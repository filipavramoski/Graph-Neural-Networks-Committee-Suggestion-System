
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import json

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


EDGE_TYPE_MENTOR = 0
EDGE_TYPE_C2 = 1
EDGE_TYPE_C3 = 2
EDGE_TYPE_RESEARCH = 3
EDGE_TYPE_COLLABORATION = 4


class RoleSpecificMultiTaskGNN(nn.Module):

    def __init__(self, thesis_feature_dim, professor_feature_dim, hidden_dim, num_relations=5):
        super(RoleSpecificMultiTaskGNN, self).__init__()


        self.thesis_transform = nn.Sequential(
            nn.Linear(thesis_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.professor_transform = nn.Sequential(
            nn.Linear(professor_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # RGCN layers (5 edge types: mentor, c2, c3, research, collaboration)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)

        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)

        # HEAD 1: Mentor prediction
        self.head_mentor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # HEAD 2: C2 prediction
        self.head_c2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # HEAD 3: C3 prediction
        self.head_c3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, thesis_features, professor_features, edge_index, edge_type):
        # Transform features
        thesis_emb = self.thesis_transform(thesis_features)
        prof_emb = self.professor_transform(professor_features)
        # Combine all node features
        all_features = torch.cat([thesis_emb, prof_emb], dim=0)
        # RGCN layers with edge type awareness
        x1 = self.conv1(all_features, edge_index, edge_type)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = x1 + all_features

        x2 = self.conv2(x1, edge_index, edge_type)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = x2 + x1

        x3 = self.conv3(x2, edge_index, edge_type)
        x3 = self.bn3(x3)
        x3 = x3 + x2

        return all_features, x3

    def predict_mentor(self, x_ini, x_final, edge_index):

        thesis_emb = x_final[edge_index[0]]
        prof_emb = x_final[edge_index[1]]

        combined = torch.cat([thesis_emb, prof_emb], dim=-1)
        scores = self.head_mentor(combined).squeeze(-1)

        return torch.sigmoid(scores)

    def predict_c2(self, x_ini, x_final, edge_index):

        thesis_emb = x_final[edge_index[0]]
        prof_emb = x_final[edge_index[1]]

        combined = torch.cat([thesis_emb, prof_emb], dim=-1)
        scores = self.head_c2(combined).squeeze(-1)

        return torch.sigmoid(scores)

    def predict_c3(self, x_ini, x_final, edge_index):

        thesis_emb = x_final[edge_index[0]]
        prof_emb = x_final[edge_index[1]]

        combined = torch.cat([thesis_emb, prof_emb], dim=-1)
        scores = self.head_c3(combined).squeeze(-1)

        return torch.sigmoid(scores)


class RoleSpecificTrainer:

    def __init__(self, thesis_feature_dim, professor_feature_dim, hidden_dim,
                 num_professors, num_theses, device, info):
        self.model = RoleSpecificMultiTaskGNN(
            thesis_feature_dim=thesis_feature_dim,
            professor_feature_dim=professor_feature_dim,
            hidden_dim=hidden_dim,
            num_relations=5  # mentor, c2, c3, research, collaboration
        ).to(device)

        self.num_professors = num_professors
        self.num_theses = num_theses
        self.professor_offset = num_theses
        self.device = device
        self.info = info
        self.hidden_dim = hidden_dim

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )


        self.diversity_weights = self._compute_diversity_weights()

        print(f" Role-Specific Model initialized on {device}")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - Architecture: RGCN with 3 role-specific heads")
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - Edge types: 5 (mentor, c2, c3, research, collaboration)")
        print(f"   - Training strategy: EDGE MASKING (70% visible, 30% masked)")
        print(f"   -  Diversity weighting: ENABLED")

    def _compute_diversity_weights(self):

        if 'activity_counts' not in self.info:
            print("Activity counts not found - diversity weighting disabled")
            return None

        weights = {}

        for role in ['mentor', 'c2', 'c3']:
            activity_counts = self.info['activity_counts'].get(role, {})

            if not activity_counts:
                weights[role] = {}
                continue

            activities = list(activity_counts.values())
            median_activity = np.median(activities)

            role_weights = {}
            for prof_name, activity in activity_counts.items():
                if activity == 0:
                    weight = 3.0
                else:
                    weight = median_activity / activity
                    weight = np.clip(weight, 0.3, 5.0)

                role_weights[prof_name] = weight

            weights[role] = role_weights

            weight_values = list(role_weights.values())
            print(f"    {role.upper()} weights: min={min(weight_values):.2f}, "
                  f"median={np.median(weight_values):.2f}, max={max(weight_values):.2f}")

        return weights

    def _get_sample_weights(self, edges, labels, role='mentor'):

        if self.diversity_weights is None or role not in self.diversity_weights:
            return torch.ones_like(labels)

        weights = torch.ones_like(labels)
        role_weights = self.diversity_weights[role]

        for i in range(edges.shape[1]):
            if labels[i] == 1.0:
                prof_node_idx = edges[1, i].item()
                prof_idx = prof_node_idx - self.professor_offset

                if 0 <= prof_idx < self.num_professors:
                    prof_name = self.info['mappings']['professors'][prof_idx]
                    weight = role_weights.get(prof_name, 1.0)
                    weights[i] = weight

        return weights

    def train_validate(self, data_train, data_val, data_test,
                       num_epochs=200, patience=15, save_path='role_specific_model.pt'):

        best_combined_metric = -1
        patience_counter = 0
        best_epoch = 0



        for epoch in range(num_epochs):
            self.model.train()
            train_loss, mentor_loss, c2_loss, c3_loss = self._train_epoch(data_train)

            if (epoch + 1) % 5 == 0:
                val_metrics = self.validate(data_val)


                combined_metric = (
                                          val_metrics['mentor']['hits@3'] +
                                          val_metrics['c2']['hits@3'] +
                                          val_metrics['c3']['hits@3']
                                  ) / 3

                self.scheduler.step(combined_metric)

                print(f'\nEpoch {epoch + 1:03d} | LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                print(f'  Mentor → Loss: {mentor_loss:.4f} | '
                      f'Hits@3: {val_metrics["mentor"]["hits@3"]:.4f} | '
                      f'AUC: {val_metrics["mentor"]["auc"]:.4f}')
                print(f'  C2     → Loss: {c2_loss:.4f} | '
                      f'Hits@3: {val_metrics["c2"]["hits@3"]:.4f} | '
                      f'AUC: {val_metrics["c2"]["auc"]:.4f}')
                print(f'  C3     → Loss: {c3_loss:.4f} | '
                      f'Hits@3: {val_metrics["c3"]["hits@3"]:.4f} | '
                      f'AUC: {val_metrics["c3"]["auc"]:.4f}')

                if combined_metric > best_combined_metric:
                    best_combined_metric = combined_metric
                    best_epoch = epoch + 1
                    patience_counter = 0

                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'thesis_dim': data_train['thesis_features'].shape[1],
                        'professor_dim': data_train['professor_features'].shape[1],
                        'hidden_dim': self.hidden_dim,  # ← CHANGED: Use instance variable
                        'num_professors': self.num_professors,
                        'num_theses': self.num_theses,
                        'best_combined_metric': best_combined_metric,
                        'best_epoch': best_epoch,
                        'mentor_metrics': val_metrics['mentor'],
                        'c2_metrics': val_metrics['c2'],
                        'c3_metrics': val_metrics['c3']
                    }, save_path)
                    print(f'   Best model saved! (Combined: {combined_metric:.4f})')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'\n Early stopping at epoch {epoch + 1}!')
                        break




        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.validate(data_test)

        print(f'\n Mentor Role Results (Negative Sampling - 1:3 ratio):')
        print(f'   - Hits@1: {test_metrics["mentor"]["hits@1"]:.4f}')
        print(f'   - Hits@3: {test_metrics["mentor"]["hits@3"]:.4f}')
        print(f'   - Hits@5: {test_metrics["mentor"]["hits@5"]:.4f}')
        print(f'   - MRR: {test_metrics["mentor"]["mrr"]:.4f}')
        print(f'   - AUC: {test_metrics["mentor"]["auc"]:.4f}')

        print(f'\n C2 Role Results (Negative Sampling - 1:3 ratio):')
        print(f'   - Hits@1: {test_metrics["c2"]["hits@1"]:.4f}')
        print(f'   - Hits@3: {test_metrics["c2"]["hits@3"]:.4f}')
        print(f'   - Hits@5: {test_metrics["c2"]["hits@5"]:.4f}')
        print(f'   - MRR: {test_metrics["c2"]["mrr"]:.4f}')
        print(f'   - AUC: {test_metrics["c2"]["auc"]:.4f}')

        print(f'\n C3 Role Results (Negative Sampling - 1:3 ratio):')
        print(f'   - Hits@1: {test_metrics["c3"]["hits@1"]:.4f}')
        print(f'   - Hits@3: {test_metrics["c3"]["hits@3"]:.4f}')
        print(f'   - Hits@5: {test_metrics["c3"]["hits@5"]:.4f}')
        print(f'   - MRR: {test_metrics["c3"]["mrr"]:.4f}')
        print(f'   - AUC: {test_metrics["c3"]["auc"]:.4f}')

        print(f'\n   - Best epoch: {best_epoch}')


        full_rank_mentor = self.evaluate_full_ranking(data_test, role='mentor')
        print(f'\n Mentor Role (Full Ranking):')
        print(f'   - Hits@1: {full_rank_mentor["hits@1"]:.4f}')
        print(f'   - Hits@3: {full_rank_mentor["hits@3"]:.4f}')
        print(f'   - Hits@5: {full_rank_mentor["hits@5"]:.4f}')
        print(f'   - Hits@10: {full_rank_mentor["hits@10"]:.4f}')
        print(f'   - MRR: {full_rank_mentor["mrr"]:.4f}')

        full_rank_c2 = self.evaluate_full_ranking(data_test, role='c2')
        print(f'\n C2 Role (Full Ranking):')
        print(f'   - Hits@1: {full_rank_c2["hits@1"]:.4f}')
        print(f'   - Hits@3: {full_rank_c2["hits@3"]:.4f}')
        print(f'   - Hits@5: {full_rank_c2["hits@5"]:.4f}')
        print(f'   - Hits@10: {full_rank_c2["hits@10"]:.4f}')
        print(f'   - MRR: {full_rank_c2["mrr"]:.4f}')

        full_rank_c3 = self.evaluate_full_ranking(data_test, role='c3')
        print(f'\n C3 Role (Full Ranking):')
        print(f'   - Hits@1: {full_rank_c3["hits@1"]:.4f}')
        print(f'   - Hits@3: {full_rank_c3["hits@3"]:.4f}')
        print(f'   - Hits@5: {full_rank_c3["hits@5"]:.4f}')
        print(f'   - Hits@10: {full_rank_c3["hits@10"]:.4f}')
        print(f'   - MRR: {full_rank_c3["mrr"]:.4f}')

        return test_metrics

    def _train_epoch(self, data):
        self.model.train()

        x_ini, x_final = self.model(
            thesis_features=data['thesis_features'],
            professor_features=data['professor_features'],
            edge_index=data['edge_index_visible'],
            edge_type=data['edge_type_visible']
        )

        mentor_scores = self.model.predict_mentor(x_ini, x_final, data['mentor_edges'])
        labels_mentor = data['mentor_labels']

        sample_weights_mentor = self._get_sample_weights(
            data['mentor_edges'], labels_mentor, role='mentor'
        )

        pos_count = labels_mentor.sum()
        neg_count = len(labels_mentor) - pos_count
        if pos_count > 0 and neg_count > 0:
            pos_weight = (neg_count / pos_count).clamp(1.0, 5.0)
        else:
            pos_weight = torch.tensor(1.0, device=self.device)

        bce_loss = F.binary_cross_entropy(mentor_scores, labels_mentor, reduction='none')
        pt = torch.where(labels_mentor == 1, mentor_scores, 1 - mentor_scores)
        focal_loss = ((1 - pt) ** 2 * bce_loss * sample_weights_mentor).mean()
        loss_mentor = focal_loss * pos_weight


        c2_scores = self.model.predict_c2(x_ini, x_final, data['c2_edges'])
        labels_c2 = data['c2_labels']

        sample_weights_c2 = self._get_sample_weights(
            data['c2_edges'], labels_c2, role='c2'
        )

        pos_count_c2 = labels_c2.sum()
        neg_count_c2 = len(labels_c2) - pos_count_c2
        if pos_count_c2 > 0 and neg_count_c2 > 0:
            pos_weight_c2 = (neg_count_c2 / pos_count_c2).clamp(1.0, 5.0)
        else:
            pos_weight_c2 = torch.tensor(1.0, device=self.device)

        bce_loss_c2 = F.binary_cross_entropy(c2_scores, labels_c2, reduction='none')
        pt_c2 = torch.where(labels_c2 == 1, c2_scores, 1 - c2_scores)
        focal_loss_c2 = ((1 - pt_c2) ** 2 * bce_loss_c2 * sample_weights_c2).mean()  # ✨ Apply weights
        loss_c2 = focal_loss_c2 * pos_weight_c2

        c3_scores = self.model.predict_c3(x_ini, x_final, data['c3_edges'])
        labels_c3 = data['c3_labels']

        sample_weights_c3 = self._get_sample_weights(
            data['c3_edges'], labels_c3, role='c3'
        )

        pos_count_c3 = labels_c3.sum()
        neg_count_c3 = len(labels_c3) - pos_count_c3
        if pos_count_c3 > 0 and neg_count_c3 > 0:
            pos_weight_c3 = (neg_count_c3 / pos_count_c3).clamp(1.0, 5.0)
        else:
            pos_weight_c3 = torch.tensor(1.0, device=self.device)

        bce_loss_c3 = F.binary_cross_entropy(c3_scores, labels_c3, reduction='none')
        pt_c3 = torch.where(labels_c3 == 1, c3_scores, 1 - c3_scores)
        focal_loss_c3 = ((1 - pt_c3) ** 2 * bce_loss_c3 * sample_weights_c3).mean()  # ✨ Apply weights
        loss_c3 = focal_loss_c3 * pos_weight_c3

        total_loss = loss_mentor + loss_c2 + loss_c3

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item(), loss_mentor.item(), loss_c2.item(), loss_c3.item()

    def validate(self, data):
        self.model.eval()

        with torch.no_grad():
            x_ini, x_final = self.model(
                thesis_features=data['thesis_features'],
                professor_features=data['professor_features'],
                edge_index=data['edge_index_visible'],
                edge_type=data['edge_type_visible']
            )

            predictions_mentor = self.model.predict_mentor(
                x_ini, x_final, data['mentor_edges']
            )
            mentor_metrics = self._calculate_metrics(
                data['mentor_edges'],
                predictions_mentor,
                data['mentor_labels']
            )

            # C2 metrics
            predictions_c2 = self.model.predict_c2(
                x_ini, x_final, data['c2_edges']
            )
            c2_metrics = self._calculate_metrics(
                data['c2_edges'],
                predictions_c2,
                data['c2_labels']
            )

            # C3 metrics
            predictions_c3 = self.model.predict_c3(
                x_ini, x_final, data['c3_edges']
            )
            c3_metrics = self._calculate_metrics(
                data['c3_edges'],
                predictions_c3,
                data['c3_labels']
            )

        return {
            'mentor': mentor_metrics,
            'c2': c2_metrics,
            'c3': c3_metrics
        }

    def _calculate_metrics(self, edges, predictions, labels):
        labels_np = labels.cpu().numpy()
        preds_np = predictions.cpu().numpy()

        # Loss
        loss = F.binary_cross_entropy(predictions, labels).item()

        # Group by thesis
        edge_org = edges[0].cpu().numpy()
        edge_des = edges[1].cpu().numpy()

        thesis_to_preds = {}
        thesis_to_labels = {}

        for t, p, score, l in zip(edge_org, edge_des, preds_np, labels_np):
            if t not in thesis_to_preds:
                thesis_to_preds[t] = {}
                thesis_to_labels[t] = []
            thesis_to_preds[t][p] = score
            if l == 1:
                thesis_to_labels[t].append(p)

        # Ranking metrics
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        mrr_sum = 0
        total = 0

        for t in thesis_to_preds:
            if not thesis_to_labels[t]:
                continue

            sorted_profs = sorted(thesis_to_preds[t].items(),
                                  key=lambda x: x[1], reverse=True)
            top_k_profs = [p for p, _ in sorted_profs]
            true_profs = set(thesis_to_labels[t])

            if top_k_profs[0] in true_profs:
                hits_at_1 += 1

            if any(p in true_profs for p in top_k_profs[:3]):
                hits_at_3 += 1

            if any(p in true_profs for p in top_k_profs[:5]):
                hits_at_5 += 1

            for rank, p in enumerate(top_k_profs, 1):
                if p in true_profs:
                    mrr_sum += 1.0 / rank
                    break

            total += 1

        hits_at_1 = hits_at_1 / total if total > 0 else 0.0
        hits_at_3 = hits_at_3 / total if total > 0 else 0.0
        hits_at_5 = hits_at_5 / total if total > 0 else 0.0
        mrr = mrr_sum / total if total > 0 else 0.0

        try:
            auc = roc_auc_score(labels_np, preds_np)
        except:
            auc = 0.5

        try:
            map_score = average_precision_score(labels_np, preds_np)
        except:
            map_score = 0.0

        return {
            "loss": loss,
            "hits@1": hits_at_1,
            "hits@3": hits_at_3,
            "hits@5": hits_at_5,
            "mrr": mrr,
            "auc": auc,
            "map": map_score
        }

    def evaluate_full_ranking(self, data, role='mentor'):

        self.model.eval()

        with torch.no_grad():
            # GNN forward pass
            x_ini, x_final = self.model(
                thesis_features=data['thesis_features'],
                professor_features=data['professor_features'],
                edge_index=data['edge_index_visible'],
                edge_type=data['edge_type_visible']
            )

            # Select the appropriate edges and labels based on role
            if role == 'mentor':
                edges = data['mentor_edges']
                labels = data['mentor_labels']
                predict_fn = self.model.predict_mentor
                activity_key = 'mentor'
            elif role == 'c2':
                edges = data['c2_edges']
                labels = data['c2_labels']
                predict_fn = self.model.predict_c2
                activity_key = 'c2'
            elif role == 'c3':
                edges = data['c3_edges']
                labels = data['c3_labels']
                predict_fn = self.model.predict_c3
                activity_key = 'c3'
            else:
                raise ValueError(f"Unknown role: {role}")

            # Get unique test thesis indices
            test_thesis_indices = torch.unique(edges[0]).cpu().numpy()

            # Get eligible professors for this role
            activity_counts = self.info['activity_counts'][activity_key]
            eligible_profs = [
                i for i, prof_name in enumerate(self.info['mappings']['professors'])
                if activity_counts.get(prof_name, 0) > 0
            ]

            print(f"   Evaluating {len(test_thesis_indices)} theses against "
                  f"{len(eligible_profs)} eligible {role}s")

            # Build mapping: thesis_idx → true_professor_idx
            thesis_to_true_prof = {}
            for i in range(edges.shape[1]):
                thesis_idx = edges[0, i].item()
                prof_node_idx = edges[1, i].item()
                label = labels[i].item()

                if label == 1.0:  # True professor
                    prof_idx = prof_node_idx - self.professor_offset
                    thesis_to_true_prof[thesis_idx] = prof_idx

            # Calculate metrics
            hits_at_1 = 0
            hits_at_3 = 0
            hits_at_5 = 0
            hits_at_10 = 0
            mrr_sum = 0
            total = 0


            for thesis_idx in test_thesis_indices:
                if thesis_idx not in thesis_to_true_prof:
                    continue

                scoring_edges = torch.tensor([
                    [thesis_idx] * len(eligible_profs),
                    [p + self.professor_offset for p in eligible_profs]
                ], dtype=torch.long, device=self.device)

                scores = predict_fn(x_ini, x_final, scoring_edges)

                sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
                ranked_profs = [eligible_profs[i] for i in sorted_indices]

                true_prof = thesis_to_true_prof[thesis_idx]

                if true_prof in ranked_profs:
                    true_rank = ranked_profs.index(true_prof) + 1

                    if true_rank <= 1:
                        hits_at_1 += 1
                    if true_rank <= 3:
                        hits_at_3 += 1
                    if true_rank <= 5:
                        hits_at_5 += 1
                    if true_rank <= 10:
                        hits_at_10 += 1

                    mrr_sum += 1.0 / true_rank
                else:
                    # True professor not in eligible list (shouldn't happen)
                    print(f"   Warning: True prof {true_prof} not in eligible list for thesis {thesis_idx}")

                total += 1

            # Calculate final metrics
            results = {
                'hits@1': hits_at_1 / total if total > 0 else 0.0,
                'hits@3': hits_at_3 / total if total > 0 else 0.0,
                'hits@5': hits_at_5 / total if total > 0 else 0.0,
                'hits@10': hits_at_10 / total if total > 0 else 0.0,
                'mrr': mrr_sum / total if total > 0 else 0.0,
            }

            return results


def prepare_role_specific_data_with_masking(graph_data, info):  # ← Removed neg_pos_ratio parameter



    thesis_features = graph_data['thesis'].x
    professor_features = graph_data['professor'].x

    num_theses = len(thesis_features)
    num_professors = len(professor_features)
    professor_offset = num_theses

    # Get edges by role
    mentor_edges_raw = graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy()
    c2_edges_raw = graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy()
    c3_edges_raw = graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy()
    research_edges_raw = graph_data[('professor', 'researches', 'thesis')].edge_index.numpy()
    collab_edges_raw = graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy()

    print(f"\n✓ Original edge counts:")
    print(f"   - Mentor edges: {mentor_edges_raw.shape[1]}")
    print(f"   - C2 edges: {c2_edges_raw.shape[1]}")
    print(f"   - C3 edges: {c3_edges_raw.shape[1]}")
    print(f"   - Research edges: {research_edges_raw.shape[1]}")
    print(f"   - Collaboration edges: {collab_edges_raw.shape[1]}")

    # Build thesis → professors mapping
    thesis_to_roles = {}

    # Mentor
    for i in range(mentor_edges_raw.shape[1]):
        prof_idx = mentor_edges_raw[0, i]
        thesis_idx = mentor_edges_raw[1, i]
        if thesis_idx not in thesis_to_roles:
            thesis_to_roles[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None}
        thesis_to_roles[thesis_idx]['mentor'] = prof_idx

    # C2
    for i in range(c2_edges_raw.shape[1]):
        prof_idx = c2_edges_raw[0, i]
        thesis_idx = c2_edges_raw[1, i]
        if thesis_idx not in thesis_to_roles:
            thesis_to_roles[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None}
        thesis_to_roles[thesis_idx]['c2'] = prof_idx

    # C3
    for i in range(c3_edges_raw.shape[1]):
        prof_idx = c3_edges_raw[0, i]
        thesis_idx = c3_edges_raw[1, i]
        if thesis_idx not in thesis_to_roles:
            thesis_to_roles[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None}
        thesis_to_roles[thesis_idx]['c3'] = prof_idx

    print(f"\n Theses with committees: {len(thesis_to_roles)}")

    professors_with_mentor_experience = set()
    professors_with_c2_experience = set()
    professors_with_c3_experience = set()

    for thesis_idx, roles in thesis_to_roles.items():
        if roles['mentor'] is not None:
            professors_with_mentor_experience.add(roles['mentor'])
        if roles['c2'] is not None:
            professors_with_c2_experience.add(roles['c2'])
        if roles['c3'] is not None:
            professors_with_c3_experience.add(roles['c3'])

    print(f"\n Role-specific professor pools:")
    print(f"   - Professors with MENTOR experience: {len(professors_with_mentor_experience)}")
    print(f"   - Professors with C2 experience: {len(professors_with_c2_experience)}")
    print(f"   - Professors with C3 experience: {len(professors_with_c3_experience)}")


    mentor_only = professors_with_mentor_experience - professors_with_c2_experience - professors_with_c3_experience
    c2_only = professors_with_c2_experience - professors_with_mentor_experience - professors_with_c3_experience
    c3_only = professors_with_c3_experience - professors_with_mentor_experience - professors_with_c2_experience

    print(f"\n   Role specialists:")
    print(f"   - Mentor-only: {len(mentor_only)} professors")
    print(f"   - C2-only: {len(c2_only)} professors")
    print(f"   - C3-only: {len(c3_only)} professors")

    if c3_only:
        print(f"   - C3-only examples: {list(info['mappings']['professors'][p] for p in list(c3_only)[:3])}")

    #  CREATE EDGE MASK
    all_thesis_ids = list(thesis_to_roles.keys())
    random.shuffle(all_thesis_ids)

    train_size = int(0.7 * len(all_thesis_ids))
    val_size = int(0.15 * len(all_thesis_ids))

    # Assign masks
    edge_mask = {}
    for i, thesis_id in enumerate(all_thesis_ids):
        if i < train_size:
            edge_mask[thesis_id] = False  # NOT masked (visible for training)
        elif i < train_size + val_size:
            edge_mask[thesis_id] = True  # MASKED (for validation)
        else:
            edge_mask[thesis_id] = True  # MASKED (for testing)

    masked_train_theses = [t for t in all_thesis_ids if not edge_mask[t]]
    masked_val_theses = [t for t in all_thesis_ids[train_size:train_size + val_size]]
    masked_test_theses = [t for t in all_thesis_ids[train_size + val_size:]]

    print(f"\n Edge Masking Statistics:")
    print(f"   - Train theses (VISIBLE edges): {len(masked_train_theses)} (70%)")
    print(f"   - Val theses (MASKED edges): {len(masked_val_theses)} (15%)")
    print(f"   - Test theses (MASKED edges): {len(masked_test_theses)} (15%)")


    visible_edges = []
    visible_edge_types = []

    # Add edges ONLY from non-masked (train) theses
    for i in range(mentor_edges_raw.shape[1]):
        thesis_idx = mentor_edges_raw[1, i]
        if not edge_mask.get(thesis_idx, True):  # If NOT masked
            visible_edges.append([thesis_idx, mentor_edges_raw[0, i] + professor_offset])
            visible_edges.append([mentor_edges_raw[0, i] + professor_offset, thesis_idx])

            visible_edge_types.append(EDGE_TYPE_MENTOR)
            visible_edge_types.append(EDGE_TYPE_MENTOR)

    for i in range(c2_edges_raw.shape[1]):
        thesis_idx = c2_edges_raw[1, i]
        if not edge_mask.get(thesis_idx, True):
            visible_edges.append([thesis_idx, c2_edges_raw[0, i] + professor_offset])
            visible_edges.append([c2_edges_raw[0, i] + professor_offset, thesis_idx])

            visible_edge_types.append(EDGE_TYPE_C2)
            visible_edge_types.append(EDGE_TYPE_C2)

    for i in range(c3_edges_raw.shape[1]):
        thesis_idx = c3_edges_raw[1, i]
        if not edge_mask.get(thesis_idx, True):
            visible_edges.append([thesis_idx, c3_edges_raw[0, i] + professor_offset])
            visible_edges.append([c3_edges_raw[0, i] + professor_offset, thesis_idx])

            visible_edge_types.append(EDGE_TYPE_C3)
            visible_edge_types.append(EDGE_TYPE_C3)

    for i in range(research_edges_raw.shape[1]):
        thesis_idx = research_edges_raw[1, i]
        if not edge_mask.get(thesis_idx, True):
            visible_edges.append([thesis_idx, research_edges_raw[0, i] + professor_offset])
            visible_edges.append([research_edges_raw[0, i] + professor_offset, thesis_idx])

            visible_edge_types.append(EDGE_TYPE_RESEARCH)
            visible_edge_types.append(EDGE_TYPE_RESEARCH)

    # Collaboration edges (always visible - professor to professor)
    for i in range(collab_edges_raw.shape[1]):
        visible_edges.append([collab_edges_raw[0, i] + professor_offset,
                              collab_edges_raw[1, i] + professor_offset])
        visible_edges.append([collab_edges_raw[1, i] + professor_offset,
                              collab_edges_raw[0, i] + professor_offset])

        visible_edge_types.append(EDGE_TYPE_COLLABORATION)
        visible_edge_types.append(EDGE_TYPE_COLLABORATION)

    print(f"\n✓ Visible graph (for training):")
    print(f"   - Total visible edges: {len(visible_edges)}")
    print(f"   - Edge types: 5 (mentor, c2, c3, research, collaboration)")
    print(f"   - Masked theses have NO edges in this graph")

    edge_index_visible = torch.tensor(visible_edges, dtype=torch.long, device=device).t()
    edge_type_visible = torch.tensor(visible_edge_types, dtype=torch.long, device=device)


    mentor_src, mentor_dst, mentor_labels, mentor_thesis_ids = [], [], [], []
    c2_src, c2_dst, c2_labels, c2_thesis_ids = [], [], [], []
    c3_src, c3_dst, c3_labels, c3_thesis_ids = [], [], [], []

    print(f"\n📊 Preparing training examples with ALL negatives:")

    for thesis_idx, roles in thesis_to_roles.items():
        committee = set([p for p in roles.values() if p is not None])


        if roles['mentor'] is not None:
            # Positive example
            mentor_src.append(thesis_idx)
            mentor_dst.append(roles['mentor'] + professor_offset)
            mentor_labels.append(1.0)
            mentor_thesis_ids.append(thesis_idx)

            negative_pool = professors_with_mentor_experience - committee

            for neg in negative_pool:
                mentor_src.append(thesis_idx)
                mentor_dst.append(neg + professor_offset)
                mentor_labels.append(0.0)
                mentor_thesis_ids.append(thesis_idx)


        if roles['c2'] is not None:
            # Positive example
            c2_src.append(thesis_idx)
            c2_dst.append(roles['c2'] + professor_offset)
            c2_labels.append(1.0)
            c2_thesis_ids.append(thesis_idx)


            negative_pool = professors_with_c2_experience - committee

            for neg in negative_pool:
                c2_src.append(thesis_idx)
                c2_dst.append(neg + professor_offset)
                c2_labels.append(0.0)
                c2_thesis_ids.append(thesis_idx)


        if roles['c3'] is not None:
            # Positive example
            c3_src.append(thesis_idx)
            c3_dst.append(roles['c3'] + professor_offset)
            c3_labels.append(1.0)
            c3_thesis_ids.append(thesis_idx)

            # ALL negative examples (exclude committee members)
            negative_pool = professors_with_c3_experience - committee

            for neg in negative_pool:
                c3_src.append(thesis_idx)
                c3_dst.append(neg + professor_offset)
                c3_labels.append(0.0)
                c3_thesis_ids.append(thesis_idx)

    print(f"   - Total MENTOR examples: {len(mentor_src)} "
          f"(~{len(mentor_src) / max(1, sum(1 for r in thesis_to_roles.values() if r['mentor'] is not None)):.1f} per thesis)")
    print(f"   - Total C2 examples: {len(c2_src)} "
          f"(~{len(c2_src) / max(1, sum(1 for r in thesis_to_roles.values() if r['c2'] is not None)):.1f} per thesis)")
    print(f"   - Total C3 examples: {len(c3_src)} "
          f"(~{len(c3_src) / max(1, sum(1 for r in thesis_to_roles.values() if r['c3'] is not None)):.1f} per thesis)")

    # === SPLIT EXAMPLES BY MASK ===
    def split_by_mask(src_list, dst_list, label_list, thesis_ids, thesis_set):

        mask = [i for i, t in enumerate(thesis_ids) if t in thesis_set]

        if not mask:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float, device=device)
            )

        edges = torch.tensor([[src_list[i], dst_list[i]] for i in mask],
                             dtype=torch.long, device=device).t()
        labels = torch.tensor([label_list[i] for i in mask],
                              dtype=torch.float, device=device)

        return edges, labels

    # Split mentor examples
    train_mentor_edges, train_mentor_labels = split_by_mask(
        mentor_src, mentor_dst, mentor_labels, mentor_thesis_ids, set(masked_train_theses)
    )
    val_mentor_edges, val_mentor_labels = split_by_mask(
        mentor_src, mentor_dst, mentor_labels, mentor_thesis_ids, set(masked_val_theses)
    )
    test_mentor_edges, test_mentor_labels = split_by_mask(
        mentor_src, mentor_dst, mentor_labels, mentor_thesis_ids, set(masked_test_theses)
    )

    # Split C2 examples
    train_c2_edges, train_c2_labels = split_by_mask(
        c2_src, c2_dst, c2_labels, c2_thesis_ids, set(masked_train_theses)
    )
    val_c2_edges, val_c2_labels = split_by_mask(
        c2_src, c2_dst, c2_labels, c2_thesis_ids, set(masked_val_theses)
    )
    test_c2_edges, test_c2_labels = split_by_mask(
        c2_src, c2_dst, c2_labels, c2_thesis_ids, set(masked_test_theses)
    )

    # Split C3 examples
    train_c3_edges, train_c3_labels = split_by_mask(
        c3_src, c3_dst, c3_labels, c3_thesis_ids, set(masked_train_theses)
    )
    val_c3_edges, val_c3_labels = split_by_mask(
        c3_src, c3_dst, c3_labels, c3_thesis_ids, set(masked_val_theses)
    )
    test_c3_edges, test_c3_labels = split_by_mask(
        c3_src, c3_dst, c3_labels, c3_thesis_ids, set(masked_test_theses)
    )

    print(f"\n Data Split (by masked edges):")
    print(
        f"   Train - Mentor: {train_mentor_edges.shape[1]}, C2: {train_c2_edges.shape[1]}, C3: {train_c3_edges.shape[1]}")
    print(f"   Val   - Mentor: {val_mentor_edges.shape[1]}, C2: {val_c2_edges.shape[1]}, C3: {val_c3_edges.shape[1]}")
    print(
        f"   Test  - Mentor: {test_mentor_edges.shape[1]}, C2: {test_c2_edges.shape[1]}, C3: {test_c3_edges.shape[1]}")

    # === PACKAGE DATA ===
    data_train = {
        'thesis_features': thesis_features.to(device),
        'professor_features': professor_features.to(device),
        'edge_index_visible': edge_index_visible,
        'edge_type_visible': edge_type_visible,
        'mentor_edges': train_mentor_edges,
        'mentor_labels': train_mentor_labels,
        'c2_edges': train_c2_edges,
        'c2_labels': train_c2_labels,
        'c3_edges': train_c3_edges,
        'c3_labels': train_c3_labels
    }

    data_val = {
        'thesis_features': thesis_features.to(device),
        'professor_features': professor_features.to(device),
        'edge_index_visible': edge_index_visible,
        'edge_type_visible': edge_type_visible,
        'mentor_edges': val_mentor_edges,
        'mentor_labels': val_mentor_labels,
        'c2_edges': val_c2_edges,
        'c2_labels': val_c2_labels,
        'c3_edges': val_c3_edges,
        'c3_labels': val_c3_labels
    }

    data_test = {
        'thesis_features': thesis_features.to(device),
        'professor_features': professor_features.to(device),
        'edge_index_visible': edge_index_visible,
        'edge_type_visible': edge_type_visible,
        'mentor_edges': test_mentor_edges,
        'mentor_labels': test_mentor_labels,
        'c2_edges': test_c2_edges,
        'c2_labels': test_c2_labels,
        'c3_edges': test_c3_edges,
        'c3_labels': test_c3_labels
    }

    print(f"\n Data prepared with ALL NEGATIVES strategy!")
    print(f"   - Training uses graph with {len(visible_edges)} edges (70% of theses)")
    print(f"   - Each thesis scored against ALL eligible professors")
    print(f"   - Validation predicts on {len(masked_val_theses)} masked theses (15%)")
    print(f"   - Testing predicts on {len(masked_test_theses)} masked theses (15%)")

    return data_train, data_val, data_test, num_theses


def main():


    graph_data = torch.load('../structure/hetero_graph_edge_labeled.pt', weights_only=False)
    with open('../structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
        info = json.load(f)


    data_train, data_val, data_test, num_theses = prepare_role_specific_data_with_masking(
        graph_data, info
    )

    thesis_dim = data_train['thesis_features'].shape[1]
    professor_dim = data_train['professor_features'].shape[1]
    num_professors = len(info['mappings']['professors'])

    # Initialize trainer
    trainer = RoleSpecificTrainer(
        thesis_feature_dim=thesis_dim,
        professor_feature_dim=professor_dim,
        hidden_dim=128,
        num_professors=num_professors,
        num_theses=num_theses,
        device=device,
        info=info
    )

    # Train
    test_metrics = trainer.train_validate(
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        num_epochs=200,
        patience=15,
        save_path='role_specific_model.pt'
    )

    print(f"\n Training complete with diversity weighting and ALL negatives!")


if __name__ == "__main__":
    main()