
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
import os
import re
import json
import numpy as np

try:
    import sys
    sys.path.insert(0, 'gnn')
    from training.training import RoleSpecificMultiTaskGNN
    GNN_AVAILABLE = True
    print(" Role-Specific GNN components available")
except ImportError as e:
    print(f" GNN components not available: {e}")
    GNN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print(" SentenceTransformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(" SentenceTransformers not available")


TRANSLATOR_AVAILABLE = False
translator_type = None


try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    translator_type = "deep_translator"
    print(" Translator available: deep-translator (MK ↔ EN)")
except ImportError:

    try:
        from googletrans import Translator
        TRANSLATOR_AVAILABLE = True
        translator_type = "googletrans"
        print(" Translator available: googletrans (MK ↔ EN)")
    except ImportError:
        TRANSLATOR_AVAILABLE = False
        print(" No translator available")
        print("  Install with: pip install deep-translator")

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


EDGE_TYPE_MENTOR = 0
EDGE_TYPE_C2 = 1
EDGE_TYPE_C3 = 2
EDGE_TYPE_RESEARCH = 3
EDGE_TYPE_COLLABORATION = 4


class RoleSpecificThesisRecommender:

    def __init__(self):
        self.graph_data = None
        self.mappings = None
        self.info = None
        self.gnn_model = None
        self.text_encoder = None
        self.gnn_loaded = False
        self.system_loaded = False
        self.num_professors = 0
        self.num_theses = 0
        self.professor_features = None
        self.professor_offset = None


        self.load_graph_structure()
        self.initialize_text_encoder()

        if GNN_AVAILABLE:
            self.load_gnn_model()
        if TRANSLATOR_AVAILABLE:
            self._initialize_translator()
        else:
            self.translator = None
            self.translator_type = None

    def load_graph_structure(self):



        try:
            if not os.path.exists('structure/hetero_graph_edge_labeled.pt'):
                print(" hetero_graph_edge_labeled.pt not found")
                return False

            if not os.path.exists('structure/graph_metadata_edge_labeled.json'):
                print(" graph_metadata_edge_labeled.json not found")
                return False

            self.graph_data = torch.load('structure/hetero_graph_edge_labeled.pt', weights_only=False)
            print(" Graph structure loaded")

            with open('structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.mappings = metadata.get('mappings', {})
            self.num_professors = len(self.mappings['professors'])
            self.num_theses = len(self.mappings['theses'])
            self.professor_features = self.graph_data['professor'].x.to(device)
            self.professor_offset = self.num_theses

            print(f" System loaded:")
            print(f"   - Professors: {self.num_professors}")
            print(f"   - Theses: {self.num_theses}")

            # Count edges by type
            mentor_edges = self.graph_data[('professor', 'mentors', 'thesis')].edge_index
            c2_edges = self.graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index
            c3_edges = self.graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index
            collab_edges = self.graph_data[('professor', 'collaborates', 'professor')].edge_index

            print(f"   - Mentor edges: {mentor_edges.shape[1]}")
            print(f"   - C2 edges: {c2_edges.shape[1]}")
            print(f"   - C3 edges: {c3_edges.shape[1]}")
            print(f"   - Collaboration edges: {collab_edges.shape[1]}")

            self.system_loaded = True
            return True

        except Exception as e:
            print(f" Error loading system: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_gnn_model(self):

        try:
            model_path = 'training/role_specific_model.pt'
            if not os.path.exists(model_path):
                print(f" {model_path} not found")
                print("   Train with layers3_roles.py first!")
                return False

            checkpoint = torch.load(model_path, map_location=device)
            print(f"\n Loading Role-Specific GNN model...")

            thesis_dim = checkpoint.get('thesis_dim', self.graph_data['thesis'].x.shape[1])
            professor_dim = checkpoint.get('professor_dim', self.graph_data['professor'].x.shape[1])
            hidden_dim = checkpoint.get('hidden_dim', 128)

            self.gnn_model = RoleSpecificMultiTaskGNN(
                thesis_feature_dim=thesis_dim,
                professor_feature_dim=professor_dim,
                hidden_dim=hidden_dim,
                num_relations=5  # mentor, c2, c3, research, collaboration
            ).to(device)

            if 'model_state_dict' in checkpoint:
                self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.gnn_model.eval()
                self.gnn_loaded = True
                print(f" Role-Specific GNN loaded successfully!")

                if 'mentor_metrics' in checkpoint:
                    mentor = checkpoint['mentor_metrics']
                    print(f"   - Mentor Hits@3: {mentor.get('hits@3', 0):.4f}")
                if 'c2_metrics' in checkpoint:
                    c2 = checkpoint['c2_metrics']
                    print(f"   - C2 Hits@3: {c2.get('hits@3', 0):.4f}")
                if 'c3_metrics' in checkpoint:
                    c3 = checkpoint['c3_metrics']
                    print(f"   - C3 Hits@3: {c3.get('hits@3', 0):.4f}")

            return True

        except Exception as e:
            print(f" Error loading GNN model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialize_text_encoder(self):

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print(" Text encoder initialized")
            except Exception as e:
                print(f" Failed to initialize text encoder: {e}")
                self.text_encoder = None
        else:
            print(" SentenceTransformers not available")

    def clean_text(self, text):

        if not text or str(text).strip() == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        text = ' '.join(text.split())
        return text

    def _initialize_translator(self):

        global translator_type

        if translator_type == "deep_translator":
            try:
                from deep_translator import GoogleTranslator
                # Create translator: Macedonian → English
                self.translator = GoogleTranslator(source='mk', target='en')
                self.translator_type = "deep_translator"
                print(" Translator ready: Macedonian → English")
            except Exception as e:
                print(f" Failed to initialize deep-translator: {e}")
                self.translator = None
                self.translator_type = None

        elif translator_type == "googletrans":
            try:
                from googletrans import Translator
                self.translator = Translator()
                self.translator_type = "googletrans"
                print(" Translator ready: Macedonian → English")
            except Exception as e:
                print(f" Failed to initialize googletrans: {e}")
                self.translator = None
                self.translator_type = None
        else:
            self.translator = None
            self.translator_type = None

    def detect_and_translate(self, text):

        if not self.translator:
            print("  No translator available, using original text")
            return text

        try:
            # Simple language detection: Check for Cyrillic characters
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)

            if not has_cyrillic:
                # No Cyrillic = English
                print("    Detected: English (no translation needed)")
                return text

            # Has Cyrillic = Macedonian → translate to English
            print("    Detected: Macedonian (Cyrillic) → translating to English")

            # ===== DEEP_TRANSLATOR =====
            if self.translator_type == "deep_translator":
                translated_text = self.translator.translate(text)

                print(f"   Original (MK): {text[:50]}...")
                print(f"   Translated (EN): {translated_text[:50]}...")
                return translated_text

            # ===== GOOGLETRANS =====
            elif self.translator_type == "googletrans":
                translated = self.translator.translate(text, src='mk', dest='en')
                translated_text = translated.text

                print(f"   Original (MK): {text[:50]}...")
                print(f"   Translated (EN): {translated_text[:50]}...")
                return translated_text

            else:
                print(f"   ️ Unknown translator type")
                return text

        except Exception as e:
            print(f"    Translation failed: {e}")
            print(f"   Attempting to use original text")




    def encode_new_thesis(self, thesis_text):
        if self.text_encoder is None:
            print(" No text encoder available!")
            return None

        try:

            if not thesis_text or thesis_text.strip() == '':
                print(" Empty thesis text!")
                return None


            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in thesis_text)

            if has_cyrillic:
                print(f" Input: Macedonian (Cyrillic detected)")
            else:
                print(f" Input: English")

            # Translate if needed (Macedonian → English)
            english_text = self.detect_and_translate(thesis_text)

            # Clean text
            cleaned_text = self.clean_text(english_text)

            if not cleaned_text:
                print(" Empty text after cleaning, trying original...")
                cleaned_text = self.clean_text(thesis_text)

            if not cleaned_text:
                print(" Cannot process empty text!")
                return None

            # Encode
            embedding = self.text_encoder.encode([cleaned_text])[0]

            print(f"✓ Encoded thesis successfully")
            if has_cyrillic:
                print(f"   MK: {thesis_text[:40]}...")
                print(f"   EN: {english_text[:40]}...")
            else:
                print(f"   Text: {cleaned_text[:40]}...")

            return torch.tensor(embedding, dtype=torch.float32).to(device)

        except Exception as e:
            print(f" Error encoding thesis: {e}")
            import traceback
            traceback.print_exc()
            return None




    def recommend_committee(self, thesis_text, primary_mentor=None):

        if not self.gnn_loaded:
            return None

        try:



            new_thesis_emb = self.encode_new_thesis(thesis_text)
            if new_thesis_emb is None:
                return None

            # Extend features
            all_thesis_features = torch.cat([
                self.graph_data['thesis'].x.to(device).clone(),
                new_thesis_emb.unsqueeze(0)
            ], dim=0)

            professor_features = self.professor_features.clone()
            new_thesis_idx = self.num_theses

            # Choose mode
            if primary_mentor and primary_mentor in self.mappings['professors']:
                return self._recommend_with_mentor(
                    all_thesis_features, professor_features, new_thesis_idx,
                    new_thesis_emb, primary_mentor
                )
            else:
                return self._recommend_without_mentor(
                    all_thesis_features, professor_features, new_thesis_idx,
                    new_thesis_emb
                )

        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
            return None



    def _recommend_with_mentor(self, all_thesis_features, professor_features,
                               new_thesis_idx, new_thesis_emb, mentor_name):


        print(f" Mode: Mentor pre-selected (GNN picks C2 & C3 from collaborators)")
        print(f"   Selected mentor: {mentor_name}")

        mentor_idx = self.mappings['professors'].index(mentor_name)
        mentor_node_idx = mentor_idx + self.professor_offset

        # Get mentor's collaborators
        collab_edges = self.graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy()
        mentor_collaborators = set()

        for i in range(collab_edges.shape[1]):
            if collab_edges[0, i] == mentor_idx:
                mentor_collaborators.add(collab_edges[1, i])
            elif collab_edges[1, i] == mentor_idx:
                mentor_collaborators.add(collab_edges[0, i])

        print(f"   {mentor_name} has {len(mentor_collaborators)} known collaborators")

        if len(mentor_collaborators) == 0:
            print(f" {mentor_name} has no collaborators in the system!")
            return [{
                'professor': mentor_name,
                'compatibility_score': 100.0,
                'role': 'Primary Mentor (Your Selection)',
                'explanation': f"Your selected primary mentor. No collaborators found in system.",
                'is_mentor': True
            }]

        if len(mentor_collaborators) < 2:
            print(
                f" {mentor_name} has only {len(mentor_collaborators)} collaborator(s). Need at least 2 for C2 and C3!")
            recommendations = [{
                'professor': mentor_name,
                'compatibility_score': 100.0,
                'role': 'Primary Mentor (Your Selection)',
                'explanation': f"Your selected primary mentor.",
                'is_mentor': True
            }]
            for i, collab_idx in enumerate(list(mentor_collaborators), 1):
                collab_name = self.mappings['professors'][collab_idx]
                recommendations.append({
                    'professor': collab_name,
                    'compatibility_score': 90.0,
                    'role': f'Committee Member {i + 1}',
                    'explanation': f"Known collaborator with {mentor_name}.",
                    'is_known_collaborator': True
                })
            return recommendations

        training_edges, training_edge_types = self._get_training_graph_edges()

        # Create new thesis → mentor edge
        new_mentor_edge = torch.tensor([
            [new_thesis_idx, mentor_node_idx],
            [mentor_node_idx, new_thesis_idx]
        ], dtype=torch.long, device=device).t()

        new_edge_types = torch.tensor(
            [EDGE_TYPE_MENTOR, EDGE_TYPE_MENTOR],  # Bidirectional mentor edge
            dtype=torch.long,
            device=device
        )

        combined_edges = torch.cat([training_edges, new_mentor_edge], dim=1)
        combined_edge_types = torch.cat([training_edge_types, new_edge_types], dim=0)

        print(f"✓ Graph constructed:")
        print(f"   - Training edges: {training_edges.shape[1]}")
        print(f"   - New thesis → mentor edge: 2 (bidirectional)")
        print(f"   - Total edges: {combined_edges.shape[1]}")
        print(f"   - Professors retain full context from training graph!")

        # GNN forward pass
        self.gnn_model.eval()
        with torch.no_grad():
            x_ini, x_final = self.gnn_model(
                thesis_features=all_thesis_features,
                professor_features=professor_features,
                edge_index=combined_edges,  # ← Training + new edge
                edge_type=combined_edge_types
            )



            print(f" GNN forward pass complete")
            print(f"   - New thesis: connected to mentor (learns from mentor's expertise)")
            print(f"   - Professors: retain full training graph context")

            collab_list = list(mentor_collaborators)
            collab_node_indices = [c + self.professor_offset for c in collab_list]

            scoring_edges = torch.tensor([
                [new_thesis_idx] * len(collab_list),
                collab_node_indices
            ], dtype=torch.long, device=device)

            c2_scores = self.gnn_model.predict_c2(x_ini, x_final, scoring_edges)
            c3_scores = self.gnn_model.predict_c3(x_ini, x_final, scoring_edges)

            c2_scores = c2_scores.cpu().numpy()
            c3_scores = c3_scores.cpu().numpy()

        print(f" Scored {len(collab_list)} collaborators:")
        print(f"   - C2 scores: [{c2_scores.min():.4f}, {c2_scores.max():.4f}]")
        print(f"   - C3 scores: [{c3_scores.min():.4f}, {c3_scores.max():.4f}]")

        best_c2_idx = c2_scores.argmax()
        best_c3_idx = c3_scores.argmax()

        if best_c2_idx == best_c3_idx and len(collab_list) > 1:
            c3_scores_copy = c3_scores.copy()
            c3_scores_copy[best_c2_idx] = -1
            best_c3_idx = c3_scores_copy.argmax()

        best_c2_prof_idx = collab_list[best_c2_idx]
        best_c3_prof_idx = collab_list[best_c3_idx]

        best_c2_name = self.mappings['professors'][best_c2_prof_idx]
        best_c3_name = self.mappings['professors'][best_c3_prof_idx]

        recommendations = [
            {
                'professor': mentor_name,
                'compatibility_score': 100.0,
                'role': 'Primary Mentor (Your Selection)',
                'explanation': f"Your selected primary mentor.",
                'is_mentor': True
            },
            {
                'professor': best_c2_name,
                'compatibility_score': float(c2_scores[best_c2_idx] * 100),
                'role': 'Committee Member 2 (C2)',
                'explanation': f"GNN-selected C2 from {mentor_name}'s network. Known collaborator. Compatibility: {c2_scores[best_c2_idx] * 100:.1f}%.",
                'is_known_collaborator': True
            },
            {
                'professor': best_c3_name,
                'compatibility_score': float(c3_scores[best_c3_idx] * 100),
                'role': 'Committee Member 3 (C3)',
                'explanation': f"GNN-selected C3 from {mentor_name}'s network. Known collaborator. Compatibility: {c3_scores[best_c3_idx] * 100:.1f}%.",
                'is_known_collaborator': True
            }
        ]

        print(f" Committee formed:")
        print(f"   - Mentor: {mentor_name} (user selected)")
        print(f"   - C2: {best_c2_name} (GNN: {c2_scores[best_c2_idx] * 100:.1f}%)")
        print(f"   - C3: {best_c3_name} (GNN: {c3_scores[best_c3_idx] * 100:.1f}%)")

        return recommendations


    def _recommend_without_mentor(self, all_thesis_features, professor_features,
                                  new_thesis_idx, new_thesis_emb):


        print(f" Mode: No mentor selected (GNN picks Mentor, C2, C3)")

        # Load role-eligible professors from metadata
        try:
            with open('structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            activity_counts = metadata.get('activity_counts', {})

            # Get professors who have held each role
            eligible_mentors = set()
            eligible_c2 = set()
            eligible_c3 = set()

            for prof_name, count in activity_counts.get('mentor', {}).items():
                if count > 0:
                    try:
                        prof_idx = self.mappings['professors'].index(prof_name)
                        eligible_mentors.add(prof_idx)
                    except ValueError:
                        continue

            for prof_name, count in activity_counts.get('c2', {}).items():
                if count > 0:
                    try:
                        prof_idx = self.mappings['professors'].index(prof_name)
                        eligible_c2.add(prof_idx)
                    except ValueError:
                        continue

            for prof_name, count in activity_counts.get('c3', {}).items():
                if count > 0:
                    try:
                        prof_idx = self.mappings['professors'].index(prof_name)
                        eligible_c3.add(prof_idx)
                    except ValueError:
                        continue

            print(f" Role eligibility:")
            print(f"   - Eligible mentors: {len(eligible_mentors)}")
            print(f"   - Eligible C2: {len(eligible_c2)}")
            print(f"   - Eligible C3: {len(eligible_c3)}")

        except Exception as e:
            print(f" Could not load eligibility data: {e}")
            # Fallback: allow all professors
            eligible_mentors = set(range(self.num_professors))
            eligible_c2 = set(range(self.num_professors))
            eligible_c3 = set(range(self.num_professors))


        training_edges, training_edge_types = self._get_training_graph_edges()

        print(f" Using training graph structure:")
        print(f"   - Total edges: {training_edges.shape[1]}")
        print(f"   - New thesis is ISOLATED (no edges)")
        print(f"   - This matches training conditions!")

        # GNN forward pass with TRAINING graph only (new thesis isolated)
        self.gnn_model.eval()
        with torch.no_grad():
            x_ini, x_final = self.gnn_model(
                thesis_features=all_thesis_features,
                professor_features=professor_features,
                edge_index=training_edges,  # ← ONLY training edges
                edge_type=training_edge_types
            )



            print(f" GNN forward pass complete")
            print(f"   - New thesis embedding: feature-only (isolated)")
            print(f"   - Professor embeddings: rich (from training graph)")


            eligible_mentors_list = list(eligible_mentors)
            eligible_c2_list = list(eligible_c2)
            eligible_c3_list = list(eligible_c3)


            mentor_edges_tensor = torch.tensor([
                [new_thesis_idx] * len(eligible_mentors_list),
                [prof_idx + self.professor_offset for prof_idx in eligible_mentors_list]
            ], dtype=torch.long, device=device)

            c2_edges_tensor = torch.tensor([
                [new_thesis_idx] * len(eligible_c2_list),
                [prof_idx + self.professor_offset for prof_idx in eligible_c2_list]
            ], dtype=torch.long, device=device)

            c3_edges_tensor = torch.tensor([
                [new_thesis_idx] * len(eligible_c3_list),
                [prof_idx + self.professor_offset for prof_idx in eligible_c3_list]
            ], dtype=torch.long, device=device)

            print(f" Created scoring pairs:")
            print(f"   - Mentor pairs: {mentor_edges_tensor.shape[1]}")
            print(f"   - C2 pairs: {c2_edges_tensor.shape[1]}")
            print(f"   - C3 pairs: {c3_edges_tensor.shape[1]}")

            mentor_scores_partial = self.gnn_model.predict_mentor(x_ini, x_final, mentor_edges_tensor)
            c2_scores_partial = self.gnn_model.predict_c2(x_ini, x_final, c2_edges_tensor)
            c3_scores_partial = self.gnn_model.predict_c3(x_ini, x_final, c3_edges_tensor)

            mentor_scores_partial = mentor_scores_partial.cpu().numpy()
            c2_scores_partial = c2_scores_partial.cpu().numpy()
            c3_scores_partial = c3_scores_partial.cpu().numpy()

        print(f" Scored eligible professors:")
        print(f"   - Mentor scores: [{mentor_scores_partial.min():.4f}, {mentor_scores_partial.max():.4f}]")
        print(f"   - C2 scores: [{c2_scores_partial.min():.4f}, {c2_scores_partial.max():.4f}]")
        print(f"   - C3 scores: [{c3_scores_partial.min():.4f}, {c3_scores_partial.max():.4f}]")

        best_mentor_local_idx = mentor_scores_partial.argmax()
        best_c2_local_idx = c2_scores_partial.argmax()
        best_c3_local_idx = c3_scores_partial.argmax()

        best_mentor_idx = eligible_mentors_list[best_mentor_local_idx]
        best_c2_idx = eligible_c2_list[best_c2_local_idx]
        best_c3_idx = eligible_c3_list[best_c3_local_idx]

        used_indices = {best_mentor_idx}

        if best_c2_idx == best_mentor_idx:
            # Find second best C2
            c2_scores_copy = c2_scores_partial.copy()
            c2_scores_copy[best_c2_local_idx] = -1
            best_c2_local_idx = c2_scores_copy.argmax()
            best_c2_idx = eligible_c2_list[best_c2_local_idx]
        used_indices.add(best_c2_idx)

        if best_c3_idx in used_indices:
            # Find next best C3
            c3_scores_copy = c3_scores_partial.copy()
            for used_prof_idx in used_indices:
                if used_prof_idx in eligible_c3_list:
                    local_idx = eligible_c3_list.index(used_prof_idx)
                    c3_scores_copy[local_idx] = -1
            best_c3_local_idx = c3_scores_copy.argmax()
            best_c3_idx = eligible_c3_list[best_c3_local_idx]

        # Get professor names
        best_mentor_name = self.mappings['professors'][best_mentor_idx]
        best_c2_name = self.mappings['professors'][best_c2_idx]
        best_c3_name = self.mappings['professors'][best_c3_idx]

        # Create recommendations
        recommendations = [
            {
                'professor': best_mentor_name,
                'compatibility_score': float(mentor_scores_partial[best_mentor_local_idx] * 100),
                'role': 'Primary Mentor',
                'explanation': f"GNN-selected as best mentor. Compatibility: {mentor_scores_partial[best_mentor_local_idx] * 100:.1f}%."
            },
            {
                'professor': best_c2_name,
                'compatibility_score': float(c2_scores_partial[best_c2_local_idx] * 100),
                'role': 'Committee Member 2 (C2)',
                'explanation': f"GNN-selected as best C2. Compatibility: {c2_scores_partial[best_c2_local_idx] * 100:.1f}%."
            },
            {
                'professor': best_c3_name,
                'compatibility_score': float(c3_scores_partial[best_c3_local_idx] * 100),
                'role': 'Committee Member 3 (C3)',
                'explanation': f"GNN-selected as best C3. Compatibility: {c3_scores_partial[best_c3_local_idx] * 100:.1f}%."
            }
        ]

        print(f" Committee formed:")
        print(f"   - Mentor: {best_mentor_name} (GNN: {mentor_scores_partial[best_mentor_local_idx] * 100:.1f}%)")
        print(f"   - C2: {best_c2_name} (GNN: {c2_scores_partial[best_c2_local_idx] * 100:.1f}%)")
        print(f"   - C3: {best_c3_name} (GNN: {c3_scores_partial[best_c3_local_idx] * 100:.1f}%)")

        return recommendations

    def _get_training_graph_edges(self):

        edges_list = []
        edge_types_list = []

        # Mentor edges (professor → thesis)
        if ('professor', 'mentors', 'thesis') in self.graph_data.edge_types:
            mentor_edges = self.graph_data[('professor', 'mentors', 'thesis')].edge_index
            edges_list.append(mentor_edges)
            edge_types_list.extend([EDGE_TYPE_MENTOR] * mentor_edges.shape[1])

        # C2 edges (professor → thesis)
        if ('professor', 'serves_as_c2', 'thesis') in self.graph_data.edge_types:
            c2_edges = self.graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index
            edges_list.append(c2_edges)
            edge_types_list.extend([EDGE_TYPE_C2] * c2_edges.shape[1])

        # C3 edges (professor → thesis)
        if ('professor', 'serves_as_c3', 'thesis') in self.graph_data.edge_types:
            c3_edges = self.graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index
            edges_list.append(c3_edges)
            edge_types_list.extend([EDGE_TYPE_C3] * c3_edges.shape[1])

        # Research edges (professor → thesis)
        if ('professor', 'researches', 'thesis') in self.graph_data.edge_types:
            research_edges = self.graph_data[('professor', 'researches', 'thesis')].edge_index
            edges_list.append(research_edges)
            edge_types_list.extend([EDGE_TYPE_RESEARCH] * research_edges.shape[1])

        # Collaboration edges (professor → professor)
        if ('professor', 'collaborates', 'professor') in self.graph_data.edge_types:
            collab_edges = self.graph_data[('professor', 'collaborates', 'professor')].edge_index
            edges_list.append(collab_edges)
            edge_types_list.extend([EDGE_TYPE_COLLABORATION] * collab_edges.shape[1])

        # Concatenate all edges
        if not edges_list:
            print("⚠ No training edges found!")
            # Return empty tensors
            return torch.empty((2, 0), dtype=torch.long, device=device), \
                torch.empty(0, dtype=torch.long, device=device)

        all_edges = torch.cat(edges_list, dim=1).to(device)
        all_edge_types = torch.tensor(edge_types_list, dtype=torch.long, device=device)

        return all_edges, all_edge_types






recommender = RoleSpecificThesisRecommender()


@app.route('/')
def index():

    return render_template('index_gnn.html')


@app.route("/recommend_professors", methods=["POST"])
def recommend_professors():

    try:
        body = request.json

        thesis_input = body.get("thesis", "").strip()
        primary_mentor_raw = body.get("primary_mentor")
        primary_mentor = primary_mentor_raw.strip() if primary_mentor_raw else None

        if not thesis_input:
            return jsonify({"error": "Please provide a thesis topic"}), 400

        if not recommender.gnn_loaded:
            return jsonify({
                "error": "Role-Specific GNN not loaded. Train with layers3_roles.py first."
            }), 500

        print(f" Thesis: '{thesis_input[:60]}...'")
        if primary_mentor:
            print(f" Selected Primary Mentor: {primary_mentor}")
            print(f" GNN will select C2 and C3 from mentor's collaborators")
        else:
            print(f" No mentor - GNN will select Mentor, C2, and C3")

        # Get recommendations
        committee = recommender.recommend_committee(
            thesis_input,
            primary_mentor=primary_mentor
        )

        if not committee:
            return jsonify({
                "error": "Failed to generate recommendations"
            }), 500

        print(f" Generated {len(committee)} recommendations")

        for rec in committee:
            print(f"   - {rec['role']}: {rec['professor']} ({rec['compatibility_score']:.1f}%)")

        response = {
            "thesis": thesis_input,
            "primary_mentor_selected": primary_mentor is not None,
            "selected_mentor": primary_mentor,
            "recommendations": committee,
            "method": "Role-Specific GNN (Mentor, C2, C3)",
            "model_info": {
                "approach": "Role-specific multi-task learning with RGCN",
                "heads": "3 separate heads (Mentor, C2, C3)",
                "edge_types": "5 types (mentor, c2, c3, research, collaboration)",
                "features": [
                    "GNN learns specific committee roles",
                    "Separate prediction heads for each role",
                    "Edge-aware message passing (5 edge types)",
                    "Mode 1: GNN picks Mentor + C2 + C3",
                    "Mode 2: User picks Mentor, GNN picks C2 + C3 from collaborators"
                ]
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f" Error in recommend_professors: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/professors", methods=["GET"])
def get_professors():

    try:
        if not recommender.mappings:
            return jsonify({"error": "System not loaded properly"}), 500

        professors = recommender.mappings['professors']
        return jsonify({
            "professors": [p for p in professors if p],
            "total_count": len([p for p in professors if p])
        })
    except Exception as e:
        return jsonify({"error": f"Error loading professors: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def status():

    try:
        status_info = {
            "system_loaded": recommender.system_loaded,
            "gnn_available": GNN_AVAILABLE,
            "gnn_loaded": recommender.gnn_loaded,
            "text_encoder_available": SENTENCE_TRANSFORMERS_AVAILABLE and recommender.text_encoder is not None,
            "professors_count": recommender.num_professors,
            "theses_count": recommender.num_theses,
            "method": "Role-Specific GNN with 3 Heads",
            "capabilities": [
                "Mode 1: No mentor - GNN selects Mentor, C2, C3",
                "Mode 2: Mentor pre-selected - GNN selects C2, C3 from collaborators",
                "Role-aware predictions (not just 'top 3')",
                "3 separate prediction heads for specific roles",
                "Edge-aware message passing with 5 edge types"
            ]
        }

        return jsonify(status_info)

    except Exception as e:
        return jsonify({"error": f"Status error: {str(e)}"}), 500


if __name__ == '__main__':


    if recommender.system_loaded:
        print(f"\n System loaded successfully!")
        print(f"   - Professors: {recommender.num_professors}")
        print(f"   - Theses: {recommender.num_theses}")
        print(f"   - GNN Model: {'Loaded ✓' if recommender.gnn_loaded else 'Not available ✗'}")
        print(f"   - Text Encoder: {'Available ✓' if recommender.text_encoder else 'Not available ✗'}")

        if recommender.gnn_loaded:
            print(f"\n    Role-Specific GNN:")
            print(f"    3 separate heads: Mentor, C2, C3")
            print(f"    GNN decides WHO should have WHICH role")
            print(f"    Mode 1: GNN picks all 3 roles")
            print(f"    Mode 2: User picks mentor, GNN picks C2 & C3")
        else:
            print(f"\n     GNN not loaded")
            print(f"   Run layers3_roles.py to train the model!")
    else:
        print("\n System failed to load!")




    app.run(host='0.0.0.0', port=5000, debug=True)