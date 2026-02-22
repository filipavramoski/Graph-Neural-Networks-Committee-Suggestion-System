import pandas as pd
import numpy as np
import torch
import re
import json
from collections import Counter, defaultdict
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer


def load_datasets():


    research_df = pd.read_csv('../datasets/research.csv')
    research_df['Professor'] = research_df.get('Mentor', '').str.strip()
    research_df['Research_Title'] = research_df.get('Thesis name', '').str.strip()
    research_df['Research_Abstract'] = research_df.get('Abstract', '').str.strip()
    research_df['Research_Field'] = research_df.get('Field of study', '').str.strip()
    research_df['Publication_Date'] = research_df.get('Publication date', '').str.strip()


    research_name_fixes = {
        'Александра Каневче Дединец': 'Александра Дединец',
        'Александра Поповкса-Митровиќ': 'Александра Поповска-Митровиќ',
        'Бојан Илиоски': 'Бојан Илијоски',
        'Ефтим Здравески': 'Ефтим Здравевски',
        'Катерина Тројачанец Динева': 'Катарина Тројачанец',
        'Катерина Тројачанец': 'Катарина Тројачанец',
        'Моника Симјаноска Мишева': 'Моника Симјаноска',
        'Наташа Стојановска-Илиевска': 'Наташа Илиевска',
        'Петар Секуловски': 'Петар Секулоски'
    }
    research_df['Professor'] = research_df['Professor'].replace(research_name_fixes)


    before_remove = len(research_df)
    research_df = research_df[research_df['Professor'] != 'Андреј Ристески']
    removed_count = before_remove - len(research_df)

    print(f"    Applied {len(research_name_fixes)} name fixes in research dataset")
    if removed_count > 0:
        print(f"    Removed Андреј Ристески: {removed_count} papers")

    print("\n2. Loading committee data (commettee_final.csv)")
    mentorship_df = pd.read_csv('../datasets/commettee_final.csv')
    mentorship_df['Mentor'] = mentorship_df.get('Mentor', '').str.strip()
    mentorship_df['C2'] = mentorship_df.get('C2', '').str.strip()
    mentorship_df['C3'] = mentorship_df.get('C3', '').str.strip()
    mentorship_df['Thesis_Title'] = mentorship_df.get('Thesis Title EN', '').str.strip()
    mentorship_df['Thesis_Abstract'] = mentorship_df.get('Thesis_Desc_EN', '').str.strip()
    mentorship_df['Application_Date'] = mentorship_df.get('Thesis Application Date', '').str.strip()

    mentorship_name_fixes = {
        'Александра Поповска Митровиќ': 'Александра Поповска-Митровиќ',
        'Верица Бакева Смиљкова': 'Верица Бакева'
    }

    for col in ['Mentor', 'C2', 'C3']:
        mentorship_df[col] = mentorship_df[col].replace(mentorship_name_fixes)

    professors_to_remove = {
        'Ѓорѓи Ќосев - 3P Development',
        'Жанета Попеска',
        'Коста Митрески',
        'Татјана Зорчец',
        'Филип Блажевски',
        'Фросина Стојановска',
        'Стефан Митески',
        'Весна Киранџиска',
        'Милка Љончева',
        'Александа Лозаноска',
        'Славица Тасевска Николовска',
        'Драган Михајлов',
        'Јозеф Шпилнер',
        'Евгенија Крајчевска',
        'Катерина Русевска'

    }

    before_remove = len(mentorship_df)
    mask = (
            mentorship_df['Mentor'].isin(professors_to_remove) |
            mentorship_df['C2'].isin(professors_to_remove) |
            mentorship_df['C3'].isin(professors_to_remove)
    )
    mentorship_df = mentorship_df[~mask]
    removed_count = before_remove - len(mentorship_df)

    print(f"    Applied {len(mentorship_name_fixes)} name fixes in mentorship dataset")
    print(f"    Removed {len(professors_to_remove)} unwanted professors: {removed_count} committee records")

    # Clean data
    research_df = research_df.dropna(subset=['Professor'])
    research_df = research_df[research_df['Professor'] != '']
    research_df = research_df[research_df['Research_Title'] != '']

    mentorship_df = mentorship_df.dropna(subset=['Mentor', 'Thesis_Title'])
    mentorship_df = mentorship_df[mentorship_df['Mentor'] != '']
    mentorship_df = mentorship_df[mentorship_df['Thesis_Title'] != '']


    research_df = research_df.drop_duplicates(subset=['Professor', 'Research_Title'])
    mentorship_df = mentorship_df.drop_duplicates(subset=['Mentor', 'Thesis_Title'])

    print(f"\n Research dataset: {len(research_df)} papers")
    print(f" Mentorship dataset: {len(mentorship_df)} theses")

    return research_df, mentorship_df


def balance_research_papers(research_df, max_papers_per_prof=100):

    balanced_rows = []
    professors_capped = 0
    total_removed = 0

    for professor in research_df['Professor'].unique():
        prof_papers = research_df[research_df['Professor'] == professor]

        if len(prof_papers) <= max_papers_per_prof:
            balanced_rows.extend(prof_papers.to_dict('records'))
        else:
            professors_capped += 1
            total_removed += len(prof_papers) - max_papers_per_prof

            if 'Publication_Date' in prof_papers.columns and prof_papers['Publication_Date'].notna().any():
                prof_papers = prof_papers.sort_values('Publication_Date', ascending=False, na_position='last')
                sampled = prof_papers.head(max_papers_per_prof)
            else:
                sampled = prof_papers.sample(n=max_papers_per_prof, random_state=42)

            balanced_rows.extend(sampled.to_dict('records'))

    balanced_df = pd.DataFrame(balanced_rows)
    print(f"Capped {professors_capped} professors")
    print(f"Removed {total_removed} papers")
    print(f"Final: {len(balanced_df)} research papers")

    return balanced_df


def cap_mentored_theses(mentorship_df, max_theses_per_prof=200):

    mentor_counts = mentorship_df.groupby('Mentor').size()
    professors_to_cap = mentor_counts[mentor_counts > max_theses_per_prof].index.tolist()

    if not professors_to_cap:
        print(f" No professors exceed {max_theses_per_prof} mentored theses")
        return mentorship_df

    capped_rows = []
    total_removed = 0

    for professor in mentorship_df['Mentor'].unique():
        prof_theses = mentorship_df[mentorship_df['Mentor'] == professor]

        if len(prof_theses) <= max_theses_per_prof:
            capped_rows.append(prof_theses)
        else:
            total_removed += len(prof_theses) - max_theses_per_prof

            if 'Application_Date' in prof_theses.columns and prof_theses['Application_Date'].notna().any():
                prof_theses = prof_theses.sort_values('Application_Date', ascending=False, na_position='last')
                sampled = prof_theses.head(max_theses_per_prof)
            else:
                sampled = prof_theses.sample(n=max_theses_per_prof, random_state=42)

            capped_rows.append(sampled)

    capped_df = pd.concat(capped_rows, ignore_index=True)
    print(f" Capped {len(professors_to_cap)} professors: {', '.join(professors_to_cap)}")
    print(f" Removed {total_removed} thesis records")
    print(f" Final: {len(capped_df)} mentored theses")

    return capped_df


def create_node_mappings(research_df, mentorship_df):

    research_profs = set(research_df['Professor'].dropna())
    mentor_profs = set(mentorship_df['Mentor'].dropna())
    c2_profs = set(mentorship_df['C2'].dropna()) - {''}
    c3_profs = set(mentorship_df['C3'].dropna()) - {''}

    all_professors = sorted(list(research_profs | mentor_profs | c2_profs | c3_profs))
    all_professors = [p for p in all_professors if p]


    research_theses = set(research_df['Research_Title'].dropna())
    mentored_theses = set(mentorship_df['Thesis_Title'].dropna())

    all_theses = sorted(list(research_theses | mentored_theses))
    all_theses = [t for t in all_theses if t]


    professor_to_idx = {prof: i for i, prof in enumerate(all_professors)}
    thesis_to_idx = {thesis: i for i, thesis in enumerate(all_theses)}

    print(f" Professor nodes: {len(all_professors)}")
    print(f" Thesis nodes: {len(all_theses)}")
    print(f"   - Research papers: {len(research_theses)}")
    print(f"   - Mentored theses: {len(mentored_theses)}")
    print(f"   - Overlap: {len(research_theses & mentored_theses)}")

    return {
        'professor_to_idx': professor_to_idx,
        'thesis_to_idx': thesis_to_idx,
        'professors': all_professors,
        'theses': all_theses
    }


def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
    text = ' '.join(text.split())
    return text


def analyze_professor_overlap(research_df, mentorship_df, mappings):

    research_profs = set(research_df['Professor'].dropna())
    mentor_profs = set(mentorship_df['Mentor'].dropna())
    c2_profs = set(mentorship_df['C2'].dropna()) - {''}
    c3_profs = set(mentorship_df['C3'].dropna()) - {''}

    mentorship_all_profs = mentor_profs | c2_profs | c3_profs

    in_both = research_profs & mentorship_all_profs
    only_research = research_profs - mentorship_all_profs
    only_mentorship = mentorship_all_profs - research_profs

    total_profs = len(mappings['professors'])

    print(f"\nProfessor Distribution:")
    print(f"   Total unique professors: {total_profs}")
    print(f"    In BOTH datasets: {len(in_both)} ({len(in_both) / total_profs * 100:.1f}%)")
    print(f"    Only in RESEARCH dataset: {len(only_research)} ({len(only_research) / total_profs * 100:.1f}%)")
    print(f"    Only in MENTORSHIP dataset: {len(only_mentorship)} ({len(only_mentorship) / total_profs * 100:.1f}%)")

    print(f"\n Embedding Strategy:")
    print(
        f"   Content-based embeddings: {len(research_profs)} professors ({len(research_profs) / total_profs * 100:.1f}%)")
    print(
        f"   Random initialization: {len(only_mentorship)} professors ({len(only_mentorship) / total_profs * 100:.1f}%)")


    if only_mentorship:
        print(f"\n Sample professors with RANDOM initialization (first 10):")
        for i, prof in enumerate(sorted(list(only_mentorship))[:10], 1):
            mentor_count = len(mentorship_df[mentorship_df['Mentor'] == prof])
            c2_count = len(mentorship_df[mentorship_df['C2'] == prof])
            c3_count = len(mentorship_df[mentorship_df['C3'] == prof])
            total = mentor_count + c2_count + c3_count
            print(f"   {i}. {prof}: {total} committee roles (M:{mentor_count}, C2:{c2_count}, C3:{c3_count})")

    if in_both:
        print(f"\n Sample professors with CONTENT-BASED embeddings (first 10):")
        for i, prof in enumerate(sorted(list(in_both))[:10], 1):
            research_count = len(research_df[research_df['Professor'] == prof])
            mentor_count = len(mentorship_df[mentorship_df['Mentor'] == prof])
            c2_count = len(mentorship_df[mentorship_df['C2'] == prof])
            c3_count = len(mentorship_df[mentorship_df['C3'] == prof])
            committee_total = mentor_count + c2_count + c3_count
            print(f"   {i}. {prof}: {research_count} papers + {committee_total} committee roles")


    for i, prof in enumerate(sorted(in_both), 1):
        research_count = len(research_df[research_df['Professor'] == prof])
        mentor_count = len(mentorship_df[mentorship_df['Mentor'] == prof])
        c2_count = len(mentorship_df[mentorship_df['C2'] == prof])
        c3_count = len(mentorship_df[mentorship_df['C3'] == prof])
        committee_total = mentor_count + c2_count + c3_count
        print(f"{i:2}. {prof:<50} {research_count:>3} papers + {committee_total:>3} committee")


    if only_research:
        for i, prof in enumerate(sorted(only_research), 1):
            research_count = len(research_df[research_df['Professor'] == prof])
            print(f"{i:2}. {prof:<50} {research_count:>3} papers")
    else:
        print("   (None)")


    if only_mentorship:
        for i, prof in enumerate(sorted(only_mentorship), 1):
            mentor_count = len(mentorship_df[mentorship_df['Mentor'] == prof])
            c2_count = len(mentorship_df[mentorship_df['C2'] == prof])
            c3_count = len(mentorship_df[mentorship_df['C3'] == prof])
            total = mentor_count + c2_count + c3_count
            print(f"{i:2}. {prof:<50} {total:>3} roles (M:{mentor_count:>2}, C2:{c2_count:>2}, C3:{c3_count:>2})")
    else:
        print("   (None)")

    print("=" * 70)

    return {
        'in_both': in_both,
        'only_research': only_research,
        'only_mentorship': only_mentorship,
        'research_profs': research_profs
    }


def create_embeddings(research_df, mentorship_df, mappings):

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = 384

    print("\n1. Creating thesis embeddings first (needed for professors without research)...")
    thesis_texts = []

    for thesis_title in mappings['theses']:
        texts = [clean_text(thesis_title)]

        mentored_data = mentorship_df[mentorship_df['Thesis_Title'] == thesis_title]
        if len(mentored_data) > 0:
            row = mentored_data.iloc[0]
            if pd.notna(row.get('Thesis_Abstract', '')) and row['Thesis_Abstract'] != '':
                texts.append(clean_text(row['Thesis_Abstract']))

        research_data = research_df[research_df['Research_Title'] == thesis_title]
        if len(research_data) > 0:
            row = research_data.iloc[0]
            if pd.notna(row.get('Research_Abstract', '')) and row['Research_Abstract'] != '':
                texts.append(clean_text(row['Research_Abstract']))

        if not texts:
            texts = [clean_text(thesis_title)]

        thesis_texts.append(' '.join(texts))

    thesis_embeddings = encoder.encode(thesis_texts, convert_to_tensor=True)
    print(f"    Thesis embeddings: {thesis_embeddings.shape}")

    print("\n2. Creating professor embeddings...")
    professor_embeddings_list = []
    profs_with_research = 0
    profs_with_committee_avg = 0
    profs_with_zero_init = 0

    for prof in mappings['professors']:
        prof_papers = research_df[research_df['Professor'] == prof]

        if len(prof_papers) > 0:
            unique_fields = set()
            for _, paper in prof_papers.iterrows():
                if pd.notna(paper.get('Research_Field', '')) and paper['Research_Field'] != '':
                    fields = str(paper['Research_Field']).split(',')
                    for field in fields:
                        field = field.strip()
                        if field:
                            unique_fields.add(clean_text(field))

            if unique_fields:
                profs_with_research += 1
                professor_profile = ' '.join(sorted(unique_fields))
                embedding = encoder.encode(professor_profile, convert_to_tensor=True)
                professor_embeddings_list.append(embedding)
                continue

        committee_thesis_indices = []

        # Mentor role
        mentor_theses = mentorship_df[mentorship_df['Mentor'] == prof]['Thesis_Title'].tolist()
        for thesis_title in mentor_theses:
            if thesis_title in mappings['thesis_to_idx']:
                committee_thesis_indices.append(mappings['thesis_to_idx'][thesis_title])

        # C2 role
        c2_theses = mentorship_df[mentorship_df['C2'] == prof]['Thesis_Title'].tolist()
        for thesis_title in c2_theses:
            if thesis_title in mappings['thesis_to_idx']:
                committee_thesis_indices.append(mappings['thesis_to_idx'][thesis_title])

        # C3 role
        c3_theses = mentorship_df[mentorship_df['C3'] == prof]['Thesis_Title'].tolist()
        for thesis_title in c3_theses:
            if thesis_title in mappings['thesis_to_idx']:
                committee_thesis_indices.append(mappings['thesis_to_idx'][thesis_title])

        if committee_thesis_indices:
            profs_with_committee_avg += 1
            committee_embeddings = thesis_embeddings[committee_thesis_indices]
            embedding = committee_embeddings.mean(dim=0)
            professor_embeddings_list.append(embedding)
        else:
            profs_with_zero_init += 1
            embedding = torch.zeros(embedding_dim)
            professor_embeddings_list.append(embedding)
            print(f"     Professor '{prof}' has no research AND no committee data - zero init")

    print(f"     Professors with research fields: {profs_with_research}")
    print(f"     Professors with committee avg: {profs_with_committee_avg}")
    print(f"     Professors with zero init: {profs_with_zero_init}")

    professor_embeddings = torch.stack(professor_embeddings_list)
    print(f"     Professor embeddings: {professor_embeddings.shape}")

    print(f"\n Embedding dimension: {embedding_dim}")
    print(f"   Strategy 1 (Research fields): {profs_with_research} professors")
    print(f"   Strategy 2 (Committee avg): {profs_with_committee_avg} professors")
    print(f"   Strategy 3 (Zero init): {profs_with_zero_init} professors")

    return {
        'professor': professor_embeddings,
        'thesis': thesis_embeddings
    }, embedding_dim


def create_labeled_edges(research_df, mentorship_df, mappings):

    print("\n CREATING LABELED EDGES ")

    edges = {}

    mentor_edges = set()
    c2_edges = set()
    c3_edges = set()
    research_edges = set()
    collaboration_edges = set()

    activity_counts = {
        'mentor': Counter(),
        'c2': Counter(),
        'c3': Counter()
    }

    print("\n1. Creating mentorship edges (MENTOR, C2, C3)...")
    for _, row in mentorship_df.iterrows():
        thesis = str(row['Thesis_Title']).strip()

        if thesis not in mappings['thesis_to_idx']:
            continue

        thesis_idx = mappings['thesis_to_idx'][thesis]

        # MENTOR edge
        mentor = str(row['Mentor']).strip()
        if mentor and mentor in mappings['professor_to_idx']:
            mentor_idx = mappings['professor_to_idx'][mentor]
            mentor_edges.add((mentor_idx, thesis_idx))
            activity_counts['mentor'][mentor] += 1

        # C2 edge
        c2 = str(row.get('C2', '')).strip()
        if c2 and c2 in mappings['professor_to_idx']:
            c2_idx = mappings['professor_to_idx'][c2]
            c2_edges.add((c2_idx, thesis_idx))
            activity_counts['c2'][c2] += 1

        # C3 edge
        c3 = str(row.get('C3', '')).strip()
        if c3 and c3 in mappings['professor_to_idx']:
            c3_idx = mappings['professor_to_idx'][c3]
            c3_edges.add((c3_idx, thesis_idx))
            activity_counts['c3'][c3] += 1

        committee = []
        if mentor and mentor in mappings['professor_to_idx']:
            committee.append(mappings['professor_to_idx'][mentor])
        if c2 and c2 in mappings['professor_to_idx']:
            committee.append(mappings['professor_to_idx'][c2])
        if c3 and c3 in mappings['professor_to_idx']:
            committee.append(mappings['professor_to_idx'][c3])

        for i in range(len(committee)):
            for j in range(i + 1, len(committee)):
                edge = tuple(sorted([committee[i], committee[j]]))
                collaboration_edges.add(edge)

    print(f"    MENTOR edges: {len(mentor_edges)}")
    print(f"    C2 edges: {len(c2_edges)}")
    print(f"    C3 edges: {len(c3_edges)}")

    print(f"\n Activity statistics (for normalization):")
    print(f"   Top 5 mentors:")
    for prof, count in activity_counts['mentor'].most_common(5):
        print(f"      {prof}: {count} theses")
    print(f"   Top 5 C2:")
    for prof, count in activity_counts['c2'].most_common(5):
        print(f"      {prof}: {count} theses")
    print(f"   Top 5 C3:")
    for prof, count in activity_counts['c3'].most_common(5):
        print(f"      {prof}: {count} theses")

    print("\n2. Creating research edges (RESEARCH)...")
    for _, row in research_df.iterrows():
        professor = str(row['Professor']).strip()
        thesis = str(row['Research_Title']).strip()

        if professor in mappings['professor_to_idx'] and thesis in mappings['thesis_to_idx']:
            prof_idx = mappings['professor_to_idx'][professor]
            thesis_idx = mappings['thesis_to_idx'][thesis]
            research_edges.add((prof_idx, thesis_idx))

    print(f"    RESEARCH edges: {len(research_edges)}")

    print("\n3. Creating collaboration edges (COLLABORATES)...")
    print(f"    Unique collaborations: {len(collaboration_edges)}")

    bidirectional_collab = []
    for edge in collaboration_edges:
        bidirectional_collab.append([edge[0], edge[1]])
        bidirectional_collab.append([edge[1], edge[0]])

    print(f"    Bidirectional collaboration edges: {len(bidirectional_collab)}")

    print("\n4. Converting to PyTorch tensors...")

    if mentor_edges:
        mentor_tensor = torch.tensor(list(mentor_edges), dtype=torch.long).t()
        edges[('professor', 'mentors', 'thesis')] = mentor_tensor
        edges[('thesis', 'mentored_by', 'professor')] = torch.stack([mentor_tensor[1], mentor_tensor[0]])

    if c2_edges:
        c2_tensor = torch.tensor(list(c2_edges), dtype=torch.long).t()
        edges[('professor', 'serves_as_c2', 'thesis')] = c2_tensor
        edges[('thesis', 'has_c2', 'professor')] = torch.stack([c2_tensor[1], c2_tensor[0]])

    if c3_edges:
        c3_tensor = torch.tensor(list(c3_edges), dtype=torch.long).t()
        edges[('professor', 'serves_as_c3', 'thesis')] = c3_tensor
        edges[('thesis', 'has_c3', 'professor')] = torch.stack([c3_tensor[1], c3_tensor[0]])

    if research_edges:
        research_tensor = torch.tensor(list(research_edges), dtype=torch.long).t()
        edges[('professor', 'researches', 'thesis')] = research_tensor
        edges[('thesis', 'researched_by', 'professor')] = torch.stack([research_tensor[1], research_tensor[0]])

    if bidirectional_collab:
        collab_tensor = torch.tensor(bidirectional_collab, dtype=torch.long).t()
        edges[('professor', 'collaborates', 'professor')] = collab_tensor

    print("\n✓ Edge type summary:")
    for edge_type, edge_tensor in edges.items():
        print(f"   {edge_type}: {edge_tensor.shape[1]} edges")

    return edges, {
        'mentor': list(mentor_edges),
        'c2': list(c2_edges),
        'c3': list(c3_edges),
        'research': list(research_edges),
        'collaboration': list(collaboration_edges)
    }, activity_counts


def analyze_edge_distribution(edge_sets, mappings):

    prof_edge_counts = Counter()
    for edge_type, edges in edge_sets.items():
        if edge_type != 'collaboration':
            for prof_idx, _ in edges:
                prof_edge_counts[prof_idx] += 1

    print("\n Top 15 Most Active Professors:")
    for prof_idx, count in prof_edge_counts.most_common(15):
        prof_name = mappings['professors'][prof_idx]

        mentor_count = sum(1 for p, _ in edge_sets['mentor'] if p == prof_idx)
        c2_count = sum(1 for p, _ in edge_sets['c2'] if p == prof_idx)
        c3_count = sum(1 for p, _ in edge_sets['c3'] if p == prof_idx)
        research_count = sum(1 for p, _ in edge_sets['research'] if p == prof_idx)

        print(f"   {prof_name}: {count} total")
        print(f"      Mentor: {mentor_count}, C2: {c2_count}, C3: {c3_count}, Research: {research_count}")

    thesis_edge_counts = Counter()
    for edge_type, edges in edge_sets.items():
        if edge_type != 'collaboration':
            for _, thesis_idx in edges:
                thesis_edge_counts[thesis_idx] += 1

    print(f"\n Thesis Edge Statistics:")
    edge_counts = list(thesis_edge_counts.values())
    print(f"   Mean edges per thesis: {np.mean(edge_counts):.2f}")
    print(f"   Median: {np.median(edge_counts):.2f}")
    print(f"   Max: {max(edge_counts)}")
    print(f"   Min: {min(edge_counts)}")


def build_hetero_data(node_features, edge_indices):

    data = HeteroData()
    data['professor'].x = node_features['professor']
    data['thesis'].x = node_features['thesis']


    for edge_type, edge_index in edge_indices.items():
        data[edge_type].edge_index = edge_index

    print(f"\n HeteroData created:")
    print(f"   Node types: {data.node_types}")
    print(f"   Edge types: {data.edge_types}")
    for edge_type in data.edge_types:
        print(f"      {edge_type}: {data[edge_type].num_edges} edges")

    return data


def save_data(data, mappings, edge_sets, embedding_dim, activity_counts):


    torch.save(data, 'hetero_graph_edge_labeled.pt')
    print("✓ Saved HeteroData to 'hetero_graph_edge_labeled.pt'")

    edge_counts = {str(k): int(v.num_edges) for k, v in data.items() if hasattr(v, 'num_edges')}

    metadata = {
        'mappings': {
            'professor_to_idx': mappings['professor_to_idx'],
            'thesis_to_idx': mappings['thesis_to_idx'],
            'professors': mappings['professors'],
            'theses': mappings['theses']
        },
        'statistics': {
            'num_professors': len(mappings['professors']),
            'num_theses': len(mappings['theses']),
            'num_edges': edge_counts,
            'embedding_dim': embedding_dim,
            'edge_type_counts': {
                'mentor': len(edge_sets['mentor']),
                'c2': len(edge_sets['c2']),
                'c3': len(edge_sets['c3']),
                'research': len(edge_sets['research']),
                'collaboration': len(edge_sets['collaboration'])
            }
        },
        'activity_counts': {
            'mentor': dict(activity_counts['mentor']),
            'c2': dict(activity_counts['c2']),
            'c3': dict(activity_counts['c3'])
        }
    }

    with open('graph_metadata_edge_labeled.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(" Saved metadata to 'graph_metadata_edge_labeled.json'")

    return metadata


def main():

    research_df, mentorship_df = load_datasets()
    research_df = balance_research_papers(research_df, max_papers_per_prof=100)
    mentorship_df = cap_mentored_theses(mentorship_df, max_theses_per_prof=200)

    # Create graph structure
    mappings = create_node_mappings(research_df, mentorship_df)

    # Analyze professor overlap BEFORE creating embeddings
    overlap_info = analyze_professor_overlap(research_df, mentorship_df, mappings)

    node_features, embedding_dim = create_embeddings(research_df, mentorship_df, mappings)
    edge_indices, edge_sets,activity_counts  = create_labeled_edges(research_df, mentorship_df, mappings)

    # Analyze
    analyze_edge_distribution(edge_sets, mappings)

    # Build and save
    data = build_hetero_data(node_features, edge_indices)
    metadata = save_data(data, mappings, edge_sets, embedding_dim,activity_counts)

    # Summary

    print(f" {len(mappings['professors'])} professor nodes")
    print(f" {len(mappings['theses'])} thesis nodes")
    print(f" {len(edge_sets['mentor'])} MENTOR edges")
    print(f" {len(edge_sets['c2'])} C2 edges")
    print(f" {len(edge_sets['c3'])} C3 edges")
    print(f" {len(edge_sets['research'])} RESEARCH edges")
    print(f" {len(edge_sets['collaboration'])} unique collaborations")
    print(f" Ready for edge-labeled GNN training!")


    return data, mappings, metadata


if __name__ == "__main__":
    main()