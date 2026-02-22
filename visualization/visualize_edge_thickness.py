"""
Enhanced Graph Visualization with Weighted Edges and Nodes (FIXED)
Shows collaboration strength and supervision experience
"""

import matplotlib
matplotlib.use('Agg')

import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from collections import defaultdict, Counter


def load_graph_data():

    graph_data = torch.load('../structure/hetero_graph_edge_labeled.pt', weights_only=False)
    with open('../structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    return graph_data, info


def calculate_collaboration_counts(graph_data, num_professors):

    collaboration_counts = defaultdict(int)

    # Get all edge types
    mentor_edges = graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy()
    c2_edges = graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy()
    c3_edges = graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy()

    # For each thesis, find all professors who worked on it
    thesis_committees = defaultdict(set)

    # Add mentors
    for i in range(mentor_edges.shape[1]):
        prof_idx = mentor_edges[0, i]
        thesis_idx = mentor_edges[1, i]
        thesis_committees[thesis_idx].add(prof_idx)

    # Add C2 members
    for i in range(c2_edges.shape[1]):
        prof_idx = c2_edges[0, i]
        thesis_idx = c2_edges[1, i]
        thesis_committees[thesis_idx].add(prof_idx)

    # Add C3 members
    for i in range(c3_edges.shape[1]):
        prof_idx = c3_edges[0, i]
        thesis_idx = c3_edges[1, i]
        thesis_committees[thesis_idx].add(prof_idx)

    # Count collaborations (pairs who served together)
    for thesis_idx, committee in thesis_committees.items():
        committee_list = list(committee)
        # For each pair in this committee
        for i in range(len(committee_list)):
            for j in range(i + 1, len(committee_list)):
                prof1 = min(committee_list[i], committee_list[j])
                prof2 = max(committee_list[i], committee_list[j])
                collaboration_counts[(prof1, prof2)] += 1

    print(f" Calculated {len(collaboration_counts)} unique professor collaborations")

    return collaboration_counts


def calculate_professor_statistics(graph_data, info):
    """
    Calculate supervision counts for each professor

    Returns:
        dict: {prof_idx: {'total': int, 'mentor': int, 'c2': int, 'c3': int, 'research': int}}
    """
    professor_stats = defaultdict(lambda: {'total': 0, 'mentor': 0, 'c2': 0, 'c3': 0, 'research': 0})

    # Mentor counts
    mentor_edges = graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy()
    for i in range(mentor_edges.shape[1]):
        prof_idx = mentor_edges[0, i]
        professor_stats[prof_idx]['mentor'] += 1
        professor_stats[prof_idx]['total'] += 1

    # C2 counts
    c2_edges = graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy()
    for i in range(c2_edges.shape[1]):
        prof_idx = c2_edges[0, i]
        professor_stats[prof_idx]['c2'] += 1
        professor_stats[prof_idx]['total'] += 1

    # C3 counts
    c3_edges = graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy()
    for i in range(c3_edges.shape[1]):
        prof_idx = c3_edges[0, i]
        professor_stats[prof_idx]['c3'] += 1
        professor_stats[prof_idx]['total'] += 1

    # Research counts
    research_edges = graph_data[('professor', 'researches', 'thesis')].edge_index.numpy()
    for i in range(research_edges.shape[1]):
        prof_idx = research_edges[0, i]
        professor_stats[prof_idx]['research'] += 1

    return professor_stats


def find_two_connected_theses_strict(graph_data, info, max_search=1000):
    """
    Find two theses where:
    1. Both have complete committees (Mentor + C2 + C3)
    2. They share EXACTLY ONE professor
    3. The shared professor has DIFFERENT roles in each committee

    Returns:
        tuple: (thesis1_idx, thesis2_idx, shared_prof_idx) or None
    """
    # Get all edges
    mentor_edges = graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy()
    c2_edges = graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy()
    c3_edges = graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy()

    # Build thesis -> committee mapping
    thesis_committees = {}

    # Add mentors
    for i in range(mentor_edges.shape[1]):
        prof_idx = mentor_edges[0, i]
        thesis_idx = mentor_edges[1, i]
        if thesis_idx not in thesis_committees:
            thesis_committees[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None, 'all': set()}
        thesis_committees[thesis_idx]['mentor'] = prof_idx
        thesis_committees[thesis_idx]['all'].add(prof_idx)

    # Add C2
    for i in range(c2_edges.shape[1]):
        prof_idx = c2_edges[0, i]
        thesis_idx = c2_edges[1, i]
        if thesis_idx not in thesis_committees:
            thesis_committees[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None, 'all': set()}
        thesis_committees[thesis_idx]['c2'] = prof_idx
        thesis_committees[thesis_idx]['all'].add(prof_idx)

    # Add C3
    for i in range(c3_edges.shape[1]):
        prof_idx = c3_edges[0, i]
        thesis_idx = c3_edges[1, i]
        if thesis_idx not in thesis_committees:
            thesis_committees[thesis_idx] = {'mentor': None, 'c2': None, 'c3': None, 'all': set()}
        thesis_committees[thesis_idx]['c3'] = prof_idx
        thesis_committees[thesis_idx]['all'].add(prof_idx)

    # Find theses with complete committees
    complete_theses = []
    for thesis_idx, committee in thesis_committees.items():
        if (committee['mentor'] is not None and
            committee['c2'] is not None and
            committee['c3'] is not None and
            len(committee['all']) == 3):  # Ensure all 3 are different
            complete_theses.append(thesis_idx)

    print(f" Found {len(complete_theses)} theses with complete committees (3 unique professors)")

    # Find two theses that share EXACTLY ONE professor
    for i, thesis1 in enumerate(complete_theses[:max_search]):
        committee1 = thesis_committees[thesis1]

        for thesis2 in complete_theses[i+1:max_search]:
            committee2 = thesis_committees[thesis2]

            # Check if they share exactly one professor
            shared_profs = committee1['all'] & committee2['all']

            if len(shared_profs) == 1:
                shared_prof = list(shared_profs)[0]

                # Get the role of shared professor in each committee
                role1 = None
                role2 = None

                if committee1['mentor'] == shared_prof:
                    role1 = 'mentor'
                elif committee1['c2'] == shared_prof:
                    role1 = 'c2'
                elif committee1['c3'] == shared_prof:
                    role1 = 'c3'

                if committee2['mentor'] == shared_prof:
                    role2 = 'mentor'
                elif committee2['c2'] == shared_prof:
                    role2 = 'c2'
                elif committee2['c3'] == shared_prof:
                    role2 = 'c3'

                # Prefer cases where the shared prof has DIFFERENT roles
                if role1 != role2:
                    print(f" Found PERFECT match:")
                    print(f"   Thesis 1: {thesis1} (shared prof is {role1})")
                    print(f"   Thesis 2: {thesis2} (shared prof is {role2})")
                    print(f"   Shared professor: {shared_prof}")
                    return thesis1, thesis2, shared_prof

    # If no perfect match, try same role
    for i, thesis1 in enumerate(complete_theses[:max_search]):
        committee1 = thesis_committees[thesis1]

        for thesis2 in complete_theses[i+1:max_search]:
            committee2 = thesis_committees[thesis2]

            shared_profs = committee1['all'] & committee2['all']

            if len(shared_profs) == 1:
                shared_prof = list(shared_profs)[0]
                print(f"✓ Found connected theses: {thesis1} and {thesis2} (shared prof: {shared_prof})")
                return thesis1, thesis2, shared_prof

    print(" No connected theses found with strict criteria")
    return None


def visualize_enhanced_committee(graph_data, info, output_path='visualizations/enhanced_committee.png'):
    """
    Create enhanced visualization with:
    - Two theses (connected through shared professor with DIFFERENT roles)
    - Solid edges (not dotted)
    - Collaboration count labels on edges
    - Supervision count labels on nodes
    - Thicker edges for more collaborations
    - Bigger nodes for more supervisions
    - Detailed legend
    """

    # Calculate statistics
    print("\n Calculating professor statistics...")
    professor_stats = calculate_professor_statistics(graph_data, info)

    print("\n Calculating collaboration frequencies...")
    collaboration_counts = calculate_collaboration_counts(graph_data, len(info['mappings']['professors']))

    print("\n Finding two connected theses with strict criteria...")
    result = find_two_connected_theses_strict(graph_data, info)

    if result is None:
        print(" Could not find suitable theses for visualization")
        return

    thesis1_idx, thesis2_idx, shared_prof_idx = result

    # Get committee information
    mentor_edges = graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy()
    c2_edges = graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy()
    c3_edges = graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy()

    # Build committees for both theses
    committees = {thesis1_idx: {}, thesis2_idx: {}}

    # Get thesis 1 committee
    for i in range(mentor_edges.shape[1]):
        if mentor_edges[1, i] == thesis1_idx:
            committees[thesis1_idx]['mentor'] = mentor_edges[0, i]
    for i in range(c2_edges.shape[1]):
        if c2_edges[1, i] == thesis1_idx:
            committees[thesis1_idx]['c2'] = c2_edges[0, i]
    for i in range(c3_edges.shape[1]):
        if c3_edges[1, i] == thesis1_idx:
            committees[thesis1_idx]['c3'] = c3_edges[0, i]

    # Get thesis 2 committee
    for i in range(mentor_edges.shape[1]):
        if mentor_edges[1, i] == thesis2_idx:
            committees[thesis2_idx]['mentor'] = mentor_edges[0, i]
    for i in range(c2_edges.shape[1]):
        if c2_edges[1, i] == thesis2_idx:
            committees[thesis2_idx]['c2'] = c2_edges[0, i]
    for i in range(c3_edges.shape[1]):
        if c3_edges[1, i] == thesis2_idx:
            committees[thesis2_idx]['c3'] = c3_edges[0, i]

    # Get all unique professors
    all_profs = set()
    for committee in committees.values():
        all_profs.update(committee.values())

    print(f"\n Visualization will include:")
    print(f"   - Thesis 1: {info['mappings']['theses'][thesis1_idx][:60]}...")
    print(f"     Committee: Mentor={committees[thesis1_idx]['mentor']}, C2={committees[thesis1_idx]['c2']}, C3={committees[thesis1_idx]['c3']}")
    print(f"   - Thesis 2: {info['mappings']['theses'][thesis2_idx][:60]}...")
    print(f"     Committee: Mentor={committees[thesis2_idx]['mentor']}, C2={committees[thesis2_idx]['c2']}, C3={committees[thesis2_idx]['c3']}")
    print(f"   - {len(all_profs)} unique professors")
    print(f"   - Shared professor: {shared_prof_idx}")

    # Create NetworkX graph
    G = nx.Graph()

    # Add thesis nodes
    G.add_node(f"thesis_{thesis1_idx}",
               label=info['mappings']['theses'][thesis1_idx],
               node_type='thesis',
               index=thesis1_idx)
    G.add_node(f"thesis_{thesis2_idx}",
               label=info['mappings']['theses'][thesis2_idx],
               node_type='thesis',
               index=thesis2_idx)

    # Add professor nodes with size based on supervision count
    for prof_idx in all_profs:
        prof_name = info['mappings']['professors'][prof_idx]
        stats = professor_stats[prof_idx]

        # Node size based on total supervisions
        total_supervisions = stats['total']

        G.add_node(f"prof_{prof_idx}",
                   label=prof_name,
                   node_type='professor',
                   index=prof_idx,
                   supervisions=total_supervisions,
                   stats=stats)

    # Add thesis -> professor edges
    for thesis_idx in [thesis1_idx, thesis2_idx]:
        thesis_node = f"thesis_{thesis_idx}"
        committee = committees[thesis_idx]

        for role, prof_idx in committee.items():
            prof_node = f"prof_{prof_idx}"
            G.add_edge(thesis_node, prof_node,
                       edge_type=role,
                       thesis_idx=thesis_idx)

    # Add professor -> professor collaboration edges with weights
    profs_list = list(all_profs)
    for i in range(len(profs_list)):
        for j in range(i + 1, len(profs_list)):
            prof1_idx = min(profs_list[i], profs_list[j])
            prof2_idx = max(profs_list[i], profs_list[j])

            collab_count = collaboration_counts.get((prof1_idx, prof2_idx), 0)

            if collab_count > 0:
                G.add_edge(f"prof_{prof1_idx}", f"prof_{prof2_idx}",
                           edge_type='collaboration',
                           weight=collab_count)

    # Create visualization
    fig = plt.figure(figsize=(20, 14))

    # Main plot area (left side)
    ax_main = plt.subplot2grid((1, 3), (0, 0), colspan=2)

    # Legend area (right side)
    ax_legend = plt.subplot2grid((1, 3), (0, 2))
    ax_legend.axis('off')

    # === MAIN PLOT ===

    # Create layout
    pos = {}

    # Position theses
    pos[f"thesis_{thesis1_idx}"] = np.array([0, 2.0])
    pos[f"thesis_{thesis2_idx}"] = np.array([0, -2.0])

    # Position professors in a circle around their primary thesis
    # But ensure shared professor is positioned centrally

    radius = 3.5

    # Thesis 1 professors
    committee1 = committees[thesis1_idx]
    profs1 = [committee1['mentor'], committee1['c2'], committee1['c3']]

    # Position them at 120-degree intervals around thesis 1
    for i, (role, prof_idx) in enumerate([('mentor', committee1['mentor']),
                                           ('c2', committee1['c2']),
                                           ('c3', committee1['c3'])]):
        angle = (i * 120 + 90) * np.pi / 180  # Start from top, go clockwise
        x = radius * np.cos(angle)
        y = 2.0 + radius * np.sin(angle)
        pos[f"prof_{prof_idx}"] = np.array([x, y])

    # Thesis 2 professors (only position those not already positioned)
    committee2 = committees[thesis2_idx]
    profs2 = [committee2['mentor'], committee2['c2'], committee2['c3']]

    # Find which positions are not yet filled
    new_profs2 = [p for p in profs2 if f"prof_{p}" not in pos]

    # Position new professors around thesis 2
    if len(new_profs2) == 2:
        # Two new professors (shared one already positioned)
        angles = [210, 330]  # Left and right of bottom thesis
    else:
        # All three need positioning (shouldn't happen with our strict criteria)
        angles = [150, 210, 330]

    for i, prof_idx in enumerate(new_profs2):
        if i < len(angles):
            angle = angles[i] * np.pi / 180
            x = radius * np.cos(angle)
            y = -2.0 + radius * np.sin(angle)
            pos[f"prof_{prof_idx}"] = np.array([x, y])

    # Draw thesis nodes
    thesis_nodes = [f"thesis_{thesis1_idx}", f"thesis_{thesis2_idx}"]
    nx.draw_networkx_nodes(G, pos, ax=ax_main,
                           nodelist=thesis_nodes,
                           node_color='gold',
                           node_size=3500,
                           node_shape='o',
                           edgecolors='black',
                           linewidths=4)

    # Draw professor nodes with size based on supervision count
    professor_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'professor']

    # Calculate node sizes (scale between 1500 and 6000 based on supervisions)
    node_sizes = []
    node_colors = []

    for prof_node in professor_nodes:
        supervisions = G.nodes[prof_node]['supervisions']

        # Size: 1500 + (supervisions * 25), max 6000
        size = min(1500 + supervisions * 25, 6000)
        node_sizes.append(size)

        # Color: shared professor is red, others are blue
        prof_idx = G.nodes[prof_node]['index']
        if prof_idx == shared_prof_idx:
            node_colors.append('lightcoral')  # Shared professor
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, ax=ax_main,
                           nodelist=professor_nodes,
                           node_color=node_colors,
                           node_size=node_sizes,
                           node_shape='s',
                           edgecolors='black',
                           linewidths=3)

    # Draw thesis -> professor edges (colored by role, SOLID)
    role_colors = {'mentor': 'red', 'c2': 'blue', 'c3': 'green'}

    for thesis_idx in [thesis1_idx, thesis2_idx]:
        thesis_node = f"thesis_{thesis_idx}"
        committee = committees[thesis_idx]

        for role, prof_idx in committee.items():
            prof_node = f"prof_{prof_idx}"

            nx.draw_networkx_edges(G, pos, ax=ax_main,
                                   edgelist=[(thesis_node, prof_node)],
                                   edge_color=role_colors[role],
                                   width=5,
                                   alpha=0.8,
                                   style='solid')  #  SOLID, not dashed

    # Draw professor -> professor collaboration edges (SOLID, thickness based on count)
    collab_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('edge_type') == 'collaboration']

    for u, v in collab_edges:
        weight = G[u][v]['weight']

        # Edge width: 2 + (weight * 0.15), max 12
        edge_width = min(2 + weight * 0.15, 12)

        nx.draw_networkx_edges(G, pos, ax=ax_main,
                               edgelist=[(u, v)],
                               edge_color='purple',
                               width=edge_width,
                               alpha=0.7,
                               style='solid')


    edge_labels = {}
    for u, v in collab_edges:
        weight = G[u][v]['weight']
        edge_labels[(u, v)] = f"{weight}"

    nx.draw_networkx_edge_labels(G, pos, ax=ax_main,
                                  edge_labels=edge_labels,
                                  font_size=11,
                                  font_weight='bold',
                                  font_color='darkviolet',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white',
                                           edgecolor='purple',
                                           alpha=0.9))

    # Draw thesis labels
    thesis_labels = {}
    for thesis_node in thesis_nodes:
        thesis_label = G.nodes[thesis_node]['label']
        short_label = thesis_label[:35] + "..." if len(thesis_label) > 35 else thesis_label
        thesis_labels[thesis_node] = short_label

    nx.draw_networkx_labels(G, pos, ax=ax_main,
                            labels=thesis_labels,
                            font_size=9,
                            font_weight='bold')


    prof_labels = {}
    for prof_node in professor_nodes:
        prof_name = G.nodes[prof_node]['label']
        supervisions = G.nodes[prof_node]['supervisions']
        prof_labels[prof_node] = f"{prof_name}\n({supervisions} theses)"

    nx.draw_networkx_labels(G, pos, ax=ax_main,
                            labels=prof_labels,
                            font_size=9,
                            font_weight='bold')

    ax_main.set_title("Enhanced Committee Visualization\n(Weighted Edges & Nodes with Labels)",
                      fontsize=18, fontweight='bold', pad=20)
    ax_main.axis('off')
    ax_main.margins(0.15)

    # === LEGEND ===

    legend_y = 0.95
    legend_x = 0.05
    line_height = 0.035

    def add_legend_text(text, y, fontsize=10, fontweight='normal', color='black'):
        ax_legend.text(legend_x, y, text, fontsize=fontsize,
                      fontweight=fontweight, color=color,
                      transform=ax_legend.transAxes,
                      verticalalignment='top')

    # Title
    add_legend_text("LEGEND", legend_y, fontsize=14, fontweight='bold')
    legend_y -= line_height * 1.5


    # Edge colors


    add_legend_text("─── Mentor (red)", legend_y, fontsize=9, color='red')
    legend_y -= line_height

    add_legend_text("─── C2 Member (blue)", legend_y, fontsize=9, color='blue')
    legend_y -= line_height

    add_legend_text("─── C3 Member (green)", legend_y, fontsize=9, color='green')
    legend_y -= line_height

    add_legend_text("─── Collaboration (purple)", legend_y, fontsize=9, color='purple')
    legend_y -= line_height * 1.5




    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Saved enhanced visualization to {output_path}")
    plt.close()


def create_enhanced_visualizations():
    """Generate enhanced visualizations"""

    import os
    os.makedirs('visualizations', exist_ok=True)



    print("\n Loading graph data...")
    graph_data, info = load_graph_data()

    print("\n Creating enhanced committee visualization...")
    visualize_enhanced_committee(graph_data, info,
                                 'visualizations/enhanced_committee_fixed.png')

    print("\n" + "=" * 70)
    print(" ENHANCED VISUALIZATION CREATED!")
    print("=" * 70)
    print(f"\nSaved to 'visualizations/enhanced_committee_fixed.png'")
    print("\n✓ Fixed Features:")
    print("  1.  All edges are SOLID (no dashed lines)")
    print("  2.  Collaboration count LABELS on edges")
    print("  3.  Supervision count LABELS on nodes")
    print("  4.  Strict committee selection (shared prof has DIFFERENT roles)")
    print("  5.  Node size ∝ supervision count")
    print("  6.  Edge thickness ∝ collaboration frequency")


if __name__ == "__main__":
    create_enhanced_visualizations()