"""
Graph Visualization for Thesis Committee Recommendation System
Visualizes the heterogeneous graph with professors, theses, and relationships
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files

import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict


def load_graph_data():
    """Load the graph and metadata"""
    graph_data = torch.load('../structure/hetero_graph_edge_labeled.pt', weights_only=False)
    with open('../structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    return graph_data, info


def create_networkx_graph(graph_data, info, max_theses=100, max_professors=None):
    """
    Convert PyTorch Geometric graph to NetworkX graph

    Args:
        max_theses: Limit number of theses to visualize (for clarity)
        max_professors: Limit professors (None = all)
    """
    G = nx.Graph()

    professors = info['mappings']['professors']
    theses = info['mappings']['theses']

    # Limit for visualization clarity
    theses_subset = theses[:max_theses] if max_theses else theses
    professors_subset = professors[:max_professors] if max_professors else professors

    print(f" Creating graph with {len(theses_subset)} theses and {len(professors_subset)} professors")

    # Add professor nodes
    for i, prof in enumerate(professors_subset):
        G.add_node(f"prof_{i}",
                   label=prof,
                   node_type='professor',
                   index=i)

    # Add thesis nodes
    for i, thesis in enumerate(theses_subset):
        # Truncate long thesis titles
        short_title = thesis[:50] + "..." if len(thesis) > 50 else thesis
        G.add_node(f"thesis_{i}",
                   label=short_title,
                   node_type='thesis',
                   index=i)

    # Add edges by type
    edge_types = {
        'mentor': graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy(),
        'c2': graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy(),
        'c3': graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy(),
        'research': graph_data[('professor', 'researches', 'thesis')].edge_index.numpy(),
        'collaboration': graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy()
    }

    # Track edge counts
    edge_counts = defaultdict(int)

    # Add mentor edges
    for i in range(edge_types['mentor'].shape[1]):
        prof_idx = edge_types['mentor'][0, i]
        thesis_idx = edge_types['mentor'][1, i]
        if thesis_idx < len(theses_subset) and prof_idx < len(professors_subset):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                       edge_type='mentor', color='red', width=3)
            edge_counts['mentor'] += 1

    # Add C2 edges
    for i in range(edge_types['c2'].shape[1]):
        prof_idx = edge_types['c2'][0, i]
        thesis_idx = edge_types['c2'][1, i]
        if thesis_idx < len(theses_subset) and prof_idx < len(professors_subset):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                       edge_type='c2', color='blue', width=2)
            edge_counts['c2'] += 1

    # Add C3 edges
    for i in range(edge_types['c3'].shape[1]):
        prof_idx = edge_types['c3'][0, i]
        thesis_idx = edge_types['c3'][1, i]
        if thesis_idx < len(theses_subset) and prof_idx < len(professors_subset):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                       edge_type='c3', color='green', width=2)
            edge_counts['c3'] += 1

    # Add research edges
    for i in range(edge_types['research'].shape[1]):
        prof_idx = edge_types['research'][0, i]
        thesis_idx = edge_types['research'][1, i]
        if thesis_idx < len(theses_subset) and prof_idx < len(professors_subset):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                       edge_type='research', color='orange', width=1)
            edge_counts['research'] += 1

    # Add collaboration edges
    for i in range(edge_types['collaboration'].shape[1]):
        prof_idx1 = edge_types['collaboration'][0, i]
        prof_idx2 = edge_types['collaboration'][1, i]
        if prof_idx1 < len(professors_subset) and prof_idx2 < len(professors_subset):
            G.add_edge(f"prof_{prof_idx1}", f"prof_{prof_idx2}",
                       edge_type='collaboration', color='purple', width=2)
            edge_counts['collaboration'] += 1

    print(f" Graph created:")
    print(f"   - Nodes: {G.number_of_nodes()} ({len(professors_subset)} professors, {len(theses_subset)} theses)")
    print(f"   - Edges: {G.number_of_edges()}")
    print(f"   - Mentor edges: {edge_counts['mentor']}")
    print(f"   - C2 edges: {edge_counts['c2']}")
    print(f"   - C3 edges: {edge_counts['c3']}")
    print(f"   - Research edges: {edge_counts['research']}")
    print(f"   - Collaboration edges: {edge_counts['collaboration']}")

    return G


def visualize_full_graph(G, output_path='visualizations/full_graph.png'):
    """Visualize the complete heterogeneous graph"""

    plt.figure(figsize=(20, 16))

    # Separate nodes by type
    professor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'professor']
    thesis_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'thesis']

    # Use bipartite layout (professors on left, theses on right)
    pos = {}

    # Position professors on the left
    for i, node in enumerate(professor_nodes):
        pos[node] = (0, i * (len(thesis_nodes) / len(professor_nodes)))

    # Position theses on the right
    for i, node in enumerate(thesis_nodes):
        pos[node] = (3, i)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=professor_nodes,
                           node_color='lightblue',
                           node_size=500,
                           node_shape='s',  # Square
                           label='Professors')

    nx.draw_networkx_nodes(G, pos,
                           nodelist=thesis_nodes,
                           node_color='lightgreen',
                           node_size=300,
                           node_shape='o',  # Circle
                           label='Theses')

    # Draw edges by type
    edge_colors = {'mentor': 'red', 'c2': 'blue', 'c3': 'green',
                   'research': 'orange', 'collaboration': 'purple'}

    for edge_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == edge_type]
        if edges:
            widths = [G[u][v].get('width', 1) for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   edge_color=color, width=widths,
                                   alpha=0.6, label=edge_type.capitalize())

    plt.title("Heterogeneous Graph: Professors, Theses, and Relationships",
              fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved full graph visualization to {output_path}")
    plt.close()


def visualize_single_thesis_subgraph(G, thesis_idx=0, output_path='visualizations/thesis_subgraph.png'):
    """Visualize committee for a single thesis"""

    thesis_node = f"thesis_{thesis_idx}"

    if thesis_node not in G:
        print(f" Thesis {thesis_idx} not in graph")
        return

    # Get thesis label
    thesis_label = G.nodes[thesis_node]['label']

    # Get committee members
    neighbors = list(G.neighbors(thesis_node))

    if not neighbors:
        print(f" Thesis {thesis_idx} has no committee members in the limited graph")
        return

    # Create subgraph
    nodes_to_include = [thesis_node] + neighbors
    subgraph = G.subgraph(nodes_to_include)

    plt.figure(figsize=(12, 10))

    # Circular layout with thesis in center
    pos = nx.spring_layout(subgraph, k=2, iterations=50)

    # Put thesis in center
    pos[thesis_node] = np.array([0, 0])

    # Draw thesis node (larger, center)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=[thesis_node],
                           node_color='gold',
                           node_size=3000,
                           node_shape='o')

    # Draw professor nodes
    professor_nodes = [n for n in neighbors if G.nodes[n]['node_type'] == 'professor']

    # Color by role
    node_colors = []
    for node in professor_nodes:
        edge_data = G[thesis_node][node]
        if edge_data['edge_type'] == 'mentor':
            node_colors.append('red')
        elif edge_data['edge_type'] == 'c2':
            node_colors.append('blue')
        elif edge_data['edge_type'] == 'c3':
            node_colors.append('green')
        else:
            node_colors.append('gray')

    if professor_nodes:
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=professor_nodes,
                               node_color=node_colors,
                               node_size=2000,
                               node_shape='s')

    # Draw edges
    for u, v, d in subgraph.edges(data=True):
        edge_color = d.get('color', 'gray')
        edge_width = d.get('width', 2)
        nx.draw_networkx_edges(subgraph, pos,
                               edgelist=[(u, v)],
                               edge_color=edge_color,
                               width=edge_width,
                               alpha=0.7)

    # Draw labels
    labels = {}
    labels[thesis_node] = f"Thesis:\n{thesis_label[:40]}..."
    for node in professor_nodes:
        edge_type = G[thesis_node][node]['edge_type']
        prof_label = G.nodes[node]['label']
        labels[node] = f"{prof_label}\n({edge_type.upper()})"

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9)

    plt.title(f"Committee for Thesis {thesis_idx}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved thesis subgraph to {output_path}")
    plt.close()


def visualize_statistics(G, output_path='visualizations/graph_statistics.png'):
    """Visualize graph statistics"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Degree distribution
    degrees = [d for n, d in G.degree()]
    axes[0, 0].hist(degrees, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Degree', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # 2. Professor supervision counts
    professor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'professor']
    supervision_counts = []
    for prof in professor_nodes:
        thesis_neighbors = [n for n in G.neighbors(prof)
                            if G.nodes[n]['node_type'] == 'thesis']
        supervision_counts.append(len(thesis_neighbors))

    axes[0, 1].hist(supervision_counts, bins=20, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Theses Supervised', fontsize=12)
    axes[0, 1].set_ylabel('Number of Professors', fontsize=12)
    axes[0, 1].set_title('Professor Supervision Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # 3. Edge type distribution
    edge_types = [d['edge_type'] for u, v, d in G.edges(data=True)]
    edge_type_counts = {}
    for et in edge_types:
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

    axes[1, 0].bar(edge_type_counts.keys(), edge_type_counts.values(),
                   color=['red', 'blue', 'green', 'orange', 'purple'])
    axes[1, 0].set_xlabel('Edge Type', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Edge Type Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3)

    # 4. Committee size distribution
    thesis_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'thesis']
    committee_sizes = []
    for thesis in thesis_nodes:
        prof_neighbors = [n for n in G.neighbors(thesis)
                          if G.nodes[n]['node_type'] == 'professor']
        committee_sizes.append(len(prof_neighbors))

    max_size = max(committee_sizes) if committee_sizes else 5
    axes[1, 1].hist(committee_sizes, bins=range(0, max_size + 2),
                    color='lightgreen', edgecolor='black')
    axes[1, 1].set_xlabel('Committee Size', fontsize=12)
    axes[1, 1].set_ylabel('Number of Theses', fontsize=12)
    axes[1, 1].set_title('Thesis Committee Size Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved statistics to {output_path}")
    plt.close()


def create_all_visualizations():
    """Generate all visualizations"""

    import os
    os.makedirs('visualizations', exist_ok=True)



    print("\n Loading graph data...")
    graph_data, info = load_graph_data()

    print("\n Creating NetworkX graph...")
    G = create_networkx_graph(graph_data, info, max_theses=100, max_professors=50)

    print("\n Generating visualizations...")

    print("\n1. Full heterogeneous graph...")
    visualize_full_graph(G, 'visualizations/1_full_graph.png')

    print("\n2. Single thesis committee with FULL committee (Mentor + C2 + C3)...")

    # Find a thesis with ALL three committee members (mentor, c2, c3)
    thesis_found = False
    for idx in range(100):
        thesis_node = f"thesis_{idx}"
        if thesis_node not in G:
            continue

        # Get all professor neighbors
        prof_neighbors = [n for n in G.neighbors(thesis_node)
                          if G.nodes[n]['node_type'] == 'professor']

        if not prof_neighbors:
            continue

        # Check what roles are present
        has_mentor = False
        has_c2 = False
        has_c3 = False

        for neighbor in prof_neighbors:
            edge_type = G[thesis_node][neighbor]['edge_type']
            if edge_type == 'mentor':
                has_mentor = True
            elif edge_type == 'c2':
                has_c2 = True
            elif edge_type == 'c3':
                has_c3 = True

        # Only visualize if ALL three roles are present
        if has_mentor and has_c2 and has_c3:
            print(f"   ✓ Found thesis {idx} with complete committee (Mentor + C2 + C3)")
            visualize_single_thesis_subgraph(G, thesis_idx=idx,
                                             output_path='visualizations/2_thesis_committee.png')
            thesis_found = True
            break

    if not thesis_found:
        print("   ️ No thesis with FULL committee (Mentor + C2 + C3) found in limited graph")
        print("    Try increasing max_theses or max_professors parameters")

    print("\n3. Graph statistics...")
    visualize_statistics(G, 'visualizations/3_statistics.png')

    print("\n" + "=" * 70)
    print(" ALL VISUALIZATIONS CREATED!")
    print("=" * 70)
    print(f"\nSaved to 'visualizations/' directory:")
    print("  1. 1_full_graph.png - Complete heterogeneous graph")
    print("  2. 2_thesis_committee.png - Example thesis with FULL committee")
    print("  3. 3_statistics.png - Graph statistics and distributions")


if __name__ == "__main__":
    create_all_visualizations()