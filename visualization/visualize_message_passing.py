"""
Visualize how message passing works in your GNN
Shows one iteration of message aggregation for thesis committee recommendation
"""

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.patches import FancyBboxPatch
import numpy as np


def visualize_message_passing_stages():
    """
    Visualize GNN message passing with a simple subgraph
    Shows 3 stages: Initial features, Message passing, Updated features
    """

    # Create simple example graph
    G = nx.Graph()

    # Central thesis
    G.add_node("Thesis A", node_type='thesis', color='gold', size=3000)

    # Committee members
    G.add_node("Prof Ана\n(Mentor)", node_type='professor', color='red', size=2000)
    G.add_node("Prof Соња\n(C2)", node_type='professor', color='blue', size=2000)
    G.add_node("Prof Билјана\n(C3)", node_type='professor', color='green', size=2000)

    # Related theses (for prof context)
    G.add_node("Thesis B", node_type='thesis', color='lightgreen', size=1500)
    G.add_node("Thesis C", node_type='thesis', color='lightgreen', size=1500)

    # Edges
    G.add_edge("Thesis A", "Prof Ана\n(Mentor)", edge_type='mentor')
    G.add_edge("Thesis A", "Prof Соња\n(C2)", edge_type='c2')
    G.add_edge("Thesis A", "Prof Билјана\n(C3)", edge_type='c3')
    G.add_edge("Prof Ана\n(Mentor)", "Thesis B", edge_type='mentor')
    G.add_edge("Prof Соња\n(C2)", "Thesis C", edge_type='c2')

    # Create 3 subplots for different stages
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Layout
    pos = {
        "Thesis A": (0, 0),
        "Prof Ана\n(Mentor)": (-2.5, 2.5),
        "Prof Соња\n(C2)": (0, 2.5),
        "Prof Билјана\n(C3)": (2.5, 2.5),
        "Thesis B": (-2.5, 5),
        "Thesis C": (0, 5)
    }

    # --- STAGE 1: Initial Features ---
    ax = axes[0]
    ax.set_title("Stage 1: Initial Node Features\n(Input Embeddings)",
                 fontsize=16, fontweight='bold', pad=20)

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add feature vectors
    ax.text(-2.5, 1.3, "h⁰ = [0.2, 0.5, 0.8, ...]", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.8),
            ha='center')
    ax.text(0, -1.3, "h⁰ = [0.8, 0.3, 0.1, ...]", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.8),
            ha='center')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                   markersize=12, label='Target Thesis'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=12, label='Mentor'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=12, label='C2 Member'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                   markersize=12, label='C3 Member'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.axis('off')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 6)

    # --- STAGE 2: Message Passing ---
    ax = axes[1]
    ax.set_title("Stage 2: Message Passing\n(Neighbor Aggregation)",
                 fontsize=16, fontweight='bold', pad=20)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Draw edges with arrows showing message flow TO thesis
    edge_colors = {'mentor': 'red', 'c2': 'blue', 'c3': 'green'}

    for u, v, d in G.edges(data=True):
        # Only show messages TO Thesis A
        if v == "Thesis A":
            source, target = u, v
            color = edge_colors.get(d.get('edge_type'), 'gray')

            # Calculate arrow positions
            x1, y1 = pos[source]
            x2, y2 = pos[target]

            # Shorten arrow to not overlap with nodes
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            dx_norm, dy_norm = dx / length, dy / length

            start_x = x1 + dx_norm * 0.3
            start_y = y1 + dy_norm * 0.3
            end_x = x2 - dx_norm * 0.3
            end_y = y2 - dy_norm * 0.3

            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=40,
                linewidth=4, color=color, alpha=0.8,
                zorder=1
            )
            ax.add_patch(arrow)

            # Add message label
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            ax.text(mid_x + 0.3, mid_y, f"m({source[:8]}→A)",
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        elif u != "Thesis A" and v != "Thesis A":
            # Draw other edges lightly
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   edge_color='gray', width=1, alpha=0.3, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add aggregation equation
    equation_text = (
        "Aggregation:\n"
        "h¹(Thesis A) = σ(W · Σ messages)\n"
        "             = σ(W · [m₁ + m₂ + m₃])"
    )
    ax.text(0, -1.8, equation_text, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.7", facecolor='lightyellow', alpha=0.9),
            ha='center', family='monospace')

    ax.axis('off')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 6)

    # --- STAGE 3: Updated Features ---
    ax = axes[2]
    ax.set_title("Stage 3: Updated Node Features\n(After One GNN Layer)",
                 fontsize=16, fontweight='bold', pad=20)

    # Make thesis node glow (updated)
    # Draw glow effect
    for i in range(5):
        alpha = 0.3 - i * 0.05
        size = 3500 + i * 200
        nx.draw_networkx_nodes(G, pos,
                               nodelist=["Thesis A"],
                               node_color='yellow',
                               node_size=size,
                               alpha=alpha, ax=ax)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=["Thesis A"],
                           node_color='gold',
                           node_size=3500,
                           alpha=1.0, ax=ax,
                           edgecolors='orange',
                           linewidths=3)

    other_nodes = [n for n in G.nodes() if n != "Thesis A"]
    other_colors = [G.nodes[n]['color'] for n in other_nodes]
    other_sizes = [G.nodes[n]['size'] for n in other_nodes]

    nx.draw_networkx_nodes(G, pos,
                           nodelist=other_nodes,
                           node_color=other_colors,
                           node_size=other_sizes,
                           alpha=0.5, ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add updated feature with highlight
    feature_text = (
        "h¹(Thesis A) = [0.7, 0.6, 0.4, ...]\n\n"
        "✓ Enriched with context from:\n"
        "  • Mentor's expertise\n"
        "  • C2 member's research\n"
        "  • C3 member's background"
    )
    ax.text(0, -2.0, feature_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.7", facecolor='lightgreen', alpha=0.9),
            ha='center')

    # Add "UPDATED!" badge
    ax.text(0, 1.2, "⭐ UPDATED ⭐", fontsize=14, fontweight='bold',
            color='orange', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

    ax.axis('off')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 6)

    plt.tight_layout()
    plt.savefig('visualizations/message_passing.png', dpi=300, bbox_inches='tight')
    print("✓ Saved message passing visualization to visualizations/message_passing.png")
    plt.close()


def visualize_multi_layer_gnn():
    """
    Visualize multiple GNN layers (3 layers like your model)
    """

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Simple graph for demonstration
    G = nx.Graph()
    nodes = ["Thesis", "Prof 1", "Prof 2", "Prof 3"]
    G.add_nodes_from(nodes)
    G.add_edges_from([
        ("Thesis", "Prof 1"),
        ("Thesis", "Prof 2"),
        ("Thesis", "Prof 3")
    ])

    pos = {
        "Thesis": (0, 0),
        "Prof 1": (-1.5, 1.5),
        "Prof 2": (0, 1.5),
        "Prof 3": (1.5, 1.5)
    }

    layers = ["Input", "Layer 1", "Layer 2", "Layer 3"]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    for idx, (ax, layer, color) in enumerate(zip(axes, layers, colors)):
        ax.set_title(f"{layer}", fontsize=16, fontweight='bold')

        # Node colors get more saturated with each layer
        node_colors = [color] * len(nodes)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=2000, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               width=2, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Add information text
        if idx == 0:
            info = "Initial\nEmbeddings"
        else:
            info = f"After {idx}\nlayer(s)"

        ax.text(0, -1.5, info, fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

        # Add receptive field info
        if idx == 0:
            receptive = "0-hop"
        elif idx == 1:
            receptive = "1-hop neighbors"
        elif idx == 2:
            receptive = "2-hop neighbors"
        else:
            receptive = "3-hop neighbors"

        ax.text(0, 2.5, f"Receptive field: {receptive}",
                fontsize=9, ha='center', style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.7))

        ax.axis('off')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 3)

    plt.suptitle("Multi-Layer GNN: Expanding Receptive Field",
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/multi_layer_gnn.png', dpi=300, bbox_inches='tight')
    print("✓ Saved multi-layer GNN visualization to visualizations/multi_layer_gnn.png")
    plt.close()


def visualize_rgcn_edge_types():
    """
    Visualize how RGCN handles different edge types
    """

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Create graph with different edge types
    G = nx.MultiDiGraph()

    # Nodes
    nodes = {
        'thesis': ["Thesis A"],
        'professor': ["Prof 1", "Prof 2", "Prof 3", "Prof 4"]
    }

    for node_type, node_list in nodes.items():
        for node in node_list:
            G.add_node(node, type=node_type)

    # Edges with types
    edges = [
        ("Prof 1", "Thesis A", "mentor", "red", 3),
        ("Prof 2", "Thesis A", "c2", "blue", 2),
        ("Prof 3", "Thesis A", "c3", "green", 2),
        ("Prof 4", "Thesis A", "research", "orange", 1),
        ("Prof 1", "Prof 2", "collaboration", "purple", 2),
    ]

    for src, dst, etype, color, width in edges:
        G.add_edge(src, dst, type=etype, color=color, width=width)

    # Layout
    pos = {
        "Thesis A": (0, 0),
        "Prof 1": (-2, 2),
        "Prof 2": (2, 2),
        "Prof 3": (-2, -2),
        "Prof 4": (2, -2)
    }

    # Draw nodes
    thesis_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'thesis']
    prof_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'professor']

    nx.draw_networkx_nodes(G, pos, nodelist=thesis_nodes,
                           node_color='gold', node_size=4000,
                           node_shape='o', ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=prof_nodes,
                           node_color='lightblue', node_size=3000,
                           node_shape='s', ax=ax)

    # Draw edges by type
    for src, dst, data in G.edges(data=True):
        edge_type = data['type']
        color = data['color']
        width = data['width'] * 2

        # Draw curved arrow
        x1, y1 = pos[src]
        x2, y2 = pos[dst]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=30,
            linewidth=width, color=color, alpha=0.7,
            connectionstyle="arc3,rad=0.1",
            zorder=1
        )
        ax.add_patch(arrow)

        # Add edge label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, edge_type.upper(),
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

    # Add RGCN equation
    equation = (
        "RGCN Message Passing:\n\n"
        "h¹(v) = σ( Σᵣ Wᵣ · Σᵤ∈Nᵣ(v) [h⁰(u) / |Nᵣ(v)|] )\n\n"
        "Where:\n"
        "• r = edge type (mentor, c2, c3, research, collaboration)\n"
        "• Wᵣ = weight matrix for edge type r\n"
        "• Nᵣ(v) = neighbors of v via edge type r\n"
        "• σ = activation function (ReLU)"
    )

    ax.text(0, -4, equation, fontsize=10, ha='center',
            family='monospace',
            bbox=dict(boxstyle="round,pad=1", facecolor='lightyellow', alpha=0.9))

    # Add title
    ax.text(0, 4, "R-GCN: Relation-Specific Message Passing",
            fontsize=18, fontweight='bold', ha='center')

    ax.text(0, 3.3, "Each edge type has its own weight matrix Wᵣ",
            fontsize=12, ha='center', style='italic')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='Mentor (W₀)'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='C2 (W₁)'),
        plt.Line2D([0], [0], color='green', linewidth=3, label='C3 (W₂)'),
        plt.Line2D([0], [0], color='orange', linewidth=3, label='Research (W₃)'),
        plt.Line2D([0], [0], color='purple', linewidth=3, label='Collaboration (W₄)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, title='Edge Types')

    ax.axis('off')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-5, 4.5)

    plt.tight_layout()
    plt.savefig('visualizations/rgcn_edge_types.png', dpi=300, bbox_inches='tight')
    print("✓ Saved RGCN edge types visualization to visualizations/rgcn_edge_types.png")
    plt.close()


def create_all_message_passing_visualizations():
    """Generate all message passing visualizations"""

    import os
    os.makedirs('visualizations', exist_ok=True)



    print("\n1. Basic message passing (3 stages)...")
    visualize_message_passing_stages()

    print("\n2. Multi-layer GNN...")
    visualize_multi_layer_gnn()

    print("\n3. RGCN edge-type-specific message passing...")
    visualize_rgcn_edge_types()


    print(f"\nSaved to 'visualizations/' directory:")
    print("  1. message_passing.png - Basic 3-stage message passing")
    print("  2. multi_layer_gnn.png - Multi-layer receptive field expansion")
    print("  3. rgcn_edge_types.png - RGCN relation-specific aggregation")


if __name__ == "__main__":
    create_all_message_passing_visualizations()