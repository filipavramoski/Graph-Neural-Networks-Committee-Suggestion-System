"""
Interactive 3D graph visualization using Plotly
Creates an HTML file with ALL edge types visible
"""

import torch
import json
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def load_graph_data():
    """Load the graph and metadata"""
    graph_data = torch.load('../structure/hetero_graph_edge_labeled.pt', weights_only=False)
    with open('../structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    return graph_data, info


def create_3d_interactive_graph_all_edges(max_theses=100, max_professors=50):
    """Create interactive 3D visualization with ALL edge types"""

    print("\n Loading graph data...")
    graph_data, info = load_graph_data()

    print(" Creating NetworkX graph with ALL edge types...")
    G = nx.Graph()

    professors = info['mappings']['professors'][:max_professors]
    theses = info['mappings']['theses'][:max_theses]

    print(f"   - Using {len(professors)} professors and {len(theses)} theses")

    # Add nodes
    for i, prof in enumerate(professors):
        G.add_node(f"prof_{i}", label=prof, node_type='professor', index=i)

    for i, thesis in enumerate(theses):
        short_title = thesis[:40] + "..." if len(thesis) > 40 else thesis
        G.add_node(f"thesis_{i}", label=short_title, node_type='thesis', index=i)

    # Add ALL edge types
    edge_types_data = {
        'mentor': (graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy(), 'red', 3),
        'c2': (graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy(), 'blue', 2),
        'c3': (graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy(), 'green', 2),
        'research': (graph_data[('professor', 'researches', 'thesis')].edge_index.numpy(), 'orange', 1),
        'collaboration': (graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy(), 'purple', 2),
    }

    edge_counts = {}

    # Add mentor edges
    for i in range(edge_types_data['mentor'][0].shape[1]):
        prof_idx = edge_types_data['mentor'][0][0, i]
        thesis_idx = edge_types_data['mentor'][0][1, i]
        if thesis_idx < len(theses) and prof_idx < len(professors):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                      edge_type='mentor', color='red', width=3)
            edge_counts['mentor'] = edge_counts.get('mentor', 0) + 1

    # Add C2 edges
    for i in range(edge_types_data['c2'][0].shape[1]):
        prof_idx = edge_types_data['c2'][0][0, i]
        thesis_idx = edge_types_data['c2'][0][1, i]
        if thesis_idx < len(theses) and prof_idx < len(professors):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                      edge_type='c2', color='blue', width=2)
            edge_counts['c2'] = edge_counts.get('c2', 0) + 1

    # Add C3 edges
    for i in range(edge_types_data['c3'][0].shape[1]):
        prof_idx = edge_types_data['c3'][0][0, i]
        thesis_idx = edge_types_data['c3'][0][1, i]
        if thesis_idx < len(theses) and prof_idx < len(professors):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                      edge_type='c3', color='green', width=2)
            edge_counts['c3'] = edge_counts.get('c3', 0) + 1

    # Add research edges
    for i in range(edge_types_data['research'][0].shape[1]):
        prof_idx = edge_types_data['research'][0][0, i]
        thesis_idx = edge_types_data['research'][0][1, i]
        if thesis_idx < len(theses) and prof_idx < len(professors):
            G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                      edge_type='research', color='orange', width=1)
            edge_counts['research'] = edge_counts.get('research', 0) + 1

    # Add collaboration edges (professor to professor)
    for i in range(edge_types_data['collaboration'][0].shape[1]):
        prof_idx1 = edge_types_data['collaboration'][0][0, i]
        prof_idx2 = edge_types_data['collaboration'][0][1, i]
        if prof_idx1 < len(professors) and prof_idx2 < len(professors):
            G.add_edge(f"prof_{prof_idx1}", f"prof_{prof_idx2}",
                      edge_type='collaboration', color='purple', width=2)
            edge_counts['collaboration'] = edge_counts.get('collaboration', 0) + 1

    print(f"\n Graph created with ALL edge types:")
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Total edges: {G.number_of_edges()}")
    print(f"   - Mentor edges: {edge_counts.get('mentor', 0)}")
    print(f"   - C2 edges: {edge_counts.get('c2', 0)}")
    print(f"   - C3 edges: {edge_counts.get('c3', 0)}")
    print(f"   - Research edges: {edge_counts.get('research', 0)}")
    print(f"   - Collaboration edges: {edge_counts.get('collaboration', 0)}")

    # 3D spring layout
    print("\n Computing 3D layout...")
    pos = nx.spring_layout(G, dim=3, k=0.5, iterations=50, seed=42)

    # Separate nodes by type
    professor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'professor']
    thesis_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'thesis']

    # Create edge traces by type
    print(" Creating edge traces...")
    edge_traces = []

    edge_type_info = {
        'mentor': ('red', 'Mentorship'),
        'c2': ('blue', 'C2 Member'),
        'c3': ('green', 'C3 Member'),
        'research': ('orange', 'Research'),
        'collaboration': ('purple', 'Collaboration')
    }

    for edge_type, (color, label) in edge_type_info.items():
        edge_x, edge_y, edge_z = [], [], []

        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == edge_type:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

        if edge_x:  # Only add if there are edges of this type
            width = 3 if edge_type == 'mentor' else (2 if edge_type in ['c2', 'c3', 'collaboration'] else 1)

            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='none',
                name=f'{label} ({edge_counts.get(edge_type, 0)})',
                showlegend=True,
                opacity=0.6,
                visible=True  # Make all edge types visible by default
            )
            edge_traces.append(edge_trace)

    # Extract coordinates for professors
    prof_x = [pos[node][0] for node in professor_nodes]
    prof_y = [pos[node][1] for node in professor_nodes]
    prof_z = [pos[node][2] for node in professor_nodes]
    prof_labels = [G.nodes[node]['label'] for node in professor_nodes]
    prof_text = [f"<b>{label}</b><br>"
                 f"Type: Professor<br>"
                 f"Total Connections: {G.degree(node)}<br>"
                 f"Supervised: {len([n for n in G.neighbors(node) if G.nodes[n]['node_type']=='thesis' and G[node][n].get('edge_type')=='mentor'])}<br>"
                 f"C2: {len([n for n in G.neighbors(node) if G.nodes[n]['node_type']=='thesis' and G[node][n].get('edge_type')=='c2'])}<br>"
                 f"C3: {len([n for n in G.neighbors(node) if G.nodes[n]['node_type']=='thesis' and G[node][n].get('edge_type')=='c3'])}<br>"
                 f"Collaborators: {len([n for n in G.neighbors(node) if G.nodes[n]['node_type']=='professor'])}"
                 for node, label in zip(professor_nodes, prof_labels)]

    # Extract coordinates for theses
    thesis_x = [pos[node][0] for node in thesis_nodes]
    thesis_y = [pos[node][1] for node in thesis_nodes]
    thesis_z = [pos[node][2] for node in thesis_nodes]
    thesis_labels = [G.nodes[node]['label'] for node in thesis_nodes]

    # Get committee info for each thesis
    thesis_text = []
    for node, label in zip(thesis_nodes, thesis_labels):
        prof_neighbors = [n for n in G.neighbors(node) if G.nodes[n]['node_type'] == 'professor']

        mentor = next((G.nodes[n]['label'] for n in prof_neighbors
                      if G[node][n].get('edge_type') == 'mentor'), 'None')
        c2_members = [G.nodes[n]['label'] for n in prof_neighbors
                     if G[node][n].get('edge_type') == 'c2']
        c3_members = [G.nodes[n]['label'] for n in prof_neighbors
                     if G[node][n].get('edge_type') == 'c3']

        text = (f"<b>{label}</b><br>"
               f"Type: Thesis<br>"
               f"Committee Size: {len(prof_neighbors)}<br>"
               f"Mentor: {mentor}<br>"
               f"C2: {', '.join(c2_members) if c2_members else 'None'}<br>"
               f"C3: {', '.join(c3_members) if c3_members else 'None'}")
        thesis_text.append(text)

    # Create node traces
    professor_trace = go.Scatter3d(
        x=prof_x, y=prof_y, z=prof_z,
        mode='markers',
        marker=dict(
            size=10,
            color='lightblue',
            symbol='square',
            line=dict(color='darkblue', width=2),
            opacity=0.9
        ),
        text=prof_text,
        hoverinfo='text',
        name=f'Professors ({len(professor_nodes)})',
        showlegend=True
    )

    thesis_trace = go.Scatter3d(
        x=thesis_x, y=thesis_y, z=thesis_z,
        mode='markers',
        marker=dict(
            size=7,
            color='lightgreen',
            symbol='circle',
            line=dict(color='darkgreen', width=1),
            opacity=0.8
        ),
        text=thesis_text,
        hoverinfo='text',
        name=f'Theses ({len(thesis_nodes)})',
        showlegend=True
    )

    # Create figure
    print(" Creating interactive plot...")
    fig = go.Figure(data=edge_traces + [professor_trace, thesis_trace])

    fig.update_layout(
        title=dict(
            text='<b>Complete 3D Academic Network: All Relationships</b><br>'
                 f'<i>{len(professors)} Professors • {len(theses)} Theses • {G.number_of_edges()} Connections</i><br>'
                 f'<span style="font-size:12px">Mentor: Red • C2: Blue • C3: Green • Research: Orange • Collaboration: Purple</span><br>'
                 '<span style="font-size:11px">Drag to rotate • Scroll to zoom • Hover for details • Click legend to toggle</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='black',
            borderwidth=2,
            font=dict(size=11)
        ),
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='rgb(245, 248, 250)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        hovermode='closest',
        width=1600,
        height=1000,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Save to HTML
    output_path = 'visualizations/interactive_3d_full_network.html'
    fig.write_html(output_path)
    print(f"\n Saved complete 3D visualization to {output_path}")
    print(f"   Open in your browser to interact with the graph!")
    print(f"\n Tips:")
    print(f"   - Click legend items to show/hide specific edge types")
    print(f"   - Drag to rotate the 3D view")
    print(f"   - Scroll to zoom in/out")
    print(f"   - Hover over nodes for detailed information")

    return fig


def create_2d_interactive_graph_all_edges(max_theses=100, max_professors=50):
    """Create interactive 2D network visualization with ALL edge types"""

    print("\n Loading graph data...")
    graph_data, info = load_graph_data()

    print(" Creating NetworkX graph with ALL edge types...")
    G = nx.Graph()

    professors = info['mappings']['professors'][:max_professors]
    theses = info['mappings']['theses'][:max_theses]

    # Add nodes
    for i, prof in enumerate(professors):
        G.add_node(f"prof_{i}", label=prof, node_type='professor')

    for i, thesis in enumerate(theses):
        short_title = thesis[:40] + "..." if len(thesis) > 40 else thesis
        G.add_node(f"thesis_{i}", label=short_title, node_type='thesis')

    # Add all edge types
    edge_types_data = {
        'mentor': (graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy(), 'red', 2.5),
        'c2': (graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy(), 'blue', 2),
        'c3': (graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy(), 'green', 2),
        'research': (graph_data[('professor', 'researches', 'thesis')].edge_index.numpy(), 'orange', 1),
        'collaboration': (graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy(), 'purple', 2),
    }

    edges_by_type = {etype: [] for etype in edge_types_data.keys()}
    edge_counts = {etype: 0 for etype in edge_types_data.keys()}

    for etype, (edges_raw, color, width) in edge_types_data.items():
        for i in range(edges_raw.shape[1]):
            prof_idx = edges_raw[0, i]

            # Handle thesis edges
            if etype != 'collaboration':
                thesis_idx = edges_raw[1, i]
                if thesis_idx < len(theses) and prof_idx < len(professors):
                    G.add_edge(f"prof_{prof_idx}", f"thesis_{thesis_idx}",
                              edge_type=etype, color=color, width=width)
                    edges_by_type[etype].append((f"prof_{prof_idx}", f"thesis_{thesis_idx}"))
                    edge_counts[etype] += 1
            # Handle collaboration edges
            else:
                prof_idx2 = edges_raw[1, i]
                if prof_idx < len(professors) and prof_idx2 < len(professors):
                    G.add_edge(f"prof_{prof_idx}", f"prof_{prof_idx2}",
                              edge_type=etype, color=color, width=width)
                    edges_by_type[etype].append((f"prof_{prof_idx}", f"prof_{prof_idx2}"))
                    edge_counts[etype] += 1

    print(f"\n Graph created:")
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Edges: {G.number_of_edges()}")
    for etype, count in edge_counts.items():
        print(f"   - {etype.capitalize()} edges: {count}")

    print("\n Computing 2D layout...")
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Create traces for each edge type
    edge_traces = []
    for etype, (_, color, width) in edge_types_data.items():
        edge_x, edge_y = [], []
        for edge in edges_by_type[etype]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        if edge_x:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='none',
                name=f'{etype.capitalize()} ({edge_counts[etype]})',
                showlegend=True,
                opacity=0.6
            )
            edge_traces.append(edge_trace)

    # Create node traces
    professor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'professor']
    thesis_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'thesis']

    prof_x = [pos[node][0] for node in professor_nodes]
    prof_y = [pos[node][1] for node in professor_nodes]
    prof_text = [f"<b>{G.nodes[node]['label']}</b><br>"
                 f"Type: Professor<br>"
                 f"Degree: {G.degree(node)}"
                 for node in professor_nodes]

    professor_trace = go.Scatter(
        x=prof_x, y=prof_y,
        mode='markers',
        marker=dict(
            size=15,
            color='lightblue',
            symbol='square',
            line=dict(color='darkblue', width=2)
        ),
        text=prof_text,
        hoverinfo='text',
        name=f'Professors ({len(professor_nodes)})',
        showlegend=True
    )

    thesis_x = [pos[node][0] for node in thesis_nodes]
    thesis_y = [pos[node][1] for node in thesis_nodes]
    thesis_text = [f"<b>{G.nodes[node]['label']}</b><br>"
                   f"Type: Thesis<br>"
                   f"Committee Size: {G.degree(node)}"
                   for node in thesis_nodes]

    thesis_trace = go.Scatter(
        x=thesis_x, y=thesis_y,
        mode='markers',
        marker=dict(
            size=10,
            color='lightgreen',
            symbol='circle',
            line=dict(color='darkgreen', width=1)
        ),
        text=thesis_text,
        hoverinfo='text',
        name=f'Theses ({len(thesis_nodes)})',
        showlegend=True
    )

    # Create figure
    print(" Creating interactive plot...")
    fig = go.Figure(data=edge_traces + [professor_trace, thesis_trace])

    fig.update_layout(
        title=dict(
            text='<b>Complete 2D Academic Network: All Relationships</b><br>'
                 f'<i>{len(professors)} Professors • {len(theses)} Theses • {G.number_of_edges()} Connections</i><br>'
                 '<i>Click legend to toggle edge types • Hover for details</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='black',
            borderwidth=2
        ),
        hovermode='closest',
        width=1600,
        height=1000,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgb(245, 248, 250)',
        paper_bgcolor='white'
    )

    # Save to HTML
    output_path = 'visualizations/interactive_2d_full_network.html'
    fig.write_html(output_path)
    print(f"\n Saved complete 2D visualization to {output_path}")
    print(f"   Open in your browser to interact with the graph!")

    return fig


def create_all_interactive_visualizations():
    """Generate all interactive visualizations"""

    import os
    os.makedirs('visualizations', exist_ok=True)

    print("=" * 70)
    print("CREATING COMPLETE INTERACTIVE VISUALIZATIONS (ALL EDGE TYPES)")
    print("=" * 70)

    print("\n1. Interactive 3D graph with ALL edge types...")
    create_3d_interactive_graph_all_edges(max_theses=100, max_professors=50)

    print("\n2. Interactive 2D graph with ALL edge types...")
    create_2d_interactive_graph_all_edges(max_theses=100, max_professors=50)


    print(f"\nSaved to 'visualizations/' directory:")
    print("  1. interactive_3d_full_network.html - Complete 3D network (ALL edges)")
    print("  2. interactive_2d_full_network.html - Complete 2D network (ALL edges)")
    print("\n Features:")
    print("  • ALL edge types visible (mentor, C2, C3, research, collaboration)")
    print("  • Click legend to show/hide specific edge types")
    print("  • Drag to rotate (3D) or pan (2D)")
    print("  • Scroll to zoom")
    print("  • Hover for detailed information")
    print("\n Open these HTML files in your web browser!")


if __name__ == "__main__":
    create_all_interactive_visualizations()