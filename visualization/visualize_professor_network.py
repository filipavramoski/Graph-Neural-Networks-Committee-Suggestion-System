"""
Interactive Professor-Specific Network Visualization
Input a professor's name to see their complete network:
- Mentored theses
- C2 committee memberships
- C3 committee memberships
- Research collaborations
- Professor collaborations
"""

import torch
import json
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys


def load_graph_data():
    """Load the graph and metadata"""
    graph_data = torch.load('../structure/hetero_graph_edge_labeled.pt', weights_only=False)
    with open('../structure/graph_metadata_edge_labeled.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    return graph_data, info


def find_professor_index(prof_name, professors):
    """Find professor index by name (case-insensitive, partial match)"""
    prof_name_lower = prof_name.lower()

    # Exact match first
    for i, prof in enumerate(professors):
        if prof.lower() == prof_name_lower:
            return i, prof

    # Partial match
    matches = []
    for i, prof in enumerate(professors):
        if prof_name_lower in prof.lower():
            matches.append((i, prof))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"\n️ Multiple professors found matching '{prof_name}':")
        for i, (idx, name) in enumerate(matches, 1):
            print(f"   {i}. {name}")
        choice = input("\nEnter number to select: ")
        try:
            selected = matches[int(choice) - 1]
            return selected
        except (ValueError, IndexError):
            print("Invalid selection")
            return None, None

    return None, None


def build_professor_network(prof_idx, graph_data, info):
    """Build a network centered on a specific professor"""

    professors = info['mappings']['professors']
    theses = info['mappings']['theses']

    G = nx.Graph()

    # Add the central professor
    central_prof = f"prof_{prof_idx}"
    G.add_node(central_prof,
               label=professors[prof_idx],
               node_type='professor',
               is_central=True)

    # Get all edge data
    edge_data = {
        'mentor': graph_data[('professor', 'mentors', 'thesis')].edge_index.numpy(),
        'c2': graph_data[('professor', 'serves_as_c2', 'thesis')].edge_index.numpy(),
        'c3': graph_data[('professor', 'serves_as_c3', 'thesis')].edge_index.numpy(),
        'research': graph_data[('professor', 'researches', 'thesis')].edge_index.numpy(),
        'collaboration': graph_data[('professor', 'collaborates', 'professor')].edge_index.numpy()
    }

    # Track statistics
    stats = {
        'mentor': [],
        'c2': [],
        'c3': [],
        'research': [],
        'collaborators': []
    }

    # Add mentor edges
    for i in range(edge_data['mentor'].shape[1]):
        p_idx = edge_data['mentor'][0, i]
        t_idx = edge_data['mentor'][1, i]
        if p_idx == prof_idx:
            thesis_node = f"thesis_{t_idx}"
            thesis_title = theses[t_idx][:60] + "..." if len(theses[t_idx]) > 60 else theses[t_idx]
            G.add_node(thesis_node,
                       label=thesis_title,
                       full_title=theses[t_idx],
                       node_type='thesis')
            G.add_edge(central_prof, thesis_node,
                       edge_type='mentor', color='red', width=3)
            stats['mentor'].append(thesis_title)

    # Add C2 edges
    for i in range(edge_data['c2'].shape[1]):
        p_idx = edge_data['c2'][0, i]
        t_idx = edge_data['c2'][1, i]
        if p_idx == prof_idx:
            thesis_node = f"thesis_{t_idx}"
            if thesis_node not in G:
                thesis_title = theses[t_idx][:60] + "..." if len(theses[t_idx]) > 60 else theses[t_idx]
                G.add_node(thesis_node,
                           label=thesis_title,
                           full_title=theses[t_idx],
                           node_type='thesis')
            G.add_edge(central_prof, thesis_node,
                       edge_type='c2', color='blue', width=2)
            stats['c2'].append(theses[t_idx][:60])

    # Add C3 edges
    for i in range(edge_data['c3'].shape[1]):
        p_idx = edge_data['c3'][0, i]
        t_idx = edge_data['c3'][1, i]
        if p_idx == prof_idx:
            thesis_node = f"thesis_{t_idx}"
            if thesis_node not in G:
                thesis_title = theses[t_idx][:60] + "..." if len(theses[t_idx]) > 60 else theses[t_idx]
                G.add_node(thesis_node,
                           label=thesis_title,
                           full_title=theses[t_idx],
                           node_type='thesis')
            G.add_edge(central_prof, thesis_node,
                       edge_type='c3', color='green', width=2)
            stats['c3'].append(theses[t_idx][:60])

    # Add research edges
    for i in range(edge_data['research'].shape[1]):
        p_idx = edge_data['research'][0, i]
        t_idx = edge_data['research'][1, i]
        if p_idx == prof_idx:
            thesis_node = f"thesis_{t_idx}"
            if thesis_node not in G:
                thesis_title = theses[t_idx][:60] + "..." if len(theses[t_idx]) > 60 else theses[t_idx]
                G.add_node(thesis_node,
                           label=thesis_title,
                           full_title=theses[t_idx],
                           node_type='thesis')
            G.add_edge(central_prof, thesis_node,
                       edge_type='research', color='orange', width=1)
            stats['research'].append(theses[t_idx][:60])

    # Add collaboration edges
    for i in range(edge_data['collaboration'].shape[1]):
        p_idx1 = edge_data['collaboration'][0, i]
        p_idx2 = edge_data['collaboration'][1, i]
        if p_idx1 == prof_idx:
            collab_node = f"prof_{p_idx2}"
            G.add_node(collab_node,
                       label=professors[p_idx2],
                       node_type='professor',
                       is_central=False)
            G.add_edge(central_prof, collab_node,
                       edge_type='collaboration', color='purple', width=2)
            stats['collaborators'].append(professors[p_idx2])
        elif p_idx2 == prof_idx:
            collab_node = f"prof_{p_idx1}"
            G.add_node(collab_node,
                       label=professors[p_idx1],
                       node_type='professor',
                       is_central=False)
            G.add_edge(central_prof, collab_node,
                       edge_type='collaboration', color='purple', width=2)
            stats['collaborators'].append(professors[p_idx1])

    return G, stats


def create_3d_professor_visualization(prof_name):
    """Create interactive 3D visualization for a specific professor"""

    print("\n Loading graph data...")
    graph_data, info = load_graph_data()

    professors = info['mappings']['professors']

    # Find professor
    print(f" Searching for professor: {prof_name}")
    prof_idx, prof_full_name = find_professor_index(prof_name, professors)

    if prof_idx is None:
        print(f"\n Professor '{prof_name}' not found!")
        print(f"\n Available professors ({len(professors)} total):")
        for i, prof in enumerate(professors[:20], 1):
            print(f"   {i}. {prof}")
        if len(professors) > 20:
            print(f"   ... and {len(professors) - 20} more")
        return None

    print(f" Found: {prof_full_name}")

    # Build network
    print(f"\n Building network for {prof_full_name}...")
    G, stats = build_professor_network(prof_idx, graph_data, info)

    if G.number_of_nodes() == 1:
        print(f"\n No connections found for {prof_full_name}")
        return None

    # Print statistics
    print(f"\n Network Statistics for {prof_full_name}:")
    print(f"   - Total nodes: {G.number_of_nodes()}")
    print(f"   - Total edges: {G.number_of_edges()}")
    print(f"   - Mentored theses: {len(stats['mentor'])}")
    print(f"   - C2 memberships: {len(stats['c2'])}")
    print(f"   - C3 memberships: {len(stats['c3'])}")
    print(f"   - Research papers: {len(stats['research'])}")
    print(f"   - Collaborators: {len(stats['collaborators'])}")

    # 3D layout
    print("\n Computing 3D layout...")
    pos = nx.spring_layout(G, dim=3, k=1.0, iterations=100, seed=42)

    # Force central professor to center
    central_prof = f"prof_{prof_idx}"
    pos[central_prof] = np.array([0, 0, 0])

    # Separate nodes by type
    central_node = [central_prof]
    thesis_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'thesis']
    collab_nodes = [n for n, d in G.nodes(data=True)
                    if d['node_type'] == 'professor' and not d.get('is_central', False)]

    # Create edge traces by type
    edge_traces = []
    edge_types = {
        'mentor': ('red', 'Mentorship'),
        'c2': ('blue', 'C2 Member'),
        'c3': ('green', 'C3 Member'),
        'research': ('orange', 'Research'),
        'collaboration': ('purple', 'Collaboration')
    }

    for edge_type, (color, label) in edge_types.items():
        edge_x, edge_y, edge_z = [], [], []
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') == edge_type:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

        if edge_x:
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=color, width=4),
                hoverinfo='none',
                name=label,
                showlegend=True,
                opacity=0.7
            )
            edge_traces.append(edge_trace)

    # Central professor trace
    central_x = [pos[central_prof][0]]
    central_y = [pos[central_prof][1]]
    central_z = [pos[central_prof][2]]

    central_trace = go.Scatter3d(
        x=central_x, y=central_y, z=central_z,
        mode='markers+text',
        marker=dict(
            size=25,
            color='gold',
            symbol='diamond',
            line=dict(color='darkorange', width=3)
        ),
        text=[prof_full_name],
        textposition='top center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertext=[f"<b>{prof_full_name}</b><br>"
                   f"<b>CENTRAL PROFESSOR</b><br>"
                   f"Total Connections: {G.degree(central_prof)}<br>"
                   f"Mentored: {len(stats['mentor'])}<br>"
                   f"C2: {len(stats['c2'])}<br>"
                   f"C3: {len(stats['c3'])}<br>"
                   f"Research: {len(stats['research'])}<br>"
                   f"Collaborators: {len(stats['collaborators'])}"],
        hoverinfo='text',
        name='Central Professor',
        showlegend=True
    )

    # Thesis nodes trace
    thesis_x = [pos[node][0] for node in thesis_nodes]
    thesis_y = [pos[node][1] for node in thesis_nodes]
    thesis_z = [pos[node][2] for node in thesis_nodes]
    thesis_text = []

    for node in thesis_nodes:
        label = G.nodes[node]['label']
        full_title = G.nodes[node].get('full_title', label)
        # Get edge type to this thesis
        edge_types_to_thesis = [G[central_prof][node]['edge_type']]
        edge_type_str = ', '.join([et.upper() for et in edge_types_to_thesis])

        thesis_text.append(
            f"<b>{label}</b><br>"
            f"Full Title: {full_title}<br>"
            f"Role: {edge_type_str}"
        )

    thesis_trace = go.Scatter3d(
        x=thesis_x, y=thesis_y, z=thesis_z,
        mode='markers',
        marker=dict(
            size=8,
            color='lightgreen',
            symbol='circle',
            line=dict(color='darkgreen', width=1)
        ),
        text=thesis_text,
        hoverinfo='text',
        name='Theses',
        showlegend=True
    )

    # Collaborator nodes trace
    if collab_nodes:
        collab_x = [pos[node][0] for node in collab_nodes]
        collab_y = [pos[node][1] for node in collab_nodes]
        collab_z = [pos[node][2] for node in collab_nodes]
        collab_text = [f"<b>{G.nodes[node]['label']}</b><br>"
                       f"Type: Collaborator<br>"
                       f"Joint Projects: {G.degree(node)}"
                       for node in collab_nodes]

        collab_trace = go.Scatter3d(
            x=collab_x, y=collab_y, z=collab_z,
            mode='markers',
            marker=dict(
                size=12,
                color='lightblue',
                symbol='square',
                line=dict(color='darkblue', width=2)
            ),
            text=collab_text,
            hoverinfo='text',
            name='Collaborators',
            showlegend=True
        )
    else:
        collab_trace = None

    # Create figure
    print(" Creating interactive 3D visualization...")
    traces = edge_traces + [central_trace, thesis_trace]
    if collab_trace:
        traces.append(collab_trace)

    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Professor Network: {prof_full_name}</b><br>'
                 f'<i>Mentored: {len(stats["mentor"])} | '
                 f'C2: {len(stats["c2"])} | '
                 f'C3: {len(stats["c3"])} | '
                 f'Research: {len(stats["research"])} | '
                 f'Collaborators: {len(stats["collaborators"])}</i><br>'
                 f'<span style="font-size:12px">Drag to rotate • Scroll to zoom • Hover for details</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=2,
            font=dict(size=12)
        ),
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='rgb(240, 245, 250)',
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
    safe_filename = prof_full_name.replace(' ', '_').replace('/', '_')
    output_path = f'visualizations/professor_{safe_filename}_network.html'
    fig.write_html(output_path)

    print(f"\n Visualization created successfully!")
    print(f" Saved to: {output_path}")
    print(f" Open in your browser to interact with the graph!")




    if stats['mentor']:
        print(f"\n Mentored Theses ({len(stats['mentor'])}):")
        for i, thesis in enumerate(stats['mentor'][:5], 1):
            print(f"   {i}. {thesis}")
        if len(stats['mentor']) > 5:
            print(f"   ... and {len(stats['mentor']) - 5} more")

    if stats['c2']:
        print(f"\n C2 Committee Memberships ({len(stats['c2'])}):")
        for i, thesis in enumerate(stats['c2'][:5], 1):
            print(f"   {i}. {thesis}")
        if len(stats['c2']) > 5:
            print(f"   ... and {len(stats['c2']) - 5} more")

    if stats['c3']:
        print(f"\n C3 Committee Memberships ({len(stats['c3'])}):")
        for i, thesis in enumerate(stats['c3'][:5], 1):
            print(f"   {i}. {thesis}")
        if len(stats['c3']) > 5:
            print(f"   ... and {len(stats['c3']) - 5} more")

    if stats['research']:
        print(f"\n Research Papers ({len(stats['research'])}):")
        for i, thesis in enumerate(stats['research'][:5], 1):
            print(f"   {i}. {thesis}")
        if len(stats['research']) > 5:
            print(f"   ... and {len(stats['research']) - 5} more")

    if stats['collaborators']:
        print(f"\n Collaborators ({len(stats['collaborators'])}):")
        for i, collab in enumerate(stats['collaborators'][:10], 1):
            print(f"   {i}. {collab}")
        if len(stats['collaborators']) > 10:
            print(f"   ... and {len(stats['collaborators']) - 10} more")

    return fig


def interactive_mode():
    """Interactive mode - prompt user for professor names"""

    import os
    os.makedirs('visualizations', exist_ok=True)

    print("=" * 70)
    print("INTERACTIVE PROFESSOR NETWORK VISUALIZATION")
    print("=" * 70)

    while True:

        prof_name = input("\n Enter professor name (or 'quit' to exit): ").strip()

        if prof_name.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break

        if not prof_name:
            print("️ Please enter a professor name")
            continue

        try:
            fig = create_3d_professor_visualization(prof_name)

            if fig is not None:
                another = input("\n Visualize another professor? (y/n): ").strip().lower()
                if another != 'y':
                    break
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(professor_names):
    """Batch mode - create visualizations for multiple professors"""

    import os
    os.makedirs('visualizations', exist_ok=True)



    for prof_name in professor_names:
        print(f"\n{'=' * 70}")
        print(f"Processing: {prof_name}")
        print(f"{'=' * 70}")

        try:
            create_3d_professor_visualization(prof_name)
        except Exception as e:
            print(f" Error processing {prof_name}: {e}")

    print(f"\n Batch processing complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Batch mode: python visualize_professor_network.py "Prof Name1" "Prof Name2"
        professor_names = sys.argv[1:]
        batch_mode(professor_names)
    else:
        # Interactive mode
        interactive_mode()