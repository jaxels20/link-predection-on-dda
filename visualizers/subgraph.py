import sys
sys.path.append('..')
from med_rt_parser.networkX_loader import get_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder for the hierarchy graphs
OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "graph_plots"

def draw_subgraph(G, root_name, depth):
    # Special node positions for COVID-19. Function is designed for this specific node.
    # Draw subgraph of graph G
    # G: networkx graph
    root_name = 'COVID-19'
    # Find root node
    for node in G.nodes:
        # Get attributes of node
        try: 
            node_name = G.nodes[node]['name']
            if node_name == root_name.split('_')[0]:
                print("Root node found")
                root_node = node
                break
        except:
            continue

    # Get nodes in BFS order starting from root_node
    edges = nx.bfs_edges(G, root_node, depth_limit=depth)
    nodes_no_root = [v for u, v in edges]
    nodes = [root_node] + nodes_no_root

    # Get subgraph
    subgraph = nx.subgraph(G, nodes)
    
    pos = nx.spring_layout(subgraph)
    
    node_labels = nx.get_node_attributes(subgraph, 'name')   # Can  be used in draw, with labels=node_labels
    edge_labels = nx.get_edge_attributes(subgraph,'association_type')
    # Draw subgraph
    nx.draw(subgraph, with_labels=True, pos=pos)
    nx.draw_networkx_edge_labels(subgraph, edge_labels=edge_labels, font_color='black', pos=pos)
    plt.show()

if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True).to_undirected()
    
    draw_subgraph(G, root_name='COVID-19', depth=2)