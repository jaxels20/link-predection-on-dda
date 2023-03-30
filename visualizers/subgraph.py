import sys
sys.path.append('..')
from med_rt_parser.networkX_loader import get_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder for the hierarchy graphs
OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "graph_plots"

def draw_subgraph(G, root_name, depth):
    # Draw subgraph of graph G
    # G: networkx graph
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
    edges = nx.bfs_edges(G.to_undirected(), root_node, depth_limit=depth)
    nodes_no_root = [v for u, v in edges]
    additional_nodes = G.to_undirected().neighbors(nodes_no_root[1])
    nodes = set([root_node] + nodes_no_root  + list(additional_nodes))

    # Get subgraph
    subgraph = nx.subgraph(G, nodes)
    
    # Set plot details
    pos = {'SARS-CoV-2 (COVID-19) vaccine, mRNA-BNT162b2 OMICRON (BA.4/BA.5)_2610243': [-0.71856347,  0.43810619],
           'Antibody Interactions [MoA]_N0000010226': [0.81156406, 0.13873962],
           'bebtelovimab_2592360': [-0.09906104, -0.52099354],
           'imdevimab_2465249': [ 0.48211375, -0.06237493],
           'COVID-19_M000732426': [-0.28186297,  0.03705112],
           'Cellular Activity Alteration [PE]_N0000008337': [0.63224975, 0.41243655],
           'SARS-COV-2 (COVID-19) vaccine, subunit, recombinant spike protein_2606074': [-0.36081018,  0.60141073],
           'bamlanivimab_2463114': [-0.27735684, -0.35276651],
           'casirivimab_2465242': [0.30540625, 0.16511813],
           'SARS-CoV-2 (COVID-19) vaccine, protein NVX-CoV2373_2606073': [-0.87794855,  0.09072453],
           'Antiviral Agent [TC]_N0000178304': [ 0.16948115, -0.28605463],
           'SARS-CoV-2 (COVID-19) vaccine, mRNA-1273 OMICRON (BA.4/BA.5)_2610326': [-0.78521191, -0.29774968],
           'I [Preparations]_N0000010591': [ 0.81156406, -0.36364758]}
    node_labels = nx.get_node_attributes(subgraph, 'name')   # Can  be used in draw, with labels=node_labels
    print(node_labels)
    for name in node_labels:
        if len(name)>30:
            modified_name = node_labels[name].split(' ')
            modified_name[len(modified_name)//2] = modified_name[len(modified_name)//2] + '\n'
            node_labels[name] = ' '.join(modified_name)
    edge_labels = nx.get_edge_attributes(subgraph,'association_type')
    color_values = []
    for node in subgraph.nodes: 
        if G.nodes[node]['type'] == 'drug':
            color_values.append('blue')
        elif G.nodes[node]['type'] == 'disease':
            color_values.append('red')
        elif G.nodes[node]['type'] in ['MoA', 'PE', 'TC']:
            color_values.append('purple')
        else:
            color_values.append('darkgreen')

    # Draw subgraph
    nx.draw_networkx(subgraph, with_labels=True, pos=pos, labels=node_labels, node_color=color_values, font_size=7)
    plt.box(False)
    nx.draw_networkx_edge_labels(subgraph, edge_labels=edge_labels, label_pos=0.5, font_color='black', pos=pos, font_size=7)
    # Legend
    plt.scatter([], [], c='blue', label='Drug')
    plt.scatter([], [], c='red', label='Disease')
    plt.scatter([], [], c='purple', label='Concept')
    plt.scatter([], [], c='darkgreen', label='Other')
    plt.legend(loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True)
    
    draw_subgraph(G, root_name='imdevimab', depth=1)