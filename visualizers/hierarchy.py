import sys
sys.path.append('..')
from med_rt_parser.networkX_loader import get_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder for the hierarchy graphs
OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "graph_plots"

def build_hierachy(graph, concept_type):
    hierarchy_graph = nx.DiGraph()
    for node in graph.nodes(data=True):
        try:
            # Check if type of node is a concept
            if node[1]['type'] == concept_type:
                # Get all in edges and out edges for the node in the directed graph
                out_associations = graph.out_edges(node[0], data=True)
                parent_of_associations = [edge for edge in out_associations if edge[2]['association_type'] == 'Parent Of']
                if len(parent_of_associations) > 0:
                    for edge in parent_of_associations:
                        hierarchy_graph.add_edge(edge[0], edge[1], **edge[2])
                else:
                    continue
        except:
            pass
    return hierarchy_graph

def draw_hierarchy(hierarchy_graph, save=False, filename="hierarchy.pdf", fig_size=(150, 100), node_size=200, arrow_size=40):
    #reversed_graph = hierarchy_graph.reverse(copy=True)
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(hierarchy_graph)):
        x_offset = len(node_list) / 2
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i)
    if save:
        f = plt.figure(figsize=fig_size)
        nx.draw_networkx_nodes(hierarchy_graph, pos_dict, nodelist=hierarchy_graph, node_color='r', node_size=node_size, alpha=1)
        nx.draw_networkx_edges(hierarchy_graph, pos_dict, width=0.5, alpha=0.5, arrowsize=arrow_size)
        # Annotate the number  of nodes and edges in the graph
        plt.annotate("Nodes: " + str(hierarchy_graph.number_of_nodes()), xy=(0.1, 0.2), xytext=(0.1, 0.2), textcoords='axes fraction', fontsize=100)
        plt.annotate("Edges: " + str(hierarchy_graph.number_of_edges()), xy=(0.1, 0.175), xytext=(0.1, 0.175), textcoords='axes fraction', fontsize=100)
        # Add title for the plot
        plt.title("Hierarchy of " + filename.split("_")[0], fontsize=150)
        f.savefig(OUTPUT_FOLDER / filename)
    else:
        nx.draw_networkx_nodes(hierarchy_graph, pos_dict, nodelist=hierarchy_graph, node_color='r', node_size=5, alpha=1)
        nx.draw_networkx_edges(hierarchy_graph, pos_dict, width=0.5, alpha=0.5)
        plt.show()

def get_hierachy_plots(G):
    MoA_hierarchy = build_hierachy(G, 'MoA')
    draw_hierarchy(MoA_hierarchy, save=True, filename="MoA_hierarchy.pdf")
    PE_hierarchy = build_hierachy(G, 'PE')
    draw_hierarchy(PE_hierarchy, save=True, filename="PE_hierarchy.pdf", fig_size=(200, 100), node_size=50)
    HC_hierarchy = build_hierachy(G, 'HC')
    draw_hierarchy(HC_hierarchy, save=True, filename="HC_hierarchy.pdf", fig_size=(200, 100), node_size=50)

if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True)
    get_hierachy_plots(G)
