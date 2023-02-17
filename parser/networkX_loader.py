import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# bfs traversal for drawing subgraph with a certain depth

def bfs(graph, start, depth=1):
    #make the graph undirected
    graph = graph.to_undirected()

    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            if depth > 0:
                queue.extend(set(graph[vertex]) - visited)
            depth -= 1
    return visited


def set_edge_attrib(edge):
    edge_attributes = {}
    # create dictionary of attributes
    edge_attributes["association_type"] = edge.find("name").text
    edge_attributes["namespace"] = edge.find("namespace").text
    edge_attributes["from_namespace"] = edge.find("from_namespace").text
    edge_attributes["to_namespace"] = edge.find("to_namespace").text
    edge_attributes["from_name"] = edge.find("from_name").text
    edge_attributes["to_name"] = edge.find("to_name").text
    edge_attributes["from_code"] = edge.find("from_code").text
    edge_attributes["to_code"] = edge.find("to_code").text

        return edge_attributes

def conditionally_add_node(G, node, node_attributes):
    if G.has_node(node) == False:
        G.add_node(node, **node_attributes)
    # draw relations with name = ...
    subgraph = nx.DiGraph([(u, v, d) for u, v, d in subgraph.edges(data=True) if d['association_type'] == "Parent Of"])
    pos = nx.circular_layout(subgraph)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph, node_color='r', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=nx.get_edge_attributes(subgraph, "association_type"), font_size=8, font_family='sans-serif')
    plt.show()

def get_networkx_graph(remove_self_loops=False, remove_isolated_nodes=False, bipartite=False):

    # get the path of the xml file
    path = Path(__file__).resolve().parents[1] / "med-rt" / "Core_MEDRT_2023.02.06_XML.xml"

    # parse the xml file
    tree = ET.parse(str(path))
    root = tree.getroot()

    # create a directed graph
    G = nx.DiGraph()

    if not bipartite:
        # add concepts as nodes to the graph from the xml file
        for element in root.findall('concept'):
            attributes = {}
            node_name = element.find("name").text
            
            for prop in element.findall('property'):
                if prop.find('name').text == "CTY":
                    attributes["type"] = prop.find('value').text

            # add the nodes with their attributes
            G.add_node(node_name, **attributes)


    # add edges to the graph from the xml file this also adds the nodes
    for edge in root.findall('association'):
        try: 
            from_node_attributes = {}
            to_node_attributes = {}

            from_node = edge.find("from_name").text
            to_node = edge.find("to_name").text

            edge_attributes = set_edge_attrib(edge)

            if edge_attributes["association_type"] == "may_treat" or edge_attributes["association_type"] == "may_prevent":
                from_node_attributes["type"] = "drug"
                to_node_attributes["type"] = "disease"
            

                # add the nodes with their attribute
                
                conditionally_add_node(G, from_node, from_node_attributes)
                conditionally_add_node(G, to_node, to_node_attributes)

                G.add_edge(from_node, to_node, **edge_attributes)
            
            if not bipartite:

                if edge_attributes["association_type"] == "has_SC":
                    from_node_attributes["type"] = "drug"
                    to_node_attributes["type"] = "SC"

                    # add the nodes with their attributes
                    conditionally_add_node(G, from_node, from_node_attributes)
                    conditionally_add_node(G, to_node, to_node_attributes)

                    G.add_edge(from_node, to_node, **edge_attributes)

                elif edge_attributes["association_type"] == "has_MoA":
                    from_node_attributes["type"] = "drug"
                    to_node_attributes["type"] = "MoA"

                    # add the nodes with their attributes
                    conditionally_add_node(G, from_node, from_node_attributes)
                    conditionally_add_node(G, to_node, to_node_attributes)

                    G.add_edge(from_node, to_node, **edge_attributes)

                elif edge_attributes["association_type"] == "has_PE":
                    from_node_attributes["type"] = "drug"
                    to_node_attributes["type"] = "PE"

                    # add the nodes with their attributes
                    conditionally_add_node(G, from_node, from_node_attributes)
                    conditionally_add_node(G, to_node, to_node_attributes)

                    G.add_edge(from_node, to_node, **edge_attributes)

                elif edge_attributes["association_type"] == "has_TC":
                    from_node_attributes["type"] = "drug"
                    to_node_attributes["type"] = "TC"

                    # add the nodes with their attributes
                    conditionally_add_node(G, from_node, from_node_attributes)
                    conditionally_add_node(G, to_node, to_node_attributes)

                    G.add_edge(from_node, to_node, **edge_attributes)

                elif edge_attributes["association_type"] == "Parent Of":
                    G.add_edge(from_node, to_node, **edge_attributes)
                    

        except:
            print("Error: ", edge.find("from_name").text, edge.find("to_name").text)


    if remove_self_loops:
        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))
    
    if remove_isolated_nodes:
        # remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    return G

if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True, bipartite=False)

    # get the subgraph with a depth of 2
    subgraph = bfs(G, "Antirheumatic Agent [EPC]", depth=1)
    pos = nx.spring_layout(G.subgraph(subgraph))

    print(G.nodes["Antirheumatic Agent [EPC]"]["type"])
    # draw the subgraph with edge labels and node attributes
    nx.draw(G.subgraph(subgraph), with_labels=True)
    nx.draw_networkx_edge_labels(G.subgraph(subgraph), pos=pos, edge_labels=nx.get_edge_attributes(G.subgraph(subgraph), 'association_type'))
    nx.draw_networkx_labels(G.subgraph(subgraph), pos=pos, labels=nx.get_node_attributes(G.subgraph(subgraph), 'type'))
    plt.show()








