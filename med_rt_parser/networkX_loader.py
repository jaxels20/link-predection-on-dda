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
    edge_attributes["association_type"] = edge.find("name").text.replace(" ", "_")
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

def get_networkx_graph(remove_self_loops=False, remove_isolated_nodes=False, bipartite=False):

    # get the path of the xml file
    path = Path(__file__).resolve().parents[1] / "med-rt" / "Core_MEDRT_2023.02.06_XML.xml"

    # parse the xml file
    tree = ET.parse(str(path))
    root = tree.getroot()

    # create a directed graph
    G = nx.DiGraph()

    if bipartite:
        G = add_drugs_and_disease(G, root)
        return G
    
    else:
        G = add_concepts(G, root)
        G = add_drugs_and_disease(G, root)
        G = add_all_edge(G, root)

    if remove_self_loops:
        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))
    
    if remove_isolated_nodes:
        # remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    return G

def add_drugs_and_disease(G, root):
    for edge in root.findall('association'):
        try:
            from_node_attributes = {}
            to_node_attributes = {}

            from_node = edge.find("from_name").text + "_" + edge.find("from_code").text
            to_node = edge.find("to_name").text + "_" + edge.find("to_code").text

            edge_attributes = set_edge_attrib(edge)

            if edge_attributes["association_type"] == "may_treat":
                from_node_attributes["type"] = "drug"
                to_node_attributes["type"] = "disease"

                # add the nodes with their attribute
                
                conditionally_add_node(G, from_node, from_node_attributes)
                conditionally_add_node(G, to_node, to_node_attributes)

                G.add_edge(from_node, to_node, **edge_attributes)
        except:
            print("Error")    
     
    return G


def add_concepts(G, root):
    # add concepts as nodes to the graph from the xml file
        for element in root.findall('concept'):
            attributes = {}
            node_name = element.find("name").text + "_" + element.find("code").text
            
            for prop in element.findall('property'):
                if prop.find('name').text == "CTY":
                    attributes["type"] = prop.find('value').text
                    
            # add the nodes with their attributes
            G.add_node(node_name, **attributes)

        return G


def add_all_edge(G, root):
    # add edges to the graph from the xml file this also adds the nodes
    for edge in root.findall('association'):
        try: 
            from_node = edge.find("from_name").text + "_" + edge.find("from_code").text
            to_node = edge.find("to_name").text + "_" + edge.find("to_code").text

            edge_attributes = set_edge_attrib(edge)

            if from_node in G.nodes and to_node in G.nodes:
                G.add_edge(from_node, to_node, **edge_attributes)

        except:
            print("Error: ", edge.find("from_name").text, edge.find("to_name").text)
    
    return G





if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True, bipartite=True)









