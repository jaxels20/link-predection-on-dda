import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


# get the path of the xml file
path = Path(__file__).resolve().parents[1] / "med-rt" / "Core_MEDRT_2023.02.06_XML.xml"

edge_labels = {}

# parse the xml file
tree = ET.parse(str(path))
root = tree.getroot()

# create a graph
G = nx.Graph()


# add nodes to the graph
for elem in root.findall('term'):
    node_name = elem.find('name').text

    # create dictionary of attributes
    attributes = dict(zip([subelem.tag for subelem in elem.iter()], [elem.text for elem in elem.iter()]))
    
    # remove the name attribute and the term tag

    attributes.pop('name')
    attributes.pop('term')

    attributes['type'] = "drug"
    
    G.add_node(node_name, **attributes)

for elem in root.findall('concept'):
    node_name = elem.find('name').text

    attributes = dict(zip([subelem.tag for subelem in elem.iter()], [elem.text for elem in elem.iter()]))

    attributes.pop('name')
    attributes.pop('concept')

    attributes['type'] = "concept"

    G.add_node(node_name, **attributes)


# add edges to the graph
for edge in root.findall('association'):
    try: 
        attributes = {}
        node1 = edge.find("from_name").text
        node2 = edge.find("to_name").text

        # create dictionary of attributes
        attributes["association_type"] = edge.find("name").text
        attributes["namespace"] = edge.find("namespace").text
        attributes["from_namespace"] = edge.find("from_namespace").text
        attributes["to_namespace"] = edge.find("to_namespace").text
        attributes["from_name"] = edge.find("from_name").text
        attributes["to_name"] = edge.find("to_name").text
        attributes["from_code"] = edge.find("from_code").text
        attributes["to_code"] = edge.find("to_code").text
        
        G.add_edge(node1, node2, **attributes)
    except:
        pass
        print("Error: ", edge.find("from_name").text, edge.find("to_name").text)




# bfs traversal for drawing subgraph with a certain depth

def bfs(graph, start, depth=1):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            if depth > 0:
                queue.extend(set(graph[vertex]) - visited)
            depth -= 1
    return visited
    
# get the subgraph

visited_nodes = bfs(G, "Cellular or Molecular Interactions [MoA]", depth=2)

# draw the subgraph with node lables and edge lables

subgraph = G.subgraph(visited_nodes)


pos = nx.spring_layout(subgraph)
nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph, node_color='r', node_size=500, alpha=0.8)
nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=nx.get_edge_attributes(subgraph, "association_type"), font_size=8, font_family='sans-serif')



plt.show()









