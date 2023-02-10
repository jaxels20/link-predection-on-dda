import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx

# get the path of the xml file
path = Path(__file__).resolve().parents[1] / "med-rt" / "Core_MEDRT_2023.02.06_XML.xml"


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
    

    G.add_node(node_name, **attributes)


# add edges to the graph
for edge in root.findall('association'):
    try: 
        node1 = edge.find("from_name").text
        node2 = edge.find("to_name").text

        # create dictionary of attributes
        attributes = dict(zip([subelem.tag for subelem in edge.iter()], [elem.text for elem in edge.iter()]))

        attributes.pop('association')
        

        G.add_edge(node1, node2)
    except:
        pass
        print("Error: ", edge.find("from_name").text, edge.find("to_name").text)






