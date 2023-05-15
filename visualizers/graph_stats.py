import sys
sys.path.append('..')
from pathlib import Path
from med_rt_parser.networkX_loader import get_networkx_graph
import networkx as nx
import plotly.express as px
from math import log
import numpy as np
from sklearn.linear_model import LinearRegression

OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "graph_plots"

def print_graph_stats(G):
    """Prints the number of nodes, edges, connected components, self loops, and isolated nodes in the graph"""

    print("Number of nodes: ", G.number_of_nodes())
    print("Number of drug nodes: ", len([n for n in G.nodes if G.nodes[n]["type"] == "drug"]))
    print("Number of disease nodes: ", len([n for n in G.nodes if G.nodes[n]["type"] == "disease"]))
    print("Number of edges: ", G.number_of_edges())
    print("Number of connected components: ", nx.number_connected_components(G))
    print("Number of isolated nodes: ", len(list(nx.isolates(G))))
    print("Density: ", nx.density(G))
    print("Maximum degree: ", max([d for n, d in G.degree()]))
    print("Minimum degree: ", min([d for n, d in G.degree()]))
    print("Average degree: ", sum([d for n, d in G.degree()]) / G.number_of_nodes())
    
def get_degree_dist(G):
    """Returns a dictionary of the degree distribution of the graph"""
    degree_dist = {}
    for n, d in G.degree():
        if d in degree_dist:
            degree_dist[d] += 1
        else:
            degree_dist[d] = 1
    return degree_dist

def plot_degree_box_violin(G):
    """Plots a boxplot and violin plot on top of each other of the degree distribution of graph G"""
    degrees = [d for n, d in G.degree()]
    fig = px.box(x=degrees, points="outliers",
                 labels={"x": "Degree"},
                 template="simple_white",
                 width=780, height=340)
    fig.update_layout(font_family = "Moderne Computer")
    fig.write_image(OUTPUT_FOLDER  / "degree_box.pdf")

def plot_scale_free_network(degree_dict):
    """Plots a scale free network of the degree distribution of graph G"""
    num_nodes = sum(degree_dict.values())
    normalize_dict = {k: v / num_nodes for k, v in degree_dict.items()}
    x = list(normalize_dict.keys())
    y = list(normalize_dict.values())

    # Regression model
    x_log1 = np.array([log(i) for i in x])
    x_log = np.array([log(i) for i in x]).reshape(-1, 1)
    y_log = np.array([log(i) for i in y])
    model = LinearRegression()
    model.fit(np.array(x_log), np.array(y_log))
    intercept = model.intercept_
    slope = model.coef_[0]
    r_sq = model.score(x_log, y_log)
    print("Intercept: ", intercept)
    print("Slope: ", slope)
    print("R^2: ", r_sq)

    test = np.array([x**(-1.7319418742438881) for x in np.sort(x_log1)])

    y_pred = model.predict(x_log)

    fig = px.scatter(x=x, y=y,
                     labels={"x": "Degree", "y": "Proportion of nodes"},
                     template="simple_white",
                     width=780, height=340)
    fig.update_layout(font_family = "Moderne Computer")
    #fig.add_trace(px.line(x=x, y=np.exp(y_pred)).data[0])
    fig.write_image(OUTPUT_FOLDER  / "degree_scatter.pdf")


if __name__ == "__main__":
    G = get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True, bipartite=True).to_undirected()
    #print_graph_stats(G)
    degree_dist = get_degree_dist(G)
    #plot_degree_box_violin(G)
    plot_scale_free_network(degree_dist)