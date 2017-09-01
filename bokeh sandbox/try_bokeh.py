__author__ = 'bc'

import networkx as nx
import numpy as np

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx

G=nx.karate_club_graph()

plot = figure(title="Networkx Integration Demonstration", x_range=(-1.1,1.1), y_range=(-1.1,1.1),
              tools="", toolbar_location=None)

graph = from_networkx(G, nx.random_layout)
plot.renderers.append(graph)

output_file("networkx_graph.html")
show(plot)

# initilize a random graph with networkx


# try to draw a random graph using bokeh



# animations on the graph