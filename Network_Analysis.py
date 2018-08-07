#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Network analysis with networkx

# use cases: social media marketing strategy, infrastructure system design,
#            financial risk manegement, public health management

# terms:
# nodes, edges, (un)directed graph, (un)directed edge, graph size(number of edges),
# graph order(number of vertices),
# degree(number of edges connected to a vertex, with loops counted twice)

# type of graph generators:
# graph drawing algorithms, network analysis algorithms, algorithmic routing for graphs,
# graph search algorithms, subgraph algorithms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import networkx as nx

#%matplotlib inline
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

G = nx.Graph()
nx.draw(G)

G.add_node(1)
nx.draw(G)

G.add_nodes_from([2,3,4,5,6,8,9,12,15,16])
nx.draw(G)

G.add_edges_from([(2,4),(2,6),(2,8),(2,12),(2,16),(3,6),(3,9),(3,12),(3,15),(4,8),(4,12),(4,16),(6,12),(8,16)])
nx.draw(G)

nx.draw_circular(G,node_color='bisque',with_labels=True)
nx.draw_spring(G)

G.remove_node(1)

sum_stats = nx.info(G)
print(sum_stats)
print(nx.degree(G))

G = nx.complete_graph(25)
nx.draw(G,node_color='bisque',with_labels=True)

G = nx.gnc_graph(7,seed=25) #directed graph
nx.draw(G,node_color='bisque',with_labels=True)

ego_G = nx.ego_graph(G,3,radius=5)
nx.draw(G,node_color='bisque',with_labels=True)


#=============================================================================


# Social network

# generate a graph object and edgelist
DG = nx.gn_graph(7,seed=25)
#for line in nx.generate_edgelist(DG,data=False):print(line)

# assign attributes to nodes
DG.node[0]['name'] = 'Alice'
print(DG.node[0])
DG.node[1]['name'] = 'Bob'
DG.node[2]['name'] = 'Claire'
DG.node[3]['name'] = 'Dennis'
DG.node[4]['name'] = 'Esther'
DG.node[5]['name'] = 'Frank'
DG.node[6]['name'] = 'George'

DG.add_nodes_from([(0,{'age':25}),(1,{'age':31}),(2,{'age':18}),(3,{'age':47}),(4,{'age':22}),(5,{'age':23}),(6,{'age':50})])
print(DG.node[0])

DG.node[0]['gender'] = 'f'
DG.node[1]['gender'] = 'm'
DG.node[2]['gender'] = 'f'
DG.node[3]['gender'] = 'm'
DG.node[4]['gender'] = 'f'
DG.node[5]['gender'] = 'm'
DG.node[6]['gender'] = 'm'

labeldict = {0: 'Alice',1:'Bob',2:'Claire',3:'Dennis',4:'Esther',5:'Frank',6:'George'}
nx.draw_circular(DG,labels=labeldict,node_color='bisque',with_labels=True)

G = DG.to_undirected()
nx.draw_spectral(G,labels=labeldict,node_color='bisque',with_labels=True)


#=============================================================================


# Analyze social networks

# degree, successors, neighbors
# in-degree, out-degree

print(nx.info(DG))
print(DG.degree())

nx.draw_circular(DG,node_color='bisque',with_labels=True)
DG.successors(3)
DG.neighbors(4)    #out-going nodes
G.neighbors(4)     #connected nodes

