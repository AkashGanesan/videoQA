# Author	: Akash.Ganesan
# Email		: akaberto@umich.edu, akaberto@gmail.com
# TimeCreated	: Wed Apr 11 23:32:57 2018

"""This module contains graph combinators required for the video analysis"""

import networkx as nx
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import json
from functools import reduce
import glob
import os
import sys
import numpy as np
from itertools import chain, combinations
import networkx as nx
from networkx.algorithms import isomorphism


# The object_attr is used for annotating objects from the scene graph
object_attr = {
    "t" : "object",
    "fillcolor" : "red",
    "style" : "filled"}

# The predicate attr is used for annotating predicates from the scene
# graph
predicate_attr = {
    "t" : "predicate",
    "fillcolor" : "yellow",
    "style" : "filled"}

# The attribute attr is used for annotating attributes from the scene
# graph
attribute_attr = {
    "t" : "attribute",
    "fillcolor" : "green",
    "style" : "filled"}


def get_sg_from_json_obj(x):
    print ("Object got", x)
    print ("Keys avail", x.keys())
    return x.results[0].sg

def get_agg_from_json(json_in_folder,  length=None):
    """ Folder -> [Scene graph]"""
    agg = []

    cnt = 0
    for i in sorted(glob.glob(os.path.join(json_in_folder, "*.json"))):
        with open(i) as f:
            x = edict(json.loads(f.read()))            
            agg = agg + [x]

        if (length != None and cnt >= length):
            break
        
        cnt = cnt+1
    return agg
        



def sg_list_to_dg_list(json_frame):
    """ [JSON]  ->  [Nx.Graph]"""
    json_frame['sgNx'] = list(map(sg_to_dg, json_frame.results[0].sg))
    return json_frame
        


def sgs_list_to_dgs_list(json_frames):
    """ [[JSON]] -> [[Nx.Graph]] """
    return list(map(sg_list_to_dg_list, json_frames))

def combine_scene_graphs_list(json_in_folder):
    """ Folder  -> [JSON + sgNx] ; All the individual frames are combined"""
    x = get_agg_from_json(json_in_folder)
    print ("Length of x", len(x))
    print ("Length of sgs_to_dgs_list", len(sgs_list_to_dgs_list(x)))
    y = sgs_list_to_dgs_list(x)
    for i in y:
        sgNxList = i['sgNx']
        i['sgNxCombined'] = nx.compose_all(sgNxList) if sgNxList !=[] else []
    return y


def compose_nx_list(nx_list):
    return nx.compose_all(nx_list) if nx_list !=[] else None

def get_list_of_sgNx(mod_json):
    """ Given a list of JSONs with sgNx lists, this will return a list of jsons"""
    return list([ i['sgNx'] for i in  mod_json])

def sg_list_to_combined_graph(sg_list):
    dg_list = sg_list_to_dg_list(sg_list)
    return nx.compose_all(dg_list)


def sg_to_dg(sg):
    dg = nx.DiGraph()



    if (sg is not None):
        
        # Get all objects in an individual scene graph
        # objs = map(lambda (cnt,sg): (cnt,sg.names), enumerate(sg.objects))
        # print ("sg.objects", sg.keys())
        objs = dict(list(map(lambda x: (x[0],x[1].names[0])  ,enumerate(sg.objects))))
        
        for rel in sg.relationships:
            dg.add_node(objs[rel['subject']],  **object_attr)
            dg.add_node(rel['predicate'],  **predicate_attr)
            dg.add_node(objs[rel['object']],  **object_attr )
            
            
            dg.add_edge(objs[rel['subject']], rel['predicate'])
            dg.add_edge(rel['predicate'], objs[rel['object']])        
            
        for rel in sg.attributes:
            dg.add_node(objs[rel['subject']], **object_attr)
            dg.add_node(rel['attribute'], **attribute_attr)        
            dg.add_edge(objs[rel['subject']], rel['attribute'])

        
    return dg
    

def dotify(g, path):
    """dotify(networkx.classes.graph.Graph, path)
    Writes the path with the given xdot"""

    nx.nx_pydot.write_dot(g, path)

def graph_sim_frob(g1,g2):

    nodes_a = set(g1.nodes)
    nodes_b = set(g2.nodes)
    nodes_combined = nodes_a.union(nodes_b)
    nodes_combined = sorted(list(nodes_combined))
    adj1 = nx.to_numpy_matrix(g1, nodelist = nodes_combined)
    adj2 = nx.to_numpy_matrix(g2, nodelist = nodes_combined)

    return 1- ((np.linalg.norm(adj1-adj2)**2)/(len(g1.edges)+len(g2.edges)))




    

        
        
def getMCS(g1,g2):
    matching_graph=nx.Graph()
    for n1,n2,attr in g2.edges(data=True):
        if g1.has_edge(n1,n2) :
            matching_graph.add_edge(n1,n2,weight=1)
    graphs = list(nx.connected_component_subgraphs(matching_graph))
    mcs_length = 0
    mcs_graph = nx.Graph()
    for i, graph in enumerate(graphs):
        if len(graph.nodes()) > mcs_length:
            mcs_length = len(graph.nodes())
            mcs_graph = graph
    return len(mcs_graph.nodes())

def sim_score_mcs(g1,g2):    
    
    D = getMCS(g1, g2)
    res1 = ((D)/ (len(g1.nodes()) + len(g1.nodes()) - D))
    res = D / min(len(g1.nodes()), len(g2.nodes()))
    return res1

class Similarity():
    
    def __init__(self, G1, G2):
        self.G1 = G1 
        self.G2 = G2
        
        
    def getMCS(self):
       matching_graph=nx.Graph()
       for n1,n2,attr in self.G2.edges(data=True):
           if self.G1.has_edge(n1,n2) :
               matching_graph.add_edge(n1,n2,weight=1)
       graphs = list(nx.connected_component_subgraphs(matching_graph))
       mcs_length = 0
       mcs_graph = nx.Graph()
       for i, graph in enumerate(graphs):
           if len(graph.nodes()) > mcs_length:
               mcs_length = len(graph.nodes())
               mcs_graph = graph
       return len(mcs_graph.nodes())

    def sim_score(self):    
        # a = self.all_subgraphs(self.G2)
        D = self.getMCS()
        res1 = ((D)/ (len(self.G1.nodes()) + len(self.G1.nodes()) - D))
        res = D / min(len(self.G1.nodes()), len(self.G2.nodes()))
        return res1


if __name__=="__main__":
    x = edict({"relationships":[],"phrase":"","objects":[{"names":["cat"]}],"attributes":[{"predicate":"is","subject":0,"attribute":"brown","text":["cat","is","brown"],"object":"brown"}],"id":0,"url":"0"})

    y = edict({"relationships":[{"predicate":"of","subject":0,"text":["bowl","of","food"],"object":1}],"phrase":"","objects":[{"names":["bowl"]},{"names":["food"]}],"attributes":[{"predicate":"is","subject":0,"attribute":"white","text":["bowl","is","white"],"object":"white"}],"id":0,"url":"0"})
    
    z = edict({"relationships":[{"predicate":"eat","subject":0,"text":["cat","eat","food"],"object":1}],"phrase":"","objects":[{"names":["cat"]},{"names":["food"]}],"attributes":[],"id":0,"url":"0"})

    k = edict({"relationships":[{"predicate":"eat","subject":1,"text":["cat","eat","food"],"object":0}],"phrase":"","objects":[{"names":["cat"]},{"names":["food"]}],"attributes":[],"id":0,"url":"0"})
    

    xG = sg_to_dg(x)
    yG = sg_to_dg(y)
    zG = sg_to_dg(z)
    kG = sg_to_dg(k)
    
    
    dotify(xG, "x.dot")
    dotify(yG, "y.dot")
    dotify(zG, "z.dot")
    dotify(kG, "z.dot")
    
    
    dotify(nx.compose_all([xG,yG, zG,kG]), "merged.dot")

