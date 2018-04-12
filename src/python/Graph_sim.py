# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:54:03 2018

@author: Shubha
"""

import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
import sys
import json
import nltk
from itertools import chain, combinations

class Similarity():
    
    def __init__(self, G1, G2):
        self.G1 = G1 
        self.G2 = G2
        
#    def powerset(self, iterable): 
#        s = list(iterable) 
#        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)) 
#    
#    def all_subgraphs(self, graph): 
#            a= list(range(len(graph.nodes()))) 
#            a.reverse() 
#            for vertices in self.powerset(a): 
#                yield graph.subgraph(vertices) 
    
    ## then use builtin-function: .subgraph_is_isomorphic() 
    ## (reversed list, to start with longest subgraphs first and then we 
    #can break, when one is found)               
#    def MCS_ret(self, subgraphs):
#        for sg2 in reversed(list(subgraphs)):
#            MCS = 0
#            if len(sg2.nodes()) == 0:
#                continue
#            GM = isomorphism.GraphMatcher(self.G1, sg2)
#            if GM.subgraph_is_isomorphic():
#                for isomorph in GM.subgraph_isomorphisms_iter():
#                    if MCS < len(isomorph):
#                        MCS = len(isomorph)
#            return MCS
        
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
    
if __name__ == "__main__":
    
    # two graphs intialized
#    G2 = nx.DiGraph()
#    G2.add_edge('1','3')
#    G2.add_edge('2','5')
#    G2.add_edge('9','3')
#    G2.add_edge('1','4')
#    G2.add_edge('2','7')
#    G2.add_edge('7','6')
#        
#    G1 = nx.DiGraph()
#    G1.add_edge('1','7')
#    G1.add_edge('3','6')
#    G1.add_edge('9','3')
#    G1.add_edge('1','5')
#    G1.add_edge('1','2')
#    G1.add_edge('8','4')
#    G1.add_edge('6','8')
#    G1.add_edge('3','4')
#    G1.add_edge('10','1')
    
    G2 = nx.DiGraph()
    G2.add_edge(1,3)
    G2.add_edge(2,5)
    G2.add_edge(9,3)
    G2.add_edge(1,4)
    G2.add_edge(2,7)
    G2.add_edge(7,6)
        
    G1 = nx.DiGraph()
    G1.add_edge(1,7)
    G1.add_edge(3,6)
    G1.add_edge(9,3)
    G1.add_edge(1,5)
    G1.add_edge(1,2)
    G1.add_edge(8,4)
    G1.add_edge(6,8)
    G1.add_edge(3,4)
    G1.add_edge(10,1)
    
    Sim = Similarity(G1,G2)
    print (Sim.sim_score())

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            	           