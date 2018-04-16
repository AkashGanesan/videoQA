# Author	: Akash.Ganesan
# Email		: akaberto@umich.edu, akaberto@gmail.com
# TimeCreated	: Thu Apr 12 00:59:16 2018

import graphcombinator
from graphcombinator import *

import networkx as nx
import numpy as np

def get_neighbors(G,q):
    """ Nx.Graph -> seq.Query -> Maybe [Nodes]"""
    if q in G:
        return list(nx.neighbors(G,q))
    return None


def get_predecessor(G,q):
    """ Nx.Graph -> seq.Query -> Maybe [Nodes]"""
    if q in G:
        return list(nx.predecessor(G,q))
    return None



def get_suggestions_bk(G,q,depth=1):
    """Nx.Graph -> seq.Query -> Mabye [Nodes] """
    neighbors = get_predecessor(G,q[0])

    if len(q) == 1:
        return get_predecessor(G,q[0])
    
    if neighbors is not None:
        for i in q[1:]:
            if i in neighbors:
                neighbors = list(nx.predecessor(G,i))
            else:                
                return None

        return list(nx.predecessor(G,i))
    else:
        return None




def get_suggestions(G,q,depth=1):
    """Nx.Graph -> seq.Query -> Mabye [Nodes] """
    neighbors = get_neighbors(G,q[0])

    if len(q) == 1:
        return get_neighbors(G,q[0])
    
    if neighbors is not None:
        for i in q[1:]:
            if i in neighbors:
                neighbors = list(nx.neighbors(G,i))
            else:                
                return None

        return list(nx.neighbors(G,i))
    else:
        return None
    


def search_seq(G,q):
    """Nx.Graph -> seq.Query -> Bool"""
    neighbors = get_neighbors(G,q[0])
    if neighbors is not None:
        for i in q[1:]:
            if i in neighbors:
                neighbors = nx.neighbors(G,i)
            else:
                return False
        return True
    else:
        return False

            
    
def search_bow(G,q):
    """ Nx.Graph -> seq.Query -> Bool """
    types_list = nx.get_node_attributes(G,'t')
    return all(map (lambda x: x in G, q))



def first_fail_index(G,q):
    """ Nx.Graph -> seq.Query -> Index """
    x =  list(map (lambda x: x in G, q))
    if all(x) is False:
        return x.index(False), q[x.index(False)]
    return None, None
        
        

        

if __name__=="__main__":


    x_a = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video4_out')
    y_a = sgs_list_to_dgs_list(x_a)
    z_a = get_list_of_sgNx(y_a)
    seq_composed_a = list(map(nx.compose_all,z_a))
    full_composed_a = nx.compose_all(seq_composed_a)
    
    
    
    x_b = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video3_out')
    y_b = sgs_list_to_dgs_list(x_b)
    z_b = get_list_of_sgNx(y_b)
    seq_composed_b = list(map(nx.compose_all,z_b))
    full_composed_b = nx.compose_all(seq_composed_b)
    
    
    
    x_c = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video1_out')
    y_c = sgs_list_to_dgs_list(x_c)
    z_c = get_list_of_sgNx(y_c)
    seq_composed_c = list(map(nx.compose_all,z_c))
    full_composed_c = nx.compose_all(seq_composed_c)
    
    
