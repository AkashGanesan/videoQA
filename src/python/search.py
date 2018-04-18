# Author	: Akash.Ganesan
# Email		: akaberto@umich.edu, akaberto@gmail.com
# TimeCreated	: Thu Apr 12 00:59:16 2018

import graphcombinator
from graphcombinator import *
import spacy
import nltk
from nltk.corpus import wordnet as wn
import networkx as nx
import numpy as np
from nltk.corpus import brown
from helpers import *
def get_neighbors(G,q):
    """ Nx.Graph -> seq.Query -> Maybe [Nodes]"""
    if q in G:
        return list(G.successors(q))
    return None


def get_predecessors(G,q):
    """ Nx.Graph -> seq.Query -> Maybe [Nodes]"""
    if q in G:
        return list(G.predecessors(q))
    return None



    

def get_suggestions_bk(G,q,depth=1):
    """Nx.Graph -> seq.Query -> Mabye [Nodes] """
    neighbors = get_predecessors(G,q[0])

    if len(q) == 1:
        return get_predecessors(G,q[0])
    
    if neighbors is not None:
        for i in q[1:]:
            if i in neighbors:
                neighbors = get_predecessors(G,i)
            else:                
                return None

        return list(get_predecessors(G,i))
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



    

def get_all_suggestions(G,q,depth=1):
    """Nx.Graph -> seq.Query -> Mabye [Nodes] """
    if depth == 0:
        return []
    if search_seq(G,q):
        sugg_list = []
        for l,v in G.succ[q[-1]].items():
            if q[0] in v:
                sugg_list.append([l] + get_all_suggestions(G,(q+[l]), depth=depth-1))
        return sugg_list
    else:
        return None


def word_sym(G,q,depth=1):
    list1 = 'woman'
    allsyns1 = set(ss for word in list1 for ss in wordnet.synsets(word))
    allsyns2 = set(ss for word in list2 for ss in wordnet.synsets(word))
    best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in 
        product(allsyns1, allsyns2))
    print(best)


def get_active_paths(G, o,i):
    """ For a given node, get probable paths """
    l = G[o][i]
    return list(l)



def get_suggestions_bk1(G, o, i ):
    """This takes in two nodes in a list and returns what are the
    predecessors of this sequence"""

        
    if search_seq(G,[o,i], False):        
        active_paths = get_active_paths(G, o, i)
        sugg_list = []
        for l,v in G.pred[o].items():
            if l in active_paths:
                sugg_list.append(l)
        return sugg_list
    else:
        return None    
    
def get_suggestions_n(G,q,depth=1):
    """Nx.Graph -> seq.Query -> Mabye [Nodes] """
    if search_seq(G,q):
        sugg_list = []
        for l,v in G.succ[q[-1]].items():
            if q[0] in v:
                sugg_list.append(l)
        return sugg_list
    else:
        return None


def reverse_search_seq(G,q):
    """ This will seach a reverse sequence """
    return search_seq(G,list(reversed(q)))





 
def search_seq(G,q,check_key=True):
    """This will search a sequence; If check_key is True, we only search
    paths which are annotated properly"""
    h = q[0]
    # Check if the first node is in the graph 
    ret = h in G
    if ret == False:
        return False
    elif (len(q) > 1):
        for i,ip1 in zip(q,q[1:]):
            try :
                if (check_key):
                # Highly execution order dependent
                    if (h in G[i][ip1] or
                        i in G[i][ip1]):
                        continue
                    else:                    
                        return False
                else:
                    if (G[i][ip1] or G[i][ip1]):
                        continue
                    else:
                        return False                    
            except KeyError:
                return False
        return True
    else:
        return ret

    
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
        
    

class FuzzySearch:
    """This class implements fuzzy similarity metric as well as getting
    actions of objects in an networkx object"""
    def __init__(self, graph,model=None):
        """model needs to be supplied.  Else, this is useless.  Uses spacy
        model"""
        self.nlp = model
        self.graph = graph
        self.nodes = sorted(list(graph.nodes))
        self.nodes_nlp = np.array([self.nlp(node) for node in self.nodes])
        self.nodes = np.array(self.nodes)
        self.verbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}
    
    def get_similar(self, key, threshold=0.75):    
        emb_key = self.nlp(key)
        sim_scores = np.array([emb_key.similarity(i) for i in self.nodes_nlp]) 
        return self.nodes[sim_scores > threshold]


    def get_actions(self, obj):
        lst = get_suggestions_n(self.graph, obj)
        val = []
        for i in lst:
            if first_word(i) in self.verbs:
                val.append(i)

        queries = map(lambda x : obj + [x], val)
        queries_search = map (lambda x : x + get_suggestions_n(self.graph, x), queries)
        return list(queries_search)



def search_a_and_b(graph_frames, a,b, check_key_a=True, check_key_b=True):
    """ Check if a seq occurs before b occurs and return indices if present """
    first_a = first_search_seq_temporal(graph_frames, a, check_key_a)
    first_b = first_search_seq_temporal(graph_frames, b, check_key_b)

    ret_val = False
    if (first_a == None or
        first_b == None):
        ret_val = False

    else:
        for frame in graph_frames[max(first_a, first_b):]:
            if (search_seq(frame, a, check_key_a) and  search_seq(frame,b, check_key_b)):
                return True
        ret_val =  False

    return ret_val


def get_relationships(graph, a,b):
    if a in graph and b in graph:
        l  = list(nx.all_shortest_paths (graph, a, b))
        return l
    else:
        return None


def search_a_then_b(graph_frames, a,b, check_key_a=False, check_key_b=False):
    """ Check if a seq occurs before b occurs and return indices if present """
    first_a = first_search_seq_temporal(graph_frames, a, check_key_a)
    first_b = first_search_seq_temporal(graph_frames, b, check_key_b)

    ret_val = False
    if (first_a == None or
        first_b == None):
        ret_val = False

    elif (first_b < first_a):
        ret_val = False
    else:
        ret_val =  True

    return ret_val, first_a, first_b
    

    
    
def search_a_then_b(graph_frames, a,b, check_key_a=False, check_key_b=False):
    """ Check if a seq occurs before b occurs and return indices if present """
    first_a = first_search_seq_temporal(graph_frames, a, check_key_a)
    first_b = first_search_seq_temporal(graph_frames, b, check_key_b)

    ret_val = False
    if (first_a == None or
        first_b == None):
        ret_val = False

    elif (first_b < first_a):
        ret_val = False
    else:
        ret_val =  True

    return ret_val, first_a, first_b
        
            

def search_seq_temporal( graph_frames, seq, check_key=True):
    """Check if the sequence is present across all frames and return a
    list of Bool values"""
    return [search_seq(i,seq,check_key) for i in graph_frames]

def first_search_seq_temporal ( graph_frames, seq, check_key=True):
    l = search_seq_temporal( graph_frames, seq, check_key=check_key)
    return l.index(True) if True in l else None
    
    

if __name__=="__main__":


    x_a = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video1_out')
    y_a = sgs_list_to_dgs_list(x_a)
    z_a = get_list_of_sgNx(y_a)
    seq_composed_a = list(map(nx.compose_all,z_a))
    full_composed_a = nx.compose_all(seq_composed_a)
    
    
        
    
    
    x_c = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video3_out')
    y_c = sgs_list_to_dgs_list(x_c)
    z_c = get_list_of_sgNx(y_c)
    seq_composed_c = list(map(nx.compose_all,z_c))
    full_composed_c = nx.compose_all(seq_composed_c)
    
    



    x_d = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video4_out')
    y_d = sgs_list_to_dgs_list(x_d)
    z_d = get_list_of_sgNx(y_d)
    seq_composed_d = list(map(nx.compose_all,z_d))
    full_composed_d = nx.compose_all(seq_composed_d)
    
    
    


    x_e = graphcombinator.combine_scene_graphs_list('/home/akash/learn/598/project/video-context-transcription/test/video5_out')
    y_e = sgs_list_to_dgs_list(x_e)
    z_e = get_list_of_sgNx(y_e)
    seq_composed_e = list(map(nx.compose_all,z_e))
    full_composed_e = nx.compose_all(seq_composed_e)



    nlp = spacy.load('en_core_web_lg')
    fuzz_a = FuzzySearch(full_composed_a,nlp)
    fuzz_c = FuzzySearch(full_composed_c,nlp)
    fuzz_d = FuzzySearch(full_composed_d,nlp)
    fuzz_e = FuzzySearch(full_composed_e,nlp)
