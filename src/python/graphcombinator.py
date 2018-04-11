import networkx as nx
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import json
from functools import reduce
object_attr = {
    "type" : "object",
    "fillcolor" : "red",
    "style" : "filled"}

predicate_attr = {
    "type" : "predicate",
    "fillcolor" : "yellow",
    "style" : "filled"}

attribute_attr = {
    "type" : "attribute",
    "fillcolor" : "green",
    "style" : "filled"}


def apply_on_sgs(func, json_in_folder):
    agg = []
    # print("Length", len(sorted(glob.glob(os.path.join(json_in_folder, "*.json")))))
    for i in sorted(glob.glob(os.path.join(json_in_folder, "*.json"))):
        with open(i) as f:
            try :
                x = edict(json.loads(f.read()))            
                agg = agg + (x.results[0].sg)
            except:
                agg = agg + None
            
    return func(list(agg))
    

def get_aggregate_graph(json_in_folder):
    return apply_on_sgs(sg_list_to_combined_graph, json_in_folder)
        
                

def sg_list_to_combined_graph(sg_list):
    dg_list = sg_list_to_dg_list(sg_list)
    return nx.compose_all(dg_list)

def sg_list_to_dg_list(sg_list):
    return map(sg_to_dg, sg_list)

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


