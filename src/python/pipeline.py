""" This is the logical pipeline of this entire project
This does the following

Frames dir --> 
Densecap --> 
CoreNLP for scene graphs --> 
Generate Graphs --> Similarity and other measures.

"""
import os
import sys
from pathlib import Path
from densecap import DenseCapFetcherLocal
from scenegraph import SceneGraph
from easydict import EasyDict as edict
import  graphcombinator
import json
import glob 


class Pipeline(object):

    def __init__(self,
                 patterns,
                 image_path,                 
                 sceneGraphParams,                 
                 dcapParams,
                 generalParams):

        
        self.patterns = patterns
        self.sceneGraphParams = sceneGraphParams
        self.dcapParams = dcapParams
                
        self.dCapFetcher = DenseCapFetcherLocal(**self.dcapParams)
        self.sceneGraphGen = SceneGraph(**self.sceneGraphParams)
        self.generalParams = generalParams
        self.sceneGraphGen.start()
        
        
        # def single_step(self,image_path):

    def preproc(self,caption):
        """Do whatever preprocessing that needs to be done before passing the
        captions on to scene graph parser"""
        return caption.strip('\n')+'.'


    def get_dense_cap(self, img_path):
        """ Gets the JSON from the dense cap library online"""
        return self.dCapFetcher.get_json(img_path)

    def get_captions(self, densecapDict):
        """Input : DenseCapDict shall a edict from the densecap fetcher
        call"""
        



        available_captions_len = len(densecapDict.results[0].captions)
        if (self.generalParams['depth_or_confidence'] == "depth"):
            depth_cnt = max(self.generalParams['depth'],available_captions_len)
        elif (self.generalParams['depth_or_confidence'] == "confidence"):
            depth_cnt = len(list((filter(lambda x: x > self.generalParams['confidence'],
                                         densecapDict.results[0].scores))))
        
        captions = list(map(self.preproc,densecapDict.results[0].captions[0:depth_cnt]))
        sceneGraphs = list(map(lambda x: edict(json.loads(self.sceneGraphGen.getVal(x))), captions))
        # sceneGraphsNx =  graphcombinator.sg_list_to_combined_graph (sceneGraphs)
        
        densecapDict.results[0].captions = densecapDict.results[0].captions[0:depth_cnt]
        densecapDict.results[0].scores = densecapDict.results[0].scores[0:depth_cnt]
        densecapDict.results[0].boxes = densecapDict.results[0].boxes[0:depth_cnt]
        densecapDict.results[0]['sg'] = sceneGraphs
        # densecapDict.results[0]['sgNx'] = sceneGraphsNx
        ## TODO : This can be turned into a lambda based dispatch.

                
        return densecapDict
        

    def single_step(self, img_path):
        densecapDict = self.get_dense_cap(img_path)
        get_captions = self.get_captions(densecapDict)
        return get_captions

    def write_to_dot(self, img_path):
        densecapDict = self.get_dense_cap(img_path)
        get_captions = self.get_captions(densecapDict)
        return get_captions
    



            
    
    def multi_step(self, img_folder_path, json_out_folder):        
        for i in sorted(glob.glob(os.path.join(img_folder_path, "*.jpg"))):
            print ("Processing: ", i)
            os.makedirs(json_out_folder,exist_ok=True)            
            write_path = (Path(json_out_folder).joinpath(Path(i).stem).with_suffix(".json"))
            with open(write_path, "w") as f:
                json.dump( self.single_step(i), f)
        



    
