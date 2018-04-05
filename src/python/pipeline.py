from densecap import DenseCapFetcher
from scenegraph import SceneGraph
from easydict import EasyDict as edict
import json


class Pipeline(object):

    def __init__(self,
                 patterns,
                 image_path,                 
                 sceneGraphParams,                 
                 dcapParams):
        
        self.patterns = patterns
        self.sceneGraphParams = sceneGraphParams
        self.dcapParams = dcapParams
                
        self.dCapFetcher = DenseCapFetcher(**self.dcapParams)
        self.sceneGraphGen = SceneGraph(**self.sceneGraphParams)        
        self.sceneGraphGen.start()
        
        
        # def single_step(self,image_path):

    def preproc(self,caption):
        return caption.strip('\n')+'.'
        
    def single_step(self, img_path):
        densecapDict = self.dCapFetcher.get_json(img_path)
        
        for captionDict in  densecapDict.output.captions:
            caption = self.preproc(captionDict.caption)
            sceneGraph = edict(json.loads(self.sceneGraphGen.getVal(caption)))
            captionDict['sg'] = sceneGraph
            break
        return densecapDict
                
