from densecap import DenseCapFetcher
from scenegraph import SceneGraph




class Pipeline(object):

    def __init__(self,
                 sceneGraphParams,
                 dcapParams):
        self.sceneGraphParams = sceneGraphParams
        self.dcapParams = dcapParams
                
        self.dCapFetcher = DenseCapFetcher(**self.dcapParams)
        self.sceneGraphGen = SceneGraph(**self.sceneGraphParams)        
        self.sceneGraphGen.start()
        
        
        # def single_step(self,image_path):
        
        
        
