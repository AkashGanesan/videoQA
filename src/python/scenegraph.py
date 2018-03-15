from subprocess import Popen, PIPE
from time import sleep
import os
from easydict import EasyDict as edict
import json



class SceneGraph(object):

    def __init__(self,
                 scene_graph_exe,
                 fw_file_name="tmpout",
                 fr_file_name="tmpout",
                 start_str="#start"):        
        self.init = 0
        self.start_str = start_str
        self.fw = open(fw_file_name, "wb")
        self.fr = open(fr_file_name, "r")
        self.p = Popen(["java", "-jar", scene_graph_exe], stdin = PIPE, stdout = self.fw, stderr = self.fw, bufsize = 1)


    def start(self):
        while(1):
            init_str = self.fr.readline().strip()
            if (init_str == self.start_str):
                self.fr.readline()
                break
            sleep(1)


        self.init = 1
        print "Scene Graph initialization is done."
    
    def getVal(self, sentence):
        if (self.init == 0):
            print "Havent inited"
            return None
        self.p.stdin.write(sentence+'\n')
        while(1):            
            val = self.fr.readline()
            if (val == ''):                
                sleep(0.1)
            else:
                return  val
        

