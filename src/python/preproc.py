"""This module will take a video and provide individual frames in a
given directory.  We can also choose what mode of frame capture do we
want
"""

import argparse
import os
import subprocess
import shlex

class GetFrames:

    def __init__(self):
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("video",
                                 type=str,
                                 help="display a square of a given number")
        self.parser.add_argument("dir",
                                 help="Out dir for frames",
                                 type=str)
        self.parser.add_argument("-a", "--action",
                                 help="Options to ffmpeg",
                                 type=str,
                                 default="")
        self.args = self.parser.parse_args()
        
        assert (os.path.isfile(self.args.video)), \
            ("Video not present : %s" % self.args.video)

        self.command = ['ffmpeg','-i', self.args.video]

        if (self.args.action == ""):
            self.options = shlex.split('-vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 2')
            print ( self.options)
        else :            
            self.options = shlex.split(self.args.action)
        print ( "Options ;", self.options)
        self.full_command = self.command + self.options        

    def run(self):    
        print (self.full_command)
        


if __name__=="__main__":

    a = GetFrames()
    a.run()
