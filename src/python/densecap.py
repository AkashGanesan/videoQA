import requests
import json
import os
from easydict import EasyDict as edict
import subprocess
import shlex
from string import Template

class DenseCapFetcher():
    def __init__(self,
                 api_key='a59cd502-6bc9-4aac-a1af-642bec4fc71c',
                 request_url="https://api.deepai.org/api/densecap"):
        self.request_url = request_url
        self.api_key = api_key
        self.dict = dict()

    def get_json(self,image_path):

        r = requests.post(
            self.request_url,
            files={
                'image': open(os.path.abspath(image_path), 'rb'),
            },
            headers={'api-key': self.api_key})

        dump_dict = r.json()

        return edict(dump_dict)



class DenseCapFetcherLocal():
    def __init__(self,
                 cmd="th run_model.lua -input_image ${input_image} -output_dir ${output_dir} -gpu -1",
                 densecap_dir="/home/akash/bin/densecap",
                 output_dir="default_out_dir"):
        self.command_format_template = Template(cmd)
        self.densecap_dir = densecap_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = os.path.abspath(output_dir)
    def get_json(self, input_image):
        cmd = self.command_format_template.substitute({"input_image": input_image,
                                                       "output_dir": self.output_dir})
        cmd_shell = shlex.split(cmd)

        subprocess.call(cmd_shell, cwd=self.densecap_dir)
        with open(os.path.join(self.output_dir, "results.json")) as f:
            return edict(json.loads(f.read()))
                  
     


    
