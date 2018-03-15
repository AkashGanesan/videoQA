import requests
import json
import os
from easydict import EasyDict as edict
class DenseCapFetcher():
    def __init__(self,api_key='a59cd502-6bc9-4aac-a1af-642bec4fc71c', request_url="https://api.deepai.org/api/densecap"):
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

        json_fname = "test.json"
        dump_dict = r.json()

        self.caps = edict(dump_dict)

