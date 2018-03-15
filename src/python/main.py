from pipeline import Pipeline


dcapParams = {"api_key" : 'a59cd502-6bc9-4aac-a1af-642bec4fc71c',
              "request_url" :"https://api.deepai.org/api/densecap"}

sceneGraphParams = {"fw_file_name" : "tmpout",
                    "fr_file_name" : "tmpout",
                    "scene_graph_exe" : os.path.abspath("../../bin/sceneGraph.jar"),
                    "start_str" : "#start"}


pipeLine = Pipeline(sceneGraphParams, dcapParams)
