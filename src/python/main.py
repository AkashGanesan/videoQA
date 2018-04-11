from pipeline import Pipeline
import sys
import glob
import os
import graphcombinator
dcapParams = {
    "cmd" : "th run_model.lua -input_image ${input_image} -output_vis_dir ${output_dir} -gpu -1",
    "densecap_dir" : "/home/akash/bin/densecap",
    "output_dir" :"default_out_dir"}

sceneGraphParams = {
    "fw_file_name" : "tmpout",
    "fr_file_name" : "tmpout",
    "scene_graph_exe" : os.path.abspath("../../bin/sceneGraph.jar"),
    "start_str" : "#start"}

generalParms = {
    "depth" : 10000,
    "confidence" : 0.5,
    "depth_or_confidence" : "confidence"
    }


pipelineParams = {
    "patterns" : ["*.jpg", "*.png"],
    "image_path" : None,
    "sceneGraphParams" : sceneGraphParams,
    "dcapParams"  : dcapParams,
    "generalParams" : generalParms }



# if __name__=="__main__":
#     if len(sys.argv) !=2:
#         error("Require a folder to get images")        
#         exit
#     vid_path = sys.argv[1]
#     if not os.path.isdir(vid_path):
#         pipeline_inst = Pipeline(**pipelineParams)
    

pipeline_inst = Pipeline(**pipelineParams)
a = pipeline_inst.multi_step('/home/akash/learn/598/project/video-context-transcription/test/video3/',
                             '/home/akash/learn/598/project/video-context-transcription/test/video3_out/')
