import numpy as np
import threading
import time
import json
from pathlib import Path
import sys, os
import shutil
from collections import namedtuple
import argparse
import re
curr_dir = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import re
from util_render import add_close, get_chars
import shutil

sys.path.append("..")
#from tools.plotting_code_planner import plot_graph_2d_v2 as plot_graph_2d

# import plotly.io
sys.path.append(os.path.join("virtualhome", "simulation"))
'''sys.path.remove(
    "/data/vision/torralba/frames/data_acquisition/projects/vh_collect_data/virtualhome/src"
)'''
import argparse
import pickle
#from flask import Flask, render_template, request, redirect, Response, send_file, flash
from virtualhome.simulation.unity_simulator import comm_unity
#import vh_tools
import random, json
from datetime import datetime
import time
from PIL import Image
import io

# import agents
import base64

# from util import utils_rl_agent

import matplotlib.pyplot as plt

sys.path.insert(0, f"{curr_dir}/../online_watch_and_help/")
from agents import language
sys.path.insert(0, f"{curr_dir}/")

#app = Flask(__name__)
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

image_top = None
comm = None
lc = None
instance_colors = None
current_image = None
images = None
prev_images = None
graph = None
id2node = None
aspect_ratio = 9.0 / 16.0
bad_class_name_ids = []
curr_task = None
last_completed = {}

extra_agent = None
next_helper_action = None

# parameters for record graph
time_start = None
record_graph_flag = True  # False
# vis_graph_flag = True
graph_save_dir = None

record_step_counter = 0

message_history = []

buffer_instr = []
locker = None

# Contains a mapping so that objects have a smaller id. One unique pr object+instance
# instead of instance. glass.234, glass.267 bcecome glass.1 glass.2
class2numobj = {}
id2classid = {}

task_index = -1  # for indexing task_id in task_group
reset_counter = -1
task_index_shuffle = []

current_goal = {}

parser = argparse.ArgumentParser(description="Collection data simulator.")
parser.add_argument(
    "--deployment", type=str, choices=["local", "remote"], default="remote"
)
parser.add_argument("--simversion", type=str, choices=["v1", "v2"], default="v1")
parser.add_argument("--res", type=int, default=400)
parser.add_argument("--hres", type=int, default=600)
parser.add_argument("--portflask", type=int, default=5005)
parser.add_argument("--portvh", type=int, default=8180)
parser.add_argument("--x-display", type=str, default='1')
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--showmodal", action="store_true")
parser.add_argument(
    "--task_group",
    type=int,
    nargs="+",
    default=None,
    help="usage: --task group 41 42 43 44",
)
parser.add_argument(
    "--exp_name", type=str, default="single_agent_partial_test_20_human_test_highres"
)
parser.add_argument(
    "--extra_agent", type=str, nargs="+", default=None
)  # none, planner, rl_mcts

parser.add_argument("--start-id", type=int, default=0)
parser.add_argument("--end-id", type=int, default=1)
parser.add_argument('--trim_type', choices=['none', 'last_comm_10', 'putback'], default='none', help='Select the type of action trimming: none, trim after last communication plus 5 steps, or trim after putback action.')
parser.add_argument("--id", type=int, default=0)
args = parser.parse_args()

env_dataset = "/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/dataset/new_datasets/full_dataset.pik"
def convert_image(img_array):
    # cv2.imwrite('current.png', img_array.astype('uint8'))
    img = Image.fromarray(img_array.astype("uint8"))
    # file_like = cStringIO.cStringIO(img)
    file_object = io.BytesIO()
    img.save(file_object, "JPEG")
    img_str = base64.b64encode(file_object.getvalue())
    return img_str


def get_camera_pos(center_p, size, comm):
    c1_p = [
        center_p[0] + size[0] * 0.4,
        center_p[1] + size[1] * 0.15,
        center_p[2] - size[2] * 0.0,
    ]
    c2_p = [
        center_p[0] + size[0] * 0.0,
        center_p[1] + size[1] * 0.15,
        center_p[2] - size[2] * 0.3,
    ]
    c3_p = [
        center_p[0] - size[0] * 0.4,
        center_p[1] + size[1] * 0.15,
        center_p[2] + size[2] * 0.0,
    ]
    c4_p = [
        center_p[0] - size[0] * 0.0,
        center_p[1] + size[1] * 0.15,
        center_p[2] + size[2] * 0.3,
    ]
    _, message = comm.add_camera(position=c1_p, rotation=[20, -90, 0], field_view=90)
    _, message = comm.add_camera(position=c2_p, rotation=[20, 0, 0], field_view=90)
    _, message = comm.add_camera(position=c3_p, rotation=[20, 90, 0], field_view=90)
    _, message = comm.add_camera(position=c4_p, rotation=[20, 180, 0], field_view=90)

    # To get the camera list
    s, c = comm.camera_count()
    return [c - 1, c - 2, c - 3, c - 4]


ROOM_LIST = ["kitchen", "livingroom", "bathroom", "bedroom"]


room2cam_id = {
    0: {"kitchen": 2, "livingroom": 3, "bedroom": 1, "bathroom": 3},
    1: {
        "kitchen": 0,
        "livingroom": 1,
        "bedroom": 3,
        "bathroom": 2,
    },  # check what's inside living room
    2: {"kitchen": 3, "livingroom": 1, "bedroom": 2, "bathroom": 1},
    3: {"kitchen": 3, "livingroom": 0, "bedroom": 1, "bathroom": 2},
    4: {"kitchen": 2, "livingroom": 1, "bedroom": 3, "bathroom": 2},
    5: {"kitchen": 1, "livingroom": 2, "bedroom": 3, "bathroom": 2},
    6: {"kitchen": 0, "livingroom": 3, "bedroom": 1, "bathroom": 1},
}


def get_id2node(g):
    return {n["id"]: n for n in g["nodes"]}


def get_room(node_id, g, id2node):
    for e in g["edges"]:
        if (
            e["from_id"] == node_id
            and e["relation_type"] == "INSIDE"
            and id2node[e["to_id"]]["class_name"] in ROOM_LIST
        ):
            return id2node[e["to_id"]]["class_name"], e["to_id"]
    return None, None

def read_one_episode(task_id, log_file_path, comm, test_list=None):
    room_list = ["kitchen", "bedroom", "livingroom", "bathroom"]
    save_dir = "/home/scai/Workspace_2/hshi33/video_new"
    port = 8080
    if task_id >= 4000:
        env_dataset = "/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/dataset/new_datasets/dataset_language_large.pik"
    else:
        env_dataset = "/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/dataset/new_datasets/full_dataset.pik"
    with open(env_dataset, "rb") as dataset_file:
        data = pickle.load(dataset_file)
    print(log_file_path)
    with open(log_file_path, "rb") as plan_data_file:
        plan_data = pickle.load(plan_data_file)
    if task_id >= 4000:
        task_id, env_id, init_graph = task_id, plan_data["env_id"], data[task_id-4000]["init_graph"]
    else:
        task_id, env_id, init_graph = task_id, plan_data["env_id"], data[task_id]["init_graph"]
    have_language = False
    for index, language in enumerate(plan_data["language"][0]):
        if language is not None:
            have_language = True
            break
    actions = plan_data["action"]
    with open("actions_pos.json", "r") as json_file:
        temp = json.load(json_file)
    if str(task_id) in temp.keys():
        actions = temp[str(task_id)]["action"]
        actions = {0: actions["0"], 1: actions["1"]}
        plan_data["language"] = temp[str(task_id)]["language"]
        plan_data["language"] = {0: temp[str(task_id)]["language"]["0"], 1: temp[str(task_id)]["language"]["1"]}
    else:
        actions = add_close(actions, plan_data["graph"], plan_data["language"])

    print("actions: ", actions)
    print("language: ", plan_data["language"]) 

    print("env_id: ", env_id)
    comm.reset(env_id)
    s, m = comm.expand_scene(init_graph)
    print("EXPAND: ", m)
    tmp_actions = []


    match = re.search(r'logs_episode\.(\d+)_iter', log_file_path)
    if match:
        episode_id = int(match.group(1))
        print(f"Episode ID: {episode_id}")
    else:
        raise ValueError("Episode ID not found in the log")
    frame_intervals = []
    all_frame_ids = []
    male_char, female_char = get_chars(task_id)

    print("Male char: ", male_char)
    print("Female char: ", female_char)

    if have_language:
        comm.add_character(male_char, initial_room=data[task_id-4000]["init_rooms"][0])
        comm.add_character(female_char, initial_room=data[task_id-4000]["init_rooms"][1])
    else:
        comm.add_character(male_char, initial_room=data[task_id]["init_rooms"][1])
        comm.add_character(female_char, initial_room="bathroom")
    t = 0
    changed = False
    continue_step = 0
    frame_prev = 0
    frame_post = 0
    file_name = 2
    have_raise = True
    s, g = comm.environment_graph()
    os.makedirs("{}/{}".format(save_dir, task_id), exist_ok=True)
    pickle.dump(
        g, open("{}/{}/init_graph.pik".format(save_dir, task_id), "wb")
    )
    while t < len(actions[0]):
        frame_prev = frame_post
        id2node = get_id2node(g)
        s, id2color = comm.instance_colors()
        pickle.dump(id2color, open("{}/{}/instance_colors_{}.pik".format(save_dir, task_id, t), "wb"))
        if not have_language and not changed:
            if actions[0][t] is None and actions[1][t] is None and not t+1== len(actions[0]) and actions[1][t+1] is not None:
                bathroom_id = [node["id"] for node in g["nodes"] if node["class_name"] == "bathroom"][0]
                actions[0][t] = "[walk] <bathroom> ({})".format(bathroom_id)
                room_id = [node["id"] for node in g["nodes"] if node["class_name"] == data[task_id]["init_rooms"][1]][0]
                actions[1][t] = "[walk] <{}> ({})".format(data[task_id]["init_rooms"][1], room_id)

                with open("{}/{}/timemark.json".format(save_dir, task_id), "w") as temp_file:
                    json.dump({"time_change_frame": frame_post}, temp_file)

                changed = True
        action_str = ""
        if actions[0][t] is not None:
            action_str += "<char0> {}".format(actions[0][t])
        if actions[1][t] is not None:
            if not action_str == "":
                action_str += "|"
            action_str += "<char1> {}".format(actions[1][t])
        script = [action_str]
        print(t, script)
        # t += 1
        if env_id in room2cam_id:
            if not have_language and actions[0][t] is None:
                print("step {}, check agent 2".format(t))
                room, romm_id = get_room(2, g, id2node)
                if actions[1][t] is not None:
                    for r in room_list:
                        if r in actions[1][t]:
                            room = r
                            break
                print(room)
            else:
                print("step {}, check agent 1".format(t))
                room, romm_id = get_room(1, g, id2node)
                if actions[0][t] is not None:
                    for r in room_list:
                        if r in actions[0][t]:
                            room = r
                            break
            if (env_id == 6 and room == "kitchen" and ("microwave" in actions[1][t] or "fridge" in actions[1][t])):
                camera_id = 3
            elif env_id == 3 and room == "kitchen" and actions[1][t] is not None and ("sofa" in actions[1][t]):
                camera_id = 1
            elif env_id == 3 and room == "livingroom" and actions[1][t] is not None and ("desk" in actions[1][t]):
                camera_id = 2
            else:
                camera_id = room2cam_id[env_id][room]
            room_node = [node for node in g["nodes"] if node["class_name"] == room][0]
            center_p = room_node["bounding_box"]["center"]
            size = room_node["bounding_box"]["size"]
            camera_ids = get_camera_pos(center_p, size, comm)
            s, cd = comm.camera_data(camera_ids)
            try:
                if not have_language:
                    s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], image_synthesis=["normal", "depth", "seg_inst"], output_folder=f"{save_dir}/{task_id}", save_pose_data=True, file_name_prefix=f"{file_name}")
                else:
                    if plan_data["language"][0][t] is not None:
                        language_info = {0: plan_data["language"][0][t], 1: plan_data["language"][1][t]}
                        language_info["time_change_frame"] = frame_post
                        if args.trim_type == 'none':
                            json_file_path = "{}/{}/language.json".format(save_dir, task_id)
                        elif args.trim_type == 'last_comm_10':
                            json_file_path = "/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_last_comm_10/{}/language.json".format(task_id)
                            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
                        elif args.trim_type == 'putback':
                            json_file_path = "/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_putback/{}/language.json".format(task_id)
                            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
                        else:
                            raise ValueError("Invalid trim type")

                        with open(json_file_path, "w") as json_file:
                            json.dump(language_info, json_file)
                        if args.trim_type == 'none':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], image_synthesis=["normal", "depth", "seg_inst"], output_folder=f"{save_dir}/{task_id}", save_pose_data=True, file_name_prefix=f"{file_name}")
                        elif args.trim_type == 'last_comm_10':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], output_folder="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_last_comm_10/{}".format(task_id), file_name_prefix="language")
                        elif args.trim_type == 'putback':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], output_folder="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_putback/{}".format(task_id), file_name_prefix="language")
                    else:
                        if args.trim_type == 'none':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], image_synthesis=["normal", "depth", "seg_inst"], output_folder=f"{save_dir}/{task_id}", save_pose_data=True, file_name_prefix=f"{file_name}")
                        elif args.trim_type == 'last_comm_10':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], output_folder="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_last_comm_10", file_name_prefix="{}".format(task_id))
                        elif args.trim_type == 'putback':
                            s, m = comm.render_script(script, recording=True, frame_rate=10, image_width=1280, image_height=720, camera_mode=[camera_ids[camera_id]], output_folder="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/video/trimmed_putback", file_name_prefix="{}".format(task_id))

            except Exception as e:
                if file_name >= 5:
                    raise Exception
                comm.close()
                print("Reset connetion to port {}".format(port))
                comm = comm_unity.UnityCommunication(file_name="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/linux_exec.v2.3.0.x86_64", port="8081", timeout_wait=100, x_display="1")
                comm.reset(env_id)
                s, m = comm.expand_scene(g)
                print("Restore scene at step: ", t)
                file_name += 1
                continue
            frame_post = 0
            for file in os.listdir("{}/{}".format(save_dir, task_id)):
                if "pik" in file or "json" in file:
                    continue
                frame_num = len([f for f in os.listdir(f"{save_dir}/{task_id}/{file}/0") if f.endswith("normal.png")])
                frame_post += frame_num
            frame_intervals.append([frame_prev, frame_post])
            print("frame interval:", t, frame_prev, frame_post)
            pickle.dump(
                cd[camera_id],
                open("{}/{}/camera_data_{}.pik".format(save_dir, task_id, t), "wb"),
            )
            pickle.dump(
                frame_intervals, open(f"{save_dir}/{task_id}/frame_intervals.pik", "wb")
            )
            if not s:
                raise Exception
        else:
            s, m = comm.render_script(script,recording=False,skip_animation=True,)
        t += 1 # TODO: to determine whether adding this line is correct
        continue_step += 1
        print(s, m)
        s, g = comm.environment_graph()
        pickle.dump(
            g, open("{}/{}/graph_{}.pik".format(save_dir, task_id, t), "wb")
        )

    pickle.dump(
        frame_intervals, open(f"{save_dir}/{task_id}/frame_intervals.pik", "wb")
    )
    frame_count = 0
    prev_frame_count = 0
    '''os.mkdir(f"{save_dir}/{task_id}/0")
    os.mkdir(f"{save_dir}/{task_id}/1")
    for file in os.listdir(f"{save_dir}/{task_id}"):
        prev_frame_count = frame_count
        if "pik" in file or "json" in file:
            continue
        if file == "1" or file == "0":
            continue
        temp_list = os.listdir(f"{save_dir}/{task_id}/{file}/0")
        temp_list.sort()
        for image in temp_list:
            if image.endswith("normal.png"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/0/{image}", f"{save_dir}/{task_id}/0/Action_{frame_count:04d}_0_normal.png")
                frame_count += 1
        frame_count = prev_frame_count
        for image in temp_list:
            if image.endswith("exr"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/0/{image}", f"{save_dir}/{task_id}/0/Action_{frame_count:04d}_0_depth.exr")
                frame_count += 1
        frame_count = prev_frame_count
        for image in temp_list:    
            if image.endswith("inst.png"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/0/{image}", f"{save_dir}/{task_id}/0/Action_{frame_count:04d}_0_seg_inst.png")
                frame_count += 1
        frame_count = prev_frame_count
        temp_list = os.listdir(f"{save_dir}/{task_id}/{file}/1")
        temp_list.sort()
        for image in temp_list:
            if image.endswith("normal.png"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/1/{image}", f"{save_dir}/{task_id}/1/Action_{frame_count:04d}_0_normal.png")
                frame_count += 1
        frame_count = prev_frame_count        
        for image in temp_list:
            if image.endswith("exr"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/1/{image}", f"{save_dir}/{task_id}/1/Action_{frame_count:04d}_0_depth.exr")
                frame_count += 1
        frame_count = prev_frame_count
        for image in temp_list:
            if image.endswith("inst.png"):
                shutil.copy(f"{save_dir}/{task_id}/{file}/1/{image}", f"{save_dir}/{task_id}/1/Action_{frame_count:04d}_0_seg_inst.png")
                frame_count += 1
        shutil.rmtree(f"{save_dir}/{task_id}/{file}")'''
    

def render():
    comm = comm_unity.UnityCommunication(file_name="/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/linux_exec.v2.3.0.x86_64", port='8081', timeout_wait=100, x_display="1")

    episode_list = [2709, 766, 1859, 404, 3367, 1917, 2726, 2808, 640, 549, 609, 538, 3289, 3096, 3412, 135, 2053, 801, 857, 144, 3082, 3325, 138, 3115, 3312, 642, 3068, 153, 1827, 3355, 3031, 3056, 3462, 263, 647, 557, 398, 1886, 784, 240, 3308, 913, 1931, 2079, 130, 634, 3058, 644, 1851, 195, 3475, 895, 1775, 3092, 780, 541, 147, 865, 389, 128, 154, 628, 3117, 3371, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 2584, 3292, 578, 2559, 571, 1802, 548, 225, 608, 218, 910, 2799, 3145, 42, 193, 2462, 3366, 211, 1758, 161, 1818, 3081, 949, 406, 3300, 3419, 397, 630, 417, 3130, 3315, 612, 3098, 3074, 1798, 121, 3166, 240, 2701, 149, 905, 801, 3119, 179, 3398, 1131, 129, 1817, 871, 1140, 3328, 532, 583, 1819, 949, 1892, 152, 655, 824, 3080, 906, 1811, 3077, 577, 790, 1858, 848, 3050, 3433, 1860, 3060, 3047, 3344, 3065, 528, 682, 931, 1765, 1813, 601, 1893, 956]
    language_list = [4375, 5039, 4181, 5427, 5442, 5257, 5082, 5175, 5103, 4164, 4181, 5427, 5470, 4043, 4370, 5379, 4014, 4172, 4063, 4002, 4363, 4057, 4110, 4034, 5024, 4009, 5176, 5158, 4005, 4224, 4576, 5165, 4542, 5422, 5335, 5356, 4567, 5203, 5302, 4617, 4575, 4436, 4438, 4018, 4162, 4117, 5350, 4453, 5499, 5433, 4033, 5355, 4054, 4520, 5464, 4234, 5477, 4077, 5042, 4516, 4329, 5105, 4546, 4331, 4445, 5412, 5010, 5313, 5461, 4332, 4452, 4529, 4101, 5305, 4271, 4347, 4658, 5154, 5365, 4488, 4102, 4556, 4327, 4316, 4560, 5330, 4656, 4280, 4517, 4505, 4069, 4059, 4526, 4439, 5381, 4056, 4512, 4145, 4173, 4432, 5506, 5068, 4269, 5504, 4367, 4482, 4262, 5036, 4665, 4284, 5025, 4324, 5478, 5126, 4017, 4416, 4081, 4594, 4150, 5261, 5065, 4369, 5090, 4485, 5403, 4551, 4285, 5138, 4559, 5184, 4200, 5017, 4365, 5084, 4429, 5127, 4568, 4037, 4113, 5258, 4312, 4338, 5421, 4499, 4458, 5256, 5080, 5344, 5265, 5398, 5486, 4041, 5228, 4023, 5173, 4463, 4469, 4124, 4184, 4047, 4118, 5262, 4100, 4123, 4328, 4072, 4106, 5066, 4604, 4419, 4667, 5099, 5230, 5014, 4227, 5098, 5123, 4413, 4070, 4455, 5408, 4343, 5201, 4176, 5374, 4606, 4178, 4657, 4552, 5247, 5252, 4584, 4618, 5224, 4506, 4160, 5093, 4374, 4483, 5509, 4339, 4134, 5362, 4311, 4508, 4371, 4229, 4510, 5177]
    test_list = episode_list + language_list
    test_list = episode_list
    #test_list = episode_list
    test_list = [args.id]
    failed = []

    for test_id in tqdm(test_list, desc='Processing Episodes'):
        if Path("/home/scai/Workspace_2/hshi33/video_new/{}".format(test_id)).is_dir():
            print("Episode {} was rendered".format(test_id))
            continue
        if test_id > 4000:
            if args.trim_type == 'none':    
                log_file_dir = "/home/scai/Workspace/hshi33/virtualhome/data/dataset_language_large/language"
            elif args.trim_type == 'last_comm_10':
                log_file_dir = "/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/1500+episodes_trimmed_last_comm_10"
                os.makedirs(log_file_dir, exist_ok=True)
            elif args.trim_type == 'putback':
                log_file_dir = "/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/1500+episodes_trimmed_putback"
                os.makedirs(log_file_dir, exist_ok=True)
            else:
                raise ValueError("Invalid trim type")
        else:
            log_file_dir = "/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/nolang_episodes"
        if test_id in episode_list:
            log_file_path = log_file_dir + "/" + "logs_episode.{}_iter.0.pik".format(test_id)
        else:
            log_file_path = log_file_dir + "/" + "logs_episode.{}_iter.0.pik".format(test_id-4000)
        print(log_file_path)
        # read_one_episode(test_id, log_file_path, comm, test_list)
        try:
            read_one_episode(test_id, log_file_path, comm, test_list)
        except Exception as e:
            print(str(e))
            print("failed:", test_id)
            failed.append(test_id)
            raise Exception
    print("failed:", failed)


if __name__ == "__main__":
    render()
