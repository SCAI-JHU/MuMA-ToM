#This script analyze object relationships from instance segmentation to get accurate object relationship. It is not used in LIMP. We provide this and you can try using instance segementation results as visual outputs
import cv2
import ipdb
import math
import json
import pickle
import imageio
import sys, os
import argparse
import numpy as np
import copy
import shutil
from tqdm import tqdm
from collections import defaultdict
import random
from numba import cuda

from env_utils import ALL_LIST, ROOM_LIST, CONTAINER_LIST, SURFACE_LIST, OBJECT_LIST, ROOM_COMPONENTS, ROOM_POSSIBILITY
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
sys.path.append('..')


def load_pickles(path, folder=True):
    if folder:
        pickle_data = {}
        for file in os.listdir(path):
            if file.endswith(".pik"):
                with open(os.path.join(path, file), 'rb') as f:
                    data = pickle.load(f)
                    pickle_data[file] = data
    else:
        if path.endswith(".pik"):
            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
    return pickle_data


def read_frame_intervals(parent_path):
    path = parent_path + 'frame_intervals.pik'
    with open(path, 'rb') as f:
        intervals = pickle.load(f)
    return intervals


def find_pixel_id(time, parent_path, parent_path_video):
    with open(parent_path + "frame_intervals.pik", "rb") as file:
        intervals = pickle.load(file)
    for index, interval in enumerate(intervals):
        if time >= interval[0] and time <= interval[1]:
            num = index
    path = parent_path + 'instance_colors_{}.pik'.format(num)
    with open(path, 'rb') as f:
        id2color = pickle.load(f)

    path = parent_path + 'graph_{}.pik'.format(num+1)
    with open(path, 'rb') as f:
        g = pickle.load(f)

    id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}
    id2color_filtered = {
            int(i): [int(x * 255) for x in rgb]
            for i, rgb in id2color.items()
            if int(i) in id2classname and id2classname[int(i)] in ALL_LIST
        }
    path = parent_path_video + f"Action_{time:04d}_0_seg_inst.png"
    imgs_seg = cv2.imread(path)
    img_shape = imgs_seg.shape
    height, width, channels = imgs_seg.shape
    img_id = np.empty((height, width, 1))
    object_list = []

    for x in range(img_shape[0]):
        for y in range(img_shape[1]):
            curr_rgb = imgs_seg[x, y, :].astype(int) 
            found_id = None
            for object_id, rgb in id2color_filtered.items():
                match = True
                for c_id in range(3):
                    if abs(curr_rgb[2 - c_id] - int(rgb[c_id])) > 1: 
                        match = False
                        break
                if match:
                    found_id = object_id
                    break

            img_id[x, y, 0] = -1
            if found_id is not None:
                if found_id in id2classname:
                    object_classname = id2classname[found_id]
                    if object_classname in ALL_LIST:
                        img_id[x, y, 0] = found_id
                        if object_classname not in object_list:
                            object_list.append(object_classname)

    return img_id, id2classname


def inverse_rot(currrot):
    new_matrix = np.zeros((4,4))
    new_matrix[:3, :3] = currrot[:3,:3].transpose()
    new_matrix[-1,-1] = 1
    new_matrix[:-1, -1] = np.matmul(currrot[:3,:3].transpose(), -currrot[:-1, -1])
    
    return new_matrix


def project_pixel_to_3d(rgb, time, num, parent_path, parent_path_video, scale=1, ax=None, ij=None):
    path = parent_path_video + f'Action_{time:04d}_0_depth.exr'
    try:
        depth = imageio.imread(path)
    except FileNotFoundError:
        time += 1
        path = parent_path_video + f'Action_{time:04d}_0_depth.exr'
        depth = imageio.imread(path)
    depth = depth[:,:,0]
    with open(parent_path + "frame_intervals.pik", "rb") as file:
        intervals = pickle.load(file)
    for index, interval in enumerate(intervals):
        if time >= interval[0] and time <= interval[1]:
            num = index
    path = parent_path + f'camera_data_{num}.pik'
    with open(path, 'rb') as f:
        params = pickle.load(f)


    (hs, ws, _) = rgb.shape
    naspect = float(ws/hs)
    aspect = params['aspect']
    
    w = np.arange(ws)
    h = np.arange(hs)
    projection = np.array(params['projection_matrix']).reshape((4,4)).transpose()
    view = np.array(params['world_to_camera_matrix']).reshape((4,4)).transpose()
    
    # Build inverse projection
    inv_view = inverse_rot(view)
    rgb = rgb.reshape(-1,3)
    col= rgb
    
    xv, yv = np.meshgrid(w, h)
    npoint = ws*hs
    
    # Normalize image coordinates to -1 to 1
    xp = xv.reshape((-1))
    yp = yv.reshape((-1))
    if ij is not None:
        index_interest = ij[0]*ws + ij[1]
    else:
        index_interest = None
     
    x = xv.reshape((-1)) * 2./ws - 1
    y = 2 - (yv.reshape((-1)) * 2./hs) - 1
    z = depth.reshape((-1))
    
    nump = x.shape[0]
    
    m00 = projection[0,0]
    m11 = projection[1,1]
    
    xn = x*z / m00
    yn = y*z / m11
    zn = -z
    XY1 = np.concatenate([xn[:, None], yn[:, None], zn[:, None], np.ones((nump,1))], 1).transpose()

    # World coordinates
    XY = np.matmul(inv_view, XY1).transpose()

    x, y, z = XY[:, 0], XY[:, 1], XY[:, 2]
    if ij is not None:
        print("3D point", x[index_interest], y[index_interest], z[index_interest])
    return x,y,z


def find_bounding_boxes(pixel2id, x, y, z, id2classname):
    id_array = pixel2id.reshape(-1)

    grouped_positions = defaultdict(list)
    for position, id_value in enumerate(id_array):
        if id_value != 255:
            grouped_positions[id_value].append(position)
    grouped_positions = dict(grouped_positions)

    bounding_boxes = {}  
    for id_value, positions in grouped_positions.items():
        if id_value in id2classname:
            bounding_box = {}
            bounding_box['class_name'] = id2classname[id_value]
            x_values = [x[pos] for pos in positions]
            bounding_box['x_max'], bounding_box['x_min'] = max(x_values), min(x_values)
            y_values = [y[pos] for pos in positions]
            bounding_box['y_max'], bounding_box['y_min'] = max(y_values), min(y_values)
            z_values = [z[pos] for pos in positions]
            bounding_box['z_max'], bounding_box['z_min'] = max(z_values), min(z_values)
            bounding_boxes[id_value] = bounding_box
    return bounding_boxes


def calculate_1d_distance(dim1, dim2):
    if dim1['max'] < dim2['min']:
        return dim2['min'] - dim1['max']
    elif dim2['max'] < dim1['min']:
        return dim1['min'] - dim2['max']
    else:
        return 0.0


def calculate_distance_from_object(bounding_boxes, id):
    try:
        if isinstance(random.choice(list(bounding_boxes.keys())), (int, float, complex)):
            object_1 = bounding_boxes[id]
        else:
            print("208")
            print(str(id))
            object_1 = bounding_boxes[str(id)]
            print("210")
            print("object 1: ", object_1)

    except KeyError:
        return None

    # Prevent comparison with itself
    self_id = id
    
    distances = {}
    print("220")
    print("id: ", id)
    print("str(self_id): ", str(self_id))
    try:
        for id, obj_data in bounding_boxes.items():
            if isinstance(random.choice(list(bounding_boxes.keys())), (int, float, complex)):
                if id == self_id:
                    print("skip character itself")
                    continue
            else:
                if id == str(self_id):
                    print("skip character itself")
                    continue

            x_dist = calculate_1d_distance({'min': object_1['x_min'], 'max': object_1['x_max']}, {'min': obj_data['x_min'], 'max': obj_data['x_max']})
            y_dist = calculate_1d_distance({'min': object_1['y_min'], 'max': object_1['y_max']}, {'min': obj_data['y_min'], 'max': obj_data['y_max']})
            z_dist = calculate_1d_distance({'min': object_1['z_min'], 'max': object_1['z_max']}, {'min': obj_data['z_min'], 'max': obj_data['z_max']})
            dist = math.sqrt(x_dist**2 + z_dist**2)

            distances[float(id)] = dist
            print("dist: ", dist)
    except KeyError:
        return None
    return distances

@cuda.jit(device=True)
def calculate_1d_distance_gpu(min1, max1, min2, max2):
    if max1 < min2:
        return min2 - max1
    elif max2 < min1:
        return min1 - max2
    else:
        return 0.0

@cuda.jit
def calculate_distance_kernel(bboxes, distances, self_index):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    
    if i < bboxes.shape[0] and i != self_index:
        x_dist = calculate_1d_distance_gpu(bboxes[self_index, 0], bboxes[self_index, 1], bboxes[i, 0], bboxes[i, 1])
        z_dist = calculate_1d_distance_gpu(bboxes[self_index, 4], bboxes[self_index, 5], bboxes[i, 4], bboxes[i, 5])
        dist = math.sqrt(x_dist**2 + z_dist**2)
        distances[i] = dist

def calculate_distance_from_object_gpu(bounding_boxes, id):
    id_to_index = {key: idx for idx, key in enumerate(bounding_boxes.keys())}
    index_to_id = {idx: key for key, idx in id_to_index.items()}  
    bboxes_array = np.array([[obj['x_min'], obj['x_max'], obj['y_min'], obj['y_max'], obj['z_min'], obj['z_max']] for obj in bounding_boxes.values()])
    distances = np.zeros(len(bounding_boxes))
    
    threads_per_block = 32
    blocks_per_grid = (len(bounding_boxes) + (threads_per_block - 1)) // threads_per_block

    if isinstance(random.choice(list(bounding_boxes.keys())), (int, float, complex)):
        self_index = id_to_index.get(id, None)
    else:
        self_index = id_to_index.get(str(id), None)

    if self_index is None:
        print(f"ID {id} not found in bounding_boxes.")
        return None

    calculate_distance_kernel[blocks_per_grid, threads_per_block](bboxes_array, distances, self_index)

    final_distances = {float(index_to_id[i]): dist for i, dist in enumerate(distances) if i != self_index}
    return final_distances

# def find_close_to(distances, id2classname, threshold = 1):
#     close_to_items = []
#     for id, distance in distances.items():
#         if distance <= threshold and id2classname[id] not in ROOM_LIST:
#             close_to_items.append(id)

#     return close_to_items

def find_close_to(distances, id2classname, threshold = 1):
    close_to_items = []
    print("distances in find close: ", distances)
    if isinstance(distances, np.ndarray):
        raise ValueError("distances should be a dictionary, not a numpy array.")
    else:
        for id, distance in distances.items():
            if distance <= threshold and id2classname[id] not in ROOM_LIST:
                close_to_items.append(id)

    return close_to_items


# def find_closeness(times, parent_path, parent_path_video, character_id):
# '''
#     character_id: 1 for agent 0, 2 for agent 1.
# '''
#     all_closeness = {}
#     all_bounding_boxes = {}
#     num = 0
#     for time in tqdm(times, "Processing frames"):
#         pixel2id, id2classname = find_pixel_id(time, parent_path, parent_path_video)
#         x, y, z = project_pixel_to_3d(pixel2id, time, num, parent_path, parent_path_video)
#         bounding_boxes = find_bounding_boxes(pixel2id, x, y, z, id2classname)
#         all_bounding_boxes[num] = bounding_boxes

#         distances = calculate_distance_from_object(bounding_boxes, character_id)
#         if distances == None:
#             close_to_items = []
#         else:
#             close_to_items = find_close_to(distances, id2classname)

#         close_to_items = [id2classname[id] for id in close_to_items]
#         all_closeness[num] = list(set(close_to_items))
#         num += 1

#     print(all_bounding_boxes)

#     return all_bounding_boxes, all_closeness

def find_bounding_boxes_for_times(times, parent_path, parent_path_video):
    all_bounding_boxes = {}
    num = 0

    for time in tqdm(times, "Processing frames"):
        pixel2id, id2classname = find_pixel_id(time, parent_path, parent_path_video)
        x, y, z = project_pixel_to_3d(pixel2id, time, num, parent_path, parent_path_video)
        bounding_boxes = find_bounding_boxes(pixel2id, x, y, z, id2classname)
        all_bounding_boxes[num] = bounding_boxes
        num += 1

    print(all_bounding_boxes)
    return all_bounding_boxes


def find_closeness_with_character(all_bounding_boxes, times, parent_path, parent_path_video, character_id):
    all_closeness = {}
    num = 0
    print("281")
    for time in tqdm(times, "Analyzing closeness"):
        print("283\n")
        pixel2id, id2classname = find_pixel_id(time, parent_path, parent_path_video)
        print("284\n")
        print("character_id: ", character_id)
        # print(all_bounding_boxes)
        if isinstance(random.choice(list(all_bounding_boxes.keys())), (int, float, complex)):
            bounding_boxes = all_bounding_boxes[num]
        else:
            bounding_boxes = all_bounding_boxes[str(num)]
        print("287\n")
        # distances = calculate_distance_from_object(bounding_boxes, character_id)
        distances = calculate_distance_from_object_gpu(bounding_boxes, character_id)
        print("distance: {}\n".format(distances))
        print("289\n")
        if distances is None:
            close_to_items = []
        else:
            close_to_items = find_close_to(distances, id2classname)

            if close_to_items:
                if isinstance(random.choice(close_to_items), (int, float, complex)):
                    close_to_items = [id2classname[id] for id in close_to_items]
                else:
                    close_to_items = [id2classname[float(id)] for id in close_to_items]

        all_closeness[num] = list(set(close_to_items))
        num += 1
    print("295")

    return all_closeness


def find_GT_closeness(parent_path):
    all_closeness = {}
    task = load_pickles(parent_path, folder=True)
    time = 0
    while True:
        graph_file_name = f'graph_{time}.pik'
        if graph_file_name not in task:
            break
        graph = task[graph_file_name]
        
        closeness = []
        edges = graph['edges']
        id2node = {node["id"]: node for node in graph["nodes"]}
        id2name = {id:  node["class_name"] for id, node in id2node.items()}

        for edge in edges:
            from_id = edge['from_id']
            to_id = edge['to_id']

            edge['from_name'] = id2name.get(from_id)
            edge['to_name'] = id2name.get(to_id)

        for edge in edges:
            if edge['from_name'] == 'character' and edge['to_name'] in ALL_LIST:
                if edge['relation_type'] == 'CLOSE':
                    closeness.append(edge['to_name'])
        
        all_closeness[time] = list(set(closeness))
        time += 1

    return all_closeness


def f1_score(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    true_positives = len(set1.intersection(set2))
    false_positives = len(set2 - set1)
    false_negatives = len(set1 - set2)

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def find_GT_inside(parent_path):
    all_inside = {}
    task = load_pickles(parent_path, folder=True)
    time = 0
    while True:
        graph_file_name = f'graph_{time}.pik'
        if graph_file_name not in task:
            break
        graph = task[graph_file_name]
        
        inside = []
        edges = graph['edges']
        id2node = {node["id"]: node for node in graph["nodes"]}
        id2name = {id:  node["class_name"] for id, node in id2node.items()}

        for edge in edges:
            from_id = edge['from_id']
            to_id = edge['to_id']

            edge['from_name'] = id2name.get(from_id)
            edge['to_name'] = id2name.get(to_id)

        for edge in edges:
            if edge['relation_type'] == 'INSIDE':
                if edge['from_name'] in CONTAINER_LIST and edge["to_name"] in ROOM_LIST:
                    inside.append(f"{edge['from_name']} inside {edge['to_name']}")
                if edge['from_name'] in OBJECT_LIST and edge["to_name"] in CONTAINER_LIST:
                    inside.append(f"{edge['from_name']} in {edge['to_name']}")
            if edge['relation_type'] == 'ON':
                if edge['from_name'] in OBJECT_LIST and edge["to_name"] in SURFACE_LIST:
                    inside.append(f"{edge['from_name']} on {edge['to_name']}")
        
        all_inside[time] = inside
        time += 1

    return all_inside

def get_room(bounding_boxes):
    location_list = []
    for obj_name, bbox in bounding_boxes.items():
        if obj_name in CONTAINER_LIST or obj_name in SURFACE_LIST:
            location_list.append(obj_name)
    room_in = None
    for room, locations in ROOM_COMPONENTS.items():
        inside = True
        for location in locations:
            if location not in location_list:
                inside = False
        if inside:
            room_in = room
            break
    return room_in


def find_inside_and_open_and_on(times, parent_path, episode_bounding_boxes):
    all_inside = {}
    all_opened = {}
    all_on = {}
    num = 0
    for time in times:
        path = parent_path + 'graph_1.pik'
        with open(path, 'rb') as f:
            g = pickle.load(f)
        id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}

        if isinstance(random.choice(list(episode_bounding_boxes.keys())), (int, float, complex)):
            bounding_boxes = episode_bounding_boxes[num]
        else:
            bounding_boxes = episode_bounding_boxes[str(num)]

        bounding_boxes = {int(float(key)): value for key, value in bounding_boxes.items()}

        inside = []
        containers = [id for id in id2classname.keys() if id2classname[id] in CONTAINER_LIST]
        for id in containers:
            if not id in bounding_boxes.keys():
                continue
            
            container_box = bounding_boxes[id]
            if container_box["x_max"] - container_box["x_min"] > 2.5 or container_box["z_max"] - container_box["z_min"] > 2.5:
                continue
            for obj_id, box in bounding_boxes.items():
                if id2classname[obj_id] not in OBJECT_LIST:
                    continue
                if obj_id == id:
                    continue
                object_box = bounding_boxes[obj_id]
                if object_box["x_min"] >= container_box["x_min"] and object_box["x_max"] <= container_box["x_max"] and object_box["z_min"] >= container_box["z_min"] and object_box["z_max"] <= container_box["z_max"] and object_box["y_min"] >= container_box["y_min"] and object_box["y_max"] <= container_box["y_max"]:
                    inside.append(f'A {id2classname[obj_id]} is inside the {id2classname[id]}')
        
        on = []
        surfaces = [id for id in id2classname.keys() if id2classname[id] in SURFACE_LIST]
        for id in surfaces:
            if id2classname[id] == "kitchencounter":
                continue
            if not id in bounding_boxes.keys():
                continue
            surface_box = bounding_boxes[id]
            if surface_box["x_max"] - surface_box["x_min"] > 2.5 or surface_box["z_max"] - surface_box["z_min"] > 2.5:
                continue
            for obj_id, box in bounding_boxes.items():
                if id2classname[obj_id] not in OBJECT_LIST:
                    continue
                if obj_id == id:
                    continue
                object_box = bounding_boxes[obj_id]
                if object_box["x_min"] >= surface_box["x_min"] and object_box["x_max"] <= surface_box["x_max"] and object_box["z_min"] >= surface_box["z_min"] and object_box["z_max"] <= surface_box["z_max"]:
                    on.append(f'A {id2classname[obj_id]} is on the {id2classname[id]}')
        
        opened = []
        if num != 0:
            if isinstance(random.choice(list(episode_bounding_boxes.keys())), (int, float, complex)):
                previous_bounding_boxes = episode_bounding_boxes[num-1]
            else:
                previous_bounding_boxes = episode_bounding_boxes[str(num-1)]
            previous_bounding_boxes = {int(float(key)): value for key, value in previous_bounding_boxes.items()}
            containers = [id for id in id2classname.keys() if id2classname[id] in CONTAINER_LIST]
            for id in containers:
                if id in bounding_boxes and id in previous_bounding_boxes:
                    current = bounding_boxes[id]
                    previous = previous_bounding_boxes[id]
                    if id2classname[id] == "kitchencabinet":
                        if (current['x_max'] - current['x_min'] > previous['x_max'] - previous['x_min'] + 0.05) or \
                        (current['y_max'] - current['y_min'] > previous['y_max'] - previous['y_min'] + 0.05) or \
                        (current['z_max'] - current['z_min'] > previous['z_max'] - previous['z_min'] + 0.05):
                            opened.append(id2classname[id])
                    else:
                        if (current['x_max'] - current['x_min'] > previous['x_max'] - previous['x_min'] + 0.1) or \
                            (current['y_max'] - current['y_min'] > previous['y_max'] - previous['y_min'] + 0.1) or \
                            (current['z_max'] - current['z_min'] > previous['z_max'] - previous['z_min'] + 0.1):
                            opened.append(id2classname[id])

        all_inside[num] = inside
        all_on[num] = on
        all_opened[num] = list(set(opened))
        num += 1

    return all_inside, all_opened, all_on

# def find_hold(times, parent_path, episode_bounding_boxes):
#     count = 0
#     hold = {}
#     num = 0
#     for time in times:
#         hold[num] = []
#         path = parent_path + 'graph_1.pik'
#         with open(path, 'rb') as f:
#             g = pickle.load(f)
#         id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}
#         bounding_boxes = episode_bounding_boxes[str(num)]
#         if "1.0" not in bounding_boxes.keys():
#             num += 1
#             continue
#         character_box = bounding_boxes["1.0"]
#         for obj_id, box in bounding_boxes.items():
#             obj_id = int(float(obj_id))
#             if id2classname[obj_id] not in OBJECT_LIST:
#                 continue
#             object_box = bounding_boxes[str(float(obj_id))]
#             if object_box["y_min"] >= character_box["y_min"] and object_box["y_max"] <= character_box["y_max"]:
#                 if not object_box["x_min"] > character_box["x_max"] and not object_box["x_max"] < character_box["x_min"] and not object_box["z_min"] > character_box["z_max"] and not object_box["z_max"] < character_box["z_min"]:
#                     #if (object_box["x_min"] >= character_box["x_min"] and object_box["x_max"] <= character_box["x_max"]) or (object_box["z_min"] >= character_box["z_min"] and object_box["z_max"] <= character_box["z_max"]):
#                     hold[num].append(f'Male character is holding a {id2classname[obj_id]} ')
#         num += 1
        
#     return hold

def find_hold_by_character(times, parent_path, episode_bounding_boxes, character_id):
    hold = {}
    num = 0
    for time in tqdm(times, "Analyzing hold"):
        hold[num] = []
        path = parent_path + 'graph_1.pik'
        with open(path, 'rb') as f:
            g = pickle.load(f)
        
        id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}
        # bounding_boxes = episode_bounding_boxes[str(num)]
        print("num: ", num)
        if isinstance(random.choice(list(episode_bounding_boxes.keys())), (int, float, complex)):
            print("594")
            bounding_boxes = episode_bounding_boxes[num]
        else:
            print("597")
            # print(episode_bounding_boxes)
            bounding_boxes = episode_bounding_boxes[str(num)]
            # print(bounding_boxes)
            
        if (character_id not in bounding_boxes.keys()) and (str(character_id) not in bounding_boxes.keys()):
            num += 1
            continue

        if isinstance(random.choice(list(episode_bounding_boxes.keys())), (int, float, complex)):
            character_box = bounding_boxes[character_id]
        else:
            character_box = bounding_boxes[str(character_id)]

        if character_box["x_max"] - character_box["x_min"] > 10 or character_box["z_max"] - character_box["z_min"] > 10:
            num += 1
            continue

        hand_box = {
            "x_min": character_box["x_min"],
            "x_max": character_box["x_max"],
            "y_min": character_box["y_min"] + (character_box["y_max"] - character_box["y_min"]) * 0.5 - 0.2,  
            "y_max": character_box["y_min"] + (character_box["y_max"] - character_box["y_min"]) * 0.5 +0.2,
            "z_min": character_box["z_min"],
            "z_max": character_box["z_max"]
        }

        
        for obj_id, box in bounding_boxes.items():
            obj_id = int(float(obj_id))
            if obj_id == 1:
                continue
            if id2classname[obj_id] not in OBJECT_LIST:
                continue

            if isinstance(random.choice(list(bounding_boxes.keys())), (int, float, complex)):
                object_box = bounding_boxes[obj_id]
            else:
                object_box = bounding_boxes["{:.1f}".format(obj_id)]

            if is_holding(hand_box, object_box):
                hold[num].append(f'Character {int(character_id) - 1} is holding a {id2classname[obj_id]} ')
        num += 1
        
    return hold

def is_holding(hand_box, obj_box):

    overlap_x = max(0, min(hand_box['x_max'], obj_box['x_max']) - max(hand_box['x_min'], obj_box['x_min']))
    overlap_y = max(0, min(hand_box['y_max'], obj_box['y_max']) - max(hand_box['y_min'], obj_box['y_min']))
    overlap_z = max(0, min(hand_box['z_max'], obj_box['z_max']) - max(hand_box['z_min'], obj_box['z_min']))
    

    overlap_volume = overlap_x * overlap_y * overlap_z
    object_volume = (obj_box['x_max'] - obj_box['x_min']) * (obj_box['y_max'] - obj_box['y_min']) * (obj_box['z_max'] - obj_box['z_min'])
    

    return overlap_volume > 0.2 * object_volume # Charles: adjust this to filter out wrong object

if __name__ == "__main__":
    from pathlib import Path
    save_dir = "/home/scai/Workspace_2/hshi33/benchmark/visual_data/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_video_path", type=str, default='videos')
    parser.add_argument("--frame_analysis_interval", type=int, default=20)
    args = parser.parse_args()

    character_ids =[1.0,2.0]

    times = True
    BOX = False
    CLOSENESS = True
    INSIDE_OPEN = True
    all_closeness = {}
    all_boxes = {}
    all_opened = {}
    all_inside = {}
    all_on = {}
    all_hold = {}
    episodes = [5302, 5379, 5381, 5082, 5175, 5039, 4375, 4043, 4198, 4140, 4525, 4063, 4415, 4323, 4057, 4034, 4441, 4527, 4567, 4575, 4117, 4077, 5042, 4331, 4529, 4101, 4103, 4488, 4102, 4519, 4280, 4540, 4069, 4372, 5197, 4059, 4526, 4145, 4487, 4078, 5091, 4482, 4133, 4017, 4416, 4081, 4083, 5163, 4369, 5090, 4485, 4449, 4551, 4623, 4559, 4200, 4105, 5017, 4365, 5084, 5127, 4037, 4473, 4113, 4385, 4312, 4499, 4023, 5173, 4463, 4184, 4047, 4118, 4123, 4334, 4328, 4106, 4419, 4667, 4641, 5123, 4413, 4455, 4343, 4166, 4098, 5121, 4490, 4339, 4190, 5103, 4164, 4370, 4172, 4423, 5024, 4009, 5176, 4005, 4224, 4576, 5165, 4542, 4018, 4162, 4453, 4054, 4520, 4329, 5105, 4546, 5010, 4332, 4452, 4658, 5154, 4556, 4327, 4560, 4656, 4505, 4439, 4512, 5095, 4173, 5068, 4367, 4262, 4284, 4324, 5126, 5049, 4594, 4150, 4285, 5138, 5184, 4429, 4568, 4338, 4458, 5080, 4041, 4469, 4621, 4124, 4604, 5099, 5014, 5098, 4355, 4070, 4176, 4606, 4178, 4552, 4584, 4618, 4506, 4160, 5093, 4374, 5509, 4510, 5177]
    '''episodes = [5197]
    episodes = [766, 2709, 404, 2726, 640, 549, 609, 538, 3289, 135, 2053, 801, 857, 144, 3082, 138, 642, 3068, 153, 1827, 3355, 263, 647, 557, 784, 3308, 913, 130, 634, 3058, 644, 195, 895, 3092, 541, 865, 389, 128, 154, 628, 3117, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 3292, 578, 2559, 1802, 548, 225, 910, 2799, 42, 193, 2462, 3366, 1758, 161, 1818, 3300, 397, 630, 956, 3130, 3315, 612, 3098, 3074, 3166, 905, 3119, 179, 1131, 129, 1817, 871, 532, 583, 1819, 152, 655, 824, 1811, 3077, 577, 790, 1858, 848, 3050, 3060, 528, 682, 1765, 601, 1893]
    episodes = [5184]'''
    episodes = [4458]
    language_list = [4017, 4018, 4023, 4034, 4037, 4041, 4043, 4054, 4057, 4059, 4063, 4070, 4077, 4078, 4081, 4083, 4098, 4103, 4105, 4106, 4124, 4145, 4150, 4162, 4166, 4172, 4184, 4190, 4198, 4200, 4224, 4284, 4324, 4327, 4328, 4329, 4331, 4334, 4338, 4343, 4367, 4369, 4370, 4372, 4374, 4385, 4416, 4419, 4423, 4429, 4439, 4441, 4449, 4452, 4453, 4469, 4473, 4482, 4485, 4487, 4488, 4490, 4499, 4505, 4506, 4510, 4512, 4519, 4520, 4525, 4540, 4542, 4546, 4552, 4556, 4559, 4567, 4568, 4575, 4594, 4604, 4606, 4618, 4623, 4641, 4656, 4658, 5010, 5017, 5039, 5042, 5068, 5080, 5084, 5091, 5095, 5099, 5103, 5105, 5121, 5138, 5154, 5165, 5173, 5175, 5197, 5302, 5379, 5381, 5509, 4005, 4009, 4047, 4101, 4102, 4113, 4117, 4123, 4133, 4140, 4160, 4164, 4173, 4176, 4178, 4280, 4285, 4312, 4324, 4328, 4331, 4332, 4365, 4375, 4415, 4458, 4463, 4526, 4527, 4529, 4551, 4560, 4576, 4584, 4621, 4667, 5014, 5049, 5082, 5093, 5098, 5123, 5126, 5127, 5163, 4455]
    before_list = [42, 128, 130, 144, 154, 223, 393, 404, 528, 541, 548, 549, 557, 634, 790, 871, 905, 1758, 1818, 1858, 2053, 2070, 3074, 3092, 3098, 3129, 3130, 3292, 3308]
    after_list = [135, 138, 153, 193, 195, 389, 532, 538, 577, 628, 630, 642, 647, 766, 848, 865, 895, 910, 956, 1811, 1856, 2559, 3050, 3058, 3068, 3117, 3119, 3315, 129, 152, 161, 212, 225, 263, 397, 578, 583, 601, 609, 640, 644, 682, 784, 801, 824, 857, 913, 1131, 1817, 1819, 2462, 3077]
    episodes = []
    episodes += before_list
    episodes += after_list
    episodes += language_list
    for episode in tqdm(episodes):
        print(f"Episode {episode}")

        # try:
        parent_path = f"/home/scai/Workspace_2/hshi33/benchmark/raw_video_frames/{episode}/"
        #parent_path = f"/weka/scratch/tshu2/xfang21/benchmark/raw_video_frames/{episode}/"
        frame_num = 0
        temp = [f for f in os.listdir(parent_path) if not "pik" in f and not "json" in f and not "mp4" in f and not f == "0"]
        temp.sort()
        count = 0
        os.makedirs(os.path.join(parent_path, "0"), exist_ok=True)
        prev_count = 0
        for f in temp:
            frame_num += len([image for image in os.listdir(os.path.join(parent_path, f, "0")) if image.endswith("normal.png")])
            list_1 = [image for image in os.listdir(os.path.join(parent_path, f, "0")) if image.endswith("normal.png")]
            list_1.sort()
            for image in list_1:
                shutil.copy(os.path.join(parent_path, f, "0", image), os.path.join(parent_path, "0", f"Action_{count:04d}_0_normal.png"))
                count += 1
            count = prev_count
            list_2 = [image for image in os.listdir(os.path.join(parent_path, f, "0")) if image.endswith("depth.exr")]
            list_2.sort()
            for image in list_2:
                shutil.copy(os.path.join(parent_path, f, "0", image), os.path.join(parent_path, "0", f"Action_{count:04d}_0_depth.exr"))
                count += 1
            count = prev_count
            list_3= [image for image in os.listdir(os.path.join(parent_path, f, "0")) if image.endswith("seg_inst.png")]
            list_3.sort()
            for image in list_3:
                shutil.copy(os.path.join(parent_path, f, "0", image), os.path.join(parent_path, "0", f"Action_{count:04d}_0_seg_inst.png"))
                count += 1
            prev_count = count

        print("Total frame: ", frame_num)
        actual_frame_num = frame_num
        if episode < 4000 and episode in after_list:
            with open(parent_path + "timemark.json", "r") as file:
                frame_init = json.load(file)["time_change_frame"]
            with open(parent_path + "frame_intervals.pik", "rb") as file:
                intervals = pickle.load(file)
            for index, interval in enumerate(intervals):
                if interval[1] == frame_init:
                    frame = intervals[index+1][1]
                    break
            actual_frame_num = frame_num - frame
        elif episode in before_list:
            with open(parent_path + "timemark.json", "r") as file:
                frame_init = json.load(file)["time_change_frame"]
            actual_frame_num = frame_init
        else:
            acutal_frame_num = frame_num

        if actual_frame_num > 1500:
            print("Episode {} is too long".format(episode))
        frame_interval = args.frame_analysis_interval
        times = []
        if actual_frame_num / frame_interval > 30:
            frame_interval = int(actual_frame_num / 30)
        print("frame interval: ", frame_interval)
        print(f"{actual_frame_num} frames to analysis")
        if episode in language_list:
            for i in range(0, frame_num, frame_interval):
                times.append(i)
        elif episode in before_list:
            for i in range(0, actual_frame_num, frame_interval):
                times.append(i)
        else:
            for i in range(frame, frame_num, frame_interval):
                times.append(i)
        with open(os.path.join("/home/scai/Workspace_2/hshi33/benchmark/visual_data/", str(episode), "boxes.json"), "r") as file:
            data = json.load(file)
        if not len(data[str(episode)].keys()) == len(times):
            raise Exception
        os.makedirs(os.path.join(save_dir, str(episode)), exist_ok=True)

        # Charles: Don't delete this part. It's needed by video and multi-modal action extraction.
        if times: 
            file_path = os.path.join(save_dir, str(episode), "frames.json")
            data_to_save = {episode: times}
            with open(file_path, "w") as file:
                json.dump(data_to_save, file)

        if BOX:
            boxes = find_bounding_boxes_for_times(times, parent_path, f"{parent_path}/0/")
            all_boxes[episode] = boxes

            file_path = os.path.join(save_dir, str(episode), "boxes.json")
            with open(file_path, "w") as file:
                json.dump(all_boxes, file)


        if CLOSENESS:
            if not all_boxes:
                print(os.path.join(save_dir, "boxes.json"))

                with open(os.path.join("/home/scai/Workspace_2/hshi33/benchmark/visual_data/", str(episode), "boxes.json"), "r") as file:
                # with open(os.path.join("/home/scai/Workspace/hshi33/virtualhome/data/visual_data", str(episode), "boxes.json"), "r") as file:
                    boxes = json.load(file)
                episode_bounding_boxes = boxes[str(episode)]
            else:
                episode_bounding_boxes = boxes

            assert(episode_bounding_boxes)

            for character_id in character_ids:
                closeness = find_closeness_with_character(episode_bounding_boxes, times, parent_path, f"{parent_path}/0/", character_id)         
                all_closeness[episode] = closeness

                file_path = os.path.join(save_dir, str(episode), "closeness_{}.json".format(int(character_id -1)))
                with open(file_path, "w") as file:
                    json.dump(all_closeness, file)


        if INSIDE_OPEN:
            if not all_boxes:
                print(os.path.join(save_dir, "boxes.json"))
                with open(os.path.join("/home/scai/Workspace_2/hshi33/benchmark/visual_data/", str(episode), "boxes.json"), "r") as file:
                # with open(os.path.join("/home/scai/Workspace/hshi33/virtualhome/data/visual_data", str(episode), "boxes.json"), "r") as file:
                    boxes = json.load(file)
                episode_bounding_boxes = boxes[str(episode)]
            else:
                episode_bounding_boxes = boxes

            assert(episode_bounding_boxes)
            
            all_inside[episode], all_opened[episode], all_on[episode] = find_inside_and_open_and_on(times, parent_path, episode_bounding_boxes)      

            for character_id in character_ids: 
                all_hold[episode] = find_hold_by_character(times, parent_path, episode_bounding_boxes, character_id)

                file_path = os.path.join(save_dir, str(episode), "hold_{}.json".format(int(character_id -1)))
                with open(file_path, "w") as file:
                    json.dump(all_hold, file)

            os.makedirs(os.path.join(save_dir, str(episode)), exist_ok=True)

            file_path = os.path.join(save_dir, str(episode), "opened.json")
            with open(file_path, "w") as file:
                json.dump(all_opened, file)

            file_path = os.path.join(save_dir, str(episode), "inside.json")
            with open(file_path, "w") as file:
                json.dump(all_inside, file)
            
            file_path = os.path.join(save_dir, str(episode), "on.json")
            with open(file_path, "w") as file:
                json.dump(all_on, file)

        # except Exception as e:
        #     print(str(e))
        #     continue
    

    
    

    