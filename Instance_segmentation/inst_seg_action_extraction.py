import json
import os
import base64
from openai import OpenAI
from tqdm import tqdm
import requests
import argparse
from collections import Counter
import sys
import random

from env_utils import ALL_LIST, ROOM_LIST, CONTAINER_LIST, SURFACE_LIST, OBJECT_LIST, ROOM_COMPONENTS, ROOM_POSSIBILITY

base_dir = "path for visual data"
parent_path = "path for video frames"
api_key = ''  # Enter your API Key when you want to use this script

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def read_and_encode_frames(episode):
    json_file = 'frames.json'
    file_path = os.path.join(base_dir, str(episode), json_file)

    with open(file_path, 'r') as json_file:
        episode_frames_dic = json.load(json_file)

    print(episode_frames_dic)
    assert(len(episode_frames_dic) == 1)
    
    for key, value in episode_frames_dic.items():
        episode = key
        frames_idx = value
    
    frames_path = os.path.join(parent_path, episode, "0")

    encoded_images = {
        "normal": [],
        "depth": [],
        "seg_inst": []
    }

    normal_images = [image for image in os.listdir(frames_path) if image.endswith("normal.png")]


    normal_images.sort()


    for idx in frames_idx:
        if idx < len(normal_images):
            encoded_image = encode_image(os.path.join(frames_path, normal_images[idx]))
            encoded_images["normal"].append(encoded_image)


    return encoded_images["normal"]

import ipdb
# by character logic
def read_json_files_by_character(episode, have_image=False, character_id=0):
    with open(os.path.join(base_dir, str(episode), "closeness_{}.json".format(character_id)), "r") as file:
        close_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "hold_{}.json".format(character_id)), "r") as file:
        hold_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "inside.json"), "r") as file:
        inside_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "on.json"), "r") as file:
        on_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "opened.json"), "r") as file:
        open_data = json.load(file)

    frame_num = len(hold_data[str(episode)].keys())
    state_list = []
    print(f"Frame number: {frame_num}")

    for i in range(frame_num):
        state = ""
        repeat = True  # Check if this scene is largely the same as the previous one

        if i == 0:
            repeat = False
        if repeat and not len(inside_data[str(episode)][str(i)]) == len(inside_data[str(episode)][str(i - 1)]):
            repeat = False
        if repeat and not len(open_data[str(episode)][str(i)]) == len(open_data[str(episode)][str(i - 1)]):
            repeat = False
        if repeat and not len(hold_data[str(episode)][str(i)]) == len(hold_data[str(episode)][str(i - 1)]):
            repeat = False
        if repeat:
            list1 = on_data[str(episode)][str(i)]
            list2 = on_data[str(episode)][str(i-1)]
            counter1 = Counter(list1)
            counter2 = Counter(list2)
            all_elements = set(counter1.keys()).union(set(counter2.keys()))

            diff_count = 0
            for element in all_elements:
                diff_count += abs(counter1[element] - counter2[element])
            if diff_count > 0:
                repeat = False
        if repeat:
            #print(f"Frame {i} is skipped.")
            continue

        encoded_str = ""
        for obj in close_data[str(episode)][str(i)]:
            if not obj in SURFACE_LIST and not obj in CONTAINER_LIST:
                continue
            encoded_str += "The character is close to {}. ".format(obj)
        for obj in open_data[str(episode)][str(i)]:
            count = 0
            for j in range(i):
                if obj in open_data[str(episode)][str(j)]:
                    count += 1
            encoded_str += "A {} is opened. ".format(obj)
        for statement in on_data[str(episode)][str(i)]:
            encoded_str += statement
            if not statement == "":
                encoded_str += ". "
        for statement in inside_data[str(episode)][str(i)]:
            encoded_str += statement
            if not statement == "":
                encoded_str += ". "
        for statement in hold_data[str(episode)][str(i)]:
            encoded_str += statement
            if not statement == "":
                encoded_str += ". "

        state_list.append((i, encoded_str))

    data = ""
    for count, (frame_index, state) in enumerate(state_list, start=0):
        if have_image:
            data += "State {} (Frame {}):\n".format(count, frame_index)
        else:
            data += "State {}:\n".format(count)
        data += state
        if count != len(state_list):
            data += "\n"

    return data

def read_json_files(episode, have_image=False, character_ids = [0, 1]): # used for read both 2 characters

    close_data_list = []
    hold_data_list = []

    for character_id in character_ids:
        with open(os.path.join(base_dir, str(episode), "closeness_{}.json".format(character_id)), "r") as file:
            close_data_list.append(json.load(file))
        with open(os.path.join(base_dir, str(episode), "hold_{}.json".format(character_id)), "r") as file:
            hold_data_list.append(json.load(file))

    with open(os.path.join(base_dir, str(episode), "inside.json"), "r") as file:
        inside_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "on.json"), "r") as file:
        on_data = json.load(file)
    with open(os.path.join(base_dir, str(episode), "opened.json"), "r") as file:
        open_data = json.load(file)

    frame_num = len(on_data[str(episode)].keys())
    state_list = []
    print(f"Frame number: {frame_num}")

    for i in range(frame_num):
        state = ""
        repeat = True  # Check if this scene is largely the same as the previous one

        if i == 0:
            repeat = False
        if repeat and not len(inside_data[str(episode)][str(i)]) == len(inside_data[str(episode)][str(i - 1)]):
            repeat = False
        if repeat and not len(open_data[str(episode)][str(i)]) == len(open_data[str(episode)][str(i - 1)]):
            repeat = False
        for j in range(len(character_ids)):
            if repeat and  len(hold_data_list[j][str(episode)][str(i)]) != len(hold_data_list[j][str(episode)][str(i - 1)]):
                repeat = False
        if repeat:
            list1 = on_data[str(episode)][str(i)]
            list2 = on_data[str(episode)][str(i-1)]
            counter1 = Counter(list1)
            counter2 = Counter(list2)
            all_elements = set(counter1.keys()).union(set(counter2.keys()))

            diff_count = 0
            for element in all_elements:
                diff_count += abs(counter1[element] - counter2[element])
            if diff_count > 0:
                repeat = False
        if repeat:
            #print(f"Frame {i} is skipped.")
            continue

        encoded_str = ""
        for j in range(len(character_ids)):
            for obj in close_data_list[j][str(episode)][str(i)]:
                if not obj in SURFACE_LIST and not obj in CONTAINER_LIST:
                    continue
                encoded_str += "The character {} is close to {}. ".format(j, obj)
        for obj in open_data[str(episode)][str(i)]:
            count = 0
            for j in range(i):
                if obj in open_data[str(episode)][str(j)]:
                    count += 1
            encoded_str += "A {} is opened. ".format(obj)
        for statement in on_data[str(episode)][str(i)]:
            encoded_str += statement
            if not statement == "":
                encoded_str += ". "
        for statement in inside_data[str(episode)][str(i)]:
            encoded_str += statement
            if not statement == "":
                encoded_str += ". "
        for j in range(len(character_ids)):
            for statement in hold_data_list[j][str(episode)][str(i)]:
                encoded_str += statement
                if not statement == "":
                    encoded_str += ". "

        state_list.append((i, encoded_str))

    data = ""
    for count, (frame_index, state) in enumerate(state_list, start=0):
        if have_image:
            data += "State {} (Frame {}):\n".format(count, frame_index)
        else:
            data += "State {}:\n".format(count)
        data += state
        if count != len(state_list):
            data += "\n"

    return data

def predict_action_by_character(data=None, encoded_images=None, episode=0, utterance=None, character_id=0):
    extra_info = ""
    if episode > 4000:
        if utterance is not None:
            names = list(utterance.keys())
            extra_info += utterance[names[1]][0]
    else:
        '''with open("/home/scai/Workspace_2/hshi33/benchmark/texts/multimodal/episode_{}.txt".format(episode), "r") as file:
            extra_info += file.read()'''
    # base64_images = [encode_image(frame) for frame in frames]
    if data and encoded_images:
        prompt = """
            Task: You will read a series of states describing the state of a household setting in chronological order. These states are extracted from a video. The changes in these states are due to actions taken an agent and potentially a partner. Infer the actions taken by the agent.
            Possible Actions:
            Walk towards a certain location
            Grab a certain object from somewhere
            Open a certain container
            Close a certain container
            Put an object somewhere
            Requirement for actions:
            Grabbing: Agent must be close to the container/surface. 
            Opening/Closing: Agent must be close to the container.
            Putting: Agent must be close to the location where the object is placed.
            State Inferences:
            If a container is opened and no objects are shown inside, it is empty.
            If the agent is holding something in one state and not in the next, the object was put somewhere.
            If the agent is not holding something in one state and is holding it in the next, the object was grabbed from somewhere the agent is close to. More likely, the agent grab it from last container opened (as long as it is still close to the container)
            An opening action is usually followed by a closing action.
            Numtiple agents: When multiple agents are present in the video, you must confirm which agent is character {} and only describe the actions of character {}.
            Steps Between States: There might be one or more steps between two states. Summarize all actions of of character {} logically and in chronological order.
            Tracing Actions:
            Walking: Trace by the objects the agent is close to.
            Putting: Trace by the objects the agent is holding. For the state before the putting action agent must hold the object, for the state after the object must be insde/on a container/surface. If this conditions are not satisfied, putting action is never possible
            Grabbing: Trace by objects agents are holding. When determining the place of the grab action, trace location agent is close to in current and previous 1 state. If the place grabbing from is not clear in the states, infer a reasonable location. 
            Clues: There might be some information that can be used to assist you in infering the actions. Focus on actions related with objects directly mentioned in the information
            Attention:
            After summarizing all the actions of character {}, if there seems to be no grabbing object in the sequence, infer a object grabbed and a reasonable place for grabbing. This is the last step of the action sequence. 
            Additionally, encoded images of the frames from original video are provided to assist in the inference. 
            
            Format. In the end, formulate the inferred actions in the following form:
            Actions of character {}:
            ["Action one", "Action two:, ...]
            Do the same for further steps. Only include the extracted actions. Do not give any extra information or explanation. Make sure outputs follow the format requirements.

            States: {}

            Extra_information: {}
        """
        print("Data:", data) 

        messages = [
            {"role": "user", "content": prompt.format(character_id, character_id, character_id, character_id, character_id, data, extra_info)},
        ]

    # state
    elif data is not None and encoded_images is None:
        prompt = """
            Task: You will read a series of states describing the state of a household setting in chronological order. The changes in these states are due to actions taken an agent and potentially a partner. Infer the actions taken by the agent.
            Possible Actions:
            Walk towards a certain location
            Grab a certain object from somewhere
            Open a certain container
            Close a certain container
            Put an object somewhere
            Requirement for actions:
            Grabbing: Agent must be close to the container/surface. 
            Opening/Closing: Agent must be close to the container.
            Putting: Agent must be close to the location where the object is placed.
            State Inferences:
            If a container is opened and no objects are shown inside, it is empty.
            If the agent is holding something in one state and not in the next, the object was put somewhere.
            If the agent is not holding something in one state and is holding it in the next, the object was grabbed from somewhere the agent is close to. More likely, the agent grab it from last container opened (as long as it is still close to the container)
            An opening action is usually followed by a closing action.
            Steps Between States: There might be one or more steps between two states. Summarize all actions logically and in chronological order.
            Tracing Actions:
            Walking: Trace by the objects the agent is close to.
            Putting: Trace by the objects the agent is holding. For the state before the putting action agent must hold the object, for the state after the object must be insde/on a container/surface. If this conditions are not satisfied, putting action is never possible
            Grabbing: Trace by objects agents are holding. When determining the place of the grab action, trace location agent is close to in current and previous 1 state. If the place grabbing from is not clear in the states, infer a reasonable location. 
            Clues: There might be some information that can be used to assist you in infering the actions. Focus on actions related with objects directly mentioned in the information
            Attention:
            After summarizing all the actions, if there seems to be no grabbing object in the sequence, infer a object grabbed and a reasonable place for grabbing. This is the last step of the action sequence. 
            Format. In the end, formulate the inferred actions in the following form:
            Actions:
            ["Action one", "Action two:, ...]
            Do the same for further steps. Only include the extracted actions. Do not give any extra information or explanation. Make sure outputs follow the format requirements.

            States: {}

            Extra_information: {}
        """

        messages = [
            {"role": "user", "content": prompt.format(data, extra_info)},
        ]

        #print(data)


    # video (not working now)
    elif data is None and encoded_images is not None:
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps extract character actions based on provided frames."},
            {"role": "user", "content": ("1. Using the provided frames, extract and describe the action sequence of the female character in chronological order. "
                                        "2. Each frame is sampled every 20 frames from the video. "
                                        "3. Each frame may correspond to part of an action sequence. "
                                        "4. Based on the sequence of frames, infer actions such as 'walking towards'. "
                                        "5. If an object moves between frames, infer that it was moved by a character. "
                                        "6. Avoid stating the proximity; instead, describe the inferred actions of the female character. "
                                        "7. Before describing each action, consider whether it logically follows from the previous actions, as the character's behavior should be logical. "
                                        "8. Use 'hold.json' to determine objects that characters are holding and incorporate this into the action descriptions. "
                                        "8. Encoded images of the frames are provided as source in the inference. "
                                        "9. Note that there may be both a male and a female character in the video, but only describe the actions of the female character."
                                        "10. After the detailed description, provide a step-by-step summary of the inferred action sequence, excluding steps where the female character is not visible.")},
        ]

        # cheating code?
        messages.append({"role": "user", "content": "11. **Besides walking, if the character performs any action in the frames, for example, bending down, it is very likely to be picking up or putting down objects. Specify what the objects are based on the following instruction!** "
                                            "**Correlate this with previous and subsequent frames to ensure the objects being picked up or put down are identified.**"})
        messages.append({"role": "user", "content": "12. **Pay close attention to what the character is holding in their hands.**"})

        # Final instruction
        messages.append({"role": "user", "content": "13. After the detailed description, provide a step-by-step summary of the inferred action sequence, excluding steps where the female character is not visible. "
                                            "**Refer to the frames to determine the object being picked up or placed.**"})

    
    else:
         raise ValueError("Invalid input")   


    if encoded_images:
        for i, base64_image in enumerate(encoded_images):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Frame {i+1}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        #print(response.json().get('choices')[0].get('message').get('content'))
        return response.json().get('choices')[0].get('message').get('content')
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

def get_action_by_character(episode, utterance=None, character_id=0):
    import ast
    import re
    data = read_json_files_by_character(episode, character_id=character_id)
    action_prediction = predict_action_by_character(data=data, episode=episode, utterance=utterance, character_id=character_id)
    print(action_prediction)
    actions_match = re.search(r'Actions:\s*(\[[^\]]*\])', action_prediction)
    actions = actions_match.group(1) if actions_match else None
    action_prediction = ast.literal_eval(actions)
    return action_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-e', '--episode',
            type=int,
            default=0,
            help='Episode number.',

        )

    parser.add_argument(
        '-m', '--modal',
        type=str,
        choices=['state', 'multi-modal', 'video'],
        required=True,
        help="Choose the mode: 'state' for using extracted info only or 'multi-modal' for combining raw frames, or 'video' for raw frames only"
    )

    parser.add_argument(
        '-i', '--character_id',
        type=int,
        choices=[0, 1],
        required=True,
    )

    args = parser.parse_args()

    character_ids = [0, 1]

    if args.episode == 0:
        episodes_list = [4070, 4150, 4324, 4469, 4510, 4529, 4576, 4667, 5080, 5126]
    else:
        episodes_list = [args.episode]
    for episode in episodes_list:
        print("Episode {}".format(episode))
        if args.modal == "multi-modal":
            data = read_json_files(episode, True, character_ids)
            frames = read_and_encode_frames(episode)
            action_prediction = predict_action_by_character(data, frames, args.episode, character_id=args.character_id)
        elif args.modal == "state":
            data = read_json_files_by_character(episode, character_id=args.character_id)
            action_prediction = predict_action_by_character(data=data, episode=episode, character_id=args.character_id)
        elif args.modal == "video": # This one is temporarily disabled
            frames = read_and_encode_frames(args.episode, True)
            action_prediction = predict_action_by_character(encoded_images=frames, episode=args.episode)
        
        print("Predicted: ", action_prediction)
