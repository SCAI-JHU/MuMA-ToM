import numpy as np
import random
#may need to adjust system path here to insert virtualhome api path
from simulation.evolving_graph.utils import load_graph_dict, load_name_equivalence
from simulation.evolving_graph.environment import EnvironmentState, EnvironmentGraph, GraphNode
import scipy.special
import ipdb
import pdb
import sys
import simulation.evolving_graph.utils as vh_utils
import json
import copy
from termcolor import colored
from openai import OpenAI

client = OpenAI(
    api_key=''
)

class Language():
    def __init__(self, from_agent_id=None, to_agent_id=None, language_type=None):
        self.from_agent_id = from_agent_id
        self.to_agent_id = to_agent_id
        self.language_type = language_type


class LanguageInquiry(Language):
    def __init__(self, obj_list=None, from_agent_id=None, to_agent_id=None, language_type=None):
        super().__init__(from_agent_id, to_agent_id, language_type)
        self.obj_list = obj_list #name
    
    # Use the helper's belief and sampled graph to extract the most certain object information
    def generate_response(self, sampled_graph, edge_belief):
        assert(self.language_type == 'location')
        assert(self.obj_list is not None)
        hinder = random.random()
        obj_ids = {} #map from object name to ids belong to that class
        for obj in self.obj_list:
            obj_ids[obj] = [node["id"] for node in sampled_graph["nodes"] if node["class_name"] == obj]
        
        '''max_obj_id_prob = [None, 0.]
        pred = None
        position_id = None'''

        #ipdb.set_trace()
        id2class = {}
        room_ids = {}
        rooms = ["bathroom", "livingroom", "bedroom", "kitchen"]
        id2room = {}
        for node in sampled_graph["nodes"]:
            id2class[node["id"]] = node["class_name"]
        for node in sampled_graph["nodes"]:
            if node["class_name"] in rooms:
                room_ids[node["id"]] = node["class_name"]
                continue
            for edge in sampled_graph["edges"]:
                if edge["from_id"] == node["id"] and edge["relation_type"] == "INSIDE" and "room" in id2class[edge["to_id"]]:
                    id2room[node["id"]] = id2class[edge["to_id"]]
                    break
        '''print(id2class)
        print(id2room)'''
        print("hinder:", hinder)
        obj_position = {}
        for obj_name in obj_ids.keys():
            obj_position[obj_name] = {}
            for obj_id in obj_ids[obj_name]:
                if obj_id not in edge_belief:
                    continue
                else:
                    distribution = scipy.special.softmax(edge_belief[obj_id]["INSIDE"][1][:])
                    distribution = distribution[distribution > 0] #filter out all zero probabilities
                    if distribution.shape[0] == 1: #case when belief is certain
                        #ipdb.set_trace()
                        obj_position[obj_name][obj_id] = []
                        maxIndex = np.argmax(edge_belief[obj_id]["INSIDE"][1][:])
                        if hinder > 0.5: #not hindering
                            obj_position[obj_name][obj_id].append({"predicate": "inside", "position": edge_belief[obj_id]["INSIDE"][0][maxIndex]})
                            length = len(obj_position[obj_name][obj_id])
                            if edge_belief[obj_id]["INSIDE"][0][maxIndex] is not None:
                                obj_position[obj_name][obj_id][length-1]["class_name"] = id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]
                                for index, room in enumerate(list(room_ids.keys())):
                                    if index == len(room_ids) - 1:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        continue
                                    if edge_belief[obj_id]["INSIDE"][0][maxIndex] > room and edge_belief[obj_id]["INSIDE"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        break
                        else:
                            temp = []
                            class_name = ""
                            if not maxIndex == 0:
                                class_name = id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]
                            for i in range(1, edge_belief[obj_id]["INSIDE"][1].shape[0]):
                                if i == maxIndex or id2class[edge_belief[obj_id]["INSIDE"][0][i]] == class_name:
                                    continue
                                temp.append(edge_belief[obj_id]["INSIDE"][0][i])
                            position = random.choice(temp) #choose a random position
                            obj_position[obj_name][obj_id].append({"predicate": "inside", "position": position, "class_name":id2class[position]})
                            length = len(obj_position[obj_name][obj_id])
                            #if edge_belief[obj_id]["INSIDE"][0][index] is not None:
                            for index, room in enumerate(list(room_ids.keys())):
                                if index == len(room_ids) - 1:
                                    obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                    continue
                                if position and position < list(room_ids.keys())[index + 1]:
                                    obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                    break
                    else:
                        entropy = -np.sum(distribution * np.log2(distribution))
                        ratio = entropy / np.log2(distribution.shape[0])
                        if ratio <= 0.5: #certain enough, for testing purpose have 1.0 now
                            if obj_id not in obj_position[obj_name].keys():
                                obj_position[obj_name][obj_id] = []
                            maxIndex = np.argmax(edge_belief[obj_id]["INSIDE"][1][:])
                            if hinder > 0.5:
                                obj_position[obj_name][obj_id].append({"predicate": "inside", "position": edge_belief[obj_id]["INSIDE"][0][maxIndex]})
                                length = len(obj_position[obj_name][obj_id])
                                if edge_belief[obj_id]["INSIDE"][0][maxIndex] is not None:
                                    obj_position[obj_name][obj_id][length-1]["class_name"] = id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]
                                    for index, room in enumerate(list(room_ids.keys())):
                                        if index == len(room_ids) - 1:
                                            obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                            continue
                                        if edge_belief[obj_id]["INSIDE"][0][maxIndex] > room and edge_belief[obj_id]["INSIDE"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                            obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                            break
                            else:
                                temp = []
                                class_name = ""
                                if not maxIndex == 0:
                                    class_name = id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]
                                for i in range(1, edge_belief[obj_id]["INSIDE"][1].shape[0]):
                                    if i == maxIndex or id2class[edge_belief[obj_id]["INSIDE"][0][i]] == class_name:
                                        continue
                                    temp.append(edge_belief[obj_id]["INSIDE"][0][i])
                                position = random.choice(temp) #choose a random position
                                obj_position[obj_name][obj_id].append({"predicate": "inside", "position": position, "class_name":id2class[position]})
                                length = len(obj_position[obj_name][obj_id])
                                #if edge_belief[obj_id]["INSIDE"][0][index] is not None:
                                for index, room in enumerate(list(room_ids.keys())):
                                    if index == len(room_ids) - 1:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        continue
                                    if position > room and position < list(room_ids.keys())[index + 1]:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        break
                            '''for index, element in enumerate(distribution):
                                if element > 1 / distribution.shape[0]:
                                    obj_position[obj_name][obj_id].append({"predicate": "inside", "position": edge_belief[obj_id]["INSIDE"][0][index]})'''
                    distribution = scipy.special.softmax(edge_belief[obj_id]["ON"][1][:])
                    distribution = distribution[distribution > 0]
                    if distribution.shape[0] == 1:
                        maxIndex = np.argmax(edge_belief[obj_id]["ON"][1][:])
                        if hinder > 0.5:
                            obj_position[obj_name][obj_id].append({"predicate": "on", "position": edge_belief[obj_id]["ON"][0][maxIndex]})
                            length = len(obj_position[obj_name][obj_id])
                            if edge_belief[obj_id]["ON"][0][maxIndex] is not None:
                                obj_position[obj_name][obj_id][length-1]["class_name"] = id2class[edge_belief[obj_id]["ON"][0][maxIndex]]
                                for index, room in enumerate(list(room_ids.keys())):
                                    if index == len(room_ids) - 1:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        continue
                                    if edge_belief[obj_id]["ON"][0][maxIndex] > room and edge_belief[obj_id]["ON"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        break
                        else:
                            temp = []
                            for i in range(1, edge_belief[obj_id]["ON"][1].shape[0]):
                                if i == maxIndex:
                                    continue
                                temp.append(edge_belief[obj_id]["ON"][0][i])
                            position = random.choice(temp) #choose a random position
                            obj_position[obj_name][obj_id].append({"predicate": "on", "position": position, "class_name":id2class[position]})
                            length = len(obj_position[obj_name][obj_id])
                            #if edge_belief[obj_id]["INSIDE"][0][index] is not None:
                            for index, room in enumerate(list(room_ids.keys())):
                                if index == len(room_ids) - 1:
                                    obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                    continue
                                if position > room and position < list(room_ids.keys())[index + 1]:
                                    obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                    break
                        continue
                    entropy = -np.sum(distribution * np.log2(distribution))
                    ratio = entropy / np.log2(distribution.shape[0])
                    if ratio <= 0.5: #certain enough, for testing purpose have 1.0 now
                        if obj_id not in obj_position[obj_name].keys():
                            obj_position[obj_name][obj_id] = []
                        maxIndex = np.argmax(edge_belief[obj_id]["ON"][1][:])
                        if hinder > 0.5:
                            obj_position[obj_name][obj_id].append({"predicate": "on", "position": edge_belief[obj_id]["ON"][0][maxIndex]})
                            length = len(obj_position[obj_name][obj_id])
                            if edge_belief[obj_id]["ON"][0][maxIndex] is not None:
                                obj_position[obj_name][obj_id][length-1]["class_name"] = id2class[edge_belief[obj_id]["ON"][0][maxIndex]]
                                for index, room in enumerate(list(room_ids.keys())):
                                    if index == len(room_ids) - 1:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        continue
                                    if edge_belief[obj_id]["ON"][0][maxIndex] > room and edge_belief[obj_id]["ON"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        break
                        else:
                            temp = []
                            for i in range(1, edge_belief[obj_id]["ON"][1].shape[0]):
                                if i == maxIndex:
                                    continue
                                temp.append(edge_belief[obj_id]["ON"][0][i])
                            position = random.choice(temp) #choose a random position
                            obj_position[obj_name][obj_id].append({"predicate": "on", "position": position, "class_name":id2class[position]})
                            #if edge_belief[obj_id]["INSIDE"][0][index] is not None:
                            length = len(obj_position[obj_name][obj_id])
                            for index, room in enumerate(room_ids):
                                for index, room in enumerate(list(room_ids.keys())):
                                    if index == len(room_ids) - 1:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        continue
                                    if position > room and position < list(room_ids.keys())[index + 1]:
                                        obj_position[obj_name][obj_id][length-1]["room"] = room_ids[room]
                                        break
        for obj_name in obj_position.keys():
            if len(obj_position[obj_name].keys()) == 0: #indicating that agent is unsure about the object
                if hinder < 0.5:
                    for obj_id in obj_ids[obj_name]:
                        distribution = edge_belief[obj_id]["INSIDE"][0] #choose inside because it will hinder more effectively
                        maxIndex = random.choice(range(1, len(distribution)))
                        obj_position[obj_name][obj_id] = [{"predicate": "inside", "position": edge_belief[obj_id]["INSIDE"][0][maxIndex], "class_name": id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]}]
                        for index, room in enumerate(list(room_ids.keys())):
                            if index == len(room_ids) - 1:
                                obj_position[obj_name][obj_id][0]["room"] = room_ids[room]
                                continue
                            if edge_belief[obj_id]["INSIDE"][0][maxIndex] > room and edge_belief[obj_id]["INSIDE"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                obj_position[obj_name][obj_id][0]["room"] = room_ids[room]
                                break
                continue
            for obj_id in obj_position[obj_name].keys():
                if len(obj_position[obj_name][obj_id]) == 1 and obj_position[obj_name][obj_id][0]["position"] is None:
                    if hinder < 0.5:
                        distribution = edge_belief[obj_id]["INSIDE"][0] #choose inside because it will hinder more effectively
                        maxIndex = random.choice(range(1, len(distribution)))
                        obj_position[obj_name][obj_id] = [{"predicate": "inside", "position": edge_belief[obj_id]["INSIDE"][0][maxIndex], "class_name": id2class[edge_belief[obj_id]["INSIDE"][0][maxIndex]]}]
                        for index, room in enumerate(list(room_ids.keys())):
                            if index == len(room_ids) - 1:
                                obj_position[obj_name][obj_id][0]["room"] = room_ids[room]
                                continue
                            if edge_belief[obj_id]["INSIDE"][0][maxIndex] > room and edge_belief[obj_id]["INSIDE"][0][maxIndex] < list(room_ids.keys())[index + 1]:
                                obj_position[obj_name][obj_id][0]["room"] = room_ids[room]
                                break
        if hinder < 0.5:
            for obj_name, info in obj_position.items():
                for obj_id, places in info.items():
                    obj_position[obj_name][obj_id] = [places[0]]
        #ipdb.set_trace()

        return LanguageResponse(obj_position, from_agent_id=self.to_agent_id, to_agent_id=self.from_agent_id, language_type="location", try_to_hinder=(hinder < 0.5))

   
    def to_language(self, mode = "full"):
        if mode == "full":
            ans = ""
            for obj in self.obj_list:
                ans += "Where is {}?\n".format(obj)
            return ans
        instruction1 = """
                Objective: Generate natural language from a language template.
                User Input: Questions with a basic templated format in the form of "Where is X? Where is Y?"
                Instructions: Convert this question into natural conversational language. Make it seem like everyday conversation. If the user asks about multiple objects, combine the objects into a single question.
                """
        
        question = ""
        for obj in self.obj_list:
            question += "Where is {}? ".format(obj)

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": instruction1},
                {"role": "user", "content": "Where is wine?"},
                {"role": "assistant", "content": "Do you know where the wine is?"},
                {"role": "user", "content": "Where is cup?"},
                {"role": "assistant", "content": "Have you seen a cup anywhere?"},
                {"role": "user", "content":"Where is spoon? Where is plate?"},
                {"role": "assistant", "content": "Have you come across a spoon and a plate by any chance?"},
                {"role": "user", "content": question}
            ],
            model="gpt-4o",
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
        


class LanguageResponse(Language):
    def __init__(self, obj_positions=None, goal_spec=None, from_agent_id=None, to_agent_id=None, language_type=None, try_to_hinder=False):
        # obj_name is used for the natural language conversation. But actually we use obj_id implicitly to avoid ambiguity, we need the obj_id
        super().__init__(from_agent_id, to_agent_id, language_type)
        if language_type == 'location':
            assert(obj_positions is not None)
            self.obj_positions = obj_positions
            self.try_to_hinder = try_to_hinder
        elif language_type == 'goal':
            assert(goal_spec is not None)
            self.language = goal_spec.keys()[0] # xinyu: since we don't need multiple goals for an single agent,  I just choose the first one here
            self.goal_spec = goal_spec
            # raise NotImplementedError('Goal language not implemented yet') 
            
        else:
            raise ValueError('Language type not recognized')
        
    def parse(self):
        return self.language.split('_')
    
    def to_language(self, mode="full"):  # interface for converting to natural language
        if self.try_to_hinder and mode == "full":
            print("This language attempt to hinder")
        if self.language_type == "location":
            ans = ""
            if mode == "full":  # all the information of communication, only for testing stage
                for obj_name in self.obj_positions.keys():
                    if len(self.obj_positions[obj_name].keys()) == 0:
                        ans += 'I do not know position of object {}.\n'.format(obj_name)
                    else:
                        ans += "Location of {}:\n".format(obj_name)
                        for obj_id in self.obj_positions[obj_name].keys():
                            ans += "{} {} ".format(obj_name, obj_id)
                            for location in self.obj_positions[obj_name][obj_id]:
                                if location["position"] is None:
                                    #ans += "not {} anything".format(location["predicate"])
                                    continue
                                ans += " {} {}".format(location["predicate"], location["position"], location["class_name"], location["room"])
                            ans += "\n"
                return ans
            if mode == "natural": # natural communication
                info = ""
                for obj_name in self.obj_positions.keys():
                    if len(self.obj_positions[obj_name].keys()) == 0:
                        info += "{} null; ".format(obj_name)
                    else:
                        for obj_id in self.obj_positions[obj_name].keys():
                            for location in self.obj_positions[obj_name][obj_id]:
                                if location["position"] is None:
                                    info += "{} null; ".format(obj_name)
                                else:
                                    info += "{} {} {} {} {}; ".format(obj_name, location["predicate"], location["position"], location["class_name"], location["room"])
                
                instruction2 = """
                        Objective: Generate natural language from language template.
                        User Input: The locations of an object with a basic templated format, with entries seperated by ;. For instance, apple on table 121 livingroom; apple inside fridge 240 kitchen; apple null; banana on counter 101 kitchen; banana null means that there is an apple on the table in the livingroom, an apple inside the fridge in the kitchen, and the location of the third apple is unknown. There is a banana on the kitchen counter, and the location of the other banana is unknown.
                        Instructions:
                        Convert this statement into natural conversational language. If the multiple locations are provided for the same object, and some of them are null, ignore the null objects in the final description. In the above example, ignore apple null and banana null since the locations of the other apples and banana are known.
                        """

                example1 = "plate on coffeetable 133 livingroom; juice inside fridge 231 kitchen"
                response1 = "I saw the plate on the coffee table in the living room, and the juice is inside the fridge in the kitchen."
                example2 = "waterglass on table 120 livingroom"
                response2 = "I found the water glass sitting on the table in the living room."
                example3 = "apple on table 121 livingroom; apple inside fridge 240 kitchen; apple on kitchencounter 301 kitchen; apple null"
                response3 = "I noticed an apple on the table in the living room, another inside the fridge in the kitchen, and one more on the kitchen counter."
                example4 = "plate null"
                response4 = "Sorry, I don't know where there plate is."
                example5 = "laptop null; chair on floor 100 livingroom"
                response5 = "I don't know where the laptop is, but I noticed the chair on the floor in the living room."
                example6 = "wine null; wine null; wine null"
                response6 = "I'm not sure where the wine is."
                example7 = "fork on table 121 livingroom; fork null; plate inside microwave 230 kitchen"
                response7 = "I saw a fork on the table in the livingroom, and a plate inside the microwave in the kitchen."

                response2 = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction2},
                        {"role": "user", "content": example1},
                        {"role": "assistant", "content": response1},
                        {"role": "user", "content": example2},
                        {"role": "assistant", "content": response2},
                        {"role": "user", "content": example2},
                        {"role": "assistant", "content": response2},
                        {"role": "user", "content": example3},
                        {"role": "assistant", "content": response3},
                        {"role": "user", "content": example4},
                        {"role": "assistant", "content": response4},
                        {"role": "user", "content": example5},
                        {"role": "assistant", "content": response5},
                        {"role": "user", "content": example6},
                        {"role": "assistant", "content": response6},
                        {"role": "user", "content": example7},
                        {"role": "assistant", "content": response7},
                        {"role": "user", "content": info}
                    ],
                    model="gpt-4o",
                    temperature=0.3
                )

                return response2.choices[0].message.content.strip()