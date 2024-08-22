import numpy as np
import json
import copy
import ipdb
import itertools
import os
import sys


curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("") #your virtualhome simulator API path

from termcolor import colored
from evolving_graph.utils import (
    load_graph_dict,
    load_name_equivalence,
    graph_dict_helper,
)
from evolving_graph.execution import ScriptExecutor, ExecutionInfo
from evolving_graph.scripts import read_script_from_string

from evolving_graph.environment import EnvironmentGraph, Relation
from evolving_graph.environment import EnvironmentState as EnvironmentStateBase


def init_from_state(env_state: EnvironmentStateBase, touched_objs, offer_objs):
    env_state_new = EnvironmentState(
        env_state._graph,
        env_state._name_equivalence,
        env_state.instance_selection,
        touched_objs,
        offer_objs,
    )
    env_state_new.executor_data = env_state.executor_data
    env_state_new._script_objects = env_state._script_objects
    env_state_new._new_nodes = env_state._new_nodes
    env_state_new._removed_edges_from = env_state._removed_edges_from
    env_state_new._new_edges_from = env_state._new_edges_from
    return env_state_new


class EnvironmentState(EnvironmentStateBase):
    def __init__(
        self,
        graph: EnvironmentGraph,
        name_equivalence,
        instance_selection: bool = False,
        touched_objs=[],
        offer_obj=[],
    ):
        self.touched_objs = touched_objs.copy()
        self.offer_obj = offer_obj.copy()

        super(EnvironmentState, self).__init__(
            graph, name_equivalence, instance_selection
        )

    def to_dict(self):
        edges = []
        from_pairs = self._new_edges_from.keys() | self._graph.get_from_pairs()
        for from_n, r in from_pairs:
            for to_n in self.get_node_ids_from(from_n, r):
                edges.append(
                    {"from_id": from_n, "relation_type": r.name, "to_id": to_n}
                )
        nodes = []
        for node in self.get_nodes():
            dict_node = node.to_dict()
            if dict_node["id"] in self.touched_objs:
                dict_node["states"].append("touched")
            nodes.append(dict_node)
        return {"nodes": nodes, "edges": edges}

    def remove_obj_offer(self, obj_id):
        if obj_id in self.offer_obj:
            self.offer_obj.remove(obj_id)

    def offer_object(self, obj_id):
        self.offer_obj.append(obj_id)

    def touch_object(self, obj_id):
        self.touched_objs.append(obj_id)


class VhGraphEnv:

    metadata = {"render.modes": ["human"]}
    action_executors = []
    actions = [
        "Walk",  # Same as Run
        # "Find",
        "Sit",
        "StandUp",
        "Grab",
        "Open",
        "Close",
        "PutBack",
        "PutIn",
        "SwitchOn",
        "SwitchOff",
        # "Drink",
        "LookAt",
        "TurnTo",
        # "Wipe",
        # "Run",
        "PutOn",
        "PutOff",
        # "Greet",
        "Drop",  # Same as Release
        # "Read",
        "PointAt",
        "Touch",
        "Lie",
        "PutObjBack",
        "Pour",
        # "Type",
        # "Watch",
        "Push",
        "Pull",
        "Move",
        # "Rinse",
        # "Wash",
        # "Scrub",
        # "Squeeze",
        "PlugIn",
        "PlugOut",
        "Cut",
        # "Eat",
        "Sleep",
        "WakeUp",
        # "Release"
    ]
    map_properties_to_pred = {
        "ON": ("on", True),
        "OPEN": ("open", True),
        "OFF": ("on", False),
        "CLOSED": ("open", False),
    }
    map_edges_to_pred = {
        "INSIDE": "inside",
        "CLOSE": "close",
        "ON": "ontop",
        "FACING": "facing",
    }
    house_obj = ["floor", "wall", "ceiling"]

    def __init__(self, n_chars=1, max_nodes=200):
        self.graph_helper = None  # graph_dict_helper()
        self.n_chars = n_chars
        self.name_equivalence = None

        self.state = None
        self.observable_state_n = [None for i in range(self.n_chars)]
        self.character_n = [None for i in range(self.n_chars)]
        self.tasks_n = [None for i in range(self.n_chars)]
        self.prev_progress_n = [None for i in range(self.n_chars)]
        self.rooms = None
        self.rooms_ids = None
        self.observable_object_ids_n = [None for i in range(self.n_chars)]
        self.pomdp = False
        self.executor_n = [
            ScriptExecutor(EnvironmentGraph(self.state), self.name_equivalence, i)
            for i in range(self.n_chars)
        ]
        self.room_doors = None

    def intermediate_doors(self, door_dict, rooms, doors):
        # Floyd algo for pairs of distances between rooms
        for room1 in rooms:
            for room2 in rooms:
                for room3 in rooms:
                    if room1 != room2 and room2 != room3:
                        d1d3, d1d2, d2d3 = doors, doors, doors
                        if (room1, room3) in door_dict:
                            d1d3 = door_dict[(room1, room3)]
                        if (room1, room2) in door_dict:
                            d1d2 = door_dict[(room1, room2)]
                        if (room2, room3) in door_dict:
                            d2d3 = door_dict[(room2, room3)]
                        if len(d1d3 + d2d3) < len(d1d2):
                            door_dict[(room1, room2)] = d1d3 + d2d3
                            door_dict[(room2, room1)] = d2d3 + d1d3[::-1]

        return door_dict

    def build_room_doors(self):
        door_dict = {}

        doors = sorted(
            [node["id"] for node in self.state["nodes"] if node["category"] == "Doors"]
        )
        rooms = sorted(
            [node["id"] for node in self.state["nodes"] if node["category"] == "Rooms"]
        )
        if doors == [52, 53, 171, 227, 363, 364]:
            door_dict[(11, 74)] = 52
            door_dict[(74, 11)] = 52
            door_dict[(11, 336)] = 363
            door_dict[(336, 11)] = 363  # DC
            door_dict[(74, 207)] = 227
            door_dict[(207, 74)] = 227

        elif doors == [46, 47, 69, 70, 212, 329, 330]:
            door_dict[(11, 50)] = 47
            door_dict[(50, 11)] = 47
            door_dict[(185, 50)] = 69
            door_dict[(50, 185)] = 69
            door_dict[(262, 50)] = 70
            door_dict[(50, 262)] = 70

        elif doors == [46, 47, 48, 204, 254, 304, 305, 374]:
            door_dict[(11, 346)] = 47
            door_dict[(346, 11)] = 47
            door_dict[(184, 241)] = 254
            door_dict[(241, 184)] = 254
            door_dict[(346, 285)] = 305
            door_dict[(285, 346)] = 305

        elif doors == [41, 249, 364, 365]:
            door_dict[(11, 172)] = 41
            door_dict[(172, 11)] = 41
            door_dict[(210, 267)] = 365
            door_dict[(267, 210)] = 365
            door_dict[(11, 210)] = 249  # CHECK
            door_dict[(210, 11)] = 249  #

        elif doors == [24, 128, 129, 240, 241, 287, 288]:
            door_dict[(11, 109)] = 128
            door_dict[(109, 11)] = 128
            door_dict[(109, 212)] = 241
            door_dict[(212, 109)] = 241
            door_dict[(11, 274)] = 288
            door_dict[(274, 11)] = 288

        elif doors == [48, 49, 112, 189, 190, 286, 287]:
            door_dict[(11, 73)] = 48
            door_dict[(73, 11)] = 48
            door_dict[(73, 170)] = 189
            door_dict[(170, 73)] = 189
            door_dict[(259, 73)] = 286  #
            door_dict[(73, 259)] = 286  #

        elif doors == [52, 209, 295, 297]:
            door_dict[(11, 56)] = 52
            door_dict[(56, 11)] = 52
            door_dict[(56, 198)] = 209
            door_dict[(198, 56)] = 209
            door_dict[(56, 294)] = 297
            door_dict[(294, 56)] = 297
        else:
            print(doors)
            ipdb.set_trace()
            raise Exception
        for doork, doorv in door_dict.items():
            door_dict[doork] = [doorv]
        door_dict = self.intermediate_doors(door_dict, rooms, doors)
        # ipdb.set_trace()
        return door_dict

    def compute_distance(self, vh_state, action, char_id, use_doors=False):
        def distance(bbox1, bbox2):
            b1 = [bbox1[0], bbox1[2]]
            b2 = [bbox2[0], bbox2[2]]
            return np.linalg.norm(np.array(b1) - np.array(b2))

        if use_doors and self.room_doors is None:
            self.room_doors = self.build_room_doors()

        script = read_script_from_string(action)

        assert len(script) == 1
        scriptline = script[0]
        object_script = scriptline.object()

        if scriptline.action.name != "WALK":
            return 0
        if object_script is None:
            return 0
        else:
            total_dist = 0.0
            # The current room of the character
            room_char = list(vh_state.get_node_ids_from(char_id, Relation.INSIDE))[0]
            close_char = list(vh_state.get_node_ids_from(char_id, Relation.CLOSE))
            close_char = [
                index
                for index in close_char
                if "GRABBABLE" not in self.id2node[index]["properties"]
            ]
            if len(close_char) == 0:
                pos_char = self.id2node[room_char]["bounding_box"]["center"]
            else:
                pos_char = self.id2node[close_char[0]]["bounding_box"]["center"]
            curr_node = vh_state.get_node(object_script.instance)

            # ipdb.set_trace()
            # if walking to a room
            is_obj = False
            if curr_node.category == "Rooms":
                curr_room = object_script.instance
                if not use_doors:
                    # If we dont consider doors, the final pos is the position of the room
                    final_pos = self.id2node[object_script.instance]["bounding_box"][
                        "center"
                    ]
            else:
                inside_obj = []

                curr_inside_obj = list(
                    vh_state.get_node_ids_from(object_script.instance, Relation.INSIDE)
                )
                inside_obj += curr_inside_obj
                while len(curr_inside_obj) > 0:
                    curr_obj = curr_inside_obj[0]
                    curr_inside_obj = list(
                        vh_state.get_node_ids_from(curr_obj, Relation.INSIDE)
                    )
                    inside_obj += curr_inside_obj
                inside_obj = list(set(inside_obj))

                try:
                    curr_room = [
                        indexr for indexr in inside_obj if indexr in self.rooms_ids
                    ][0]
                except:
                    ipdb.set_trace()
                if (
                    "GRABBABLE"
                    not in self.id2node[object_script.instance]["properties"]
                ):
                    final_pos = self.id2node[object_script.instance]["bounding_box"][
                        "center"
                    ]
                else:
                    final_pos = list(
                        vh_state.get_node_ids_from(
                            object_script.instance, Relation.INSIDE
                        )
                    )
                    final_pos += list(
                        vh_state.get_node_ids_from(object_script.instance, Relation.ON)
                    )
                    final_pos = [
                        index
                        for index in final_pos
                        if "GRABBABLE" not in self.id2node[index]["properties"]
                        and final_pos not in self.rooms_ids
                    ]
                    if len(close_char) > 0:
                        final_pos = self.id2node[final_pos[0]]["bounding_box"]["center"]
                    else:
                        final_pos = self.id2node[curr_room]["bounding_box"]["center"]

                is_obj = True

            if not use_doors or room_char == curr_room:
                doors = []
            else:
                try:
                    doors = self.room_doors[(room_char, curr_room)]
                except:
                    ipdb.set_trace()

            if len(doors) > 0:
                doors_pos = [
                    self.id2node[did]["bounding_box"]["center"] for did in doors
                ]
                total_dist = distance(pos_char, doors_pos[0])

                for did in range(1, len(doors)):
                    total_dist += distance(doors_pos[did], doors_pos[did - 1])

                if is_obj:
                    # add distance from last door to new object
                    if len(doors) > 0:
                        total_dist += distance(doors_pos[-1], final_pos)
            else:
                if not use_doors:
                    total_dist += distance(pos_char, final_pos)
                else:
                    if not is_obj:
                        #     ipdb.set_trace()
                        total_dist += 0  # TODO: should avoid this
                    else:
                        total_dist += distance(pos_char, final_pos)
        return total_dist

    def to_pomdp(self):
        self.pomdp = True
        for i in range(self.n_chars):
            if self.observable_object_ids_n[i] is None:
                self.observable_state_n[i] = self._mask_state(self.state, i)
                self.observable_object_ids_n[i] = [
                    node["id"] for node in self.observable_state_n[i]["nodes"]
                ]

    def to_fomdp(self):
        self.pomdp = False
        self.observable_object_ids = [None for i in range(self.n_chars)]

    def _remove_house_obj(self, state):
        delete_ids = [
            x["id"] for x in state["nodes"] if x["class_name"].lower() in self.house_obj
        ]
        state["nodes"] = [x for x in state["nodes"] if x["id"] not in delete_ids]
        state["edges"] = [
            x
            for x in state["edges"]
            if x["from_id"] not in delete_ids and x["to_id"] not in delete_ids
        ]
        return state

    def get_observations(self, graph_env=None, char_index=0):
        if graph_env is None:
            state = self.vh_state.to_dict()
        else:
            state = graph_env

        observable_state = self._mask_state(state, char_index) if self.pomdp else state
        return observable_state

    # def step(self, scripts):
    #     obs_n = []
    #     info_n = {'n':[]}
    #     reward_n = []

    #     if self.pomdp:
    #         for i in range(self.n_chars):
    #             if i not in scripts:
    #                 continue
    #             assert self._is_action_valid(scripts.get(i), i)

    #     # State transition: Sequentially performing actions
    #     # TODO: Detect action conflicts
    #     # convert action to a single action script
    #     objs_in_use = []
    #     for i in range(self.n_chars):
    #         if i not in scripts:
    #             continue
    #         script = read_script_from_string(scripts.get(i, ""))

    #         is_executable, msg = self._is_action_executable(script, i, objs_in_use)
    #         if (is_executable):
    #             objs_in_use += script.obtain_objects()
    #             succeed, self.vh_state = self.executor_n[i].execute_one_step(script, self.vh_state)
    #             info_n['n'].append({
    #                 "succeed": succeed,
    #                 "error_message": {i: self.executor_n[i].info.get_error_string() for i in range(self.n_chars)}
    #             })
    #         else:
    #             info_n['n'].append({
    #                 "succeed": False,
    #                 "error_message": {i: msg}
    #             })

    #     state = self.vh_state.to_dict()
    #     self.state = state

    #     for i in range(self.n_chars):
    #         observable_state = self._mask_state(state, i) if self.pomdp else state
    #         self.observable_state_n[i] = observable_state
    #         self.observable_object_ids_n[i] = [node["id"] for node in observable_state["nodes"]]
    #         obs_n.append(observable_state)
    #         # Reward Calculation

    #         # progress = self.tasks_n[i].measure_progress(self.observable_state_n[i], i)
    #         #progress_per_task = [task.measure_progress(self.observable_state_n[i], i) for task in self.tasks_n[i]]
    #         #progress = sum(progress_per_task) / float(len(progress_per_task))
    #         progress = 0
    #         reward_n.append(progress - self.prev_progress_n[i])
    #         self.prev_progress_n[i] = progress

    #         # if abs(progress - 1.0) < 1e-6:
    #         #     info_n['n'][i].update({'terminate': True})
    #         # else:
    #         #     info_n['n'][i].update({'terminate': False})

    #     # Information

    #     return reward_n, obs_n, info_n

    def reward(self, agent_id, state):
        # progress_per_task = [task.measure_progress(self._mask_state(state, agent_id), agent_id) for task in self.tasks_n[agent_id]]
        return 0
        # return sum(progress_per_task) / float(len(progress_per_task))

    def transition(self, vh_state, scripts, do_assert=False):

        # print(scripts, self.observable_object_ids_n[0])

        # print(
        #     "init",
        #     [
        #         e
        #         for e in vh_state.to_dict()["edges"]
        #         if e["from_id"] == 1 or e["to_id"] == 1
        #     ],
        # )

        if do_assert:
            if self.pomdp:
                for i in range(self.n_chars):
                    observable_nodes = self._mask_state(vh_state.to_dict(), i)["nodes"]
                    observable_object_ids = [node["id"] for node in observable_nodes]
                    assert self._is_action_valid_sim(
                        scripts.get(i), observable_object_ids
                    )

        next_vh_state = None
        for i in range(self.n_chars):
            script_string = scripts.get(i, "")
            if len(script_string) == 0:
                continue
            script = read_script_from_string(script_string)

            touched_objs = vh_state.touched_objs
            if "[offer]" in script_string:
                obj_id = script.obtain_objects()[0][1]
                obj_grabbed_person = list(
                    vh_state.get_node_ids_from(obj_id, Relation.HOLDS_RH)
                ) + list(vh_state.get_node_ids_from(obj_id, Relation.HOLDS_LH))
                next_vh_state = init_from_state(
                    next_vh_state, vh_state.touched_objs, vh_state.offer_obj
                )
                if len(obj_grabbed_person) > 0 and obj_grabbed_person[0] == i + 1:
                    succeed = True
                    next_vh_state.offer_object(obj_id)
                else:
                    succeed = False

            elif "[touch]" in script_string:

                succeed, next_vh_state = self.executor_n[i].execute_one_step(
                    script, copy.deepcopy(vh_state)
                )
                next_vh_state = init_from_state(next_vh_state, touched_objs)
                obj_id = script.obtain_objects()[0][1]
                next_vh_state.touch_object(obj_id)
            else:

                try:
                    succeed, next_vh_state = self.executor_n[i].execute_one_step(
                        script, copy.deepcopy(vh_state), in_place=True
                    )
                except:
                    print(script_string)
                    ipdb.set_trace()
                next_vh_state = init_from_state(
                    next_vh_state, touched_objs, vh_state.offer_obj
                )
                if script[0].action.name in ["GRAB", "PUTBACK", "PUTIN", "PUTOBJBACK"]:
                    # # inside_obj += curr_inside_obj
                    # print(
                    #     [
                    #         e
                    #         for e in vh_state.to_dict()["edges"]
                    #         if e["from_id"] == 1 or e["to_id"] == 1
                    #     ]
                    # )
                    # print(
                    #     [
                    #         e
                    #         for e in next_vh_state.to_dict()["edges"]
                    #         if e["from_id"] == 1 or e["to_id"] == 1
                    #     ]
                    # )
                    # ipdb.set_trace()
                    obj_id = script.obtain_objects()[0][1]
                    next_vh_state.remove_obj_offer(obj_id)
            if not succeed:
                # print(
                #     "Failed",
                #     [
                #         e
                #         for e in vh_state.to_dict()["edges"]
                #         if e["from_id"] == 1 or e["to_id"] == 1
                #     ],
                # )
                # raise ValueError
                # ipdb.set_trace()
                return False, next_vh_state

        if next_vh_state is None:
            ipdb.set_trace()
        # state = next_vh_state.to_dict()
        return True, next_vh_state

    def get_vh_state(self, state, name_equivalence=None, instance_selection=True):
        if name_equivalence is None:
            name_equivalence = self.name_equivalence

        # Remove touched state
        touched_objs = []
        for node in state["nodes"]:
            if "TOUCHED" in node["states"]:
                touched_objs.append(node["id"])
                node["states"] = [st for st in node["states"] if st != "TOUCHED"]
        env = EnvironmentState(
            EnvironmentGraph(state),
            self.name_equivalence,
            instance_selection=True,
            touched_objs=touched_objs,
        )
        for node in state["nodes"]:
            if node["id"] in touched_objs:
                node["states"].append("TOUCHED")
        return env

    def fill_missing_states(self, state):
        for node in state["nodes"]:
            object_name = node["class_name"]
            states_graph_old = node["states"]
            bin_vars = self.graph_helper.get_object_binary_variables(object_name)
            bin_vars_missing = [
                x
                for x in bin_vars
                if x.positive not in states_graph_old
                and x.negative not in states_graph_old
            ]
            print(node["class_name"], [x.default for x in bin_vars_missing])
            states_graph = states_graph_old + [x.default for x in bin_vars_missing]
            # fill out the rest of info regarding the states
            node["states"] = states_graph

    # TODO: Now the random function doesn't align with the manually set seed
    # task_goals_n is a list of list that represents the goals of every agent
    def reset(self, state):
        ############ State ############

        state = self._remove_house_obj(state)

        # Fill out the missing states
        # self.fill_missing_states(state)
        # ipdb.set_trace()
        for i in range(self.n_chars):
            self.executor = ScriptExecutor(
                EnvironmentGraph(state), self.name_equivalence, i
            )

        self.character_n = [None for i in range(self.n_chars)]
        chars = [node for node in state["nodes"] if node["category"] == "Characters"]
        chars.sort(key=lambda node: node["id"])

        self.character_n = chars

        self.rooms = []
        for node in state["nodes"]:
            if node["category"] == "Rooms":
                self.rooms.append(node)
        self.rooms_ids = [n["id"] for n in self.rooms]
        self.state = state
        self.id2node = {node["id"]: node for node in state["nodes"]}
        self.vh_state = self.get_vh_state(state)

        ############ Reward ############
        observable_state_n = [
            self._mask_state(state, i) if self.pomdp else state
            for i in range(self.n_chars)
        ]
        self.observable_state_n = observable_state_n
        self.observable_object_ids_n = [
            [node["id"] for node in obs_state["nodes"]]
            for obs_state in observable_state_n
        ]

        return observable_state_n

    def render(self, mode="human", close=False):
        return

    def _is_action_valid(self, string: str, char_index):

        script = read_script_from_string(string)

        valid = True
        for object_and_id in script.obtain_objects():
            id = object_and_id[1]
            if id not in self.observable_object_ids_n[char_index]:
                valid = False
                break

        return valid

    def _is_action_executable(self, script, char_index, objs_in_use):
        # if there's agent already interacting with the object in this step
        for obj in script.obtain_objects():
            if obj in objs_in_use:
                return False, "object <{}> ({}) is interacted by other agent".format(
                    obj[0], obj[1]
                )
        # # if object is held by others
        #     for i in range(self.n_chars):
        #         if i != char_index:
        #             node = Node(obj[1])
        #             if self.vh_state.evaluate(ExistsRelation(CharacterNode(char_index), Relation.HOLDS_RH, NodeInstance(node))) or \
        #                self.vh_state.evaluate(ExistsRelation(CharacterNode(char_index), Relation.HOLDS_LH, NodeInstance(node))):
        #                 return False, "object <{}> ({}) is held by other agent".format(obj[0], obj[1])
        return True, None

    def _is_action_valid_sim(self, string: str, observable_object_ids):

        script = read_script_from_string(string)

        valid = True
        for object_and_id in script.obtain_objects():
            id = object_and_id[1]
            if id not in observable_object_ids:
                valid = False
                break

        return valid

    def get_action_space(
        self,
        vh_state=None,
        char_index=0,
        action=None,
        obj1=None,
        obj2=None,
        structured_actions=False,
    ):
        # TODO: this could probably just go into virtualhome

        if vh_state is None:
            vh_state = self.vh_state
            nodes = self.observable_state_n[char_index]["nodes"]
        else:
            nodes = self._mask_state(vh_state.to_dict(), char_index)["nodes"]
        node_ids = [x["id"] for x in nodes]

        action_executors = self.executor_n[char_index]._action_executors

        if obj1 is not None and obj1["id"] not in node_ids:
            return []

        action_list = []
        action_candidates = self.actions if action is None else [action]
        action_list_sep = []

        for action in action_candidates:
            curr_action = Action[action.upper()]
            num_params = curr_action.value[1]
            objects = [[] for _ in range(num_params)]
            for param in range(num_params):
                properties_params = curr_action.value[2][param]
                if param == 0:
                    node_candidates = nodes if obj1 not in nodes else [obj1]
                elif param == 1:
                    node_candidates = nodes if obj2 not in nodes else [obj2]
                else:
                    node_candidates = nodes
                # if param == 0:
                #     node_candidates = nodes if obj1 is None else [obj1]
                # elif param == 1:
                #     node_candidates = nodes if obj2 is None else [obj2]
                # else:
                #     node_candidates = nodes

                # remove character from candidates
                node_candidates = [
                    x for x in node_candidates if x["class_name"] != "character"
                ]

                # if obj1 is not None and obj1['id'] == 2038:
                #     print('node candidates:', [node['id'] for node in node_candidates])

                for node in node_candidates:
                    if (
                        len(properties_params) == 0
                        or len(set(node["properties"]).intersection(properties_params))
                        > 0
                    ):
                        objects[param].append(node)

            if any([len(x) == 0 for x in objects]):
                continue
            prod = list(itertools.product(*objects))
            for obj_candidates in prod:
                obj_cand_list = list(obj_candidates)
                string_instr = self.obtain_formatted_action(action, obj_cand_list)
                action_list_tuple = [action] + obj_cand_list
                if action in ["Walk", "Find", "Run"]:
                    succeed = True
                else:
                    script = read_script_from_string(string_instr)
                    # This fails, it is modifyng the graph
                    succeed = self.executor_n[char_index].check_one_step(
                        script, vh_state
                    )
                    self.executor_n[char_index].info = ExecutionInfo()
                if succeed:
                    action_list.append(string_instr.lower())
                    action_list_sep.append(action_list_tuple)

        if structured_actions:
            return action_list_sep
        else:
            return action_list

    def obtain_formatted_action(self, action, obj_cand_list, debug=False):
        if len(obj_cand_list) == 0:
            return "[{}]".format(action)
        if debug:
            import pdb

            pdb.set_trace()
        obj_list = " ".join(
            [
                "<{}> ({})".format(node_obj["class_name"], node_obj["id"])
                for node_obj in obj_cand_list
            ]
        )
        string_instr = "[{}] {}".format(action, obj_list)
        return string_instr

    def _mask_state(self, state, char_index):
        # Assumption: inside is not transitive. For every object, only the closest inside relation is recorded
        character = self.character_n[char_index]
        # find character
        character_id = character["id"]
        id2node = {node["id"]: node for node in state["nodes"]}
        inside_of, is_inside, edge_from = {}, {}, {}

        grabbed_ids = []
        for edge in state["edges"]:

            if edge["relation_type"] == "INSIDE":

                if edge["to_id"] not in is_inside.keys():
                    is_inside[edge["to_id"]] = []

                is_inside[edge["to_id"]].append(edge["from_id"])
                inside_of[edge["from_id"]] = edge["to_id"]

            elif "HOLDS" in edge["relation_type"]:
                if edge["from_id"] == character["id"]:
                    grabbed_ids.append(edge["to_id"])

        character_inside_ids = inside_of[character_id]
        room_id = character_inside_ids

        object_in_room_ids = is_inside[room_id]

        # Some object are not directly in room, but we want to add them
        curr_objects = list(object_in_room_ids)
        while len(curr_objects) > 0:
            objects_inside = []
            for curr_obj_id in curr_objects:
                new_inside = (
                    is_inside[curr_obj_id] if curr_obj_id in is_inside.keys() else []
                )
                objects_inside += new_inside

            object_in_room_ids += list(objects_inside)
            curr_objects = list(objects_inside)

        # Only objects that are inside the room and not inside something closed
        # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
        object_hidden = (
            lambda ido: inside_of[ido] not in self.rooms_ids
            and "OPEN" not in id2node[inside_of[ido]]["states"]
        )
        observable_object_ids = [
            object_id
            for object_id in object_in_room_ids
            if not object_hidden(object_id)
        ] + self.rooms_ids
        observable_object_ids += grabbed_ids

        partilly_observable_state = {
            "edges": [
                edge
                for edge in state["edges"]
                if edge["from_id"] in observable_object_ids
                and edge["to_id"] in observable_object_ids
            ],
            "nodes": [id2node[id_node] for id_node in observable_object_ids],
        }

        return partilly_observable_state

    def _find_node_by_id(self, state, id):
        for node in state["nodes"]:
            if node["id"] == id:
                return node
        return None

    def _filter_edge(self, state, filter):

        target = []
        for edge in state["edges"]:
            if filter(edge):
                target.append(edge)

        return target if len(target) > 0 else None

    def _filter_node(self, state, filter):

        target = []
        for node in state["nodes"]:
            if filter(node):
                target.append(node)

        return target if len(target) > 0 else None

    def _find_targets(self, state, from_id, relation, to_id):

        assert sum([from_id == None, relation == None, to_id == None]) <= 1

        target = []
        if from_id is None:
            for e in state["edges"]:
                if e["relation_type"] == relation and e["to_id"] == to_id:
                    target.append(e["from_id"])

        elif to_id is None:
            for e in state["edges"]:
                if e["relation_type"] == relation and e["from_id"] == from_id:
                    target.append(e["to_id"])

        return target if len(target) > 0 else None

    def __str__(self):

        s = ""
        for i in range(self.n_chars):
            s += "Character {}".format(self.character_n[i]["id"]) + "\n"
            s += "Task goal: ({})".format(self.task.goal_n[i]) + "\n"

        return s


def _test1():

    env = VhGraphEnv()
    task_goals = (
        "(and (ontop phone[247] kitchen_counter_[230]) (inside character[65]"
        " dining_room[201]))"
    )
    state_path = "/scratch/gobi1/andrewliao/programs_processed_precond_nograb_morepreconds/init_and_final_graphs/TrimmedTestScene1_graph/results_intentions_march-13-18/file1003_2.json"
    s = env.reset(state_path, task_goals)

    env.to_pomdp()
    r, s, info = env.step("[walk] <dining_room> (201)")
    r, s, info = env.step("[walk] <phone> (247)")
    r, s, info = env.step("[grab] <phone> (247)")
    print(r, info)


if __name__ == "__main__":
    import ipdb

    _test1()
