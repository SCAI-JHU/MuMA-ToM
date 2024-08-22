import pickle
import pdb
import sys
import os
import random
from termcolor import colored
import json
import ipdb
import numpy as np
import copy
import argparse


curr_dir = os.path.dirname(os.path.abspath(__file__))


class SetInitialGoal:
    def __init__(
        self,
        obj_position,
        class_name_size, #environment parameter
        init_pool_tasks, #from json file
        task_name, #task name like set up table
        same_room=True,
        goal_template=None,
        rand=None,
        nprand=None,
        set_curr_goal=True,
        set_random_goal=False,
    ):
        self.task_name = task_name
        self.init_pool_tasks = init_pool_tasks
        self.obj_position = obj_position
        self.class_name_size = class_name_size
        self.object_id_count = 1000
        self.surface_size = {}
        self.surface_used_size = {}
        self.max_num_place = 50
        self.goal_template = goal_template

        self.min_num_other_object = 15  # 15
        self.max_num_other_object = 25  # 45
        self.set_random_goal = set_random_goal
        self.add_goal_obj_success = True
        self.set_curr_goal = set_curr_goal
        if rand is not None:
            self.rand = rand
            self.nprand = nprand
        else:
            self.rand = random.Random()
            self.nprand = np.random.RandomState()

        if nprand is not None:
            self.nprand = nprand
        else:
            self.nprand = np.random.RandomState()
        self.set_goal()

        self.same_room = same_room

    def set_goal(self):
        self.init_pool = {}
        if self.task_name in [
            "setup_table_toy",
            "setup_table",
            "clean_table",
            "put_dishwasher",
            "unload_dishwasher",
            "put_fridge",
            "read_book",
            "prepare_food",
            "watch_tv",
            "setup_table_toy_2",
            "setup_table_toy_1",
            "setup_desk",
            "prepare_drink",
            "collect_document",
            "collect_toy",
            "clear_fridge",
            "clear_table",
            "clear_desk"
        ]:
            self.init_pool[self.task_name] = self.init_pool_tasks[self.task_name]

        elif self.task_name == "setup_table_prepare_food":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["setup_table"])
            self.init_pool.update(self.init_pool_tasks["prepare_food"])

        elif self.task_name == "setup_table_read_book":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["setup_table"])
            self.init_pool.update(self.init_pool_tasks["read_book"])

        elif self.task_name == "setup_table_watch_tv":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["setup_table"])
            self.init_pool.update((self.init_pool_tasks["watch_tv"]))

        elif self.task_name == "setup_table_put_fridge":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["setup_table"])
            self.init_pool.update(self.init_pool_tasks["put_fridge"])

        elif self.task_name == "setup_table_put_dishwasher":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["setup_table"])
            self.init_pool.update(self.init_pool_tasks["put_dishwasher"])

        elif self.task_name == "prepare_food_put_dishwasher":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["prepare_food"])
            self.init_pool.update(self.init_pool_tasks["put_dishwasher"])

        elif self.task_name == "put_fridge_put_dishwasher":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["put_fridge"])
            self.init_pool.update(self.init_pool_tasks["put_dishwasher"])

        elif self.task_name == "put_dishwasher_read_book":
            self.init_pool = copy.deepcopy(self.init_pool_tasks["put_dishwasher"])
            self.init_pool.update(self.init_pool_tasks["read_book"])
        
        elif "and" in self.task_name:
            tasks = self.task_name.split("_and_")
            self.init_pool[tasks[0]] = copy.deepcopy(self.init_pool_tasks[tasks[0]])
            self.init_pool[tasks[1]] = copy.deepcopy(self.init_pool_tasks[tasks[1]])

        ## make sure the goal is not empty
        self.goal_random_agent = {}
        deb = """
        while 1:
            self.goal = {}
            for k,v in self.init_pool.items():
                self.goal[k] = random.randint(v['min_num'], v['max_num'])

            # break

            count = 0
            for k,v in self.goal.items():
                count+=v
        """
        if self.set_curr_goal:
            if self.goal_template is not None:
                self.goal = {}
                for predicate, count in self.goal_template.items():
                    elements = predicate.split("_")
                    for e in elements:
                        if e in self.init_pool:
                            self.goal[e] = count
                print(self.goal_template)
                print(self.goal)
            else:
                while 1:
                    self.goal = {}
                    for k, v in self.init_pool.items():
                        self.goal[k] = self.rand.randint(v["min_num"], v["max_num"])
                        #self.goal[k] = 0
                    #self.goal["plate"] = 1
                    #self.goal["cupcake"] = 1

                    # break

                    count = 0
                    for k, v in self.goal.items():
                        count += v

                    if (
                        self.task_name == "read_book"
                        and 2 <= count <= 4
                        or self.task_name == "watch_tv"
                        and 2 <= count <= 4
                    ):
                        break

                    if (
                        2 <= count <= 6
                        and self.task_name not in ["clean_table", "unload_dishwasher"]
                        or 3 <= count <= 6
                    ):
                        break

        # Select goals for random agent
        if self.set_random_goal:
            objects_not_pick = [
                goal_name
                for goal_name, goal_count in self.goal.items()
                if goal_count > 0
            ]
            total_count = self.rand.randint(2, 6)

            possible_objects_dict = self.init_pool_tasks["noise"]
            objects_used = []
            object_candidates = list(possible_objects_dict.keys())
            self.rand.shuffle(object_candidates)
            it_obj = 0
            while total_count > 0 and it_obj < len(object_candidates):
                object_name = object_candidates[it_obj]
                it_obj += 1
                if object_name in objects_not_pick:
                    continue
                v = possible_objects_dict[object_name]
                count = min(self.rand.randint(v["min_num"], v["max_num"]), total_count)
                total_count -= count
                self.goal_random_agent[object_name] = count

    def get_obj_room(self, obj_id):
        room_ids = [
            node["id"] for node in graph["nodes"] if node["category"] == "Rooms"
        ]
        room_info = [
            edge["to_id"]
            for edge in graph["edges"]
            if edge["to_id"] in room_ids
            and edge["relation_type"] == "INSIDE"
            and edge["from_id"] == obj_id
        ]
        assert len(room_info) == 1

        objs_in_room = [
            edge["from_id"]
            for edge in graph["edges"]
            if edge["to_id"] == room_info[0] and edge["relation_type"] == "INSIDE"
        ]

        return objs_in_room

    def check_graph(self, graph, apartment, original_graph):
        current_objects = {node["id"]: node["class_name"] for node in graph["nodes"]}
        current_object_ids = list(current_objects.keys())

        OBJ_LIST = [
            "plate",
            "waterglass",
            "wineglass",
            "cutleryfork",
            "cupcake",
            "juice",
            "pancake",
            "poundcake",
            "wine",
            "pudding",
            "apple",
            "coffeepot",
            "cutleryknife",
        ]

        # nodes_to_check = [node_id for node_id in current_object_ids if node_id not in initial_object_ids]
        # nodes_to_check = [node['id'] for node in graph['nodes'] if node['class_name'] in OBJ_LIST]
        nodes_to_check = current_object_ids

        id2node = {node["id"]: node for node in graph["nodes"]}
        connected_edges = {id: [] for id in nodes_to_check}
        for edge in graph["edges"]:
            if (
                edge["from_id"] in nodes_to_check
                and edge["relation_type"] != "CLOSE"
                and id2node[edge["to_id"]]["category"] != "Rooms"
            ):
                connected_edges[edge["from_id"]].append(edge)

        ori_id2node = {node["id"]: node for node in original_graph["nodes"]}
        ori_connected_edges = {id: [] for id in nodes_to_check}
        for edge in original_graph["edges"]:
            if (
                edge["from_id"] in nodes_to_check
                and edge["relation_type"] != "CLOSE"
                and ori_id2node[edge["to_id"]]["category"] != "Rooms"
            ):
                ori_connected_edges[edge["from_id"]].append(edge)

        print("num nodes:")
        print(len(connected_edges), len(ori_connected_edges))

        for node_id, edges in connected_edges.items():
            if len(edges) < 1:
                if node_id in ori_connected_edges:
                    pass

                    # print(len(ori_connected_edges[node_id]))
                    # print(len(edges))

                    # if len(edges)!=len(ori_connected_edges[node_id]):
                    #     pdb.set_trace()

                    # assert id2node[node_id]['class_name']==ori_id2node[node_id]['class_name']

                    # # print(node_id)
                    # # print(edges)
                    # # print(ori_connected_edges[node_id])

                    # # print(id2node[node_id]['class_name'])
                    # # print(ori_id2node[node_id]['class_name'])

                    # if len(ori_connected_edges[node_id])>=1:
                    #     print('old object error')
                    #     print(node_id, id2node[node_id]['class_name'])
                    #     return False

                else:
                    print("add new object error")
                    print(node_id, id2node[node_id]["class_name"])
                    return False

        return True

    def check_goal_achievable(self, graph, comm, env_goal, apartment):
        graph_copy = copy.deepcopy(graph)

        # comm.reset(apartment)
        # success, message = comm.expand_scene(graph_copy)
        # print(success, message)
        # pdb.set_trace()
        if (
            ("setup_table" in self.task_name)
            or ("put_dishwasher" in self.task_name)
            or ("put_fridge" in self.task_name)
            or ("prepare_food" in self.task_name)
        ):
            curr_task_name = list(env_goal.keys())[0]
            for goal in env_goal[curr_task_name]:
                # print(self.object_id_count)
                subgoal_name = list(goal.keys())[0]
                num_obj = list(goal.values())[0]
                obj = subgoal_name.split("_")[1]
                target_id = int(subgoal_name.split("_")[3])

                if self.same_room:
                    objs_in_room = self.get_obj_room(target_id)
                else:
                    objs_in_room = None

                obj_ids = [
                    node["id"]
                    for node in graph_copy["nodes"]
                    if obj == node["class_name"]
                ]
                if len(obj_ids) < num_obj:
                    print(subgoal_name, num_obj, obj_ids)
                    # pdb.set_trace()
                    return 0

                graph_copy = self.remove_obj(graph_copy, obj_ids)

                self.object_id_count, graph, success_add_obj = self.add_obj(
                    graph_copy,
                    obj,
                    num_obj,
                    self.object_id_count,
                    objs_in_room=objs_in_room,
                    only_position=target_id,
                )

                if not success_add_obj:
                    return False
                # comm.reset(apartment)
                # success, message = comm.expand_scene(graph_copy)
                # print(success, message)
                # pdb.set_trace()

            comm.reset(apartment)
            success, message = comm.expand_scene(graph_copy)
            id2node = {node["id"]: node for node in graph_copy["nodes"]}
            if not success:
                if "unaligned_ids" in message:
                    for id in message["unaligned_ids"]:
                        print(id2node[id])
                elif "unplaced" in message:
                    for string in message["unplaced"]:
                        elements = string.split(".")
                        obj_id = int(elements[1])
                        print(
                            [
                                edge
                                for edge in graph_copy["edges"]
                                if edge["from_id"] == obj_id
                            ]
                        )
        else:
            success = 1
            message = self.task_name

        return success

    def convert_size(self, envsize):
        size = envsize[0] * envsize[2]
        return size

    def check_placeable(self, graph, surface_id, obj_name, relation="ON"):
        if obj_name == "apple":
            print(obj_name, surface_id)

        obj_size = self.convert_size(self.class_name_size[obj_name])

        surface_node = [node for node in graph["nodes"] if node["id"] == surface_id]

        if surface_id not in self.surface_size:
            surface_node = [node for node in graph["nodes"] if node["id"] == surface_id]
            assert len(surface_node)
            self.surface_size[surface_id] = self.convert_size(
                self.class_name_size[surface_node[0]["class_name"]]
            )

        classes_discount = ["kitchencounterdrawer", "oventray"]
        if surface_id not in self.surface_used_size:
            objs_on_surface = [
                edge["from_id"]
                for edge in graph["edges"]
                if edge["to_id"] == surface_id and edge["relation_type"] == relation
            ]

            objs_on_surface_node = [
                node
                for node in graph["nodes"]
                if node["id"] in objs_on_surface
                and node["class_name"] not in classes_discount
            ]
            objs_on_surface_size = [
                self.convert_size(self.class_name_size[node["class_name"]])
                for node in objs_on_surface_node
            ]
            self.surface_used_size[surface_id] = np.sum(
                objs_on_surface_size
            )  # get size from the initial graph

        # if surface_node[0]['class_name'] == 'microwave':
        #     ipdb.set_trace()

        # print(self.surface_size[surface_id])
        # print(self.surface_used_size[surface_id], obj_size, self.surface_used_size[surface_id]+obj_size)
        # print(obj_name, surface_node[0]['class_name'])

        if (
            self.surface_size[surface_id] / 2
            > self.surface_used_size[surface_id] + obj_size
        ):
            self.surface_used_size[surface_id] += obj_size
            #print("1")

            return 1
        else:
            # if obj_name == 'book':
            #     objs_on_surface = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == surface_id and edge['relation_type'] == relation]

            #     print(obj_name, len(objs_on_surface), surface_node[0]['class_name'])

            # if surface_node[0]['class_name'] == "cabinet":
            #     pdb.set_trace()
            # ipdb.set_trace()
            #print("0")
            return 0

    def remove_obj(self, graph, obj_ids):
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in obj_ids]
        graph["edges"] = [
            edge
            for edge in graph["edges"]
            if edge["from_id"] not in obj_ids and edge["to_id"] not in obj_ids
        ]
        return graph

    def add_obj(
        self,
        graph,
        obj_name,
        num_obj,
        object_id,
        objs_in_room=None,
        only_position=None,
        except_position=None,
        goal_obj=False,
        enforced_adding=None
    ):
        # Place num_obj of type obj_name, starting with object_id

        if enforced_adding is not None:
            nodes = []
            edges = []
            for _ in range(num_obj):
                new_node = {
                        "id": object_id,
                        "class_name": obj_name,
                        "properties": ["GRABBABLE"],
                        "states": [],
                        "category": "added_object",
                    }
                nodes.append(new_node)
                edges.append(
                    {
                        "from_id": object_id,
                        "relation_type": enforced_adding[1],
                        "to_id": enforced_adding[0],
                    }
                )
                object_id += 1
            graph["nodes"] += nodes
            graph["edges"] += edges
            return object_id, graph, True

        if isinstance(except_position, int):
            except_position = [except_position]
        if isinstance(only_position, int):
            only_position = [only_position]

        edges = []
        nodes = []
        ids_class = {}
        for node in graph["nodes"]:
            class_name = node["class_name"]
            if class_name not in ids_class:
                ids_class[class_name] = []
            ids_class[class_name].append(node["id"])

        # candidates = [(obj_rel_name[0], obj_rel_name[1]) for obj_rel_name in obj_position_pool[obj_name] if obj_rel_name[1] in ids_class.keys() and (except_position is None or obj_rel_name[1] not in except_position) and (only_position is None or obj_rel_name[1] in only_position)]

        candidates = [
            (obj_rel_name[0], obj_rel_name[1])
            for obj_rel_name in self.obj_position[obj_name]
            if obj_rel_name[1] in ids_class.keys()
        ]
        init_candidates = [
            (obj_rel_name[0], obj_rel_name[1])
            for obj_rel_name in self.obj_position[obj_name]
            if obj_rel_name[1] in ids_class.keys()
        ]
        # print(candidates)
        #print(
        #    "Placing: {}. Candidates: {} {}".format(
        #        obj_name, candidates, self.obj_position[obj_name]
        #    )
        #)
        id2node = {node["id"]: node for node in graph["nodes"]}
        success_add = 0

        for i in range(num_obj):
            # TODO: we need to check the properties and states, probably the easiest is to get them from the original set of graphs

            num_place = 0

            while 1:
                if num_place > self.max_num_place:
                    break

                if only_position != None:
                    num_place2 = 0
                    while 1:
                        if num_place2 > self.max_num_place:
                            break
                        target_id = self.rand.choice(only_position)
                        if self.same_room and goal_obj:
                            if target_id in objs_in_room:
                                break
                            else:
                                num_place2 += 1
                        else:
                            break

                    # target_id = self.rand.choice(only_position)

                    target_id_name = [
                        node["class_name"]
                        for node in graph["nodes"]
                        if node["id"] == target_id
                    ]
                    if "livingroom" in target_id_name and obj_name == "plate":
                        pdb.set_trace()

                    target_pool = [k for k, v in ids_class.items() if target_id in v]
                    target_position_pool = [
                        tem[0]
                        for tem in self.obj_position[obj_name]
                        if tem[1] in target_pool
                    ]

                    if (
                        len(target_pool) == 0
                        or len(target_position_pool) == 0
                        or (num_place2 > self.max_num_place)
                    ):
                        num_place += 1
                        continue
                    else:
                        relation = self.rand.choice(target_position_pool)

                else:
                    num_place2 = 0
                    while 1:
                        if num_place2 > self.max_num_place:
                            break
                        if len(candidates) == 0:
                            ipdb.set_trace()
                        relation, target_classname = self.rand.choice(candidates)

                        candidate_obj = [x[1] for x in candidates]
                        target_id = self.rand.choice(ids_class[target_classname])

                        target_id_name = [
                            node["class_name"]
                            for node in graph["nodes"]
                            if node["id"] == target_id
                        ]

                        if "livingroom" in target_id_name and obj_name == "plate":
                            pdb.set_trace()

                        if self.same_room and goal_obj:
                            if target_id in objs_in_room:
                                ipdb.set_trace()
                                break
                            else:
                                num_place2 += 1
                        else:
                            break

                    # for tem in candidates:
                    #     if 'plate' in tem:
                    #         print(candidates)
                    #         pdb.set_trace()

                    ## target in except_position

                    if (
                        (except_position != None) and (target_id in except_position)
                    ) or (num_place2 > self.max_num_place):
                        num_place += 1
                        continue

                ## check if it is possible to put object in this surface
                # if target_classname == 'microwave' or target_classname == 'stove':
                #     ipdb.set_trace()
                placeable = self.check_placeable(
                    graph, target_id, obj_name, relation=relation
                )

                # print(target_classname, 'placed in ', obj_name, 'Success', placeable)
                # print(obj_name, id2node[target_id]['class_name'], placeable)
                # print('placing %s: %dth (total %d), success: %d' % (obj_name, i+1, num_obj, placeable))

                if placeable:
                    new_node = {
                        "id": object_id,
                        "class_name": obj_name,
                        "properties": ["GRABBABLE"],
                        "states": [],
                        "category": "added_object",
                    }
                    nodes.append(new_node)
                    edges.append(
                        {
                            "from_id": object_id,
                            "relation_type": relation,
                            "to_id": target_id,
                        }
                    )
                    object_id += 1
                    success_add += 1
                    break
                else:
                    num_place += 1
        #print(nodes)
        #print(edges)
        graph["nodes"] += nodes
        graph["edges"] += edges

        if goal_obj:
            # print(success_add, num_obj)
            if success_add != num_obj:
                # print(obj_name)
                # ipdb.set_trace()
                return None, None, False

        return object_id, graph, True

    def setup_other_objs(
        self,
        graph,
        object_id,
        objs_in_room=None,
        except_position=None,
        except_objects=[],
    ):
        new_object_pool = [
            tem
            for tem in self.init_pool_tasks["obj_random"]
            if tem not in list(self.goal.keys()) + except_objects
        ]  # remove objects in goal

        self.num_other_obj = self.rand.randint(
            self.min_num_other_object, self.max_num_other_object
        )
        obj_in_graph = [
            node["class_name"] for node in graph["nodes"]
        ]  # if the object already in env, skip
        # print(self)
        added_objects, failed_objects = [], [] # TODO(xinyu): seems it's a bug, since failed_objects is not used
        for i in range(self.num_other_obj):
            obj_name = self.rand.choice(new_object_pool)
            if obj_name in obj_in_graph:
                continue

            object_id, graph, success = self.add_obj(
                graph,
                obj_name,
                1,
                object_id,
                objs_in_room=objs_in_room,
                only_position=None,
                except_position=except_position,
            )

            if success:
                added_objects.append(obj_name)

        # ipdb.set_trace()
        print("Adding objects:")
        print(colored(added_objects, "green"))
        print(colored(failed_objects, "red"))
        return object_id, graph

    def set_tv_off(self, graph, tv_id):
        node = [n for n in graph["nodes"] if n["id"] == tv_id]
        assert len(node) == 1
        node[0]["states"] = ["OFF"]
        # + [state for state in node[0]['states'] if state not in ['ON', 'OFF']]
        return graph


def debug_function(comm):
    with open("data/object_info.json", "r") as file:
        obj_position = json.load(file)

    success_edges = []
    fail_target_nodes = []

    for obj_name in obj_position["objects_grab"]:
        object_id = 2000
        new_node = {
            "id": object_id,
            "class_name": obj_name,
            "properties": ["GRABBABLE"],
            "states": [],
            "category": "added_object",
        }
        nodes = [new_node]

        for target_name in obj_position["objects_inside"]:
            s, graph = comm.environment_graph()

            target_node = [
                node for node in graph["nodes"] if node["class_name"] == target_name
            ]
            if len(target_node) == 0:
                print(target_name)
                fail_target_nodes.append(target_name)
                continue

            target_id = target_node[0]["id"]

            edges = [
                {"from_id": object_id, "relation_type": "INSIDE", "to_id": target_id}
            ]

            graph["nodes"] += nodes
            graph["edges"] += edges
            comm.reset(apartment)
            success, message = comm.expand_scene(graph)
            # print(success, message)

            if success:
                success_edges.append(
                    {
                        "from_id": obj_name,
                        "relation_type": "INSIDE",
                        "to_id": target_name,
                    }
                )
            else:
                print(
                    {
                        "from_id": obj_name,
                        "relation_type": "INSIDE",
                        "to_id": target_name,
                    }
                )

        for target_name in obj_position["objects_surface"]:
            s, graph = comm.environment_graph()

            target_node = [
                node for node in graph["nodes"] if node["class_name"] == target_name
            ]
            if len(target_node) == 0:
                print(target_name)
                fail_target_nodes.append(target_name)
                continue

            target_id = target_node[0]["id"]

            edges = [{"from_id": object_id, "relation_type": "ON", "to_id": target_id}]

            graph["nodes"] += nodes
            graph["edges"] += edges
            comm.reset(apartment)
            success, message = comm.expand_scene(graph)
            # print(success, message)

            if success:
                success_edges.append(
                    {"from_id": obj_name, "relation_type": "ON", "to_id": target_name}
                )
            else:
                print(
                    {"from_id": obj_name, "relation_type": "ON", "to_id": target_name}
                )
