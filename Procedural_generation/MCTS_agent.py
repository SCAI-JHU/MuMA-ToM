import numpy as np
import scipy.special
import traceback
import logging
import random
import time
import math
import copy
import importlib
import json
import multiprocessing as mp
from functools import partial
import ipdb
import pdb
import pickle


from . import belief
from graph_env import VhGraphEnv
from language import Language, LanguageInquiry, LanguageResponse

#
import pdb
import MCTS_algorithm

import sys

import utils_environment as utils_env
import utils_exception


def find_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    # find_{index}

    target = int(object_target.split("_")[-1])
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    containerdict = {}
    hold = False
    for edge in env_graph["edges"]:
        if edge["relation_type"] == "INSIDE" and edge["from_id"] not in containerdict:
            containerdict[edge["from_id"]] = edge["to_id"]
        elif "HOLDS" in edge["relation_type"] and agent_id == 1:  # only for main agent
            containerdict[edge["to_id"]] = edge["from_id"]
            if edge["to_id"] == target:
                hold = True

    # containerdict = {
    #     edge['from_id']: edge['to_id']
    #     for edge in env_graph['edges']
    #     if edge['relation_type'] == 'INSIDE'
    # }

    observation_ids = [x["id"] for x in observations["nodes"]]

    # if agent_id == 1 and hold:
    #     print('container_ids find:', object_target, containerdict)

    try:
        room_char = [
            edge["to_id"]
            for edge in env_graph["edges"]
            if edge["from_id"] == agent_id and edge["relation_type"] == "INSIDE"
        ][0]
    except:
        print("Error")
        # ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     )ipdb.set_trace()
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            raise Exception
        # If the object is a room, we have to walk to what is insde

        if id2node[container]["category"] == "Rooms":
            action_list = [
                ("walk", (id2node[target]["class_name"], target), None)
            ] + action_list
            cost_list = [1] + cost_list

        elif "CLOSED" in id2node[container]["states"] or (
            "OPEN" not in id2node[container]["states"]
        ):
            if id2node[container]["class_name"] != "character":
                action = ("open", (id2node[container]["class_name"], container), None)
                action_list = [action] + action_list
                cost_list = [0.05] + cost_list

        target = container
        # if hold:
        #     print(target)

    ids_character = [
        x["to_id"]
        for x in observations["edges"]
        if x["from_id"] == agent_id and x["relation_type"] == "CLOSE"
    ] + [
        x["from_id"]
        for x in observations["edges"]
        if x["to_id"] == agent_id and x["relation_type"] == "CLOSE"
    ]

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [
            ("walk", (id2node[target]["class_name"], target), None)
        ] + action_list
        cost_list = [1] + cost_list

    # if hold:
    #     print(action_list)

    return action_list, cost_list, f"find_{target}"


def touch_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    target_action = [("touch", (target_node["class_name"], target_id), None)]
    cost = [0.05]

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"touch_{target_id}"
    else:
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"touch_{target_id}"


def grab_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("grab", (target_node["class_name"], target_id), None)]
        cost = [0.0]
    else:
        target_action = []
        cost = []

    # if agent_id == 1:
    #     print('observed_ids grab:', target_id, observed_ids)

    if len(agent_close) > 0 and target_id in observed_ids:
        if agent_id == 1 and target_id == 351:
            print(target_action)
        return target_action, cost, f"grab_{target_id}"
    else:
        #print("here")
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        if agent_id == 1 and target_id == 351:
            print(find_actions + target_action)
        return find_actions + target_action, find_costs + cost, f"grab_{target_id}"


def turnOn_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("switchon", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"turnon_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"turnon_{target_id}"


def sit_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    on_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "ON" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in on_ids:
        target_action = [("sit", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"sit_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"sit_{target_id}"


def put_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, target, verbose=False
):
    # Modif, now put heristic is only the immaediate after action
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]
    if verbose:
        ipdb.set_trace()
    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        # ipdb.set_trace()
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and agent_id == 2
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object is being placed by another agent
        # ipdb.set_trace()
        return [], 0, []

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
    target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )
    if verbose:
        ipdb.set_trace()
    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )

    res = grab_obj1 + find_obj2
    cost_list = cost_grab_obj1 + cost_find_obj2

    if verbose:
        ipdb.set_trace()
    if target_put > 2:  # not character
        action = [
            (
                "putback",
                (target_node["class_name"], target_grab),
                (target_node2["class_name"], target_put),
            )
        ]
        cost = [0.05]
        res += action
        cost_list += cost
    else:
        action = [("walk", (target_node2["class_name"], target_put), None)]
        cost = [1]
        res += action
        cost_list += cost
    # print(res, target)
    return res, cost_list, f"put_{target_grab}_{target_put}"


def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # TODO: change this as well
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and agent_id == 2
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object has been placed
        return None, 0, None

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
    target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name

    else:
        grab_obj1 = []
        cost_grab_obj1 = []

        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )
        target_put_state = target_node2["states"]
        action_open = [("open", (target_node2["class_name"], target_put))]
        action_put = [
            (
                "putin",
                (target_node["class_name"], target_grab),
                (target_node2["class_name"], target_put),
            )
        ]
        cost_open = [0.0]
        cost_put = [0.0]

        remained_to_put = 0
        for predicate, count in unsatisfied.items():
            if predicate.startswith("inside"):
                remained_to_put += count
        if remained_to_put == 1:  # or agent_id > 1:
            action_close = []
            cost_close = []
        else:
            action_close = []
            cost_close = []
            # action_close = [('close', (target_node2['class_name'], target_put))]
            # cost_close = [0.05]

        if "CLOSED" in target_put_state or "OPEN" not in target_put_state:
            res = grab_obj1 + find_obj2 + action_open + action_put + action_close
            cost_list = (
                cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
            )
        else:
            res = grab_obj1 + find_obj2 + action_put + action_close
            cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

        # print(res, target)
        grab_node = target_node["id"]
        place_node = target_node2["id"]
        return res, cost_list, f"putin_{grab_node}_{place_node}"


def clean_graph(state, goal_spec, last_opened):
    # TODO: document well what this is doing
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate, val_goal in goal_spec.items():
        elements = predicate.split("_")
        nodes_missing += val_goal["grab_obj_ids"]
        nodes_missing += val_goal["container_ids"]
    # get all grabbed object ids
    for edge in state["edges"]:
        if "HOLD" in edge["relation_type"] and edge["to_id"] not in nodes_missing:
            nodes_missing += [edge["to_id"]]

    nodes_missing += [
        node["id"]
        for node in state["nodes"]
        if node["class_name"] == "character"
        or node["category"] in ["Rooms", "Doors"]
        or node["class_name"]
        in [
            "kitchencabinet",
            "bathroomcabinet",
            "cabinet",
            "fridge",
            "microwave",
            "stove",
            "dishwasher",
            "coffeetable",
            "kitchentable",
            "desk",
            "sofa",
        ]
    ]

    def clean_node(curr_node):
        return {
            "id": curr_node["id"],
            "class_name": curr_node["class_name"],
            "category": curr_node["category"],
            "states": curr_node["states"],
            "properties": curr_node["properties"],
        }

    id2node = {node["id"]: clean_node(node) for node in state["nodes"]}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state["edges"]:
        if edge["relation_type"] == "INSIDE":
            if edge["from_id"] not in inside.keys():
                inside[edge["from_id"]] = []
            inside[edge["from_id"]].append(edge["to_id"])

    while len(nodes_missing) > 0:
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [
                    node_in
                    for node_in in inside[node_missing]
                    if node_in not in ids_interaction
                ]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key, value in goal_spec.items():
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["containers"][0]]["class_name"] in [
                "dishwasher",
                "kitchentable",
            ]:
                augmented_class_names += [
                    "kitchencabinets",
                    "kitchencounterdrawer",
                    "kitchencounter",
                ]
                break
    for key in goal_spec:
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["container_ids"][0]]["class_name"] in ["sofa", "chair"]:
                augmented_class_names += ["coffeetable"]
                break
    containers = [
        [node["id"], node["class_name"]]
        for node in state["nodes"]
        if node["class_name"] in augmented_class_names
    ]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    new_graph = {
        "edges": [
            edge
            for edge in state["edges"]
            if edge["from_id"] in ids_interaction and edge["to_id"] in ids_interaction
        ],
        "nodes": [id2node[id_node] for id_node in ids_interaction],
    }

    return new_graph


def mp_run_mcts(root_node, mcts, nb_steps, last_subgoal, opponent_subgoal):
    heuristic_dict = {
        "offer": put_heuristic,
        "find": find_heuristic,
        "grab": grab_heuristic,
        "put": put_heuristic,
        "putIn": putIn_heuristic,
        "sit": sit_heuristic,
        "turnOn": turnOn_heuristic,
        "touch": touch_heuristic,
    }
    # res = root_node * 2
    try:
        new_mcts = copy.deepcopy(mcts)
        res = new_mcts.run(
            root_node, nb_steps, heuristic_dict, last_subgoal, opponent_subgoal
        )
    except Exception as e:
        print("plan fail in index", root_node.particle_id)
        # traceback.print_stack()
        # print("raising")
        # print("Exception...")
        # print(utils_exception.ExceptionWrapper(e))
        # print('---')
        return utils_exception.ExceptionWrapper(e)
    return res


def mp_run_2(
    process_id, root_node, mcts, nb_steps, last_subgoal, opponent_subgoal, res
):
    res[process_id] = mp_run_mcts(
        root_node=root_node,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )


def get_plan(
    mcts,
    particles,
    env,
    nb_steps,
    goal_spec,
    last_subgoal,
    last_action,
    opponent_subgoal=None,
    num_process=10,
    length_plan=5,
    verbose=True,
):
    root_nodes = []
    for particle_id in range(len(particles)):
        root_action = None
        root_node = Node(
            id=(root_action, [goal_spec, 0, ""]),
            particle_id=particle_id,
            plan=[],
            state=copy.deepcopy(particles[particle_id]),
            num_visited=0,
            sum_value=0,
            is_expanded=False,
        )
        root_nodes.append(root_node)

    # root_nodes = list(range(10))
    mp_run = partial(
        mp_run_mcts,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )

    if len(root_nodes) == 0:
        print("No root nodes")
        raise Exception
    if num_process > 0:

        manager = mp.Manager()
        res = manager.dict()
        num_root_nodes = len(root_nodes)
        for start_root_id in range(0, num_root_nodes, num_process):
            end_root_id = min(start_root_id + num_process, num_root_nodes)
            jobs = []
            for process_id in range(start_root_id, end_root_id):
                # print(process_id)
                p = mp.Process(
                    target=mp_run_2,
                    args=(
                        process_id,
                        root_nodes[process_id],
                        mcts,
                        nb_steps,
                        last_subgoal,
                        opponent_subgoal,
                        res,
                    ),
                )
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        info = [res[x] for x in range(len(root_nodes))]

    else:
        info = [mp_run(rn) for rn in root_nodes]

    for info_item in info:
        if isinstance(info_item, utils_exception.ExceptionWrapper):
            print("rasiing")
            info_item.re_raise()

    if num_process > 0:
        print("Plan Done")
    rewards_all = [inf[-1] for inf in info]
    plans_all = [inf[1] for inf in info]
    goals_all = [inf[-2] for inf in info]
    index_action = 0
    # length_plan = 5
    prev_index_particles = list(range(len(info)))

    final_actions, final_goals = [], []
    lambd = 0.5
    # ipdb.set_trace()
    while index_action < length_plan:
        max_action = None
        max_score = None
        action_count_dict = {}
        action_reward_dict = {}
        action_goal_dict = {}
        # Which particles we select now
        index_particles = [
            p_id for p_id in prev_index_particles if len(plans_all[p_id]) > index_action
        ]
        # print(index_particles)
        if len(index_particles) == 0:
            index_action += 1
            continue
        for ind in index_particles:
            action = plans_all[ind][index_action]
            if action is None:
                continue
            try:
                reward = rewards_all[ind][index_action]
                goal = goals_all[ind][index_action]
            except:
                ipdb.set_trace()
            if not action in action_count_dict:
                action_count_dict[action] = []
                action_goal_dict[action] = []
                action_reward_dict[action] = 0
            action_count_dict[action].append(ind)
            action_reward_dict[action] += reward
            action_goal_dict[action].append(goal)

        for action in action_count_dict:
            # Average reward of this action
            average_reward = (
                action_reward_dict[action] * 1.0 / len(action_count_dict[action])
            )
            # Average proportion of particles
            average_visit = len(action_count_dict[action]) * 1.0 / len(index_particles)
            score = average_reward * lambd + average_visit
            goal = action_goal_dict[action]

            if max_score is None or max_score < score:
                max_score = score
                max_action = action
                max_goal = goal

        index_action += 1
        prev_index_particles = action_count_dict[max_action]
        # print(max_action, prev_index_particles)
        final_actions.append(max_action)
        final_goals.append(max_goal)

    # ipdb.set_trace()
    # If there is no action predicted but there were goals missing...
    #if len(final_actions) == 0:
        #print("No actions")
    if verbose:
        ipdb.set_trace()

    plan = final_actions
    subgoals = final_goals

    # ipdb.set_trace()
    # subgoals = [[None, None, None], [None, None, None]]
    # next_root, plan, subgoals = mp_run_mcts(root_nodes[0])
    next_root = None

    # ipdb.set_trace()
    # print('plan', plan)
    # if 'put' in plan[0]:
    #     ipdb.set_trace()
    if verbose:
        print("plan", plan)
        print("subgoal", subgoals)
    sample_id = None

    if sample_id is not None:
        res[sample_id] = plan
    else:

        return plan, next_root, subgoals


class MCTS_agent_particle_v2_instance:
    """
    MCTS for a single agent
    """

    def __init__(
        self,
        agent_id,
        char_index,
        max_episode_length,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        num_particles=20,
        recursive=False,
        num_samples=1,
        num_processes=1,
        comm=None,
        logging=False,
        logging_graphs=False,
        agent_params={},
        get_plan_states=False,
        get_plan_cost=False,
        seed=None,
    ):
        self.agent_type = "MCTS"
        self.verbose = False
        self.recursive = recursive

        # self.env = unity_env.env
        if seed is None:
            seed = random.randint(0, 100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.last_obs = None
        self.last_plan = None
        self.last_loc = {}
        self.failed_action = False

        self.agent_id = agent_id
        self.char_index = char_index

        self.sim_env = VhGraphEnv(n_chars=self.agent_id)
        self.sim_env.pomdp = True
        self.belief = None

        self.belief_params = agent_params["belief"]
        self.agent_params = agent_params
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.num_particles = num_particles
        self.get_plan_states = get_plan_states
        self.get_plan_cost = get_plan_cost

        self.previous_belief_graph = None
        self.verbose = False

        # self.should_close = True
        # if self.planner_params:
        #     if 'should_close' in self.planner_params:
        #         self.should_close = self.planner_params['should_close']

        self.mcts = None
        # MCTS_particles_v2(self.agent_id, self.char_index, self.max_episode_length,
        #                 self.num_simulation, self.max_rollout_steps,
        #                 self.c_init, self.c_base, agent_params=self.agent_params)

        self.particles = [None for _ in range(self.num_particles)]
        self.particles_full = [None for _ in range(self.num_particles)]
        self.have_asked = False

        # if self.mcts is None:
        #    raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph["edges"]:
            key = (edge["from_id"], edge["to_id"])
            if key not in edge_dict:
                edge_dict[key] = [edge["relation_type"]]
                new_edges.append(edge)
            else:
                if edge["relation_type"] not in edge_dict[key]:
                    edge_dict[key] += [edge["relation_type"]]
                    new_edges.append(edge)

        graph["edges"] = new_edges
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [
            node["id"] for node in graph["nodes"] if node["class_name"] == "character"
        ][0]
        edges = [edge for edge in graph["edges"] if edge["from_id"] == char_id]
        print("Character:")
        print(edges)
        print("---")

    def new_obs(obs, ignore_ids=None):
        curr_visible_ids = [node["id"] for node in obs["nodes"]]
        relations = {"ON": 0, "INSIDE": 1}
        num_relations = len(relations)
        if set(curr_visible_ids) != set(self.last_obs["ids"]):
            new_obs = True
        else:
            state_ids = np.zeros((len(curr_visible_ids), 4))
            edge_ids = np.zeros(
                (len(curr_visible_ids), len(curr_visible_ids), num_relations)
            )
            idnode2id = []
            for idnode, node in enumerate(nodes):
                idnode2id[node["id"]] = idnode
                state_ids[idnode, 0] = "OPEN" in node["states"]
                state_ids[idnode, 1] = "CLOSED" in node["states"]
                state_ids[idnode, 0] = "ON" in node["states"]
                state_ids[idnode, 1] = "OFF" in node["states"]
            for edge in node["edges"]:
                if edge["relation_type"] in relations.keys():
                    edge_id = relations[edge["relation_type"]]
                    from_id, to_id = (
                        idnode2id[edge["from_id"]],
                        idnode2id[edge["to_id"]],
                    )
                    edge_ids[from_id, to_id, edge_id] = 1

            if ignore_ids != None:
                # We will ignore some edges, for instance if we are grabbing an object
                self.last_obs["edges"][ignore_ids, :] = edge_ids[ignore_ids, :]
                self.last_obs["edges"][:, ignore_ids] = edge_ids[:, ignore_ids]

            if (
                state_ids != self.last_obs["state"]
                or edge_ids != self.last_obs["edges"]
            ):
                new_obs = True
                self.last_obs["state"] = state_ids
                self.last_obs["edges"] = edge_ids
        return new_obs

    def get_location_in_goal(self, obs, obj_id):
        curr_loc = [edge for edge in obs["edges"] if edge["from_id"] == obj_id]
        curr_loc += [edge for edge in obs["edges"] if edge["to_id"] == obj_id]
        print(curr_loc)
        curr_loc_on = [
            edge
            for edge in curr_loc
            if edge["relation_type"] == "ON" or "hold" in edge["relation_type"].lower()
        ]
        curr_loc_inside = [
            edge for edge in curr_loc if edge["relation_type"] == "INSIDE"
        ]
        if len(curr_loc_on) + len(curr_loc_inside) > 0:
            if len(curr_loc_on) > 0:
                if "hold" in curr_loc_on[0]["relation_type"].lower():
                    curr_loc_index = curr_loc_on[0]["from_id"]
                else:
                    curr_loc_index = curr_loc_on[0]["to_id"]
            else:
                curr_loc_index = curr_loc_inside[0]["to_id"]
            if len(curr_loc_on) > 1 or len(curr_loc_inside) > 1:
                print(curr_loc_on)
                print(curr_loc_inside)
                ipdb.set_trace()

        return curr_loc_index

    # xinyu: in_same_room is used to indicate whether the two agents are in the same room
    def get_action(
        self, obs, goal_spec, opponent_subgoal=None, length_plan=5, must_replan=True, 
        language=None, inquiry=False, in_same_room=True, modify_observation=False, modified_rooms=None
    ): #inquiry: whether agent is possible to ask for help; in same room: whether the agent is in same room with other or not; 
       #modify_observation: whether agent will receive manipulated gt graph, modified_rooms: list of rooms where obs will be false
        change_goal = False # indicate whether the agent should change its goal
        if modify_observation:
            obs = self.modify_observation(modified_rooms)
        
        if language is not None:
            print("Agent {} received message from {}".format(self.agent_id - 1, language.from_agent_id - 1))
            print(language.to_language())
            '''if type(language) == LanguageInquiry:
                if language.language_type == "location":
                    print("ask for location of {}".format(language.obj_name))
                elif language.language_type == "goal":
                    print("ask for goal to offer help")
                else:
                    ValueError("Language type not recognized")

            if type(language) == LanguageResponse:
                print("information of {} {} {}".format(language.language.split("_")[0], language.language.split("_")[1], language.language.split("_")[2]))'''

        # xinyu: the commutication about goals should always be started by agent 1, and the communication about location should always be started by agent 0,
        # Thus,  I commented the following code.
        # if self.agent_id == 1:
        #     inquiry = False
                
        language_to_be_sent = None

        if type(language) == LanguageInquiry:
            if language.language_type == "location":
                language_to_be_sent = language.generate_response(self.belief.sampled_graph, self.belief.edge_belief)
                #TODO: should we use observation here? or sampled graph

            
            elif language.language_type == "goal":
                language_to_be_sent = LanguageResponse(None,
                                                    None, 
                                                    None, 
                                                    None, 
                                                    goal_spec, 
                                                    self.agent_id, 
                                                    language.from_agent_id,
                                                    "goal")
            else:
                ValueError("Language type not recognized")
        # ipdb.set_trace()
        if len(goal_spec) == 0:
            ipdb.set_trace()

        # Create the particles
        # pdb.set_trace()

        self.belief.update_belief(obs)

        # when the agent is about give a response, it shouldn't start a inquiry
        if language_to_be_sent is not None:
            inquiry = False


        # TODO: maybe we will want to keep the previous belief graph to avoid replanning
        # self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})
        if inquiry and in_same_room and not self.have_asked:
            # agent 1 asks agent 2 for help
            if self.agent_id == 1:
                obj_seek = self.whether_to_ask(goal_spec, 0.05, "location")
                print("whether to ask", obj_seek)
                if obj_seek is not None:
                    language_to_be_sent = LanguageInquiry(obj_seek, 1, 2, "location")
                    self.have_asked = True 
        last_action = self.last_action
        last_subgoal = self.last_subgoal[0] if self.last_subgoal is not None else None
        subgoals = self.last_subgoal
        last_plan = self.last_plan

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose

        # If the current obs is the same as the last obs
        ignore_id = None

        should_replan = True

        goal_ids_all = []
        for goal_name, goal_val in goal_spec.items():
            if goal_val["count"] > 0:
                goal_ids_all += goal_val["grab_obj_ids"]

        goal_ids = [
            nodeobs["id"] for nodeobs in obs["nodes"] if nodeobs["id"] in goal_ids_all
        ]
        close_ids = [
            edge["to_id"]
            for edge in obs["edges"]
            if edge["from_id"] == self.agent_id
            and edge["relation_type"] in ["CLOSE", "INSIDE"]
        ]
        plan = []

        if last_plan is not None and len(last_plan) > 0:
            should_replan = False

            # If there is a goal object that was not there before
            next_id_interaction = []
            if len(last_plan) > 1:
                next_id_interaction.append(
                    int(last_plan[1].split("(")[1].split(")")[0])
                )

            new_observed_objects = (
                set(goal_ids)
                - set(self.last_obs["goal_objs"])
                - set(next_id_interaction)
            )
            # self.last_obs = {'goal_objs': goal_ids}
            if len(new_observed_objects) > 0:
                # New goal, need to replan
                should_replan = True
            else:

                visible_ids = {node["id"]: node for node in obs["nodes"]}
                curr_plan = last_plan

                first_action_non_walk = [
                    act for act in last_plan[1:] if "walk" not in act
                ]

                # If the first action other than walk is OPEN/CLOSE and the object is already open/closed...
                if len(first_action_non_walk):
                    first_action_non_walk = first_action_non_walk[0]
                    if "open" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:
                            if "OPEN" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS OPEN")
                    elif "close" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:

                            if "CLOSED" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS CLOSED")

                if (
                    "open" in last_plan[0]
                    or "close" in last_plan[0]
                    or "put" in last_plan[0]
                    or "grab" in last_plan[0]
                    or "touch" in last_plan[0]
                ):
                    if len(last_plan) == 1:
                        should_replan = True
                    else:
                        curr_plan = last_plan[1:]
                        subgoals = (
                            self.last_subgoal[1:]
                            if self.last_subgoal is not None
                            else None
                        )
                if (
                    "open" in curr_plan[0]
                    or "close" in curr_plan[0]
                    or "put" in curr_plan[0]
                    or "grab" in curr_plan[0]
                    or "touch" in curr_plan[0]
                ):
                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                    if not obj_id in close_ids or not obj_id in visible_ids:
                        should_replan = True

                next_action = not should_replan
                while next_action and "walk" in curr_plan[0]:

                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])

                    # If object is not visible, replan
                    if obj_id not in visible_ids:
                        should_replan = True
                        next_action = False
                    else:
                        if obj_id in close_ids:
                            if len(curr_plan) == 1:
                                should_replan = True
                                next_action = False
                            else:
                                curr_plan = curr_plan[1:]
                                subgoals = (
                                    subgoals[1:] if subgoals is not None else None
                                )
                        else:
                            # Keep with previous action
                            next_action = False

                if not should_replan:
                    plan = curr_plan

        obj_grab = -1
        curr_loc_index = -1

        self.last_obs = {"goal_objs": goal_ids}
        if self.failed_action:
            should_replan = True
            self.failed_action = False
        else:
            # obs = utils_env.inside_not_trans(obs)
            #if not should_replan and not must_replan:
                # If the location of the object you wanted to grab has changed
                #if last_subgoal is not None and "grab" in last_subgoal[0]:
                    #obj_grab = int(last_subgoal[0].split("_")[1])
                    #curr_loc_index = self.get_location_in_goal(obs, obj_grab)
                    #if obj_grab not in self.last_loc:
                    #    ipdb.set_trace()
                    #if curr_loc_index != self.last_loc[obj_grab]:
                        # The object I wanted to get now changed position, so I should replan
                        # self.last_loc = curr_loc_index
                        #should_replan = True

                # If you wanted to put an object but it is not in your hands anymore
                if last_subgoal is not None and "put" in last_subgoal[0]:
                    object_put = int(last_subgoal[0].split("_")[1])
                    hands_char = [
                        edge["to_id"]
                        for edge in obs["edges"]
                        if "hold" in edge["relation_type"].lower()
                        and edge["from_id"] == self.agent_id
                    ]
                    if object_put not in hands_char:
                        should_replan = True

        time1 = time.time()
        lg = ""
        if last_subgoal is not None and len(last_subgoal) > 0:
            lg = last_subgoal[0]
        if self.verbose:
            print(
                "-------- Agent {}: {} --------".format(
                    self.agent_id, "replan" if should_replan else "no replan"
                )
            )
        if should_replan or must_replan:
            # print("must_replan")
            # ipdb.set_trace()
            for particle_id, particle in enumerate(self.particles):
                belief_states = []
                obs_ids = [node["id"] for node in obs["nodes"]]

                # if True: #particle is None:
                if type(language) == LanguageResponse:
                    if language.language_type == "location":
                        new_graph = self.belief.update_graph_from_gt_graph(
                            obs, resample_unseen_nodes=True, update_belief=True, language_response=language
                        )
                    elif language.language_type == "goal":
                        new_graph = self.belief.update_graph_from_gt_graph(
                            obs, resample_unseen_nodes=True, update_belief=False
                        )
                        assert self.agent_id == 2
                        goal_spec = language.goal_spec # make agent 2's goal the same as agent 1's
                        change_goal = True

                        
                elif type(language) == LanguageInquiry:
                    new_graph = self.belief.update_graph_from_gt_graph(
                        obs, resample_unseen_nodes=True, update_belief=False
                    )
                else:
                    new_graph = self.belief.update_graph_from_gt_graph(
                        obs, resample_unseen_nodes=True, update_belief=False
                    )
                # print('new_graph:')
                # print([n['id'] for n in new_graph['nodes']])
                # print(
                #     'obs:',
                #     [edge for edge in obs['edges'] if 'HOLD' in edge['relation_type']],
                # )

                # print(
                #     'new_graph:',
                #     [
                #         edge
                #         for edge in new_graph['edges']
                #         if 'HOLD' in edge['relation_type']
                #     ],
                # )
                init_state = clean_graph(new_graph, goal_spec, self.mcts.last_opened)
                satisfied, unsatisfied = utils_env.check_progress2(
                    init_state, goal_spec
                )
                if "offer" in list(goal_spec.keys())[0]:
                    if self.verbose:
                        print("offer:")
                        print(satisfied)
                        print(unsatisfied)
                    # ipdb.set_trace()
                '''for edge in init_state["edges"]:
                    if edge["from_id"] == 457:
                        print(edge)'''
                init_vh_state = self.sim_env.get_vh_state(init_state)
                # print(colored(unsatisfied, "yellow"))
                self.particles[particle_id] = (
                    init_vh_state,
                    init_state,
                    satisfied,
                    unsatisfied,
                )
                # print(
                #     'init_state:',
                #     [
                #         edge
                #         for edge in init_state['edges']
                #         if 'HOLD' in edge['relation_type']
                #     ],
                # )

                self.particles_full[particle_id] = new_graph
            # print('-----')
            should_stop = False
            if self.agent_id == 2:
                # If agent 1 is grabbing an object, make sure that is not part of the plan
                new_goal_spec = copy.deepcopy(goal_spec)
                ids_grab_1 = [
                    edge["to_id"]
                    for edge in obs["edges"]
                    if edge["from_id"] == 1 and "hold" in edge["relation_type"].lower()
                ]
                if len(ids_grab_1) > 0:
                    for kgoal, elemgoal in new_goal_spec.items():
                        elemgoal["grab_obj_ids"] = [
                            ind
                            for ind in elemgoal["grab_obj_ids"]
                            if ind not in ids_grab_1
                        ]
                    should_stop = True
                goal_spec = new_goal_spec

            plan, root_node, subgoals = get_plan(
                self.mcts,
                self.particles,
                self.sim_env,
                nb_steps,
                goal_spec,
                last_plan,
                last_action,
                opponent_subgoal,
                length_plan=length_plan,
                verbose=self.verbose,
                num_process=self.num_processes,
            )

            if self.verbose:
                print("here")
                ipdb.set_trace()
            # update last_loc, we will store the location of the objects we are trying to grab
            # at the moment of planning, if something changes, then we will replan when time comes
            elems_grab = []
            if subgoals is not None:
                for goal in subgoals:
                    if goal is not None and goal[0] is not None and "grab" in goal[0]:
                        elem_grab = int(goal[0].split("_")[1])
                        elems_grab.append(elem_grab)
            self.last_loc = {}
            #if must_replan is False:
                #for goal_id in elems_grab:
                    #self.last_loc[goal_id] = self.get_location_in_goal(obs, goal_id) #what if goal is not visible in intitial observation

            # if len(plan) == 0 and self.agent_id == 1:
            #     ipdb.set_trace()

            if self.verbose:
                print(colored(plan[: min(len(plan), 10)], "cyan"))
            # ipdb.set_trace()
        # else:
        #     subgoals = [[None, None, None], [None, None, None]]
        # if len(plan) == 0 and not must_replan:
        #     ipdb.set_trace()
        #     print("Plan empty")
        #     raise Exception
        # print('-------- Plan {}: {}, {} ------------'.format(self.agent_id, lg, plan))
        if len(plan) > 0:
            action = plan[0]
            action = action.replace("[walk]", "[walktowards]")
        else:
            action = None
        if self.logging:
            info = {
                "plan": plan,
                "subgoals": subgoals,
                "belief": copy.deepcopy(self.belief.edge_belief),
                "belief_room": copy.deepcopy(self.belief.room_node),
            }

            if self.get_plan_states or self.get_plan_cost:
                plan_states = []
                plan_cost = []
                env = self.sim_env
                env.pomdp = True
                particle_id = 0
                vh_state = self.particles[particle_id][0]
                plan_states.append(vh_state.to_dict())
                for action_item in plan:

                    if self.get_plan_cost:
                        plan_cost.append(
                            env.compute_distance(vh_state, action_item, self.agent_id)
                        )

                    # if self.char_index == 1:
                    #     ipdb.set_trace()

                    success, vh_state = env.transition(
                        vh_state, {self.char_index: action_item}
                    )
                    vh_state_dict = vh_state.to_dict()
                    # print(action_item, [edge['to_id'] for edge in vh_state_dict['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] == 'INSIDE'])
                    plan_states.append(vh_state_dict)

                if self.get_plan_states:
                    info["plan_states"] = plan_states
                if self.get_plan_cost:
                    info["plan_cost"] = plan_cost

            if self.logging_graphs:
                info.update({"obs": obs["nodes"].copy()})
        else:
            info = {"plan": plan, "subgoals": subgoals}

        self.last_action = action
        self.last_subgoal = (
            subgoals if subgoals is not None and len(subgoals) > 0 else None
        )
        self.last_plan = plan
        # print(info['subgoals'])
        # print(action)
        time2 = time.time()
        # print("Time: ", time2 - time1)
        if self.verbose:
            print("Replanning... ", should_replan or must_replan)
        if should_replan:
            if self.verbose:
                print(
                    "Agent {} did replan: ".format(self.agent_id),
                    self.last_loc,
                    obj_grab,
                    curr_loc_index,
                    plan,
                )
            # if len(plan) == 0 and self.agent_id == 1:
            #     ipdb.set_trace()

        else:
            if self.verbose:
                print(
                    "Agent {} not replan: ".format(self.agent_id),
                    self.last_loc,
                    obj_grab,
                    curr_loc_index,
                    plan,
                )

        if action is not None and "grab" in action:
            if self.agent_id == 2:
                grab_id = int(action.split()[2][1:-1])
                grabbed_obj = [
                    edge
                    for edge in obs["edges"]
                    if edge["to_id"] == grab_id
                    and "hold" in edge["relation_type"].lower()
                ]
                if len(grabbed_obj):
                    ipdb.set_trace()
            # if len([edge for edge in obs['edges'] if edge['from_id'] == 369 and edge['to_id'] == 103]) > 0:
            #     print("Bad plan")
            #     ipdb.set_trace()

        return action, info, language_to_be_sent, change_goal
    
    def whether_to_ask(self, goal_spec=None, boundary=None, language_type=None): #decide whether to ask for help
        if language_type == "location":
            uncertain_objects = []
            for goal_name, info in goal_spec.items():
                record = []
                for obj_id in info['grab_obj_ids']:
                    if obj_id not in self.belief.edge_belief.keys():
                        continue
                    distribution = scipy.special.softmax(self.belief.edge_belief[obj_id]["INSIDE"][1][:])
                    distribution = distribution[distribution > 0]
                    if distribution.shape[0] == 1:
                        record.append(0.0)
                        continue
                    entropy = -np.sum(distribution * np.log2(distribution))
                    ratio = entropy / np.log2(distribution.shape[0])
                    record.append(ratio)
                    distribution = scipy.special.softmax(self.belief.edge_belief[obj_id]["ON"][1][:])
                    distribution = distribution[distribution > 0]
                    if distribution.shape[0] == 1:
                        index = np.argmax(self.belief.edge_belief[obj_id]["ON"][1][:])
                        if index == 0:
                            continue
                        record.append(0.0)
                        continue
                    entropy = -np.sum(distribution * np.log2(distribution))
                    ratio = entropy / np.log2(distribution.shape[0])
                    record.append(ratio)
                record.sort()
                try:
                    if record[info["count"] - 1] >= boundary:
                        uncertain_objects.append(goal_name.split('_')[1])
                except IndexError:
                    continue
            if len(uncertain_objects) == 0:
                return None
            return uncertain_objects
        elif language_type == "goal":
            return True
        else:
            ValueError("Language type not recognized")



    def reset(
        self,
        observed_graph,
        gt_graph,
        task_goal,
        seed=0,
        simulator_type="python",
        is_alice=False,
    ):
        self.have_asked = False
        self.last_action = None
        self.last_subgoal = None
        self.failed_action = False
        self.init_gt_graph = gt_graph
        """TODO: do no need this?"""
        # if 'waterglass' not in [node['class_name'] for node in self.init_gt_graph['nodes']]:
        #    ipdb.set_trace()
        self.belief = belief.Belief(
            gt_graph,
            agent_id=self.agent_id,
            seed=seed,
            belief_params=self.belief_params,
        )
        self.sim_env.reset(gt_graph)
        add_bp = self.num_processes == 0
        self.mcts = MCTS_particles_v2_instance(
            gt_graph,
            self.agent_id,
            self.char_index,
            self.max_episode_length,
            self.num_simulation,
            self.max_rollout_steps,
            self.c_init,
            self.c_base,
            seed=seed,
            agent_params=self.agent_params,
            add_bp=add_bp,
        )

        # self.mcts.should_close = self.should_close
    
    def modify_observation(self, room_list=[]): #function for generating a false graph at the beginning for agent 1
        obs = self.init_gt_graph #set as ground truth graph in the beginning
        for node in obs['nodes']:
            if node["class_name"] == "coffeemaker": #for debug
                node["states"].append('CLOSED')
        '''node_to_remove = []
        edge_to_remove = []
        for node in obs['nodes']:
            if node["class_name"] == "bookshelf": #for debug
                removed = node["id"]
                node_to_remove.append(node)
                for edge in obs["edges"]:
                    if edge["from_id"] == removed or edge["to_id"] == removed:
                        edge_to_remove.append(edge)
                        if edge["relation_type"] in ["INSIDE", "ON"]:
                            node_to_remove.append([node for obs["nodes"] if node["id"] == edge["from_id"] or node["id"] == edge["to_id"]])
        for node in node_to_remove:
            obs["nodes"].remove(node)
        for edge in edge_to_remove:
            obs["edges"].remove(edge)
        for edge in obs["edges"]:
            if edge["from_id"] == 140 or edge["to_id"] == 140:
                print(edge)'''
        id2node = {node["id"] : node for node in obs["nodes"]}
        id2room = {}
        room_names = ["bathroom", "bedroom", "kitchen", "livingroom"]
        for id, node in id2node.items():
            if node["class_name"] in room_names:
                continue
            for edge in obs["edges"]:
                if edge["from_id"] == id and edge["relation_type"] == "INSIDE" and id2node[edge["to_id"]]["class_name"] in room_names:
                    id2room[id] = edge["to_id"]
        container_list = self.belief.edge_belief[list(self.belief.edge_belief.keys())[0]]["INSIDE"][0][1:]
        surface_list = self.belief.edge_belief[list(self.belief.edge_belief.keys())[0]]["ON"][0][1:]
        for obj_id in self.belief.edge_belief.keys():
            for container in self.belief.edge_belief[obj_id]["INSIDE"][0][1:]:
                if container not in container_list:
                    container_list.append(container)
        for obj_id in self.belief.edge_belief.keys():
            for container in self.belief.edge_belief[obj_id]["ON"][0][1:]:
                if container not in surface_list:
                    surface_list.append(container) #formulate list of containers and surfaces
        for room in room_list: #walk through all rooms intend to have wrong information
            for node in obs["nodes"]:
                if node["class_name"] == room:
                    room_id = node["id"]
            room_obj_list = {}
            for edge in obs["edges"]:
                if edge["to_id"] == room_id and edge["relation_type"] == "INSIDE":
                    if id2node[edge["from_id"]]["class_name"] not in ["floor", "window", "ceiling", "curtain", "wall"]:
                        room_obj_list[edge["from_id"]] = edge #all object id inside the room
            for node in obs["nodes"]:
                if node["id"] not in room_obj_list.keys() and node["id"] not in ["floor", "window", "ceiling", "curtain", "wall"]:
                    for edge in obs["edges"]:
                        if edge["from_id"] == node["id"] and edge["to_id"] in room_obj_list.keys():
                            room_obj_list[node["id"]] = edge #some object will not have edge to room, but their containers have
            for obj in room_obj_list.keys():
                temp = copy.deepcopy(obs["edges"])
                for edge in temp:
                    if edge["from_id"] == obj and edge["to_id"] in container_list and edge["relation_type"] == "INSIDE":
                        for edge1 in obs["edges"]:
                            if edge1["from_id"] == obj: #remove old edges
                                obs["edges"].remove(edge1)
                        container_list.remove(edge["to_id"]) #can't be at old place
                        n = random.random()
                        if n > len(container_list) / len(container_list) + len(surface_list):
                            index = random.randint(0, len(surface_list) - 1)
                            new_position = surface_list[index]
                            obs["edges"].append({"from_id": obj, "to_id": new_position, "relation_type": "ON"})
                            obs["edges"].append({"from_id": obj, "to_id": id2room[new_position], "relation_type": "INSIDE"}) #for on relationship, add inside room
                        else:
                            index = random.randint(0, len(container_list) - 1)
                            new_position = container_list[index]
                            obs["edges"].append({"from_id": obj, "to_id": new_position, "relation_type": "INSIDE"}) #inside container will not have inside room
                        container_list.append(edge["to_id"])
                        break
                    if edge["from_id"] == obj and edge["to_id"] in surface_list and edge["relation_type"] == "ON":
                        for edge1 in obs["edges"]:
                            if edge1["from_id"] == obj: #remove old edges
                                obs["edges"].remove(edge1)
                        surface_list.remove(edge["to_id"])
                        n = random.random()
                        if n > len(surface_list) / len(surface_list) + len(container_list):
                            index = random.randint(0, len(container_list) - 1)
                            new_position = container_list[index]
                            obs["edges"].append({"from_id": obj, "to_id": new_position, "relation_type": "INSIDE"})
                        else:
                            index = random.randint(0, len(surface_list) - 1)
                            new_position = surface_list[index]
                            obs["edges"].append({"from_id": obj, "to_id": new_position, "relation_type": "ON"})
                            obs["edges"].append({"from_id": obj, "to_id": id2room[new_position], "relation_type": "INSIDE"})
                        surface_list.append(edge["to_id"])
                        break
        return obs
                            


            
             
