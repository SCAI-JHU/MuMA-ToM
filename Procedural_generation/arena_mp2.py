import random
import cv2
import logging
import utils_exception
import copy
import numpy as np
from tqdm import tqdm
import time
import ipdb
import traceback
import atexit
import utils_environment as utils
from typing import List, Optional
from language import Language, LanguageInquiry, LanguageResponse
# @ray.remote
class ArenaMP(object):
    def __init__(
        self,
        max_number_steps,
        arena_id,
        environment_fn,
        agent_fn,
        use_sim_agent=False,
        save_belief=True,
    ):
        self.agents = []
        self.sim_agents = []
        self.save_belief = save_belief
        self.env_fn = environment_fn
        self.agent_fn = agent_fn
        self.arena_id = arena_id
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.use_sim_agent = use_sim_agent
        print("Init Env")
        self.env = environment_fn(arena_id)
        assert self.env.num_agents == len(
            agent_fn
        ), "The number of agents defined and the ones in the env defined mismatch"
        for agent_type_fn in agent_fn:
            self.agents.append(agent_type_fn(arena_id, self.env))
            if self.use_sim_agent:
                self.sim_agents.append(agent_type_fn(arena_id, self.env))

        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.saved_info = None
        atexit.register(self.close)
        self.language_infos: List[Optional[Language]] = [None for _ in range(self.num_agents)]



    def close(self):
        #print(traceback.print_exc())

        #print(traceback.print_stack())

        self.env.close()

    def get_port(self):
        return self.env.port_number

    def reset(self, task_id=None):
        ob = None
        count = 0
        while ob is None and count < 10:
            try:
                ob = self.env.reset(task_id=task_id)
                count += 1
            except Exception:
                return None
        if ob is None:
            return None
        print(ob.keys(), self.num_agents)

        for it, agent in enumerate(self.agents):
            self.language_infos[it] = None
            if "MCTS" in agent.agent_type or "Random" in agent.agent_type:
                agent.reset(
                    ob[it], self.env.full_graph, self.env.task_goal, seed=agent.seed
                )
                if self.use_sim_agent:
                    self.sim_agents[it].reset(
                        ob[it],
                        self.env.full_graph,
                        self.env.task_goal,
                        seed=self.agents[1 - it].seed,
                    )
            else:
                agent.reset(self.env.full_graph)
                if self.use_sim_agent:
                    self.sim_agents.reset(self.env.full_graph)
        return ob

    def set_weigths(self, epsilon, weights):
        for agent in self.agents:
            if "RL" in agent.agent_type:
                agent.epsilon = epsilon
                agent.actor_critic.load_state_dict(weights)

    def get_actions(
        self,
        obs,
        action_space=None,
        true_graph=False,
        length_plan=5,
        must_replan=None,
        agent_id=None,
        inferred_goal=None,
        opponent_subgoal=None,
        tp=True,
        inquiry=False,
        modify_graph=False,
        room_list=None
    ):
        # ipdb.set_trace()
        dict_actions, dict_info = {}, {}
        language_info = {}
        op_subgoal = {0: None, 1: None}
        # pdb.set_trace()

        for it, agent in enumerate(self.agents):
            if agent_id is not None and it != agent_id:
                continue
            if inferred_goal is None:
                if self.task_goal is None:
                    goal_spec = self.env.get_goal2(
                        self.env.task_goal[it], self.env.agent_goals[it]
                    )

                else:
                    goal_spec = self.env.get_goal2(
                        self.task_goal[it], self.env.agent_goals[it]
                    )
            else:
                goal_spec = self.env.get_goal2(inferred_goal, self.env.agent_goals[it])
            # ipdb.set_trace()
            if agent.agent_type in ["MCTS", "Random"]:

                language = self.language_infos[it]
                # opponent_subgoal = None
                # if agent.recursive:
                #     opponent_subgoal = self.agents[1 - it].last_subgoal
                # ipdb.set_trace()
                in_same_room = self.whether_in_same_room(self.agents)
                print("in same room:", in_same_room)
                #in_same_room = True
                
                dict_actions[it], dict_info[it], language_rsps, change_goal = agent.get_action(
                    obs[it],
                    goal_spec,
                    opponent_subgoal,
                    length_plan=length_plan,
                    must_replan=True if must_replan is None else must_replan[it], # TODO: already modified this
                    language=language,
                    inquiry=inquiry,
                    in_same_room=in_same_room,
                    modify_observation=modify_graph,
                    modified_rooms=room_list
                )
                language_info[it] = language_rsps
                if language is not None:
                    self.language_infos[language.to_agent_id - 1] = None
                if language_rsps is not None:
                    self.language_infos[language_rsps.to_agent_id - 1] = language_rsps

                # set the agent 2 goal to be the same as agent 1    
                if change_goal:
                    if inferred_goal is None:
                        if self.task_goal is not None:
                            self.task_goal[agent.agent_id-1] = self.task_goal[(agent.agent_id-2) % 2]
                            self.env.agents_goals[agent.agent_id-1] = self.env.agents_goals[(agent.agent_id-2) % 2]
                        else:
                            self.env.task_goal[agent.agent_id-1] = self.env.task_goal[(agent.agent_id-2) % 2]
                            self.env.agent_goals[agent.agent_id-1] = self.env.agent_goals[(agent.agent_id-2) % 2]
                    else:
                        inferred_goal[agent.agent_id-1] = inferred_goal[(agent.agent_id-2) % 2]
                        self.env.agent_goals[agent.agent_id-1] = self.env.agent_goals[(agent.agent_id-2) % 2]

                if tp is True and dict_actions[it] is not None:
                    dict_actions[it] = dict_actions[it].replace("walktowards", "walk")

                ind = it
                curr_obs = obs
                selected_actions = dict_actions

                # if selected_actions[ind] is not None and 'grab' in selected_actions[ind] and '369' in selected_actions[ind]:
                #    if len([edge for edge in curr_obs[ind]['edges'] if edge['from_id'] == 369 and edge['to_id'] == 103]) > 0:
                #        print("Bad plan")
                #        ipdb.set_trace()

            elif "RL" in agent.agent_type:
                if "MCTS" in agent.agent_type or "Random" in agent.agent_type:
                    if true_graph:
                        full_graph = self.env.get_graph()
                    else:
                        full_graph = None
                    dict_actions[it], dict_info[it] = agent.get_action(
                        obs[it],
                        goal_spec,
                        action_space_ids=action_space[it],
                        full_graph=full_graph,
                    )

                else:
                    # RL_RL agemt
                    dict_actions[it], dict_info[it] = agent.get_action(
                        obs[it], self.task_goal, action_space_ids=action_space[it]
                    )

        return dict_actions, dict_info, language_info

    def pred_actions(
        self,
        obs,
        action_space=None,
        true_graph=False,
        length_plan=5,
        must_replan=None,
        agent_id=None,
        inferred_goal=None,
        opponent_subgoal=None,
    ):
        # ipdb.set_trace()
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}
        # pdb.set_trace()
        # ipdb.set_trace()
        for it, agent in enumerate(self.sim_agents):
            if agent_id is not None and it != agent_id:
                continue
            if inferred_goal is None:
                if self.task_goal is None:
                    goal_spec = self.env.get_goal2(
                        self.env.task_goal[it], self.env.agent_goals[it]
                    )

                else:
                    goal_spec = self.env.get_goal2(
                        self.task_goal[it], self.env.agent_goals[it]
                    )
            else:
                goal_spec = self.env.get_goal2(inferred_goal, self.env.agent_goals[it])
            # ipdb.set_trace()
            if agent.agent_type in ["MCTS", "Random"]:
                # opponent_subgoal = None
                # if agent.recursive:
                #     opponent_subgoal = self.agents[1 - it].last_subgoal

                # agent.last_subgoal = None#[self.agents[it].last_subgoal[0]]
                # agent.last_plan = None#[self.agents[it].last_plan[0]]

                dict_actions[it], dict_info[it] = agent.get_action(
                    obs[it],
                    goal_spec,
                    opponent_subgoal,
                    length_plan=length_plan,
                    must_replan=False if must_replan is None else must_replan[it],
                )

            elif "RL" in agent.agent_type:
                if "MCTS" in agent.agent_type or "Random" in agent.agent_type:
                    if true_graph:
                        full_graph = self.env.get_graph()
                    else:
                        full_graph = None
                    dict_actions[it], dict_info[it] = agent.get_action(
                        obs[it],
                        goal_spec,
                        action_space_ids=action_space[it],
                        full_graph=full_graph,
                    )

                else:
                    # RL_RL agemt
                    dict_actions[it], dict_info[it] = agent.get_action(
                        obs[it], self.task_goal, action_space_ids=action_space[it]
                    )
        return dict_actions, dict_info

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

    def rollout_reset(
        self, logging=False, record=False, episode_id=None, is_train=True, goals=None
    ):
        try:
            res = self.rollout(
                logging, record, episode_id=episode_id, is_train=is_train, goals=goals
            )
            return res
        except:
            self.env.close()
            self.env = self.env_fn(self.arena_id)

            for agent in self.agents:
                if "RL" in agent.agent_type:
                    prev_eps = agent.epsilon
                    prev_weights = agent.actor_critic.state_dict()

            self.agents = []
            for agent_type_fn in self.agent_fn:
                self.agents.append(agent_type_fn(self.arena_id, self.env))

            self.set_weigths(prev_eps, prev_weights)
            return self.rollout(
                logging, record, episode_id=episode_id, is_train=is_train, goals=goals
            )

    def rollout(
        self, logging=0, record=False, episode_id=None, is_train=True, goals=None
    ):
        t1 = time.time()
        print("rollout", episode_id, is_train)
        if episode_id is not None:
            self.reset(episode_id)
        else:
            self.reset()

        t2 = time.time()
        t_reset = t2 - t1
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        agent_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []

        if goals is not None:
            self.task_goal = goals
        else:
            self.task_goal = None

        if logging > 0:
            info_rollout["pred_goal"] = []
            info_rollout["pred_close"] = []
            info_rollout["gt_goal"] = []
            info_rollout["gt_close"] = []
            info_rollout["mask_nodes"] = []

        if logging > 1:
            info_rollout["step_info"] = []
            info_rollout["action"] = {0: [], 1: []}
            info_rollout["script"] = []
            info_rollout["graph"] = []
            info_rollout["action_space_ids"] = []
            info_rollout["visible_ids"] = []
            info_rollout["action_tried"] = []
            info_rollout["predicate"] = []
            info_rollout["reward"] = []
            info_rollout["goals_finished"] = []
            info_rollout["obs"] = []

        rollout_agent = {}

        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            if "RL" in agent.agent_type:
                rollout_agent[agent_id] = []

        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec[0]
            goal_class = [elem_name.split("_")[1] for elem_name in list(pred.keys())]
            id2node = {node["id"]: node for node in init_graph["nodes"]}
            info_goals = []
            info_goals.append(
                [
                    node
                    for node in init_graph["nodes"]
                    if node["class_name"] in goal_class
                ]
            )
            ids_target = [
                node["id"]
                for node in init_graph["nodes"]
                if node["class_name"] in goal_class
            ]
            info_goals.append(
                [
                    (
                        id2node[edge["to_id"]]["class_name"],
                        edge["to_id"],
                        edge["relation_type"],
                        edge["from_id"],
                    )
                    for edge in init_graph["edges"]
                    if edge["from_id"] in ids_target
                ]
            )
            info_rollout["target"] = [pred, info_goals]

        agent_id = [
            id
            for id, enum_agent in enumerate(self.agents)
            if "RL" in enum_agent.agent_type
        ][0]
        reward_step = 0
        prev_reward_step = 0
        curr_num_steps = 0
        prev_reward = 0
        init_step_agent_info = {}
        local_rollout_actions = []
        if not is_train:
            pbar = tqdm(total=self.max_episode_length)
        while (
            not done
            and nb_steps < self.max_episode_length
            and agent_steps < self.max_number_steps
        ):
            (obs, reward, done, env_info), agent_actions, agent_info = self.step(
                true_graph=is_train
            )
            step_failed = env_info["failed_exec"]
            if step_failed:
                print("FAILING in task")
                print(agent_actions)
                print(local_rollout_actions)
                print("----")
            # print(agent_actions[agent_id], reward)
            local_rollout_actions.append(agent_actions[0])
            if not is_train:
                pbar.update(1)
            if logging:
                curr_graph = env_info["graph"]
                agentindex = self.agents[agent_id].agent_id
                observed_nodes = agent_info[agent_id]["visible_ids"]
                # pdb.set_trace()
                node_id = [
                    node["bounding_box"]
                    for node in obs[agent_id]["nodes"]
                    if node["id"] == agentindex
                ][0]
                edges_char = [
                    (
                        id2node[edge["to_id"]]["class_name"],
                        edge["to_id"],
                        edge["relation_type"],
                    )
                    for edge in curr_graph["edges"]
                    if edge["from_id"] == agentindex and edge["to_id"] in observed_nodes
                ]

                if logging > 0:
                    if "pred_goal" in agent_info[agent_id].keys():
                        info_rollout["pred_goal"].append(
                            agent_info[agent_id]["pred_goal"]
                        )
                        info_rollout["pred_close"].append(
                            agent_info[agent_id]["pred_close"]
                        )
                        info_rollout["gt_goal"].append(agent_info[agent_id]["gt_goal"])
                        info_rollout["gt_close"].append(
                            agent_info[agent_id]["gt_close"]
                        )
                        info_rollout["mask_nodes"].append(
                            agent_info[agent_id]["mask_nodes"]
                        )

                if logging > 1:
                    info_rollout["step_info"].append((node_id, edges_char))
                    info_rollout["script"].append(agent_actions[agent_id])
                    info_rollout["goals_finished"].append(env_info["satisfied_goals"])
                    info_rollout["finished"] = env_info["finished"]

                    # pdb.set_trace()
                    for agenti in range(len(self.agents)):
                        info_rollout["action"][agenti].append(agent_actions[agenti])
                        info_rollout["obs"].append(agent_info[agenti]["obs"])

                    info_rollout["action_tried"].append(
                        agent_info[agent_id]["action_tried"]
                    )
                    if "predicate" in agent_info[agent_id]:
                        info_rollout["predicate"].append(
                            agent_info[agent_id]["predicate"]
                        )
                    info_rollout["graph"].append(curr_graph)
                    info_rollout["action_space_ids"].append(
                        agent_info[agent_id]["action_space_ids"]
                    )
                    info_rollout["visible_ids"].append(
                        agent_info[agent_id]["visible_ids"]
                    )
                    info_rollout["reward"].append(reward)

            nb_steps += 1
            curr_num_steps += 1
            diff_reward = reward - prev_reward
            prev_reward = reward
            reward_step += diff_reward
            if "bad_predicate" in agent_info[agent_id]:
                reward_step -= 0.2
                # pdb.set_trace()

            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += diff_reward
                # action_dict[agent_index] = agent_info[agent_index]['action']

            if record:
                actions.append(agent_actions)

            # append to memory
            if is_train:
                for agent_id in range(self.num_agents):
                    if (
                        "RL" == self.agents[agent_id].agent_type
                        or self.agents[agent_id].agent_type == "RL_MCTS"
                        and "mcts_action" not in agent_info[agent_id]
                    ):
                        init_step_agent_info[agent_id] = agent_info[agent_id]

                    # If this is the end of the action
                    if (
                        "RL" == self.agents[agent_id].agent_type
                        or self.agents[agent_id].agent_type == "RL_MCTS"
                        and self.agents[agent_id].action_count == 0
                    ):
                        agent_steps += 1
                        state = init_step_agent_info[agent_id]["state_inputs"]
                        policy = [
                            log_prob.data
                            for log_prob in init_step_agent_info[agent_id]["probs"]
                        ]
                        action = agent_info[agent_id]["actions"]
                        rewards = reward_step
                        entropy_action.append(
                            -(
                                (
                                    init_step_agent_info[agent_id]["probs"][0] + 1e-9
                                ).log()
                                * init_step_agent_info[agent_id]["probs"][0]
                            )
                            .sum()
                            .item()
                        )
                        entropy_object.append(
                            -(
                                (
                                    init_step_agent_info[agent_id]["probs"][1] + 1e-9
                                ).log()
                                * init_step_agent_info[agent_id]["probs"][1]
                            )
                            .sum()
                            .item()
                        )
                        observation_space.append(
                            init_step_agent_info[agent_id]["num_objects"]
                        )
                        action_space.append(
                            init_step_agent_info[agent_id]["num_objects_action"]
                        )
                        last_agent_info = init_step_agent_info

                        rollout_agent[agent_id].append(
                            (
                                self.env.task_goal[agent_id],
                                state,
                                policy,
                                action,
                                rewards,
                                curr_num_steps,
                                1,
                            )
                        )
                        prev_reward_step = 0
                        reward_step = 0
                        curr_num_steps = 0

        # pdb.set_trace()
        if not is_train:
            pbar.close()
        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info["finished"]

        info_rollout["success"] = success_r_all[0]
        info_rollout["nsteps"] = nb_steps
        info_rollout["epsilon"] = self.agents[agent_id].epsilon
        info_rollout["entropy"] = (entropy_action, entropy_object)
        info_rollout["observation_space"] = np.mean(observation_space)
        info_rollout["action_space"] = np.mean(action_space)
        info_rollout["t_reset"] = t_reset
        info_rollout["t_steps"] = t_steps

        # pdb.set_trace()
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info["finished"]

        info_rollout["env_id"] = self.env.env_id
        info_rollout["goals"] = list(self.env.task_goal[0].keys())
        # padding
        # TODO: is this correct? Padding that is valid?

        # Rollout max
        # max_length_batchmem = self.max_episode_length
        if is_train:
            while nb_steps < self.max_number_steps:
                nb_steps += 1
                for agent_id in range(self.num_agents):
                    if "RL" in self.agents[agent_id].agent_type:
                        state = last_agent_info[agent_id]["state_inputs"]
                        if "edges" in obs.keys():
                            pdb.set_trace()
                        policy = [
                            log_prob.data
                            for log_prob in last_agent_info[agent_id]["probs"]
                        ]
                        action = last_agent_info[agent_id]["actions"]
                        # rewards = reward
                        rollout_agent[agent_id].append(
                            (
                                self.env.task_goal[agent_id],
                                state,
                                policy,
                                action,
                                0,
                                0,
                                0,
                            )
                        )

        return c_r_all, info_rollout, rollout_agent
    
    def whether_in_same_room(self, agents):
        full_graph = self.env.get_graph()
        all_in_same_room = True
        room_id = None
        for agent in agents:
            new_room_id = [
            edge["to_id"]
            for edge in full_graph["edges"]
            if edge["from_id"] == agent.agent_id and edge["relation_type"] == "INSIDE"
        ][0]
            if room_id is None:
                room_id = new_room_id
            else:
                if room_id != new_room_id:
                    all_in_same_room = False
                    break
        return all_in_same_room

    def step(self, true_graph=False, inquiry=False, modify_graph=False, room_list=None):

        if self.env.steps == 0:
            pass
            # self.env.changed_graph = True
        obs = self.env.get_observations()

        action_space = self.env.get_action_space()
        dict_actions, dict_info, language = self.get_actions(
            obs, action_space, true_graph=true_graph, inquiry=inquiry, modify_graph=modify_graph, room_list=room_list
        )
        print("MCTS returned", dict_actions)

        try:
            step_info = self.env.step(dict_actions)
            # for agent_id, agent in enumerate(self.agents):
            #    agent.step(dict_actions[agent_id])

        except:
            raise utils_exception.UnityException
        return step_info, dict_actions, dict_info, language

    def step_given_action(self, actionss, true_graph=False):
        for i in range(len(self.agents)):
            self.agents[i].failed_action = False
        try:
            step_info = self.env.step(actionss)
        except:
            raise utils_exception.UnityException

        script_exec = step_info[-1]["executed_script"]
        for index_agent in actionss:
            if index_agent not in script_exec:
                self.agents[0].failed_action = True
        return step_info

    def run(self, random_goal=False, pred_goal=None, save_img=None):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """
        self.task_goal = copy.deepcopy(self.env.task_goal)
        if len(self.task_goal[0].keys()) == 0 or len(self.task_goal[1].keys()) == 0:
            print("episode {} have improper goal".format(self.env.task_id))
            return False, 0, {"obs": []}
        for goal, _ in self.task_goal[0].items():
            if "donut" in goal:
                print("episode {} failed".format(self.env.task_id))
                return False, 0, {"obs": []}
        for goal, _ in self.task_goal[1].items():
            if "donut" in goal:
                print("episode {} failed".format(self.env.task_id))
                return False, 0, {"obs": []}
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        saved_info = {
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "task_name": self.env.task_name,
            "gt_goals": self.env.task_goal[0],
            "goals": self.task_goal,
            "action": {0: [], 1: []},
            "plan": {0: [], 1: []},
            "finished": None,
            "init_unity_graph": self.env.init_graph,
            "goals_finished": [],
            "belief": {0: [], 1: []},
            "belief_room": {0: [], 1: []},
            "belief_graph": {0: [], 1: []},
            "graph": [self.env.init_unity_graph],
            "obs": [],
            "language": {0: [], 1: []},
            "language_object": {0: [], 1: []},
            "have_belief": False,
            "false_belief_rooms": []
        }
        saved_info["have_belief"] = True
        num_nomove = 0
        success = False
        num_failed = 0
        num_repeated = 0
        prev_action_0 = None
        prev_action_1 = None
        self.saved_info = saved_info
        step = 0
        prev_agent_position = np.array([0, 0, 0]).astype(np.float32)
        while True:
            step += 1
            if save_img is not None:
                img_info = {"image_width": 224, "image_height": 224}
                obs = self.env.get_observation(0, "image", info=img_info)
                cv2.imwrite("{}/img_{:04d}.png".format(save_img, step), obs)
            if step == 1 and saved_info["have_belief"]:
                (obs, reward, done, infos), actions, agent_info, language = self.step(modify_graph=True, room_list=[])
            else:
                (obs, reward, done, infos), actions, agent_info, language = self.step()
            # ipdb.set_trace()
            new_agent_position = np.array(
                list(infos["graph"]["nodes"][0]["bounding_box"]["center"])
            ).astype(np.float32)
            distance = np.linalg.norm(new_agent_position - prev_agent_position)
            step_failed = infos["failed_exec"]
            if actions[0] == prev_action_0 and actions[1] == prev_action_1:
                num_repeated += 1
                if distance < 0.3:
                    num_nomove += 1
            else:
                prev_action_0 = actions[0]
                prev_action_1 = actions[1]
                num_repeated = 0
                num_nomove = 0
            if step_failed:
                num_failed += 1
            else:
                num_failed = 0
            if num_failed > 10 or num_repeated > 20 or num_nomove > 5:
                print("Many failures", num_failed, num_repeated)
                return False, self.env.steps, saved_info
                # logging.info("Many failures")
                raise utils_exception.ManyFailureException

            print("\nAgent Step:")
            print("----------")
            # print("Goals:", self.env.task_goal)
            print("Action: ", actions, new_agent_position)
            prev_agent_position = new_agent_position
            #logging.info(" | ".join(actions.values()))
            print("Plan of agent 0:", agent_info[0]["plan"][:4])

            if self.num_agents == 2:
                print("Plan of agent 1:", agent_info[1]["plan"][:4])
                
            print("----------")
            success = infos["finished"]
            if "satisfied_goals" in infos:
                saved_info["goals_finished"].append(infos["satisfied_goals"])
            for agent_id, action in actions.items():
                saved_info["action"][agent_id].append(action)

            if "graph" in infos:
                saved_info["graph"].append(infos["graph"])

            for agent_id, info in agent_info.items():
                # if 'belief_graph' in info:
                #    saved_info['belief_graph'][agent_id].append(info['belief_graph'])
                if self.save_belief:
                    if "belief_room" in info:
                        saved_info["belief_room"][agent_id].append(info["belief_room"])
                    if "belief" in info:
                        saved_info["belief"][agent_id].append(info["belief"])
                if "plan" in info:
                    saved_info["plan"][agent_id].append(info["plan"][:3])
                if "obs" in info:
                    # print("TEST", len(info['obs']), len(saved_info['graph'][-2]['nodes']))
                    saved_info["obs"].append([node["id"] for node in info["obs"]])
                    # print([node['states'] for node in info['obs'] if node['id'] == 103])
                    # ipdb.set_trace()
                # if len(saved_info['obs']) > 1 and set(saved_info['obs'][0]) != set(saved_info['obs'][1]):
                #    ipdb.set_trace()
            saved_info["language_object"][0].append(language[0])
            saved_info["language_object"][1].append(language[1])
            if language[0] is not None:
                saved_info["language"][0].append(language[0].to_language(mode="natural"))
            else:
                saved_info["language"][0].append(None)
            if language[1] is not None:
                saved_info["language"][1].append(language[1].to_language(mode="natural"))
            else:
                saved_info["language"][1].append(None)

            # ipdb.set_trace()
            if done:
                break
            self.saved_info = saved_info

        saved_info["obs"].append([node["id"] for node in obs[0]["nodes"]])
        # saved_info['obs'].append()

        saved_info["finished"] = success
        self.saved_info = saved_info
        return success, self.env.steps, saved_info
            
            

        