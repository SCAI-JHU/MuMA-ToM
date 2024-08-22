import sys
import shutil
import os
print(sys.path)
print(os.getcwd())
import logging
import traceback
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2, MCTS_agent_particle, MCTS_agent_particle_v2_instance
from arguments import get_args
#from algos.arena_mp2 import ArenaMP
from algos.arena_mp2 import ArenaMP
from utils import utils_goals



def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}_changeroomcost{}'.format(
        agent_args['obs_type'],  
        agent_args['open_cost'],
        agent_args['should_close'], 
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'],
        agent_args['change_room_cost'])
    return mode_str

if __name__ == '__main__':
    args = get_args()

    num_tries = 1
    args.executable_file = '/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/path_sim_dev/linux_exec.v2.3.0.x86_64'
    args.max_episode_length = 100
    args.num_per_apartment = 20
    args.dataset_path = '/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/dataset/new_datasets/dataset_language_large.pik'

    agent_types = [
            ['full', 0, 0.05, False, 0, 0.5],
            ['partial', 0, 0.05, False, 0, 0.5],
            ['partial', 0, 0.05, False, 0.1, 0.5],
            ['partial', 500, 0.05, False, 0.01, 0.5],
            ['partial', -500, 0.05, False, 0.01, 0.5],
            ['partial', 0, 2.00, False, 0.01, 10, 0.5],
    ]

    # Initialize an agent and a helper
    for agent_id in range(0, 2): #len(agent_types)): 
        # the agent and the helper both use agent type 1 for now
        args.obs_type, open_cost, walk_cost, should_close, forget_rate, change_room_cost = agent_types[1]
        datafile = args.dataset_path.split('/')[-1].replace('.pik', '')
        agent_args = {
            'obs_type': args.obs_type,
            'open_cost': open_cost,
            'should_close': True,
            'walk_cost': walk_cost,
            'belief': {'forget_rate': forget_rate},
            'change_room_cost': change_room_cost
        }
        args.mode = '{}_'.format(agent_id+1) + get_class_mode(agent_args)
        args.mode += 'v9_particles_v2'

        
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    print(f"length of env_task_set: {len(env_task_set)}")

    for env in env_task_set:
        init_gr = env['init_graph']
        gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
        init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
        init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]
        for node in init_gr['nodes']:
            if node['class_name'] == 'cutleryfork':
                node['obj_transform']['position'][1] += 0.1

    args.record_dir = '../data/{}/{}'.format(datafile, "language")
    args.record_dir = "/home/scai/Workspace_2/hshi33/training_set/raw_log/"
    error_dir = "/home/scai/Workspace_2/hshi33/training_set/"
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(601))
    #episode_ids = list(range(700, 1001))
    '''for episode_id in episode_ids:
        episode_data = env_task_set[episode_id]
        temp = {}
        for goal in episode_data["task_goal"][0]:
            temp[goal.split("_")[1]] = goal.split("_")[2]
            if goal.split("_")[1] == "toy":
                env_task_set[episode_id]["task_goal"][0][goal] = 2
        for goal in episode_data["task_goal"][1]:
            if goal.split("_")[1] == "toy":
                env_task_set[episode_id]["task_goal"][1][goal] = 2
            if goal.split("_")[1] in temp.keys() and not goal.split("_")[2] == temp[goal.split("_")[1]]:
                if not Path("/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/1500+episodes/logs_episode.{}_iter.0.pik".format(episode_id)).is_file():
                    new_episode_ids.append(episode_id)
                    break'''

    episode_ids = sorted(episode_ids)
    print('episode_ids:', episode_ids)
    print(len(episode_ids))
    # episode_ids = episode_ids[10:]
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]
    
    test_results = {}
    #episode_ids = [episode_ids[0]]
    
    def env_fn(env_id):
        return UnityEnvironment(num_agents=2,
                                max_episode_length=args.max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                observation_types=[args.obs_type for _ in range(2)],
                                use_editor=args.use_editor,
                                executable_args=executable_args,
                                base_port=8088,
                                convert_goal=True)


    args_common = dict(recursive=False,
                            max_episode_length=100,
                            num_simulation=200,
                            max_rollout_steps=5,
                            c_init=0.1,
                            c_base=1000000,
                            num_samples=1,
                            num_processes=20, 
                            num_particles=20,
                            logging=True,
                            logging_graphs=True)

    args_agent1 = {'agent_id': 1, 'char_index': 0} 
    args_agent2 = {'agent_id': 2, 'char_index': 1} # Defined as the helper

    args_agent1.update(args_common)
    args_agent2.update(args_common)

    args_agent1['agent_params'] = agent_args
    args_agent2['agent_params'] = agent_args

    agents = [lambda x, y: MCTS_agent_particle_v2_instance(**args_agent1), 
                lambda x, y: MCTS_agent_particle_v2_instance(**args_agent2)]
    temp = random.random() # 0 neutral, 1 help, -1 hinder
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)
    for iter_id in range(num_tries):
        #if iter_id > 0:

        cnt = 0
        steps_list, failed_tasks = [], []
        current_tried = iter_id

        if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
            test_results = {}
        else:
            test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))
        
        logger = logging.getLogger() 
        logger.setLevel(logging.INFO)
        for episode_id in episode_ids:
            #if episode_id == 0:
            #    continue
            #if episode_id in [2, 6, 7, 12, 17, 20]:
            #    continue
            #curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(
            #env_task_set[episode_id]['task_id'],
            #env_task_set[episode_id]['task_name'],
            #iter_id)

            log_file_name = args.record_dir + '/logs_episode.{}_iter.{}.pik'.format(episode_id, iter_id)
            failure_file = '{}/{}_{}.txt'.format(error_dir, episode_id, iter_id)
            if os.path.isfile(log_file_name):# or os.path.isfile(failure_file):
                continue
            if os.path.isfile("/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/2_partial_opencost0_closecostFalse_walkcost0.05_forgetrate0_changeroomcost0.5v9_particles_v2/logs_episode.{}_iter.0.pik".format(episode_id)):
                continue
            if os.path.isfile(failure_file):
                os.remove(failure_file)
            fileh = logging.FileHandler(failure_file, 'a')
            fileh.setLevel(logging.DEBUG)
            logger.addHandler(fileh)


            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            
            # try:
                
            obs = arena.reset(episode_id)
            print(f"arena reset to {episode_id}")
            if obs is None:
                failed_tasks.append(episode_id)
                pickle.dump({"obs": []}, open(log_file_name, 'wb'))
                continue
            success, steps, saved_info = arena.run()
            print(f"arena run finished. success: {success}, steps: {steps}")
            print('-------------------------------------')
            print('success' if success else 'failure')
            print('steps:', steps)
            print('-------------------------------------')
            if not success:
                failed_tasks.append(episode_id)
            else:
                steps_list.append(steps)
            is_finished = 1 if success else 0

            Path(args.record_dir).mkdir(parents=True, exist_ok=True)
            if len(saved_info['obs']) > 0:
                if not steps == args.max_episode_length and not success:
                    saved_info["fail_to_execute"] = True
                else:
                    saved_info["fail_to_execute"] = False
                pickle.dump(saved_info, open(log_file_name, 'wb'))
            else:
                with open(log_file_name, 'w+') as f:
                    f.write(json.dumps(saved_info, indent=4))
                #failed episodes: saved_info["obs"] == 0

            logger.removeHandler(logger.handlers[0])
            os.remove(failure_file)
            # except:
            #     #with open(failure_file, 'w+') as f:
            #     #    error_str = 'Failure'
            #     #    error_str += '\n'
            #     #    stack_form = ''.join(traceback.format_stack())
            #     #    error_str += stack_form

            #     #    f.write(error_str)
            #     logging.exception("Error")
            #     print("ERROR")
            #     logger.removeHandler(logger.handlers[0])
            #     exit()
            #     arena.reset_env()
            #     continue

            '''S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {'S': S[episode_id],
                                        'L': L[episode_id]}'''
                                        
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

