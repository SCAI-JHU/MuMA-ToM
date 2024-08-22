import numpy as np
import random
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
home_path = "../../"
sys.path.insert(0, "") #your virtualhome api path
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

def get_rooms_category(belief_type):
    room_names = ['kitchen', 'bedroom', 'bathroom', 'livingroom']
    if belief_type == 'uniform':
        room_array = np.ones(len(room_names))
    elif belief_type == 'spiked':

        room_array = np.log([0.8, 0.1, 0.1, 0.1])
    elif belief_type == 'spiked2':
        room_array = np.ones(len(room_names))
    elif belief_type == 'spiked3':
        raise Exception
    elif belief_type == 'spiked4':

        room_array = np.log([0.8, 0.1, 0.1, 0.1])

    else:
        raise Exception

    return room_names, room_array

def get_rooms(id2node, belief_type, room_ids):

    if belief_type == 'uniform':
        room_array = np.ones(len(room_ids))
    elif belief_type == 'spiked':
        # TODO_belief: set to sometihng sensible
        init_values = np.ones(len(room_ids))
        id_kitchen = [(id_room, index_cont) for index_cont, id_room in enumerate(room_ids) if id_room != None and id2node[id_room]['class_name'] ==  'kitchen']
        if len(id_kitchen) > 0:
            # Object is in the cabinet
            # Raw apprixmation
            init_values *= (0.2/len(room_ids))
            init_values[id_kitchen[0][1]] = 0.8
            init_values = np.log(init_values)
        room_array = init_values
    elif belief_type == 'spiked2':
        room_array = np.ones(len(room_ids))
    elif belief_type == 'spiked3':
        raise Exception
    elif belief_type == 'spiked4':
        # TODO_belief: set to sometihng sensible
        init_values = np.ones(len(room_ids))
        id_kitchen = [(id_room, index_cont) for index_cont, id_room in enumerate(room_ids) if id_room != None and id2node[id_room]['class_name'] ==  'kitchen']
        if len(id_kitchen) > 0:
            # Object is in the cabinet
            # Raw apprixmation
            init_values *= (0.2/len(room_ids))
            init_values[id_kitchen[0][1]] = 0.8
            init_values = np.log(init_values)
        room_array = init_values

    else:
        raise Exception

    return room_array


def get_container_prior_category(belief_type):
    container_names = [
        'none',
        'bathroomcabinet',
        'kitchencabinet',
        'cabinet',
        'fridge',
        'stove',
        'dishwasher',
        'microwave']
    
    if belief_type == 'uniform':
        init_values = np.ones(len(container_names))/len(container_names)
    elif belief_type == 'spiked':
        # This belief is that the object is either in the cabinet or in the bathroom
        init_values = np.log([0.025, 0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025])
    elif belief_type == 'spiked2':
        init_values = np.log([0.99, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125])
    elif belief_type == 'spiked3':
        raise Exception
    elif belief_type == 'spiked4':
        init_values = np.log([0.02, 0.02, 0.3, 0.02, 0.3, 0.3, 0.02, 0.02])
        
    else:
        raise Exception
    return  container_names, init_values


def get_container_prior(id2node, belief_type, container_ids):
    if belief_type == 'uniform':
        init_values = np.ones(len(container_ids))/len(container_ids)
    elif belief_type == 'spiked':
        # This belief is that the object is either in the cabinet or in the bathroom
        init_values = np.ones(len(container_ids))/len(container_ids)
        try:
            id_cabinet = [(id_obj, index_cont) for index_cont, id_obj in enumerate(container_ids) if id_obj != None and id2node[id_obj]['class_name'] ==  'cabinet']
        except:
            ipdb.set_trace()
        if len(id_cabinet) > 0:
            # Object is in the cabinet
            # Raw apprixmation
            init_values *= 0.2
            init_values[id_cabinet[0][1]] = 0.8
            init_values = np.log(init_values)
    elif belief_type == 'spiked2':
        init_values = np.ones(len(container_ids))/len(container_ids)
        init_values *= 0.01
        init_values[0] = 0.99
        init_values = np.log(init_values)
    elif belief_type == 'spiked3': #TODO: what is this?
        raise Exception
    elif belief_type == 'spiked4':
        class_names = ['fridge', 'stove', 'kitchencabinet']
        init_values = np.ones(len(container_ids))/len(container_ids)
        try:
            id_cabinet = [(id_obj, index_cont) for index_cont, id_obj in enumerate(container_ids) if id_obj != None and id2node[id_obj]['class_name'] in class_names]
        except:
            ipdb.set_trace()
        if len(id_cabinet) > 0:
            # Object is in the cabinet
            # Raw apprixmation
            init_values *= 0.1
            for idc in id_cabinet:
                init_values[idc[1]] = 0.9/(len(id_cabinet))
            init_values = np.log(init_values)
    else:
        raise Exception
    return  init_values

class Belief():
    def __init__(self, graph_gt, agent_id, prior=None, seed=None, belief_params={}):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.knowledge_containers = True #know all the containers in the environment
        self.forget_rate = 0.
        self.belief_type = "uniform"

        if len(belief_params) > 0:
            if 'forget_rate' in belief_params:
                self.forget_rate = belief_params['forget_rate']
            if 'knowledge_containers' in belief_params:
                self.knowledge_containers = belief_params['knowledge_containers']
            if 'belief_type' in belief_params:
                self.belief_type = belief_params['belief_type']

        # Possible beliefs for some objects
        self.container_restrictions = {
                'book': ['cabinet', 'kitchencabinet']  #only book?
        }

        self.id_restrictions_inside = {

        }

        self.debug = False

        # self.high_prob = 1e9
        self.low_prob = -1e9
        self.name_equivalence = load_name_equivalence()
        self.map_properties_to_pred = {
            'ON': ('on', True),
            'OPEN': ('open', True),
            'OFF': ('on', False),
            'CLOSED': ('open', False)
        }
        
        self.map_edges_to_pred = {
            'INSIDE': 'inside',
            'CLOSE': 'close',
            'ON': 'ontop',
            'FACING': 'facing'
        }
        self.house_obj = [
                'floor',
                'wall',
                'ceiling'
        ]

        self.class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain', 'window']
        self.categories_delete = ['Doors']

        self.agent_id = agent_id
        self.grabbed_object = []



        self.graph_helper = vh_utils.graph_dict_helper()
        self.binary_variables = self.graph_helper.binary_variables

        self.prohibit_ids = [node['id'] for node in graph_gt['nodes'] if node['class_name'].lower() in self.class_nodes_delete or 
                             node['category'] in self.categories_delete]
        new_graph = {
            'nodes': [copy.deepcopy(node) for node in graph_gt['nodes'] if node['id'] not in self.prohibit_ids],
            'edges': [edge for edge in graph_gt['edges'] if edge['to_id'] not in self.prohibit_ids and edge['from_id'] not in self.prohibit_ids]
        } 
        self.graph_init = graph_gt #observation
        # ipdb.set_trace()
        self.sampled_graph = new_graph #own belief no prohibited id
        

        self.states_consider = ['OFF', 'CLOSED']
        self.edges_consider = ['INSIDE', 'ON']
        self.node_to_state_belief = {}
        self.room_node = {}
        self.room_nodes = []
        self.container_ids = []
        self.surface_ids = []

        # Binary Variable Dict
        self.bin_var_dict = {}
        for bin_var in self.binary_variables:
            self.bin_var_dict[bin_var.negative] = [[bin_var.positive, bin_var.negative], 1]
            self.bin_var_dict[bin_var.positive] = [[bin_var.positive, bin_var.negative], 0]

        id2node = {}
        for x in graph_gt['nodes']:
            id2node[x['id']] = x

        # Door edges: Will be used to make the graph walkable
        self.door_edges = {}
        for edge in self.graph_init['edges']:
            if edge['relation_type'] == 'BETWEEN':
                if id2node[edge['from_id']]['category'] == 'Doors':
                    if edge['from_id'] not in self.door_edges.keys():
                        self.door_edges[edge['from_id']] = []
                    self.door_edges[edge['from_id']].append(edge['to_id'])

        # Assume that between 2 nodes there is only one edge
        self.edge_belief = {}
        self.init_belief()
        
        self.first_belief = copy.deepcopy(self.edge_belief) 
        self.first_room = copy.deepcopy(self.room_node) 


    def update(self, origin, final):
        origin_sm = scipy.special.softmax(origin)
        final_sm = scipy.special.softmax(final)
        # pdb.set_trace()
        maxdelta = np.abs(origin_sm - final_sm)
        signdelta = (final_sm - origin_sm ) * 1./(maxdelta+1e-9)
        ratio_delta = np.ones(maxdelta.shape)*self.forget_rate
        delta = signdelta * np.minimum(ratio_delta, maxdelta)
        final = origin_sm + delta + 1e-9

        return np.log(final) 
        # dist_total = origin - final
        # ratio = (1 - np.exp(-self.forget_rate*np.abs(origin-final)))
        
        # print(dist_total, ratio)
        # return origin - ratio*dist_total

    def reset_to_priot_if_invalid(belief_node):
        # belief_node: [names, probs]
        if belief_node[1].max() == self.low_prob:

            belief_node[1] = prior

    def update_to_prior(self):
        for node_name in self.edge_belief:
            self.edge_belief[node_name]['INSIDE'][1] = self.update(self.edge_belief[node_name]['INSIDE'][1], self.first_belief[node_name]['INSIDE'][1])

            self.edge_belief[node_name]['ON'][1] = self.update(self.edge_belief[node_name]['ON'][1], self.first_belief[node_name]['ON'][1])

        for node in self.room_node:
            self.room_node[node][1] = self.update(self.room_node[node][1], self.first_room[node][1])


    def _remove_house_obj(self, state):
        delete_ids = [x['id'] for x in state['nodes'] if x['class_name'].lower() in self.class_nodes_delete]
        state['nodes'] = [x for x in state['nodes'] if x['id'] not in delete_ids]
        state['edges'] = [x for x in state['edges'] if x['from_id'] not in delete_ids and x['to_id'] not in delete_ids]
        return state

    def init_belief(self):
        # Set belief on object states
        id2node = {}
        for node in self.sampled_graph['nodes']:
            id2node[node['id']] = node
            object_name = node['class_name']
            bin_vars = self.graph_helper.get_object_binary_variables(object_name)

            bin_vars = [x for x in bin_vars if x.default in self.states_consider]
            belief_dict = {}
            for bin_var in bin_vars:
                if bin_var.positive == 'OFF' and 'light' not in object_name:
                    # TODO: set a max prob
                    belief_dict[bin_var.positive] = 0.
                elif 'light' in object_name:
                    belief_dict[bin_var.positive] = 1. # Lights on 
                else:
                    belief_dict[bin_var.positive] = 0. # Objectd are closed

            self.node_to_state_belief[node['id']] = belief_dict

        # Surface classes
        surface_classes = [
            'kitchentable',
            'coffeetable',
            'sofa'
        ]

        # TODO: ths class should simply have a surface property
        container_classes = [
        'bathroomcabinet',
        'kitchencabinet',
        'cabinet',
        'fridge',
        'stove',
        # 'kitchencounterdrawer',
        'dishwasher',
        'microwave']


        # Solve these cases
        
        # Set belief for edges
        # TODO: this should be specified in the properties
        object_containers = [node for node in self.sampled_graph['nodes'] if node['class_name'] in container_classes]
        object_surfaces = [node for node in self.sampled_graph['nodes'] if node['class_name'] in surface_classes]
        # object_containers = [node for node in self.sampled_graph['nodes'] if 'CAN_OPEN' in node['properties'] and 'CONTAINERS' in node['properties']]
        



        grabbable_nodes = [node for node in self.sampled_graph['nodes'] if 'GRABBABLE' in node['properties']]

       
        self.room_nodes = [node for node in self.sampled_graph['nodes'] if node['category'] == 'Rooms']


        self.room_ids = [x['id'] for x in self.room_nodes]
        self.container_ids = [None] + [x['id'] for x in object_containers]
        self.surface_ids = [None] + [x['id'] for x in object_surfaces]

        self.room_index_belief_dict = {x: it for it, x in enumerate(self.room_ids)}
        self.container_index_belief_dict = {x: it for it, x in enumerate(self.container_ids) if x is not None}
        self.surface_index_belief_dict = {x: it for it, x in enumerate(self.surface_ids) if x is not None}


        for obj_name in self.container_restrictions:
            possible_classes = self.container_restrictions[obj_name]
            # the ids that should not go here
            restricted_ids = [x['id'] for x in object_containers if x['class_name'] not in possible_classes]
            self.id_restrictions_inside[obj_name] = np.array([self.container_index_belief_dict[rid] for rid in restricted_ids])

        """TODO: better initial belief"""
        object_room_ids = {}
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE' and edge['to_id'] in self.room_ids:
                object_room_ids[edge['from_id']] = edge['to_id']

        #objects inside any container
        nodes_inside_ids = [x['from_id'] for x in self.sampled_graph['edges'] if x['to_id'] not in self.room_ids and x['relation_type'] == 'INSIDE']
        nodes_inside = [node for node in self.sampled_graph['nodes'] if node['id'] in nodes_inside_ids and 'GRABBABLE' not in node['properties']] #TODO: "grabble" should in node['properties']

        objects_for_belief_reasoning = grabbable_nodes+nodes_inside
        # ipdb.set_trace()
        for node in objects_for_belief_reasoning:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            id1 = node['id']
            self.edge_belief[id1] = {}
            
            init_values = get_container_prior(id2node, self.belief_type, self.container_ids)

            init_values_on = np.ones(len(self.surface_ids))/len(self.surface_ids)

            # Much more likely to be on nothing than the opposite, otherwise it will always go there
            init_values_on[0] = 0.5
            init_values_on[1:] = 0.5 / (len(self.surface_ids) - 1)




            if node['class_name'] in self.id_restrictions_inside.keys():
                init_values[self.id_restrictions_inside[node['class_name']]] = self.low_prob

            # The probability of being inside itself is 0
            if id1 in self.container_ids:
                init_values[self.container_ids.index(id1)] = self.low_prob

            if id1 in self.surface_ids:
                init_values[1:] = self.low_prob


            self.edge_belief[id1]['INSIDE'] = [self.container_ids, init_values]
            self.edge_belief[id1]['ON'] = [self.surface_ids, init_values_on]
        
        # Room belief. Will be used for nodes that are not in the belief
        for node in self.sampled_graph['nodes']:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            if node not in self.room_nodes:
                if node['id'] in self.container_ids and self.knowledge_containers:
                    # We will place it in the original room
                    room_container = [edge['to_id'] for edge in self.graph_init['edges'] if edge['from_id'] == node['id'] and edge['relation_type'] == 'INSIDE']
                    assert(len(room_container) == 1)
                    room_index = [it for it,index in enumerate(self.room_ids) if index == room_container[0]][0]
                    room_array = self.low_prob * np.ones(len(self.room_ids))
                    room_array[room_index] = 1
                else:
                    room_array = get_rooms(id2node, self.belief_type, self.room_ids)

                self.room_node[node['id']] = [self.room_ids, room_array]
        self.sampled_graph['edges'] = []


    def reset_belief(self):
        self.sampled_graph['edges'] = []
        self.init_belief()

    def sample_from_belief(self, as_vh_state=False, ids_update=None, obs=None): 
        # ids_update: ids need to be updated
        # Sample states. Sample first inside to check if the object is inside a container
        # If inside nothing, sample a room, if a room is sampled, sample whether it is on an object

        if ids_update is None:
            self.sampled_graph['edges'] = []

        for node in self.sampled_graph['nodes']:
            if ids_update is not None and node['id'] not in ids_update:
                continue
            if node['id'] not in self.node_to_state_belief:
                continue
            belief_node = self.node_to_state_belief[node['id']]
            states = []
            for var_name, var_belief_value in belief_node.items():
                rand_number = random.random()
                value_binary = 1 if rand_number < var_belief_value else 0
                states.append(self.bin_var_dict[var_name][0][value_binary])

            node['states'] = states
        #sample state of updated node from current belief
        node_inside, node_on = {}, {}
        object_grabbed = []
        in_room = []
        # Sample edges
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                node_inside[edge['from_id']] = edge['to_id']

            if edge['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                object_grabbed.append(edge['to_id']) 

            if edge['relation_type'] == 'ON':
                node_on[edge['from_id']] = edge['to_id']

        #inside relationships
        for node in self.sampled_graph['nodes']:
            if ids_update is not None and node['id'] not in ids_update:
                continue

            # Objects that cannot be inside or on anything, but we still want to check if they are in some room
            if node['id'] not in self.edge_belief:
                if node['id'] not in self.room_node.keys():
                    continue
                node_room_cands =  self.room_node[node['id']]
                node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                final_rel = (node_room, 'INSIDE')
            else:
                edge_belief_inside = self.edge_belief[node['id']]['INSIDE']
                if node['id'] in node_inside:
                    # The relationships between unseen objects should stay the same
                    sample_inside = node_inside[node['id']]
                else:
                    try:
                        sample_inside = np.random.choice(edge_belief_inside[0], p=scipy.special.softmax(edge_belief_inside[1]))
                    except:
                        print('Error with {}'.format(node['id']))
                        ipdb.set_trace()
                
                if sample_inside is None:
                    # Sample in a room
                    node_room_cands =  self.room_node[node['id']]
                    node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                    final_rel = (node_room, 'INSIDE')
                    in_room.append((node['id'], node_room))
                    
                else:
                    if sample_inside == node['id']:
                        pass
                    final_rel = (sample_inside, 'INSIDE')

            if final_rel[1] == 'INSIDE':
                node_inside[node['id']] = final_rel[0]
            new_edge = {'from_id': node['id'], 'to_id': final_rel[0], 'relation_type': final_rel[1]}
            
            # if node['id'] in [462, 458]:
            #     print(colored("Objcect sample", "magenta"), node['id'], 
            #         scipy.special.softmax(self.edge_belief[node['id']]['INSIDE'][1]), scipy.special.softmax(self.edge_belief[node['id']]['ON'][1]))
            #     print(new_edge)
            self.sampled_graph['edges'].append(new_edge)

            


        # ON relationships
        room2surface = {}
        for room_id in self.room_ids:
            room2surface[room_id] = []


        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE' and edge['from_id'] in self.surface_ids and edge['to_id'] in self.room_ids:
                room2surface[edge['to_id']].append(edge['from_id'])

        for node_id, room_id in in_room:
            # For all the objects in a room, check if they are in a surface
            surfaces = room2surface[room_id]
            if node_id not in self.edge_belief:
                # print(id2node[node_id])
                # ipdb.set_trace()
                continue

            # if node_id in [462, 458]:
            #     print(colored("Objcect sample 2", "magenta"), node['id'], 
            #         scipy.special.softmax(self.edge_belief[node['id']]['INSIDE'][1]), scipy.special.softmax(self.edge_belief[node['id']]['ON'][1]))
            #     print(new_edge)
            surface_cands = self.edge_belief[node_id]['ON']
            node_surface = np.random.choice(surface_cands[0], p=scipy.special.softmax(surface_cands[1]))
            if node_surface in surfaces:
                new_edge = {'from_id': node_id, 'to_id': node_surface, 'relation_type': 'ON'}
                self.sampled_graph['edges'].append(new_edge)
        # try:
        #
        #     nodes_inside_graph = [edge['from_id'] for edge in self.sampled_graph['edges'] if
        #                           edge['relation_type'] == 'INSIDE']
        #     objects_grabbed = [edge['to_id'] for edge in self.sampled_graph['edges'] if
        #                        'HOLDS' in edge['relation_type']]
        #     nodes_inside_graph += objects_grabbed
        #     assert (len(set(self.edge_belief.keys()) - set(nodes_inside_graph)) == 0)
        # except:
        #     pdb.set_trace()

        # Include the doors
        for node_door in self.door_edges.keys():
            node_1, node_2 = self.door_edges[node_door]
            self.sampled_graph['edges'].append({'to_id': node_1, 'from_id': node_door, 'relation_type': 'BETWEEN'})
            self.sampled_graph['edges'].append({'to_id': node_2, 'from_id': node_door, 'relation_type': 'BETWEEN'})

        if as_vh_state:
            return self.to_vh_state(self.sampled_graph)

        
        return self.sampled_graph

    def to_vh_state(self, graph):
        state = self._remove_house_obj(graph)
        vh_state = EnvironmentState(EnvironmentGraph(state), 
                                    self.name_equivalence, instance_selection=True)
        return vh_state

    def canopen_and_open(self, node):
        return 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']

    def is_surface(self, node):
        return 'SURFACE' in node['properties']

    def update_belief(self, gt_graph):
        self.update_to_prior()
        self.update_from_gt_graph(gt_graph)

    def update_graph_from_gt_graph(self, gt_graph, sampled_graph=None, resample_unseen_nodes=False, update_belief=True, language_response=None):
        """
        Updates the current sampled graph with a set of observations
        """
        # Here we have a graph sampled from our belief, and want to update it with gt graph

        if sampled_graph is not None:
            self.sampled_graph = sampled_graph


        id2node = {} 
        gt_graph = {
            'nodes': [node for node in gt_graph['nodes'] if node['id'] not in self.prohibit_ids],
            'edges': [edge for edge in gt_graph['edges'] if edge['from_id'] not in self.prohibit_ids and edge['to_id'] not in self.prohibit_ids]
        }
        ids_visible = [node['id'] for node in gt_graph['nodes']]
        
        edges_gt_graph = gt_graph['edges']
        for x in gt_graph['nodes']:
            id2node[x['id']] = x

        if update_belief:
            self.update_belief_from_languages(language_response) # TODO: does the order of update_belief_prior and update_belief_from_languages matter?
            self.update_belief(gt_graph)

        char_node = self.agent_id


        inside = {}
        for x in edges_gt_graph:
            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'], id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])
                    import ipdb
                    ipdb.set_trace()
                    raise Exception

                inside[x['from_id']] = x['to_id']

        for node in self.sampled_graph['nodes']:
            if node['id'] in id2node.keys():
                # Update the state of the visible nodes
                states_graph_old = id2node[node['id']]['states']
                object_name = id2node[node['id']]['class_name']
                bin_vars = self.graph_helper.get_object_binary_variables(object_name)
                bin_vars = [x for x in bin_vars if x.default in self.states_consider]
                bin_vars_missing = [x for x in bin_vars if x.positive not in states_graph_old and x.negative not in states_graph_old]
                states_graph = states_graph_old + [x.default for x in bin_vars_missing]
                # fill out the rest of info regarding the states
                node['states'] = states_graph
                id2node[node['id']]['states'] = states_graph



        edges_keep = []
        ids_to_update = []

        if resample_unseen_nodes:
            ids_to_update = list(set([node['id'] for node in self.sampled_graph['nodes']]) - set(ids_visible))
        else:
            for edge in self.sampled_graph['edges']:
                if (edge['from_id'] == char_node and edge['relation_type'] == 'INSIDE'):
                    continue

                # Grabbed objects we don't need to keep them
                if edge['from_id'] == char_node and 'HOLD' in edge['relation_type']:
                    continue

                # If the object should be visible but it is not in the observation, remove close relationship
                if (edge['from_id'] == char_node or edge['to_id'] == char_node) and edge['relation_type'] == 'CLOSE':
                    continue

                # Objects that are visible, we do not care anymore
                if edge['from_id'] in id2node.keys() and edge['from_id'] != char_node:
                    # The second condition is for relations such as facing
                    continue

                # The object is not visible but the container is visible


                if edge['to_id'] in id2node.keys() and edge['to_id'] != char_node:
                    # If it is a room and we have not seen it, the belief remains
                    if id2node[edge['to_id']]['category'] == 'Rooms' and edge['relation_type'] == 'INSIDE':
                        if inside[char_node] == edge['to_id']:
                            if edge['from_id'] not in id2node.keys():
                                ids_to_update.append(edge['from_id'])
                            else:
                                pass
                            continue
                    else:
                        if edge['relation_type'] == 'ON':
                            ids_to_update.append(edge['from_id'])
                            continue
                        if edge['relation_type'] == 'INSIDE' and 'OPEN' in id2node[edge['to_id']]['states']:
                            ids_to_update.append(edge['from_id'])
                            continue
                edges_keep.append(edge)

        ids_to_update = list(set(ids_to_update))

        self.sampled_graph['edges'] = edges_keep + edges_gt_graph

        # For objects that are inside in the belief, character should also be close to those, so that when we open the object
        # we are already close to what is inside

        nodes_close = [x['to_id'] for x in edges_gt_graph if x['from_id'] == char_node and x['relation_type'] == 'CLOSE']
        inside_belief = {}
        for edge in edges_keep:
            if edge['relation_type'] == 'INSIDE':
                if edge['to_id'] not in inside_belief: inside_belief[edge['to_id']] = []
                inside_belief[edge['to_id']].append(edge['from_id'])

        for node in nodes_close:
            if node not in inside_belief:
                continue

            for node_inside in inside_belief[node]:
                close_edges = [
                        {'from_id': char_node, 'to_id': node_inside, 'relation_type': 'CLOSE'},
                        {'to_id': char_node, 'from_id': node_inside, 'relation_type': 'CLOSE'}
                ]
                self.sampled_graph['edges'] += close_edges

        # sample new edges that have not been seen
        self.sample_from_belief(ids_update=ids_to_update)
        # print(colored("objects on table","yellow"), [edge['from_id'] for edge in self.sampled_graph['edges'] if edge['to_id'] == 232 and edge['relation_type'] == 'ON'])
        return self.sampled_graph
    

    def update_from_gt_graph(self, gt_graph):
        # Update the states of nodes that we can see in the belief. Note that this does not change the sampled graph
        id2node = {}
        for x in gt_graph['nodes']:
            id2node[x['id']] = x
       
        inside, on = {}, {}

        grabbed_object = []
        for x in gt_graph['edges']:
            if x['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                grabbed_object.append(x['to_id'])

            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'], id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])
                    raise Exception #?
                inside[x['from_id']] = x['to_id']
            
            if x['relation_type'] == 'ON' and x['to_id'] in self.surface_ids:
                if x['from_id'] in on.keys() and x['from_id'] in self.edge_belief.keys() and x['to_id'] in self.surface_ids:
                    print("Already on", id2node[x['from_id']]['class_name'], id2node[on[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])
                    raise Exception
                on[x['from_id']] = x['to_id']

        visible_ids = [x['id'] for x in gt_graph['nodes']]
        edge_tuples = [(x['from_id'], x['to_id']) for x in gt_graph['edges']]

        for x in gt_graph['nodes']:
            try:
                dict_state = self.node_to_state_belief[x['id']]
                for state in x['states']:
                    pred_name = self.bin_var_dict[state][0][0]
                    dict_state[pred_name] = self.bin_var_dict[state][1]
            except:
                pass
        
        char_node = self.agent_id

        visible_room = inside[char_node]

        
        deleted_edges = []
        id_updated = []

        # Keep track of things with impossible belief
        # Objects and rooms we are just seeing
        ids_known_info = [self.room_index_belief_dict[visible_room], []]
        for id_node in self.edge_belief.keys():
            id_updated.append(id_node)
            if id_node in grabbed_object:
                continue

            if id_node in visible_ids:  
                # TODO: what happens when object grabbed
                assert(id_node in inside.keys())
                inside_obj = inside[id_node]
                
                # Some objects have the relationship inside but they are not part of the belief because
                # they are visible anyways like bookshelf. In that case we consider them to just be
                # inside the room
                if inside_obj not in self.room_ids and inside_obj not in self.container_index_belief_dict:
                    inside_obj = inside[inside_obj]
                
                # If object is inside a room, for sure it is not insde another object
                if inside_obj in self.room_ids:
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][0] = 1.
                    self.room_node[id_node][1][:] = self.low_prob
                    self.room_node[id_node][1][self.room_index_belief_dict[inside_obj]] = 1.
                else:
                    # If object is inside an object, for sure it is not insde another object
                    index_inside = self.container_index_belief_dict[inside_obj]
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][index_inside] = 1.

                if id_node in on.keys():
                    # object is on something
                    on_obj = on[id_node]
                    index_on = self.surface_index_belief_dict[on_obj]
                    self.edge_belief[id_node]['ON'][1][:] = self.low_prob
                    self.edge_belief[id_node]['ON'][1][index_on] = 1.
                else:
                    # object is for suure on nothing
                    self.edge_belief[id_node]['ON'][1][:] = self.low_prob
                    self.edge_belief[id_node]['ON'][1][0] = 1.

            else:
                # If not visible. for sure not in this room
                curr_prob = scipy.special.softmax(self.room_node[id_node][1])
                prob_room = curr_prob[self.room_index_belief_dict[visible_room]]
                self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob
                #if (self.room_node[id_node][1] > self.low_prob).sum() == 0:
                if prob_room > 0.99:
                    if id_node in self.edge_belief:
                        # If not in any room, needs to be inside something
                        self.edge_belief[id_node]['INSIDE'][1][0] = self.low_prob

        # update belief for container objects
        for id_node in self.container_ids:
            if id_node in visible_ids and 'OPEN' in id2node[id_node]['states']:
                for id_node_child in self.edge_belief.keys():
                    if id_node_child not in inside.keys() or inside[id_node_child] != id_node:
                        ids_known_info[1].append(self.container_index_belief_dict[id_node])
                        self.edge_belief[id_node_child]['INSIDE'][1][self.container_index_belief_dict[id_node]] = self.low_prob
                
        # Update belief for surface objects
        for id_node in self.surface_ids:
            if id_node in visible_ids:
                for id_node_child in self.edge_belief.keys():
                    if id_node_child not in on.keys() or on[id_node_child] != id_node:
                        ids_known_info[1].append(self.surface_index_belief_dict[id_node])
                        self.edge_belief[id_node_child]['ON'][1][self.surface_index_belief_dict[id_node]] = self.low_prob


        # Some furniture has no edges, only has info about inside rooms
        # We need to udpate its location
        for id_node in self.room_node.keys():
            if id_node not in id_updated:
                if id_node in visible_ids:
                    inside_obj = inside[id_node]
                    if inside_obj == visible_room:
                        self.room_node[id_node][1][:]  = self.low_prob
                        self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = 1.
                    else:
                        assert('Error: A grabbable object is inside something else than a room')
                else:
                    # Either the node goes inside somehting in the room... or ti should not be
                    # in this room
                    self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob

        mask_house = np.ones(len(self.room_nodes))
        mask_obj = np.ones(len(self.container_ids))
        mask_house[ids_known_info[0]] = 0

        assert (len(self.room_nodes) > 0)
        if len(ids_known_info[1]):
            mask_obj[np.array(ids_known_info[1])] = 0

        mask_obj = (mask_obj == 1)
        mask_house = (mask_house == 1)

        # Check for impossible beliefs
        for id_node in self.room_node.keys():
            if id_node in self.edge_belief.keys():
                # the object should be in a room or inside something
                if np.max(self.edge_belief[id_node]['INSIDE'][1]) == self.low_prob:
                    # Sample locations except for marked ones
                    self.edge_belief[id_node]['INSIDE'][1] = self.first_belief[id_node]['INSIDE'][1]
                    # Sample rooms except marked
                    try:
                        self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]
                    except:
                        pdb.set_trace()
            else:
                if np.max(self.room_node[id_node][1]) == self.low_prob:
                    self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]

        # print("New belief", self.edge_belief[458]['ON'])
                    
    def update_belief_from_languages(self, language_response = None):
        if language_response == None:
            return
        
        # check if the language type is location
        assert(language_response.language_type == 'location')
        '''print(self.container_index_belief_dict)
        print(self.edge_belief[414]["INSIDE"])
        print(self.edge_belief[414]["ON"])'''

        for obj_name, obj_info in language_response.obj_positions.items():
            if len(obj_info.keys()) == 0:
                continue
            for obj_id, locations in obj_info.items():
                if obj_id not in self.edge_belief.keys():
                    continue
                self.edge_belief[obj_id]['INSIDE'][1][:] = self.low_prob
                self.edge_belief[obj_id]['ON'][1][:] = self.low_prob
                for location in locations:
                    if location["predicate"].upper() == "INSIDE":
                        if location["position"] is None:
                            self.edge_belief[obj_id]['INSIDE'][1][0] = 1.
                        else:
                            self.edge_belief[obj_id]['INSIDE'][1][self.container_index_belief_dict[location["position"]]] = 1.
                    if location["predicate"].upper() == "ON":
                        if location["position"] is None:
                            self.edge_belief[obj_id]['ON'][1][0] = 1.
                        else:
                            try:
                                self.edge_belief[obj_id]['ON'][1][self.surface_index_belief_dict[location["position"]]] = 1.
                            except KeyError:
                              ipdb.set_trace()  
        '''# Update the belief from the language
        pred, obj_name, position_id = language_response.parse()

        obj_ids = [node['id'] for node in self.sampled_graph['nodes'] if node['class_name'] == obj_name]

        if len(obj_ids) == 0:
            ipdb.set_trace()
        else:
            obj_id = obj_ids[0] # TODO: choose the first object for now
            if pred.upper() == 'INSIDE':
                self.edge_belief[obj_id]['INSIDE'][1][:] = self.low_prob
                self.edge_belief[obj_id]['INSIDE'][1][self.container_index_belief_dict[position_id]] = 1.
            elif pred.upper() == 'ON':
                self.edge_belief[obj_id]['ON'][1][:] = self.low_prob
                self.edge_belief[obj_id]['ON'][1][self.surface_index_belief_dict[position_id]] = 1.
            else:
                raise Exception'''
            
            







if __name__ == '__main__':
    graph_init = '../../example_graph/example_graph.json' 
    with open(graph_init, 'r') as f:
        graph = json.load(f)['init_graph']
    Belief(graph)
