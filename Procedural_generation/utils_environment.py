import ipdb
import copy
import random


def convert_goal(task_goal, init_graph):
    new_task_goal = {}
    ids_from_class = {}

    for node in init_graph['nodes']:
        if node['class_name'] not in ids_from_class:
            ids_from_class[node['class_name']] = []
        ids_from_class[node['class_name']].append(node['id'])

    newgoals = {}
    for goal_name, count in task_goal.items():
        if type(count) == int:
            cont_id = int(goal_name.split('_')[-1])
            class_name = goal_name.split('_')[1]
            if class_name not in ids_from_class.keys():
                print("Potential bug: ", class_name)
                continue
            obj_grab = ids_from_class[class_name]
            newgoals[goal_name] = {
                'count': count,
                'grab_obj_ids': obj_grab,
                'container_ids': [cont_id],
            }
        else:
            newgoals[goal_name] = count
    return newgoals


def clean_house_obj(graph):
    house_obj = ['window', 'door', 'floor', 'ceiling', 'wall']
    ids = [
        node['id'] for node in graph['nodes'] if node['class_name'].lower() in house_obj
    ]
    id2node = {node['id']: node for node in graph['nodes']}

    def weird_edge(edge, id2node):
        weird_on = [
            'dishwasher',
            'kitchencounterdrawer',
            'dishbowl',
            'mousemat',
            'wine',
            'plate',
        ]
        if (
            edge['relation_type'] == 'ON'
            and id2node[edge['to_id']]['class_name'] in weird_on
        ):
            return True
        return False

    return {
        'nodes': [node for node in graph['nodes'] if node['id'] not in ids],
        'edges': [
            edge
            for edge in graph['edges']
            if edge['from_id'] not in ids
            and edge['to_id'] not in ids
            and not weird_edge(edge, id2node)
        ],
    }


def inside_not_trans(graph):
    # print([{'from_id': 425, 'to_id': 396, 'relation_type': 'ON'}, {'from_id': 425, 'to_id': 396, 'relation_type': 'INSIDE'}])
    id2node = {node['id']: node for node in graph['nodes']}
    parents = {}
    grabbed_objs = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':

            if edge['from_id'] not in parents:
                parents[edge['from_id']] = [edge['to_id']]
            else:
                parents[edge['from_id']] += [edge['to_id']]

        elif edge['relation_type'].startswith('HOLDS'):
            grabbed_objs.append(edge['to_id'])

    edges = []
    for edge in graph['edges']:
        if (
            edge['relation_type'] == 'INSIDE'
            and id2node[edge['to_id']]['category'] == 'Rooms'
        ):
            if len(parents[edge['from_id']]) == 1:
                edges.append(edge)

        else:
            edges.append(edge)
    graph['edges'] = edges

    # # add missed edges
    # missed_edges = []
    # for obj_id, action in self.obj2action.items():
    #     elements = action.split(' ')
    #     if elements[0] == '[putback]':
    #         surface_id = int(elements[-1][1:-1])
    #         found = False
    #         for edge in edges:
    #             if edge['relation_type'] == 'ON' and edge['from_id'] == obj_id and edge['to_id'] == surface_id:
    #                 found = True
    #                 break
    #         if not found:
    #             missed_edges.append({'from_id': obj_id, 'relation_type': 'ON', 'to_id': surface_id})
    # graph['edges'] += missed_edges

    parent_for_node = {}

    char_close = {1: [], 2: []}
    for char_id in range(1, 3):
        for edge in graph['edges']:
            if edge['relation_type'] == 'CLOSE':
                if (
                    edge['from_id'] == char_id
                    and edge['to_id'] not in char_close[char_id]
                ):
                    char_close[char_id].append(edge['to_id'])
                elif (
                    edge['to_id'] == char_id
                    and edge['from_id'] not in char_close[char_id]
                ):
                    char_close[char_id].append(edge['from_id'])
    ## Check that each node has at most one parent
    objects_to_check = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] in parent_for_node and not id2node[edge['from_id']][
                'class_name'
            ].startswith('closet'):
                print('{} has > 1 parent'.format(edge['from_id']))
                raise Exception
            parent_for_node[edge['from_id']] = edge['to_id']
            # add close edge between objects in a container and the character
            if id2node[edge['to_id']]['class_name'] in [
                'fridge',
                'kitchencabinet',
                'cabinet',
                'microwave',
                'dishwasher',
                'stove',
            ]:
                objects_to_check.append(edge['from_id'])
                for char_id in range(1, 3):
                    if (
                        edge['to_id'] in char_close[char_id]
                        and edge['from_id'] not in char_close[char_id]
                    ):
                        graph['edges'].append(
                            {
                                'from_id': edge['from_id'],
                                'relation_type': 'CLOSE',
                                'to_id': char_id,
                            }
                        )
                        graph['edges'].append(
                            {
                                'from_id': char_id,
                                'relation_type': 'CLOSE',
                                'to_id': edge['from_id'],
                            }
                        )

    ## Check that all nodes except rooms have one parent
    nodes_not_rooms = [
        node['id']
        for node in graph['nodes']
        if node['category'] not in ['Rooms', 'Doors']
    ]
    nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
    nodes_without_parent = [
        node for node in nodes_without_parent if node not in grabbed_objs
    ]
    graph['edges'] = [
        edge
        for edge in graph['edges']
        if not (edge['from_id'] in objects_to_check and edge['relation_type'] == 'ON')
    ]
    if len(nodes_without_parent) > 0:
        for nd in nodes_without_parent:
            print(id2node[nd])
        ipdb.set_trace()
        raise Exception
    return graph


def convert_action(action_dict):
    action_dict = copy.deepcopy(action_dict)
    '''if 1 in action_dict:
        if action_dict[1] is not None and 'walk' in action_dict[1]:
            obj_str = ' '.join(action_dict[1].split()[1:])
            action_dict[1] = '[walktowards] {} :3:'.format(obj_str)'''

    agent_do = [item for item, action in action_dict.items() if action is not None]
    # Make sure only one agent interact with the same object
    if len(action_dict.keys()) > 1:
        if (
            None not in list(action_dict.values())
            and sum(['walk' in x for x in action_dict.values()]) < 2
        ):
            # continue
            objects_interaction = [
                x.split('(')[1].split(')')[0] for x in action_dict.values()
            ]
            if len(set(objects_interaction)) == 1:
                agent_do = [random.choice([0, 1])]

    script_list = ['']

    new_action_dict = {index: action_dict[index] for index in agent_do}
    
    for agent_id in agent_do:
        script = action_dict[agent_id]
        if script is None:
            continue
        current_script = ['<char{}> {}'.format(agent_id, script)]

        script_list = [
            x + '|' + y if len(x) > 0 else y
            for x, y in zip(script_list, current_script)
        ]

    # if self.follow:
    # script_list = [x.replace('[walk]', '[walktowards]') for x in script_list]
    # script_all = script_list

    return script_list, new_action_dict


def separate_new_ids_graph(graph, max_id):
    new_graph = copy.deepcopy(graph)
    for node in new_graph['nodes']:
        if node['id'] > max_id:
            node['id'] = node['id'] - max_id + 1000
    for edge in new_graph['edges']:
        if edge['from_id'] > max_id:
            edge['from_id'] = edge['from_id'] - max_id + 1000
        if edge['to_id'] > max_id:
            edge['to_id'] = edge['to_id'] - max_id + 1000
    return new_graph


def check_progress(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    reward = 0.0
    id2node = {node['id']: node for node in state['nodes']}
    class2id = {}
    for node in state['nodes']:
        if node['class_name'] not in class2id:
            class2id[node['class_name']] = []
        class2id[node['class_name']].append(node['id'])

    for key, value in goal_spec.items():

        elements = key.split('_')
        unsatisfied[key] = value[0] if elements[0] not in ['offOn', 'offInside'] else 0
        satisfied[key] = [None] * 2
        satisfied[key]
        satisfied[key] = []
        for edge in state['edges']:
            if elements[0] in 'close':
                if (
                    edge['relation_type'].lower().startswith('close')
                    and id2node[edge['to_id']]['class_name'] == elements[1]
                    and edge['from_id'] == int(elements[2])
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            if elements[0] in ['on', 'inside']:
                if (
                    edge['relation_type'].lower() == elements[0]
                    and edge['to_id'] == int(elements[2])
                    and (
                        id2node[edge['from_id']]['class_name'] == elements[1]
                        or str(edge['from_id']) == elements[1]
                    )
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'offOn':
                if (
                    edge['relation_type'].lower() == 'on'
                    and edge['to_id'] == int(elements[2])
                    and (
                        id2node[edge['from_id']]['class_name'] == elements[1]
                        or str(edge['from_id']) == elements[1]
                    )
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    unsatisfied[key] += 1
            elif elements[0] == 'offInside':
                if (
                    edge['relation_type'].lower() == 'inside'
                    and edge['to_id'] == int(elements[2])
                    and (
                        id2node[edge['from_id']]['class_name'] == elements[1]
                        or str(edge['from_id']) == elements[1]
                    )
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    unsatisfied[key] += 1
            elif elements[0] == 'holds':
                if (
                    edge['relation_type'].lower().startswith('holds')
                    and id2node[edge['to_id']]['class_name'] == elements[1]
                    and edge['from_id'] == int(elements[2])
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'sit':
                if (
                    edge['relation_type'].lower().startswith('sit')
                    and edge['to_id'] == int(elements[2])
                    and edge['from_id'] == int(elements[1])
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
        if elements[0] == 'turnOn':
            if 'ON' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1
        if elements[0] == 'touch':
            for id_touch in class2id[elements[1]]:
                if 'TOUCHED' in [st.upper() for st in id2node[id_touch]['states']]:
                    predicate = '{}_{}_{}'.format(elements[0], id_touch, 1)
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
    # ipdb.set_trace()
    if len(satisfied) == 0 and len(unsatisfied) == 0:
        ipdb.set_trace()
    return satisfied, unsatisfied


def check_progress2(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    reward = 0.0
    id2node = {node['id']: node for node in state['nodes']}
    class2id = {}
    for node in state['nodes']:
        if node['class_name'] not in class2id:
            class2id[node['class_name']] = []
        class2id[node['class_name']].append(node['id'])

    for key, value in goal_spec.items():

        elements = key.split('_')

        preds = []
        objects_int = value['grab_obj_ids']
        container_id = value['container_ids'][0]
        count = value['count'] if elements[0] not in ['offOn', 'offInside'] else 0
        grabbed_objs = []
        for edge in state['edges']:
            if elements[0] == 'close':
                if (
                    edge['relation_type'].lower().startswith('close')
                    and edge['to_id'] in objects_int
                    and edge['from_id'] == int(elements[2])
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    preds.append(predicate)
                    count -= 1
            if elements[0] in ['on', 'inside']:
                if (
                    edge['relation_type'].lower() == elements[0]
                    and edge['to_id'] == int(elements[2])
                    and edge['from_id'] in objects_int
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    preds.append(predicate)
                    count -= 1
            elif elements[0] == 'offOn':
                if (
                    edge['relation_type'].lower() == 'on'
                    and edge['to_id'] == int(elements[2])
                    and edge['from_id'] in objects_int
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    count += 1
            elif elements[0] == 'offInside':
                if (
                    edge['relation_type'].lower() == 'inside'
                    and edge['to_id'] == int(elements[2])
                    and edge['from_id'] in objects_int
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['from_id'], elements[2]
                    )
                    count += 1
            elif elements[0] == 'holds':
                if (
                    edge['relation_type'].lower().startswith('holds')
                    and edge['to_id'] in objects_int
                    and edge['from_id'] == container_id
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    preds.append(predicate)
                    count -= 1
            elif elements[0] == 'sit':
                if (
                    edge['relation_type'].lower().startswith('sit')
                    and edge['to_id'] == int(elements[2])
                    and edge['from_id'] == int(elements[1])
                ):
                    predicate = '{}_{}_{}'.format(
                        elements[0], edge['to_id'], elements[2]
                    )
                    preds.append(predicate)
                    count -= 1
            elif elements[0] == 'offer':
                # if object is already grabbed by the other agent or not grabbed by me
                if (
                    edge['relation_type'].lower().startswith('hold')
                    and edge['to_id'] in objects_int
                ):
                    if edge['from_id'] == container_id:
                        to_id = edge['to_id']
                        predicate = 'offer_{}_{}'.format(to_id, container_id)
                        preds.append(predicate)
                        count -= 1
                    else:
                        # TODO: Grabbed by me, note this will break with more agents
                        grabbed_objs.append(edge['to_id'])
                elif (
                    edge['relation_type'] == 'CLOSE'
                    and edge['from_id'] in objects_int
                    and edge['to_id'] == container_id
                ):
                    to_id = edge['to_id']
                    predicate = 'offer_{}_{}'.format(to_id, container_id)
                    preds.append(predicate)
                    count -= 1

        # if elements[0] == 'offer':
        #     # The objects that are not grabbed anymore, should be satisfied
        #     objects_not_grabbed = set(objects_int) - set(grabbed_objs)
        #     for obj_to_id in objects_not_grabbed:
        #         predicate = 'offer_{}_{}'.format(obj_to_id, container_id)
        #         if predicate not in preds:
        #             preds.append(predicate)
        #             count -= 1

        if elements[0] == 'turnOn':
            if 'ON' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                preds.append(predicate)
                count -= 1
        if elements[0] == 'touch':
            for id_touch in class2id[elements[1]]:
                if 'TOUCHED' in [st.upper() for st in id2node[id_touch]['states']]:
                    predicate = '{}_{}_{}'.format(elements[0], id_touch, 1)
                    preds.append(predicate)
                    count -= 1
        '''if count == 2 and "156" in key:
            ipdb.set_trace()'''

        satisfied[key] = preds
        unsatisfied[key] = count
        # if unsatisfied[key] < 0:
        #     ipdb.set_trace()
    # ipdb.set_trace()
    # if len(satisfied) == 0 and len(unsatisfied) == 0:
    #     ipdb.set_trace()
    return satisfied, unsatisfied
