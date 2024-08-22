import pdb
import ipdb
import copy


def get_container_task(init_goal_manager, graph, containers):
    containers = list(set(containers))

    container_id_map = {}
    container_ids = []
    container_preds = []
    kitchen = [
        node["id"] for node in graph["nodes"] if node["class_name"] == "kitchen"
    ][0]
    ids_kichen = [
        edge["from_id"]
        for edge in graph["edges"]
        if edge["to_id"] == kitchen and edge["relation_type"] == "INSIDE"
    ]

    for cont, pred in containers:
        cont_ids = [
            node["id"] for node in graph["nodes"] if (cont == node["class_name"])
        ]
        if cont == "sink":
            cont_ids = [ctid for ctid in cont_ids if ctid in ids_kichen]
        if len(cont_ids) == 0:
            ipdb.set_trace()
        cont_id = init_goal_manager.rand.choice(cont_ids)
        container_id_map[cont] = cont_id
        container_ids.append(cont_id)
        container_preds.append(pred)

    if len(container_ids) == 0:
        ipdb.set_trace()

    # if 'stove' not in container_id_map:
    #     ipdb.set_trace()
    return container_ids, container_preds, container_id_map


def remove_objects_from_ids(init_goal_manager, graph, container_ids, rel_dict):
    # Remove the obejcts inside the index

    rels_cont = ["ON", "INSIDE"]
    ids_in_container = []
    for edge in graph["edges"]:
        if edge["to_id"] in container_ids:
            curr_rel = rel_dict[edge["to_id"]]
            if edge["relation_type"] in curr_rel:
                ids_in_container.append(edge["from_id"])

    # ids_in_container = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] in container_ids and edge['relation_type'] in rels_cont]
    graph = init_goal_manager.remove_obj(graph, ids_in_container)
    return graph


def cleanup_graph(init_goal_manager, graph, start):
    if not start:
        return graph

    # ipdb.set_trace()
    # Clean the containers where we will place stuff
    objects_rel_clean = [
        ("kitchentable", ["ON"]),
        ("dishwasher", ["INSIDE"]),
        ("fridge", ["INSIDE"]),
        ("stove", ["INSIDE"]),
        ("microwave", ["INSIDE"]),
        ("coffeetable", ["ON"]),
    ]
    objects_clean = [x[0] for x in objects_rel_clean]
    rel_class_dict = {x[0]: x[1] for x in objects_rel_clean}

    objects_grab_clean = list(init_goal_manager.init_pool_tasks["obj_random"])

    container_ids_clean = []
    rel_dict = {}
    for node in graph["nodes"]:
        if node["id"] in objects_clean:
            rel_dict[node["id"]] = rel_class_dict[node["class_name"]]
    graph = remove_objects_from_ids(
        init_goal_manager, graph, container_ids_clean, rel_dict
    )

    # Remove the objects that have to do with the goal
    ids_obj = [
        node["id"]
        for node in graph["nodes"]
        if node["class_name"] in objects_grab_clean
    ]
    graph = init_goal_manager.remove_obj(graph, ids_obj)
    id2node = {node["id"]: node for node in graph["nodes"]}
    # print(container_ids_clean)
    # ipdb.set_trace()
    # print(ids_obj)
    # print([id2node[edge['from_id']]['class_name'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE'])
    return graph


def build_env_goal(
    task_name,
    init_goal_manager,
    container_ids,
    container_pred=[],
    container_ids_random=[],
    container_random_pred=[],
):
    env_goal = {task_name: []}
    for k, v in init_goal_manager.goal.items():
        env_goal[task_name].append(
            {"put_{}_{}_{}".format(k, container_pred[0], container_ids[0]): v}
        )

    ## get goal
    env_goal["noise"] = []
    for k, v in init_goal_manager.goal_random_agent.items():
        env_goal["noise"].append(
            {
                "put_{}_{}_{}".format(
                    k, container_random_pred[0], container_ids_random[0]
                ): v
            }
        )
    return env_goal


class Task:
    @staticmethod
    def setup_table(init_goal_manager, graph, start=True):
        # ipdb.set_trace()

        # Make sure candidates are available
        candidates = init_goal_manager.init_pool["setup_table"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]

        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["setup_table"]["counts"]["min"],
            init_goal_manager.init_pool["setup_table"]["counts"]["max"],
        )

        pr_graph = copy.deepcopy(graph)
        graph = cleanup_graph(init_goal_manager, graph, start)

        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)

        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool["setup_table"]["objects"]

        extra_object = init_goal_manager.rand.choice(["wine", "juice", "mug"])
        objects_select = [extra_object] + ["spoon", "wineglass"]
        for object_name in objects_select:
            init_goal_manager.goal[object_name] = counts_objects

        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects (in this version, only sample neccessary #objects that can satisfy the goal)
        for k, v in init_goal_manager.goal.items():
            try:
                num_obj = v
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
            )

        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal(
            "setup_table",
            init_goal_manager,
            container_ids,
            container_pred,
            container_ids_random,
            container_random_pred,
        )

        return graph, env_goal, True

    @staticmethod
    def put_dishwasher(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool["put_dishwasher"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["put_dishwasher"]["counts"]["min"],
            init_goal_manager.init_pool["put_dishwasher"]["counts"]["max"],
        )

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        object_candidates = ["coffeepot", "wineglass", "spoon", "mug", "glass", "pot"]
        different_classes = init_goal_manager.rand.randint(2, 3)
        objects_selected = init_goal_manager.rand.choices(
            object_candidates, k=different_classes
        )
        how_many_objects = init_goal_manager.rand.randint(min_count, max_count)
        obj_final = []
        for obj_name in objects_selected:
            obj_final += [obj_name] * how_many_objects
        # Get a list of the number of objecs we want to add

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(
                v, v
            )  # random select objects >= goal
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
                except_objects=object_candidates,
            )

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal(
            "put_dishwasher", init_goal_manager, container_ids, container_pred
        )

        return graph, env_goal, True

    @staticmethod
    def put_fridge(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool["put_fridge"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["put_fridge"]["counts"]["min"],
            init_goal_manager.init_pool["put_fridge"]["counts"]["max"],
        )

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        object_candidates = ["carrot", "potato", "bread", "milk", "juice", "beer"]
        different_classes = init_goal_manager.rand.randint(2, 3)
        objects_selected = init_goal_manager.rand.choices(
            object_candidates, k=different_classes
        )
        # how_many_objects = init_goal_manager.rand.randint(3, 7)
        # all_object_pool = []
        # for obj_name in objects_selected:
        #     all_object_pool += [obj_name] * how_many_objects
        # # Get a list of the number of objecs we want to add
        # obj_final = init_goal_manager.rand.choices(all_object_pool, k=how_many_objects)
        how_many_objects = init_goal_manager.rand.randint(min_count, max_count)
        obj_final = []
        for obj_name in objects_selected:
            obj_final += [obj_name] * how_many_objects

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(
                v, v
            )  # random select objects >= goal
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
                except_objects=object_candidates,
            )

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal(
            "put_fridge", init_goal_manager, container_ids, container_pred
        )

        return graph, env_goal, True
    
    @staticmethod
    def setup_desk(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool["setup_desk"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["setup_desk"]["counts"]["min"],
            init_goal_manager.init_pool["setup_desk"]["counts"]["max"],
        )

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        object_candidates = ["book", "cellphone", "remotecontrol", "folder", "notes"]
        different_classes = init_goal_manager.rand.randint(2, 3)
        objects_selected = init_goal_manager.rand.choices(
            object_candidates, k=different_classes
        )
        # how_many_objects = init_goal_manager.rand.randint(3, 7)
        # all_object_pool = []
        # for obj_name in objects_selected:
        #     all_object_pool += [obj_name] * how_many_objects
        # # Get a list of the number of objecs we want to add
        # obj_final = init_goal_manager.rand.choices(all_object_pool, k=how_many_objects)
        how_many_objects = init_goal_manager.rand.randint(min_count, max_count)
        obj_final = []
        for obj_name in objects_selected:
            obj_final += [obj_name] * how_many_objects

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(
                v, v
            )  # random select objects >= goal
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
                except_objects=object_candidates,
            )

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal(
            "setup_desk", init_goal_manager, container_ids, container_pred
        )

        return graph, env_goal, True
    
    @staticmethod
    def prepare_drink(init_goal_manager, graph, start=True):
        # ipdb.set_trace()

        # Make sure candidates are available
        candidates = init_goal_manager.init_pool["prepare_drink"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]

        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["prepare_drink"]["counts"]["min"],
            init_goal_manager.init_pool["prepare_drink"]["counts"]["max"],
        )

        pr_graph = copy.deepcopy(graph)
        graph = cleanup_graph(init_goal_manager, graph, start)

        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)

        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool["prepare_drink"]["objects"]

        extra_object = init_goal_manager.rand.sample(["wine", "beer", "milk", "juice"], 2)
        objects_select = extra_object + ["wineglass"]
        for object_name in objects_select:
            init_goal_manager.goal[object_name] = counts_objects

        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects (in this version, only sample neccessary #objects that can satisfy the goal)
        for k, v in init_goal_manager.goal.items():
            try:
                num_obj = v
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
            )

        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal(
            "prepare_drink",
            init_goal_manager,
            container_ids,
            container_pred,
            container_ids_random,
            container_random_pred,
        )

        return graph, env_goal, True

    @staticmethod
    def prepare_food(init_goal_manager, graph, start=True):
        # ipdb.set_trace()

        # Make sure candidates are available
        candidates = init_goal_manager.init_pool["prepare_food"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]

        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["prepare_food"]["counts"]["min"],
            init_goal_manager.init_pool["prepare_food"]["counts"]["max"],
        )

        pr_graph = copy.deepcopy(graph)
        graph = cleanup_graph(init_goal_manager, graph, start)

        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)

        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool["prepare_food"]["objects"]

        extra_object = init_goal_manager.rand.sample(["potato", "carrot", "bread"], 2)
        objects_select = extra_object
        for object_name in objects_select:
            init_goal_manager.goal[object_name] = counts_objects

        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects (in this version, only sample neccessary #objects that can satisfy the goal)
        for k, v in init_goal_manager.goal.items():
            try:
                num_obj = v
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
            )
        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal(
            "prepare_food",
            init_goal_manager,
            container_ids,
            container_pred,
            container_ids_random,
            container_random_pred,
        )

        return graph, env_goal, True
    
    @staticmethod
    def collect_document(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool["collect_document"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["collect_document"]["counts"]["min"],
            init_goal_manager.init_pool["collect_document"]["counts"]["max"],
        )

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        object_candidates = ["book", "folder", "magazine", "check", "notes", "address_book"]
        different_classes = init_goal_manager.rand.randint(2, 3)
        objects_selected = init_goal_manager.rand.choices(
            object_candidates, k=different_classes
        )
        how_many_objects = init_goal_manager.rand.randint(min_count, max_count)
        obj_final = []
        for obj_name in objects_selected:
            obj_final += [obj_name] * init_goal_manager.rand.randint(min_count, max_count)
        # Get a list of the number of objecs we want to add

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(
                v, v
            )  # random select objects >= goal
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
                except_objects=object_candidates,
            )

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal(
            "collect_document", init_goal_manager, container_ids, container_pred
        )

        return graph, env_goal, True
    
    @staticmethod
    def collect_toy(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool["collect_toy"]["candidates"]

        class_names = [node["class_name"] for node in graph["nodes"]]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = (
            init_goal_manager.init_pool["collect_toy"]["counts"]["min"],
            init_goal_manager.init_pool["collect_toy"]["counts"]["max"],
        )

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(
            init_goal_manager, graph, [(container_name, pred_name)]
        )

        container_ids_random, container_random_pred, container_ids_random_map = (
            [],
            [],
            {},
        )

        id2node = {node["id"]: node for node in graph["nodes"]}

        object_candidates = ["toy", "board_game"]
        objects_selected = init_goal_manager.rand.choices(
            object_candidates, k=1
        )
        how_many_objects = init_goal_manager.rand.randint(min_count, max_count)
        obj_final = []
        for obj_name in objects_selected:
            obj_final += [obj_name] * init_goal_manager.rand.randint(min_count, max_count)
        # Get a list of the number of objecs we want to add

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(
                v, v
            )  # random select objects >= goal
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    k,
                    num_obj,
                    init_goal_manager.object_id_count,
                    objs_in_room=objs_in_room,
                    except_position=except_position_ids,
                    goal_obj=True,
                )
            except:
                return None, None, False
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=objs_in_room,
                except_position=except_position_ids,
                except_objects=object_candidates,
            )

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal(
            "collect_toy", init_goal_manager, container_ids, container_pred
        )

        return graph, env_goal, True
    
    @staticmethod
    def unload_dishwasher(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        init_goal_manager.goal = {}
        object_candidates = ["spoon", "coffeepot"]
        objects_selected = ["wineglass"]
        for obj in object_candidates:
            if init_goal_manager.rand.random() > 0.2:
                objects_selected.append(obj)
        # Get a list of the number of objecs we want to add

        '''init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1'''

        dishwasher_ids = [node["id"] for node in graph["nodes"] if node["class_name"] == "dishwasher"]
        desired_id = init_goal_manager.rand.choice(dishwasher_ids)

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]

        for obj in objects_selected:
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    obj,
                    2,
                    init_goal_manager.object_id_count,
                    goal_obj=True,
                    enforced_adding=[desired_id, "INSIDE"]
                )
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False
        

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=None,
                except_position=except_position_ids + [desired_id],
                except_objects=object_candidates,
            )
        env_goal = {"unload_dishwasher": []}
        for obj in objects_selected:
            locations = init_goal_manager.obj_position[obj]
            startIndex = init_goal_manager.rand.randint(0, len(locations) - 1)
            for i in range (len(locations)):
                index = (startIndex + i) % len(locations)
                predicate, location = locations[index]
                if location == "dishwasher":
                    continue
                ids = [node["id"] for node in graph["nodes"] if node["class_name"] == location]
                if len(ids) == 0:
                    continue
                place = init_goal_manager.rand.choice(ids)
                env_goal["unload_dishwasher"].append({"put_{}_{}_{}".format(predicate.lower(), obj, place): 2})
                break
        return graph, env_goal, True

    @staticmethod
    def clear_fridge(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        init_goal_manager.goal = {}
        object_candidates = ["carrot", "potato", "wine"]
        objects_selected = ["beer"]
        for obj in object_candidates:
            if init_goal_manager.rand.random() > 0.2:
                objects_selected.append(obj)
        # Get a list of the number of objecs we want to add

        '''init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1'''

        dishwasher_ids = [node["id"] for node in graph["nodes"] if node["class_name"] == "fridge"]
        desired_id = init_goal_manager.rand.choice(dishwasher_ids)

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]

        for obj in objects_selected:
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    obj,
                    2,
                    init_goal_manager.object_id_count,
                    goal_obj=True,
                    enforced_adding=[desired_id, "INSIDE"]
                )
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False
        

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=None,
                except_position=except_position_ids + [desired_id],
                except_objects=object_candidates,
            )
        env_goal = {"clear_fridge": []}
        for obj in objects_selected:
            locations = init_goal_manager.obj_position[obj]
            startIndex = init_goal_manager.rand.randint(0, len(locations) - 1)
            for i in range (len(locations)):
                index = (startIndex + i) % len(locations)
                predicate, location = locations[index]
                if location == "fridge":
                    continue
                ids = [node["id"] for node in graph["nodes"] if node["class_name"] == location]
                if len(ids) == 0:
                    continue
                place = init_goal_manager.rand.choice(ids)
                env_goal["clear_fridge"].append({"put_{}_{}_{}".format(predicate.lower(), obj, place): 2})
                break
        return graph, env_goal, True
    
    @staticmethod
    def clear_table(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        init_goal_manager.goal = {}
        object_candidates = ["wine", "wineglass", "spoon"]
        objects_selected = []
        for obj in object_candidates:
            if init_goal_manager.rand.random() > 0.2:
                objects_selected.append(obj)
        # Get a list of the number of objecs we want to add

        '''init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1'''

        dishwasher_ids = [node["id"] for node in graph["nodes"] if node["class_name"] == "coffeetable" or node["class_name"] == "kitchentable"]
        desired_id = init_goal_manager.rand.choice(dishwasher_ids)

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]

        for obj in objects_selected:
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    obj,
                    2,
                    init_goal_manager.object_id_count,
                    goal_obj=True,
                    enforced_adding=[desired_id, "ON"]
                )
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False
        

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=None,
                except_position=except_position_ids + [desired_id],
                except_objects=object_candidates,
            )
        env_goal = {"clear_table": []}
        for obj in objects_selected:
            locations = init_goal_manager.obj_position[obj]
            startIndex = init_goal_manager.rand.randint(0, len(locations) - 1)
            for i in range (len(locations)):
                index = (startIndex + i) % len(locations)
                predicate, location = locations[index]
                if location == "kitchentable" or location == "coffeetable":
                    continue
                ids = [node["id"] for node in graph["nodes"] if node["class_name"] == location]
                if len(ids) == 0:
                    continue
                place = init_goal_manager.rand.choice(ids)
                env_goal["clear_table"].append({"put_{}_{}_{}".format(predicate.lower(), obj, place): 2})
                break
        return graph, env_goal, True
    
    @staticmethod
    def clear_desk(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        init_goal_manager.goal = {}
        object_candidates = ["cellphone", "remotecontrol"]
        objects_selected = ["book"]
        for obj in object_candidates:
            if init_goal_manager.rand.random() > 0.2:
                objects_selected.append(obj)
        # Get a list of the number of objecs we want to add

        '''init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1'''

        dishwasher_ids = [node["id"] for node in graph["nodes"] if node["class_name"] == "desk"]
        desired_id = init_goal_manager.rand.choice(dishwasher_ids)

        except_position_ids = [
            node["id"] for node in graph["nodes"] if ("floor" in node["class_name"])
        ]

        for obj in objects_selected:
            try:
                (
                    init_goal_manager.object_id_count,
                    graph,
                    success,
                ) = init_goal_manager.add_obj(
                    graph,
                    obj,
                    2,
                    init_goal_manager.object_id_count,
                    goal_obj=True,
                    enforced_adding=[desired_id, "ON"]
                )
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False
        

        # pdb.set_trace()
        if start:
            (
                init_goal_manager.object_id_count,
                graph,
            ) = init_goal_manager.setup_other_objs(
                graph,
                init_goal_manager.object_id_count,
                objs_in_room=None,
                except_position=except_position_ids + [desired_id],
                except_objects=object_candidates,
            )
        env_goal = {"clear_desk": []}
        for obj in objects_selected:
            locations = init_goal_manager.obj_position[obj]
            startIndex = init_goal_manager.rand.randint(0, len(locations) - 1)
            for i in range (len(locations)):
                index = (startIndex + i) % len(locations)
                predicate, location = locations[index]
                if location == "desk":
                    continue
                ids = [node["id"] for node in graph["nodes"] if node["class_name"] == location]
                if len(ids) == 0:
                    continue
                place = init_goal_manager.rand.choice(ids)
                env_goal["clear_desk"].append({"put_{}_{}_{}".format(predicate.lower(), obj, place): 2})
                break
        return graph, env_goal, True

    @staticmethod
    def double_task(init_goal_manager, graph, task_name):
        if "and" in task_name:
            tasks = task_name.split("_and_")
        else:
            tasks = [task_name, task_name]
        print(tasks)
        graph, env_goal_0, s1 = getattr(Task, tasks[0])(init_goal_manager, graph, start=True)
        if not s1:
            #ipdb.set_trace()
            return None, None, False
        graph, env_goal_1, s2 = getattr(Task, tasks[1])(init_goal_manager, graph, start=False)
        if not s2:
            #ipdb.set_trace()
            return None, None, False
        env_goal = {task_name: [], "noise": []}
        env_goal[task_name] = env_goal_0[tasks[0]]
        env_goal["noise"] = env_goal_1[tasks[1]]
        return graph, env_goal, True
    
