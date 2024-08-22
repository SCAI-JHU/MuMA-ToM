import re
import os
import random
def parse_string(s):
    # Define a regular expression pattern to match the parts of the string
    pattern = r'\[(.*?)\] <(.*?)> \((\d+)\) <(.*?)> \((\d+)\)'
    match = re.match(pattern, s)
    
    if match:
        # Extract the parts and convert the number to an integer
        action = match.group(1)
        device = match.group(2)
        number = match.group(3)
        device1 = match.group(4)
        number1 = match.group(5)
        return [action, device, number, device1, number1]
    
    else:
        # If the pattern does not match, return an empty list or raise an error
        pattern1 = r'\[(.*?)\] <(.*?)> \((\d+)\)'
        match = re.match(pattern1, s)
        if match:
            action = match.group(1)
            device = match.group(2)
            number = match.group(3)
            return [action, device, number]
def add_close(actions, graphs, languages):
    ans = {0: [], 1: []}
    room_name = ["bedroom", "livingroom", "kitchen", "bathroom"]
    graph = graphs[0]
    for index, action in enumerate(actions[0]):
        ans[0].append(action)
    for index, action in enumerate(actions[1]):
        ans[1].append(action)  
    room_ids = [node["id"] for node in graph["nodes"] if node["class_name"] in room_name]
    #add close after opening every container
    for index, action in enumerate(actions[0]):
        if action is None:
            continue
        temp = 0
        for new_index, new_action in enumerate(ans[0]):
            if new_action == action:
                temp = new_index
                break
        if "open" in action or "putin" in action:
            if "open" in action:
                container_name = parse_string(action)[1]
                container_id = parse_string(action)[2]
            else:
                container_name = parse_string(action)[3]
                container_id = parse_string(action)[4]
            if need_close(actions[0], index, container_id, graphs):
                if not index == len(actions[0]) - 1 and actions[0][index+1] is not None and ("grab" in actions[0][index+1] or "putin" in actions[0][index+1]):
                    ans[0].insert(temp+2, "[close] <{}> ({})".format(container_name, container_id))
                    languages[0].insert(temp+2, None)
                    ans[1].insert(temp+2, None)
                    languages[1].insert(temp+2, None)
                else:
                    ans[0].insert(temp+1, "[close] <{}> ({})".format(container_name, container_id))
                    languages[0].insert(temp+1, None)
                    ans[1].insert(temp+1, None)
                    languages[1].insert(temp+1, None)
    
    for index, action in enumerate(actions[1]):
        if action is None:
            continue
        temp = 0
        for new_index, new_action in enumerate(ans[1]):
            if new_action == action:
                temp = new_index
                break
        if "open" in action or "putin" in action:
            if "open" in action:
                container_name = parse_string(action)[1]
                container_id = parse_string(action)[2]
            else:
                container_name = parse_string(action)[3]
                container_id = parse_string(action)[4]
            if index == len(actions[1]) - 1:
                continue
            if need_close(actions[1], index, container_id, graphs):
                if not index == len(actions[1]) - 1 and actions[1][index+1] is not None and ("grab" in actions[1][index+1] or "putin" in actions[1][index+1]):
                    ans[1].insert(temp+2, "[close] <{}> ({})".format(container_name, container_id))
                    languages[1].insert(temp+2, None)
                    ans[0].insert(temp+2, None)
                    languages[0].insert(temp+2, None)
                else:
                    ans[1].insert(temp+1, "[close] <{}> ({})".format(container_name, container_id))
                    languages[1].insert(temp+1, None)
                    ans[0].insert(temp+1, None)
                    languages[0].insert(temp+1, None)
    #check every grab to avoid grabbing from a previously closed container
    for index, action in enumerate(actions[0]):
        if action is None:
            continue
        if "grab" in action or "putin" in action:
            if not index == 0:
                if actions[0][index-1] is not None:
                    if "grab" in actions[0][index - 1]:
                        continue
                state = graphs[index-1]
                grabbed_id = (int)(parse_string(action)[2])
                container = None
                if "grab" in action:
                    for edge in state["edges"]:
                        if edge["from_id"] == grabbed_id and edge["relation_type"] == "INSIDE" and edge["to_id"] not in room_ids:
                            container = edge["to_id"]
                            container_name = [node["class_name"] for node in state["nodes"] if node["id"] == container][0]
                            break
                else:
                    container = parse_string(action)[4]
                    container_name = parse_string(action)[3]
                if container is not None:
                    temp = 0
                    for new_index, new_action in enumerate(ans[0]):
                        if new_action == action:
                            temp = new_index
                            break
                    for i in range(temp):
                        if ans[1][i] is not None and (str)(container) in ans[1][i] and "close" in ans[1][i]:
                            ans[0].insert(temp, "[open] <{}> ({})".format(container_name, container))
                            languages[0].insert(temp, None)
                            ans[1].insert(temp, None)
                            languages[1].insert(temp, None)
                            if need_close(actions[0], index, (str)(container), graphs):
                                if not temp == len(ans[0]) - 2 and ans[0][temp+2] is not None and "grab" in ans[0][temp+2]:
                                    ans[0].insert(temp+3, "[close] <{}> ({})".format(container_name, container))
                                    languages[0].insert(temp+3, None)
                                    ans[1].insert(temp+3, None)
                                    languages[1].insert(temp+3, None)
                                else:
                                    ans[0].insert(temp+2, "[close] <{}> ({})".format(container_name, container))
                                    languages[0].insert(temp+2, None)
                                    ans[1].insert(temp+2, None)
                                    languages[1].insert(temp+2, None)
                                    break
                        if ans[0][i] is not None and (str)(container) in ans[0][i] and "close" in ans[0][i]:
                            ans[0].insert(temp, "[open] <{}> ({})".format(container_name, container))
                            languages[0].insert(temp, None)
                            ans[1].insert(temp, None)
                            languages[1].insert(temp, None)
                            if need_close(actions[0], index, (str)(container), graphs):
                                if not temp == len(ans[0]) - 2 and ans[0][temp+2] is not None and "grab" in ans[0][temp+2]:
                                    ans[0].insert(temp+3, "[close] <{}> ({})".format(container_name, container))
                                    languages[0].insert(temp+3, None)
                                    ans[1].insert(temp+3, None)
                                    languages[1].insert(temp+3, None)
                                else:
                                    ans[0].insert(temp+2, "[close] <{}> ({})".format(container_name, container))
                                    languages[0].insert(temp+2, None)
                                    ans[1].insert(temp+2, None)
                                    languages[1].insert(temp+2, None)
                                    break
        elif "walk" in action:
            if not index == 0:
                state = graphs[index-1]
                grabbed_id = (int)(parse_string(action)[2])
                container = None
                for edge in state["edges"]:
                    if edge["from_id"] == grabbed_id and edge["relation_type"] == "INSIDE" and edge["to_id"] not in room_ids:
                        container = edge["to_id"]
                        container_name = [node["class_name"] for node in state["nodes"] if node["id"] == container][0]
                        break
                if container is not None:
                    temp = 0
                    for new_index, new_action in enumerate(ans[0]):
                        if new_action == action:
                            temp = new_index
                            break
                    ans[0][temp] = "[walk] <{}> ({})".format(container_name, container)
    for index, action in enumerate(actions[1]):
        if action is None:
            continue
        if "grab" in action or "putin" in action:
            if not index == 0:
                if actions[1][index-1] is not None:
                    if "grab" in actions[1][index - 1]:
                        continue
                state = graphs[index-1]
                grabbed_id = (int)(parse_string(action)[2])
                container = None
                if "grab" in action:
                    for edge in state["edges"]:
                        if edge["from_id"] == grabbed_id and edge["relation_type"] == "INSIDE" and edge["to_id"] not in room_ids:
                            container = edge["to_id"]
                            container_name = [node["class_name"] for node in state["nodes"] if node["id"] == container][0]
                else:
                    container = parse_string(action)[4]
                    container_name = parse_string(action)[3]
                if container is not None:
                    temp = 0
                    for new_index, new_action in enumerate(ans[1]):
                        if new_action == action:
                            temp = new_index
                            break
                    for i in range(temp):
                        if ans[0][i] is not None and (str)(container) in ans[0][i] and "close" in ans[0][i]:
                            ans[1].insert(temp, "[open] <{}> ({})".format(container_name, container))
                            languages[1].insert(temp, None)
                            ans[0].insert(temp, None)
                            languages[0].insert(temp, None)
                            if need_close(actions[1], index, (str)(container), graphs):
                                if not temp == len(ans[1]) - 2 and ans[1][temp+2] is not None and "grab" in ans[1][temp+2]:
                                    ans[1].insert(temp+3, "[close] <{}> ({})".format(container_name, container))
                                    languages[1].insert(temp+3, None)
                                    ans[0].insert(temp+3, None)
                                    languages[0].insert(temp+3, None)
                                else:
                                    ans[1].insert(temp+2, "[close] <{}> ({})".format(container_name, container))
                                    languages[1].insert(temp+2, None)
                                    ans[0].insert(temp+2, None)
                                    languages[0].insert(temp+2, None)
                                break
                        if ans[1][i] is not None and (str)(container) in ans[1][i] and "close" in ans[1][i]:
                            ans[1].insert(temp, "[open] <{}> ({})".format(container_name, container))
                            languages[1].insert(temp, None)
                            ans[0].insert(temp, None)
                            languages[0].insert(temp, None)
                            if need_close(actions[1], index, (str)(container), graphs):
                                if not temp == len(ans[1]) - 2 and ans[1][temp+2] is not None and "grab" in ans[1][temp+2]:
                                    ans[1].insert(temp+3, "[close] <{}> ({})".format(container_name, container))
                                    languages[1].insert(temp+3, None)
                                    ans[0].insert(temp+3, None)
                                    languages[0].insert(temp+3, None)
                                else:
                                    ans[1].insert(temp+2, "[close] <{}> ({})".format(container_name, container))
                                    languages[1].insert(temp+2, None)
                                    ans[0].insert(temp+2, None)
                                    languages[0].insert(temp+2, None)
                                    break
        elif "walk" in action:
            if not index == 0:
                state = graphs[index-1]
                grabbed_id = (int)(parse_string(action)[2])
                container = None
                for edge in state["edges"]:
                    if edge["from_id"] == grabbed_id and edge["relation_type"] == "INSIDE" and edge["to_id"] not in room_ids:
                        container = edge["to_id"]
                        container_name = [node["class_name"] for node in state["nodes"] if node["id"] == container][0]
                        break
                if container is not None:
                    temp = 0
                    for new_index, new_action in enumerate(ans[1]):
                        if new_action == action:
                            temp = new_index
                            break
                    ans[1][temp] = "[walk] <{}> ({})".format(container_name, container)
    return ans
def get_chars(episode_id):
    male_names = ["John", "Michael", "David", "Kevin"]
    female_names = ["Mary", "Sarah", "Jessica", "Emma"]

    male_chars = ["Chars/Male1", "Chars/Male2", "Chars/Male1", "Chars/Male6"]
    female_chars = ["Chars/Female1", "Chars/Female2", "Chars/Female4", "Chars/Female4"]
    
    if episode_id < 4000:
        trimmed_path = f"/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/GPT/episode_descriptions_no_lang/episode_{episode_id}.txt"
    else:
        trimmed_path = f"/home/scai/Workspace/hshi33/virtualhome/online_watch_and_help/GPT/episode_descriptions/episode_{episode_id}.txt"

    if os.path.exists(trimmed_path):
        with open(trimmed_path, 'r') as file:
            content = file.read()
    else:
        return (random.choice(male_chars), random.choice(female_chars))
    
    found_male = None
    found_female = None
    
    for name in male_names:
        if name in content:
            found_male = name
            print("Male: ", found_male)
            break
    
    for name in female_names:
        if name in content:
            found_female = name
            print("Female: ", found_female)
            break
    
    if found_male and found_female:
        return (male_chars[male_names.index(found_male)], female_chars[female_names.index(found_female)])
    else:
        return (random.choice(male_chars), random.choice(female_chars))
def need_close(actions, index, container, graphs):
    for temp, action in enumerate(actions):
        if action is None:
            continue
        if temp <= index:
            continue
        if container in action:
            return False
    return True
if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    import json
    import ipdb
    episode_list = [766, 2709, 1859, 404, 3367, 1917, 2726, 2808, 640, 549, 609, 538, 3289,  3096, 3412, 135, 2053, 801, 857, 144, 3082, 3325, 138, 3115, 3312, 642, 3068, 153, 1827, 3355, 3031, 3056, 3462, 263, 647, 557, 398, 1886, 784, 240, 3308, 913, 1931, 2079, 130, 634, 3058, 644, 1851, 195, 3475, 895, 1775, 3092, 780, 541, 147, 865, 389, 128, 154, 628, 3117, 3371, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 2584, 3292, 578, 2559, 571, 1802, 548, 225, 608, 218, 910, 2799, 3145, 42, 193, 2462, 3366, 211, 1758, 161, 1818, 3081, 949, 406, 3300, 3419, 397, 630, 956, 417, 3130, 3315, 612, 3098, 3074, 1798, 121, 3166, 240, 2701, 149, 905, 801, 3119, 179, 3398, 869, 1131, 129, 1817, 871, 2748, 1140, 3328, 532, 583, 1819, 949, 1892, 152, 655, 824, 3080, 906, 2759, 1811, 3077, 550, 577, 2813, 790, 1858, 848, 3050, 3433, 1860, 767, 3060, 3047, 3344, 3065, 528, 682, 2045, 931, 3323, 3140, 1765, 1813, 3536, 601, 1893] #without language
    language_list = [5442, 5257, 5082, 5175, 5103, 4164, 4181, 5427, 4375, 5470, 4043, 4370, 5379, 4198, 4014, 4172, 4140, 4525, 4063, 4415, 4002, 4363, 4423, 4323, 4057, 4471, 4110, 4034, 4441, 5024, 4009, 5187, 5176, 4558, 4652, 5158, 4005, 4224, 4576, 5165, 4542, 5422, 4527, 5335, 5356, 4567, 5203, 5302, 5474, 4617, 4575, 4436, 4438, 4018, 4162, 4086, 4117, 5350, 4453, 5499, 5433, 4033, 5355, 4054, 4520, 5464, 4234, 5477, 4077, 5042, 4516, 4329, 5105, 4546, 4331, 4445, 5412, 5010, 5313, 5461, 4332, 4452, 4529, 4101, 5305, 4103, 4271, 4347, 4658, 5154, 5365, 4488, 4102, 4519, 4556, 4327, 5241, 4316, 4560, 5330, 4656, 4280, 4540, 5303, 4517, 4505, 4069, 5318, 4372, 5197, 4059, 4526, 5001, 4439, 5381, 4056, 4512, 4145, 5095, 4341, 4487, 4173, 4078, 5240, 4432, 5091, 5506, 5068, 4269, 5504, 4367, 4482, 4262, 5246, 5036, 4665, 5220, 4284, 5025, 5410, 4324, 5478, 5126, 4133, 5049, 4017, 4416, 4081, 4083, 5163, 4594, 4150, 5236, 5261, 5065, 4369, 5090, 5222, 4485, 5403, 4449, 4551, 4285, 4623, 5138, 4559, 5184, 4200, 4105, 5017, 4365, 5084, 4429, 5127, 4568, 4037, 4473, 4113, 5258, 4385, 4312, 4338, 5421, 5481, 4499, 4458, 5256, 5080, 5344, 5265, 5398, 5486, 4041, 5228, 4023, 5173, 4463, 4469, 4621, 4124, 4184, 4047, 4118, 5262, 5229, 4100, 5413, 4123, 4334, 4328, 4072, 4106, 5066, 4604, 4419, 5354, 4667, 5099, 5230, 5014, 4227, 4641, 5098, 4355, 5123, 4413, 4070, 4455, 5392, 5408, 4343, 5201, 4176, 4166, 5374, 4098, 5075, 4606, 4178, 4657, 4552, 5247, 5252, 4584, 4618, 5224, 4506, 4160, 5121, 5093, 4374, 4490, 5341, 4483, 5509, 4339, 4190, 4134, 5362, 4311, 4508, 5037, 4371, 4229, 4510, 5177]
    episode_list += language_list
    saved = {}
    for episode in tqdm(episode_list, "Filtering actions"):
        if episode in language_list:
            log_file = "dataset_language_large/language/logs_episode.{}_iter.0.pik".format(episode-4000)
        else:
            log_file = "full_dataset/nolang_episodes/logs_episode.{}_iter.0.pik".format(episode)
        with open(log_file, "rb") as file:
            data = pickle.load(file)
        actions = add_close(data["action"], data["graph"], data["language"])
        saved[episode] = {"action": actions, "language": data["language"]}
    with open("actions_pos.json", "w") as json_file:
        json.dump(saved, json_file, indent=4)