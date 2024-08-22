import os
import pickle
import json
import sys
import copy
import copy
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key=''
)
instruction_prompt = """
Objective: Create a description of a two-agent interaction scenario based on the provided language template.
User Input: A list of actions by each agent, Verbal communication between the agents.
Structure: Actions: A list of actions taken by agent 0 and agent 1, Language: Verbal communication between the agents in a list format.
Instructions:
1. Synchronization guidelines: Synchronize actions and language, the first entry in the "language" list corresponds to the first action step, the second entry in the "language" list corresponds to the second action and so on. If a language entry is null, there is no communication at that timestep. Synchronize descriptions of actions and language strictly by timesteps.
2. Agent names: Make sure to randomize the names. Use a random female name for agent 0 and a random male name for agent 1
3. Description guidelines: Describe the actions of agent 0 first, then agent 1. Clearly show this time sequence. Agent 1 would rearrange some objects placed by agent 0. Clearly show this by emphasizing that the object is placed by agent 0 previously when agent 1 move them (If two objects have exactly same ID, they are the same object, otherwise not). Avoid adjectives and excessive descriptions. Do not summarize the interaction process in the end. Do not skip any action or language steps.
4. After establishing the timeline, make the description shorter, more concise and flow a lot like a story.
"""

further_instructions = """
Note that the first action of agent 0 occurs at the same time as the first action of agent 1, and so on. If action section is null, it means agent does not take any action at this timestep. In this case, ignore action of this agent at this timestep. Never say some thing like "agent does nothing".
For example, reason through the user's input like this before generating the response. 
Actions: 0: ["[walk] <kitchen> (111)", "[walk] <fridge> (157)", "[open] <fridge> (157)", "[grab] <milk> (387)", "[close] <fridge> (157)", null, null, null, null, null]
1: [null, null, null, null, null, null, "[walk] <livingroom> (101)", "[walk] <table> (138)", "[grab] <book> (377)", "[putback] <book> (377) <table> (138)"]
Language: 0: [null, null, null, null, null, null, null, null, null, null] 1: [null, null, null, null, null, null, null, null, null, null]
Description:
First timestep: Lily walks into the kitchen while Tom does nothing.
Second timestep: Lily walks to a fridge while Tom does nothing.
Third timestep: Lily opens the fridge while Tom does nothing.
Fourth timestep: Lily grabs the milk while Tom does nothing.
Fifth timestep: Lily closes the fridge while Tom does nothing.
Sixth timestep: Both Lily and Tom do nothing.
Seventh timestep: Tom walks to the living room while Lily does nothing.
Eighth timestep: Tom walks to the table while Lily does nothing.
Ninth timestep: Tom grabs the book while Lily does nothing.
Tenth timtestep: Tom put the book on the table while Lily does nothing.
Generated response:
Lily walked into the kitchen and headed to the fridge. He opened fridge and grabbed a carton of milk from it, then closed the fridge. After that, Tom walked into the Kitchen and headed to the table. She grabbed the book and then put it down on the same table.
"""

user_prompt_1 = """
{"task_id":609,"goals":{"0":{"on_mug_135":{"count":1,"grab_obj_ids":[384],"container_ids":[135]},"on_spoon_135":{"count":1,"grab_obj_ids":[412,420],"container_ids":[135]},"on_wineglass_135":{"count":1,"grab_obj_ids":[411,421,430,435,436],"container_ids":[135]}},"1":{"on_juice_215":{"count":1,"grab_obj_ids":[173,174,419],"container_ids":[215]},"on_spoon_215":{"count":1,"grab_obj_ids":[412,420],"container_ids":[215]},"on_wineglass_215":{"count":1,"grab_obj_ids":[411,421,430,435,436],"container_ids":[215]}}},"action":{"0":["[walk] <kitchen> (11)","[walk] <kitchencabinet> (138)","[open] <stove> (163)","[open] <kitchencabinet> (145)","[open] <kitchencabinet> (144)","[grab] <spoon> (412)","[walk] <kitchentable> (135)","[putback] <spoon> (412) <kitchentable> (135)","[walk] <bedroom> (346)","[walk] <mug> (384)","[grab] <mug> (384)","[walk] <kitchen> (11)","[putback] <mug> (384) <kitchentable> (135)",null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"1":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,"[walk] <kitchen> (11)","[grab] <juice> (174)","[walk] <wineglass> (421)","[grab] <wineglass> (421)","[walk] <livingroom> (183)","[putback] <juice> (174) <coffeetable> (215)","[putback] <wineglass> (421) <coffeetable> (215)","[walk] <kitchen> (11)","[grab] <spoon> (412)","[walk] <livingroom> (183)","[putback] <spoon> (412) <coffeetable> (215)",null,null,null,null,null,null]},"finished":null,"language":{"0":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"1":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]},"have_belief":true,"false_belief_rooms":["kitchen"]}
"""

response_1 = """
John walked into the kitchen, opened a kitchen cabinet, stove, and then several other kitchen cabinets. He grabbed a spoon from last kitchen cabinet he opened and moved it to the kitchen table. He then walked into the bedroom, grabbed a mug and moved it back to kitchentable. Afterward, Tom walked into the kitchen, grabbed a cup of juice and a wineglass. She then walked into the living room and placed juice and wineglass on coffee table. She then go back to the kitchen, grab the spoon John previously placed on kitchen table, and moved it to coffeetable inside livingroom.
"""

user_prompt_2 = """
{"task_id":549,"goals":{"0":{"on_mug_111":{"count":1,"grab_obj_ids":[194,447],"container_ids":[111]},"on_spoon_111":{"count":1,"grab_obj_ids":[462,463,466],"container_ids":[111]},"on_wineglass_111":{"count":1,"grab_obj_ids":[197,198,298,299,454,456,464,465,467],"container_ids":[111]}},"1":{"on_juice_371":{"count":2,"grab_obj_ids":[460,461],"container_ids":[371]},"on_spoon_371":{"count":2,"grab_obj_ids":[462,463,466],"container_ids":[371]},"on_wineglass_371":{"count":2,"grab_obj_ids":[197,198,298,299,454,456,464,465,467],"container_ids":[371]}}},"action":{"0":["[walk] <mug> (447)","[grab] <mug> (447)","[walk] <bedroom> (73)","[putback] <mug> (447) <coffeetable> (111)","[walk] <livingroom> (335)","[grab] <spoon> (466)","[walk] <bedroom> (73)","[putback] <spoon> (466) <coffeetable> (111)",null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"1":[null,null,null,null,null,null,null,null,null,"[grab] <wineglass> (454)","[grab] <wineglass> (198)","[walk] <kitchen> (205)","[walk] <bathroom> (11)","[walk] <livingroom> (335)","[putback] <wineglass> (198) <coffeetable> (371)","[putback] <wineglass> (454) <coffeetable> (371)","[walk] <bedroom> (73)","[grab] <spoon> (466)","[walk] <livingroom> (335)","[putback] <spoon> (466) <coffeetable> (371)","[walk] <kitchen> (205)","[walk] <microwave> (313)","[open] <kitchencabinet> (237)","[grab] <spoon> (463)","[grab] <juice> (461)","[walk] <livingroom> (335)","[putback] <juice> (461) <coffeetable> (371)","[putback] <spoon> (463) <coffeetable> (371)","[walk] <kitchen> (205)","[walk] <kitchencabinet> (234)","[open] <stove> (311)","[open] <kitchencabinet> (234)","[open] <kitchencabinet> (235)","[grab] <juice> (460)","[walk] <livingroom> (335)","[putback] <juice> (460) <coffeetable> (371)",null,null,null,null,null,null]},"finished":null,"language":{"0":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"1":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]},"have_belief":true,"false_belief_rooms":[]}
"""

response_2 = """
Lily started by walking to a mug, grabbing it and moving it to coffeetable inside bedroom. He then walked into the livingroom, grabbed a spoon and also moved it to coffeetable inside bedroom. Afterward, Angela grab two wineglasses. After walking into kitchen, bathroom and livingroom in consecutive order, she placed two wineglasses on the coffee table inside livingroom. She then walked into the bedroom, grabbed spoon that Lily previously placed, and moved it to coffeetable. She then walked into the kitchen, open a kitchen cabinet and found a spoon and a cup of juice inside, and carried these objects to coffeetable inside kitchen. She walked back the kitchen then, searching stove and several kitchen cabinets, finally grabbing a cup of juice from last kitchen cabinet and moved it to the coffeetable inside livingroom.
"""
import os
import json
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{curr_dir}/../../online_watch_and_help/")
from agents import language
sys.path.insert(0, f"{curr_dir}/")
directory = "/home/scai/Workspace_2/hshi33/benchmark/texts/full_version"
def remove_sections(data, keys_to_remove):
    for key in keys_to_remove:
        if key in data:
            del data[key]
def main(episode_id):
    '''if os.path.exists(os.path.join(directory, "episode_{}.txt".format(episode_id))):
        print("Description of episode {} have been generated".format(episode_id))
        return'''
    pickle_file_path = '/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/nolang_episodes/logs_episode.{}_iter.0.pik'.format(episode_id)
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    json_file_path = '/home/scai/Workspace/hshi33/virtualhome/data/actions_pos.json'
    with open(json_file_path, "r") as json_file:
        action_data = json.load(json_file)
    data["action"] = action_data[str(episode_id)]["action"]
    data["language"] = action_data[str(episode_id)]["language"]
    for index, action in enumerate(data["action"]['0']):
        if action is None and data["action"]['1'][index] is None and data["action"]['1'][index + 1] is not None:
            bathroom_id = [node["id"] for node in data["init_unity_graph"]["nodes"] if node["class_name"] == "bathroom"][0]
            data["action"]['0'][index] = "[walk] <bathroom> ({})".format(bathroom_id)
            break
    keys_to_remove = ['gt_goals','init_unity_graph','plan','goals_finished','belief','belief_room','belief_graph', 'graph','obs', "env_id", "task_name", "language_object", "fail_to_execute"]
    remove_sections(data, keys_to_remove)
    information_file_path = f'/home/scai/Workspace_2/hshi33/benchmark/extracted_information/episode_{episode_id}.json'
    with open(information_file_path, 'w') as json_file:
        json.dump(data, json_file, separators=(',', ':'))
    with open(information_file_path, 'r') as json_file:
        final_prompt = json_file.read()
    print(final_prompt)


    response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "system", "content": further_instructions},
        {"role": "user", "content": user_prompt_1},
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": user_prompt_2},
        {"role": "assistant", "content": response_2},
        {"role": "user", "content": final_prompt}
    ],
    model="gpt-4o",
    temperature=1.0
    )

    episode_description = response.choices[0].message.content.strip()
    print(episode_description)
    

    output_path = os.path.join(directory, f'episode_{episode_id}.txt')

    with open(output_path,'w') as text_file:
        text_file.write(episode_description)
    
if __name__ == "__main__":
    episodes_list = [2709, 766, 1859, 404, 3367, 1917, 2726, 2808, 640, 549, 609, 538, 3289,  3096, 3412, 135, 2053, 801, 857, 144, 3082, 3325, 138, 3115, 3312, 642, 3068, 153, 1827, 3355, 3031, 3056, 3462, 263, 647, 557, 398, 1886, 784, 240, 3308, 913, 1931, 2079, 130, 634, 3058, 644, 1851, 195, 3475, 895, 1775, 3092, 780, 541, 147, 865, 389, 128, 154, 628, 3117, 3371, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 2584, 3292, 578, 2559, 571, 1802, 548, 225, 608, 218, 910, 2799, 3145, 42, 193, 2462, 3366, 211, 1758, 161, 1818, 3081, 949, 406, 3300, 3419, 397, 630, 417, 3130, 3315, 612, 3098, 3074, 1798, 121, 3166, 240, 2701, 149, 905, 801, 3119, 179, 3398, 869, 1131, 129, 1817, 871, 2748, 1140, 3328, 532, 583, 1819, 949, 1892, 152, 655, 824, 3080, 906, 2759, 1811, 3077, 550, 577, 790, 1858, 848, 3050, 3433, 1860, 3060, 3047, 3344, 3065, 528, 682, 931, 1765, 1813, 601, 1893, 767]
    #episodes_list = [135, 138, 153, 193, 195, 389, 532, 538, 577, 628, 630, 642, 647, 682, 766, 790, 848, 865, 871, 895, 905, 910, 956, 1811, 1856, 2559, 3050, 3058, 3068, 3092, 3117, 3119, 3315]
    #episodes_list = [42, 128, 130, 144, 154, 223, 393, 404, 528, 541, 548, 549, 557, 634, 790, 871, 905, 1758, 1818, 1858, 2053, 2070, 3074, 3092, 3098, 3129, 3130, 3292, 3308]
    episodes_list = [129, 152, 161, 212, 225, 263, 397, 578, 583, 601, 609, 640, 644, 682, 784, 801, 824, 857, 858, 913, 1131, 1817, 1819, 2462, 3077]
    episodes_list = [682]
    #episodes_list = [655]
    for episode in tqdm(episodes_list, desc='Processing Episodes'):
        main(episode)