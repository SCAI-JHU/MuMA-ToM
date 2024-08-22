import os
import pickle
import json
import sys
import copy
from openai import OpenAI
import argparse
from tqdm import tqdm

instruction_prompt = """
Objective: Create a description of a two-agent interaction scenario based on the provided language template.
User Input: A list of actions by each agent, Verbal communication between the agents.
Structure: Actions: A list of actions taken by agent 0 and agent 1, Language: Verbal communication between the agents in a list format.
Instructions:
1. Synchronization guidelines: Synchronize actions and language, the first entry in the "language" list corresponds to the first action step, the second entry in the "language" list corresponds to the second action and so on. If a language entry is null, there is no communication at that timestep. Synchronize descriptions of actions and language strictly by timesteps.
2. Agent names: Choose from a predefined list of common names. Use  a random name from ["Mary", "Sarah", "Jessica", "Emma"] for agent 0and  a random name from ["John", "Michael", "David", "Kevin"] for agent 1.
3. Description guidelines: Describe the actions and language of both agents together, step by step. Avoid adjectives and excessive descriptions. Do not skip any action or language steps.
4. After establishing the timeline, make the description shorter, more concise and flow a lot like a story. Do not skip any actions
5. Place more emphasis on the events immediately following the language conversation (if any)
6. When describing a action involved with grabbing objects, make sure to also include the original place of the object
"""

further_instructions = """
Note that the first action of agent 0 occurs at the same time as the first action of agent 1, and so on.
For example, reason through the user's input like this before generating the response. 
Actions: 0: ["[walk] <kitchen> (111)", "[walk] <fridge> (157)", "[open] <fridge> (157)", "[grab] <milk> (387)", "[close] <fridge> (157)", "[walk] <livingroom> (101)"]
1: ["[walk] <livingroom> (101)", "[walk] <table> (138)", "[grab] <book> (377)", "[putdown] <book> (377) <table> (138)", "[walk] <kitchen> (111)", null]
Language: 0: [null, null, "Have you seen the remote control?", null, null, null] 1: [null, null, "I saw a remote control on the sofa", null, null, null]
Description:
First timestep:  Jessica walks into the kitchen while Michael heads to the living room. (No language communication)
Second timestep:  Jessica walks to the fridge as Michael approaches the table. (No language communication)
Third timestep:  Jessica opens the fridge while Michael grabs a book from the table. "Have you seen the remote control?"  Jessica asks. "I saw a remote control on the sofa" Michael responds.
Fourth timestep:  Jessica grabs the milk from the fridge. Michael puts down the book.  (No language communication)
Fifth timestep:  Jessica closes the fridge. Michael walks to the kitchen (No language communication)
Sixth timestep:  Jessica walks into the living room (No language communication)
Generated response:
 Jessica walked into the kitchen and moved to the fridge as Michael headed to the living room as moved toward the table. "Have you seen the remote control?"  Jessica asked, opening the fridge. "I saw it on the sofa," Michael replied, grabbing the book from the table.  Jessica took the milk from the fridge and closed it, just as Michael put down the book and walked into the kitchen. Finally,  Jessica made her way to the living room.
"""

user_prompt_1 = """
{"task_id":742,"goals":{"0":{"inside_potato_162":{"count":1,"grab_obj_ids":[368],"container_ids":[162]}},"1":{"inside_bread_154":{"count":1,"grab_obj_ids":[369],"container_ids":[154]}}},"action":{"0":["[walk] <kitchen> (111)","[walk] <kitchencabinet> (145)","[open] <stove> (154)","[walk] <kitchencabinet> (144)","[open] <dishwasher> (156)","[walk] <kitchencabinet> (147)","[walk] <kitchencabinet> (144)","[open] <kitchencabinet> (143)","[open] <kitchencabinet> (149)","[open] <kitchencabinet> (142)","[open] <kitchencabinet> (148)","[open] <kitchencabinet> (147)","[grab] <potato> (368)","[walk] <microwave> (162)","[open] <microwave> (162)","[putin] <potato> (368) <microwave> (162)"],"1":["[walk] <kitchen> (111)","[walk] <kitchencabinet> (146)","[open] <kitchencabinet> (146)","[grab] <bread> (369)","[walk] <stove> (154)","[putin] <bread> (369) <stove> (154)",null,null,null,null,null,null,null,null,null,null]},"finished":true,"language":{"0":[null,"Do you know where the potato is?",null,null,null,null,null,null,null,null,null,null,null,null,null,null],"1":[null,"I found a potato inside the stove in the kitchen.",null,null,null,null,null,null,null,null,null,null,null,null,null,null]},"have_belief":true,"false_belief_rooms":[],"fail_to_execute":false}
"""

response_1 = """
Sarah and David both walked into the kitchen. While walking to a kitchen cabinet, Sarah asked "Do you know where the potato is?", and David answered "I found a potato inside the stove in the kitchen" while walking to a kitchen cabinet. Following David's instructions, Sarah walked to the stove, opened it and then headed to a kitchen cabinet, while simultaneously David opened a kitchen cabinet and grabbed a bread from it. David then walked to the stove and put the bread inside, while Sarah searched several kitchen cabinets and dishwasher. Finally, Sarah found potato from the last kitchen cabinet. She grabbed the potato and carried it to the microwave.
"""

user_prompt_2 = """
{"task_id":558,"goals":{"0":{"inside_wine_75":{"count":1,"grab_obj_ids":[369],"container_ids":[75]}},"1":{"on_magazine_277":{"count":1,"grab_obj_ids":[370],"container_ids":[277]}}},"action":{"0":["[walk] <kitchencabinet> (74)","[open] <stove> (105)","[walk] <kitchencabinet> (73)","[open] <kitchencabinet> (73)","[grab] <wine> (369)","[walk] <kitchencabinet> (75)","[open] <kitchencabinet> (75)","[putin] <wine> (369) <kitchencabinet> (75)"],"1":["[walk] <kitchen> (11)","[walk] <magazine> (370)","[grab] <magazine> (370)","[walk] <livingroom> (268)","[walk] <sofa> (277)","[putback] <magazine> (370) <sofa> (277)",null,null]},"finished":true,"language":{"0":[null,"Hey, do you know where the wine is?",null,null,null,null,null,null],"1":[null,"I found the wine inside the kitchen cabinet.",null,null,null,null,null,null]},"have_belief":true,"false_belief_rooms":[],"fail_to_execute":false}
"""

response_2 = """
Jessica walked to a kitchencabinet, while Michael walked into the kitchen. While opening the stove, Jessica asked, "Hey, do you know where the wine is?", and Michael respond "I found the wine inside the kitchen cabinet" while walking to a magazine. After ward, Jessica walked to a kitchen cabinet following Michael's instruction, found the wine there and carried the wind to a kitchen cabinet, while Michael grabbed the magazine and moved it to the sofa in the livingroom.

Jessica returned to the kitchen, opened the cabinet, and searched for the bread. Michael placed the carrot on the counter and opened the fridge. He grabbed a potato while Jessica continued searching. He placed the potato on the counter and grabbed another.

Jessica found the bread and put it in the fridge. Michael grabbed a potato and a carrot, placing them on the counter. Jessica checked the dishwasher and microwave as Michael continued adding vegetables to the counter.

Jessica briefly returned to the bedroom, then came back to the kitchen and the living room. She resumed searching the cabinets until she found the milk, which she placed in the fridge.
"""

def remove_sections(data, keys_to_remove):
    for key in keys_to_remove:
        if key in data:
            del data[key]

def find_trim_index(actions, communications, start_from):
    for i in range(start_from, len(actions)):
        if '[putback]' in actions[i]:
            return i
    return start_from # if no putback action is found, trim from the last communication index + 10

def trim_actions_10_steps_after_language(actions, communications):
    last_comm_index = max((i for i, comm in enumerate(communications) if comm), default=-1)
    return actions[:last_comm_index + 11], communications[:last_comm_index + 11] # Assuming we keep 1 action after the last communication

# Trim actions after the putback action, start seraching from the last communication index + 5
def trim_actions_after_putback(actions, communications):
    last_comm_index = max((i for i, comm in enumerate(communications) if comm), default=-1)
    start_search = last_comm_index + 5
    trim_index = find_trim_index(actions[0], communications, start_search)
    return actions[:trim_index + 1], communications[:trim_index + 1]
client = OpenAI(
    api_key=''
)
def generate(episode_id):
    pickle_file_path = '/home/scai/Workspace/hshi33/virtualhome/data/dataset_language_large/language/logs_episode.{}_iter.0.pik'.format(episode_id-4000)
    description_file_path = "/home/scai/Workspace_2/hshi33/benchmark/texts/full_version/episode_{}.txt".format(episode_id)
    '''if os.path.exists(description_file_path):
        print("Description of episode {} has been generated".format(episode_id))
        return'''

    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"File not found: {pickle_file_path}")

    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    
    
    json_file_path = '/home/scai/Workspace/hshi33/virtualhome/data/actions_pos.json'
    with open(json_file_path, "r") as json_file:
        action_data = json.load(json_file)
    data["action"] = action_data[str(episode_id)]["action"]
    data["language"] = action_data[str(episode_id)]["language"]

    keys_to_remove = ['gt_goals','init_unity_graph','plan','goals_finished','belief','belief_room','belief_graph', 'graph','obs', "env_id", "task_name", "language_object"]

    remove_sections(data, keys_to_remove)
    information_file_path = "/home/scai/Workspace_2/hshi33/benchmark/extracted_information/episode_{}.json".format(episode_id)
    with open(information_file_path, 'w') as json_file:
        json.dump(data, json_file, separators=(',', ':'))


    with open(information_file_path, 'r') as json_file:
        final_prompt = json_file.read()

    response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "system", "content": further_instructions},
        {"role": "user", "content": user_prompt_1},
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": user_prompt_2},
        {"role": "assistant", "content": response_2},
        {"role": "user", "content": final_prompt},
    ],
    model="gpt-4o",
    temperature=0.1
    )

    episode_description = response.choices[0].message.content.strip()

    print(episode_description)
    output_path = f'/home/scai/Workspace_2/hshi33/benchmark/texts/full_version/episode_{episode_id}.txt'

    exist = True
    ref_text = ""
    try:
        with open("/home/scai/Workspace_2/hshi33/benchmark/texts/multimodal/episode_{}.txt".format(episode_id), "r") as file:
            ref_text = file.read()
    except FileNotFoundError:
        exist = False
    male_list = ["John", "Michael", "David", "Kevin"]
    female_list = ["Mary", "Sarah", "Jessica", "Emma"]
    male_past = ""
    male_present = ""
    female_past = ""
    female_present = ""
    for male in male_list:
        if male in ref_text:
            male_past = male
        if male in episode_description:
            male_present = male
    for female in female_list:
        if female in ref_text:
            female_past = female
        if female in episode_description:
            female_present = female
    

    with open(output_path,'w') as text_file:
        if exist:
            episode_description = episode_description.replace(female_present, female_past)
            episode_description = episode_description.replace(male_present, male_past)
        else:
            episode_description = episode_description.replace(female_present, "Jessica")
            episode_description = episode_description.replace(male_present, "John")
        text_file.write(episode_description)
    
def main():
    parser = argparse.ArgumentParser(description='Process episodes with optional action trimming.')
    parser.add_argument('--trim_type', choices=['none', 'last_comm_10', 'putback'], default='none', help='Select the type of action trimming: none, trim after last communication plus 5 steps, or trim after putback action.')

    args = parser.parse_args()
    
    if args.trim_type == 'last_comm_10':
        new_dir = "/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/1500+episodes_trimmed_lang_10/"
        os.makedirs(new_dir, exist_ok=True)
    elif args.trim_type == 'putback':
        new_dir = "/home/scai/Workspace/hshi33/virtualhome/data/full_dataset/1500+episodes_trimmed_putback/"
        os.makedirs(new_dir, exist_ok=True)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, f"{curr_dir}/../../online_watch_and_help/") # comment out this line if run by caesar
    from agents import language
    sys.path.insert(0, f"{curr_dir}/")

    episodes_list = [5442, 5257, 5082, 5175, 5103, 4164, 4181, 5427, 5470, 4043, 4370, 5379, 4014, 4172, 4063, 4002, 4363, 4057, 4110, 4034, 5024, 4009, 5176, 5158, 4005, 4224, 4576, 5165, 4542, 5422, 5335, 5356, 4567, 5203, 5302, 4617, 4575, 4436, 4438, 4018, 4162, 4117, 5350, 4453, 5499, 5433, 4033, 5355, 4054, 4520, 5464, 4234, 5477, 4077, 5042, 4516, 4329, 5105, 4546, 4331, 4445, 5412, 5010, 5313, 5461, 4332, 4452, 4529, 4101, 5305, 4271, 4347, 4658, 5154, 5365, 4488, 4102, 4556, 4327, 4316, 4560, 5330, 4656, 4280, 4517, 4505, 4069, 4059, 4526, 4439, 5381, 4056, 4512, 4145, 4173, 4078, 4432, 5506, 5068, 4269, 5504, 4367, 4482, 4262, 5036, 4665, 4284, 5025, 4324, 5478, 5126, 4017, 4416, 4081, 4594, 4150, 5261, 5065, 4369, 5090, 4485, 5403, 4551, 4285, 5138, 4559, 5184, 4200, 5017, 4365, 5084, 4429, 5127, 4568, 4037, 4113, 5258, 4312, 4338, 5421, 4499, 4458, 5256, 5080, 5344, 5265, 5398, 5486, 4041, 5228, 4023, 5173, 4463, 4469, 4124, 4184, 4047, 4118, 5262, 4100, 4123, 4328, 4072, 4106, 5066, 4604, 4419, 4667, 5099, 5230, 5014, 4227, 5098, 5123, 4413, 4070, 4455, 5408, 4343, 5201, 4176, 5374, 4606, 4178, 4657, 4552, 5247, 5252, 4584, 4618, 5224, 4506, 4160, 5093, 4374, 4483, 5509, 4339, 4134, 5362, 4311, 4508, 4371, 4229, 4510, 5177]
    episodes_list = [5103]
    episodes_list = [4037, 4162, 4173, 4190, 4285, 4334, 4343, 4370, 4429, 4487, 4490, 4525, 4529, 4559, 4618, 4656, 5010, 5068, 5103, 5127]
    episodes_list = [
    4017, 4018, 4023, 4034, 4037, 4041, 4043, 4054, 4057, 4059, 4063, 4070, 4077, 4078, 4081, 4083, 4098,
    4103, 4105, 4106, 4124, 4145, 4150, 4162, 4166, 4172, 4184, 4190, 4198, 4200, 4224, 4284, 4324, 4327,
    4328, 4329, 4331, 4334, 4338, 4343, 4367, 4369, 4370, 4372, 4374, 4385, 4416, 4419, 4423, 4429, 4439,
    4441, 4449, 4452, 4453, 4455, 4469, 4473, 4482, 4485, 4487, 4488, 4490, 4499, 4505, 4506, 4510, 4512,
    4519, 4520, 4525, 4540, 4542, 4546, 4552, 4556, 4559, 4567, 4568, 4575, 4594, 4604, 4606, 4618, 4623,
    4641, 4656, 4658, 5010, 5017, 5039, 5042, 5068, 5080, 5084, 5091, 5095, 5099, 5103, 5105, 5121, 5138,
    5154, 5165, 5173, 5175, 5197, 5302, 5379, 5381, 5509
    ]
    episodes_list = [4005, 4009, 4047, 4101, 4102, 4113, 4117, 4123, 4133, 4140, 4160, 4164, 4173, 4176, 4178, 4280, 4285, 4312, 4324, 4328, 4331, 4332, 4365, 4375, 4415, 4458, 4463, 4526, 4527, 4529, 4551, 4560, 4576, 4584, 4621, 4667, 5014, 5049, 5082, 5093, 5098, 5123, 5126, 5127, 5163]
    for episode in tqdm(episodes_list, desc='Processing Episodes'):
        generate(episode)


if __name__ == "__main__":
    main()
