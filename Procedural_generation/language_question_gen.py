import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
client = OpenAI(
    api_key=''
)

template_question_1_1_most = """
Question:
Given the above interaction, if [second agent's name] has been trying to help [first agent's name] achieve [his/her] goal, which of the following statements is MOST likely to be true?
A) When giving information, [second agent's name] believed that there is a [object that second agent mentioned] [inside/on location second agent mentioned]
B) When giving information, [second agent's name] believed that there is a [object that appears in the episode but was not mentioned by the second agent] [inside\on the location second agent mentioned]
C) When giving information, [second agent's name] believed that there is a [object that does not show up in the episode] [inside\on the location second agent mentioned]

Answer: A) When giving information, [second agent's name] believed that there was a [object that second agent mentioned] [inside/on location second agent mentioned]
Label: 1.1
"""

template_question_1_2_least = """
Question:
Given the above interaction, if [second agent's name] has been trying to hinder [first agent's name] achieve [his/her] goal, which of the following statements is LEAST likely to be true?
A) When giving information, [second agent's name] believed that there is a [object that second agent mentioned] [inside/on location second agent mentioned the object was located]
B) When giving information, [second agent's name] believed that there is a [object that appears in the episode but was not mentioned by the second person] [inside/on location second agent mentioned the object was located]
C) When giving information, [second agent's name] believed that there is a [object that does not show up in the episode] [inside/on location second agent mentioned the object was located]
Answer: A) When giving information, [second agent's name] believed that there is a [object that second agent mentioned] [inside/on location second agent mentioned the object was located]
Label: 1.2
"""

template_question_2_1_most = """
Question:
Given the above interaction, assuming that [second agent's name] knows what is [inside\on a location that second agent mentioned in the communication], which of the following statement is MOST likely?
A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
C) When giving information, [second agent's name] has no particular inclination toward [first agent's goal] and just respond what her believed
Answer: A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
Label: 2.1
"""

template_question_2_1_least = """
Question:
Given the above interaction, assuming that [second agent's name] knows what is [inside\on a location that second agent mentioned in the communication], which of the following statement is LEAST likely?
A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
C) When giving information, [second agent's name] was indifferent towards [first agent's name]'s goals
Answer: B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
Label: 2.1
"""

template_question_2_2_most = """
Question:
Given the above interaction, assuming that [second agent's name] knows what is [inside\on a location that second agent mentioned in the communication], which of the following statement is MOST likely?
A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
C) When giving information, [second agent's name] was indifferent towards [first agent's name]'s goals
Answer: B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
Label: 2.2
"""

template_question_2_2_least = """
Question:
Given the above interaction, assuming that [second agent's name] knows what is [inside\on a location that second agent mentioned in the communication], which of the following statement is LEAST likely?
A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
B) When giving information, [second agent's name] has been trying to prevent [first agent's name] from finding [the object that second agent mentioned]
C) When giving information, [second agent's name] was indifferent towards [first agent's name]'s goals
Answer: A) When giving information, [second agent's name] has been trying to help [first agent's name] locate [the object that second agent mentioned]
Label: 2.2
"""

example_episode = """
Kevin walked into the kitchen while Emma followed. He moved towards a kitchen cabinet and asked, "Do you know where the potato is?" Emma replied, "I found a potato inside the stove in the kitchen," as she opened another kitchen cabinet and grabbed a bread.
Kevin opened the stove but didn't find the potato. He proceeded to check another kitchen cabinet and then the dishwasher. Emma walked to the stove with the bread and put it inside. Meanwhile, Kevin opened several kitchen cabinets, meticulously searching.
Finally, Kevin found the potato he was looking for. He walked over to the microwave, opened it, and placed the potato inside. Emma's task was already complete, and she left the kitchen.
"""

response_question_1 = """
Question:
Given the above interaction, if Emma has been trying to help Kevin achieve his goal, which of the following statements is MOST likely?
A) When giving information, Emma believed that there was a potato inside the stove
B) When giving information, Emma believed that there was bread inside the stove
C) When giving information, Emma believed that there was a tomato inside the stove
Answer: A) When giving information, Emma believed that there was a potato inside the stove
Label: 1.1
"""

response_question_4 = """
Question:
Given the above interaction, if Emma has been trying to hinder Kevin from achieving his goal, which of the following statements is LEAST likely?
A) When giving information, Emma believed that there was a potato inside the stove
B) When giving information, Emma believed that there was bread inside the stove
C) When giving information, Emma believed that there was a tomato inside the stove
Answer: A) When giving information, Emma believed that there was a potato inside the stove
Label: 1.2
"""

response_question_5 = """
Question:
Given the above interaction, assuming that Emma knows what is inside the stove, which of the following statements is MOST likely? 
A) When giving information, Emma has been trying to help Kevin locate the potato
B) When giving information, Emma has been trying to prevent Kevin from finding the potato
C) When giving information, Emma was indifferent towards Kevin's goals
Answer: A) When giving information, Emma has been trying to help Kevin locate the potato
Label: 2.1
"""

response_question_6 = """
Question:
Given the above interaction, assuming that Emma knows what is inside the stove, which of the following statements is LEAST likely? 
A) When giving information, Emma has been trying to help Kevin locate the potato
B) When giving information, Emma has been trying to prevent Kevin from finding the potato
C) When giving information, Emma was indifferent towards Kevin's goals
Answer: B) When giving information, Emma has been trying to prevent Kevin from finding the potato
Label: 2.1
"""

response_question_7 = """
Given the above interaction, assuming that Emma knows what is inside the stove, which of the following statements is MOST likely? 
A) When giving information, Emma has been trying to help Kevin locate the potato
B) When giving information, Emma has been trying to prevent Kevin from finding the potato
C) When giving information, Emma was indifferent towards Kevin's goals
Answer: B) When giving information, Emma has been trying to prevent Kevin from finding the potato
Label: 2.2
"""

response_question_8 = """
Question:
Given the above interaction, assuming that Emma knows what is inside the stove, which of the following statements is LEAST likely? 
A) When giving information, Emma has been trying to help Kevin locate the potato
B) When giving information, Emma has been trying to prevent Kevin from finding the potato
C) When giving information, Emma was indifferent towards Kevin's goals
Answer: A) When giving information, Emma has been trying to help Kevin locate the potato
Label: 2.2
"""

def generate(episode_id):
    instruction = """
    Objective: Generate questions from the language template by filling in the blank.
    User input: A description of the episode.
    Expected output: Questions following this templated format, filling in the blanks, denoted by [] where necessary.

    """
    input_file_path = '/home/scai/Workspace_2/hshi33/benchmark/texts/full_version/episode_{}.txt'.format(episode_id)
    question_file_path = "/home/scai/Workspace_2/hshi33/benchmark/questions/episode_{}.json".format(episode_id)
    '''if os.path.exists(question_file_path):
        print("Question of episode {} has been generated".format(episode_id))
        return'''
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File not found: {input_file_path}")

    with open(input_file_path, 'rb') as file:
        data = file.read()
    
    example_response = ""
    data = str(data)




    input_template_list_path = '/home/scai/Workspace/hshi33/virtualhome/data/dataset_language_large/language/question_types.json'
    episode_key = 'logs_episode.{}_iter.0.pik'.format(episode_id-4000)
    
    with open(input_template_list_path, 'rb') as file:
        template_list = json.load(file)
    
    if episode_key in template_list:
        templates_to_ask = template_list[episode_key]
   
    instruction += template_question_1_1_most
    example_response += response_question_1
    instruction += "\n"
    example_response += "\n"
    instruction += template_question_1_2_least
    example_response += response_question_4
    instruction += "\n"
    example_response += "\n"

    
    if "2.1.1" in templates_to_ask:
        instruction += template_question_2_1_most
        example_response += response_question_5
        instruction += "\n"
        example_response += "\n"
        instruction += template_question_2_1_least
        example_response += response_question_6
    else:
        instruction += template_question_2_2_most
        example_response += response_question_7
        instruction += "\n"
        example_response += "\n"
        instruction += template_question_2_2_least
        example_response += response_question_8
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": example_episode},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": data}
        ],
        model="gpt-4o",
        temperature=0.1
        )
    gpt_output = response.choices[0].message.content.strip()

    json_data = parse_gpt_response(gpt_output, data, episode_id)

    json_file_path = '/home/scai/Workspace_2/hshi33/benchmark/questions/episode_{}.json'.format(episode_id)

    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def parse_gpt_response(text, description, episode_num):
    question_pattern = re.compile(r'Question:\s*(.*?)\s*Answer:\s*(.*?)\s*Label:\s*(\d\.\d)\s*(?=Question:|\Z)', re.DOTALL)
    matches = question_pattern.findall(text)
    
    questions = {}
    answers = {}
    labels = {}
    
    count = 1
    for match in matches:
        question_content, answer_content, label_content = match
        questions[count] = question_content.strip()
        answers[count] = answer_content.strip()
        labels[count] = label_content.strip()
        count += 1

    return {
        episode_num: {
            "description": description,
            "questions": questions,
            "answers": answers,
            "labels": labels
        }
    }
if __name__ == "__main__":
    episodes_list = [4047, 4077, 4081, 4105, 4113, 4124, 4190, 4200, 4262, 4285, 4327, 4370, 4413, 4419, 4505, 4542, 4567, 4618, 5010, 5066]
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
    episodes_list = [4173, 4285, 4324, 4328, 4331, 4529, 5127]
    episodes_list = [212, 640]
    for episode in tqdm(episodes_list, desc='Processing Episodes'):
        generate(episode)