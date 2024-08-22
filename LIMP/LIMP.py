import os
import json
import text_parsing
import ast
import compute_prob_GPT
import scipy.special
import numpy as np
from openai import OpenAI
import visual_action_extraction
import ipdb
import random
from tqdm import tqdm
client = OpenAI(
    api_key=''
)

def extract_name_from_question(question):
    prompt = """You will read a question asking about a person's mental state or actions. From the prompt and options, extract any name of the people you encountered. Determine the person whose mental state or action the question is asking about. Produce your output in this form: [main person's name, name2, name3, ...]. Do not record names appearing multiple times, and do not give any extra information. An example question is like this:
    Example Question: Given that Emma has seen David walking to school yesterday, what will Emma most likely believe
    A David will walk to school tomorrow
    B David will drive to school tomorrow
    C David will not come to school tomorrow
    Example Output: ["Emma", "David"]

    Input Question: {}
    """


    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt.format(question)},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    temp_str = response.choices[0].message.content.strip()
    name_list = ast.literal_eval(temp_str)
    return name_list
def get_choice(final_prob, prompt):
    final_answer = f"""You will read a question with choices and likelihood of each statement for choices in a probability formal. Based on these information, answer the question and only include the letter of choice in your answer. 
    Question: {prompt}
    
    """
    choice_list = ["A", "B", "C", "D", "E"]
    for index, prob in enumerate(final_prob):
        final_answer += f"Probability of statement in choice {choice_list[index]} is True: {prob}\n"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": final_answer.format(prompt)},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    model_choice = response.choices[0].message.content.strip()[0]
    return model_choice
if __name__ == "__main__":
    correct = 0
    total = 0
    episode_list = [4005, 4009, 4017, 4018, 4023, 4034, 4037, 4041, 4043, 4054, 4057, 4059, 4063, 4070, 4077, 4078, 4081, 4083, 4098, 4103, 4105, 4106, 4124, 4145, 4150, 4162, 4172, 4184, 4190, 4198, 4200, 4284, 4324, 4327, 4331, 4338, 4343, 4367, 4369, 4370, 4372, 4374, 4385, 4416, 4419, 4423, 4429, 4439, 4441, 4449, 4452, 4453, 4469, 4473, 4482, 4485, 4487, 4488, 4490, 4499, 4505, 4506, 4510, 4512, 4520, 4525, 4540, 4542, 4546, 4552, 4556, 4559, 4567, 4568, 4594, 4604, 4606, 4618, 4623, 4641, 4656, 4658, 5010, 5017, 5039, 5042, 5068, 5080, 5084, 5091, 5095, 5099, 5103, 5105, 5121, 5138, 5154, 5165, 5173, 5175, 5197, 5302, 5379, 5381, 5509, 4047, 4101, 4102, 4113, 4117, 4123, 4133, 4140, 4160, 4173, 4176, 4178, 4280, 4285, 4312, 4328, 4332, 4365, 4415, 4458, 4463, 4526, 4527, 4529, 4551, 4560, 4576, 4584, 4621, 4667, 5014, 5049, 5082, 5093, 5098, 5123, 5126, 5127, 4455, 4375, 4164, 4224, 4329, 4575, 5163, 135, 138, 153, 193, 389, 532, 538, 577, 628, 630, 642, 647, 766, 848, 865, 895, 910, 956, 1811, 1856, 2559, 3050, 3058, 3068, 3315, 129, 152, 161, 225, 263, 397, 578, 583, 601, 609, 640, 644, 682, 784, 801, 824, 857, 913, 1131, 1817, 1819, 2462, 3077, 42, 128, 130, 144, 154, 223, 393, 404, 528, 541, 548, 549, 557, 634, 790, 871, 905, 1758, 1818, 2053, 2070, 3074, 3092, 3098, 3129, 3130, 3308]
    episode_list = [4416]
    with open("../Files/questions.json", "r") as file:
        question_data = json.load(file)
    with open("../Files/texts.json", "r") as file:
        text_data = json.load(file)
    for episode in tqdm(episode_list, "Answering questions"):
        try:
            print("Episode ", episode)
            questions = question_data[str(episode)]
            for question_id, prompt in questions["questions"].items():
                print("Question ", question_id)
                name_list = extract_name_from_question(questions["questions"]["1"])
                text = text_data[str(episode)]
                main_person = name_list[0]
                info = {}
                have_utterance = False
                for name in name_list:
                    person_info = text_parsing.parse_text_info(text, name)
                    if person_info["utterance"] is not None:
                        person_info["action"] = None
                        have_utterance = True
                    info[name] = person_info
                utterance = None
                if have_utterance:
                    utterance = {}
                    for name in info.keys():
                        utterance[name] = info[name]["utterance"]
                print(utterance)
                count = 0
                if episode > 4000:
                    info[name_list[1]]["action"] = visual_action_extraction.get_action(episode)
                else:
                    if info[main_person]["action"] is None:
                        info[main_person]["action"] = visual_action_extraction.get_action(episode)
                    else:
                        info[name_list[1]]["action"] = visual_action_extraction.get_action(episode)
                init_state, latent_var_options = text_parsing.latent_variable_extraction(info, prompt)
                prob_list = []
                choices = list(latent_var_options.keys())
                for choice, latent_var in latent_var_options.items():
                    probability = compute_prob_GPT.compute_prob(init_state, latent_var, info, main_person, prompt)
                    prob_list.append(probability)
                final_prob = scipy.special.softmax(prob_list)
                print(final_prob)
                model_choice = get_choice(final_prob, prompt)
                print("Model choose ", model_choice)
                print("Correct answer ", questions["answers"][question_id][0])
                if model_choice == questions["answers"][question_id][0]:
                    correct += 1
                total += 1
        except Exception as e:
            raise e
            print(str(e))
            print("Episode {} have error when processing".format(episode))
            continue
    print("Total accuracy rate: ", correct/total)
        