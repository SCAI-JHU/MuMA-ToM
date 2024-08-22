import os
import json
import re
import random
from openai import OpenAI
from tqdm import tqdm
client = OpenAI(
    api_key=''
)
instruction = """
Objective: Generate 3 likely and 2 unlikely choices from the language template by filling in the blank. Remember the male and female name in the episode.
Expected output: Choices following this templated format, filling in the blanks, denoted by [] where necessary.
Additional output: Cut the input episode and only keep female agent's actions. Output episode description after cutting.

Description:
[Cut description]

Likely:
1. [Female agent's name] believed that [male agent's name] wants to place the [object that both agents moved] [on/inside the place second agent placed the object]: She moved [object that both agents moved] to help [male agent's name].
2. [Female agent's name] believed that [male agent's name] placed the [object that both agents moved (same with previous one)] at his desired location: She intentionally moved [object that both agents moved] to the [place that second agent moved the object to] to hinder [male agent's name].
3. [Female agent's name] doesn't know [male agent's name]'s goal and moves the [object that both agents moved (same with previous one)] without thinking about what he wants

Unlikely:
1. [Female agent's name] believed that [male agent's name] placed the [object that both agents moved (same with previous one)] at his desired location: She moved [object that both agents moved (same with previous one)] to the [place that second agent moved the object to] to help [male agent's name].
2. [Female agent's name] believed that [male agent's name] wants to place the [object that both agents moved (same with previous one)] [on/inside the place second agent placed the object]: She intentionally moved [object that both agents moved (same with previous one)] to hinder [male agent's name].
"""

example_episode = """
David walked into the kitchen, grabbed a spoon, and headed to the living room. He placed the spoon on the coffee table in the living room. He then walked to the bedroom. Sophia headed to the living room, grabbed the spoon placed by David, walked back to the kitchen and placed that inside the dishwasher.
"""

example_response = """
Description:
Sophia headed to the living room, grabbed the spoon placed by David, walked back to the kitchen and placed that inside the dishwasher.

Likely: 
1. Sophia believed that David wants to place spoon inside dishwasher: she moved the spoon to help David.
2. Sophia believed that David placed the spoon at his desired location: she intentionally moved the spoon to the dishwasher to hinder David.
3. Sophia doesn't know David's goal, and is moving the spoon without thinking about what he wants.

Unlikely:
1. Sophia believed that David placed the spoon at his desired location: she moved the spoon to the dishwasher to help David.
2. Sophia believed David wants to place the spoon inside the dishwasher: she intentionally moved the spoon to hinder David.
"""

example_episode_2 = """
Michael walked into the kitchen. He proceeded to the stove, opened it, and grabbed a potato. He closed the stove and walked to the kitchen table, placing the potato there. Then, he walked to the bathroom. Later, Emily went to where Michael placed the potato, grabbed it, and moved it to the kitchen counter.
"""

example_response_2 = """
Description:
Later, Emily went to where Michael placed the potato, grabbed it, and moved it to the kitchen counter.

Likely:
1. Emily believed that Michael wants to place the potato on kitchen counter: she moved the potato to the kitchen counter help Michael
2. Emily believed that Michael placed the potato at his desired location: she intentionally moved the potato to the kitchen counter to hinder Michael
3. Emily doesn't know Michael's goal and moves the potato without thinking about what he wants.

Unlikely:
1. Emily believed that Michael placed the potato at his desired location: she moves the potato to the kitchen counter to help Michael
2. Emily believed that Michael wants to place the potato on the kitchen counter: she moved the potato to hinder Michael.
"""

def parse_gpt_response(text, description, episode_num):
    description_pattern = re.compile(r"Description:\s*(.*?)\s*Likely:", re.DOTALL)
    likely_pattern = re.compile(r"Likely:\s*(.*?)\s*Unlikely", re.DOTALL)
    unlikely_pattern = re.compile(r"Unlikely\s*(.*)", re.DOTALL)

    description_match = description_pattern.search(text)
    filtered_description = description_match.group(1).strip() if description_match else ""

    likely_match = likely_pattern.search(text)
    likely_text = likely_match.group(1).strip() if likely_match else ""
    likely_choices = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', likely_text, re.DOTALL)

    unlikely_match = unlikely_pattern.search(text)
    unlikely_text = unlikely_match.group(1).strip() if unlikely_match else ""
    unlikely_choices = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', unlikely_text, re.DOTALL)

    most_question = "Given the above interaction, based on the actions of the agents, which of the following statements is MOST likely?"
    least_question = "Given the above interaction, based on the actions of the agents, which of the following statements is LEAST likely?"

    random_likely_choice_mostQ = random.choice(likely_choices)
    random_unlikely_choices_mostQ = random.sample(unlikely_choices, 2)
    all_choices_mostQ = [random_likely_choice_mostQ] + random_unlikely_choices_mostQ
    random.shuffle(all_choices_mostQ)

    random_likely_choices_leastQ = random.sample(likely_choices, 2)
    random_unlikely_choice_leastQ = random.choice(unlikely_choices)
    all_choices_leastQ = random_likely_choices_leastQ + [random_unlikely_choice_leastQ]
    random.shuffle(all_choices_leastQ)

    letters = ['A', 'B', 'C']
    labeled_choices_mostQ = {letters[i]: choice for i, choice in enumerate(all_choices_mostQ)}
    labeled_choices_leastQ = {letters[i]: choice for i, choice in enumerate(all_choices_leastQ)}

    correct_answer_mostQ = next((f"{letter}) {choice}" for letter, choice in labeled_choices_mostQ.items() if choice == random_likely_choice_mostQ), None)
    correct_answer_leastQ = next((f"{letter}) {choice}" for letter, choice in labeled_choices_leastQ.items() if choice == random_unlikely_choice_leastQ), None)

    filtered_choices = [choice for choice in likely_choices if choice != random_likely_choice_mostQ]
    random_likely_choice_mostQ_2 = random.choice(filtered_choices)
    random_unlikely_choices_mostQ_2 = random.sample(unlikely_choices, 2)
    all_choices_mostQ_2 = [random_likely_choice_mostQ_2] + random_unlikely_choices_mostQ_2
    random.shuffle(all_choices_mostQ_2)

    random_likely_choices_leastQ_2 = random.sample(likely_choices, 2)
    filtered_unlikely_choices = [choice for choice in unlikely_choices if choice != random_unlikely_choice_leastQ]
    random_unlikely_choice_leastQ_2 = random.choice(filtered_unlikely_choices)
    all_choices_leastQ_2 = random_likely_choices_leastQ_2 + [random_unlikely_choice_leastQ_2]
    random.shuffle(all_choices_leastQ_2)

    letters = ['A', 'B', 'C']
    labeled_choices_mostQ = {letters[i]: choice for i, choice in enumerate(all_choices_mostQ)}
    labeled_choices_leastQ = {letters[i]: choice for i, choice in enumerate(all_choices_leastQ)}
    labeled_choices_mostQ_2 = {letters[i]: choice for i, choice in enumerate(all_choices_mostQ_2)}
    labeled_choices_leastQ_2 = {letters[i]: choice for i, choice in enumerate(all_choices_leastQ_2)}

    correct_answer_mostQ = next((f"{letter}) {choice}" for letter, choice in labeled_choices_mostQ.items() if choice == random_likely_choice_mostQ), None)
    correct_answer_leastQ = next((f"{letter}) {choice}" for letter, choice in labeled_choices_leastQ.items() if choice == random_unlikely_choice_leastQ), None)
    correct_answer_mostQ_2 = next((f"{letter}) {choice}" for letter, choice in labeled_choices_mostQ_2.items() if choice == random_likely_choice_mostQ_2), None)
    correct_answer_leastQ_2 = next((f"{letter}) {choice}" for letter, choice in labeled_choices_leastQ_2.items() if choice == random_unlikely_choice_leastQ_2), None)

    questions = {
        "1": f"{most_question.rstrip()}\n" + "\n".join([f"{letter}) {choice.strip()}" for letter, choice in labeled_choices_mostQ.items() if choice.strip()]),
        "2": f"{least_question.rstrip()}\n" + "\n".join([f"{letter}) {choice.strip()}" for letter, choice in labeled_choices_leastQ.items() if choice.strip()]),
        "3": f"{most_question.rstrip()}\n" + "\n".join([f"{letter}) {choice.strip()}" for letter, choice in labeled_choices_mostQ_2.items() if choice.strip()]),
        "4": f"{least_question.rstrip()}\n" + "\n".join([f"{letter}) {choice.strip()}" for letter, choice in labeled_choices_leastQ_2.items() if choice.strip()])
    }

    answers = {
        "1": correct_answer_mostQ,
        "2": correct_answer_leastQ,
        "3": correct_answer_mostQ_2,
        "4": correct_answer_leastQ_2
    }

    labels = {
        "1": "3.1",
        "2": "3.2",
        "3": "3.1",
        "4": "3.2",
    }

    with open("/home/scai/Workspace_2/hshi33/benchmark/texts/multimodal/episode_{}.txt".format(episode_num), "w") as file:
        file.write(filtered_description)

    return {
        episode_num: {
            "description": description,
            "filtered_description": filtered_description,
            "questions": questions,
            "answers": answers,
            "labels": labels
        }
    }

def main(episode_id):
    
    input_file_path = '/home/scai/Workspace_2/hshi33/benchmark/texts/full_version/episode_{}.txt'.format(episode_id)
    question_file_path = "/home/scai/Workspace_2/hshi33/benchmark/questions/episode_{}.json".format(episode_id)
    '''if os.path.exists(question_file_path):
        print("Question of episode {} has been generated".format(episode_id))
        return'''
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File not found: {input_file_path}")

    with open(input_file_path, 'rb') as file:
        data = file.read()

    data = str(data)
    print(data)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": example_episode},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": example_episode_2},
            {"role": "assistant", "content": example_response_2},
            {"role": "user", "content": data}
        ],
        model="gpt-4o",
        temperature=0.1
        )
    gpt_output = response.choices[0].message.content.strip()
    print(gpt_output)
    parse_gpt_response(gpt_output, data, episode_id)

    
    json_data = parse_gpt_response(gpt_output, data, episode_id)
    print(json_data)
    json_file_path = question_file_path
    
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    episodes_list = [2709, 766, 1859, 404, 3367, 1917, 2726, 2808, 640, 549, 609, 538, 3289, 3096, 3412, 135, 2053, 801, 857, 144, 3082, 3325, 138, 3115, 3312, 642, 3068, 153, 1827, 3355, 3031, 3056, 3462, 263, 647, 557, 398, 1886, 784, 240, 3308, 913, 1931, 2079, 130, 634, 3058, 644, 1851, 195, 3475, 895, 1775, 3092, 780, 541, 147, 865, 389, 128, 154, 628, 3117, 3371, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 2584, 3292, 578, 2559, 571, 1802, 548, 225, 608, 218, 910, 2799, 3145, 42, 193, 2462, 3366, 211, 1758, 161, 1818, 3081, 949, 406, 3300, 3419, 397, 630, 417, 3130, 3315, 612, 3098, 3074, 1798, 121, 3166, 240, 2701, 149, 905, 801, 3119, 179, 3398, 1131, 129, 1817, 871, 1140, 3328, 532, 583, 1819, 949, 1892, 152, 655, 824, 3080, 906, 1811, 3077, 577, 790, 1858, 848, 3050, 3433, 1860, 3060, 3047, 3344, 3065, 528, 682, 931, 1765, 1813, 601, 1893]
    episodes_list = [128, 784, 2726, 3082, 195, 628, 2070, 3031]
    episodes_list = [130, 161, 397, 404, 541, 647, 865, 871, 1758, 1817, 1827, 2053, 2799, 3050, 3068, 3092, 3119, 3130, 3355, 824]
    episodes_list = [193, 212, 389, 630, 640, 655, 766, 865, 910, 1758, 1811, 1858, 1893, 2559, 3050] #, 1758, 1817, 1827, 2053, 2799, 3050, 3068, 3092, 3119, 3130, 3355, 824]
    #episodes_list = [130, 541, 766, 865]
    #episodes_list = [135, 138, 153, 193, 195, 389, 532, 538, 577, 628, 630, 642, 647, 766, 848, 865, 895, 910, 956, 1811, 1856, 2559, 3050, 3058, 3068, 3117, 3119, 3315]
    episodes_list = [42, 128, 130, 144, 154, 223, 393, 404, 528, 541, 548, 549, 557, 634, 790, 871, 905, 1758, 1818, 1858, 2053, 2070, 3074, 3092, 3098, 3129, 3130, 3292, 3308]
    #episodes_list = [129, 152, 161, 212, 225, 263, 397, 578, 583, 601, 609, 640, 644, 682, 784, 801, 824, 857, 858, 913, 1131, 1817, 1819, 2462, 3077]
    #episodes_list = [790, 871, 905, 3092]
    episodes_list = [212, 640]
    for episode in tqdm(episodes_list, desc='Processing Episodes'):
        print("Episode: {}".format(episode))
        main(episode)

