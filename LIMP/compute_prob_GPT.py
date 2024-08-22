from openai import OpenAI
import json
import math
import re

client = OpenAI(
    api_key=''
)

def parse_latent_var(latent_var):
    belief = re.search(r'Belief:\s*(.*?)(?=\; Social goal)', latent_var).group(1)
    social_goal = re.search(r'Social goal:\s*(.*?)(?=\; Believed Goal)', latent_var).group(1)
    believed_goal = re.search(r'Believed Goal:\s*(.*)', latent_var).group(1)
    return {"Belief": belief, "Social Goal": social_goal, "Believed Goal": believed_goal}

def compute_prob(init_state, latent_var, info, main_person, prompt):
    latent_vars = parse_latent_var(latent_var)
    belief = latent_vars["Belief"]
    social_goal = latent_vars["Social Goal"]
    believed_goal = latent_vars["Believed Goal"]
    names = list(info.keys())
    other_name = [name for name in names if not name == main_person][0]
    if info[main_person]["utterance"] is not None:
        probability = compute_prob_utterance(other_name, main_person, info[other_name]["utterance"][0], info[main_person]["utterance"][0], social_goal, belief, believed_goal, None, exclude=["Believed_Goal"])
    else:
        probability = 1.0
    if info[main_person]["action"] is not None:
        for index, action in enumerate(info[main_person]["action"]):
            previous_actions = f"{other_name}'s actions:\n"
            for action1 in info[other_name]["action"]:
                previous_actions += action1
                previous_actions += "\n"
            previous_actions += f"{main_person}'s actions:\n"
            for i in range(index):
                previous_actions += info[main_person]["action"][i]
                previous_actions += "\n"
            prob = compute_prob_action(other_name, main_person, init_state, previous_actions, action, social_goal, belief, believed_goal)
            print(f"Probability of step {index}: {prob}")
            probability = probability * prob
    return probability
                



def compute_prob_utterance(name_agent_0, name_agent_1, utterance_agent_0, utterance_agent_1, a1_social_goal, a1_belief, a1_belief_of_goal, init_state, exclude=[]):
    evaluation_prompt = f"""
    {name_agent_1}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    """
    if "Believed_Goal" not in exclude:
        evaluation_prompt += f"{name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}\n"
    evaluation_prompt += f"{name_agent_0}'s Utterance': {utterance_agent_0}\n"
    if init_state is not None:
        evaluation_prompt += f"Initial state of environment: {init_state}\n"
    evaluation_prompt += f"""
    Based on the information, decide if it is likely for {name_agent_1} to say this word given conditions above. Compare the utterance and the belief of {name_agent_1}. 
    When trying to hinder, {name_agent_1} is likely to give different information with belief. For example, saying that some object is there when {name_agent_1} believe that there is some other things or nothing there, or the object is at a different place.
    Respond with only either A or B:
    {name_agent_1}'s Utterance: {utterance_agent_1}
    A) Likely
    B) Unlikely
    """
    #print("prompt:", evaluation_prompt)
    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )

    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None

    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']

    prob_a = math.exp(logprob_a) if logprob_a is not None else 0.0
    return prob_a


def compute_prob_action(name_agent_0, name_agent_1, init_state, previous_actions, a1_action, a1_social_goal, a1_belief, a1_belief_of_goal):
    evaluation_prompt = f"""
    Decide if {name_agent_1}'s action is likely with the information provided, respond with only either A or B:
    {name_agent_0}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    {name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}
    Initial state: {init_state}
    Check {name_agent_0}'s action to get the location of object when {name_agent_1} starts to act. 
    When {name_agent_1} tries to hinder, it's likely to grab object from its believed goal location for other agent, and unlikely to move objects to the believed goal location
    When {name_agent_1} tries to help, it's likely to grab object from somewhere else and put it to believed goal location, and unlikely to grab object from believed goal location
    Walking towards or grabbing from some unrelated location should be considered likely
    Previous Actions: {previous_actions}
    {name_agent_1}'s Action: {a1_action}
    A) Likely
    B) Unlikely
    """

    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )

    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None


    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']

    prob_a = math.exp(logprob_a) if logprob_a is not None else None
    return prob_a