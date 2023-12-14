import json
from tqdm import tqdm
import numpy as np
import inductive_reasoning
import deductive_reasoning
from utils import get_embedding, llm

with open(choice_example_path, "r", encoding='utf-8') as f:
    mix_choice_example = json.load(f)

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_icl_example_by_similarity(data):
    describe_embedding_whole = get_embedding(data['sentence'])
    icl_example_similarities = sorted([
        (vector_similarity(describe_embedding_whole, example['whole']), example) for example in mix_choice_example
    ], reverse=True, key=lambda x: x[0])
    return icl_example_similarities

def choice_reasoning_type(data):
    with open(template_path,'r',encoding='utf-8') as f:
        all_template = json.load(f)
        template = all_template['mix_reasoning']
    example = ''
    example_similarities = order_icl_example_by_similarity(data)
    for item in example_similarities[0:5]:
        example += 'Describe: ' + item[1]['sentence'] + '\nChoice: ' + item[1]['choice']  + '\n'
    question = 'Describe: ' + data['sentence'] + '\nChoice: '
    reasoning_type = llm('chatgpt',template['instruction'],template['example'] + example,question,None,None)
    if 'inductive' in reasoning_type.lower(): 
        reasoning_type = 'inductive_reasoning'
    elif 'deductive' in reasoning_type.lower():
        reasoning_type = 'deductive_reasoning'
    else:
        reasoning_type = 'inductive_reasoning'
    return reasoning_type

def reasoning(data,model_type,model=None,tokenizer=None,to_print=True):
    type_choice = choice_reasoning_type(data)
    if type_choice == 'inductive_reasoning':
        answer = inductive_reasoning.reasoning(data,model_type,model,tokenizer,to_print)
        if 'Finsih[unsure]' in answer:
            answer = deductive_reasoning.reasoning(data,model_type,model,tokenizer,to_print)
    elif type_choice == 'deductive_reasoning':
        answer = deductive_reasoning.reasoning(data,model_type,model,tokenizer,to_print)
        if 'Finsih[unsure]' in answer:
            answer = inductive_reasoning.reasoning(data,model_type,model,tokenizer,to_print)
    return answer
