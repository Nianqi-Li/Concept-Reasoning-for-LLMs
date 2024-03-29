import json
from tqdm import tqdm
import numpy as np
from utils import get_embedding, llm

def get_surrounding_text(text, entity):
    texts = text.split(',')
    surrounding_begin_index = -1
    surrounding_end_index = -1
    entity_text_index = -1
    for i in range(len(texts)):
        if entity in texts[i]:
            surrounding_begin_index = i
            surrounding_end_index = i
            entity_text_index = i
            break
    for i in range(len(texts)):
        texts[i] = texts[i].strip(' ')
    if len(texts[entity_text_index].split(' ')) < 4:
        while surrounding_begin_index > 0 and len(texts[surrounding_begin_index].split(' ')) < 4:
            surrounding_begin_index = surrounding_begin_index - 1
        while surrounding_end_index < len(texts) - 1 and len(texts[surrounding_end_index].split(' ')) < 4:
            surrounding_end_index = surrounding_end_index + 1
    surrounding_text = ''
    for i in range(surrounding_begin_index, surrounding_end_index + 1):
        surrounding_text += texts[i] + ', '
    surrounding_text = surrounding_text.strip(', ')
    return surrounding_text

def get_two_sides_text(text, entity):
    surrounding_text = get_surrounding_text(text, entity)
    entity_index = surrounding_text.find(entity)
    left_text = surrounding_text[0:entity_index]
    right_text = surrounding_text[entity_index + len(entity):]
    return left_text, right_text

def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

def jaccard_similarity(data, example_data):
    left_text, right_text = get_two_sides_text(data['sentence'],data['entity'])
    example_left_text, example_right_text = get_two_sides_text(example_data['sentence'],example_data['entity'])
    left_jaccard_similarity = calculate_jaccard_similarity(left_text,example_left_text)
    right_jaccard_similarity = calculate_jaccard_similarity(right_text,example_right_text)
    jaccard_similarity = (left_jaccard_similarity + right_jaccard_similarity)/2
    return jaccard_similarity

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_icl_example_by_similarity(data, icl_example=icl_example):
    surrounding_text = get_surrounding_text(data['sentence'], data['entity'])
    describe_embedding_surrounding = get_embedding(surrounding_text)
    describe_embedding_whole = get_embedding(data['sentence'])
    icl_example_similarities = sorted([
        (vector_similarity(describe_embedding_surrounding, icl['surrounding']) + vector_similarity(describe_embedding_whole, icl['whole']), jaccard_similarity(data, icl), icl) for icl in icl_example
    ], reverse=True, key=lambda x: x[0])
    return icl_example_similarities

with open(icl_example_path, "r", encoding='utf-8') as f:
    icl_example = list(json.load(f))

def reasoning(data,model_type,model=None,tokenizer=None,icl_example=icl_example,to_print=True):
    icl_example_similarities = order_icl_example_by_similarity(data)
    with open(template_path,'r',encoding='utf-8') as f:
        all_template = json.load(f)
        template = all_template['inductive_reasoning']
    examples = ''
    icl_example_similarity = sorted([item for item in icl_example_similarities[0:5]],key=lambda x: x[1], reverse=True)
    for item in icl_example_similarity[0:3]:
        examples += item[2]['icl'] + '\n'
    if to_print:
        print('examples:\n ',examples)
    question = 'Describe: ' + data['sentence'] + '\nQuestion: Please infer what concept \"' + data['entity'] + '\" belong to?\nThought: '
    if to_print:
        print('question:\n',question)
    answer = llm(model_type,template['instruction'],examples,question,model,tokenizer,['Describe:'])
    if 'Finish' not in answer:
        answer = answer.strip('\n')
        answer2 = llm(model_type,template['instruction'],examples,question + answer + '\nFinish[',model,tokenizer,['\n'])
        answer = answer + '\nFinish[' + answer2
    if to_print:
        print('answer:\n',answer)
    return question + answer
