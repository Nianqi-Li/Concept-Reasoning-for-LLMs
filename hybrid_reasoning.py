import json
import openai
from tqdm import tqdm
import numpy as np
import inductive_reasoning
import deductive_reasoning

openai.api_key = 'sk-*'
GENERATING_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(model=model,input=text)
    return result["data"][0]["embedding"]

def llm(model_type,instruction,examples,question,model=None,tokenizer=None,stop_words=['\n']):
    if model_type == 'chatgpt':
        return chatgpt(instruction,examples,question,stop_words)
    elif model_type == 'llama':
        return llama(instruction,examples,question,model,tokenizer,stop_words)

def chatgpt(instruction,examples,question, stop=["\n"]):
    response = openai.ChatCompletion.create(
      model=GENERATING_MODEL,
      messages=[{"role": "system", "content": instruction},
               {"role": "assistant", "content": examples},
               {"role": "user", "content": question}],
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["message"]["content"]

def llama(instruction,examples,question,model,tokenizer,stop_words=['\n']):
    text = instruction + examples + question
    text_token = tokenizer(text, return_tensors="pt")
    output_ids = model.generate(text_token.input_ids.to(model.device), max_new_tokens=256)
    ans = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ans = ans[len(text):]
    if len(stop_words) > 0:
        ans = ans[:min([ans.find(w) for w in stop_words if ans.find(w)!=-1] + [len(ans)])]
    return ans

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

with open(choice_example_path, "r", encoding='utf-8') as f:
    mix_choice_example = json.load(f)

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_icl_example_by_similarity(data):
    describe_embedding_whole = get_embedding(data['sentence'])
    icl_example_similarities = sorted([
        (vector_similarity(describe_embedding_whole, example['whole']), example) for example in mix_choice_example if data['sentence'] != example['sentence']
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
