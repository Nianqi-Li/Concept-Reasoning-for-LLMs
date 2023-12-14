import json
import random
from bs4 import BeautifulSoup
import openai
from tqdm import tqdm
import numpy as np
import requests
import wikipedia

openai.api_key = 'sk-*'
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATING_MODEL = "gpt-3.5-turbo"
soup = BeautifulSoup(features="html.parser")

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

def entity_search(data,mention):
    with open(template_path,'r',encoding='utf-8') as f:
        all_template = json.load(f)
        template = all_template['deductive_reasoning']['ment2ent']      
    entity = ''  
    entity_obs = ''
    ment2ents = wikipedia.search(mention)
    ment2ents = [m for m in ment2ents if mention.lower() in m.lower()]
    if len(ment2ents) > 5:
        ment2ents = ment2ents[0:5]
    if len(ment2ents) > 1:
        ment2ent_instruction = template['instruction']
        ment2ent_examples = template['example']            
        ment2ent_question = 'Describe:' + data['sentence'] + '.\nPlease select the entity from the options below that best matches the mention \"' + mention + '\" in the above description:\n'
        ment2ent_question_index = 1
        for ent in ment2ents:
            if '(' not in ent and ')' not in ent:
                try:
                    ent_describe = wikipedia.summary(ent,sentences=1)
                    ent_describe = ent_describe.split(',')
                    ment2ent_question += str(ment2ent_question_index) + '.' + ent + '(' + ent_describe[0] + ')\n'            
                except:
                    ment2ent_question += str(ment2ent_question_index) + '.' + ent + '\n'
            else:
                ment2ent_question += str(ment2ent_question_index) + '.' + ent + '\n'    
            ment2ent_question_index += 1        
        ment2ent_index = llm('chatgpt',ment2ent_instruction,ment2ent_examples,ment2ent_question + 'Answer:',None,None,['.'])
        if not str(ment2ent_index).isdigit():
            entity = random.choice(ment2ents)
        elif int(ment2ent_index) != 0:
            entity = ment2ents[int(ment2ent_index)-1]
    elif len(ment2ents) == 1:
        entity = ment2ents[0]                    
    if entity != '':
        try:
            entity_obs = wikipedia.summary(entity,sentences=1)
        except:
            template = all_template['deductive_reasoning']['search_entity']
            question = 'Describe:' + data['sentence'] + '\nQuestion:Retrieve the first sentence of the entry \"' + entity + '\"\nAnswer:'
            entity_obs = llm('chatgpt',template['instruction'],template['example'],question)
    else:
        template = all_template['deductive_reasoning']['search_entity']
        question = 'Describe:' + data['sentence'] + '\nQuestion:Retrieve the first sentence of the entry \"' + mention + '\"\nAnswer:'
        entity_obs = llm('chatgpt',template['instruction'],template['example'],question)
    return entity_obs

def thought(data,model_type,model,tokenizer,last_thought,index):
    with open(template_path,'r',encoding='utf-8') as f:
        all_template = json.load(f)
        template = all_template['deductive_reasoning']
    this_thought = llm(model_type,template['instruction'],template['example'],last_thought + f'Thought {index}:',model,tokenizer,['Obs'])
    this_thought = f'Thought {index}:' + this_thought.strip('\n') + '\n'
    if 'Act' not in this_thought:
        this_thought2 = llm(model_type,template['instruction'],template['example'],last_thought + this_thought + f'Act {index}:',model,tokenizer,['Obs'])
        this_thought = this_thought + f'Act {index}:' + this_thought2.strip('\n') + '\n'
    if 'Entity Search' not in this_thought and 'Finish' not in this_thought:
        if 'What is' in this_thought:
            thought_index = this_thought.find('Act')
            this_thought = this_thought[0:thought_index]
            this_thought2 = llm(model_type,template['instruction'],template['example'],last_thought + this_thought + f'Act {index}:Entity Search[',model,tokenizer,['Obs'])
            this_thought = this_thought + f'Act {index}:Entity Search[' + this_thought2.strip('\n') + '\n'
        else:
            thought_index = this_thought.find('Act')
            this_thought = this_thought[0:thought_index]
            this_thought2 = llm(model_type,template['instruction'],template['example'],last_thought + this_thought + f'Act {index}:Finish[',model,tokenizer,['\n'])
            this_thought = this_thought + f'Act {index}:Finish[' + this_thought2.strip('\n') + '\n'
    this_thought = this_thought.strip('\n')
    if 'Entity Search' in this_thought:
        mention_begin_index = this_thought.rfind('Entity Search[')
        mention_end_index = this_thought.rfind(']')
        mention = this_thought[mention_begin_index + len('Entity Search['):mention_end_index]
        if data['entity'] in mention or mention in data['entity'] or 'entity' in mention:
            template = all_template['cot_reasoning']
            this_thought = llm(model_type,template['instruction'],template['example'],last_thought + f'\nThought {index}:',model,tokenizer,['\nThought'])
            this_thought = f'\nThought {index}:' + this_thought
            if 'Finish[' not in this_thought:
                this_thought2 = llm(model_type,template['instruction'],template['example'],last_thought + this_thought + '\nFinish[',model,tokenizer,['\n'])
                this_thought = this_thought + '\nFinish[' + this_thought2
            return this_thought
        else:
            obs = entity_search(data,mention)
        this_thought = this_thought + f'\nObs {index}:' + obs.strip('\n') + '\n'
    this_thought = this_thought.strip('\n') + '\n'
    return this_thought  

def reasoning(data,model_type,model=None,tokenizer=None,to_print=True):
    reasoning_thought = 'Describe:' + data['sentence'] + '\nQuestion:Please infer what concept \"' + data['entity'] + '\" belong to?\n'    
    if to_print:
        print(reasoning_thought)
    index = 1
    while 'Finish' not in reasoning_thought and index <= 7:
        this_thought = thought(data,model_type,model,tokenizer,reasoning_thought,index)
        if to_print:
            print(this_thought)
        reasoning_thought += this_thought
        index += 1
    return reasoning_thought
