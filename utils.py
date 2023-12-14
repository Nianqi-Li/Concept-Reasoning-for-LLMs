import openai

openai.api_key = 'sk-*'
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATING_MODEL = "gpt-3.5-turbo"

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
    output_ids = model.generate(text_token.input_ids.to(model.device), max_new_tokens=128)
    ans = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ans = ans[len(text):]
    if len(stop_words) > 0:
        ans = ans[:min([ans.find(w) for w in stop_words if ans.find(w)!=-1] + [len(ans)])]
    return ans

