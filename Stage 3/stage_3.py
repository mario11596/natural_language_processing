import pandas as pd
import re
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList, MinLengthLogitsProcessor, TemperatureLogitsWarper
import torch
from llama_cpp import Llama

def load_tweets(filename):
    file_path = filename
    df = pd.read_excel(file_path)
    array_df = df.to_numpy()
    i = 0

    for sentence in array_df:
        modification = re.sub(r'[\d\W]+$', '', sentence[0])
        array_df[i] = modification

        i = i + 1

    return array_df

def text_generation(input):
    #input = re.sub(r'[\d\W]+$', '', input)
    prompt = "Tweet: " + input
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

    inputs = tokenizer.encode(prompt, return_tensors='pt')

    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(5, eos_token_id=tokenizer.eos_token_id),
        TemperatureLogitsWarper(temperature=0.8),
    ])

    outputs = model.generate(inputs, max_length=150, do_sample=True, num_beams=5, no_repeat_ngram_size=3, early_stopping=True,
                             num_return_sequences=1,  logits_processor=logits_processor)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    #print(text)
    return text

def text_chat_generation(input, person):
    llm = Llama(model_path="../model/llama-2-7b-chat.Q8_0.gguf", n_ctx=0, n_gpu_layers=32)

    system = f"You are {person}."
    question = f"Question: can you generate a follow-up tweet to the following tweet: " + input + " Write a maximum 280 Characters."
    prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""
    
    output = llm(prompt, max_tokens=2048, temperature=1.0)
    output_text = output["choices"][0]["text"]
    #print(text)
    return output_text

def text_style_change(input, style):
    person = style
    #tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    #model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    llm = Llama(model_path="../model/llama-2-7b-chat.Q8_0.gguf", n_ctx=0, n_gpu_layers=32)

    system = f"You are {person}."
    question = f"Question: can you repeat the tweet in the style of {person}. Write a maximum 280 Characters: " + input
    prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""
    
    #prompt = f"repeat the following words in the style of {person}: " + input
    #inputs = tokenizer.encode(prompt, return_tensors='pt')

    #logits_processor = LogitsProcessorList([
    #    MinLengthLogitsProcessor(5, eos_token_id=tokenizer.eos_token_id),
    #    TemperatureLogitsWarper(temperature=0.7),
    #])

    output = llm(prompt, max_tokens=2048)
    output_text = output["choices"][0]["text"]
    #print(text)
    return output_text


def pipeline():
    file = open('./data/initial_tweet_musk.txt', 'r')

    musk_initial_tweet = file.read()
    file.close()

    # i think this is for style transfer?
    musk_tweets = load_tweets('./data/data_stage3_1_musk.xlsx')
    trump_tweets = load_tweets('./data/data_stage3_2_trump.xlsx')
    convo = []

    m = text_generation(musk_initial_tweet)
    convo.append(m)
    for _ in range(1):
        print("S--------------------")
        t = text_style_change(m, "Donald Trump")
        print(t)
        print("--------------------")
        tt = text_generation(t)
        #tt = text_chat_generation(t, "Donald Trump")
        print(tt)
        print("--------------------")
        mt = text_style_change(tt, "Elon Musk")
        print(mt)
        print("--------------------")
        m = text_generation(mt)
        #m = text_chat_generation(mt, "Elon Musk")
        print(m)
        print("--------------------E")
        convo.extend([t, tt, mt, m])

    df = pd.DataFrame(data=convo)
    df.to_csv("./data/output.csv", sep=",", index=False)

if __name__ == '__main__':
    pipeline()