import nltk
import random
import re
import os

from hmmlearn import hmm
import numpy as np

from llama_cpp import Llama
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def text_cleaner(text):
    #newString = text.lower()
    newString = text.split(":")[1:]
    newString = ''.join(newString)
    newString = re.sub(r"(\(.*?\) )|(\(.*?\))", "", newString)
    newString = re.sub(r"([^\w\s\.\!]\s)", " ", newString)
    newString = re.sub(r"([^\w\s\.\!])", "", newString)
    #remove numbers
    return newString

def output_cleaner(text):
    if text[-1] != ".":
        newString = text+".\n"
    else:
        newString = text+"\n"
    newString = re.sub(r"(\s)([^\w\s])\s", r"\2 ", newString)
    return newString

def style_cleaner(text):
    new_string = re.sub(r"(<dummy32006>\s\n<dummy32005>\sassistant\s|<<SYS>>)([\s\S]*)(<dummy32006> )", r"\2", text)
    split_string = new_string.split(":")
    if len(split_string) > 1:
        new_string = "".join(split_string[1:])
    return new_string

def read_content(read_path, write_path):
    if os.path.exists(write_path):
        # Delete the file
        os.remove(write_path)
        print(f"The file {write_path} has been deleted.")
    else:
        print(f"The file {write_path} does not exist.")
    
    with open(read_path, 'r') as file:
        content = file.read()
    return content

def train_hmm_model(tokens, components=2):
    token_dict = {token: idx for idx, token in enumerate(set(tokens))}
    idx_dict = {idx: token for token, idx in token_dict.items()}
    
    token_idxs = [token_dict[token] for token in tokens]
    X = np.array(token_idxs).reshape(-1,1)
    
    #model = hmm.CategoricalHMM(n_components=components, n_iter=100)
    #model = model.fit(X)
    
    return idx_dict#, model

def text_generation(samples=260):
    contents = {}
    contents["kogler"] = read_content('data_stage2_1_kogler.txt', 'data_stage_2_generation1.txt')
    contents["kickl"] = read_content('data_stage2_2_kickl.txt', 'data_stage_2_generation2.txt')

    for iter,text_idx in enumerate(contents):
        #print(contents[text_idx])
        clean_text = text_cleaner(contents[text_idx])
        tokens = nltk.word_tokenize(clean_text)
        print(len(tokens))

        idx_dict = train_hmm_model(tokens)

        gen_index = model.sample(n_samples=samples)[0].flatten()
        gen_tokens = [idx_dict[idx] for idx in gen_index]

        path = f"data_stage_2_generation{iter+1}.txt"
        with open(path, "a") as file:
            text = ' '.join(gen_tokens)
            file.write(output_cleaner(text))

    return

def text_style_change():
    contents = {}
    contents["Werner Kogler"] = read_content('data_stage_2_generation2.txt', 'data_stage_2_new_style1.txt')
    contents["Herbert Kickl"] = read_content('data_stage_2_generation1.txt', 'data_stage_2_new_style2.txt')

    #https://huggingface.co/TheBloke/leo-hessianai-13B-chat-GGUF
    llm = Llama(model_path="../model/leo-hessianai-13b-chat.Q5_K_M.gguf", n_ctx=0, n_gpu_layers=32)
    #llm = Llama(model_path="../model/llama-2-7b-chat.Q6_K.gguf", n_ctx=0, n_gpu_layers=32)

    for iter,text_idx in enumerate(contents):
        #system = "You are a Austrian politician {text_idx}. "
        system = f"Du bist der österreichische Politiker {text_idx}."
        #question = "Question: Can you repeat the following text but in german and in the text-style of {text_idx}. Write at least 2000 words without emojis: " + contents1[-1800:]
        question = f"Frage: Kannst du den folgenden Text im stil von {text_idx} wiedergeben. Schreibe bitte mindestens 300 Wörter ohne emojis zu benutzen: " + contents[text_idx][-1800:]
        prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""
        output = llm(prompt, max_tokens=2048)
        output_text = output["choices"][0]["text"]
        final_text = re.sub("(\*.*?\* )|(\*.*?\*)", "", output_text)
        with open(f"data_stage_2_new_style{iter+1}.txt", "a") as file:
            file.write(final_text)
            file.write("\n\n")
            file.write(style_cleaner(final_text))
    return

def main():
    #text_generation(260)
    text_style_change()

if __name__ == '__main__':
    main()
