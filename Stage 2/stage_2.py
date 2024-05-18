import nltk
import random
import re
import os

from hmmlearn import hmm
import numpy as np

#from llama_cpp import Llama
#_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

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

def train_hmm_model(tokens, components=2):
    token_dict = {token: idx for idx, token in enumerate(set(tokens))}
    idx_dict = {idx: token for token, idx in token_dict.items()}
    
    token_idxs = [token_dict[token] for token in tokens]
    X = np.array(token_idxs).reshape(-1,1)
    
    model = hmm.CategoricalHMM(n_components=components, n_iter=100)
    model = model.fit(X)
    
    return model, idx_dict

def text_generation(samples=260):
    if os.path.exists('data_stage_2_generation1.txt'):
        # Delete the file
        os.remove('data_stage_2_generation1.txt')
        print(f"The file data_stage_2_generation1.txt has been deleted.")
    else:
        print(f"The file data_stage_2_generation1.txt does not exist.")
    
    if os.path.exists('data_stage_2_generation2.txt'):
        # Delete the file
        os.remove('data_stage_2_generation2.txt')
        print(f"The file data_stage_2_generation2.txt has been deleted.")
    else:
        print(f"The file data_stage_2_generation2.txt does not exist.")

    contents = {}

    with open('data_stage2_1_kogler.txt', 'r') as file:
        contents["kogler"] = file.read()
    
    with open('data_stage2_2_kickl.txt', 'r') as file:
        contents["kickl"] = file.read()

    for iter,text_idx in enumerate(contents):
        #print(contents[text_idx])
        clean_text = text_cleaner(contents[text_idx])
        tokens = nltk.word_tokenize(clean_text)
        print(len(tokens))

        model, idx_dict = train_hmm_model(tokens)

        gen_index = model.sample(n_samples=samples)[0].flatten()
        gen_tokens = [idx_dict[idx] for idx in gen_index]
        
        path = f"data_stage_2_generation{iter+1}.txt"
        with open(path, "a") as file:
            text = ' '.join(gen_tokens)
            file.write(output_cleaner(text))

    return

def main():
    text_generation(260)

if __name__ == '__main__':
    main()
