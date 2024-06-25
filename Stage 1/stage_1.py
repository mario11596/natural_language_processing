import nltk
import random
import re
import os
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

from llama_cpp import Llama
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)

    return newString

def text_generation():
    if os.path.exists('data_stage_1_new_file.txt'):
        # Delete the file
        os.remove('data_stage_1_new_file.txt')
        print(f"The file data_stage_1_new_file.txt has been deleted.")
    else:
        print(f"The file data_stage_1_new_file.txt does not exist.")

    with open('data_stage_1.txt', 'r') as file:
        contents = file.read()
    file.close()

    clean_text = text_cleaner(contents)

    tokens = nltk.word_tokenize(clean_text)
    print(len(tokens))

    ngrams = {}
    n = 2
    sentences_more = True
    number_of_new_words = 0

    # Construct the n-grams
    for i in range(len(tokens) - n):
        gram = ' '.join(tokens[i:i + n])
        if gram not in ngrams.keys():
            ngrams[gram] = []
        ngrams[gram].append(tokens[i + n])

    while sentences_more:
        number_of_word = random.randint(10, 30)
        number = random.randint(1, len(tokens) - n)
        curr_sequence = ' '.join(tokens[number:number + n])
        output = curr_sequence

        for i in range(number_of_word):
            if curr_sequence not in ngrams.keys():
                break
            possible_words = ngrams[curr_sequence]
            next_word = random.choice(possible_words)
            output += ' ' + next_word
            rwords = nltk.word_tokenize(output)
            curr_sequence = ' '.join(rwords[len(rwords) - n:len(rwords)])

        number_of_new_words = number_of_word + number_of_new_words

        with open("data_stage_1_new_file.txt", "a") as file:
            file.write(output + '.\n')

        if number_of_new_words >= 2000:
            sentences_more = False
    return

def text_generation_llm():
    if os.path.exists('data_stage_1_new_file_llm.txt'):
        # Delete the file
        os.remove('data_stage_1_new_file_llm.txt')
        print(f"The file data_stage_1_new_file_llm.txt has been deleted.")
    else:
        print(f"The file data_stage_1_new_file_llm.txt does not exist.")

    with open('data_stage_1.txt', 'r') as file:
        contents = file.read()
    file.close()

    clean_text = text_cleaner(contents)
    llm = Llama(model_path="../model/llama-2-7b.Q5_K_M.gguf", n_ctx=0, n_gpu_layers=32)
    
    prompt = clean_text[-1420:]
    output = llm(prompt, max_tokens=2600, repeat_penalty=1.1) #2667*0.75 for average 4 tokens per 3 words -> 2000 Words
    
    final_text = ""
    for line in output["choices"][0]["text"].splitlines():
        if line != "":
            line = re.sub("[^a-zA-Z ]", "", line)
            line = _RE_COMBINE_WHITESPACE.sub(" ", line).strip()
            final_text += line + '.\n'
    
    with open("data_stage_1_new_file_llm.txt", "a") as file:
        file.write(final_text)
    
    return

def text_style_change():
    if os.path.exists('data_stage_1_new_style.txt'):
        # Delete the file
        os.remove('data_stage_1_new_style.txt')
        print(f"The file data_stage_1_new_style.txt has been deleted.")
    else:
        print(f"The file data_stage_1_new_style.txt does not exist.")
    
    with open('data_stage_1_new_file.txt', 'r') as file:
        contents = file.read()
    file.close()

    llm = Llama(model_path="../model/llama-2-7b-chat.Q8_0.gguf", n_ctx=0, n_gpu_layers=32)
    
    system = "You are a helpful assistant pretending to be Spongebob Squarepants. "
    question = "Question: Can you repeat the following text but in the text-style of Spongebob Squarepants. Write at least 2000 words without emojis: " + contents[-1800:]
    prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""
    
    output = llm(prompt, max_tokens=2660) #2667*0.75 for average 4 tokens per 3 words -> 2000 Words
    
    output_text = output["choices"][0]["text"]
    final_text = re.sub("(\*.*?\* )|(\*.*?\*)", "", output_text)

    with open("data_stage_1_new_style.txt", "a") as file:
        file.write(final_text)
    return

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def Rouge(ground_true, generated_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(ground_true, generated_text)

    print(scores)

def Bleu(ground_true, generated_text):
    ground_true = word_tokenize(ground_true.lower())
    generated_text = word_tokenize(generated_text.lower())
    bleu_score = corpus_bleu([[ground_true]], [generated_text])

    print("BLEU Score:", bleu_score)


def text_evaluation():
    ground_true = read_file('data_stage_1.txt')
    generated_text = read_file('../hand_in/group16_stage1_generation_nollm.txt')

    Rouge(ground_true, generated_text)

    Bleu(ground_true, generated_text)


def main():
    text_generation()
    #text_generation_llm()
    text_style_change()
    text_evaluation()

if __name__ == '__main__':
    main()
