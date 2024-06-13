import pandas as pd
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList, MinLengthLogitsProcessor, TemperatureLogitsWarper
import torch


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
    prompt = input
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

    inputs = tokenizer.encode(prompt, return_tensors='pt')

    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(5, eos_token_id=tokenizer.eos_token_id),
        TemperatureLogitsWarper(temperature=0.7),
    ])

    outputs = model.generate(inputs, max_length=70, do_sample=True, num_beams=5, no_repeat_ngram_size=3, early_stopping=True,
                             num_return_sequences=1,  logits_processor=logits_processor)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    print(text)


def pipeline():
    file = open('./data/initial_tweet_musk.txt', 'r')

    musk_initial_tweet = file.read()
    file.close()

    # i think this is for style transfer?
    musk_tweets = load_tweets('./data/data_stage3_1_musk.xlsx')
    trump_tweets = load_tweets('./data/data_stage3_2_trump.xlsx')

    for i in range(100):
        text_generation(musk_initial_tweet)


if __name__ == '__main__':
    pipeline()