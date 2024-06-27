import pandas as pd
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList, MinLengthLogitsProcessor, TemperatureLogitsWarper
from llama_cpp import Llama

# loading tweets from CSV file
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

def clean_output(filename):
    file_path = "./hand_in/" + filename + ".csv"
    df = pd.read_csv(file_path, sep=",")
    regex = re.compile(r"[\S\s]*style of Elon Musk[\S\s]*:|[\S\s]*style of Donald Trump[\S\s]*:|[\S\s]*Here's a[\S\s]*:|[\S\s]*in his style[\S\s]*:")
    df = df.replace(regex, "")
    df.to_csv("./hand_in/" + filename + "_cleaned.csv", sep=",", index=False)

# approach with GPT 2 which we tried
def text_generation(input):
    prompt = "Tweet: " + input
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

    inputs = tokenizer.encode(prompt, return_tensors='pt')

    logits_setup = LogitsProcessorList([
        MinLengthLogitsProcessor(5, eos_token_id=tokenizer.eos_token_id),
        TemperatureLogitsWarper(temperature=0.8),
    ])

    gpt_outputs = model.generate(inputs, max_length=150, do_sample=True, early_stopping=True,
                             num_return_sequences=1,  logits_processor=logits_setup,top_p=0.98,top_k=50)
    generated_text = tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

    # clean the prompt from input
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

# text generation with LLM
def text_chat_generation(input, person):
    llm = Llama(model_path="../model/llama-2-7b-chat.Q8_0.gguf", n_ctx=0, n_gpu_layers=32)

    system = f"You are {person}."
    question = f"Question: generate a different follow-up tweet to the following tweet without repeating it: " + input + " Write a maximum 200 Characters without repeating the previous tweet."
    prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""
    
    output = llm(prompt, max_tokens=2048, temperature=1.0)
    output_text = output["choices"][0]["text"]

    return output_text

# text style transfer with LLM
def text_style_change(input, style):
    person = style
    llm = Llama(model_path="../model/llama-2-7b-chat.Q8_0.gguf", n_ctx=0, n_gpu_layers=32)

    system = f"You are {person}."
    question = f"Question: can you repeat the tweet in the style of {person}. Write a maximum 280 Characters: " + input
    prompt = f"""<s>[INST] <<SYS>>{system}<</SYS>>{question} [/INST]"""

    output = llm(prompt, max_tokens=2048)
    output_text = output["choices"][0]["text"]

    return output_text

# the pipeline for text generation/text style transfer
def pipeline():
    file = open('./data/initial_tweet_musk.txt', 'r')

    musk_initial_tweet = file.read()
    file.close()
    convo = []

    m = text_chat_generation(musk_initial_tweet, "Elon Musk")
    convo.append(m)
    for _ in range(100):
        t = text_style_change(m, "Donald Trump")
        tt = text_chat_generation(t, "Donald Trump")
        mt = text_style_change(tt, "Elon Musk")
        m = text_chat_generation(mt, "Elon Musk")
        convo.extend([t, tt, mt, m])

    df = pd.DataFrame(data=convo)
    df.to_csv("./data/output.csv", sep=",", index=False)

if __name__ == '__main__':
    pipeline()

    # additional option
    #clean_output("output")
