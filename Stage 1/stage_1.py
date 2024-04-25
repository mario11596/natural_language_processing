import nltk
import random
import re
import os

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
            file.write(output + '\n')

        if number_of_new_words >= 2000:
            sentences_more = False
    return

def main():
    text_generation()

if __name__ == '__main__':
    main()
