import pandas as pd
from wordfreq import zipf_frequency
import spacy
import json
import re

#sentence set
df = pd.read_csv('simplification_rating\multipleye.csv')

sentences_set = set([])

for index, line in df.iterrows():
    sentences_set.add(line["text"])

#mean length and standard deviation
mean_word_count = 27
stdev_word_count = 9
#minimum and maximum sentence length wanted 
min_word_count = mean_word_count - 2 * stdev_word_count
max_word_count = mean_word_count + 2 * stdev_word_count

#special characters to be excluded
excluded_characters = set(['*', '@', '[', ']', '\\', '/', '<', '>', '=', '^', '_', '`', '{', '}', '|', '~'])
nr_excluded = len(excluded_characters)

#romanian diacritics and punctuation
romanian_diacritics = set([536, 537, 538, 539, 350, 351, 354, 355, 258, 259, 194, 226, 206, 238, 8220, 8211, 8212, 8221, 8222, 8230, 8217, 770, 807, 806, 774, 171, 187])

#ignore sentence if it checks certain criterias
def validate_sentence(sentence):
    #check sentence length based on mean and stdev
    sentence_token_count = len(sentence)
    if sentence_token_count < min_word_count or sentence_token_count > max_word_count:
        return False
    
    #get sentence and all its' words in string format
    sentence_tokens = [str(x) for x in sentence]
    sent_str = str(sentence)

    #check if sentence contains special characters
    sentence_chars = set(sent_str)
    if len(excluded_characters - sentence_chars) < nr_excluded:
        return False
    
    #check if more then 50% of all characters are uppercase
    nr_upper = sum(1 for c in sent_str if c.isupper())
    if nr_upper >= len(sent_str) / 2:
        return False
    
    #check if more than 50% of tokens contain digits
    nr_with_digits = sum(1 for t in sentence_tokens if bool(re.search(r'\d', t)))
    if nr_with_digits >= sentence_token_count / 2:
        return False
    
    #check if sentence contains more than one token longer than 20 characters
    nr_long = sum(1 for t in sentence_tokens if len(t) > 20)
    if nr_long >= 1:
        return False
    
    #check if more than 60% of tokens are capitalized or numbers (are names, trademarks, dates etc.)
    nr_capital = sum(1 for t in sentence_tokens if t[0].isupper() or t[0].isdigit())
    if nr_capital >= sentence_token_count * 3 / 5:
        return False
    
    #check if sentence contains non-ascii characters (excluding diacritics and romanian punctuation)
    nr_non_ascii = sum(1 for c in sent_str if ord(c) > 127 and ord(c) not in romanian_diacritics)
    if nr_non_ascii >= 1:
        return False
    
    #check if sentence contains more than 4 consecutive punctuation characters
    charsearch = re.search("[.?!,;:'\u201c\u201d\u201b\u2013\u2014\u2019\u2025]{5,}", sent_str)
    if bool(charsearch):
        return False
    
    #have proper start of sentence (capitalized word)
    if not sentence[0].is_alpha or not sentence[0].is_title:
        return False
    
    #sentence has end
    if not '.' in sent_str:
        return False
    
    count_words_to_simplify = 0
    score_0 = 0
    for token in sentence_tokens:
        zipf_score = zipf_frequency(token, 'ro')
        if zipf_score <= 4.25:
            count_words_to_simplify += 1
        if zipf_score == 0:
            score_0 += 1

    if count_words_to_simplify < 5:
        return False
    
    if score_0 > 5:
        return False
    
    #sentence passes all checks
    return True


#list of solected sentences
selected_sentences = []

#tokenizer to get all sentences and words
nlp = spacy.blank("ro")
nlp.add_pipe("sentencizer")

#for each sentence check if it's valid
for sent in sentences_set:
    sentence = nlp(sent)
    if validate_sentence(sentence):
        selected_sentences.append({'sentence': str(sentence)})

#save data to a json file
with open('simplification_rating/selected_sentences_multipleye.json', 'w') as f:
    json.dump(selected_sentences, f, indent=4)