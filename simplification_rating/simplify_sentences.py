import json
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import RoTextSimpModel

with open('simplification_rating/selected_sentences.json', 'r') as f:
    sentence_list = json.load(f)

with open('simplification_rating/selected_sentences_multipleye.json', 'r') as f:
    sentence_list_multipleye = json.load(f)

def simplify_sentences(unmasker):
    simpModel = RoTextSimpModel(unmasker)
    result = []

    for sentence_set in sentence_list:
        simplification = simpModel.sentence_simplification(sentence_set["sentence"])
        result.append({"original_sentence": sentence_set["sentence"], "simplified_sentence": simplification, 
                       "file": sentence_set["file"], "subfolder": sentence_set["subfolder"]})
        
    return result


def simplify_sentences_multipleye(unmasker):
    simpModel = RoTextSimpModel(unmasker)
    result = []

    for sentence_set in sentence_list_multipleye:
        simplification = simpModel.sentence_simplification(sentence_set["sentence"])
        result.append({"original_sentence": sentence_set["sentence"], "simplified_sentence": simplification})
        
    return result


from transformers import pipeline
from gpt4omlm import gpt4o_substitution_suggestions
from llama3romlm import llama3ro_substitution_suggestions

"""
with open('simplification_rating/bertRo_sentence_simplifications.json', 'w') as f:
    unmaskerRo = pipeline('fill-mask', model='dumitrescustefan/bert-base-romanian-cased-v1', top_k=10)
    json.dump(simplify_sentences(unmaskerRo), f, indent=4)

with open('simplification_rating/gpt4o_sentence_simplifications.json', 'w') as f:
    unmaskerGPT = gpt4o_substitution_suggestions
    json.dump(simplify_sentences(unmaskerGPT), f, indent=4)

with open('simplification_rating/bertRo_sentence_simplifications_multipleye.json', 'w') as f:
    unmaskerRo = pipeline('fill-mask', model='dumitrescustefan/bert-base-romanian-cased-v1', top_k=10)
    json.dump(simplify_sentences_multipleye(unmaskerRo), f, indent=4)

with open('simplification_rating/gpt4o_sentence_simplifications_multipleye.json', 'w') as f:
    unmaskerGPT = gpt4o_substitution_suggestions
    json.dump(simplify_sentences_multipleye(unmaskerGPT), f, indent=4)
"""

with open('simplification_rating/llama3ro_sentence_simplifications.json', 'w') as f:
    unmaskerLlama = llama3ro_substitution_suggestions
    json.dump(simplify_sentences(unmaskerLlama), f, indent=4)

with open('simplification_rating/llama3ro_sentence_simplifications_multipleye.json', 'w') as f:
    unmaskerLlama = llama3ro_substitution_suggestions
    json.dump(simplify_sentences_multipleye(unmaskerLlama), f, indent=4)