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

from MILES.simplifier import models
from MILES.simplifier import config
from MILES.simplifier import simplifier

config.lang = "ro"
models.embeddings = models.load_embeddings(config.lang)

def simplify_sentences():
    result = []

    for sentence_set in sentence_list:
        simplification = simplifier.simplify_text(sentence_set["sentence"])
        result.append({"original_sentence": sentence_set["sentence"], "simplified_sentence": simplification, 
                       "file": sentence_set["file"], "subfolder": sentence_set["subfolder"], "id": sentence_set["id"]})
        
    return result


def simplify_sentences_multipleye():
    result = []

    for sentence_set in sentence_list_multipleye:
        simplification = simplifier.simplify_text(sentence_set["sentence"])
        result.append({"original_sentence": sentence_set["sentence"], "simplified_sentence": simplification, "id": sentence_set["id"]})
        
    return result

with open('simplification_rating/miles_sentence_simplifications.json', 'w') as f:
    json.dump(simplify_sentences(), f, indent=4)

with open('simplification_rating/miles_sentence_simplifications_multipleye.json', 'w') as f:
    json.dump(simplify_sentences_multipleye(), f, indent=4)