import sys
import os
import json

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import RoTextSimpModel

with open('scoring/ranking_data.json', 'r') as f:
    test_sentences = json.load(f)

def test_suggestions(unmasker):
    simpModel = RoTextSimpModel(unmasker)
    result = []

    for sentence_set in test_sentences:
        suggestions = simpModel.word_simplifications(sentence_set["sentence"], sentence_set["word"])
        suggestions_fixed = []
        for suggestion in suggestions[0]["suggestions"]:
            if not isinstance(suggestion[2], float):
                suggestions_fixed.append((suggestion[0], suggestion[1], suggestion[2].tolist()[0]))
            else:
                suggestions_fixed.append(suggestion)

        result.append({"sentence": sentence_set["sentence"], 
                   "sentence_id": sentence_set["sentence_id"], 
                   "word": sentence_set["word"], 
                   "suggestions": suggestions_fixed})

    return result


from transformers import pipeline
import sys
import os
import json
from gpt4omlm import gpt4o_substitution_suggestions
from llama3romlm import llama3ro_substitution_suggestions

"""
with open('scoring/bertRo_suggestions.json', 'w') as f:
    unmaskerRo = pipeline('fill-mask', model='dumitrescustefan/bert-base-romanian-cased-v1', top_k=10)
    json.dump(test_suggestions(unmaskerRo), f, indent=4)

with open('scoring/bertMulti_suggestions.json', 'w') as f:
    unmaskerMulti = pipeline('fill-mask', model='google-bert/bert-base-multilingual-cased', top_k=10)
    json.dump(test_suggestions(unmaskerMulti), f, indent=4)

with open('scoring/gpt4o_suggestions.json', 'w') as f:
    unmaskerGPT = gpt4o_substitution_suggestions
    json.dump(test_suggestions(unmaskerGPT), f, indent=4)

with open('scoring/RoBert-l_suggestions.json', 'w') as f:
    unmaskerRo = pipeline('fill-mask', model='readerbench/RoBERT-large', top_k=10)
    json.dump(test_suggestions(unmaskerRo), f, indent=4)
"""

with open('scoring/llama3ro_suggestions.json', 'w') as f:
    unmaskerLlama = llama3ro_substitution_suggestions
    json.dump(test_suggestions(unmaskerLlama), f, indent=4)