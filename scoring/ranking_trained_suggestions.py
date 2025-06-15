import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import json

tokenizer = BertTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = BertForMaskedLM.from_pretrained("./ranking_output/checkpoint-18")
model.eval()

def predict_mask(model, tokenizer, sentence, word_to_mask, top_k=10):
    """Get top-k predictions for masked word"""
    # Replace word with [MASK]
    masked_sentence = sentence.replace(word_to_mask, "[MASK]", 1)
    
    # Tokenize
    inputs = tokenizer(masked_sentence, return_tensors="pt", max_length=128, truncation=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    
    # Find mask position
    mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0].item()
    
    # Get top predictions
    mask_logits = logits[mask_pos]
    probs = F.softmax(mask_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    
    # Convert to words
    predictions = []
    for token_id, prob in zip(top_ids, top_probs):
        word = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if word:
            predictions.append({"token_str": word, "score": prob.item()})
    
    return predictions

"""
sentence = "Capitolul al doilea investigheaz\u0103 aceast\u0103 leg\u0103tur\u0103 pu\u021bin mai am\u0103nun\u021bit, discut\u00e2nd gimnastica mental\u0103 a rezolv\u0103rii problemelor."
word = "gimnastica"
predictions = predict_mask(model, tokenizer, sentence, word)
"""

with open('scoring/validation_split.json', 'r') as f:
    test_sentences = json.load(f)

def test_suggestions(model):
    result = []

    for sentence_set in test_sentences:
        suggestions = predict_mask(model, tokenizer, sentence_set["sentence"], sentence_set["word"])
        suggestions_fixed = [(x["token_str"], x["score"]) for x in suggestions]
        result.append({"sentence": sentence_set["sentence"], 
                   "sentence_id": sentence_set["sentence_id"], 
                   "word": sentence_set["word"], 
                   "suggestions": suggestions_fixed})

    return result

with open('scoring/suggestions/trained2_bertRo_suggestions.json', 'w') as f:
    json.dump(test_suggestions(model), f, indent=4)

model2 = BertForMaskedLM.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
with open('scoring/suggestions/bertNormal_suggestions.json', 'w') as f:
    json.dump(test_suggestions(model2), f, indent=4)