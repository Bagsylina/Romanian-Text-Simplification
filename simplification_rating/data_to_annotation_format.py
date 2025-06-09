import json

with open('simplification_rating/selected_sentences_multipleye.json', 'r') as f:
    sentence_list = json.load(f)
    sentence_dict = {sentence["id"]: sentence for sentence in sentence_list}

with open('simplification_rating/bertRo_sentence_simplifications_multipleye.json', 'r') as f:
    bertRo_list = json.load(f)
    bertRo_dict = {sentence["id"]: sentence for sentence in bertRo_list}

with open('simplification_rating/gpt4o_sentence_simplifications_multipleye.json', 'r') as f:
    gpt4o_list = json.load(f)
    gpt4o_dict = {sentence["id"]: sentence for sentence in gpt4o_list}

with open('simplification_rating/llama3ro_sentence_simplifications_multipleye.json', 'r') as f:
    llama3ro_list = json.load(f)
    llama3ro_dict = {sentence["id"]: sentence for sentence in llama3ro_list}

with open('simplification_rating/miles_sentence_simplifications_multipleye.json', 'r') as f:
    miles_list = json.load(f)
    miles_dict = {sentence["id"]: sentence for sentence in miles_list}

ro_annotate_dataset = []

for id in sentence_dict:
    if id in bertRo_dict and id in gpt4o_dict and id in llama3ro_dict and id in miles_dict:
        ro_annotate_dataset.append({"id": id, "original_sentence": sentence_dict[id]["sentence"], 
                                    "simplifications": [
                                        {"id": 1, "body": bertRo_dict[id]["simplified_sentence"]}, 
                                        {"id": 2, "body": gpt4o_dict[id]["simplified_sentence"]}, 
                                        {"id": 3, "body": llama3ro_dict[id]["simplified_sentence"]}, 
                                        {"id": 4, "body": miles_dict[id]["simplified_sentence"]}], 
                                    "sentenceBert": bertRo_dict[id]["simplified_sentence"], "sentenceGPT": gpt4o_dict[id]["simplified_sentence"],
                                    "sentenceLlama": llama3ro_dict[id]["simplified_sentence"], "sentenceMiles": miles_dict[id]["simplified_sentence"]})
    
with open('simplification_rating/meye_annotate_dataset.json', 'w') as f:
    json.dump(ro_annotate_dataset, f, indent=4)