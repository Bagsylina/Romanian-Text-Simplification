import json
import spacy

NLP = spacy.load("ro_core_news_lg")

with open('scoring/ranking_data.json', 'r') as f:
    annotated_data = json.load(f)

with open('scoring/bertRo_suggestions.json', 'r') as f:
    bertRo_data = json.load(f)

with open('scoring/RoBert-l_suggestions.json', 'r') as f:
    RoBert_l_data = json.load(f)

with open('scoring/gpt4o_suggestions.json', 'r') as f:
    gpt4o_data = json.load(f)

with open('scoring/llama3ro_suggestions.json', 'r') as f:
    llama3_data = json.load(f)

with open('scoring/bertMulti_suggestions.json', 'r') as f:
    bertML_data = json.load(f)

with open('scoring/trained_bertRo_suggestions.json', 'r') as f:
    trained_bertRo_data = json.load(f)

def calculate_results(annotated_data, predicted_data):
    count_acc = 0
    count_acc_1 = 0
    count_acc_3 = 0
    count_acc_5 = 0
    count_acc_lemma = 0
    count_recall = 0
    count_recall_1 = 0
    count_recall_3 = 0
    count_recall_5 = 0
    count_recall_lemma = 0
    count_potential = 0
    count_potential_1 = 0
    count_potential_3 = 0
    count_potential_5 = 0
    count_potential_lemma = 0
    count_precision = 0
    count_precision_1 = 0
    count_precision_3 = 0
    count_precision_5 = 0
    count_precision_lemma = 0

    sum_ap_3 = 0
    sum_ap_4 = 0
    sum_ap_5 = 0

    count_total_ann_words = 0
    count_total_pred_words = 0
    count_pred_words_1 = 0
    count_pred_words_3 = 0
    count_pred_words_5 = 0
    count_matchups = 0

    dict_ann = {}
    dict_pred = {}
    dict_ann_lemma = {}
    dict_pred_lemma = {}


    for data in annotated_data:
        dict_ann[data["sentence_id"]] = data["sorted_candidates"]

    for data in predicted_data:
        dict_pred[data["sentence_id"]] = [x[0] for x in data["suggestions"]]

    for data in annotated_data:
        dict_ann_lemma[data["sentence_id"]] = [NLP(x)[0].lemma_ for x in data["sorted_candidates"]]

    for data in predicted_data:
        dict_pred_lemma[data["sentence_id"]] = [NLP(x[0])[0].lemma_ for x in data["suggestions"]]

    for sent_id in dict_ann:
        if sent_id in dict_pred and len(dict_pred[sent_id]) > 0:
            count_total_ann_words += len(dict_ann[sent_id])
            count_total_pred_words += len(dict_pred[sent_id])
            count_pred_words_1 += min(1, len(dict_pred[sent_id]))
            count_pred_words_3 += min(3, len(dict_pred[sent_id]))
            count_pred_words_5 += min(5, len(dict_pred[sent_id]))
            count_matchups += 1

            ok_potential = False
            ok_potential_1 = False
            ok_potential_3 = False
            ok_potential_5 = False
            ok_potential_lemma = False

            if dict_ann[sent_id][0] == dict_pred[sent_id][0]:
                count_acc_1 += 1

            if dict_ann[sent_id][0] in dict_pred[sent_id][:3]:
                count_acc_3 += 1

            if dict_ann[sent_id][0] in dict_pred[sent_id][:5]:
                count_acc_5 += 1

            if dict_ann[sent_id][0] in dict_pred[sent_id]:
                count_acc += 1

            if dict_ann_lemma[sent_id][0] in dict_pred_lemma[sent_id]:
                count_acc_lemma += 1

            for word in dict_ann[sent_id]:
                if word in dict_pred[sent_id]:
                    ok_potential = True
                    count_recall += 1

                    if dict_pred[sent_id].index(word) < 1:
                        count_recall_1 += 1
                        ok_potential_1 = True

                    if dict_pred[sent_id].index(word) < 3:
                        count_recall_3 += 1
                        ok_potential_3 = True

                    if dict_pred[sent_id].index(word) < 5:
                        count_recall_5 += 1
                        ok_potential_5 = True

            if ok_potential:
                count_potential += 1
            if ok_potential_1:
                count_potential_1 += 1
            if ok_potential_3:
                count_potential_3 += 1
            if ok_potential_5:
                count_potential_5 += 1

            for word in dict_ann_lemma[sent_id]:
                if word in dict_pred_lemma[sent_id]:
                    ok_potential_lemma = True
                    count_recall_lemma += 1

            if ok_potential_lemma:
                count_potential_lemma += 1

            for word in dict_pred[sent_id]:
                if word in dict_ann[sent_id]:
                    count_precision += 1
            
            for word in dict_pred[sent_id][:1]:
                if word in dict_ann[sent_id]:
                    count_precision_1 += 1

            for word in dict_pred[sent_id][:3]:
                if word in dict_ann[sent_id]:
                    count_precision_3 += 1

            for word in dict_pred[sent_id][:5]:
                if word in dict_ann[sent_id]:
                    count_precision_5 += 1

            for word in dict_pred_lemma[sent_id]:
                if word in dict_ann_lemma[sent_id]:
                    count_precision_lemma += 1

            score = 0
            hits = 0
            for i, word in enumerate(dict_pred_lemma[sent_id][:3]):
                if word in dict_ann_lemma[sent_id]:
                    hits += 1
                    score += hits / (i + 1)
            #sum_ap_3 += score / min(3, len(dict_ann_lemma[sent_id]))
            sum_ap_3 += score / 3

            score = 0
            hits = 0
            for i, word in enumerate(dict_pred_lemma[sent_id][:4]):
                if word in dict_ann_lemma[sent_id]:
                    hits += 1
                    score += hits / (i + 1)
            #sum_ap_4 += score / min(4, len(dict_ann_lemma[sent_id]))
            sum_ap_4 += score / 4

            score = 0
            hits = 0
            for i, word in enumerate(dict_pred_lemma[sent_id][:5]):
                if word in dict_ann_lemma[sent_id]:
                    hits += 1
                    score += hits / (i + 1)
            #sum_ap_5 += score / min(5, len(dict_ann_lemma[sent_id]))
            sum_ap_5 += score / 5

    return {"Accuracy@1@top1": count_acc_1 / count_matchups, "Accuracy@3@top1": count_acc_3 / count_matchups, "Accuracy@5@top1": count_acc_5 / count_matchups,
            "Accuracy@top1": count_acc / count_matchups, "Accuracy_lemma@top1": count_acc_lemma / count_matchups,
            "MAP@3": sum_ap_3 / count_matchups, "MAP@4": sum_ap_4 / count_matchups, "MAP@5": sum_ap_5 / count_matchups,
            "Recall": count_recall / count_total_ann_words, "Recall@1": count_recall_1 / count_total_ann_words, "Recall@3": count_recall_3 / count_total_ann_words, 
            "Recall@5": count_recall_5 / count_total_ann_words, "Recall_lemma": count_recall_lemma / count_total_ann_words,
            "Potential": count_potential / count_matchups, "Potential@1": count_potential_1 / count_matchups, "Potential@3": count_potential_3 / count_matchups, 
            "Potential@5": count_potential_5 / count_matchups, "Potential_lemma": count_potential_lemma / count_matchups,
            "Precision": count_precision / count_total_pred_words, "Precision@1": count_precision_1 / count_pred_words_1, "Precision@3": count_precision_3 / count_pred_words_3, 
            "Precision@5": count_precision_5 / count_pred_words_5, "Precision_lemma": count_precision_lemma / count_total_pred_words}


results_list = []

results_bertRo = calculate_results(annotated_data, bertRo_data)
results_bertRo["model"] = "dumitrescustefan/bert-base-romanian-cased-v1"
results_list.append(results_bertRo)

results_RoBert = calculate_results(annotated_data, RoBert_l_data)
results_RoBert["model"] = "readerbench/RoBERT-large"
results_list.append(results_RoBert)

results_bertML = calculate_results(annotated_data, bertML_data)
results_bertML["model"] = "google-bert/bert-base-multilingual-cased"
results_list.append(results_bertML)

results_trainedBertRo = calculate_results(annotated_data, trained_bertRo_data)
results_trainedBertRo["model"] = "trained dumitrescustefan/bert-base-romanian-cased-v1 on this data"
results_list.append(results_trainedBertRo)

results_gpt4o = calculate_results(annotated_data, gpt4o_data)
results_gpt4o["model"] = "GPT-4o-mini"
results_list.append(results_gpt4o)

results_llama3Ro = calculate_results(annotated_data, llama3_data)
results_llama3Ro["model"] = "OpenLLM-Ro/RoLlama3-8b-Instruct-2025-04-23"
results_list.append(results_llama3Ro)

with open('scoring/model_comparisons.json', 'w') as f:
    json.dump(results_list, f, indent=4)