import json
from scipy.stats import kendalltau
import math

with open('scoring/ranking_data.json', 'r') as f:
    annotated_data = json.load(f)

with open('scoring/gpt4o_suggestions.json', 'r') as f:
    gpt4o_data = json.load(f)

with open('scoring/bertRo_suggestions.json', 'r') as f:
    bertRo_data = json.load(f)

with open('scoring/bertMulti_suggestions.json', 'r') as f:
    bertML_data = json.load(f)

count_ann = 0
count_gpt = 0
count_bro = 0
count_bml = 0

common = 0
nr_ranking_sets = 0
sum_tau = 0

dict_ann = {}
dict_gpt = {}
dict_bro = {}
dict_bml = {}

for data in annotated_data:
    dict_ann[data["sentence_id"]] = data["sorted_candidates"]

for data in gpt4o_data:
    dict_gpt[data["sentence_id"]] = [x[0] for x in data["suggestions"]]

for data in bertRo_data:
    dict_bro[data["sentence_id"]] = [x[0] for x in data["suggestions"]]

for data in bertML_data:
    dict_bml[data["sentence_id"]] = [x[0] for x in data["suggestions"]]


for key in dict_ann:
    if key in dict_ann and key in dict_gpt:
        common_list = []

        count_ann += len(dict_ann[key])
        count_gpt += len(dict_gpt[key])

        for word in dict_gpt[key]:
            if word in dict_ann[key]:
                common += 1
                common_list.append(word)

        if len(common_list) > 1:
            list_ann = [word for word in dict_ann[key] if word in common_list]
            list_gpt = [word for word in dict_gpt[key] if word in common_list]

            ranking_dict_ann = {word: i for i, word in enumerate(list_ann)}
            ranking_dict_gpt = {word: i for i, word in enumerate(list_gpt)}

            ranking_ann = [ranking_dict_ann[word] for word in common_list]
            ranking_gpt = [ranking_dict_gpt[word] for word in common_list]

            tau, _ = kendalltau(ranking_ann, ranking_gpt)

            if tau is not None and not math.isnan(tau):
                nr_ranking_sets += 1
                sum_tau += tau 

results_ann_gpt = {'Title': 'Comparison between annotated dataset and gpt4o suggestions',
                   'Number of annotated suggestions': count_ann, 'Number of gpt4o suggestions': count_gpt,
                   'Number of common suggestions': common, '% of annotated words suggested by gpt4o': common/count_ann,
                   '% of words suggested by gpt4o in annotated list': common/count_gpt, 'Number of ranking sets': nr_ranking_sets,
                   'Average Kendall Tau': sum_tau / nr_ranking_sets}


count_ann = 0
common = 0
nr_ranking_sets = 0
sum_tau = 0

for key in dict_ann:
    if key in dict_ann and key in dict_bro:
        common_list = []

        count_ann += len(dict_ann[key])
        count_bro += len(dict_bro[key])

        for word in dict_bro[key]:
            if word in dict_ann[key]:
                common += 1
                common_list.append(word)
        
        if len(common_list) > 1:
            list_ann = [word for word in dict_ann[key] if word in common_list]
            list_bro = [word for word in dict_bro[key] if word in common_list]

            ranking_dict_ann = {word: i for i, word in enumerate(list_ann)}
            ranking_dict_bro = {word: i for i, word in enumerate(list_bro)}

            ranking_ann = [ranking_dict_ann[word] for word in common_list]
            ranking_bro = [ranking_dict_bro[word] for word in common_list]

            tau, _ = kendalltau(ranking_ann, ranking_bro)

            if tau is not None and not math.isnan(tau):
                nr_ranking_sets += 1
                sum_tau += tau 

results_ann_bro = {'Title': 'Comparison between annotated dataset and romanian bert suggestions',
                   'Number of annotated suggestions': count_ann, 'Number of romanian bert suggestions': count_bro,
                   'Number of common suggestions': common, '% of annotated words suggested by bert': common/count_ann,
                   '% of words suggested by bert in annotated list': common/count_bro, 'Number of ranking sets': nr_ranking_sets,
                   'Average Kendall Tau': sum_tau / nr_ranking_sets}


count_ann = 0
common = 0

for key in dict_ann:
    if key in dict_ann and key in dict_bml:
        count_ann += len(dict_ann[key])
        count_bml += len(dict_bml[key])

        for word in dict_bml[key]:
            if word in dict_ann[key]:
                common += 1

results_ann_bml = {'Title': 'Comparison between annotated dataset and multilingual bert suggestions',
                   'Number of annotated suggestions': count_ann, 'Number of multilingual bert suggestions': count_bml,
                   'Number of common suggestions': common, '% of annotated words suggested by bert': common/count_ann,
                   '% of words suggested by bert in annotated list': common/count_bml}


count_bro = 0
count_gpt = 0
common = 0
nr_ranking_sets = 0
sum_tau = 0

for key in dict_gpt:
    if key in dict_gpt and key in dict_bro:
        common_list = []

        count_gpt += len(dict_gpt[key])
        count_bro += len(dict_bro[key])

        for word in dict_bro[key]:
            if word in dict_gpt[key]:
                common += 1
                common_list.append(word)

        if len(common_list) > 1:
            list_gpt = [word for word in dict_gpt[key] if word in common_list]
            list_bro = [word for word in dict_bro[key] if word in common_list]

            ranking_dict_gpt = {word: i for i, word in enumerate(list_gpt)}
            ranking_dict_bro = {word: i for i, word in enumerate(list_bro)}

            ranking_gpt = [ranking_dict_gpt[word] for word in common_list]
            ranking_bro = [ranking_dict_bro[word] for word in common_list]

            tau, _ = kendalltau(ranking_gpt, ranking_bro)

            if tau is not None and not math.isnan(tau):
                nr_ranking_sets += 1
                sum_tau += tau 

results_gpt_bro = {'Title': 'Comparison between gpt4o suggestions and romanian bert suggestions',
                   'Number of gpt4o suggestions': count_gpt, 'Number of romanian bert suggestions': count_bro,
                   'Number of common suggestions': common, '% of gpt words suggested by bert': common/count_gpt,
                   '% of bert words suggested by gpt': common/count_bro, 'Number of ranking sets': nr_ranking_sets,
                   'Average Kendall Tau': sum_tau / nr_ranking_sets}


count_bml = 0
count_gpt = 0
common = 0

for key in dict_gpt:
    if key in dict_gpt and key in dict_bml:
        count_gpt += len(dict_gpt[key])
        count_bml += len(dict_bml[key])

        for word in dict_bml[key]:
            if word in dict_gpt[key]:
                common += 1

results_gpt_bml = {'Title': 'Comparison between gpt4o suggestions and multilingual bert suggestions',
                   'Number of gpt4o suggestions': count_gpt, 'Number of multilingual bert suggestions': count_bml,
                   'Number of common suggestions': common, '% of gpt words suggested by bert': common/count_gpt,
                   '% of bert words suggested by gpt': common/count_bml}


count_bro = 0
count_bml = 0
common = 0

for key in dict_bml:
    if key in dict_bml and key in dict_bro:
        count_bml += len(dict_bml[key])
        count_bro += len(dict_bro[key])

        for word in dict_bro[key]:
            if word in dict_bml[key]:
                common += 1

results_bro_bml = {'Title': 'Comparison between romanian bert suggestions and multilingual bert suggestions',
                   'Number of romanian bert suggestions': count_bro, 'Number of multilingual bert suggestions': count_bml,
                   'Number of common suggestions': common, '% of ro bert words suggested by ml bert': common/count_bro,
                   '% of ml bert words suggested by ro bert': common/count_bml}


import spacy
NLP = spacy.load("ro_core_news_lg")

for data in annotated_data:
    dict_ann[data["sentence_id"]] = [NLP(x)[0].lemma_ for x in data["sorted_candidates"]]

for data in gpt4o_data:
    dict_gpt[data["sentence_id"]] = [NLP(x[0])[0].lemma_ for x in data["suggestions"]]

for data in bertRo_data:
    dict_bro[data["sentence_id"]] = [NLP(x[0])[0].lemma_ for x in data["suggestions"]]

for data in bertML_data:
    dict_bml[data["sentence_id"]] = [NLP(x[0])[0].lemma_ for x in data["suggestions"]]

count_ann = 0
count_gpt = 0
common_ann = 0
common_gpt = 0


for key in dict_ann:
    if key in dict_ann and key in dict_gpt:
        count_ann += len(dict_ann[key])
        count_gpt += len(dict_gpt[key])

        for word in dict_gpt[key]:
            if word in dict_ann[key]:
                common_gpt += 1

        for word in dict_ann[key]:
            if word in dict_gpt[key]:
                common_ann += 1

results_lemma_ann_gpt = {'Title': 'Comparison of lemmas of the words between annotated dataset and gpt4o suggestions',
                         'Number of annotated suggestions': count_ann, 'Number of gpt4o suggestions': count_gpt,
                         '% of annotated words suggested by gpt4o': common_ann/count_ann, '% of words suggested by gpt4o in annotated list': common_gpt/count_gpt}


count_bro = 0
count_ann = 0
common_ann = 0
common_bro = 0

for key in dict_ann:
    if key in dict_ann and key in dict_bro:
        count_ann += len(dict_ann[key])
        count_bro += len(dict_bro[key])

        for word in dict_bro[key]:
            if word in dict_ann[key]:
                common_bro += 1

        for word in dict_ann[key]:
            if word in dict_bro[key]:
                common_ann += 1

results_lemma_ann_bro = {'Title': 'Comparison of lemmas of the words between annotated dataset and romanian bert suggestions',
                         'Number of annotated suggestions': count_ann, 'Number of romanian bert suggestions': count_bro,
                         '% of annotated words suggested by bert': common_ann/count_ann, '% of words suggested by bert in annotated list': common_bro/count_bro}


count_bro = 0
count_gpt = 0
common_gpt = 0
common_bro = 0

for key in dict_gpt:
    if key in dict_gpt and key in dict_bro:
        count_gpt += len(dict_gpt[key])
        count_bro += len(dict_bro[key])

        for word in dict_bro[key]:
            if word in dict_gpt[key]:
                common_bro += 1

        for word in dict_gpt[key]:
            if word in dict_bro[key]:
                common_gpt += 1

results_lemma_gpt_bro = {'Title': 'Comparison of lemmas of the words between gpt4o suggestions and romanian bert suggestions',
                         'Number of gpt4o suggestions': count_gpt, 'Number of romanian bert suggestions': count_bro,
                         '% of gpt words suggested by bert': common_gpt/count_gpt, '% of bert words suggested by gpt': common_bro/count_bro}


results = [results_ann_gpt, results_ann_bro, results_ann_bml, results_gpt_bro, results_gpt_bml, results_bro_bml, results_lemma_ann_gpt, results_lemma_ann_bro, results_lemma_gpt_bro]

with open('scoring/model_comparisons.json', 'w') as f:
    json.dump(results, f, indent=4)
