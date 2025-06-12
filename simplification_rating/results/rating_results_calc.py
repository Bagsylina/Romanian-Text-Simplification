import json
from os.path import isfile, join, dirname
from os import listdir

#open folder with all text files of the data
source_texts_dir = "simplification_rating/annotations"

#list with all text files
json_files = [f for f in listdir(source_texts_dir) if isfile(join(source_texts_dir, f))]

annotations = []

for file in json_files:
    filename = join(source_texts_dir, file)
    cur_ann = []
    with open(filename, 'r', encoding='utf-8') as f:
        cur_ann = json.load(f)

    for ann in cur_ann:
        if not ann["annotations"][0]["result"]:
            continue

        ann_data = {}
        ann_data["sent_id"] = ann["data"]["id"]

        for result in ann["annotations"][0]["result"]:
            if result["from_name"] == "ranking":
                ann_data["ranking"] = result["value"]["ranker"]["ranking"]
            elif result["from_name"] == "scoreBert":
                ann_data["scoreBert"] = result["value"]["number"]
            elif result["from_name"] == "scoreGPT":
                ann_data["scoreGPT"] = result["value"]["number"]
            elif result["from_name"] == "scoreLlama":
                ann_data["scoreLlama"] = result["value"]["number"]
            elif result["from_name"] == "scoreMiles":
                ann_data["scoreMiles"] = result["value"]["number"]

        if "ranking" not in ann_data:
            continue

        annotations.append(ann_data)

with open('simplification_rating/rating_annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)

with open('simplification_rating/selected_sentences.json', 'r') as f:
    sentence_list = json.load(f)

sentence_dict = {}
for sent_data in sentence_list:
    sentence_dict[sent_data["id"]] = sent_data["subfolder"]

sum_ro = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
domain = ["Research Articles", "Philosophy", "History", "Romanian Literature", "Translated Literature", "Textbooks", "News"]
sum_rank_ro = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
sum_proc_ro = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
sum_proc2_ro = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
count_ro = [0, 0, 0, 0, 0, 0, 0, 0]
sum_meye = [0, 0, 0, 0]
sum_rank_meye = [0, 0, 0, 0]
sum_proc_meye = [0, 0, 0, 0]
sum_proc2_meye = [0, 0, 0, 0]
count_meye = 0

for ann in annotations:
    if ann["sent_id"][:2] == "ro":
        count_ro[0] += 1
        sum_ro[0][0] += ann["scoreBert"] 
        sum_ro[0][1] += ann["scoreGPT"] 
        sum_ro[0][2] += ann["scoreLlama"] 
        sum_ro[0][3] += ann["scoreMiles"] 

        x = 1
        match sentence_dict[ann["sent_id"]]:
            case "Articole_Cercetari - Research_Articles":
                x = 1
            case "Filosoly - Philosophy":
                x = 2
            case "Istorie-History":
                x = 3
            case "Literatura_Tradusa - Translated_Literature":
                x = 5
            case "Manuale - Textbooks":
                x = 6
            case "Stiri - News":
                x = 7
            case _:
                x = 4

        count_ro[x] += 1
        sum_ro[x][0] += ann["scoreBert"] 
        sum_ro[x][1] += ann["scoreGPT"] 
        sum_ro[x][2] += ann["scoreLlama"] 
        sum_ro[x][3] += ann["scoreMiles"] 

        if ann["scoreBert"] > 8:
            sum_proc_ro[0][0] += 1
            sum_proc_ro[x][0] += 1
        if ann["scoreGPT"] > 8:
            sum_proc_ro[0][1] += 1
            sum_proc_ro[x][1] += 1
        if ann["scoreLlama"] > 8:
            sum_proc_ro[0][2] += 1
            sum_proc_ro[x][2] += 1
        if ann["scoreMiles"] > 8:
            sum_proc_ro[0][3] += 1
            sum_proc_ro[x][3] += 1

        if ann["scoreBert"] > 7:
            sum_proc2_ro[0][0] += 1
            sum_proc2_ro[x][0] += 1
        if ann["scoreGPT"] > 7:
            sum_proc2_ro[0][1] += 1
            sum_proc2_ro[x][1] += 1
        if ann["scoreLlama"] > 7:
            sum_proc2_ro[0][2] += 1
            sum_proc2_ro[x][2] += 1
        if ann["scoreMiles"] > 7:
            sum_proc2_ro[0][3] += 1
            sum_proc2_ro[x][3] += 1

        sum_rank_ro[0][int(ann["ranking"][0]) - 1] += 1
        sum_rank_ro[0][int(ann["ranking"][1]) - 1] += 2
        sum_rank_ro[0][int(ann["ranking"][2]) - 1] += 3
        sum_rank_ro[0][int(ann["ranking"][3]) - 1] += 4

        sum_rank_ro[x][int(ann["ranking"][0]) - 1] += 1
        sum_rank_ro[x][int(ann["ranking"][1]) - 1] += 2
        sum_rank_ro[x][int(ann["ranking"][2]) - 1] += 3
        sum_rank_ro[x][int(ann["ranking"][3]) - 1] += 4

    else:
        count_meye += 1
        sum_meye[0] += ann["scoreBert"] 
        sum_meye[1] += ann["scoreGPT"] 
        sum_meye[2] += ann["scoreLlama"] 
        sum_meye[3] += ann["scoreMiles"] 

        if ann["scoreBert"] > 8:
            sum_proc_meye[0] += 1
        if ann["scoreGPT"] > 8:
            sum_proc_meye[1] += 1
        if ann["scoreLlama"] > 8:
            sum_proc_meye[2] += 1
        if ann["scoreMiles"] > 8:
            sum_proc_meye[3] += 1

        if ann["scoreBert"] > 7:
            sum_proc2_meye[0] += 1
        if ann["scoreGPT"] > 7:
            sum_proc2_meye[1] += 1
        if ann["scoreLlama"] > 7:
            sum_proc2_meye[2] += 1
        if ann["scoreMiles"] > 7:
            sum_proc2_meye[3] += 1

        sum_rank_meye[int(ann["ranking"][0]) - 1] += 1
        sum_rank_meye[int(ann["ranking"][1]) - 1] += 2
        sum_rank_meye[int(ann["ranking"][2]) - 1] += 3
        sum_rank_meye[int(ann["ranking"][3]) - 1] += 4


results = [{}, {}]
results[0]["title"] = "Romanian Language Repository"

results[0]["scoreBert"] = sum_ro[0][0] / count_ro[0]
results[0]["scoreGPT"] = sum_ro[0][1] / count_ro[0]
results[0]["scoreLlama"] = sum_ro[0][2] / count_ro[0]
results[0]["scoreMiles"] = sum_ro[0][3] / count_ro[0]

results[0]["goodSimpBert"] = sum_proc_ro[0][0] / count_ro[0]
results[0]["goodSimpGPT"] = sum_proc_ro[0][1] / count_ro[0]
results[0]["goodSimpLlama"] = sum_proc_ro[0][2] / count_ro[0]
results[0]["goodSimpMiles"] = sum_proc_ro[0][3] / count_ro[0]

results[0]["acceptSimpBert"] = sum_proc2_ro[0][0] / count_ro[0]
results[0]["acceptSimpGPT"] = sum_proc2_ro[0][1] / count_ro[0]
results[0]["acceptSimpLlama"] = sum_proc2_ro[0][2] / count_ro[0]
results[0]["acceptSimpMiles"] = sum_proc2_ro[0][3] / count_ro[0]

results[0]["rankBert"] = sum_rank_ro[0][0] / count_ro[0]
results[0]["rankGPT"] = sum_rank_ro[0][1] / count_ro[0]
results[0]["rankLlama"] = sum_rank_ro[0][2] / count_ro[0]
results[0]["rankMiles"] = sum_rank_ro[0][3] / count_ro[0]

results[0]["subdomains"] = []

for i in range(1, 8):
    domain_result = {}
    domain_result["title"] = domain[i - 1]

    domain_result["scoreBert"] = sum_ro[i][0] / count_ro[i]
    domain_result["scoreGPT"] = sum_ro[i][1] / count_ro[i]
    domain_result["scoreLlama"] = sum_ro[i][2] / count_ro[i]
    domain_result["scoreMiles"] = sum_ro[i][3] / count_ro[i]

    domain_result["goodSimpBert"] = sum_proc_ro[i][0] / count_ro[i]
    domain_result["goodSimpGPT"] = sum_proc_ro[i][1] / count_ro[i]
    domain_result["goodSimpLlama"] = sum_proc_ro[i][2] / count_ro[i]
    domain_result["goodSimpMiles"] = sum_proc_ro[i][3] / count_ro[i]

    domain_result["acceptSimpBert"] = sum_proc2_ro[i][0] / count_ro[i]
    domain_result["acceptSimpGPT"] = sum_proc2_ro[i][1] / count_ro[i]
    domain_result["acceptSimpLlama"] = sum_proc2_ro[i][2] / count_ro[i]
    domain_result["acceptSimpMiles"] = sum_proc2_ro[i][3] / count_ro[i]

    domain_result["rankBert"] = sum_rank_ro[i][0] / count_ro[i]
    domain_result["rankGPT"] = sum_rank_ro[i][1] / count_ro[i]
    domain_result["rankLlama"] = sum_rank_ro[i][2] / count_ro[i]
    domain_result["rankMiles"] = sum_rank_ro[i][3] / count_ro[i]

    results[0]["subdomains"].append(domain_result)


results[1]["title"] = "Multipleye"

results[1]["scoreBert"] = sum_meye[0] / count_meye
results[1]["scoreGPT"] = sum_meye[1] / count_meye
results[1]["scoreLlama"] = sum_meye[2] / count_meye
results[1]["scoreMiles"] = sum_meye[3] / count_meye

results[1]["goodSimpBert"] = sum_proc_meye[0] / count_meye
results[1]["goodSimpGPT"] = sum_proc_meye[1] / count_meye
results[1]["goodSimpLlama"] = sum_proc_meye[2] / count_meye
results[1]["goodSimpMiles"] = sum_proc_meye[3] / count_meye

results[1]["acceptSimpBert"] = sum_proc2_meye[0] / count_meye
results[1]["acceptSimpGPT"] = sum_proc2_meye[1] / count_meye
results[1]["acceptSimpLlama"] = sum_proc2_meye[2] / count_meye
results[1]["acceptSimpMiles"] = sum_proc2_meye[3] / count_meye

results[1]["rankBert"] = sum_rank_meye[0] / count_meye
results[1]["rankGPT"] = sum_rank_meye[1] / count_meye
results[1]["rankLlama"] = sum_rank_meye[2] / count_meye
results[1]["rankMiles"] = sum_rank_meye[3] / count_meye

with open('simplification_rating/model_comparisons.json', 'w') as f:
    json.dump(results, f, indent=4)