import matplotlib.pyplot as plt
import json

with open('scoring/model_comparisons.json', 'r') as f:
    models_results = json.load(f)

k = [1, 3, 5, 7, 9]
k_labels = ['1', '3', '5', 'max', 'lemma']

acc_bert_ro = [models_results[0]["Accuracy@1@top1"] * 100, models_results[0]["Accuracy@3@top1"] * 100, models_results[0]["Accuracy@5@top1"] * 100, models_results[0]["Accuracy@top1"] * 100, models_results[0]["Accuracy_lemma@top1"] * 100]
acc_robert = [models_results[1]["Accuracy@1@top1"] * 100, models_results[1]["Accuracy@3@top1"] * 100, models_results[1]["Accuracy@5@top1"] * 100, models_results[1]["Accuracy@top1"] * 100, models_results[1]["Accuracy_lemma@top1"] * 100]
acc_bert_ml = [models_results[2]["Accuracy@1@top1"] * 100, models_results[2]["Accuracy@3@top1"] * 100, models_results[2]["Accuracy@5@top1"] * 100, models_results[2]["Accuracy@top1"] * 100, models_results[2]["Accuracy_lemma@top1"] * 100]
acc_bert_ro_trained = [models_results[3]["Accuracy@1@top1"] * 100, models_results[3]["Accuracy@3@top1"] * 100, models_results[3]["Accuracy@5@top1"] * 100, models_results[3]["Accuracy@top1"] * 100, models_results[3]["Accuracy_lemma@top1"] * 100]
acc_gpt_4o = [models_results[4]["Accuracy@1@top1"] * 100, models_results[4]["Accuracy@3@top1"] * 100, models_results[4]["Accuracy@5@top1"] * 100, models_results[4]["Accuracy@top1"] * 100, models_results[4]["Accuracy_lemma@top1"] * 100]
acc_llama3_ro = [models_results[5]["Accuracy@1@top1"] * 100, models_results[5]["Accuracy@3@top1"] * 100, models_results[5]["Accuracy@5@top1"] * 100, models_results[5]["Accuracy@top1"] * 100, models_results[5]["Accuracy_lemma@top1"] * 100]

plt.plot(k_labels, acc_bert_ro, 'r', marker='s', label='bert-romanian')
plt.plot(k_labels, acc_robert, 'y', marker='s', label='RoBERT-large')
plt.plot(k_labels, acc_bert_ml, 'g', marker='s', label='bert-multilingual')
plt.plot(k_labels, acc_bert_ro_trained, 'c', marker='s', label='trained bert-romanian')
plt.plot(k_labels, acc_gpt_4o, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, acc_llama3_ro, 'm', marker='s', label='RoLlama3')

plt.title('Accuracy@k@top1')
plt.xlabel('number of candidates (k)')
plt.ylabel('score (%)')

plt.grid(True)
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('scoring/plots/Accuracy@k@top1.pdf')

plt.close()

k = [3, 4, 5]

map_bert_ro = [models_results[0]["MAP@3"] * 100, models_results[0]["MAP@4"] * 100, models_results[0]["MAP@5"] * 100]
map_robert = [models_results[1]["MAP@3"] * 100, models_results[1]["MAP@4"] * 100, models_results[1]["MAP@5"] * 100]
map_bert_ml = [models_results[2]["MAP@3"] * 100, models_results[2]["MAP@4"] * 100, models_results[2]["MAP@5"] * 100]
map_bert_ro_trained = [models_results[3]["MAP@3"] * 100, models_results[3]["MAP@4"] * 100, models_results[3]["MAP@5"] * 100]
map_gpt_4o = [models_results[4]["MAP@3"] * 100, models_results[4]["MAP@4"] * 100, models_results[4]["MAP@5"] * 100]
map_llama3_ro = [models_results[5]["MAP@3"] * 100, models_results[5]["MAP@4"] * 100, models_results[5]["MAP@5"] * 100]

plt.plot(k, map_bert_ro, 'r', marker='s', label='bert-romanian')
plt.plot(k, map_robert, 'y', marker='s', label='RoBERT-large')
plt.plot(k, map_bert_ml, 'g', marker='s', label='bert-multilingual')
plt.plot(k, map_bert_ro_trained, 'c', marker='s', label='trained bert-romanian')
plt.plot(k, map_gpt_4o, 'b', marker='s', label='GPT-4o mini')
plt.plot(k, map_llama3_ro, 'm', marker='s', label='RoLlama3')

plt.title('MAP@k')
plt.xlabel('number of candidates (k)')
plt.ylabel('score (%)')

plt.grid(True)
plt.xticks([3, 4, 5])
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('scoring/plots/MAP@k.pdf')

plt.close()

k = [1, 3, 5, 7, 9]
k_labels = ['1', '3', '5', 'max', 'lemma']

recall_bert_ro = [models_results[0]["Recall@1"] * 100, models_results[0]["Recall@3"] * 100, models_results[0]["Recall@5"] * 100, models_results[0]["Recall"] * 100, models_results[0]["Recall_lemma"] * 100]
recall_robert = [models_results[1]["Recall@1"] * 100, models_results[1]["Recall@3"] * 100, models_results[1]["Recall@5"] * 100, models_results[1]["Recall"] * 100, models_results[1]["Recall_lemma"] * 100]
recall_bert_ml = [models_results[2]["Recall@1"] * 100, models_results[2]["Recall@3"] * 100, models_results[2]["Recall@5"] * 100, models_results[2]["Recall"] * 100, models_results[2]["Recall_lemma"] * 100]
recall_bert_ro_trained = [models_results[3]["Recall@1"] * 100, models_results[3]["Recall@3"] * 100, models_results[3]["Recall@5"] * 100, models_results[3]["Recall"] * 100, models_results[3]["Recall_lemma"] * 100]
recall_gpt_4o = [models_results[4]["Recall@1"] * 100, models_results[4]["Recall@3"] * 100, models_results[4]["Recall@5"] * 100, models_results[4]["Recall"] * 100, models_results[4]["Recall_lemma"] * 100]
recall_llama3_ro = [models_results[5]["Recall@1"] * 100, models_results[5]["Recall@3"] * 100, models_results[5]["Recall@5"] * 100, models_results[5]["Recall"] * 100, models_results[5]["Recall_lemma"] * 100]

plt.plot(k_labels, recall_bert_ro, 'r', marker='s', label='bert-romanian')
plt.plot(k_labels, recall_robert, 'y', marker='s', label='RoBERT-large')
plt.plot(k_labels, recall_bert_ml, 'g', marker='s', label='bert-multilingual')
plt.plot(k_labels, recall_bert_ro_trained, 'c', marker='s', label='trained bert-romanian')
plt.plot(k_labels, recall_gpt_4o, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, recall_llama3_ro, 'm', marker='s', label='RoLlama3')

plt.title('Recall@k')
plt.xlabel('number of candidates (k)')
plt.ylabel('score (%)')

plt.grid(True)
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('scoring/plots/Recall@k.pdf')

plt.close()

k = [1, 3, 5, 7, 9]
k_labels = ['1', '3', '5', 'max', 'lemma']

potential_bert_ro = [models_results[0]["Potential@1"] * 100, models_results[0]["Potential@3"] * 100, models_results[0]["Potential@5"] * 100, models_results[0]["Potential"] * 100, models_results[0]["Potential_lemma"] * 100]
potential_robert = [models_results[1]["Potential@1"] * 100, models_results[1]["Potential@3"] * 100, models_results[1]["Potential@5"] * 100, models_results[1]["Potential"] * 100, models_results[1]["Potential_lemma"] * 100]
potential_bert_ml = [models_results[2]["Potential@1"] * 100, models_results[2]["Potential@3"] * 100, models_results[2]["Potential@5"] * 100, models_results[2]["Potential"] * 100, models_results[2]["Potential_lemma"] * 100]
potential_bert_ro_trained = [models_results[3]["Potential@1"] * 100, models_results[3]["Potential@3"] * 100, models_results[3]["Potential@5"] * 100, models_results[3]["Potential"] * 100, models_results[3]["Potential_lemma"] * 100]
potential_gpt_4o = [models_results[4]["Potential@1"] * 100, models_results[4]["Potential@3"] * 100, models_results[4]["Potential@5"] * 100, models_results[4]["Potential"] * 100, models_results[4]["Potential_lemma"] * 100]
potential_llama3_ro = [models_results[5]["Potential@1"] * 100, models_results[5]["Potential@3"] * 100, models_results[5]["Potential@5"] * 100, models_results[5]["Potential"] * 100, models_results[5]["Potential_lemma"] * 100]

plt.plot(k_labels, potential_bert_ro, 'r', marker='s', label='bert-romanian')
plt.plot(k_labels, potential_robert, 'y', marker='s', label='RoBERT-large')
plt.plot(k_labels, potential_bert_ml, 'g', marker='s', label='bert-multilingual')
plt.plot(k_labels, potential_bert_ro_trained, 'c', marker='s', label='trained bert-romanian')
plt.plot(k_labels, potential_gpt_4o, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, potential_llama3_ro, 'm', marker='s', label='RoLlama3')

plt.title('Potential@k')
plt.xlabel('number of candidates (k)')
plt.ylabel('score (%)')

plt.grid(True)
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('scoring/plots/Potential@k.pdf')

plt.close()

k = [1, 3, 5, 7, 9]
k_labels = ['1', '3', '5', 'max', 'lemma']

precision_bert_ro = [models_results[0]["Precision@1"] * 100, models_results[0]["Precision@3"] * 100, models_results[0]["Precision@5"] * 100, models_results[0]["Precision"] * 100, models_results[0]["Precision_lemma"] * 100]
precision_robert = [models_results[1]["Precision@1"] * 100, models_results[1]["Precision@3"] * 100, models_results[1]["Precision@5"] * 100, models_results[1]["Precision"] * 100, models_results[1]["Precision_lemma"] * 100]
precision_bert_ml = [models_results[2]["Precision@1"] * 100, models_results[2]["Precision@3"] * 100, models_results[2]["Precision@5"] * 100, models_results[2]["Precision"] * 100, models_results[2]["Precision_lemma"] * 100]
precision_bert_ro_trained = [models_results[3]["Precision@1"] * 100, models_results[3]["Precision@3"] * 100, models_results[3]["Precision@5"] * 100, models_results[3]["Precision"] * 100, models_results[3]["Precision_lemma"] * 100]
precision_gpt_4o = [models_results[4]["Precision@1"] * 100, models_results[4]["Precision@3"] * 100, models_results[4]["Precision@5"] * 100, models_results[4]["Precision"] * 100, models_results[4]["Precision_lemma"] * 100]
precision_llama3_ro = [models_results[5]["Precision@1"] * 100, models_results[5]["Precision@3"] * 100, models_results[5]["Precision@5"] * 100, models_results[5]["Precision"] * 100, models_results[5]["Precision_lemma"] * 100]

plt.plot(k_labels, precision_bert_ro, 'r', marker='s', label='bert-romanian')
plt.plot(k_labels, precision_robert, 'y', marker='s', label='RoBERT-large')
plt.plot(k_labels, precision_bert_ml, 'g', marker='s', label='bert-multilingual')
plt.plot(k_labels, precision_bert_ro_trained, 'c', marker='s', label='trained bert-romanian')
plt.plot(k_labels, precision_gpt_4o, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, precision_llama3_ro, 'm', marker='s', label='RoLlama3')

plt.title('Precision@k')
plt.xlabel('number of candidates (k)')
plt.ylabel('score (%)')

plt.grid(True)
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('scoring/plots/Precision@k.pdf')