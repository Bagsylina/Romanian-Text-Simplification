import matplotlib.pyplot as plt
import json

with open('simplification_rating/results/model_comparisons.json', 'r') as f:
    models_results = json.load(f)

k = [1, 2, 3, 4, 5, 6, 7]
k_labels = ["Research\nArticles", "Philosophy", "History", "Romanian\nLiterature", "Translated\nLiterature", "Textbooks", "News"]

scores_bert = []
scores_gpt = []
scores_llama = []
scores_miles = []

for categ in models_results[0]["subdomains"]:
    scores_bert.append(categ["scoreBert"])
    scores_gpt.append(categ["scoreGPT"])
    scores_llama.append(categ["scoreLlama"])
    scores_miles.append(categ["scoreMiles"])

plt.figure(figsize=(10, 6))

plt.plot(k_labels, scores_bert, 'r', marker='s', label='BERT-Romanian')
plt.plot(k_labels, scores_gpt, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, scores_llama, 'm', marker='s', label='RoLlama3')
plt.plot(k_labels, scores_miles, 'g', marker='s', label='MILES')

plt.title('Ratings')
plt.xlabel('category')
plt.ylabel('score')

plt.grid(True)
plt.ylim(bottom=2, top=8.5) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('simplification_rating/plots/Ratings.pdf')

plt.close()


scores_bert = []
scores_gpt = []
scores_llama = []
scores_miles = []

for categ in models_results[0]["subdomains"]:
    scores_bert.append(categ["rankBert"])
    scores_gpt.append(categ["rankGPT"])
    scores_llama.append(categ["rankLlama"])
    scores_miles.append(categ["rankMiles"])

plt.figure(figsize=(10, 6))

plt.plot(k_labels, scores_bert, 'r', marker='s', label='BERT-Romanian')
plt.plot(k_labels, scores_gpt, 'b', marker='s', label='GPT-4o mini')
plt.plot(k_labels, scores_llama, 'm', marker='s', label='RoLlama3')
plt.plot(k_labels, scores_miles, 'g', marker='s', label='MILES')

plt.title('Average Rankings')
plt.xlabel('category')
plt.ylabel('ranking')

plt.grid(True)
plt.ylim(bottom=1, top=4) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.gca().invert_yaxis()
plt.savefig('simplification_rating/plots/Rankings.pdf')

plt.close()


scores_bert = []
scores_gpt = []
scores_llama = []
scores_miles = []

for categ in models_results[0]["subdomains"]:
    scores_bert.append(categ["goodSimpBert"] * 100)
    scores_gpt.append(categ["goodSimpGPT"] * 100)
    scores_llama.append(categ["goodSimpLlama"] * 100)
    scores_miles.append(categ["goodSimpMiles"] * 100)

plt.figure(figsize=(10, 6))

plt.plot(k_labels, scores_bert, 'r', marker='s', label='BERT-Romanian (>8)')
plt.plot(k_labels, scores_gpt, 'b', marker='s', label='GPT-4o mini (>8)')
plt.plot(k_labels, scores_llama, 'm', marker='s', label='RoLlama3 (>8)')
plt.plot(k_labels, scores_miles, 'g', marker='s', label='MILES (>8)')

scores_bert = []
scores_gpt = []
scores_llama = []
scores_miles = []

for categ in models_results[0]["subdomains"]:
    scores_bert.append(categ["acceptSimpBert"] * 100)
    scores_gpt.append(categ["acceptSimpGPT"] * 100)
    scores_llama.append(categ["acceptSimpLlama"] * 100)
    scores_miles.append(categ["acceptSimpMiles"] * 100)

plt.plot(k_labels, scores_bert, 'darkred', marker='o', label='BERT-Romanian (>7)')
plt.plot(k_labels, scores_gpt, 'darkblue', marker='o', label='GPT-4o mini (>7)')
plt.plot(k_labels, scores_llama, 'darkmagenta', marker='o', label='RoLlama3 (>7)')
plt.plot(k_labels, scores_miles, 'darkgreen', marker='o', label='MILES (>7)')

plt.title('Number of Good Simplifications')
plt.xlabel('category')
plt.ylabel('score (%)')

plt.grid(True)
plt.ylim(bottom=0) 
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.savefig('simplification_rating/plots/GoodSimp.pdf')

plt.close()