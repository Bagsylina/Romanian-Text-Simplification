import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.functional as F
import numpy as np
from transformers import TrainerCallback

from transformers import BertForMaskedLM, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertConfig

import random
import numpy as np
import torch
import transformers

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

set_seed(42)

def print_pretty_metrics(metrics):
    print("\n=== Evaluation Metrics ===")
    for key in metrics.keys():
        print(f"{key:<20}: {metrics[key]:.2f}")
    print("=" * 28 + "\n")


class PrettyPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print_pretty_metrics(metrics)


def make_target_probs(length, device, decay_type='linear', mask=0):
    # asta era folosit pentru Kullback-Leibler divergence
    # https://colah.github.io/posts/2015-09-Visual-Information/
    # sa fac niste distributii de ponderi ca sa conteze mai mult primele decat ultimele
    # mask e pentru situatia in care avem o lungime mare
    # sa zicem lenght=10 dar numai primele 3 sunt importante, resutul de la 4 incolo
    # sunt valori de padding; nu facem nimic de genul aici
    if mask > length:
        raise ValueError("mask must be <= length")
    if mask <= 0:
        mask = length
    if decay_type == "linear":
        weights = torch.arange(mask, 0, -1, dtype=torch.float32, device=device)
    elif decay_type == "exp":
        weights = F.softmax(torch.arange(mask, 0, -1, dtype=torch.float32, device=device))
        return weights
        #weights = torch.exp(-torch.arange(mask, dtype=torch.float32, device=device))
    elif decay_type == "uniform":
        weights = torch.ones(mask, dtype=torch.float32, device=device)
    elif decay_type == "one-hot":
        weights = torch.full((mask,), 0.1 / (mask - 1) if mask > 1 else 0.0, dtype=torch.float32, device=device)
        weights[0] = 0.9
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    weights = weights / weights.sum()
    full_probs = torch.zeros(length, dtype=torch.float32, device=device)
    full_probs[:mask] = weights
    return full_probs


class RankingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_candidates = 20

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence = row["sentence"]
        word = row["word"]
        candidates = row["candidates"]
        # Only keep the first token of each candidate
        candidate_token_ids = [
            self.tokenizer.encode(c, add_special_tokens=False)[0] for c in candidates
        ]
        # tried removing multi-token words
        # maybe better try mt5 https://github.com/huggingface/transformers/issues/3985 in the future
        #candidate_token_ids = [c[0] for c in candidate_token_ids if len(c) == 1]
        #if not candidate_token_ids:
        #    candidate_token_ids = [1]
        # no padding for batch size of 1
        #candidate_token_ids += [-100] * (self.max_candidates - len(candidate_token_ids))
        masked_sentence = sentence.replace(word, "[MASK]", 1)
        enc = self.tokenizer(masked_sentence,
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # there is no difference between candidates and labels :)
            # le-am pastrat pe ambele ca m-am tot jucat si am avut mai multe iteratii
            "candidates": candidate_token_ids,
            # astea de mai jos nici n-ar fi necesare
            # unele sunt pt debug, altele pentru inferenta
            "labels": torch.tensor([candidate_token_ids], dtype=torch.int),
            "sentence": row['sentence'],
            "sentence_id": row['sentence_id'],
            "word": row["word"],
            "candidate_words": row['candidates']
        }


def beam_search_single_mask(model, tokenizer, sentence, word, beam_width=5):
    '''Asta nu e beam search si are niste return values care nu sunt folosite;
    L-am facut asa pentru debug.
    Initial incercam sa fac [MASK] de mai multe ori, intre 1 si 4 tokens
    sa fac un beam search, dar l-am sters si a ramas hibridul asta
    un fill_mask simplu ar fi suficient 
    '''
    device = model.device
    masked_input = sentence.replace(word, "[MASK]", 1)
    enc = tokenizer(masked_input, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]  # [L, V]
    mask_pos = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        return []
    mask_index = mask_pos[0].item()
    log_probs = F.log_softmax(logits[mask_index], dim=-1)
    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
    results = []
    token_results = []
    for token_id, log_p in zip(topk_ids.tolist(), topk_log_probs.tolist()):
        current_word = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if current_word:
            results.append((current_word, log_p))
            token_results.append(token_id) #.append((token_id, log_p))
    return results, token_results


class RankingLossTrainer(Trainer):
    def __init__(self, *args, decay_type="linear", processing_class=BertTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1"), top_k=5, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_type = decay_type
        self.top_k = top_k
        self.processing_class = processing_class
        self.mask_token_id = self.processing_class.mask_token_id
        self.alpha = alpha
        self.total_epochs = self.args.num_train_epochs

    def ce_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        candidates = inputs["candidates"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]
        batch_size = input_ids.size(0)
        loss = 0.0
        for b in range(batch_size):
            mask_index = (input_ids[b] == self.mask_token_id).nonzero(as_tuple=True)[0][0].item()
            log_probs = logits[b, mask_index]
            candidate_ids = candidates[b]
            current_sentence_loss = 0.0
            #weights = make_target_probs(len(candidate_ids), device=log_probs.device, decay_type='linear')
            for idx, tid in enumerate(candidate_ids):
                crossent = F.cross_entropy(log_probs.unsqueeze(0),
                                           torch.tensor([tid], device=log_probs.device),
                                           label_smoothing=0.3)
                # make each position matter differently ?
                #current_sentence_loss += weights[idx]*crossent
                current_sentence_loss += crossent
            loss += (current_sentence_loss / len(candidate_ids))
        return (loss, {"logits": logits}) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self.ce_loss(model,
                               inputs,
                               return_outputs=return_outputs,
                               num_items_in_batch=num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        predictions = []
        for sentence, word in zip(inputs['sentence'], inputs['word']):
            candidates, cand_ids = beam_search_single_mask(model,
                                                           self.processing_class,
                                                           sentence,
                                                           word,
                                                           beam_width=self.top_k)
            predictions.append(cand_ids)
        labels = inputs.get("labels", torch.tensor([0] * len(predictions)))
        predictions = torch.tensor(predictions)
        return None, predictions, labels


def filter_true_ids(true_token_ids):
    return set(true_token_ids[true_token_ids != -100].tolist())


def get_hits(pred_ids, true_ids):
    return [1 if pid in true_ids else 0 for pid in pred_ids]


def compute_recall_at_k(pred_ids, true_ids, k):
    top_k_preds = pred_ids[:k]
    hits = set(top_k_preds).intersection(true_ids)
    return len(hits) / len(true_ids) if true_ids else 0.0


def compute_precision_at_k(pred_ids, true_ids, k):
    top_k_preds = pred_ids[:k]
    hits = set(top_k_preds).intersection(true_ids)
    return len(hits) / k


def compute_mrr(hits):
    for rank, hit in enumerate(hits, 1):
        if hit:
            return 1.0 / rank
    return 0.0


def compute_average_precision(hits, true_count):
    if true_count == 0:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, hit in enumerate(hits):
        if hit:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / true_count

def compute_average_precision_at_k(pred_ids, true_ids, k):
    score = 0.0
    hits = 0
    for i, pid in enumerate(pred_ids[:k]):
        if pid in true_ids:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(true_ids), k) if true_ids else 0.0

def compute_potential_at_k(pred_ids, true_ids, k):
    return 1.0 if any(pid in true_ids for pid in pred_ids[:k]) else 0.0

def compute_acc_n_top_gold(pred_ids, gold_top, n):
    return 1.0 if gold_top in pred_ids[:n] else 0.0


def compute_metrics(eval_pred):
    # metricile sunt facute cu chatgpt, nu le-am verificat daca sunt corecte
    predictions, labels = eval_pred.predictions, eval_pred.label_ids[0]
    total = len(predictions)
    ks = [1, 3, 5, 10]
    map_scores = {f"MAP@{k}": 0.0 for k in ks}
    potential_scores = {f"Potential@{k}": 0.0 for k in ks}
    acc_top_gold = {f"ACC@{n}@top_gold_1": 0.0 for n in [1, 2, 3]}
    for pred_ids, label_ids in zip(predictions, labels):
        true_ids = [tid for tid in label_ids if tid != -100]
        if not true_ids:
            continue
        gold_top = true_ids[0]
        for k in ks:
            map_scores[f"MAP@{k}"] += compute_average_precision_at_k(pred_ids, true_ids, k)
            potential_scores[f"Potential@{k}"] += compute_potential_at_k(pred_ids, true_ids, k)
        for n in [1, 2, 3]:
            acc_top_gold[f"ACC@{n}@top_gold_1"] += compute_acc_n_top_gold(pred_ids, gold_top, n)
    # Normalize
    metrics = {}
    metrics.update({k: v / total for k, v in map_scores.items()})
    metrics.update({k: v / total for k, v in potential_scores.items()})
    metrics.update({k: v / total for k, v in acc_top_gold.items()})
    #for k,v in metrics.items():
    #    metrics[k] = np.round(v, 2)
    return metrics


def ranking_data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "candidates": [f["candidates"] for f in features],
        "labels": [f["labels"] for f in features],
        "sentence": [f['sentence'] for f in features],
        "word": [f['word'] for f in features],
        "candidate_words": [f['candidate_words'] for f in features],
    }



model_name = "dumitrescustefan/bert-base-romanian-cased-v1"

# does not generalize as well
#model_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
# way worse: MAP 0.07 by default
#model_name = "readerbench/RoBERT-large"
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

for param in model.bert.parameters():
    param.requires_grad = False

# am incercat si sa fac freeze la tot mai putin ultimul block
# n-a mers mai bine, sunt prea putine date
# doar head-ul de clasificare are sens sa fie frozen
#for name, param in model.bert.named_parameters():
#    if "encoder.layer.11" in name:
#        param.requires_grad = True
#    else:
#        param.requires_grad = False

#df = pd.read_csv("data/ranking_data.csv")
#df["candidates"] = df["candidates"].apply(eval)
import json
with open("scoring/ranking_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={
    'sorted_candidates': 'candidates'
})

#gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
#train_idx, val_idx = next(gss.split(df, groups=df.sentence))
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df.sentence_id))
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

train_dataset = RankingDataset(train_df, tokenizer)
val_dataset = RankingDataset(val_df, tokenizer)

BS = 1
EPOCHS = 5
# pentru ca unele propozitii au 2 candidati iar altele 4, altele 1
# nu putem face batches de dimensiuni egale cu usurinta
# si atunci am facut SGD, adica batch size de 1
# dar la BS=1 loss-ul țopăie de colo colo
# asa ca am pus un gradient accumulation ca sa imit
# un batch size mai mare
# ideal ar fi să avem batch size fix si sa nu facem accumulation
# dar nu m-am mai chinuit
# o posibila solutie ar fi:
# 1. pentru fiecare candidat facem un singur exemplu si antrenam MLM cu cross-entropy 
# 2. dupa ce facem un fine-tuning re-rankingul candidatilor il facem cu o metoda de LCP
ACC_STEPS=16

args = TrainingArguments(
    output_dir="./ranking_output",
    overwrite_output_dir=True,
    save_safetensors=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    #warmup_steps=30,
    #lr_scheduler_type="cosine",
    learning_rate=0.0006,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    gradient_accumulation_steps=ACC_STEPS,
    weight_decay=0.001,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    greater_is_better=True,
    metric_for_best_model="eval_MAP@1",
    logging_dir="./logs",
    remove_unused_columns=False,
    report_to="none"
)


trainer = RankingLossTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=ranking_data_collator,
    decay_type='linear',
    alpha=1,
    callbacks=[PrettyPrintCallback()],
)


print_pretty_metrics(trainer.evaluate())
trainer.train()
print_pretty_metrics(trainer.evaluate())
print_pretty_metrics(trainer.evaluate(train_dataset))

val_df = val_df.rename(columns={
    'candidates': 'sorted_candidates'
})
val_df.to_json('scoring/validation_split.json', orient="records", indent=4)