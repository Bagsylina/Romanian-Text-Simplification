import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RankedMaskDataset(Dataset):
    """Dataset for masked language modeling with ranked replacement words"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """
        Load data from JSON file. Expected format:
        [
            {
                "masked_sentence": "The cat [MASK] on the mat.",
                "ranked_words": ["sat", "slept", "lay", "stood"]
            },
            ...
        ]
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        masked_sentence = sentence.replace(item['word'], '[MASK]', 1)
        ranked_words = item['sorted_candidates']
        
        # Tokenize the masked sentence
        encoding = self.tokenizer(
            masked_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find the mask position
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (encoding['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_positions) == 0:
            raise ValueError(f"No mask token found in sentence: {masked_sentence}")
        
        mask_position = mask_positions[0].item()
        
        # Tokenize ranked words and get their token IDs
        ranked_token_ids = []
        for word in ranked_words:
            # Use add_special_tokens=False to get just the word tokens
            word_tokens = self.tokenizer(word, add_special_tokens=False)['input_ids']
            if len(word_tokens) == 1:  # Only single-token words for simplicity
                ranked_token_ids.append(word_tokens[0])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'mask_position': mask_position,
            'ranked_token_ids': ranked_token_ids,
            'original_sentence': masked_sentence
        }

class RankedBERTModel(nn.Module):
    """BERT model with ranking loss for masked language modeling"""
    
    def __init__(self, model_name: str, vocab_size: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask, mask_position=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        if mask_position is not None:
            # Get the hidden state at the mask position
            batch_size = input_ids.size(0)
            mask_output = sequence_output[torch.arange(batch_size), mask_position]
        else:
            # Use [CLS] token for classification
            mask_output = sequence_output[:, 0]
            
        mask_output = self.dropout(mask_output)
        logits = self.classifier(mask_output)
        
        return logits
    
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model state dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # Save the config
        config = {
            'model_type': 'RankedBERTModel',
            'vocab_size': self.classifier.out_features,
            'hidden_size': self.bert.config.hidden_size,
            'base_model': self.bert.config.name_or_path if hasattr(self.bert.config, 'name_or_path') else None
        }
        
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path):
        """Load the model from a directory"""
        import os
        
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        base_model = config.get('base_model', 'dumitrescustefan/bert-base-romanian-cased-v1')
        model = cls(base_model, config['vocab_size'])
        
        # Load state dict
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

class RankingLoss(nn.Module):
    """Custom loss function for ranked predictions"""
    
    def __init__(self, margin: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, logits, ranked_token_ids_batch):
        """
        Compute ranking loss where higher-ranked words should have higher scores
        """
        total_loss = 0
        batch_size = len(ranked_token_ids_batch)
        
        for i in range(batch_size):
            ranked_token_ids = ranked_token_ids_batch[i]
            if len(ranked_token_ids) < 2:
                continue
                
            batch_logits = logits[i]
            
            # Get scores for ranked tokens
            ranked_scores = batch_logits[ranked_token_ids]
            
            # Pairwise ranking loss
            loss = 0
            pairs = 0
            
            for j in range(len(ranked_scores)):
                for k in range(j + 1, len(ranked_scores)):
                    # Higher ranked (lower index) should have higher score
                    higher_ranked_score = ranked_scores[j]
                    lower_ranked_score = ranked_scores[k]
                    
                    # Margin ranking loss
                    loss += torch.clamp(
                        self.margin - (higher_ranked_score - lower_ranked_score), 
                        min=0
                    )
                    pairs += 1
            
            if pairs > 0:
                total_loss += loss / pairs
                
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)

def collate_fn(batch):
    """Custom collate function to handle variable-length ranked lists"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    mask_positions = torch.tensor([item['mask_position'] for item in batch])
    ranked_token_ids = [item['ranked_token_ids'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'mask_position': mask_positions,
        'ranked_token_ids': ranked_token_ids
    }

def train_model(
    model_name: str = 'dumitrescustefan/bert-base-romanian-cased-v1',
    train_data_path: str = 'train_data.json',
    val_data_path: str = None,
    output_dir: str = './bert_ranked_model',
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    warmup_steps: int = 500,
    save_steps: int = 1000
):
    """Train the BERT model with ranking loss"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    
    model = RankedBERTModel(model_name, vocab_size)
    model.to(device)
    
    # Load datasets
    train_dataset = RankedMaskDataset(train_data_path, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_data_path:
        val_dataset = RankedMaskDataset(val_data_path, tokenizer, max_length)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = RankingLoss(margin=1.0, temperature=1.0)
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_positions = batch['mask_position'].to(device)
            ranked_token_ids = batch['ranked_token_ids']
            
            # Forward pass
            logits = model(input_ids, attention_mask, mask_positions)
            
            # Compute loss
            loss = criterion(logits, ranked_token_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (progress_bar.n + 1):.4f}'
            })
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_path = f"{output_dir}/checkpoint-{global_step}"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }, checkpoint_path)
                logger.info(f"Saved checkpoint at step {global_step}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    mask_positions = batch['mask_position'].to(device)
                    ranked_token_ids = batch['ranked_token_ids']
                    
                    logits = model(input_ids, attention_mask, mask_positions)
                    loss = criterion(logits, ranked_token_ids)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            model.train()
        
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader):.4f}")
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def predict_ranked_words(model_path: str, sentence: str, top_k: int = 10):
    """Use trained model to predict ranked words for a masked sentence"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RankedBERTModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(sentence, return_tensors='pt').to(device)
    
    # Find mask position
    mask_positions = (encoding['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_positions) == 0:
        raise ValueError("No mask token found in sentence")
    
    mask_position = mask_positions[0]
    
    # Get predictions
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'], mask_position.unsqueeze(0))
        probabilities = torch.softmax(logits[0], dim=-1)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        
        # Convert to words
        predicted_words = []
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
            word = tokenizer.decode(idx.item())
            predicted_words.append((word, prob.item(), i + 1))
    
    return predicted_words

# Example usage
if __name__ == "__main__":
    # Train the model
    """
    train_model(
        model_name='dumitrescustefan/bert-base-romanian-cased-v1',
        train_data_path='scoring/ranking_data.json',
        #val_data_path='val_data.json',  # Optional
        output_dir='./bert_ranked_romanian',
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=10
    )
    """
    
    # Example prediction
    """
    predicted = predict_ranked_words(
        './bert_ranked_romanian',
        "Capitolul al doilea investigheaz\u0103 aceast\u0103 leg\u0103tur\u0103 pu\u021bin mai am\u0103nun\u021bit, discut\u00e2nd [MASK] mental\u0103 a rezolv\u0103rii problemelor.",
        top_k=10
    )
    print("Predicted words:", predicted)

    predicted = predict_ranked_words(
        './bert_ranked_romanian',
        "Capitolul al doilea investigheaz\u0103 aceast\u0103 leg\u0103tur\u0103 pu\u021bin mai [MASK], discut\u00e2nd gimnastica mental\u0103 a rezolv\u0103rii problemelor.",
        top_k=10
    )
    print("Predicted words:", predicted)

    predicted = predict_ranked_words(
        './bert_ranked_romanian',
        "\u00cen prezent, un loc este [MASK] cu aproximativ 600 000 de aleg\u0103tori.",
        top_k=10
    )
    print("Predicted words:", predicted)
    """

    def unmasker_model(sentence):
        predicted = predict_ranked_words(
            './bert_ranked_romanian',
            sentence,
            top_k=10
        )

        suggestions = []

        for prediction in predicted:
            suggestions.append({"score": prediction[1], "token_str": prediction[0]})

        return suggestions
    
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from model import RoTextSimpModel

    with open('scoring/ranking_data.json', 'r') as f:
        test_sentences = json.load(f)

    def test_suggestions(unmasker):
        result = []

        for sentence_set in test_sentences:
            masked_sentence = sentence_set["sentence"].replace(sentence_set["word"], "[MASK]", 1)
            suggestions = unmasker(masked_sentence)

            result.append({"sentence": sentence_set["sentence"], 
                    "sentence_id": sentence_set["sentence_id"], 
                    "word": sentence_set["word"], 
                    "suggestions": suggestions})

        return result
    
    with open('scoring/trained_bertRo_suggestions_2.json', 'w') as f:
        unmasker = lambda sentence: predict_ranked_words('./bert_ranked_romanian', sentence, top_k=10)
        json.dump(test_suggestions(unmasker), f, indent=4)
