import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset

class HindiCharacterEmbedding:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.char_to_idx = defaultdict(lambda: len(self.char_to_idx))
        self.embeddings = None
        
    def fit(self, hindi_texts):
        # Create vocabulary of unique characters
        for text in tqdm(hindi_texts, desc="Building vocabulary"):
            if isinstance(text, str):  # Make sure we only process strings
                for char in text:
                    _ = self.char_to_idx[char]
        
        # Initialize embedding layer
        vocab_size = len(self.char_to_idx)
        print(f"Vocabulary size: {vocab_size} characters")
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        
    def get_character_embedding(self, char):
        if self.embeddings is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if char not in self.char_to_idx:
            raise ValueError(f"Character '{char}' not in vocabulary")
            
        char_idx = self.char_to_idx[char]
        with torch.no_grad():
            return self.embeddings(torch.tensor([char_idx])).numpy()

    def get_text_embedding(self, text):
        if self.embeddings is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Check if all characters are in vocabulary
        unknown_chars = [char for char in text if char not in self.char_to_idx]
        if unknown_chars:
            raise ValueError(f"Characters {unknown_chars} not in vocabulary")
            
        char_indices = [self.char_to_idx[char] for char in text]
        with torch.no_grad():
            char_embeddings = self.embeddings(torch.tensor(char_indices))
            return torch.mean(char_embeddings, dim=0).numpy()

def get_hindi_dataset():
    """Load Hindi dataset from Hugging Face"""
    try:
        # Load the IIT Bombay English-Hindi dataset
        dataset = load_dataset("cfilt/iitb-english-hindi")
        
        # Extract Hindi texts from the training split
        hindi_texts = dataset['train']['translation']
        hindi_texts = [item['hi'] for item in hindi_texts]
        
        print(f"Successfully loaded {len(hindi_texts)} Hindi texts")
        return hindi_texts
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # Fallback to sample data if loading fails
        return [
            "नमस्ते भारत",
            "हिंदी भाषा बहुत सुंदर है",
            "मैं हिंदी सीख रहा हूं",
            "भारत एक महान देश है",
            "आज का दिन बहुत अच्छा है",
            "यह एक बहुत अच्छा उदाहरण है",
            "मैं प्रोग्रामिंग सीख रहा हूं",
            "पाइथन एक अच्छी प्रोग्रामिंग भाषा है",
            "मशीन लर्निंग बहुत रोचक विषय है",
            "डेटा साइंस का भविष्य उज्जवल है"
        ]

def main():
    # Load the Hindi dataset
    hindi_texts = get_hindi_dataset()
    
    # Filter out empty lines
    hindi_texts = [text for text in hindi_texts if text.strip()]
    print(f"Total number of texts after filtering: {len(hindi_texts)}")
    
    # Take a subset for demonstration if dataset is too large
    max_texts = 1000
    if len(hindi_texts) > max_texts:
        print(f"Using first {max_texts} texts for demonstration")
        hindi_texts = hindi_texts[:max_texts]
    
    # Initialize and train the embedding model
    embedding_model = HindiCharacterEmbedding(embedding_dim=100)
    embedding_model.fit(hindi_texts)
    
    # Example usage
    sample_chars = ["न", "म", "स्", "त", "े"]
    print("\nCharacter embeddings examples:")
    for char in sample_chars:
        try:
            embedding = embedding_model.get_character_embedding(char)
            print(f"Character: {char}, Embedding shape: {embedding.shape}")
        except ValueError as e:
            print(f"Error processing character '{char}': {str(e)}")
    
    sample_texts = [
        "नमस्ते",
        "भारत",
        "हिंदी"
    ]
    
    print("\nText embeddings examples:")
    for text in sample_texts:
        try:
            embedding = embedding_model.get_text_embedding(text)
            print(f"Text: {text}, Embedding shape: {embedding.shape}")
        except ValueError as e:
            print(f"Error processing text '{text}': {str(e)}")

if __name__ == "__main__":
    main()