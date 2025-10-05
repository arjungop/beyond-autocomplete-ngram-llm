"""
N-gram Language Models for Text Prediction
==========================================
Implementation of multi-order n-gram language models with smoothing.
"""

import numpy as np
import math
from collections import defaultdict, Counter
from data_preprocessing import get_tokenized_data


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """Count n-grams with proper sentence boundary handling"""
    n_grams = {}
    
    for sentence in data:
        # Add start and end tokens
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)
        
        # Extract n-grams
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i + n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
                
    return n_grams


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """Calculate perplexity for a sentence"""
    sentence_start = ['<s>'] + sentence + ['<e>']
    sentence_length = len(sentence_start)
    
    log_probability = 0.0
    
    for i in range(1, sentence_length):
        n_gram = tuple(sentence_start[i-1:i+1])
        n_minus1_gram = tuple(sentence_start[i-1:i])
        
        n_plus1_gram_count = n_plus1_gram_counts.get(n_gram, 0)
        n_gram_count = n_gram_counts.get(n_minus1_gram, 0)
        
        probability = (n_plus1_gram_count + k) / (n_gram_count + k * vocabulary_size)
        log_probability += math.log(probability)
    
    perplexity = math.exp(-log_probability / sentence_length)
    return perplexity


def preprocess_data(train_data, test_data, count_threshold):
    """Preprocess training and test data"""
    word_counts = {}
    
    # Count words in training data
    for sentence in train_data:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Build vocabulary with count threshold
    vocabulary = ['<unk>']
    for word, count in word_counts.items():
        if count >= count_threshold:
            vocabulary.append(word)
    
    vocab_set = set(vocabulary)
    
    # Replace rare words with <unk>
    def replace_oov_words(data):
        processed = []
        for sentence in data:
            new_sentence = []
            for word in sentence:
                if word in vocab_set:
                    new_sentence.append(word)
                else:
                    new_sentence.append('<unk>')
            processed.append(new_sentence)
        return processed
    
    train_processed = replace_oov_words(train_data)
    test_processed = replace_oov_words(test_data)
    
    return train_processed, test_processed, vocabulary


def preprocess_data(train_data, test_data, count_threshold):
    """Preprocess training and test data with vocabulary filtering"""
    word_counts = {}
    
    # Count words in training data with better filtering
    for sentence in train_data:
        for word in sentence:
            if len(word) > 0 and not word.isspace():  # Filter empty/whitespace words
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Build vocabulary with adaptive threshold
    vocabulary = ['<unk>', '<s>', '<e>']
    
    # Use adaptive thresholding for better vocabulary coverage
    for word, count in word_counts.items():
        if count >= count_threshold:
            vocabulary.append(word)
        elif count >= max(1, count_threshold // 2) and len(word) > 2:
            vocabulary.append(word)  # Keep longer words with lower threshold
    
    vocab_set = set(vocabulary)
    
    # Enhanced OOV replacement with better handling
    def replace_oov_words_enhanced(data):
        processed = []
        for sentence in data:
            if not sentence:  # Skip empty sentences
                continue
            new_sentence = []
            for word in sentence:
                if word in vocab_set:
                    new_sentence.append(word)
                elif len(word) > 0:  # Only replace non-empty words
                    new_sentence.append('<unk>')
            if new_sentence:  # Only add non-empty sentences
                processed.append(new_sentence)
        return processed
    
    train_processed = replace_oov_words_enhanced(train_data)
    test_processed = replace_oov_words_enhanced(test_data)
    
    return train_processed, test_processed, vocabulary


class NGramModel:
    """Multi-order N-gram model for text prediction"""
    
    def __init__(self, max_n=4):
        self.max_n = max_n
        self.vocabulary = []
        self.n_gram_counts_list = []
        self.train_data = None
        
    def train(self, tokenized_sentences, count_threshold=1):  # Lower threshold for more coverage
        """Train the n-gram language model"""
        print(f"Training n-gram model with {len(tokenized_sentences)} sentences...")
        
        # Use 90% for training to get more data for higher-order n-grams
        train_size = int(len(tokenized_sentences) * 0.9)
        train_data = tokenized_sentences[:train_size]
        test_data = tokenized_sentences[train_size:]
        
        # For higher-order n-grams, use even lower count threshold
        lower_threshold = max(1, count_threshold // 2)
        
        # Preprocess training data
        train_processed, test_processed, vocabulary = preprocess_data(
            train_data, test_data, lower_threshold
        )
        
        self.train_data = train_processed
        self.test_data = test_processed
        self.vocabulary = vocabulary
        
        print(f"Vocabulary size: {len(vocabulary)}")
        
        # Count n-grams for all orders
        self.n_gram_counts_list = []
        for n in range(1, self.max_n + 1):
            n_gram_counts = count_n_grams(train_processed, n)
            self.n_gram_counts_list.append(n_gram_counts)
            print(f"{n}-gram: {len(n_gram_counts)} unique n-grams")
        
        print("Training completed!")
        
    def calculate_probability(self, n_gram, k=0.001):
        """Calculate probability using interpolated smoothing"""
        # Interpolation weights (higher for higher-order models)
        weights = [0.1, 0.2, 0.3, 0.4]  # weights for 1,2,3,4-gram
        
        interpolated_prob = 0.0
        
        for order in range(1, min(len(n_gram) + 1, self.max_n + 1)):
            if order > len(self.n_gram_counts_list):
                continue
                
            # Get the order-gram from the full n_gram
            current_gram = n_gram[-order:] if order <= len(n_gram) else n_gram
            context = current_gram[:-1] if len(current_gram) > 1 else ()
            
            # Get counts
            gram_counts = self.n_gram_counts_list[order - 1]
            context_counts = self.n_gram_counts_list[order - 2] if order > 1 else None
            
            gram_count = gram_counts.get(current_gram, 0)
            context_count = context_counts.get(context, 0) if context_counts and context else len(self.train_data)
            
            # Calculate probability with smoothing
            vocab_size = len(self.vocabulary) + 2
            if context_count > 0:
                prob = (gram_count + k) / (context_count + k * vocab_size)
            else:
                prob = 1.0 / vocab_size
            
            # Add weighted probability
            weight = weights[order - 1] if order <= len(weights) else 0.05
            interpolated_prob += weight * prob
        
        return interpolated_prob
        
    def calculate_perplexity(self, sentence, n_order, k=0.001):
        """Calculate perplexity for a sentence using n-gram model"""
        sentence_start = ['<s>'] * (n_order - 1) + sentence + ['<e>']
        sentence_length = len(sentence_start) - (n_order - 1)
        
        log_probability = 0.0
        
        for i in range(n_order - 1, len(sentence_start)):
            # Get the n-gram
            start_idx = max(0, i - n_order + 1)
            n_gram = tuple(sentence_start[start_idx:i+1])
            
            # Calculate probability using smoothing
            probability = self.calculate_probability(n_gram, k)
            
            if probability <= 0:
                probability = 1.0 / (len(self.vocabulary) + 2)  # Uniform fallback
            
            log_probability += math.log(probability)
        
        perplexity = math.exp(-log_probability / sentence_length) if sentence_length > 0 else float('inf')
        return perplexity
        """Calculate perplexity using backoff smoothing"""
        sentence_start = ['<s>'] * (n_order - 1) + sentence + ['<e>']
        sentence_length = len(sentence_start) - (n_order - 1)
        
        log_probability = 0.0
        
        for i in range(n_order - 1, len(sentence_start)):
            # Try higher-order n-gram first, then backoff
            probability = None
            
            for backoff_order in range(n_order, 0, -1):
                if backoff_order > len(self.n_gram_counts_list):
                    continue
                    
                start_idx = i - backoff_order + 1
                if start_idx < 0:
                    continue
                    
                n_gram = tuple(sentence_start[start_idx:i+1])
                context = tuple(sentence_start[start_idx:i])
                
                n_gram_counts = self.n_gram_counts_list[backoff_order - 1]
                context_counts = self.n_gram_counts_list[backoff_order - 2] if backoff_order > 1 else None
                
                n_gram_count = n_gram_counts.get(n_gram, 0)
                context_count = context_counts.get(context, 0) if context_counts else len(self.train_data)
                
                if context_count > 0:
                    vocabulary_size = len(self.vocabulary) + 2
                    probability = (n_gram_count + k) / (context_count + k * vocabulary_size)
                    break
            
            if probability is None or probability <= 0:
                probability = 1.0 / (len(self.vocabulary) + 2)  # Uniform fallback
            
            log_probability += math.log(probability)
        
        perplexity = math.exp(-log_probability / sentence_length) if sentence_length > 0 else float('inf')
        return perplexity
    def evaluate_perplexity(self, test_sentences=None, k=1.0):
        """Evaluate perplexity for each n-gram order using backoff smoothing"""
        if test_sentences is None:
            test_sentences = self.test_data[:100]  # Use first 100 test sentences
        
        results = {}
        
        # Evaluate each n-gram order with backoff
        for n in range(2, self.max_n + 1):  # Start from 2-gram
            perplexities = []
            
            for sentence in test_sentences:
                try:
                    perplexity = self.calculate_perplexity(sentence, n, k)
                    if perplexity < float('inf') and perplexity > 0:
                        perplexities.append(perplexity)
                except:
                    continue
            
            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                results[f'{n}-gram'] = avg_perplexity
            else:
                results[f'{n}-gram'] = float('inf')
        
        return results
    
    def generate_text(self, starting_sentence, max_length=10, n_order=2, k=1.0):
        """Generate text using n-gram model"""
        if n_order > self.max_n:
            n_order = self.max_n
        
        words = starting_sentence.split()
        generated_words = []
        
        for _ in range(max_length):
            if len(words) < n_order - 1:
                context = ['<s>'] * (n_order - 1 - len(words)) + words
            else:
                context = words[-(n_order-1):]
            
            # Get candidates
            candidates = []
            context_tuple = tuple(context)
            
            n_gram_counts = self.n_gram_counts_list[n_order - 1]
            
            for n_gram, count in n_gram_counts.items():
                if n_gram[:-1] == context_tuple:
                    candidates.append((n_gram[-1], count))
            
            if not candidates:
                break
                
            # Select next word (simple: pick most frequent)
            candidates.sort(key=lambda x: x[1], reverse=True)
            next_word = candidates[0][0]
            
            if next_word == '<e>':
                break
                
            generated_words.append(next_word)
            words.append(next_word)
            
        return ' '.join(generated_words)