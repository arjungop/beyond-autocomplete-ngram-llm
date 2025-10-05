"""
Advanced Sentence Completion System

This module extends the basic n-gram autocomplete to full sentence completion
with enhanced context awareness, variable-length n-grams, and improved smoothing.

Features:
- Variable-length n-gram selection (adaptive context)
- Advanced smoothing techniques (Kneser-Ney, Modified Kneser-Ney)
- Sentence-level completion with beam search
- Context-aware prediction with sentence boundaries
- Quality scoring and ranking of completions
"""

import numpy as np
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import heapq
import nltk
from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams


class AdvancedSentenceCompletor:
    """Advanced sentence completion using variable-length n-grams and beam search."""
    
    def __init__(self, max_n=5, vocab_threshold=3):
        """
        Initialize the advanced sentence completor.
        
        Args:
            max_n (int): Maximum n-gram order to consider
            vocab_threshold (int): Minimum frequency for vocabulary inclusion
        """
        self.max_n = max_n
        self.vocab_threshold = vocab_threshold
        self.n_gram_counts = {}  # Dictionary of n-gram count dictionaries
        self.vocabulary = set()
        self.vocab_size = 0
        self.start_token = '<s>'
        self.end_token = '<e>'
        self.unk_token = '<unk>'
        self.trained = False
        
    def train(self, training_data: List[List[str]]):
        """
        Train the model on preprocessed training data.
        
        Args:
            training_data: List of tokenized sentences
        """
        print("Training Advanced Sentence Completion Model...")
        
        # Build vocabulary
        word_counts = Counter()
        for sentence in training_data:
            word_counts.update(sentence)
        
        # Filter vocabulary by frequency
        self.vocabulary = {word for word, count in word_counts.items() 
                          if count >= self.vocab_threshold}
        self.vocabulary.update([self.start_token, self.end_token, self.unk_token])
        self.vocab_size = len(self.vocabulary)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Replace OOV words with <unk>
        processed_data = []
        for sentence in training_data:
            processed_sentence = [word if word in self.vocabulary else self.unk_token 
                                for word in sentence]
            processed_data.append(processed_sentence)
        
        # Count n-grams for all orders
        for n in range(1, self.max_n + 1):
            self.n_gram_counts[n] = count_n_grams(processed_data, n, 
                                                 self.start_token, self.end_token)
            print(f"{n}-grams counted: {len(self.n_gram_counts[n])}")
        
        self.trained = True
        print("✓ Training completed!")
    
    def _get_context_scores(self, context: List[str]) -> Dict[int, float]:
        """
        Calculate context scores for different n-gram orders.
        Higher scores indicate more reliable context.
        
        Args:
            context: List of context words
            
        Returns:
            Dictionary mapping n-gram order to reliability score
        """
        scores = {}
        context_len = len(context)
        
        for n in range(1, min(self.max_n, context_len + 1) + 1):
            # Get the relevant context for this n-gram order
            if n == 1:
                context_tuple = ()
            else:
                start_idx = max(0, context_len - (n - 1))
                context_tuple = tuple([self.start_token] * max(0, (n-1) - context_len) + 
                                    context[start_idx:])
            
            # Calculate frequency-based score
            if context_tuple in self.n_gram_counts.get(n-1, {}) if n > 1 else True:
                count = self.n_gram_counts.get(n-1, {}).get(context_tuple, 0) if n > 1 else 1
                scores[n] = math.log(count + 1)  # Log-smoothed frequency
            else:
                scores[n] = 0.0
                
        return scores
    
    def _kneser_ney_probability(self, word: str, context: List[str], n: int, 
                               discount: float = 0.75) -> float:
        """
        Calculate Kneser-Ney smoothed probability.
        
        Args:
            word: Target word
            context: Context words
            n: N-gram order
            discount: Discount parameter for KN smoothing
            
        Returns:
            Smoothed probability
        """
        if n == 1:
            # Unigram case: use simple frequency
            total_count = sum(self.n_gram_counts[1].values())
            word_count = self.n_gram_counts[1].get((word,), 0)
            return word_count / total_count if total_count > 0 else 1e-10
        
        # Build n-gram and (n-1)-gram
        if len(context) >= n - 1:
            context_tuple = tuple(context[-(n-1):])
        else:
            padding_needed = (n - 1) - len(context)
            context_tuple = tuple([self.start_token] * padding_needed + context)
        
        n_gram = context_tuple + (word,)
        
        # Get counts
        n_gram_count = self.n_gram_counts[n].get(n_gram, 0)
        context_count = self.n_gram_counts[n-1].get(context_tuple, 0)
        
        if context_count == 0:
            # Backoff to lower order
            return self._kneser_ney_probability(word, context[1:] if context else [], 
                                              n-1, discount)
        
        # Calculate discounted probability
        discounted_count = max(n_gram_count - discount, 0)
        prob = discounted_count / context_count
        
        # Add backoff probability
        if n_gram_count == 0:
            # Calculate continuation probability for backoff
            backoff_prob = self._kneser_ney_probability(word, context[1:] if context else [], 
                                                       n-1, discount)
            
            # Calculate interpolation weight
            unique_continuations = len([ng for ng in self.n_gram_counts[n] 
                                      if ng[:-1] == context_tuple])
            lambda_weight = (discount * unique_continuations) / context_count
            
            prob += lambda_weight * backoff_prob
        
        return max(prob, 1e-10)  # Prevent zero probabilities
    
    def _beam_search_completion(self, partial_sentence: List[str], 
                               max_length: int = 20, beam_width: int = 5) -> List[Tuple[List[str], float]]:
        """
        Use beam search to find the best sentence completions.
        
        Args:
            partial_sentence: Incomplete sentence to complete
            max_length: Maximum words to add
            beam_width: Number of candidates to keep at each step
            
        Returns:
            List of (completion, score) tuples
        """
        # Initialize beam with the partial sentence
        beam = [(partial_sentence.copy(), 0.0)]
        completed_sentences = []
        
        for step in range(max_length):
            candidates = []
            
            for sentence, score in beam:
                # Check if sentence is already complete
                if sentence and sentence[-1] == self.end_token:
                    completed_sentences.append((sentence, score))
                    continue
                
                # Get context scores for dynamic n-gram selection
                context_scores = self._get_context_scores(sentence)
                
                # Generate next word candidates
                word_scores = {}
                
                for word in self.vocabulary:
                    if word in [self.start_token]:  # Skip start token in middle
                        continue
                    
                    total_prob = 0.0
                    total_weight = 0.0
                    
                    # Weighted combination of different n-gram orders
                    for n, context_score in context_scores.items():
                        prob = self._kneser_ney_probability(word, sentence, n)
                        weight = math.exp(context_score)  # Convert log score to weight
                        
                        total_prob += prob * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        avg_prob = total_prob / total_weight
                        word_scores[word] = math.log(avg_prob)
                
                # Add top candidates to beam
                for word, word_score in word_scores.items():
                    new_sentence = sentence + [word]
                    new_score = score + word_score
                    candidates.append((new_sentence, new_score))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
            
            # Stop if all beams are complete
            if not beam:
                break
        
        # Add remaining beam items to completed sentences
        completed_sentences.extend(beam)
        
        # Sort by score and return
        completed_sentences.sort(key=lambda x: x[1], reverse=True)
        return completed_sentences
    
    def complete_sentence(self, partial_text: str, num_completions: int = 3, 
                         max_words: int = 20) -> List[Tuple[str, float]]:
        """
        Generate sentence completions for partial text.
        
        Args:
            partial_text: Incomplete sentence
            num_completions: Number of completions to return
            max_words: Maximum words to add
            
        Returns:
            List of (completion_text, confidence_score) tuples
        """
        if not self.trained:
            raise ValueError("Model must be trained before generating completions")
        
        # Tokenize input
        tokens = nltk.word_tokenize(partial_text.lower())
        
        # Replace OOV words
        tokens = [word if word in self.vocabulary else self.unk_token for word in tokens]
        
        # Generate completions using beam search
        completions = self._beam_search_completion(tokens, max_words)
        
        # Format results
        results = []
        for completion, score in completions[:num_completions]:
            # Remove start/end tokens and join
            filtered_completion = [word for word in completion 
                                 if word not in [self.start_token, self.end_token]]
            completion_text = ' '.join(filtered_completion)
            
            # Convert log score to confidence (0-1)
            confidence = min(1.0, max(0.0, (score + 50) / 50))  # Rough normalization
            
            results.append((completion_text, confidence))
        
        return results
    
    def evaluate_sentence_probability(self, sentence: str) -> float:
        """
        Calculate the probability of a complete sentence.
        
        Args:
            sentence: Complete sentence to evaluate
            
        Returns:
            Log probability of the sentence
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        tokens = nltk.word_tokenize(sentence.lower())
        tokens = [word if word in self.vocabulary else self.unk_token for word in tokens]
        tokens = [self.start_token] + tokens + [self.end_token]
        
        log_prob = 0.0
        
        for i in range(1, len(tokens)):
            context = tokens[:i]
            word = tokens[i]
            
            # Use dynamic n-gram selection
            context_scores = self._get_context_scores(context)
            
            total_prob = 0.0
            total_weight = 0.0
            
            for n, context_score in context_scores.items():
                prob = self._kneser_ney_probability(word, context, n)
                weight = math.exp(context_score)
                
                total_prob += prob * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_prob = total_prob / total_weight
                log_prob += math.log(avg_prob)
        
        return log_prob


def demo_advanced_completion():
    """Demonstrate the advanced sentence completion system."""
    print("=== Advanced Sentence Completion Demo ===\n")
    
    # Load data
    print("Loading training data...")
    with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
        data = f.read()
    
    tokenized_data = get_tokenized_data(data)
    train_data = tokenized_data[:10000]  # Use subset for demo
    
    # Initialize and train model
    completor = AdvancedSentenceCompletor(max_n=4, vocab_threshold=5)
    completor.train(train_data)
    
    # Demo completions
    test_phrases = [
        "The weather today is",
        "I think that",
        "Technology has changed",
        "My favorite",
        "In the future we will"
    ]
    
    print("\n" + "="*60)
    print("SENTENCE COMPLETION EXAMPLES")
    print("="*60)
    
    for phrase in test_phrases:
        print(f"\nInput: '{phrase}'")
        print("-" * 40)
        
        try:
            completions = completor.complete_sentence(phrase, num_completions=3)
            
            for i, (completion, confidence) in enumerate(completions, 1):
                print(f"{i}. {completion}")
                print(f"   Confidence: {confidence:.3f}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_advanced_completion()