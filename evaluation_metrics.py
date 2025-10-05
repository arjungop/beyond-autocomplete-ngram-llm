"""
Evaluation Metrics for N-gram Language Model
============================================

This module provides comprehensive evaluation metrics for autocomplete systems
and text generation models, including BLEU scores, accuracy metrics, and
semantic coherence measures.

Author: Research Team
Date: October 2025
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from data_preprocessing import tokenize_sentences


class BLEUScore:
    """
    BLEU (Bilingual Evaluation Understudy) Score Calculator
    
    BLEU is a metric for evaluating the quality of machine-generated text
    by comparing it to reference text(s). It measures n-gram precision
    with a brevity penalty.
    """
    
    def __init__(self, max_n: int = 4):
        """
        Initialize BLEU score calculator.
        
        Args:
            max_n: Maximum n-gram order to consider (default: 4)
        """
        self.max_n = max_n
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return Counter()
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def _modified_precision(self, candidate: List[str], references: List[List[str]], n: int) -> float:
        """
        Calculate modified n-gram precision.
        
        Args:
            candidate: Generated text tokens
            references: List of reference text token lists
            n: N-gram order
            
        Returns:
            Modified precision score
        """
        candidate_ngrams = self._get_ngrams(candidate, n)
        
        if not candidate_ngrams:
            return 0.0
        
        # Get maximum counts from all references
        max_ref_counts = Counter()
        for reference in references:
            ref_ngrams = self._get_ngrams(reference, n)
            for ngram in candidate_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Calculate clipped counts
        clipped_counts = {
            ngram: min(count, max_ref_counts[ngram])
            for ngram, count in candidate_ngrams.items()
        }
        
        return sum(clipped_counts.values()) / sum(candidate_ngrams.values())
    
    def _brevity_penalty(self, candidate_length: int, reference_lengths: List[int]) -> float:
        """
        Calculate brevity penalty to penalize short translations.
        
        Args:
            candidate_length: Length of candidate translation
            reference_lengths: Lengths of reference translations
            
        Returns:
            Brevity penalty factor
        """
        # Find closest reference length
        closest_ref_length = min(reference_lengths, 
                               key=lambda ref_len: abs(ref_len - candidate_length))
        
        if candidate_length > closest_ref_length:
            return 1.0
        elif candidate_length == 0:
            return 0.0
        else:
            return math.exp(1 - closest_ref_length / candidate_length)
    
    def calculate_sentence_bleu(self, candidate: str, references: List[str], 
                              weights: Optional[List[float]] = None) -> float:
        """
        Calculate BLEU score for a single sentence.
        
        Args:
            candidate: Generated sentence
            references: List of reference sentences
            weights: Weights for different n-gram orders (default: uniform)
            
        Returns:
            BLEU score between 0 and 1
        """
        if weights is None:
            weights = [1.0 / self.max_n] * self.max_n
        
        # Tokenize
        candidate_tokens = candidate.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]
        
        if not candidate_tokens:
            return 0.0
        
        # Calculate precision for each n-gram order
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._modified_precision(candidate_tokens, reference_tokens, n)
            if precision == 0:
                return 0.0  # If any precision is 0, BLEU is 0
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        log_precisions = [math.log(p) * w for p, w in zip(precisions, weights)]
        geometric_mean = math.exp(sum(log_precisions))
        
        # Apply brevity penalty
        candidate_length = len(candidate_tokens)
        reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens]
        bp = self._brevity_penalty(candidate_length, reference_lengths)
        
        return bp * geometric_mean
    
    def calculate_corpus_bleu(self, candidates: List[str], references_list: List[List[str]], 
                            weights: Optional[List[float]] = None) -> float:
        """
        Calculate BLEU score for a corpus of sentences.
        
        Args:
            candidates: List of generated sentences
            references_list: List of reference sentence lists for each candidate
            weights: Weights for different n-gram orders
            
        Returns:
            Corpus-level BLEU score
        """
        if weights is None:
            weights = [1.0 / self.max_n] * self.max_n
        
        # Aggregate counts across all sentences
        total_candidate_length = 0
        total_reference_length = 0
        total_clipped_counts = [Counter() for _ in range(self.max_n)]
        total_candidate_counts = [Counter() for _ in range(self.max_n)]
        
        for candidate, references in zip(candidates, references_list):
            candidate_tokens = candidate.lower().split()
            reference_tokens = [ref.lower().split() for ref in references]
            
            total_candidate_length += len(candidate_tokens)
            
            # Find closest reference length for this sentence
            closest_ref_length = min([len(ref_tokens) for ref_tokens in reference_tokens],
                                   key=lambda ref_len: abs(ref_len - len(candidate_tokens)))
            total_reference_length += closest_ref_length
            
            # Aggregate n-gram counts
            for n in range(1, self.max_n + 1):
                candidate_ngrams = self._get_ngrams(candidate_tokens, n)
                total_candidate_counts[n-1].update(candidate_ngrams)
                
                # Get maximum reference counts
                max_ref_counts = Counter()
                for ref_tokens in reference_tokens:
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    for ngram in candidate_ngrams:
                        max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
                
                # Update clipped counts
                for ngram, count in candidate_ngrams.items():
                    total_clipped_counts[n-1][ngram] += min(count, max_ref_counts[ngram])
        
        # Calculate precision for each n-gram order
        precisions = []
        for n in range(self.max_n):
            if sum(total_candidate_counts[n].values()) == 0:
                return 0.0
            
            precision = sum(total_clipped_counts[n].values()) / sum(total_candidate_counts[n].values())
            if precision == 0:
                return 0.0
            precisions.append(precision)
        
        # Calculate geometric mean
        log_precisions = [math.log(p) * w for p, w in zip(precisions, weights)]
        geometric_mean = math.exp(sum(log_precisions))
        
        # Apply brevity penalty
        if total_candidate_length > total_reference_length:
            bp = 1.0
        elif total_candidate_length == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - total_reference_length / total_candidate_length)
        
        return bp * geometric_mean


class AutocompleteAccuracy:
    """
    Accuracy metrics specifically designed for autocomplete systems.
    """
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self.total_predictions = 0
        self.correct_predictions = 0
        self.top_k_correct = defaultdict(int)
        self.prediction_ranks = []
        self.confidence_scores = []
        self.actual_words = []
    
    def exact_match_accuracy(self, predicted: str, actual: str) -> bool:
        """
        Check if prediction exactly matches the actual word.
        
        Args:
            predicted: Predicted word
            actual: Actual/target word
            
        Returns:
            True if exact match, False otherwise
        """
        return predicted.lower().strip() == actual.lower().strip()
    
    def top_k_accuracy(self, predictions: List[str], actual: str, k: int = 5) -> bool:
        """
        Check if actual word appears in top-k predictions.
        
        Args:
            predictions: List of predicted words (ranked)
            actual: Actual/target word
            k: Number of top predictions to consider
            
        Returns:
            True if actual word is in top-k, False otherwise
        """
        top_k_preds = [pred.lower().strip() for pred in predictions[:k]]
        return actual.lower().strip() in top_k_preds
    
    def mean_reciprocal_rank(self, predictions: List[str], actual: str) -> float:
        """
        Calculate reciprocal rank of the actual word in predictions.
        
        Args:
            predictions: List of predicted words (ranked)
            actual: Actual/target word
            
        Returns:
            Reciprocal rank (1/rank if found, 0 if not found)
        """
        actual_lower = actual.lower().strip()
        for i, pred in enumerate(predictions):
            if pred.lower().strip() == actual_lower:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_prediction(self, predictions: List[str], actual: str, 
                          confidence: Optional[float] = None):
        """
        Evaluate a single prediction and update metrics.
        
        Args:
            predictions: List of predicted words (ranked by confidence)
            actual: Actual/target word
            confidence: Confidence score for top prediction
        """
        if not predictions:
            return
        
        self.total_predictions += 1
        self.actual_words.append(actual)
        
        if confidence is not None:
            self.confidence_scores.append(confidence)
        
        # Exact match accuracy (top-1)
        if self.exact_match_accuracy(predictions[0], actual):
            self.correct_predictions += 1
        
        # Top-k accuracy for various k values
        for k in [1, 3, 5, 10]:
            if self.top_k_accuracy(predictions, actual, k):
                self.top_k_correct[k] += 1
        
        # Mean reciprocal rank
        mrr = self.mean_reciprocal_rank(predictions, actual)
        self.prediction_ranks.append(mrr)
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """
        Calculate and return all accuracy metrics.
        
        Returns:
            Dictionary with various accuracy metrics
        """
        if self.total_predictions == 0:
            return {
                'exact_match_accuracy': 0.0,
                'top_1_accuracy': 0.0,
                'top_3_accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'top_10_accuracy': 0.0,
                'mean_reciprocal_rank': 0.0,
                'total_predictions': 0
            }
        
        metrics = {
            'exact_match_accuracy': self.correct_predictions / self.total_predictions,
            'top_1_accuracy': self.top_k_correct[1] / self.total_predictions,
            'top_3_accuracy': self.top_k_correct[3] / self.total_predictions,
            'top_5_accuracy': self.top_k_correct[5] / self.total_predictions,
            'top_10_accuracy': self.top_k_correct[10] / self.total_predictions,
            'mean_reciprocal_rank': np.mean(self.prediction_ranks) if self.prediction_ranks else 0.0,
            'total_predictions': self.total_predictions
        }
        
        if self.confidence_scores:
            metrics['mean_confidence'] = np.mean(self.confidence_scores)
            metrics['confidence_std'] = np.std(self.confidence_scores)
        
        return metrics


class TextGenerationQuality:
    """
    Quality metrics for generated text including coherence and fluency measures.
    """
    
    @staticmethod
    def calculate_perplexity(text: str, language_model) -> float:
        """
        Calculate perplexity of generated text using the language model.
        
        Args:
            text: Generated text to evaluate
            language_model: Trained language model instance
            
        Returns:
            Perplexity score (lower is better)
        """
        return language_model.calculate_perplexity(text)
    
    @staticmethod
    def repetition_penalty(text: str, n: int = 3) -> float:
        """
        Calculate repetition penalty based on repeated n-grams.
        
        Args:
            text: Text to analyze
            n: N-gram size for repetition detection
            
        Returns:
            Repetition score (0-1, where 1 is no repetition)
        """
        tokens = text.lower().split()
        if len(tokens) < n:
            return 1.0
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 1.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return unique_ngrams / total_ngrams
    
    @staticmethod
    def length_consistency(generated_texts: List[str], target_length: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze length consistency of generated texts.
        
        Args:
            generated_texts: List of generated text samples
            target_length: Expected length (if any)
            
        Returns:
            Length statistics
        """
        lengths = [len(text.split()) for text in generated_texts]
        
        stats = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
        
        if target_length:
            deviations = [abs(length - target_length) for length in lengths]
            stats['mean_deviation'] = np.mean(deviations)
            stats['length_accuracy'] = sum(1 for d in deviations if d <= 2) / len(deviations)
        
        return stats


def evaluate_autocomplete_system(language_model, test_data: List[Tuple[str, str]], 
                                num_suggestions: int = 5) -> Dict[str, float]:
    """
    Comprehensive evaluation of autocomplete system.
    
    Args:
        language_model: Trained language model
        test_data: List of (context, target_word) tuples
        num_suggestions: Number of suggestions to generate
        
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy_evaluator = AutocompleteAccuracy()
    
    for context, target_word in test_data:
        # Get predictions from the model
        words = context.split()
        if len(words) >= 2:
            # Get suggestions using the model's suggestion method
            try:
                suggestions = language_model.get_user_input_suggestions(
                    words[-2:], num_suggestions
                )
                predictions = [word for word, _ in suggestions]
                confidence = suggestions[0][1] if suggestions else 0.0
                
                accuracy_evaluator.evaluate_prediction(predictions, target_word, confidence)
            except Exception as e:
                print(f"Error evaluating context '{context}': {e}")
                continue
    
    return accuracy_evaluator.get_accuracy_metrics()


def evaluate_text_generation(generated_texts: List[str], reference_texts: List[str], 
                           language_model=None) -> Dict[str, float]:
    """
    Comprehensive evaluation of text generation quality.
    
    Args:
        generated_texts: List of generated text samples
        reference_texts: List of reference/target texts
        language_model: Language model for perplexity calculation
        
    Returns:
        Dictionary with evaluation metrics
    """
    bleu_calculator = BLEUScore()
    metrics = {}
    
    # BLEU scores
    if len(generated_texts) == len(reference_texts):
        bleu_scores = []
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            bleu_score = bleu_calculator.calculate_sentence_bleu(gen_text, [ref_text])
            bleu_scores.append(bleu_score)
        
        metrics['mean_bleu'] = np.mean(bleu_scores)
        metrics['bleu_std'] = np.std(bleu_scores)
        
        # Corpus BLEU
        references_list = [[ref] for ref in reference_texts]
        metrics['corpus_bleu'] = bleu_calculator.calculate_corpus_bleu(
            generated_texts, references_list
        )
    
    # Quality metrics
    repetition_scores = [TextGenerationQuality.repetition_penalty(text) 
                        for text in generated_texts]
    metrics['mean_repetition_penalty'] = np.mean(repetition_scores)
    
    # Length consistency
    length_stats = TextGenerationQuality.length_consistency(generated_texts)
    metrics.update(length_stats)
    
    # Perplexity (if language model provided)
    if language_model:
        perplexities = []
        for text in generated_texts:
            try:
                perplexity = TextGenerationQuality.calculate_perplexity(text, language_model)
                perplexities.append(perplexity)
            except:
                continue
        
        if perplexities:
            metrics['mean_perplexity'] = np.mean(perplexities)
            metrics['perplexity_std'] = np.std(perplexities)
    
    return metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Testing BLEU Score Calculator...")
    
    bleu = BLEUScore()
    
    # Test sentence BLEU
    candidate = "the cat sat on the mat"
    references = ["a cat sat on the mat", "the cat was sitting on the mat"]
    
    score = bleu.calculate_sentence_bleu(candidate, references)
    print(f"Sentence BLEU score: {score:.4f}")
    
    # Test corpus BLEU
    candidates = [
        "the cat sat on the mat",
        "it is raining today"
    ]
    references_list = [
        ["a cat sat on the mat", "the cat was sitting on the mat"],
        ["it is raining outside", "today it is raining"]
    ]
    
    corpus_score = bleu.calculate_corpus_bleu(candidates, references_list)
    print(f"Corpus BLEU score: {corpus_score:.4f}")
    
    print("\nTesting Autocomplete Accuracy...")
    
    accuracy = AutocompleteAccuracy()
    
    # Simulate some predictions
    test_cases = [
        (["cat", "dog", "bird"], "cat"),  # Correct prediction
        (["house", "home", "building"], "car"),  # Incorrect prediction
        (["run", "walk", "jog"], "run"),  # Correct prediction
        (["apple", "banana", "orange"], "banana"),  # Top-3 correct
    ]
    
    for predictions, actual in test_cases:
        accuracy.evaluate_prediction(predictions, actual, 0.8)
    
    metrics = accuracy.get_accuracy_metrics()
    print("Accuracy Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")