"""
Comprehensive Evaluation Script for N-gram Language Model
========================================================

This script evaluates the autocomplete system and text generation capabilities
using various metrics including BLEU scores, accuracy metrics, and quality measures.

Usage:
    python evaluate_system.py
    
Author: Research Team
Date: October 2025
"""

import os
import json
from typing import List, Tuple, Dict
import random
from evaluation_metrics import (
    BLEUScore, AutocompleteAccuracy, TextGenerationQuality,
    evaluate_autocomplete_system, evaluate_text_generation
)
from language_model import LanguageModel
from data_preprocessing import preprocess_data
from markov_text_generator import MarkovTextGenerator


class SystemEvaluator:
    """
    Comprehensive evaluator for the autocomplete and text generation system.
    """
    
    def __init__(self, data_file: str = "data/en_US.twitter.txt"):
        """
        Initialize the evaluator with data and models.
        
        Args:
            data_file: Path to the training data file
        """
        self.data_file = data_file
        self.language_model = None
        self.markov_generator = None
        self.test_data = []
        
    def setup_models(self):
        """Initialize and train the language models."""
        print("Setting up language models...")
        
        # Initialize and train n-gram language model
        if os.path.exists(self.data_file):
            # Load and preprocess data similar to main.py
            from data_preprocessing import get_tokenized_data
            
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = f.read()
            
            # Split into train/test
            tokenized_data = get_tokenized_data(data)
            random.shuffle(tokenized_data)
            
            train_size = int(len(tokenized_data) * 0.8)
            train_data = tokenized_data[0:train_size]
            test_data = tokenized_data[train_size:]
            
            # Preprocess the data
            minimum_freq = 5
            train_data_processed, test_data_processed, vocabulary = preprocess_data(
                train_data, test_data, minimum_freq
            )
            
            self.language_model = LanguageModel()
            self.language_model.fit(train_data_processed, vocabulary)
            print(f"N-gram model trained on {len(train_data_processed)} sentences")
            
            # Initialize Markov text generator
            self.markov_generator = MarkovTextGenerator()
            self.markov_generator.train_from_file(self.data_file)
            print("Markov text generator trained")
        else:
            print(f"Warning: Data file {self.data_file} not found!")
    
    def create_test_data(self, num_samples: int = 100) -> List[Tuple[str, str]]:
        """
        Create test data for autocomplete evaluation.
        
        Args:
            num_samples: Number of test samples to create
            
        Returns:
            List of (context, target_word) tuples
        """
        print(f"Creating {num_samples} test samples...")
        
        if not os.path.exists(self.data_file):
            print("No data file found for creating test samples")
            return []
        
        test_samples = []
        
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            random.shuffle(lines)
            
            samples_created = 0
            for line in lines:
                if samples_created >= num_samples:
                    break
                
                words = line.strip().split()
                # Filter words with minimum length and ensure they're alphabetic
                words = [word.lower() for word in words if len(word) >= 3 and word.isalpha()]
                
                if len(words) >= 4:  # Need at least 3 words for context + 1 target
                    # Take random position for target word (not first or last)
                    target_idx = random.randint(2, len(words) - 1)
                    context = " ".join(words[:target_idx])
                    target_word = words[target_idx]
                    
                    test_samples.append((context, target_word))
                    samples_created += 1
        
        self.test_data = test_samples
        print(f"Created {len(test_samples)} test samples")
        return test_samples
    
    def evaluate_autocomplete(self) -> Dict[str, float]:
        """
        Evaluate the autocomplete system performance.
        
        Returns:
            Dictionary with autocomplete evaluation metrics
        """
        print("\n" + "="*50)
        print("EVALUATING AUTOCOMPLETE SYSTEM")
        print("="*50)
        
        if not self.language_model:
            print("Language model not initialized!")
            return {}
        
        if not self.test_data:
            self.create_test_data()
        
        # Use the evaluation function from evaluation_metrics
        metrics = evaluate_autocomplete_system(
            self.language_model, 
            self.test_data[:50],  # Use first 50 samples for faster evaluation
            num_suggestions=10
        )
        
        print("\nAutocomplete Accuracy Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric:25s}: {value:.4f}")
            else:
                print(f"{metric:25s}: {value}")
        
        return metrics
    
    def evaluate_text_generation(self, num_samples: int = 20) -> Dict[str, float]:
        """
        Evaluate text generation quality using BLEU and other metrics.
        
        Args:
            num_samples: Number of text samples to generate and evaluate
            
        Returns:
            Dictionary with text generation evaluation metrics
        """
        print("\n" + "="*50)
        print("EVALUATING TEXT GENERATION")
        print("="*50)
        
        if not self.markov_generator:
            print("Markov generator not initialized!")
            return {}
        
        # Generate text samples
        print(f"Generating {num_samples} text samples...")
        generated_texts = []
        reference_texts = []
        
        # Use random sentences from test data as references
        random.shuffle(self.test_data)
        
        for i in range(min(num_samples, len(self.test_data))):
            # Generate text using Markov generator
            try:
                generated_text = self.markov_generator.generate_text(50)  # Use num_words parameter
                generated_texts.append(generated_text)
                
                # Use part of test context as reference
                context, target = self.test_data[i]
                reference_text = context + " " + target
                reference_texts.append(reference_text)
                
            except Exception as e:
                print(f"Error generating text sample {i}: {e}")
                continue
        
        if not generated_texts:
            print("No text samples generated!")
            return {}
        
        print(f"Generated {len(generated_texts)} text samples")
        
        # Evaluate using comprehensive metrics
        metrics = evaluate_text_generation(
            generated_texts, 
            reference_texts, 
            self.language_model
        )
        
        print("\nText Generation Quality Metrics:")
        print("-" * 35)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric:30s}: {value:.4f}")
            else:
                print(f"{metric:30s}: {value}")
        
        # Show some example generations
        print("\nExample Generated Texts:")
        print("-" * 25)
        for i, (gen, ref) in enumerate(zip(generated_texts[:3], reference_texts[:3])):
            print(f"\nExample {i+1}:")
            print(f"Generated: {gen[:100]}...")
            print(f"Reference: {ref[:100]}...")
        
        return metrics
    
    def evaluate_bleu_detailed(self) -> Dict[str, float]:
        """
        Detailed BLEU score evaluation with different configurations.
        
        Returns:
            Dictionary with detailed BLEU metrics
        """
        print("\n" + "="*50)
        print("DETAILED BLEU SCORE ANALYSIS")
        print("="*50)
        
        if not self.markov_generator or not self.test_data:
            print("Required models or data not available!")
            return {}
        
        bleu_calculator = BLEUScore()
        
        # Generate pairs for BLEU evaluation
        num_pairs = 10
        candidates = []
        references_list = []
        
        for i in range(min(num_pairs, len(self.test_data))):
            try:
                # Generate candidate text
                candidate = self.markov_generator.generate_text(20)  # Use num_words parameter
                candidates.append(candidate)
                
                # Use test data as reference
                context, target = self.test_data[i]
                reference = context + " " + target
                references_list.append([reference])
                
            except Exception as e:
                continue
        
        if not candidates:
            print("No valid text pairs generated!")
            return {}
        
        # Calculate different BLEU variants
        metrics = {}
        
        # Standard BLEU-4
        corpus_bleu = bleu_calculator.calculate_corpus_bleu(candidates, references_list)
        metrics['corpus_bleu_4'] = corpus_bleu
        
        # BLEU with different n-gram weights
        weights_configs = {
            'bleu_1': [1.0, 0.0, 0.0, 0.0],
            'bleu_2': [0.5, 0.5, 0.0, 0.0],
            'bleu_3': [0.33, 0.33, 0.34, 0.0],
            'bleu_4': [0.25, 0.25, 0.25, 0.25]
        }
        
        for name, weights in weights_configs.items():
            score = bleu_calculator.calculate_corpus_bleu(
                candidates, references_list, weights
            )
            metrics[name] = score
        
        # Individual sentence BLEU scores
        sentence_bleus = []
        for candidate, references in zip(candidates, references_list):
            score = bleu_calculator.calculate_sentence_bleu(candidate, references[0:1])
            sentence_bleus.append(score)
        
        metrics['mean_sentence_bleu'] = sum(sentence_bleus) / len(sentence_bleus)
        metrics['max_sentence_bleu'] = max(sentence_bleus)
        metrics['min_sentence_bleu'] = min(sentence_bleus)
        
        print("\nDetailed BLEU Scores:")
        print("-" * 25)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        return metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, Dict]:
        """
        Run complete evaluation suite and return all metrics.
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("STARTING COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Setup
        self.setup_models()
        
        if not self.language_model:
            print("Failed to initialize models!")
            return {}
        
        # Create test data
        self.create_test_data(100)
        
        # Run all evaluations
        results = {
            'autocomplete_metrics': self.evaluate_autocomplete(),
            'text_generation_metrics': self.evaluate_text_generation(),
            'detailed_bleu_metrics': self.evaluate_bleu_detailed()
        }
        
        # Summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Key metrics summary
        autocomplete = results.get('autocomplete_metrics', {})
        text_gen = results.get('text_generation_metrics', {})
        bleu_detailed = results.get('detailed_bleu_metrics', {})
        
        print(f"\nKey Performance Indicators:")
        print(f"- Autocomplete Top-1 Accuracy: {autocomplete.get('top_1_accuracy', 0):.3f}")
        print(f"- Autocomplete Top-5 Accuracy: {autocomplete.get('top_5_accuracy', 0):.3f}")
        print(f"- Mean Reciprocal Rank: {autocomplete.get('mean_reciprocal_rank', 0):.3f}")
        print(f"- Text Generation BLEU-4: {bleu_detailed.get('bleu_4', 0):.3f}")
        print(f"- Corpus BLEU Score: {text_gen.get('corpus_bleu', 0):.3f}")
        print(f"- Mean Perplexity: {text_gen.get('mean_perplexity', 'N/A')}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main evaluation function."""
    evaluator = SystemEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Save results
    evaluator.save_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("Check 'evaluation_results.json' for detailed results.")


if __name__ == "__main__":
    main()