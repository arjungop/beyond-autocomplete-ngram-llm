"""
N-gram Language Model Perplexity Evaluation
==========================================
Professional implementation for evaluating n-gram language model perplexity.
"""

import time
from ngram_model import NGramModel
from data_preprocessing import get_tokenized_data


def evaluate_model_perplexity():
    """Evaluate n-gram language models with perplexity metrics"""
    print("N-gram Language Model Evaluation")
    print("=" * 40)
    
    # Load training data
    with open('data/en_US.twitter.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    tokenized_sentences = get_tokenized_data(data)
    total_sentences = len(tokenized_sentences)
    
    # Use comprehensive dataset for robust evaluation
    training_size = min(45000, total_sentences)
    tokenized_sentences = tokenized_sentences[:training_size]
    
    print(f"Dataset: {training_size:,} sentences")
    print(f"Total available: {total_sentences:,} sentences")
    print()
    
    # Train model
    start_time = time.time()
    model = NGramModel(max_n=4)
    model.train(tokenized_sentences, count_threshold=1)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Vocabulary size: {len(model.vocabulary):,} words")
    print()
    
    # Evaluate perplexity for different smoothing parameters
    print("Perplexity Evaluation Results")
    print("-" * 30)
    
    smoothing_values = [0.0005, 0.001, 0.002, 0.005]
    best_results = {}
    
    for k in smoothing_values:
        perplexities = model.evaluate_perplexity(k=k)
        print(f"\nSmoothing parameter k={k}:")
        
        for order, perp in sorted(perplexities.items()):
            print(f"  {order}: {perp:.1f}")
            if order not in best_results or perp < best_results[order]:
                best_results[order] = perp
    
    # Display final results
    print(f"\nOptimal Results Summary")
    print("=" * 25)
    
    for order, perp in sorted(best_results.items()):
        print(f"{order}: {perp:.1f}")
    
    print(f"\nModel Performance:")
    print(f"- Training time: {training_time:.1f}s")
    print(f"- Dataset size: {training_size:,} sentences")
    print(f"- Vocabulary: {len(model.vocabulary):,} words")
    
    return best_results


if __name__ == "__main__":
    evaluate_model_perplexity()