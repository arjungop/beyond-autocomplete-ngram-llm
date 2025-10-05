"""
Evaluation Metrics for N-gram Language Models
============================================
BLEU Score and Perplexity evaluation for autocomplete models.
"""

import math
from collections import Counter
from ngram_model import NGramModel
from data_preprocessing import get_tokenized_data


def calculate_bleu(candidate, reference):
    """Calculate BLEU score between candidate and reference text"""
    if not candidate or not reference:
        return 0.0
    
    c_words = candidate.lower().split()
    r_words = reference.lower().split()
    
    if not c_words:
        return 0.0
    
    # Calculate precision for 1-gram and 2-gram
    c_1grams = Counter(c_words)
    r_1grams = Counter(r_words)
    
    matches_1 = sum(min(c_1grams[word], r_1grams[word]) for word in c_1grams if word in r_1grams)
    precision_1 = matches_1 / len(c_words) if c_words else 0.0
    
    # 2-gram precision
    if len(c_words) >= 2 and len(r_words) >= 2:
        c_2grams = Counter(zip(c_words[:-1], c_words[1:]))
        r_2grams = Counter(zip(r_words[:-1], r_words[1:]))
        
        matches_2 = sum(min(c_2grams[bg], r_2grams[bg]) for bg in c_2grams if bg in r_2grams)
        precision_2 = matches_2 / max(len(c_words) - 1, 1)
        
        # Geometric mean
        if precision_1 > 0 and precision_2 > 0:
            bleu = (precision_1 * precision_2) ** 0.5
        else:
            bleu = precision_1
    else:
        bleu = precision_1
    
    # Brevity penalty
    ref_len = len(r_words)
    cand_len = len(c_words)
    
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    return bleu * bp


def run_evaluation():
    """Run model evaluation with BLEU and perplexity metrics"""
    print("📊 N-GRAM MODEL EVALUATION")
    print("=" * 40)
    
    # Load data  
    with open('data/en_US.twitter.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    tokenized_sentences = get_tokenized_data(data)
    data_size = min(10000, len(tokenized_sentences))
    tokenized_sentences = tokenized_sentences[:data_size]
    
    # Train n-gram model
    model = NGramModel(max_n=4)
    model.train(tokenized_sentences, count_threshold=2)
    
    # Test samples
    test_samples = [
        {"prompt": "i am", "reference": "i am going to work"},
        {"prompt": "the weather", "reference": "the weather is nice"},
        {"prompt": "this is", "reference": "this is great"},
        {"prompt": "you can", "reference": "you can do this"},
        {"prompt": "we should", "reference": "we should go out"},
    ]
    
    print("\n📋 EVALUATION RESULTS:")
    print("-" * 40)
    
    bleu_scores = []
    
    for i, sample in enumerate(test_samples, 1):
        # Generate text
        generated = model.generate_text(sample["prompt"], max_length=6, n_order=2, k=0.05)
        full_generated = sample["prompt"] + " " + generated
        
        # Calculate BLEU
        bleu = calculate_bleu(full_generated, sample["reference"])
        bleu_scores.append(bleu)
        
        print(f"Sample {i}: BLEU={bleu:.3f}")
        print(f"  Generated: '{full_generated}'")
        print(f"  Reference: '{sample['reference']}'")
        print()
    
    # Final results
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print("📈 EVALUATION RESULTS:")
    print("-" * 25)
    print(f"Average BLEU Score: {avg_bleu:.3f}")
    
    # Calculate perplexity
    perplexities = model.evaluate_perplexity([sample["reference"].split() for sample in test_samples])
    avg_perplexity = perplexities.get('2-gram', 0)
    
    print(f"Average Perplexity: {avg_perplexity:.1f}")
    print(f"Model trained on {len(tokenized_sentences)} sentences")
    
    return avg_bleu


if __name__ == "__main__":
    run_evaluation()