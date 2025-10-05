"""
Evaluation Metrics for N-gram Autocomplete System
=================================================
Comprehensive evaluation including BLEU scores and model performance metrics.
"""

import math
import random
from collections import Counter
from ngram_model import NGramModel
from data_preprocessing import get_tokenized_data


def calculate_bleu(candidate, reference):
    """
    Calculate BLEU score between candidate and reference text.
    
    Args:
        candidate (str): Generated text
        reference (str): Reference/ground truth text
        
    Returns:
        float: BLEU score (0.0 to 1.0)
    """
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
        
        # Geometric mean of 1-gram and 2-gram precision
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


def evaluate_autocomplete_bleu():
    """
    Evaluate autocomplete system using BLEU scores.
    Tests how well the system completes partial sentences.
    """
    print("🎯 AUTOCOMPLETE BLEU EVALUATION")
    print("=" * 50)
    
    # Load and prepare data
    with open('data/en_US.twitter.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    tokenized_sentences = get_tokenized_data(data)
    
    # Use a larger subset for better evaluation
    evaluation_size = min(15000, len(tokenized_sentences))
    eval_sentences = tokenized_sentences[:evaluation_size]
    
    # Train the model
    print("Training N-gram model...")
    model = NGramModel(max_n=4)
    model.train(eval_sentences, count_threshold=1)
    
    # Test samples for autocomplete evaluation
    test_samples = [
        {"prompt": "i am", "reference": "i am going to work today"},
        {"prompt": "the weather", "reference": "the weather is really nice today"},
        {"prompt": "this is", "reference": "this is a great example"},
        {"prompt": "you can", "reference": "you can do this easily"},
        {"prompt": "we should", "reference": "we should go out tonight"},
        {"prompt": "i love", "reference": "i love this new song"},
        {"prompt": "it was", "reference": "it was an amazing experience"},
        {"prompt": "i think", "reference": "i think this is correct"},
        {"prompt": "have a", "reference": "have a wonderful day"},
        {"prompt": "thank you", "reference": "thank you so much for helping"},
    ]
    
    print(f"\nEvaluating with {len(test_samples)} test samples...")
    print("-" * 50)
    
    bleu_scores = []
    
    for i, sample in enumerate(test_samples, 1):
        try:
            # Generate completion using the model with optimized parameters
            generated = model.generate_text(sample["prompt"], max_length=7, n_order=4, k=0.005)
            full_generated = sample["prompt"] + " " + generated
            
            # Calculate BLEU score
            bleu = calculate_bleu(full_generated, sample["reference"])
            bleu_scores.append(bleu)
            
            print(f"Sample {i:2d}: BLEU = {bleu:.3f}")
            print(f"  Prompt:    '{sample['prompt']}'")
            print(f"  Generated: '{full_generated}'")
            print(f"  Reference: '{sample['reference']}'")
            print()
            
        except Exception as e:
            print(f"Error with sample {i}: {e}")
            bleu_scores.append(0.0)
    
    # Calculate overall results
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        max_bleu = max(bleu_scores)
        min_bleu = min(bleu_scores)
        
        print("📊 BLEU EVALUATION RESULTS:")
        print("-" * 30)
        print(f"Average BLEU Score: {avg_bleu:.3f}")
        print(f"Maximum BLEU Score: {max_bleu:.3f}")
        print(f"Minimum BLEU Score: {min_bleu:.3f}")
        print(f"Samples Evaluated:  {len(bleu_scores)}")
        print(f"Training Sentences: {len(eval_sentences)}")
        
        # Performance assessment
        if avg_bleu > 0.3:
            assessment = "Excellent 🌟"
        elif avg_bleu > 0.2:
            assessment = "Good 👍"
        elif avg_bleu > 0.1:
            assessment = "Fair 👌"
        else:
            assessment = "Needs Improvement 📈"
        
        print(f"Performance Level:  {assessment}")
        
        return avg_bleu
    else:
        print("No valid BLEU scores calculated.")
        return 0.0


def compare_text_generation_bleu():
    """
    Compare different text generation approaches using BLEU scores.
    """
    print("\n🔄 COMPARING TEXT GENERATION METHODS")
    print("=" * 50)
    
    # Load data
    with open('data/en_US.twitter.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    tokenized_sentences = get_tokenized_data(data)
    eval_sentences = tokenized_sentences[:8000]
    
    # Train model with better parameters
    model = NGramModel(max_n=4)
    model.train(eval_sentences, count_threshold=1)
    
    # Test different n-gram orders and optimized smoothing parameters
    test_configs = [
        {"n_order": 2, "k": 0.005, "name": "Bigram (k=0.005)"},
        {"n_order": 3, "k": 0.005, "name": "Trigram (k=0.005)"},
        {"n_order": 4, "k": 0.005, "name": "4-gram (k=0.005)"},
        {"n_order": 4, "k": 0.001, "name": "4-gram (k=0.001)"},
    ]
    
    reference_text = "i am going to the store today to buy some groceries"
    prompt = "i am going"
    
    print(f"Reference: '{reference_text}'")
    print(f"Prompt: '{prompt}'")
    print("-" * 50)
    
    results = []
    
    for config in test_configs:
        try:
            generated = model.generate_text(prompt, max_length=10, 
                                          n_order=config["n_order"], 
                                          k=config["k"])
            full_generated = prompt + " " + generated
            bleu = calculate_bleu(full_generated, reference_text)
            
            results.append({
                "name": config["name"],
                "generated": full_generated,
                "bleu": bleu
            })
            
            print(f"{config['name']:20} | BLEU: {bleu:.3f} | '{full_generated}'")
            
        except Exception as e:
            print(f"{config['name']:20} | Error: {e}")
    
    if results:
        best_result = max(results, key=lambda x: x["bleu"])
        print(f"\n🏆 Best Method: {best_result['name']} (BLEU: {best_result['bleu']:.3f})")


def main():
    """Main function to run BLEU evaluation"""
    try:
        # Run autocomplete BLEU evaluation
        avg_bleu = evaluate_autocomplete_bleu()
        
        # Compare different generation methods
        compare_text_generation_bleu()
        
        print(f"\n✅ BLEU evaluation completed successfully!")
        print(f"Overall average BLEU score: {avg_bleu:.3f}")
        
    except FileNotFoundError:
        print("❌ Error: Training data file 'data/en_US.twitter.txt' not found.")
        print("Please ensure the data file is available.")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


if __name__ == "__main__":
    main()