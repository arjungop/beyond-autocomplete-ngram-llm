"""
Demonstration of BLEU Score and Accuracy Metrics
=================================================

This script demonstrates how to use the BLEU score and accuracy metrics
for evaluating autocomplete and text generation systems with practical examples.

Author: Research Team
Date: October 2025
"""

from evaluation_metrics import BLEUScore, AutocompleteAccuracy, TextGenerationQuality


def demonstrate_bleu_scores():
    """Demonstrate BLEU score calculations with various examples."""
    print("BLEU SCORE DEMONSTRATION")
    print("=" * 50)
    
    bleu_calculator = BLEUScore()
    
    print("\n1. High Quality Translation (Expected: High BLEU)")
    candidate = "the cat is sitting on the mat"
    references = ["the cat is sitting on the mat", "a cat sits on the mat"]
    score = bleu_calculator.calculate_sentence_bleu(candidate, references)
    print(f"Candidate: '{candidate}'")
    print(f"References: {references}")
    print(f"BLEU Score: {score:.4f}")
    
    print("\n2. Partial Match (Expected: Medium BLEU)")
    candidate = "the cat sits on mat"
    references = ["the cat is sitting on the mat"]
    score = bleu_calculator.calculate_sentence_bleu(candidate, references)
    print(f"Candidate: '{candidate}'")
    print(f"References: {references}")
    print(f"BLEU Score: {score:.4f}")
    
    print("\n3. Poor Quality (Expected: Low BLEU)")
    candidate = "dog runs fast today"
    references = ["the cat is sitting on the mat"]
    score = bleu_calculator.calculate_sentence_bleu(candidate, references)
    print(f"Candidate: '{candidate}'")
    print(f"References: {references}")
    print(f"BLEU Score: {score:.4f}")
    
    print("\n4. Corpus BLEU Example")
    candidates = [
        "the cat sits on the mat",
        "it is raining today",
        "python is a programming language"
    ]
    references_list = [
        ["the cat is sitting on the mat", "a cat sits on the mat"],
        ["it is raining outside", "today it is raining"],
        ["python is a programming language", "python is used for programming"]
    ]
    
    corpus_score = bleu_calculator.calculate_corpus_bleu(candidates, references_list)
    print(f"Corpus BLEU Score: {corpus_score:.4f}")
    
    # Individual scores vs corpus score
    print("\nIndividual vs Corpus BLEU:")
    individual_scores = []
    for cand, refs in zip(candidates, references_list):
        score = bleu_calculator.calculate_sentence_bleu(cand, refs)
        individual_scores.append(score)
        print(f"  '{cand}' -> {score:.4f}")
    
    avg_individual = sum(individual_scores) / len(individual_scores)
    print(f"Average Individual: {avg_individual:.4f}")
    print(f"Corpus BLEU: {corpus_score:.4f}")


def demonstrate_autocomplete_accuracy():
    """Demonstrate autocomplete accuracy metrics."""
    print("\n\nAUTOCOMPLETE ACCURACY DEMONSTRATION")
    print("=" * 50)
    
    accuracy_evaluator = AutocompleteAccuracy()
    
    print("\nSimulating autocomplete predictions...")
    
    # Test case 1: Perfect prediction
    predictions1 = ["cat", "dog", "bird", "fish", "rabbit"]
    actual1 = "cat"
    accuracy_evaluator.evaluate_prediction(predictions1, actual1, confidence=0.95)
    print(f"Test 1 - Predictions: {predictions1[:3]}... | Actual: '{actual1}' | Top-1 Match: ✓")
    
    # Test case 2: Top-3 match
    predictions2 = ["house", "home", "car", "bike", "train"]
    actual2 = "car"
    accuracy_evaluator.evaluate_prediction(predictions2, actual2, confidence=0.60)
    print(f"Test 2 - Predictions: {predictions2[:3]}... | Actual: '{actual2}' | Top-3 Match: ✓")
    
    # Test case 3: Top-5 match
    predictions3 = ["run", "walk", "jump", "swim", "fly"]
    actual3 = "fly"
    accuracy_evaluator.evaluate_prediction(predictions3, actual3, confidence=0.30)
    print(f"Test 3 - Predictions: {predictions3[:3]}... | Actual: '{actual3}' | Top-5 Match: ✓")
    
    # Test case 4: No match
    predictions4 = ["apple", "banana", "orange", "grape", "lemon"]
    actual4 = "computer"
    accuracy_evaluator.evaluate_prediction(predictions4, actual4, confidence=0.80)
    print(f"Test 4 - Predictions: {predictions4[:3]}... | Actual: '{actual4}' | No Match: ✗")
    
    # Get final metrics
    metrics = accuracy_evaluator.get_accuracy_metrics()
    
    print("\nFinal Accuracy Metrics:")
    print("-" * 25)
    print(f"Exact Match (Top-1) Accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.3f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.3f}")
    print(f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.3f}")
    print(f"Total Predictions: {metrics['total_predictions']}")


def demonstrate_text_quality_metrics():
    """Demonstrate text generation quality metrics."""
    print("\n\nTEXT GENERATION QUALITY DEMONSTRATION")
    print("=" * 50)
    
    print("\n1. Repetition Penalty")
    texts = [
        "This is a good example of varied text with different words.",
        "Good good good good text text text with with repeated words.",
        "Perfect example without any repetition of terms or phrases."
    ]
    
    for i, text in enumerate(texts, 1):
        penalty = TextGenerationQuality.repetition_penalty(text, n=2)
        print(f"Text {i}: {penalty:.3f} - {text[:50]}...")
    
    print("\n2. Length Consistency Analysis")
    generated_texts = [
        "Short text here",
        "This is a medium length text with several words",
        "Brief sample",
        "Another example of medium length text generation",
        "Tiny"
    ]
    
    stats = TextGenerationQuality.length_consistency(generated_texts, target_length=8)
    print(f"Mean Length: {stats['mean_length']:.1f} words")
    print(f"Standard Deviation: {stats['std_length']:.1f}")
    print(f"Length Range: {stats['min_length']}-{stats['max_length']} words")
    if 'mean_deviation' in stats:
        print(f"Mean Deviation from Target (8): {stats['mean_deviation']:.1f}")
        print(f"Length Accuracy (±2 words): {stats['length_accuracy']:.1%}")


def demonstrate_comprehensive_evaluation():
    """Show how all metrics work together."""
    print("\n\nCOMPREHENSIVE EVALUATION EXAMPLE")
    print("=" * 50)
    
    print("\nScenario: Evaluating an autocomplete system for mobile texting")
    print("-" * 60)
    
    # Simulate autocomplete evaluation
    accuracy = AutocompleteAccuracy()
    
    test_cases = [
        # (predictions, actual_word, scenario)
        (["the", "they", "there", "them", "then"], "the", "Common word prediction"),
        (["going", "good", "great", "getting", "giving"], "going", "Verb prediction"),
        (["because", "before", "between", "beautiful", "believe"], "because", "Longer word"),
        (["hello", "help", "here", "home", "hope"], "hello", "Greeting"),
        (["python", "program", "project", "problem", "process"], "python", "Technical term"),
    ]
    
    print("\nAutocomplete Test Results:")
    for predictions, actual, scenario in test_cases:
        accuracy.evaluate_prediction(predictions, actual, confidence=0.75)
        top1_match = "✓" if predictions[0] == actual else "✗"
        top3_match = "✓" if actual in predictions[:3] else "✗"
        print(f"  {scenario:20s}: Top-1 {top1_match} | Top-3 {top3_match} | Target: '{actual}'")
    
    metrics = accuracy.get_accuracy_metrics()
    print(f"\nOverall Performance:")
    print(f"  Top-1 Accuracy: {metrics['top_1_accuracy']:.1%}")
    print(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.1%}")
    print(f"  Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}")
    
    # Simulate text generation evaluation
    print(f"\nText Generation Evaluation:")
    generated = "I am going to the store today to buy some groceries"
    reference = "I'm going to the store to buy groceries today"
    
    bleu = BLEUScore()
    bleu_score = bleu.calculate_sentence_bleu(generated, [reference])
    repetition = TextGenerationQuality.repetition_penalty(generated)
    
    print(f"  Generated: '{generated}'")
    print(f"  Reference: '{reference}'")
    print(f"  BLEU Score: {bleu_score:.3f}")
    print(f"  Repetition Score: {repetition:.3f}")
    
    # Overall assessment
    print(f"\nSystem Assessment:")
    overall_score = (metrics['top_1_accuracy'] + bleu_score + repetition) / 3
    print(f"  Overall Quality Score: {overall_score:.3f}/1.0")
    
    if overall_score > 0.7:
        print("  ✓ System performs well")
    elif overall_score > 0.4:
        print("  ⚠ System needs improvement")
    else:
        print("  ✗ System requires significant work")


if __name__ == "__main__":
    print("EVALUATION METRICS DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the BLEU score and accuracy metrics")
    print("for autocomplete and text generation evaluation.")
    
    demonstrate_bleu_scores()
    demonstrate_autocomplete_accuracy()
    demonstrate_text_quality_metrics()
    demonstrate_comprehensive_evaluation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• BLEU scores measure similarity between generated and reference text")
    print("• Accuracy metrics evaluate autocomplete prediction quality")
    print("• Multiple metrics provide comprehensive evaluation")
    print("• Use these metrics to compare different model versions")
    print("• Higher scores generally indicate better performance")