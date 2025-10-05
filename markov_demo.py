"""
Markov Text Generator Demo

This script demonstrates the Markov chain text generator integrated with the existing
n-gram language model project. It provides multiple ways to use the generator:
1. Interactive mode for experimenting with different parameters
2. Integration with the project's Twitter dataset
3. Comparison with the existing n-gram model
"""

import os
import sys
from markov_text_generator import MarkovTextGenerator


def demo_with_twitter_data():
    """Demo the Markov generator using the Twitter dataset."""
    data_file = "./data/en_US.twitter.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Twitter data file not found at {data_file}")
        print("Please ensure the data file exists in the data directory.")
        return False
    
    print("=== Markov Text Generator Demo with Twitter Data ===\n")
    
    # Load and train the model
    print("Loading and training Markov model with Twitter data...")
    generator = MarkovTextGenerator()
    
    try:
        generator.train_from_file(data_file)
        print("✓ Model trained successfully!\n")
    except Exception as e:
        print(f"Error training model: {e}")
        return False
    
    # Generate some example texts
    word_counts = [20, 50, 100]
    
    for word_count in word_counts:
        print(f"--- Generated text ({word_count} words) ---")
        try:
            generated = generator.generate_text(word_count)
            print(generated)
        except Exception as e:
            print(f"Error generating text: {e}")
        print()
    
    return True


def interactive_demo():
    """Interactive demo allowing user to experiment with the generator."""
    data_file = "./data/en_US.twitter.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Twitter data file not found at {data_file}")
        return
    
    print("=== Interactive Markov Text Generator ===\n")
    print("Training model with Twitter data...")
    
    generator = MarkovTextGenerator()
    generator.train_from_file(data_file)
    print("Model trained successfully!\n")
    
    while True:
        try:
            user_input = input("Enter number of words to generate (or 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            try:
                num_words = int(user_input)
                if num_words <= 0:
                    print("Please enter a positive number.")
                    continue
                
                print(f"\nGenerating {num_words} words...")
                print("-" * 50)
                generated = generator.generate_text(num_words)
                print(generated)
                print("-" * 50)
                print()
                
            except ValueError:
                print("Please enter a valid number.")
                continue
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def compare_models_demo():
    """Compare the Markov generator with the existing n-gram model."""
    print("=== Model Comparison Demo ===\n")
    
    # This function would integrate with the existing n-gram model
    # For now, we'll just show the Markov generator
    print("Note: This demo shows the Markov generator output.")
    print("To compare with the n-gram model, you would need to:")
    print("1. Train both models on the same dataset")
    print("2. Generate text with both models")
    print("3. Compare coherence, creativity, and computational efficiency")
    print()
    
    demo_with_twitter_data()


def create_custom_training_demo():
    """Demo showing how to train on custom text."""
    print("=== Custom Training Demo ===\n")
    
    custom_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully.
    Brown foxes are known for their agility. Foxes hunt at night and sleep during the day.
    Dogs and foxes are both canines. Canines are intelligent animals.
    Animals in the wild have different behaviors than domestic animals.
    Domestic animals like dogs are loyal companions to humans.
    Humans have domesticated many animals throughout history.
    History shows us how relationships between humans and animals evolved.
    Evolution is a fascinating process that shapes all living things.
    """
    
    print("Training Markov model on custom text...")
    generator = MarkovTextGenerator()
    generator.train_from_text(custom_text)
    
    print("Generating text from custom training data:\n")
    
    for i in range(3):
        print(f"Generation {i+1}:")
        try:
            generated = generator.generate_text(30)
            print(generated)
        except Exception as e:
            print(f"Error: {e}")
        print()


def main():
    """Main menu for the demo."""
    print("Markov Text Generator Demo")
    print("=" * 30)
    
    while True:
        print("\nChoose a demo option:")
        print("1. Demo with Twitter data")
        print("2. Interactive generator")
        print("3. Model comparison (placeholder)")
        print("4. Custom training demo")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            demo_with_twitter_data()
        elif choice == '2':
            interactive_demo()
        elif choice == '3':
            compare_models_demo()
        elif choice == '4':
            create_custom_training_demo()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()