"""
Markov Chain Text Generation Demo
================================
Simple demonstration of Markov chain text generation using the Twitter dataset.
"""

import os
from markov_text_generator import MarkovTextGenerator


def run_text_generation_demo():
    """Demonstrate Markov chain text generation"""
    print("Markov Chain Text Generation")
    print("=" * 35)
    
    data_file = "./data/en_US.twitter.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Dataset file {data_file} not found.")
        print("Please ensure the Twitter dataset is available.")
        return
    
    # Initialize generator
    generator = MarkovTextGenerator()
    
    print("Loading training data...")
    generator.train_from_file(data_file)
    print("Training completed!")
    print()
    
    # Generate sample texts
    print("Generated Text Samples:")
    print("-" * 25)
    
    sample_lengths = [20, 30, 40]
    
    for i, length in enumerate(sample_lengths, 1):
        print(f"\nSample {i} ({length} words):")
        text = generator.generate_text(length)
        print(f'"{text}"')
    
    print(f"\nGeneration completed using {len(generator.possibles)} unique word pairs.")


def interactive_generation():
    """Interactive text generation with user-specified parameters"""
    print("\nInteractive Generation Mode")
    print("-" * 28)
    
    generator = MarkovTextGenerator()
    
    # Train the model
    print("Training model...")
    generator.train_from_file("./data/en_US.twitter.txt")
    
    while True:
        try:
            length = input("\nEnter text length (or 'quit' to exit): ")
            if length.lower() == 'quit':
                break
            
            length = int(length)
            if length <= 0:
                print("Please enter a positive number.")
                continue
            
            text = generator.generate_text(length)
            print(f'\nGenerated: "{text}"')
            
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    run_text_generation_demo()
    
    # Ask if user wants interactive mode
    response = input("\nWould you like to try interactive generation? (y/n): ")
    if response.lower().startswith('y'):
        interactive_generation()