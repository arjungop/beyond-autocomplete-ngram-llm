#!/usr/bin/env python3
"""
Custom Text Generation Demo

This script shows how to train the Markov generator with your own input text
for more controlled and coherent text generation.
"""

from markov_text_generator import MarkovTextGenerator
import tempfile
import os


def generate_from_custom_text():
    """Generate text from user-provided input."""
    print("=== Custom Text Generation ===\n")
    print("Enter your training text (end with 'END' on a new line):")
    
    # Collect user input
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    if not lines:
        print("No text provided.")
        return
    
    training_text = '\n'.join(lines)
    print(f"\nTraining text length: {len(training_text)} characters")
    print(f"Training text preview: {training_text[:100]}...")
    
    # Train the model
    generator = MarkovTextGenerator()
    generator.train_from_text(training_text)
    print("✓ Model trained successfully!\n")
    
    # Generate text
    while True:
        try:
            words_input = input("Enter number of words to generate (or 'q' to quit): ")
            if words_input.lower() == 'q':
                break
            
            num_words = int(words_input)
            if num_words <= 0:
                print("Please enter a positive number.")
                continue
            
            print(f"\nGenerated text ({num_words} words):")
            print("-" * 50)
            generated = generator.generate_text(num_words)
            print(generated)
            print("-" * 50)
            print()
            
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def generate_from_file():
    """Generate text from a user-specified file."""
    print("=== File-based Text Generation ===\n")
    
    filename = input("Enter the path to your text file: ").strip()
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return
    
    try:
        # Train the model
        generator = MarkovTextGenerator()
        generator.train_from_file(filename)
        print("✓ Model trained successfully!\n")
        
        # Generate text
        while True:
            try:
                words_input = input("Enter number of words to generate (or 'q' to quit): ")
                if words_input.lower() == 'q':
                    break
                
                num_words = int(words_input)
                if num_words <= 0:
                    print("Please enter a positive number.")
                    continue
                
                print(f"\nGenerated text ({num_words} words):")
                print("-" * 50)
                generated = generator.generate_text(num_words)
                print(generated)
                print("-" * 50)
                print()
                
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function with menu options."""
    while True:
        print("\n=== Custom Text Generation Options ===")
        print("1. Enter text manually")
        print("2. Load text from file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            generate_from_custom_text()
        elif choice == '2':
            generate_from_file()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()