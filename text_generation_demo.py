"""
Markov Chain Text Generation Demo

This module provides a user-friendly demonstration of Markov chain text generation
using the Twitter dataset. It showcases both automated generation with predefined
parameters and interactive generation where users can specify their own parameters.

Key Features:
- Automated demo with multiple text samples
- Interactive mode for custom text generation
- Integration with the MarkovTextGenerator class
- Error handling for missing data files
"""

import os  # For file system operations and path checking
from markov_text_generator import MarkovTextGenerator  # Our custom Markov text generator


def run_text_generation_demo():
    """
    Demonstrate Markov chain text generation with predefined sample parameters.
    
    This function provides an automated demonstration of text generation using
    the Markov chain model. It generates multiple text samples of different
    lengths to showcase the model's capabilities and text variety.
    """
    print("Markov Chain Text Generation")
    print("=" * 35)
    
    # Define the path to our training data file
    data_file = "./data/en_US.twitter.txt"
    
    # Check if the required data file exists before proceeding
    if not os.path.exists(data_file):
        print(f"Error: Dataset file {data_file} not found.")
        print("Please ensure the Twitter dataset is available.")
        return
    
    # Step 1: Initialize the Markov text generator
    # This creates an empty generator ready for training
    generator = MarkovTextGenerator()
    
    # Step 2: Train the generator using the Twitter dataset
    print("Loading training data...")
    generator.train_from_file(data_file)
    print("Training completed!")
    print()
    
    # Step 3: Generate sample texts with different lengths
    # This demonstrates how text quality and coherence vary with length
    print("Generated Text Samples:")
    print("-" * 25)
    
    # Define different sample lengths to showcase text generation variety
    # Shorter texts tend to be more coherent, longer texts show more diversity
    sample_lengths = [20, 30, 40]
    
    # Generate and display text samples for each specified length
    for i, length in enumerate(sample_lengths, 1):
        print(f"\nSample {i} ({length} words):")
        
        # Generate text using the trained Markov model
        # The model creates text by following bigram transition probabilities
        text = generator.generate_text(length)
        
        # Display the generated text in quotes for clear presentation
        print(f'"{text}"')
    
    # Display training statistics to show model complexity
    print(f"\nGeneration completed using {len(generator.possibles)} unique word pairs.")


def interactive_generation():
    """
    Interactive text generation mode with user-specified parameters.
    
    This function provides an interactive interface where users can specify
    their own text generation parameters. It continues generating text based
    on user input until the user chooses to quit, allowing for experimentation
    with different text lengths and repeated generation.
    """
    print("\nInteractive Generation Mode")
    print("-" * 28)
    
    # Initialize a new generator for interactive use
    generator = MarkovTextGenerator()
    
    # Train the model using the Twitter dataset
    # This step is necessary before any text generation can occur
    print("Training model...")
    generator.train_from_file("./data/en_US.twitter.txt")
    print("Model training completed!")
    print("You can now generate text of any length.")
    
    # Main interactive loop - continues until user chooses to quit
    while True:
        try:
            # Get user input for desired text length
            # Provide clear instructions and exit option
            length = input("\nEnter text length (or 'quit' to exit): ")
            
            # Check if user wants to exit the interactive mode
            if length.lower() == 'quit':
                print("Exiting interactive mode. Goodbye!")
                break
            
            # Convert input to integer and validate
            length = int(length)
            if length <= 0:
                print("Please enter a positive number.")
                continue
            
            # Generate text with the specified length
            # The generator uses trained bigram probabilities to create coherent text
            text = generator.generate_text(length)
            
            # Display the generated text with clear formatting
            print(f'\nGenerated: "{text}"')
            
        except ValueError:
            # Handle non-numeric input gracefully
            print("Please enter a valid number.")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting...")
            break
        
        except Exception as e:
            # Handle any unexpected errors
            print(f"An error occurred: {e}")
            print("Please try again with a different input.")


if __name__ == "__main__":
    # Entry point when script is run directly
    
    # Step 1: Run the automated demonstration
    # This shows predefined examples of text generation
    run_text_generation_demo()
    
    # Step 2: Offer interactive mode to the user
    # This allows users to experiment with their own parameters
    response = input("\nWould you like to try interactive generation? (y/n): ")
    
    # Check user response and launch interactive mode if requested
    if response.lower().startswith('y'):
        interactive_generation()
    else:
        print("Thank you for trying the Markov text generation demo!")