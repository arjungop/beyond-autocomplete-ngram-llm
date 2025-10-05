"""
Markov Chain Text Generator

This module implements a simple 2-gram (bigram) Markov chain text generator
that builds a probabilistic model of text and generates random sentences
based on the learned patterns.

Usage:
    python markov_text_generator.py <num_words> < input_text_file.txt

Example:
    python markov_text_generator.py 100 < data/en_US.twitter.txt
"""

import collections
import random
import sys
import textwrap


class MarkovTextGenerator:
    """A simple Markov chain text generator using bigrams."""
    
    def __init__(self):
        """Initialize the generator with an empty possibles table."""
        self.possibles = collections.defaultdict(list)
        self.trained = False
    
    def train_from_text(self, text):
        """
        Train the Markov model from input text.
        
        Args:
            text (str): The input text to train on
        """
        # Initialize prefix words
        w1 = w2 = ''
        
        # Process each line of text
        for line in text.splitlines():
            for word in line.split():
                # Add word as a possible continuation of the current bigram
                self.possibles[w1, w2].append(word)
                # Shift the window: previous w2 becomes w1, current word becomes w2
                w1, w2 = w2, word
        
        # Handle end of input to avoid empty possibles lists
        self.possibles[w1, w2].append('')
        self.possibles[w2, ''].append('')
        
        self.trained = True
    
    def train_from_file(self, filename):
        """
        Train the Markov model from a text file.
        
        Args:
            filename (str): Path to the text file
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            self.train_from_text(text)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Could not decode file '{filename}'. Please check the encoding.")
            sys.exit(1)
    
    def train_from_stdin(self):
        """Train the Markov model from standard input."""
        # Initialize prefix words
        w1 = w2 = ''
        
        # Process each line from stdin
        for line in sys.stdin:
            for word in line.split():
                # Add word as a possible continuation of the current bigram
                self.possibles[w1, w2].append(word)
                # Shift the window: previous w2 becomes w1, current word becomes w2
                w1, w2 = w2, word
        
        # Handle end of input to avoid empty possibles lists
        self.possibles[w1, w2].append('')
        self.possibles[w2, ''].append('')
        
        self.trained = True
    
    def generate_text(self, num_words):
        """
        Generate random text using the trained Markov model.
        
        Args:
            num_words (int): Number of words to generate
            
        Returns:
            str: Generated text wrapped to 70 columns
        """
        if not self.trained:
            raise ValueError("Model must be trained before generating text")
        
        if not self.possibles:
            raise ValueError("No training data available")
        
        # Find all bigrams that start with a capitalized word
        capitalized_starts = [k for k in self.possibles.keys() if k[0] and k[0][:1].isupper()]
        
        if not capitalized_starts:
            # If no capitalized starts, use any available bigram
            capitalized_starts = [k for k in self.possibles.keys() if k[0]]
        
        if not capitalized_starts:
            raise ValueError("No suitable starting bigrams found in training data")
        
        # Start with a random capitalized prefix
        w1, w2 = random.choice(capitalized_starts)
        output = [w1, w2]
        
        # Generate the requested number of additional words
        for _ in range(num_words):
            if (w1, w2) not in self.possibles or not self.possibles[w1, w2]:
                # If no continuations available, try to find a new starting point
                available_starts = [k for k in self.possibles.keys() if k[0] and self.possibles[k]]
                if not available_starts:
                    break
                w1, w2 = random.choice(available_starts)
            
            # Choose a random word from the possible continuations
            word = random.choice(self.possibles[w1, w2])
            
            # If we hit an empty string (end marker), we might want to start a new sentence
            if word == '':
                # Try to find a new sentence start
                sentence_starts = [k for k in self.possibles.keys() if k[0] and k[0][:1].isupper() and self.possibles[k]]
                if sentence_starts:
                    w1, w2 = random.choice(sentence_starts)
                    word = w1  # Use the first word of the new bigram
                    if len(output) > 0 and output[-1]:  # Add period if the last word isn't empty
                        output.append('.')
                else:
                    break
            
            output.append(word)
            w1, w2 = w2, word
        
        # Filter out empty strings and format output
        filtered_output = [word for word in output if word]
        return textwrap.fill(' '.join(filtered_output), width=70)


def main():
    """Main function to run the Markov text generator from command line."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python markov_text_generator.py <num_words>")
        print("       echo 'text' | python markov_text_generator.py <num_words>")
        print("       python markov_text_generator.py <num_words> < input_file.txt")
        sys.exit(1)
    
    try:
        num_words = int(sys.argv[1])
        if num_words <= 0:
            raise ValueError("Number of words must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a valid positive integer for the number of words.")
        sys.exit(1)
    
    # Create and train the generator
    generator = MarkovTextGenerator()
    
    try:
        generator.train_from_stdin()
        generated_text = generator.generate_text(num_words)
        print(generated_text)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()