"""
Markov Chain Text Generator
===========================
Implementation of a Markov chain text generator for natural language generation.
This module creates text by learning bigram transitions from training data.
"""

# Import required libraries
import collections  # For defaultdict to store word transitions
import random       # For random word selection during generation
import sys         # For system-level operations
import textwrap    # For formatting generated text output


class MarkovTextGenerator:
    """
    Markov chain text generator using bigram transitions.
    
    This class implements a simple Markov chain model where each word is predicted
    based on the previous two words (bigram context). The model learns transition
    probabilities from training text and uses them for text generation.
    """
    
    def __init__(self):
        """
        Initialize the Markov text generator.
        
        Sets up empty data structures for storing word transitions and
        initializes the training state flag.
        """
        # Dictionary to store possible next words for each word pair (bigram)
        # Key: (word1, word2) tuple, Value: list of possible next words
        self.possibles = collections.defaultdict(list)
        
        # Flag to track if model has been trained
        self.trained = False
    
    def train_from_text(self, text):
        """
        Train the Markov model from input text string.
        
        This method processes the input text to build bigram transitions.
        For each sequence of three consecutive words (w1, w2, w3), it learns
        that w3 can follow the bigram (w1, w2).
        
        Args:
            text (str): The input text to train on
        """
        # Initialize the sliding window with empty strings
        # w1 and w2 represent the previous two words in our bigram context
        w1 = w2 = ''
        
        # Process each line of the input text separately
        for line in text.splitlines():
            # Split line into individual words
            for word in line.split():
                # Record that 'word' can follow the bigram (w1, w2)
                # This builds our transition probability table
                self.possibles[w1, w2].append(word)
                
                # Slide the window forward: shift words left and add new word
                # Previous w2 becomes new w1, current word becomes new w2
                w1, w2 = w2, word
        
        # Handle end of input to create proper sentence endings
        # Add empty string as possible continuation to indicate text end
        self.possibles[w1, w2].append('')
        self.possibles[w2, ''].append('')
        
        # Mark model as trained
        self.trained = True
    
    def train_from_file(self, filename):
        """
        Train the Markov model from a text file.
        
        This method reads a text file and trains the model on its contents.
        Handles file reading errors gracefully with appropriate error messages.
        
        Args:
            filename (str): Path to the text file to read and train on
        """
        try:
            # Open file with UTF-8 encoding to handle international characters
            with open(filename, 'r', encoding='utf-8') as f:
                # Read entire file content into memory
                text = f.read()
            
            # Train the model using the file content
            self.train_from_text(text)
            
        except FileNotFoundError:
            # Handle case where file doesn't exist
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)  # Exit with error code
            
        except UnicodeDecodeError:
            # Handle case where file encoding is incompatible
            print(f"Error: Could not decode file '{filename}'. Please check the encoding.")
            sys.exit(1)  # Exit with error code
    
    def train_from_stdin(self):
        """
        Train the Markov model from standard input (stdin).
        
        This method allows training from piped input or interactive text entry.
        Useful for command-line usage where text is provided via pipe or redirect.
        """
        # Initialize the sliding window with empty strings
        w1 = w2 = ''
        
        # Read and process each line from standard input
        for line in sys.stdin:
            # Split each line into individual words
            for word in line.split():
                # Record that 'word' can follow the bigram (w1, w2)
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