"""
Interactive Autocomplete System

This is the main module that implements an interactive autocomplete system using
N-gram language models. It loads data, trains multiple n-gram models, and provides
a command-line interface for users to get word suggestions in real-time.
"""

import random  # For shuffling data and random sampling
import nltk    # Natural Language Toolkit for tokenization

# Add current directory to NLTK data path for accessing downloaded resources
nltk.data.path.append('.')

# Import custom modules for data processing and language modeling
from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions


def load_and_preprocess_data():
    """
    Load training data from file and preprocess it for language model training.
    
    This function handles the complete data loading and preprocessing pipeline:
    1. Loads raw text data from the Twitter dataset file
    2. Tokenizes and splits into training/test sets
    3. Filters vocabulary and handles out-of-vocabulary words
    4. Returns processed data ready for n-gram model training
    
    Returns:
        tuple: (train_data_processed, test_data_processed, vocabulary)
    """
    print("Loading and preprocessing data...")
    
    # Step 1: Load raw text data from the dataset file
    # The Twitter dataset contains informal text suitable for autocomplete training
    with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
        # Read entire file content as a single string
        data = f.read()
    
    # Step 2: Tokenize the raw text into sentences and words
    # This converts the text into a list of lists (sentences containing words)
    tokenized_data = get_tokenized_data(data)
    
    # Step 3: Shuffle data for randomization to avoid any ordering bias
    # Set random seed for reproducible results across runs
    random.seed(87)
    random.shuffle(tokenized_data)

    # Step 4: Split data into training and test sets (80/20 split)
    # Training data will be used to build n-gram counts
    # Test data will be used to evaluate model performance
    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[0:train_size]           # First 80% for training
    test_data = tokenized_data[train_size:]             # Last 20% for testing
    
    # Step 5: Apply vocabulary filtering and OOV word replacement
    # minimum_freq=5 means words must appear at least 5 times to be included
    # This reduces noise from typos and very rare words
    minimum_freq = 5  # Threshold for including words in vocabulary
    train_data_processed, test_data_processed, vocabulary = preprocess_data(
        train_data, test_data, minimum_freq
    )

    # Display data statistics for user information
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training data size: {len(train_data_processed)} sentences")
    
    return train_data_processed, test_data_processed, vocabulary


def get_user_input_suggestions(vocabulary, n_gram_counts_list, k=1.0):
    """
    Interactive loop for providing autocomplete suggestions to users.
    
    This function implements the main user interface for the autocomplete system.
    It continuously prompts users for input, tokenizes their text, generates
    suggestions using multiple n-gram models, and displays ranked results.
    
    Args:
        vocabulary (list): List of vocabulary words from training
        n_gram_counts_list (list): List of n-gram count dictionaries for different orders
        k (float): Smoothing parameter for probability calculations
    """
    print("Interactive Autocomplete System")
    print("Type sentences and get word suggestions!")
    print("Enter 'q' to quit the program.\n")
    
    # Main interaction loop - continues until user types 'q'
    while True:
        # Get user input and normalize it
        # Convert to lowercase to match training data format
        # Strip whitespace to handle extra spaces
        user_input = input("Enter a sentence (or 'q' to quit): ").lower().strip()
        
        # Check for quit command
        if user_input == 'q':
            print("Goodbye!")
            break

        # Tokenize user input into individual words
        # This converts the sentence string into a list of words
        # Uses NLTK tokenizer to handle punctuation and word boundaries
        tokens = nltk.word_tokenize(user_input)

        # Generate suggestions using all available n-gram models
        # This returns suggestions from unigram, bigram, trigram, etc. models
        # Each model provides one suggestion based on different context lengths
        suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k)

        # Remove duplicate suggestions and keep the highest probability for each word
        # Multiple n-gram models might suggest the same word with different probabilities
        unique_suggestions = {}
        for word, prob in suggestions:
            # Keep track of the highest probability seen for each word
            if word not in unique_suggestions or prob > unique_suggestions[word]:
                unique_suggestions[word] = prob
        
        # Sort suggestions by probability in descending order
        # This puts the most confident predictions first for better user experience
        sorted_suggestions = sorted(unique_suggestions.items(), key=lambda x: x[1], reverse=True)

        # Display the ranked suggestions to the user
        print("\nSuggestions:")
        for i, (word, prob) in enumerate(sorted_suggestions, 1):
            # Format probability to 6 decimal places for readability
            print(f"{i}. {word} (probability: {prob:.6f})")

        # Add spacing for better readability
        print("\n")


def main():
    """
    Main function that orchestrates the entire autocomplete system.
    
    This function coordinates all the components of the autocomplete system:
    1. Loads and preprocesses the training data
    2. Builds n-gram models of different orders (1-gram through 4-gram)
    3. Launches the interactive user interface for getting suggestions
    
    The system uses multiple n-gram models to provide diverse suggestions
    based on different context lengths, giving users comprehensive autocomplete options.
    """
    print("Initializing N-gram Autocomplete System...")
    
    # Step 1: Load and preprocess all training data
    # This handles file loading, tokenization, train/test split, and vocabulary filtering
    train_data_processed, test_data_processed, vocabulary = load_and_preprocess_data()
    
    print("Building n-gram language models...")

    # Step 2: Build n-gram count dictionaries for multiple model orders
    # We create models from unigram (1-gram) up to 4-gram for comprehensive coverage
    # Each order captures different amounts of context for predictions
    n_gram_counts_list = []
    
    # Build models for n-gram orders 1 through 4
    for n in range(1, 5):
        print(f"Building {n}-gram model...")
        
        # Count all n-grams of order 'n' in the processed training data
        # This creates frequency dictionaries for probability calculations
        n_gram_counts = count_n_grams(train_data_processed, n)
        
        # Add this model's counts to our collection
        n_gram_counts_list.append(n_gram_counts)

    print("Models built successfully!")
    print("Starting interactive autocomplete system...\n")

    # Step 3: Launch the interactive autocomplete interface
    # This starts the main user interaction loop for getting suggestions
    get_user_input_suggestions(vocabulary, n_gram_counts_list)


if __name__ == "__main__":
    # Entry point: only run main() if this script is executed directly
    # This allows the module to be imported without automatically running
    main()
