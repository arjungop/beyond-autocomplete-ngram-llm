"""
Data Preprocessing Module for N-gram Language Model

This module provides functions for cleaning and preparing text data for training
N-gram language models. It handles tokenization, vocabulary filtering, and 
out-of-vocabulary (OOV) word replacement to create consistent training data.
"""

import nltk  # Natural Language Toolkit for tokenization

# Add current directory to NLTK data path for accessing downloaded resources
nltk.data.path.append('.')


def split_to_sentences(data):
    """
    Split raw text data into individual sentences.
    
    This function takes a multi-line text string and converts it into a list
    of clean sentences, removing empty lines and whitespace.
    
    Args:
        data (str): Raw text data with multiple lines
        
    Returns:
        list: List of cleaned sentences as strings
    """
    # Split the input text at line breaks (\n) to get individual lines
    sentences = data.splitlines()
    
    # Clean each sentence by removing leading and trailing whitespace
    # This handles cases where lines have extra spaces or tabs
    sentences = [s.strip() for s in sentences]
    
    # Filter out empty sentences (lines that contained only whitespace)
    # This ensures we only keep sentences with actual content
    sentences = [s for s in sentences if len(s) > 0]
    
    return sentences


def tokenize_sentences(sentences):
    """
    Convert sentences into lists of individual word tokens.
    
    This function processes each sentence by:
    1. Converting to lowercase for case normalization
    2. Using NLTK's word tokenizer to split into words
    3. Handling punctuation as separate tokens
    
    Args:
        sentences (list): List of sentence strings
        
    Returns:
        list: List of lists, where each inner list contains word tokens for one sentence
    """
    # Initialize an empty list to store tokenized sentences
    # Each element will be a list of tokens representing one sentence
    tokenized_sentences = []
    
    # Process each sentence individually
    for sentence in sentences:
        # Convert sentence to lowercase to normalize case
        # This ensures "The" and "the" are treated as the same word
        sentence = sentence.lower()
        
        # Use NLTK's word tokenizer to split sentence into individual words
        # This handles punctuation, contractions, and other word boundaries
        tokenized = nltk.word_tokenize(sentence)
        
        # Add the list of tokens for this sentence to our collection
        tokenized_sentences.append(tokenized)
    
    return tokenized_sentences


def get_tokenized_data(data):
    """
    Complete preprocessing pipeline to convert raw text to tokenized sentences.
    
    This is a convenience function that combines sentence splitting and tokenization
    into a single operation. It's the main entry point for text preprocessing.
    
    Args:
        data (str): Raw text data as a multi-line string
        
    Returns:
        list: List of lists containing tokenized sentences
    """
    # Step 1: Split the raw text into individual sentences
    # This removes empty lines and cleans whitespace
    sentences = split_to_sentences(data)
    
    # Step 2: Tokenize each sentence into individual words
    # This normalizes case and handles punctuation
    tokenized_sentences = tokenize_sentences(sentences)
    
    return tokenized_sentences


def count_words(tokenized_sentences):
    """
    Count the frequency of each word token across all sentences.
    
    This function creates a frequency dictionary by iterating through all
    tokenized sentences and counting how many times each word appears.
    This is essential for vocabulary filtering and rare word handling.
    
    Args:
        tokenized_sentences (list): List of lists containing word tokens
        
    Returns:
        dict: Dictionary mapping each word to its frequency count
    """
    # Initialize an empty dictionary to store word frequencies
    # Key: word token (string), Value: count (integer)
    word_counts = {}
    
    # Iterate through each sentence in the tokenized data
    for sentence in tokenized_sentences:
        # Process each individual token (word) in the current sentence
        for token in sentence:
            # Check if this is the first time we've seen this token
            if token not in word_counts:
                # Initialize count to 1 for new tokens
                word_counts[token] = 1
            else:
                # Increment count for tokens we've seen before
                word_counts[token] += 1
    
    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Filter vocabulary to include only words that appear frequently enough.
    
    This function creates a "closed vocabulary" by keeping only words that
    appear at least 'count_threshold' times in the training data. This helps
    reduce noise from rare words and keeps the model focused on common patterns.
    
    Args:
        tokenized_sentences (list): List of lists containing word tokens
        count_threshold (int): Minimum frequency required to include a word
        
    Returns:
        list: List of words that meet the frequency threshold
    """
    # Initialize empty list to store words that pass the frequency threshold
    # This will become our "closed vocabulary" for training
    closed_vocab = []
    
    # Get frequency counts for all words in the tokenized sentences
    # This uses our count_words function to build the frequency dictionary
    word_counts = count_words(tokenized_sentences)
    
    # Examine each word and its count in the frequency dictionary
    for word, cnt in word_counts.items():
        # Check if this word appears frequently enough to include in vocabulary
        # Only words with count >= count_threshold will be kept
        if cnt >= count_threshold:
            # Add qualifying words to our closed vocabulary list
            closed_vocab.append(word)
    
    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace out-of-vocabulary (OOV) words with unknown tokens.
    
    This function processes tokenized sentences and replaces any words that
    are not in the approved vocabulary with a special unknown token. This
    ensures consistent handling of rare or unseen words during training.
    
    Args:
        tokenized_sentences (list): List of lists containing word tokens
        vocabulary (list): List of approved vocabulary words
        unknown_token (str): Token to use for OOV words (default: "<unk>")
        
    Returns:
        list: List of lists with OOV words replaced by unknown tokens
    """
    # Convert vocabulary list to a set for O(1) lookup time
    # This optimization is crucial when vocabulary is large
    vocabulary = set(vocabulary)

    # Initialize list to store sentences after OOV replacement
    # Each element will be a modified sentence with unknown tokens
    replaced_tokenized_sentences = []

    # Process each sentence individually
    for sentence in tokenized_sentences:
        # Initialize list to store the current sentence after replacements
        # This will contain the same number of tokens as the original sentence
        replaced_sentence = []
        
        # Examine each token in the current sentence
        for token in sentence:
            # Check if the current token exists in our approved vocabulary
            # Set lookup is O(1) average case, much faster than list lookup
            if token in vocabulary:
                # Token is approved - keep it as is
                replaced_sentence.append(token)
            else:
                # Token is out-of-vocabulary - replace with unknown token
                # This handles rare words, typos, and words not in training data
                replaced_sentence.append(unknown_token)

        # Add the processed sentence to our collection of replaced sentences
        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>",
                    get_words_with_nplus_frequency=get_words_with_nplus_frequency,
                    replace_oov_words_by_unk=replace_oov_words_by_unk):
    """
    Complete data preprocessing pipeline for training and testing data.
    
    This function orchestrates the entire preprocessing workflow:
    1. Builds vocabulary from training data based on frequency threshold
    2. Replaces OOV words in both training and test data with unknown tokens
    3. Ensures consistent vocabulary handling between train and test sets
    
    Args:
        train_data (list): Tokenized training sentences
        test_data (list): Tokenized test sentences  
        count_threshold (int): Minimum frequency to include word in vocabulary
        unknown_token (str): Token for OOV words (default: "<unk>")
        get_words_with_nplus_frequency (function): Function to filter vocabulary
        replace_oov_words_by_unk (function): Function to replace OOV words
        
    Returns:
        tuple: (processed_train_data, processed_test_data, vocabulary)
    """
    # Step 1: Build vocabulary from training data only
    # We use only training data to avoid data leakage from test set
    # Only words appearing >= count_threshold times are included
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # Step 2: Apply vocabulary filtering to training data
    # Replace words not in vocabulary with unknown tokens
    # This creates consistent training data with controlled vocabulary size
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)

    # Step 3: Apply same vocabulary filtering to test data
    # Use the same vocabulary built from training data to ensure consistency
    # Test data may have more OOV words since vocabulary is based on training
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)

    # Return all three components needed for model training and evaluation
    return train_data_replaced, test_data_replaced, vocabulary


