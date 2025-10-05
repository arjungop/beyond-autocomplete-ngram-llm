"""
N-gram Language Model Implementation

This module implements a complete N-gram language model for autocomplete and text
generation tasks. It includes functions for counting n-grams, calculating probabilities
with smoothing, estimating perplexity, and generating word suggestions.
"""

import numpy as np  # For numerical operations and matrix calculations
import pandas as pd  # For creating readable count matrices and data manipulation


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count occurrences of n-grams in the training data.
    
    This function processes tokenized sentences to count how many times each
    n-gram (sequence of n words) appears. It adds start and end tokens to
    handle sentence boundaries properly.
    
    Args:
        data (list): List of tokenized sentences (each sentence is a list of words)
        n (int): Order of n-grams to count (e.g., 2 for bigrams, 3 for trigrams)
        start_token (str): Token to mark sentence beginnings
        end_token (str): Token to mark sentence endings
        
    Returns:
        dict: Dictionary mapping n-gram tuples to their occurrence counts
    """
    # Initialize empty dictionary to store n-gram counts
    # Key: tuple of n words, Value: integer count
    n_grams = {}
    
    # Process each sentence in the training data
    for sentence in data:
        # Add n start tokens to the beginning of each sentence
        # This allows the model to predict the first n words of sentences
        # Add one end token to mark where sentences finish
        sentence = [start_token] * n + sentence + [end_token]
        
        # Convert list to tuple so it can be used as dictionary key
        # Lists are mutable and can't be dictionary keys, tuples are immutable
        sentence = tuple(sentence)
        
        # Extract all possible n-grams from this sentence using sliding window
        # Range goes from 0 to (sentence_length - n + 1) to ensure valid n-grams
        for i in range(len(sentence) - n + 1):
            # Extract n consecutive words starting at position i
            # This creates the n-gram tuple from position i to i+n-1
            n_gram = sentence[i:i + n]
            
            # Update count for this n-gram
            if n_gram in n_grams.keys():
                # Increment count if we've seen this n-gram before
                n_grams[n_gram] += 1
            else:
                # Initialize count to 1 for new n-grams
                n_grams[n_gram] = 1
    
    return n_grams


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probability of a word given the previous n-gram context.
    
    This function implements Add-k smoothing (Laplace smoothing) to handle
    unseen n-grams and provide robust probability estimates. The probability
    is calculated as: P(word|context) = (count(context+word) + k) / (count(context) + k*V)
    
    Args:
        word (str): The word whose probability we want to estimate
        previous_n_gram (list): The n-gram context (previous n words)
        n_gram_counts (dict): Dictionary of n-gram counts
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary_size (int): Total size of vocabulary for smoothing
        k (float): Smoothing parameter (default: 1.0 for Laplace smoothing)
        
    Returns:
        float: Probability of the word given the context
    """
    # Convert list to tuple for use as dictionary key
    # Dictionary keys must be immutable, so we use tuples
    previous_n_gram = tuple(previous_n_gram)

    # Get count of the context n-gram from the counts dictionary
    # If the n-gram doesn't exist, default to 0 count
    # This handles unseen contexts gracefully
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)

    # Calculate denominator using Add-k smoothing formula
    # Add k times vocabulary size to account for all possible next words
    # This ensures probabilities sum to 1 and handles unseen combinations
    denominator = previous_n_gram_count + k * vocabulary_size

    # Create the (n+1)-gram by combining context with the target word
    # This represents the full sequence we're calculating probability for
    n_plus1_gram = previous_n_gram + (word,)

    # Get count of the complete (n+1)-gram from the counts dictionary
    # If this specific sequence wasn't seen, default to 0 count
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)

    # Calculate numerator using Add-k smoothing
    # Add k to the observed count to ensure non-zero probabilities
    numerator = n_plus1_gram_count + k

    # Calculate final probability as smoothed count ratio
    # This gives us P(word|previous_n_gram) with smoothing applied
    probability = numerator / denominator

    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>',
                           unknown_token="<unk>", k=1.0):
    """
    Calculate probabilities for all possible next words given a context.
    
    This function computes probability distributions over the entire vocabulary
    for a given n-gram context. It's essential for autocomplete systems where
    we need to consider multiple possible next words.
    
    Args:
        previous_n_gram (list): The n-gram context (previous n words)
        n_gram_counts (dict): Dictionary of n-gram counts
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary (list): List of vocabulary words
        end_token (str): Token marking sentence end
        unknown_token (str): Token for out-of-vocabulary words
        k (float): Smoothing parameter
        
    Returns:
        dict: Dictionary mapping each word to its probability given the context
    """
    # Convert list to tuple for dictionary key usage
    # Ensures we can use the n-gram as a hash key
    previous_n_gram = tuple(previous_n_gram)

    # Extend vocabulary with special tokens for complete probability distribution
    # Include end token for sentence completion and unknown token for OOV words
    # Note: start token <s> is excluded since it should never be predicted as next word
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    
    # Initialize dictionary to store probability for each possible next word
    probabilities = {}
    
    # Calculate probability for each word in the extended vocabulary
    for word in vocabulary:
        # Use the estimate_probability function to get P(word|previous_n_gram)
        # This applies Add-k smoothing and handles unseen combinations
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        # Store the probability for this word
        probabilities[word] = probability
    
    return probabilities


def make_count_matrix(n_plus1_gram_counts, vocabulary):
    """
    Create a matrix representation of n-gram counts for visualization and analysis.
    
    This function converts the n-gram count dictionary into a pandas DataFrame
    where rows represent n-gram contexts and columns represent possible next words.
    This matrix format is useful for analysis and debugging.
    
    Args:
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary (list): List of vocabulary words
        
    Returns:
        pandas.DataFrame: Count matrix with n-grams as rows and words as columns
    """
    # Add special tokens to vocabulary for complete representation
    # End token and unknown token are needed for sentence boundaries and OOV words
    # Start token is omitted since it should not appear as a next word prediction
    vocabulary = vocabulary + ["<e>", "<unk>"]

    # Extract unique n-gram contexts from the (n+1)-gram keys
    # Each (n+1)-gram consists of: (context_word_1, ..., context_word_n, next_word)
    # We want just the context part: (context_word_1, ..., context_word_n)
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        # Take all elements except the last one (which is the next word)
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    
    # Remove duplicates by converting to set and back to list
    # This gives us all unique n-gram contexts that appeared in training
    n_grams = list(set(n_grams))

    # Create mapping from n-gram contexts to matrix row indices
    # This allows efficient lookup when populating the matrix
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    
    # Create mapping from vocabulary words to matrix column indices
    # This allows efficient lookup when populating the matrix
    col_index = {word: j for j, word in enumerate(vocabulary)}

    # Get matrix dimensions
    nrow = len(n_grams)  # Number of unique n-gram contexts
    ncol = len(vocabulary)  # Number of possible next words
    
    # Initialize count matrix with zeros
    # Matrix[i,j] will contain count of word j following n-gram i
    count_matrix = np.zeros((nrow, ncol))
    
    # Populate the count matrix with actual counts from the dictionary
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        # Split the (n+1)-gram into context and next word
        n_gram = n_plus1_gram[0:-1]  # Context part
        word = n_plus1_gram[-1]      # Next word part
        
        # Skip words not in our vocabulary (shouldn't happen with proper preprocessing)
        if word not in vocabulary:
            continue
            
        # Get matrix coordinates for this n-gram and word combination
        i = row_index[n_gram]  # Row index for the context
        j = col_index[word]    # Column index for the next word
        
        # Set the count in the matrix
        count_matrix[i, j] = count

    # Convert numpy array to pandas DataFrame for better readability
    # Use n-grams as row labels and vocabulary words as column labels
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    
    return count_matrix


def make_probability_matrix(n_plus1_gram_counts, vocabulary, k, unique_words=None):
    """
    Create a probability matrix from count matrix using Add-k smoothing.
    
    This function converts raw counts into probability distributions by applying
    smoothing and normalizing each row to sum to 1. Each row represents the
    probability distribution over next words for a given n-gram context.
    
    Args:
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary (list): List of vocabulary words
        k (float): Smoothing parameter for Add-k smoothing
        unique_words (list): Optional list of words to use instead of vocabulary
        
    Returns:
        pandas.DataFrame: Probability matrix with normalized rows
    """
    # Create count matrix from the n-gram counts
    # This gives us raw frequency counts for each context-word combination
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    
    # Apply Add-k smoothing by adding k to all counts
    # This ensures no probability is zero and handles unseen combinations
    count_matrix += k
    
    # Normalize each row to create probability distributions
    # Each row sums to 1, representing P(next_word | n_gram_context)
    # axis=1 means we divide each row by its sum, axis=0 specifies row-wise operation
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    
    return prob_matrix


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>',
                         end_token='<e>', k=1.0):
    """
    Calculate perplexity of a sentence using the n-gram language model.
    
    Perplexity measures how well the model predicts the sentence - lower values
    indicate better prediction. It's calculated as the geometric mean of the
    inverse probabilities: perplexity = (1/P(sentence))^(1/N)
    
    Args:
        sentence (list): Tokenized sentence to evaluate
        n_gram_counts (dict): Dictionary of n-gram counts
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary_size (int): Size of vocabulary for smoothing
        start_token (str): Token marking sentence beginning
        end_token (str): Token marking sentence end
        k (float): Smoothing parameter
        
    Returns:
        float: Perplexity score (lower is better)
    """
    # Determine n-gram order from the length of keys in n_gram_counts
    # All keys should have the same length, so we take the first one
    n = len(list(n_gram_counts.keys())[0])

    # Add sentence boundary tokens for proper probability calculation
    # Prepend n start tokens to handle the beginning of the sentence
    # Append one end token to handle the end of the sentence
    sentence = [start_token] * n + sentence + [end_token]

    # Convert list to tuple for consistent processing
    # This matches the format used in our n-gram dictionaries
    sentence = tuple(sentence)

    # Get total length of the augmented sentence
    # This includes the added start and end tokens
    N = len(sentence)

    # Initialize the product of inverse probabilities
    # We multiply 1/P(word_i | context_i) for each word in the sentence
    # This will be used to calculate the geometric mean
    product_pi = 1.0

    # Calculate inverse probability for each word position
    # Start from position n (first real word after start tokens)
    # Go up to N-1 (the end token position)
    for t in range(n, N):
        # Extract n-gram context preceding the current word
        # This gives us the previous n words that condition the current word
        n_gram = sentence[t - n:t]

        # Get the current word we're trying to predict
        word = sentence[t]

        # Calculate P(word | n_gram) using our probability estimation function
        # This applies Add-k smoothing and handles unseen combinations
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)

        # Multiply by the inverse probability: 1/P(word | context)
        # We accumulate the product of all inverse probabilities
        product_pi *= 1 / probability

    # Calculate perplexity as the N-th root of the product
    # This gives us the geometric mean of the inverse probabilities
    # Formula: perplexity = (∏(1/P_i))^(1/N)
    perplexity = product_pi ** (1 / N)

    return perplexity


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    Suggest the single most probable next word given previous tokens.
    
    This function finds the word with the highest probability of following
    the given context. It optionally filters suggestions by a prefix string
    for autocomplete functionality.
    
    Args:
        previous_tokens (list): List of previous words in the context
        n_gram_counts (dict): Dictionary of n-gram counts
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram counts
        vocabulary (list): List of vocabulary words
        k (float): Smoothing parameter
        start_with (str): Optional prefix to filter suggestions
        
    Returns:
        tuple: (suggested_word, probability) - the best suggestion and its probability
    """
    # Determine n-gram order from the dictionary keys
    # All keys should have the same length (the n-gram order)
    n = len(list(n_gram_counts.keys())[0])

    # Extract the most recent n words as context for prediction
    # If user typed fewer than n words, use what's available
    # This creates the n-gram context for probability calculation
    previous_n_gram = previous_tokens[-n:]

    # Calculate probability distribution over all possible next words
    # This gives us P(word | previous_n_gram) for every word in vocabulary
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # Initialize variables to track the best suggestion
    suggestion = None      # Will store the word with highest probability
    max_prob = 0          # Will store the highest probability found

    # Examine each word and its probability to find the best suggestion
    for word, prob in probabilities.items():
        
        # Apply prefix filtering if start_with parameter is provided
        # This is useful for autocomplete when user has typed partial word
        if start_with is not None:
            # Check if current word doesn't start with the required prefix
            if not word.startswith(start_with):
                # Skip this word and continue to the next one
                continue

        # Check if this word has higher probability than current best
        if prob > max_prob:
            # Update our best suggestion with this word
            suggestion = word
            # Update the maximum probability
            max_prob = prob

    # Return the best word and its probability
    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    """
    Get word suggestions from multiple n-gram models of different orders.
    
    This function generates suggestions using n-gram models of different orders
    (unigram, bigram, trigram, etc.) to provide diverse autocomplete options.
    Each model contributes one suggestion based on different context lengths.
    
    Args:
        previous_tokens (list): List of previous words in the context
        n_gram_counts_list (list): List of n-gram count dictionaries for different orders
        vocabulary (list): List of vocabulary words
        k (float): Smoothing parameter
        start_with (str): Optional prefix to filter suggestions
        
    Returns:
        list: List of (word, probability) tuples from different n-gram orders
    """
    # Get the number of different n-gram models available
    # This is typically [unigram_counts, bigram_counts, trigram_counts, ...]
    model_counts = len(n_gram_counts_list)
    
    # Initialize list to store suggestions from different models
    suggestions = []
    
    # Generate suggestions from each consecutive pair of n-gram models
    # We use pairs because we need both n-gram and (n+1)-gram counts for probability calculation
    for i in range(model_counts - 1):
        # Get n-gram counts (context counts)
        n_gram_counts = n_gram_counts_list[i]
        
        # Get (n+1)-gram counts (context + next word counts)
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        # Generate suggestion using this specific n-gram order
        # This gives us the best word prediction for this context length
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        
        # Add this model's suggestion to our collection
        suggestions.append(suggestion)
    
    return suggestions


class LanguageModel:
    """
    N-gram Language Model class for autocomplete and text generation evaluation.
    
    This class encapsulates a complete n-gram language model with training,
    prediction, and evaluation capabilities. It supports multiple n-gram orders
    and provides both single-word suggestions and perplexity calculations.
    """
    
    def __init__(self, n_max=4, k=1.0):
        """
        Initialize the language model with specified parameters.
        
        Args:
            n_max (int): Maximum n-gram order to use (default: 4 for up to 4-grams)
            k (float): Smoothing parameter for Add-k smoothing (default: 1.0)
        """
        # Store model configuration parameters
        self.n_max = n_max  # Maximum n-gram order (1=unigram, 2=bigram, etc.)
        self.k = k          # Smoothing parameter for handling unseen n-grams
        
        # Initialize model state variables
        self.vocabulary = None          # Will store the vocabulary after training
        self.n_gram_counts_list = []   # Will store count dictionaries for each n-gram order
        self.vocabulary_size = 0       # Will store vocabulary size for smoothing calculations
    def fit(self, training_data, vocabulary):
        """
        Train the language model on the given training data.
        
        This method builds n-gram count dictionaries for all orders from 1 to n_max
        using the provided training data. It processes the data to count all n-gram
        occurrences and stores them for later probability calculations.
        
        Args:
            training_data (list): List of tokenized sentences for training
            vocabulary (list): Vocabulary words to use for the model
        """
        # Store the vocabulary and calculate its size
        # Vocabulary size is needed for Add-k smoothing calculations
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        
        # Build n-gram count dictionaries for all orders from 1 to n_max
        # This creates unigram, bigram, trigram, ... up to n_max-gram counts
        self.n_gram_counts_list = []
        
        # Process each n-gram order sequentially
        for n in range(1, self.n_max + 1):
            # Count all n-grams of order 'n' in the training data
            # This uses the count_n_grams function to process all sentences
            n_gram_counts = count_n_grams(training_data, n)
            
            # Store the counts for this n-gram order
            self.n_gram_counts_list.append(n_gram_counts)
    def get_user_input_suggestions(self, previous_tokens, num_suggestions=5, start_with=None):
        """
        Get ranked word suggestions for autocomplete given previous tokens.
        
        This method provides the main autocomplete functionality by finding the most
        probable next words given the user's input context. It tries different
        context lengths and returns the top suggestions ranked by probability.
        
        Args:
            previous_tokens (list): List of previous words typed by user
            num_suggestions (int): Maximum number of suggestions to return
            start_with (str): Optional prefix for filtering suggestions (for partial word completion)
            
        Returns:
            list: List of (word, probability) tuples sorted by probability (highest first)
        """
        # Check if model has been trained
        if not self.n_gram_counts_list or not self.vocabulary:
            # Return empty list if model hasn't been trained yet
            return []
        
        # Initialize dictionary to collect suggestions from different context lengths
        # Key: word, Value: highest probability found for that word
        all_suggestions = {}
        
        # Try different context lengths from longest to shortest
        # Longer contexts give more specific predictions but may have less data
        # We start with the longest available context and work down
        for context_len in range(min(len(previous_tokens), self.n_max - 1), 0, -1):
            # Skip if context length exceeds available tokens
            if context_len > len(previous_tokens):
                continue
                
            # Extract the most recent 'context_len' words as context
            # This gives us the n-gram context for making predictions
            context = previous_tokens[-context_len:]
            
            # Check if we have the required n-gram counts for this context length
            # We need both n-gram and (n+1)-gram counts for probability calculation
            if context_len < len(self.n_gram_counts_list):
                # Get n-gram counts (for context frequency)
                n_gram_counts = self.n_gram_counts_list[context_len - 1]
                # Get (n+1)-gram counts (for context+word frequency)
                n_plus1_gram_counts = self.n_gram_counts_list[context_len]
                
                # Calculate probability distribution over all possible next words
                # This gives P(word | context) for every word in vocabulary
                probabilities = estimate_probabilities(
                    context, n_gram_counts, n_plus1_gram_counts, 
                    self.vocabulary, k=self.k
                )
                
                # Filter suggestions and keep track of best probabilities
                for word, prob in probabilities.items():
                    # Apply prefix filter if specified (for autocomplete of partial words)
                    if start_with is None or word.startswith(start_with):
                        # Keep the maximum probability if word appears in multiple contexts
                        # This gives preference to the most confident prediction
                        if word not in all_suggestions or prob > all_suggestions[word]:
                            all_suggestions[word] = prob
        
        # Sort all suggestions by probability in descending order
        # This puts the most probable words first for better user experience
        sorted_suggestions = sorted(all_suggestions.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        # Return only the top num_suggestions
        return sorted_suggestions[:num_suggestions]
    def calculate_perplexity(self, sentence):
        """
        Calculate perplexity of a sentence using the trained language model.
        
        Perplexity measures how well the model predicts the given sentence.
        Lower perplexity values indicate better model performance. This is
        a key evaluation metric for language models.
        
        Args:
            sentence (str or list): Input sentence as string or list of tokens
            
        Returns:
            float: Perplexity score (lower values indicate better model fit)
        """
        # Handle both string and token list inputs
        if isinstance(sentence, str):
            # If input is a string, tokenize it using NLTK
            import nltk
            # Convert to lowercase and tokenize to match training data format
            tokens = nltk.word_tokenize(sentence.lower())
        else:
            # If input is already tokenized, use as-is
            tokens = sentence
        
        # Check if model has been trained and has sufficient n-gram orders
        if not self.n_gram_counts_list or len(self.n_gram_counts_list) < 2:
            # Return infinite perplexity if model isn't properly trained
            return float('inf')
        
        # Use the highest order n-gram model available for perplexity calculation
        # We need both n-gram and (n+1)-gram counts, so use the last two count dictionaries
        n_gram_counts = self.n_gram_counts_list[-2]     # n-gram counts (context)
        n_plus1_gram_counts = self.n_gram_counts_list[-1]  # (n+1)-gram counts (context+word)
        
        # Calculate perplexity using the global calculate_perplexity function
        # This applies Add-k smoothing and handles sentence boundary tokens
        return calculate_perplexity(
            tokens, n_gram_counts, n_plus1_gram_counts, 
            self.vocabulary_size, k=self.k
        )
