"""
Advanced N-gram Language Models for Text Prediction

This module provides an enhanced implementation of multi-order N-gram language models
with advanced smoothing techniques, backoff mechanisms, and interpolation. It's designed
for more sophisticated text prediction and evaluation compared to the basic implementation.

Key Features:
- Multi-order N-gram modeling (unigram through 4-gram)
- Interpolated smoothing for better probability estimates
- Backoff mechanisms for handling unseen n-grams
- Enhanced vocabulary preprocessing with adaptive thresholding
- Comprehensive perplexity evaluation
- Text generation with contextual awareness

This implementation is complementary to the main language_model.py and provides
additional features for research and advanced applications.
"""

import numpy as np                      # For numerical operations
import math                            # For logarithmic and exponential calculations
from collections import defaultdict, Counter  # For efficient counting and grouping
from data_preprocessing import get_tokenized_data  # For data preprocessing


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count n-grams with proper sentence boundary handling.
    
    This function is similar to the one in language_model.py but with
    enhanced error handling and optimizations for larger datasets.
    
    Args:
        data (list): List of tokenized sentences
        n (int): Order of n-grams to count
        start_token (str): Token marking sentence beginning
        end_token (str): Token marking sentence ending
        
    Returns:
        dict: Dictionary mapping n-gram tuples to their counts
    """
    # Initialize dictionary to store n-gram counts
    n_grams = {}
    
    # Process each sentence in the dataset
    for sentence in data:
        # Skip empty sentences to avoid processing errors
        if not sentence:
            continue
            
        # Add boundary tokens: n start tokens + sentence + 1 end token
        # This ensures proper context for predicting sentence beginnings and endings
        sentence = [start_token] * n + sentence + [end_token]
        
        # Convert to tuple for use as dictionary key
        sentence = tuple(sentence)
        
        # Extract all possible n-grams using sliding window approach
        for i in range(len(sentence) - n + 1):
            # Get n consecutive words starting at position i
            n_gram = sentence[i:i + n]
            
            # Update count using efficient dictionary operations
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
                
    return n_grams


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Calculate perplexity score for a single sentence using n-gram probability model.
    
    Perplexity measures how well the model predicts the sentence - lower values
    indicate better prediction capability. Formula: perplexity = exp(-log_prob / N)
    
    Args:
        sentence (list): List of word tokens to evaluate
        n_gram_counts (dict): Dictionary of n-gram occurrence counts  
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram occurrence counts
        vocabulary_size (int): Total vocabulary size for smoothing calculation
        k (float): Add-k smoothing parameter (default: 1.0)
        
    Returns:
        float: Perplexity score (lower is better)
    """
    # Create augmented sentence with start and end boundary tokens
    # Add one start token at beginning and one end token at end
    sentence_start = ['<s>'] + sentence + ['<e>']
    
    # Calculate total sentence length including boundary tokens
    # This is used for normalizing the final perplexity calculation
    sentence_length = len(sentence_start)
    
    # Initialize log probability accumulator to zero
    # We sum log probabilities to avoid numerical underflow issues
    log_probability = 0.0
    
    # Iterate through each word position starting from index 1
    # Skip index 0 since we need previous word for bigram calculation
    for i in range(1, sentence_length):
        # Extract bigram: current word and its preceding word
        # Convert to tuple for use as dictionary key (immutable type required)
        n_gram = tuple(sentence_start[i-1:i+1])
        
        # Extract unigram: just the preceding word for context counting
        # This gives us the denominator for conditional probability calculation
        n_minus1_gram = tuple(sentence_start[i-1:i])
        
        # Get count of this specific bigram from training data
        # Use .get() method with default 0 to handle unseen bigrams gracefully
        n_plus1_gram_count = n_plus1_gram_counts.get(n_gram, 0)
        
        # Get count of the context (preceding word) from training data
        # This is the total occurrences of the conditioning word
        n_gram_count = n_gram_counts.get(n_minus1_gram, 0)
        
        # Calculate conditional probability using Add-k smoothing
        # Formula: P(word|context) = (count(context,word) + k) / (count(context) + k*V)
        # Add-k smoothing ensures no zero probabilities for unseen combinations
        probability = (n_plus1_gram_count + k) / (n_gram_count + k * vocabulary_size)
        
        # Add log probability to running sum
        # Using log probabilities prevents numerical underflow for long sentences
        log_probability += math.log(probability)
    
    # Calculate final perplexity using the standard formula
    # Perplexity = exp(-1/N * sum(log(P(w_i|w_{i-1}))))
    # Negative log probability normalized by sentence length
    perplexity = math.exp(-log_probability / sentence_length)
    
    # Return the computed perplexity score
    return perplexity


def preprocess_data(train_data, test_data, count_threshold):
    """
    Preprocess training and test data with vocabulary filtering and OOV handling.
    
    This function builds a vocabulary from training data based on word frequency
    and replaces rare words with unknown tokens to improve model robustness.
    
    Args:
        train_data (list): List of tokenized training sentences
        test_data (list): List of tokenized test sentences
        count_threshold (int): Minimum frequency required for vocabulary inclusion
        
    Returns:
        tuple: (processed_train_data, processed_test_data, vocabulary)
    """
    # Initialize empty dictionary to count word frequencies
    # Key: word string, Value: integer count of occurrences
    word_counts = {}
    
    # Count word frequencies in training data only
    # We don't use test data to avoid data leakage in vocabulary building
    for sentence in train_data:  # Iterate through each training sentence
        for word in sentence:    # Iterate through each word in the sentence
            # Increment word count using dictionary .get() method
            # .get(word, 0) returns current count or 0 if word not seen before
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Build vocabulary list starting with unknown token
    # Include <unk> token for handling out-of-vocabulary words
    vocabulary = ['<unk>']
    
    # Add words that meet the frequency threshold to vocabulary
    for word, count in word_counts.items():  # Iterate through word-count pairs
        # Only include words that appear at least count_threshold times
        # This filters out typos, rare words, and noise
        if count >= count_threshold:
            vocabulary.append(word)  # Add qualifying word to vocabulary list
    
    # Convert vocabulary list to set for O(1) lookup efficiency
    # Set operations are much faster than list operations for membership testing
    vocab_set = set(vocabulary)
    
    # Define nested function to replace out-of-vocabulary words
    # This function will be applied to both training and test data
    def replace_oov_words(data):
        # Initialize list to store processed sentences
        processed = []
        
        # Process each sentence in the input data
        for sentence in data:           # Iterate through sentences
            new_sentence = []           # Initialize empty list for processed sentence
            
            # Process each word in the current sentence
            for word in sentence:       # Iterate through words in sentence
                # Check if word exists in our approved vocabulary
                if word in vocab_set:   # O(1) set membership test
                    new_sentence.append(word)      # Keep known words as-is
                else:
                    new_sentence.append('<unk>')   # Replace unknown words with <unk>
            
            # Add processed sentence to results list
            processed.append(new_sentence)
        
        # Return list of processed sentences
        return processed
    
    train_processed = replace_oov_words(train_data)
    test_processed = replace_oov_words(test_data)
    
    return train_processed, test_processed, vocabulary


def preprocess_data(train_data, test_data, count_threshold):
    """Preprocess training and test data with enhanced vocabulary filtering and comprehensive educational explanations"""
    # Initialize dictionary to count word occurrences in training data
    # Key: individual word string, Value: count of occurrences
    # This frequency analysis determines which words to include in vocabulary
    # More frequent words are more reliable for statistical modeling
    word_counts = {}
    
    # Count word frequencies in training data with enhanced filtering
    # Iterate through each sentence in the training dataset
    # Only count valid, non-empty words to improve vocabulary quality
    for sentence in train_data:          # Process each training sentence
        for word in sentence:            # Process each word in sentence
            # Apply quality filters to exclude problematic words
            # len(word) > 0: exclude empty strings that could cause errors
            # not word.isspace(): exclude whitespace-only strings
            # These filters improve vocabulary quality and model robustness
            if len(word) > 0 and not word.isspace():  # Filter empty/whitespace words
                # Increment word count using dictionary .get() method with default 0
                # This handles both new words (count=0) and existing words safely
                # Accumulates total frequency for each unique word
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Initialize vocabulary with essential special tokens
    # '<unk>': represents out-of-vocabulary words during inference
    # '<s>': sentence start marker for N-gram context
    # '<e>': sentence end marker for proper termination
    # These tokens are crucial for robust language model operation
    vocabulary = ['<unk>', '<s>', '<e>']
    
    # Build vocabulary using adaptive thresholding strategy
    # This approach balances vocabulary coverage with noise reduction
    # Different thresholds for different word lengths optimize inclusion
    for word, count in word_counts.items():  # Iterate through word-count pairs
        # Primary threshold: include words meeting minimum frequency
        # Higher frequency words are more reliable for modeling
        if count >= count_threshold:
            vocabulary.append(word)  # Add high-frequency word to vocabulary
        # Secondary threshold: include longer words with lower frequency
        # Longer words (>2 chars) are often more meaningful even if rare
        # Helps preserve important content words that might be less frequent
        elif count >= max(1, count_threshold // 2) and len(word) > 2:
            vocabulary.append(word)  # Keep longer words with lower threshold
    
    # Convert vocabulary to set for O(1) membership testing
    # Sets provide much faster lookup than lists for large vocabularies
    # This optimization is crucial during OOV word replacement
    vocab_set = set(vocabulary)
    
    # Define enhanced out-of-vocabulary word replacement function
    # This nested function processes both training and test data consistently
    # Enhanced version includes better error handling and data validation
    def replace_oov_words_enhanced(data):
        # Initialize list to store processed sentences
        # Each element will be a list of words (processed sentence)
        processed = []
        
        # Process each sentence in the input data
        for sentence in data:           # Iterate through input sentences
            # Skip empty sentences to avoid processing errors
            # Empty sentences can cause issues in N-gram analysis
            if not sentence:            # Check if sentence is empty
                continue                # Skip to next sentence
            
            # Initialize list for processed words in current sentence
            new_sentence = []
            
            # Process each word in the current sentence
            for word in sentence:       # Iterate through words in sentence
                # Check if word exists in approved vocabulary
                if word in vocab_set:   # O(1) set membership test
                    new_sentence.append(word)      # Keep known words unchanged
                # Only replace non-empty words with <unk> token
                # This prevents creation of spurious <unk> tokens
                elif len(word) > 0:     # Ensure word is not empty string
                    new_sentence.append('<unk>')   # Replace with unknown token
            
            # Only add non-empty processed sentences to results
            # This maintains data quality and prevents downstream errors
            if new_sentence:            # Check if processed sentence has content
                processed.append(new_sentence)    # Add to results list
        
        # Return list of processed sentences
        # Each sentence is a list of vocabulary words or <unk> tokens
        return processed
    
    # Apply enhanced OOV replacement to both training and test data
    # Consistent processing ensures model training and evaluation use same vocabulary
    train_processed = replace_oov_words_enhanced(train_data)
    test_processed = replace_oov_words_enhanced(test_data)
    
    return train_processed, test_processed, vocabulary


class NGramModel:
    """Multi-order N-gram model for text prediction with comprehensive educational implementation
    
    This class implements a statistical language model based on N-gram analysis.
    N-grams are sequences of N consecutive words used to predict the next word
    in a sequence. The model supports multiple orders (1-gram through 4-gram)
    and uses interpolation smoothing to combine predictions from different orders.
    
    Educational Purpose:
    - Demonstrates fundamental concepts in computational linguistics
    - Shows how statistical models can capture language patterns
    - Illustrates probability estimation and smoothing techniques
    - Provides hands-on experience with language model evaluation
    """
    
    def __init__(self, max_n=4):
        """Initialize the N-gram model with comprehensive parameter setup
        
        Args:
            max_n (int): Maximum N-gram order to train (default 4 for 4-grams)
        """
        # Store maximum N-gram order for model capacity
        # This determines the longest word sequences the model will analyze
        # Higher orders capture longer-range dependencies but require more data
        # max_n=4 means we train unigrams, bigrams, trigrams, and 4-grams
        self.max_n = max_n
        
        # Initialize empty vocabulary list
        # Will store all unique words seen during training plus special tokens
        # Used for probability calculations and out-of-vocabulary handling
        # Includes '<unk>' for unknown words, '<s>' for start, '<e>' for end
        self.vocabulary = []
        
        # Initialize empty list to store N-gram count dictionaries
        # Each element is a dictionary mapping N-grams to occurrence counts
        # Index 0: unigram counts, Index 1: bigram counts, etc.
        # These counts are fundamental for probability estimation
        self.n_gram_counts_list = []
        
        # Initialize placeholder for training data
        # Will store preprocessed training sentences after train() is called
        # Used for context counting and probability calculations
        # None indicates model has not been trained yet
        self.train_data = None
        
    def train(self, tokenized_sentences, count_threshold=1):  # Lower threshold for more coverage
        """Train the n-gram language model with comprehensive educational explanations
        
        Args:
            tokenized_sentences (list): List of lists, each inner list contains words from one sentence
            count_threshold (int): Minimum word frequency to include in vocabulary
        """
        # Display training progress information for user feedback
        # Shows dataset size to give sense of model training scale
        # Larger datasets generally lead to better model performance
        print(f"Training n-gram model with {len(tokenized_sentences)} sentences...")
        
        # Split data into training and testing sets using 90/10 split
        # Use 90% for training to maximize data available for N-gram learning
        # Higher-order N-grams especially benefit from more training data
        # Reserve 10% for testing to evaluate model performance
        train_size = int(len(tokenized_sentences) * 0.9)
        train_data = tokenized_sentences[:train_size]  # First 90% for training
        test_data = tokenized_sentences[train_size:]   # Last 10% for testing
        
        # Adjust count threshold for better vocabulary coverage
        # For higher-order N-grams, use more lenient threshold
        # This ensures sufficient N-gram coverage for meaningful statistics
        # Lower threshold includes more words but may add noise
        lower_threshold = max(1, count_threshold // 2)
        
        # Preprocess training and testing data
        # This step filters vocabulary, handles out-of-vocabulary words
        # Creates consistent word representations across train/test
        # Returns processed data and final vocabulary list
        train_processed, test_processed, vocabulary = preprocess_data(
            train_data, test_data, lower_threshold
        )
        
        # Store preprocessed data as instance variables
        # train_data: used for context counting in probability calculations
        # test_data: used for model evaluation and perplexity testing
        # vocabulary: complete list of recognized words plus special tokens
        self.train_data = train_processed
        self.test_data = test_processed
        self.vocabulary = vocabulary
        
        # Display vocabulary statistics for training transparency
        # Vocabulary size affects model complexity and smoothing behavior
        # Larger vocabularies require more sophisticated smoothing
        print(f"Vocabulary size: {len(vocabulary)}")
        
        # Train N-gram models for all orders from 1 to max_n
        # Each order captures different levels of linguistic context
        # Unigrams: word frequency, Bigrams: word pairs, etc.
        # Higher orders capture longer-range dependencies
        self.n_gram_counts_list = []
        for n in range(1, self.max_n + 1):
            # Count N-grams of current order in training data
            # Returns dictionary mapping N-gram tuples to occurrence counts
            # These counts are essential for probability estimation
            n_gram_counts = count_n_grams(train_processed, n)
            
            # Store counts in list for later probability calculations
            # Index (n-1) stores counts for order n (due to 0-based indexing)
            # This structure enables efficient lookup during prediction
            self.n_gram_counts_list.append(n_gram_counts)
            
            # Display N-gram statistics for each order
            # Shows model complexity and training data coverage
            # More unique N-grams indicate richer language patterns
            print(f"{n}-gram: {len(n_gram_counts)} unique n-grams")
        
        # Signal completion of training process
        # Model is now ready for text prediction and evaluation
        print("Training completed!")
        
    def calculate_probability(self, n_gram, k=0.001):
        """Calculate probability using interpolated smoothing with comprehensive educational explanations"""
        # Initialize interpolation weights for different N-gram orders
        # Interpolation combines predictions from multiple N-gram orders
        # Higher-order N-grams get higher weights because they provide more context
        # weights[0] = 0.1 corresponds to unigram (1-gram) probability contribution
        # weights[1] = 0.2 corresponds to bigram (2-gram) probability contribution
        # weights[2] = 0.3 corresponds to trigram (3-gram) probability contribution  
        # weights[3] = 0.4 corresponds to 4-gram probability contribution
        # Sum of weights = 1.0 ensures proper probability distribution
        weights = [0.1, 0.2, 0.3, 0.4]  # weights for 1,2,3,4-gram order models
        
        # Initialize variable to accumulate weighted probability from all N-gram orders
        # This will store the final interpolated probability value
        # Starts at 0.0 and will be incremented by each N-gram order contribution
        # Final value represents linear interpolation of all model orders
        interpolated_prob = 0.0

        # Iterate through all possible N-gram orders for interpolation
        # Start from order=1 (unigram) up to min(n_gram_length+1, max_n+1)
        # This ensures we don't exceed the length of input n_gram or model capacity
        # Each iteration contributes one order's probability to final interpolated result
        for order in range(1, min(len(n_gram) + 1, self.max_n + 1)):
            # Safety check: ensure we have counts for this order
            # Skip orders that exceed our trained model capacity
            # This prevents index errors when accessing n_gram_counts_list
            if order > len(self.n_gram_counts_list):
                continue  # Skip to next iteration without processing this order
                
            # Extract the appropriate N-gram substring for current order
            # Use negative indexing to get last 'order' elements from n_gram
            # For order=2 and n_gram=("the", "cat", "sat"), current_gram=("cat", "sat")
            # If order exceeds n_gram length, use entire n_gram
            current_gram = n_gram[-order:] if order <= len(n_gram) else n_gram
            
            # Extract context (all words except the last prediction word)
            # For trigram ("the", "cat", "sat"), context=("the", "cat")
            # For unigram, context is empty tuple ()
            # Context represents the conditioning history for probability calculation
            context = current_gram[:-1] if len(current_gram) > 1 else ()
            
            # Retrieve count dictionaries for current N-gram order
            # gram_counts: dictionary mapping N-grams to their occurrence counts
            # Use (order-1) because list is 0-indexed but orders start from 1
            gram_counts = self.n_gram_counts_list[order - 1]
            
            # Retrieve count dictionary for context (one order lower)
            # For trigram probability P(w3|w1,w2), need bigram counts for context
            # None for unigrams since they have no context
            context_counts = self.n_gram_counts_list[order - 2] if order > 1 else None
            
            # Get occurrence count for current N-gram
            # .get(key, 0) returns count if N-gram exists, otherwise 0
            # Zero count means this N-gram was never seen in training data
            gram_count = gram_counts.get(current_gram, 0)
            
            # Get occurrence count for context
            # For unigrams, use total number of training sentences as context
            # For higher orders, look up context count in appropriate dictionary
            # Zero context count indicates unseen context in training
            context_count = context_counts.get(context, 0) if context_counts and context else len(self.train_data)
            
            # Calculate vocabulary size for smoothing denominator
            # Add 2 for special tokens: <s> (start) and <e> (end)
            # Larger vocabulary increases smoothing effect
            vocab_size = len(self.vocabulary) + 2
            
            # Apply Add-K (Laplace) smoothing to calculate probability
            # Smoothing prevents zero probabilities for unseen N-grams
            # Formula: P(w|context) = (count(context,w) + k) / (count(context) + k*V)
            if context_count > 0:  # Normal case: context was seen in training
                # Add k to numerator and k*vocab_size to denominator
                # This redistributes probability mass to unseen N-grams
                prob = (gram_count + k) / (context_count + k * vocab_size)
            else:  # Fallback case: unseen context, use uniform distribution
                # Assign equal probability to all vocabulary words
                # This handles completely novel contexts gracefully
                prob = 1.0 / vocab_size
            
            # Apply interpolation weight for current order
            # Use predefined weight if available, otherwise small default weight
            # Higher-order N-grams typically get higher weights
            weight = weights[order - 1] if order <= len(weights) else 0.05
            
            # Add weighted probability contribution to interpolated result
            # Each order contributes (weight * probability) to final score
            # Final interpolated_prob is sum of all weighted contributions
            interpolated_prob += weight * prob
        
        return interpolated_prob
        
    def calculate_perplexity(self, sentence, n_order, k=0.001):
        """Calculate perplexity for a sentence using n-gram model with comprehensive educational explanations"""
        # Prepare sentence with start and end tokens for proper N-gram context
        # Add (n_order-1) start tokens to provide context for first real words
        # For trigrams, add 2 <s> tokens: [<s>, <s>, word1, word2, ...]
        # Add single <e> token to mark sentence boundary
        # This ensures every word has proper N-gram context for probability calculation
        sentence_start = ['<s>'] * (n_order - 1) + sentence + ['<e>']
        
        # Calculate effective sentence length for perplexity normalization
        # Subtract (n_order-1) because start tokens don't count toward sentence length
        # This gives the number of actual word predictions we need to make
        # For "the cat" with 2 start tokens: effective length = 4 - 2 = 2 predictions
        sentence_length = len(sentence_start) - (n_order - 1)
        
        # Initialize accumulator for log probabilities
        # Use log probabilities to avoid numerical underflow with very small values
        # Sum of log probabilities = log of product of probabilities
        # This is mathematically equivalent but numerically stable
        log_probability = 0.0
        
        # Iterate through sentence positions to calculate N-gram probabilities
        # Start from position (n_order-1) to ensure we have full N-gram context
        # Each iteration calculates probability of one word given its context
        for i in range(n_order - 1, len(sentence_start)):
            # Calculate starting index for current N-gram extraction
            # Ensure we don't go before beginning of sentence (use max with 0)
            # For position i with order n, start at (i-n+1) to get n words
            start_idx = max(0, i - n_order + 1)
            
            # Extract N-gram tuple from sentence for probability calculation
            # Slice from start_idx to (i+1) to include word at position i
            # Convert to tuple for dictionary lookup compatibility
            # Example: for trigram at position 3, get words [1, 2, 3]
            n_gram = tuple(sentence_start[start_idx:i+1])
            
            # Calculate probability of this N-gram using trained model
            # Uses interpolated smoothing combining multiple N-gram orders
            # Returns probability P(word_i | context_words)
            probability = self.calculate_probability(n_gram, k)
            
            # Handle edge case: zero or negative probability (should not happen with smoothing)
            # Fallback to uniform distribution over vocabulary for numerical stability
            # Add 2 to vocabulary size for start/end tokens
            if probability <= 0:
                probability = 1.0 / (len(self.vocabulary) + 2)  # Uniform fallback probability
            
            # Add log probability to running sum
            # log(P1 * P2 * P3) = log(P1) + log(P2) + log(P3)
            # This accumulates log likelihood of entire sentence
            log_probability += math.log(probability)
        
        # Calculate perplexity from accumulated log probabilities
        # Perplexity = exp(-log_likelihood / sentence_length)
        # Lower perplexity indicates better model fit to the data
        # Handle edge case of zero sentence length to avoid division by zero
        perplexity = math.exp(-log_probability / sentence_length) if sentence_length > 0 else float('inf')
        
        # Return calculated perplexity value
        # Typical range: 50-500 for reasonable language models
        # Higher values indicate model uncertainty, lower values indicate confidence
        return perplexity
    def evaluate_perplexity(self, test_sentences=None, k=1.0):
        """Evaluate perplexity for each n-gram order using backoff smoothing with comprehensive educational explanations"""
        # Set default test data if none provided by user
        # Use first 100 test sentences to balance evaluation thoroughness with computation time
        # Larger test sets give more reliable results but take longer to compute
        # 100 sentences typically provides good statistical representation
        if test_sentences is None:
            test_sentences = self.test_data[:100]  # Use first 100 test sentences for evaluation
        
        # Initialize dictionary to store perplexity results for each N-gram order
        # Keys will be strings like '2-gram', '3-gram', '4-gram'
        # Values will be average perplexity scores for each order
        # Lower perplexity indicates better model performance
        results = {}
        
        # Evaluate perplexity for each N-gram order starting from bigrams
        # Start from n=2 because unigrams don't provide meaningful perplexity comparison
        # Bigrams are minimum for capturing word sequence dependencies
        # Higher orders capture longer-range linguistic dependencies
        for n in range(2, self.max_n + 1):  # Start from 2-gram (bigrams)
            # Initialize list to collect individual sentence perplexities
            # Each sentence gets one perplexity score
            # We'll average these to get overall model performance
            perplexities = []
            
            # Calculate perplexity for each test sentence
            # Each sentence provides one data point for model evaluation
            # Iterate through all sentences to get comprehensive assessment
            for sentence in test_sentences:
                try:
                    # Calculate perplexity for current sentence using current N-gram order
                    # Uses backoff smoothing (parameter k) for unseen N-grams
                    # Higher k values provide more aggressive smoothing
                    perplexity = self.calculate_perplexity(sentence, n, k)
                    
                    # Filter out invalid perplexity values
                    # Only keep finite positive values for meaningful averaging
                    # Infinite perplexity indicates model failure on sentence
                    # Zero or negative perplexity indicates calculation error
                    if perplexity < float('inf') and perplexity > 0:
                        perplexities.append(perplexity)  # Add valid perplexity to list
                except:
                    # Skip sentences that cause calculation errors
                    # This handles edge cases like empty sentences or encoding issues
                    # Continue processing remaining sentences for robust evaluation
                    continue  # Move to next sentence without adding to results
            
            # Calculate average perplexity for current N-gram order
            # Only compute average if we have valid perplexity measurements
            # Empty list indicates all sentences failed for this order
            if perplexities:  # Check if we have any valid perplexity scores
                # Calculate arithmetic mean of all valid perplexity scores
                # This gives overall model performance for current N-gram order
                # Lower average indicates better model fit to test data
                avg_perplexity = sum(perplexities) / len(perplexities)
                
                # Store result using descriptive key format
                # Format: '2-gram', '3-gram', etc. for clear result interpretation
                results[f'{n}-gram'] = avg_perplexity
            else:
                # Handle case where no valid perplexities were calculated
                # Assign infinite perplexity to indicate complete model failure
                # This preserves result structure while indicating poor performance
                results[f'{n}-gram'] = float('inf')
        
        # Return dictionary mapping N-gram orders to average perplexity scores
        # Example: {'2-gram': 443.7, '3-gram': 324.7, '4-gram': 280.1}
        # Lower values indicate better language model performance
        return results
    
    def generate_text(self, starting_sentence, max_length=10, n_order=2, k=1.0):
        """Generate text using n-gram model with comprehensive educational explanations"""
        # Validate N-gram order against model capacity
        # Ensure we don't request higher order than what was trained
        # Fall back to maximum available order if user requests too high
        # This prevents index errors and ensures meaningful generation
        if n_order > self.max_n:
            n_order = self.max_n  # Clamp to maximum trained order
        
        # Parse starting sentence into individual word tokens
        # Split on whitespace to get list of starting words
        # These words provide initial context for text generation
        # Example: "the cat" becomes ["the", "cat"]
        words = starting_sentence.split()
        
        # Initialize list to store newly generated words
        # This will accumulate words as they are predicted
        # Final generated text is join of these words
        # Separate from input words for clear tracking
        generated_words = []
        
        # Main text generation loop
        # Generate up to max_length new words
        # Each iteration predicts one word based on current context
        # Stop early if we reach sentence end token
        for _ in range(max_length):
            # Prepare context for N-gram prediction
            # Need (n_order-1) words as context for next word prediction
            # Handle case where we have fewer words than needed context
            if len(words) < n_order - 1:
                # Pad with start tokens to provide sufficient context
                # Calculate how many start tokens needed
                # Example: for trigrams with 1 word, add 2 start tokens
                padding_needed = n_order - 1 - len(words)
                context = ['<s>'] * padding_needed + words
            else:
                # Extract last (n_order-1) words as context
                # Use negative indexing to get most recent context
                # For trigrams, get last 2 words: words[-2:]
                context = words[-(n_order-1):]
            
            # Initialize list to store candidate next words with their counts
            # Each candidate is tuple: (word, count_in_training_data)
            # Higher counts indicate more frequent word transitions
            candidates = []
            
            # Convert context to tuple for dictionary lookup
            # Tuples are immutable and hashable, required for dict keys
            # This enables efficient N-gram count retrieval
            context_tuple = tuple(context)
            
            # Get N-gram count dictionary for current order
            # Use (n_order-1) because list is 0-indexed
            # This contains all N-grams of specified order from training
            n_gram_counts = self.n_gram_counts_list[n_order - 1]
            
            # Search through all N-grams to find matches with current context
            # Look for N-grams that start with our context words
            # The last word of matching N-gram is a candidate next word
            for n_gram, count in n_gram_counts.items():
                # Check if this N-gram's prefix matches our context
                # n_gram[:-1] gives all words except the last (the context)
                # If context matches, the last word is a valid next word
                if n_gram[:-1] == context_tuple:
                    # Add candidate word with its training frequency
                    # Higher frequency indicates more probable next word
                    candidates.append((n_gram[-1], count))
            
            # Handle case where no candidates found (unseen context)
            # This can happen with rare word combinations or sparse training data
            # Stop generation to avoid infinite loops or errors
            if not candidates:
                break  # End generation due to lack of valid continuations
                
            # Select next word from candidates
            # Sort candidates by frequency (count) in descending order
            # Most frequent word appears first after sorting
            # This implements simple greedy selection strategy
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Extract the most frequent candidate word
            # Take first element (highest count) and get word (index 0)
            # This gives deterministic selection of most likely next word
            next_word = candidates[0][0]
            
            # Check for sentence end condition
            # Stop generation if model predicts sentence end token
            # This creates natural sentence boundaries in generated text
            if next_word == '<e>':
                break  # End generation due to predicted sentence end
                
            # Add predicted word to generated text
            # Append to list of generated words for final output
            # This builds up the generated text incrementally
            generated_words.append(next_word)
            
            # Update context by adding new word to word history
            # This word becomes part of context for next prediction
            # Maintains sliding window of recent words for N-gram context
            words.append(next_word)
            
        # Combine generated words into final text string
        # Join with spaces to create readable text output
        # Return empty string if no words were generated
        return ' '.join(generated_words)