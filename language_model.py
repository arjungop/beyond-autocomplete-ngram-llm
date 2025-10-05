import numpy as np
import pandas as pd


# Function to generate n-grams
def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    # Go through each sentence in the data
    for sentence in data:  # complete this line
        # prepend start token n times, and  append <e> one time
        sentence = [start_token] * n + sentence + [end_token]
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)
        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.
        for i in range(len(sentence) - n + 1):
            # Get the n-gram from i to i+n
            n_gram = sentence[i:i + n]
            # check if the n-gram is in the dictionary
            if n_gram in n_grams.keys():
                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1
    return n_grams


# Function to estimate the probability of a word given the prior 'n' words using the n-gram counts.
def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # Set the denominator
    # If the previous n-gram exists in the dictionary of n-gram counts,
    # Get its count.  Otherwise set the count to zero
    # Use the dictionary that has counts for n-grams
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0

    # Calculate the denominator using the count of the previous n gram
    # and apply k-smoothing
    denominator = previous_n_gram_count + k * vocabulary_size

    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)

    # Set the count to the count in the dictionary,
    # otherwise 0 if not in the dictionary
    # use the dictionary that has counts for the n-gram plus current word
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

    # Define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count + k

    # Calculate the probability as the numerator divided by denominator
    probability = numerator / denominator

    return probability


# Function defined below loops over all words in vocabulary to calculate probabilities for all possible words.
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>',
                           unknown_token="<unk>", k=1.0):
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability
    return probabilities


# Function to get the count matrix
def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]

    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))

    # mapping from n-gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    # mapping from next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count

    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix


# Function calculates the probabilities of each word given the previous n-gram, and stores this in matrix form.
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k, unique_words=None):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix


# Function calculates perplexity which is used as an evaluation metric for our model
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>',
                         end_token='<e>', k=1.0):
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # prepend <s> and append <e>
    sentence = [start_token] * n + sentence + [end_token]

    # Cast the sentence from a list to a tuple
    sentence = tuple(sentence)

    # length of sentence (after adding <s> and <e> tokens)
    N = len(sentence)

    # The variable p will hold the product
    # that is calculated inside the n-root
    # Update this in the code below
    product_pi = 1.0

    # Index t ranges from n to N - 1, inclusive on both ends
    for t in range(n, N):
        # get the n-gram preceding the word at position t
        n_gram = sentence[t - n:t]

        # get the word at position t
        word = sentence[t]

        # Estimate the probability of the word given the n-gram
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)

        # Update the product of the probabilities
        product_pi *= 1 / probability

    # Take the Nth root of the product
    perplexity = product_pi ** (1 / N)

    return perplexity


# Function to suggest a word
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    # Estimate the probabilities that each word in the vocabulary
    # is the next word,
    # given the previous n-gram, the dictionary of n-gram counts,
    # the dictionary of n plus 1 gram counts, and the smoothing constant
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None

    # Initialize the highest word probability to 0
    # this will be set to the highest probability
    # of all words to be suggested
    max_prob = 0

    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items():  # complete this line

        # If the optional start_with string is set
        if start_with is not None:  # complete this line

            # Check if the beginning of word does not match with the letters in 'start_with'
            if not word.startswith(start_with):  # complete this line

                # if they don't match, skip this word (move onto the next word)
                continue  # complete this line

        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob:  # complete this line

            # If so, save this word as the best suggestion (so far)
            suggestion = word

            # Save the new maximum probability
            max_prob = prob

    return suggestion, max_prob


# Function to get multiple suggestions
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions


class LanguageModel:
    """
    N-gram Language Model class for autocomplete and text generation evaluation.
    """
    
    def __init__(self, n_max=4, k=1.0):
        """
        Initialize the language model.
        
        Args:
            n_max: Maximum n-gram order to use
            k: Smoothing parameter
        """
        self.n_max = n_max
        self.k = k
        self.vocabulary = None
        self.n_gram_counts_list = []
        self.vocabulary_size = 0
        
    def fit(self, training_data, vocabulary):
        """
        Train the language model on the given data.
        
        Args:
            training_data: List of tokenized sentences
            vocabulary: Set/list of vocabulary words
        """
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        
        # Build n-gram counts for all orders
        self.n_gram_counts_list = []
        for n in range(1, self.n_max + 1):
            n_gram_counts = count_n_grams(training_data, n)
            self.n_gram_counts_list.append(n_gram_counts)
    
    def get_user_input_suggestions(self, previous_tokens, num_suggestions=5, start_with=None):
        """
        Get word suggestions for autocomplete given previous tokens.
        
        Args:
            previous_tokens: List of previous words in the context
            num_suggestions: Number of suggestions to return
            start_with: Optional prefix for suggested words
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        if not self.n_gram_counts_list or not self.vocabulary:
            return []
        
        # Collect suggestions from different n-gram orders
        all_suggestions = {}
        
        # Try different context lengths (from longest to shortest)
        for context_len in range(min(len(previous_tokens), self.n_max - 1), 0, -1):
            if context_len > len(previous_tokens):
                continue
                
            context = previous_tokens[-context_len:]
            
            if context_len < len(self.n_gram_counts_list):
                n_gram_counts = self.n_gram_counts_list[context_len - 1]
                n_plus1_gram_counts = self.n_gram_counts_list[context_len]
                
                # Get all possible next words and their probabilities
                probabilities = estimate_probabilities(
                    context, n_gram_counts, n_plus1_gram_counts, 
                    self.vocabulary, k=self.k
                )
                
                # Filter by start_with if provided
                for word, prob in probabilities.items():
                    if start_with is None or word.startswith(start_with):
                        # Use maximum probability if word appears multiple times
                        if word not in all_suggestions or prob > all_suggestions[word]:
                            all_suggestions[word] = prob
        
        # Sort by probability and return top suggestions
        sorted_suggestions = sorted(all_suggestions.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        return sorted_suggestions[:num_suggestions]
    
    def calculate_perplexity(self, sentence):
        """
        Calculate perplexity of a sentence using the trained model.
        
        Args:
            sentence: Input sentence (string or list of tokens)
            
        Returns:
            Perplexity score
        """
        if isinstance(sentence, str):
            import nltk
            tokens = nltk.word_tokenize(sentence.lower())
        else:
            tokens = sentence
        
        if not self.n_gram_counts_list or len(self.n_gram_counts_list) < 2:
            return float('inf')
        
        # Use the highest order model available
        n_gram_counts = self.n_gram_counts_list[-2]  # n-gram counts
        n_plus1_gram_counts = self.n_gram_counts_list[-1]  # (n+1)-gram counts
        
        return calculate_perplexity(
            tokens, n_gram_counts, n_plus1_gram_counts, 
            self.vocabulary_size, k=self.k
        )
