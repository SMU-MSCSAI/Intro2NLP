# More workflow are in the noteboook

def synsetSimValue(model_wv, words):
    """
    Calculate similarity statistics for a list of words using pre-trained word vectors.

    Args:
    model_wv (gensim.models.keyedvectors.KeyedVectors): The word vectors from a Word2Vec model.
    words (list): A list of words (strings).

    Returns:
    list: A list containing similarity statistics (average, standard deviation, minimum, maximum).
    """

    # Check if the input is a list, if not but a string of len 1 convert it to list one word
    if not isinstance(words, list):
        if isinstance(words, str):
            words = [words]
        else:
            raise ValueError("The input must be a list of words or a single word.")
        
     # Filter words that are not in the model's vocabulary
    words = [word for word in words if word in model_wv.key_to_index]

    # Check the number of words in the synset
    num_words = len(words)

    if num_words == 1:
        return []
    elif num_words == 2:
        return [model_wv.similarity(words[0], words[1])]
    else:
        # Calculate the similarity between all pairs of words in the synset
        sim_values = []
        for i in range(num_words):
            for j in range(i+1, num_words):
                sim_values.append(model_wv.similarity(words[i], words[j]))

        # Calculate and return the statistics
         # Calculate statistics
        avg = np.mean(sim_values)
        sd = np.std(sim_values)
        min_sim = np.min(sim_values)
        max_sim = np.max(sim_values)

        return [avg, sd, min_sim, max_sim]
    
# Calculate similarity statistics for pairs of words from two different sets using a Word2Vec model
def crossSynsetSimValue(model, words1, words2):
    """
    Calculate similarity statistics for pairs of words from two different sets using a Word2Vec model.

    Args:
    model (gensim.models.Word2Vec): The Word2Vec model.
    words1 (list): A list of words from the first set.
    words2 (list): A list of words from the second set.

    Returns:
    list: A list containing similarity statistics (average, standard deviation, minimum, maximum).
          If both sets have only one word, returns a list with the similarity between the two words.
    """
    # Filter words that are not in the model's vocabularry
    words1 = [word for word in words1 if word in model.key_to_index]
    words2 = [word for word in words2 if word in model.key_to_index]
    # print(f"{len(words1)} words out of {len(words1)} are in the vocabulary.")
    # print(f"{len(words2)} words out of {len(words2)} are in the vocabulary.")

    # return similarity between two words if both sets have only one word
    if len(words1) == 1 and len(words2) == 1:
        return [model.similarity(words1[0], words2[0])]
    else:
        # Calculate similarities between all pairs of words, one from each set
        similarities = []
        for word1 in words1:
            for word2 in words2:
                similarities.append(model.similarity(word1, word2))

        # Calculate statistics
        avg = np.mean(similarities)
        sd = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)

        return [avg, sd, min_sim, max_sim]
    

# Get BertVector for sentences
def genBERTVector(model, word, sentences):
    """
    Generate BERT vectors for a word in a list of sentences.
    """
    vectors = []
    for sentence in sentences:
        # Get BERT embeddings and tokenized sentence
        # embeddings, tokenized_sentence = get_bert_embeddings(sentence, model)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize the input sentence and convert to PyTorch tensors
        inputs = tokenizer(sentence, padding=True, return_tensors="pt")
        tokenized_sentence = tokenizer.tokenize(sentence)

        # Forward pass through the BERT model to get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the hidden states (embeddings) of the last layer
        last_hidden_states = outputs.last_hidden_state

        # Convert tensor to numpy array
        embeddings = last_hidden_states.numpy()


        # Find the position of the word or its subwords in the tokenized sentence
        word_positions = [i for i, token in enumerate(tokenized_sentence) if word in token]

        # If the word or its subwords are not in the sentence, add an empty list
        if not word_positions:
            vectors.append([])
            continue

        # Extract the embedding for the first occurrence of the word or its subwords
        word_vector = embeddings[0, word_positions[0], :]
        vectors.append(word_vector)

    return vectors