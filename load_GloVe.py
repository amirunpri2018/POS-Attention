import os
import numpy as np


def load_word_vectors(word_index, GLOVE_DIR, GLOVE_FILE, EMBEDDING_DIM, VOCAB_SIZE):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    
    # prepare embedding matrix
    num_words = min(VOCAB_SIZE, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    
    return embedding_matrix