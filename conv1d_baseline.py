
import numpy as np

from load_data import load_20newsgroups_data
from load_GloVe import load_word_vectors

TEXT_DATA_DIR = "data/20_newsgroup/"
GLOVE_DIR = "GloVe/"
GLOVE_FILE = "glove.twitter.27B.200d.txt"
MAX_SEQUENCE_LENGTH = 1000
VOCAB_SIZE = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2 


texts, labels, labels_index = load_20newsgroups_data(TEXT_DATA_DIR)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
# tokenizer = Tokenizer() # Keep all words to begin with
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
NUM_WORDS = min(VOCAB_SIZE, len(word_index) + 1)

print('Found %s unique tokens.' % len(word_index))
print('Using Vocabulary of size %s .' % NUM_WORDS)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


embedding_matrix = load_word_vectors(word_index, GLOVE_DIR, GLOVE_FILE, EMBEDDING_DIM, VOCAB_SIZE)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

from keras.layers import Embedding

embedding_layer = Embedding(NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
# x = GlobalAveragePooling1D()(embedded_sequences)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=100, batch_size=256)

























