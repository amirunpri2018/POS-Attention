
import numpy as np
import nltk
from sklearn.preprocessing import OneHotEncoder

from load_data import load_20newsgroups_data
from load_GloVe import load_word_vectors

TEXT_DATA_DIR = "data/20_newsgroup/"
GLOVE_DIR = "GloVe/"
GLOVE_FILE = "glove.twitter.27B.200d.txt"
MAX_SEQUENCE_LENGTH = 1000
VOCAB_SIZE = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

pos_tag_mapping = {"CD": 1, \
                   "JJ": 2, "JJR": 2, "JJS": 2, \
                   "NN": 3, "NNS": 3, "NNP": 3 , "NNPS": 3, \
                   "VB": 4, "VBD": 4, "VBG": 4, "VBN": 4, "VBP": 4, "VBZ": 4, \
                   "WP": 5, "WP$": 5, \
                   "WRB": 6}

pos_tag_embedding = np.identity(7)

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

# processed_texts = tokenizer.sequences_to_matrix(data.tolist())
# print len(word_index)
reversed_word_index = ["x"]*(len(word_index) + 1)
for word, i in word_index.items():
    reversed_word_index[i] = word

text_processed = []
for text in data:
    text_processed.append([reversed_word_index[wordidx] for wordidx in text])

    
print('POS Tagging Begins ...')
tagged_sentences = nltk.pos_tag_sents(text_processed)
print('POS Tagging done ...')

pos_tags = []
for sentence in tagged_sentences:
    tag_list = [pos_tag_mapping.get(tag_tuple[-1], 0) for tag_tuple in sentence]
    pos_tags.append(tag_list)

pos_tags = np.array(pos_tags)

# pos_tags = np.random.randint(7, size=(data.shape[0], data.shape[1]))

assert pos_tags.shape[1] == MAX_SEQUENCE_LENGTH, "POS Tag array shape mismatch"
assert pos_tags.shape[0] == data.shape[0],       "POS Tag array shape mismatch"

assert np.min(pos_tags) == 0, "POS Tag array values mismatch"
assert np.max(pos_tags) == 6, "POS Tag array values mismatch"

labels = to_categorical(np.asarray(labels))
print 'Shape of data tensor:', data.shape
print 'Shape of label tensor:', labels.shape


embedding_matrix = load_word_vectors(word_index, GLOVE_DIR, GLOVE_FILE, EMBEDDING_DIM, VOCAB_SIZE)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
pos_train = pos_tags[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

print x_train.shape, pos_train.shape, y_train.shape
print embedding_matrix.shape, pos_tag_embedding.shape


x_val = data[-nb_validation_samples:]
pos_val = pos_tags[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

from keras.layers import Embedding

embedding_layer = Embedding(NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


pos_tag_embedding_layer = Embedding(7,
                            7,
                            weights=[pos_tag_embedding],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)



from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, Activation, RepeatVector, multiply
from keras.models import Model
from keras.constraints import non_neg
import keras.backend as K



def repeat_attention(x):
    return MAX_SEQUENCE_LENGTH*K.repeat_elements(attention, EMBEDDING_DIM, axis=-1)

def repeat_attention_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    assert shape[-1] == 1  # only valid for attention
    shape[-1] *= EMBEDDING_DIM
    return tuple(shape)

initial_pos_weights = np.array([[0.0], [0.2], [0.4], [0.75], [0.2], [0.1], [0.1]])
initial_pos_bias = np.array([0.0])

pos_tag_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_pos_tags = pos_tag_embedding_layer(pos_tag_input)
attention =  Dense(1, activation='softmax', kernel_constraint=non_neg(), name='pos_weights', weights = [initial_pos_weights, initial_pos_bias])(embedded_pos_tags)
# attention = Activation('softmax')(attention_weights)
# attention = K.expand_dims(attention, axis=-1)
attention = Lambda(repeat_attention, output_shape=repeat_attention_output_shape)(attention)
# attention = Permute([2, 1])(attention)
# (batch_size, seq, 1)
# (batch_size, seq, embedding_dim)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = multiply([attention, embedded_sequences])
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(inputs = [sequence_input, pos_tag_input], outputs = preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print model.summary()

# happy learning!
model.fit([x_train, pos_train], y_train, validation_data=([x_val, pos_val], y_val),
          epochs=100, batch_size=256)


for layer in model.layers:
    if layer.name == 'pos_weights':
        print layer.get_weights()[0]
        print layer.get_weights()[1]
