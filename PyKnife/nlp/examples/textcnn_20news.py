# _*_ coding: utf-8 _*_

"""
This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
"""

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (Embedding, Conv1D, MaxPooling1D,
                          GlobalMaxPooling1D, Dense, Dropout)
from keras.initializers import Constant

# Parameters
# =====================================================================================================

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')

# Dataset parameters
MAX_SEQ_LEN = 1000  # max length of a sequence
VOCAB_SIZE = 20000
VAL_PERCENT = 0.2
NUM_CLASSES = 20

# Training parameters
EMBED_DIM = 100
BATCH_SIZE = 128
EPOCHS = 2

# Load pre-trained word embeddings, and build a mapping from word to its corresponding embedding vector
# =====================================================================================================

print("[INFO] Load pre-trained Glove word embeddings...")
word_2_vec = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        word, vec = line.split(maxsplit=1)
        vec = np.fromstring(vec, dtype=np.float32, sep=' ')
        word_2_vec[word] = vec
print("[INFO] Finish! Found %d word vectors." % (len(word_2_vec)))

# Load & tokenize text data
# =====================================================================================================

print("[INFO] Load text data...")
texts = []
labels = []
label_2_id = {}
for cate in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, cate)
    if os.path.isdir(path):
        label_id = len(label_2_id)
        label_2_id[cate] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)
print("[INFO] Finish! Found %d texts." % (len(texts)))

print("[INFO] Tokenize text data...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
seqs = tokenizer.texts_to_sequences(texts)
print("[INFO] Finish!")

word_2_id = tokenizer.word_index
data = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
labels = to_categorical(np.asarray(labels))

# Split data into training set and validation set
# =====================================================================================================

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_val_samples = int(VAL_PERCENT * data.shape[0])

X_train = data[:-num_val_samples]
y_train = labels[:-num_val_samples]
X_val = data[-num_val_samples:]
y_val = labels[-num_val_samples:]

# Prepare embedding matrix
# =====================================================================================================

num_words = min(VOCAB_SIZE, len(word_2_id) + 1)
embedding_matrix = np.zeros((num_words, EMBED_DIM))
for word, id in word_2_id.items():
    if id >= num_words:
        continue
    embedding_vector = word_2_vec.get(word, None)
    if embedding_vector is not None:
        embedding_matrix[id] = embedding_vector

# Define our model, and start to train
# =====================================================================================================

model = Sequential([
    Embedding(input_dim=num_words,
              output_dim=EMBED_DIM,
              embeddings_initializer=Constant(embedding_matrix),
              input_length=MAX_SEQ_LEN,
              trainable=False),  # (batch_size, 1000) => (batch_size, 1000, 100)
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # (batch_size, 1000, 100) => (batch_size, 996, 128)
    MaxPooling1D(pool_size=5),  # (batch_size, 996, 128) => (batch_size, 199, 128)
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # (batch_size, 199, 128) => (batch_size, 195, 128)
    MaxPooling1D(pool_size=5),  # (batch_size, 195, 128) => (batch_size, 39, 128)
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # (batch_size, 39, 128) => (batch_size, 35, 128)
    GlobalMaxPooling1D(),  # (batch_size, 35, 128) => (batch_size, 128)
    Dense(128, activation='relu'),  # (batch_size, 128) => (batch_size, 128)
    Dropout(0.2),
    Dense(20, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

