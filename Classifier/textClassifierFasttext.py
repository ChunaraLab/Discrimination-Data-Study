'''This example demonstrates the use of fasttext for text classification

Based on Joulin et al's paper:

Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer

import data_helpers

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 1
MAX_NB_WORDS = 20000

VERBOSE_LEVEL = 1
NUM_EPOCH = 10


print('Loading data...')


########################################################################################################################
# dataset_train, datasets_test = data_helpers.get_datasets_ag_news(train_csv_file ='/Users/i342044/Documents/DATA/ag_news_csv/train.csv',
#                                                                   test_csv_file = '/Users/i342044/Documents/DATA/ag_news_csv/test.csv',
#                                                                   classes_file = '/Users/i342044/Documents/DATA/ag_news_csv/classes.txt')

dataset_train, datasets_test = data_helpers.get_datasets_burberry(train_csv_file ='train.csv',
                                                                  test_csv_file = 'test.csv',
                                                                  text_column = 'TEXT', label_column = 'REASON_DESCRIPTION__C')

########################################################################################################################

data_text, y_train = data_helpers.load_data_labels(dataset_train, do_cleaning=False)
x_test, y_test = data_helpers.load_data_labels(datasets_test, do_cleaning=False)


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data_text)
sequences_data = tokenizer.texts_to_sequences(data_text)

sequences_test = tokenizer.texts_to_sequences(x_test)

num_output_categories = y_train.shape[1]
print("number of output classes :", num_output_categories)


#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(sequences_data), 'train sequences')
print(len(sequences_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, sequences_data)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, sequences_test)), dtype=int)))

x_train = sequences_data
x_test = sequences_test
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(num_output_categories, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

print("model evaluating - fasttext")
score = model.evaluate(x_test, y_test,
                       batch_size=128, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print("predict model on test set...")
y_predict = model.predict( x_test, batch_size=128, verbose=VERBOSE_LEVEL)
from sklearn.metrics import classification_report
print("classification report ......\n ", classification_report(y_test.argmax(axis=1), y_predict.argmax(axis=1), digits =5))


##################################################
print('experiments using fasttext wrapper....' )
import pandas
fasttext_df_train = pandas.DataFrame(
    {'text': x_train,
     'label': y_train
    })

# df = df[field]
# df['text'] = df[use_field].apply(lambda x: ' '.join(str(x)), axis=1)
# df[label] = df[label].map(lambda x: '__label__{}'.format(x.replace(' ','_')))
# df = df [['text', label] ]
