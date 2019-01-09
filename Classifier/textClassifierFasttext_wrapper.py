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

dataset_train, dataset_test = data_helpers.get_datasets_burberry(train_csv_file ='/Users/i342044/Documents/DATA/Burberry_Baseline_Data/burberry_categ3_train.csv',
                                                                  test_csv_file = '/Users/i342044/Documents/DATA/Burberry_Baseline_Data/burberry_categ3_test.csv',
                                                                  text_column = 'TEXT', label_column = 'REASON_DETAIL__C')

########################################################################################################################

#x_train, y_train = data_helpers.load_data_labels(dataset_train, do_cleaning=False)
#x_test, y_test = data_helpers.load_data_labels(dataset_test, do_cleaning=False)

##################################################
print('experiments using fasttext wrapper....' )
import pandas
fasttext_df_train = pandas.DataFrame(
    {'text': dataset_train['data'],
     'target': dataset_train['target']
    })

fasttext_df_train['label'] = fasttext_df_train['target'].map(lambda x: '__label__{}'.format(x))
fasttext_df_train = fasttext_df_train[['text', 'label']]
fasttext_df_train['text'] = fasttext_df_train['text'].str.lower().replace('\n', ' ').replace('\r', ' ')
fasttext_df_train.to_csv('train.txt', header=None, index=None, sep=' ')



fasttext_df_test = pandas.DataFrame(
    {'text': dataset_test['data'],
     'target': dataset_test['target']
    })

fasttext_df_test['label'] = fasttext_df_test['target'].map(lambda x: '__label__{}'.format(x))
fasttext_df_test = fasttext_df_test[['text', 'label']]
fasttext_df_test['text'] = fasttext_df_test['text'].str.lower().replace('\n', ' ').replace('\r', ' ')
fasttext_df_test.to_csv('test.txt', header=None, index=None, sep=' ')

print('no train: {}'.format(len(fasttext_df_train)))
print('no test: {}'.format(len(fasttext_df_test)))

import fasttext
#model = fasttext.cbow('train.txt', 'model')

classifier = fasttext.supervised('train.txt', 'model', label_prefix='__label__')

results = classifier.test('test.txt')
print("----Performance----")
print("P@1:\t\t[%.3f]" % results.precision)
print("R@1:\t\t[%.3f]" % results.recall)
print("Samples:\t[%d]" % results.nexamples)

from sklearn.metrics import classification_report, accuracy_score

ypred = classifier.predict(fasttext_df_test['text'].values, k=1)
fasttext_df_test['label'] = fasttext_df_test['label'].map(lambda x: x.replace('__label__', ''))
ypred = np.asarray(ypred).squeeze()
report = classification_report(fasttext_df_test['label'], ypred)
accu = accuracy_score(fasttext_df_test['label'], ypred)
print(report)
print('test accuracy: {}'.format(accu))
