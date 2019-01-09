import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
import pandas
import os
from sklearn import preprocessing


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter_ver1(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print("start epoch {}".format(epoch) + '.' * 20)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            #print("running batch number {:d}/{:d}".format(batch_num, num_batches_per_epoch))
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def batch_iter(data, batch_size,  shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        #print("running batch number {:d}/{:d}".format(batch_num, num_batches_per_epoch))
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets

def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    #save data to csv file
    #for inx, item in enumerate(datasets['data']):
    #df = pandas.DataFrame(
    #        {'text': datasets['data'],
    #         'label':  datasets['target']
    #         })
    #df.to_csv("data/rt-polaritydata/rt-polarity.csv", encoding='utf-8')


    return datasets

def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(datasets, do_cleaning = True):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    if do_cleaning:
        x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

def load_embedding_vectors_random(vocabulary, vector_size):
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    return embedding_vectors

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                #word, vector = parts[0], list(map('float32', parts[1:]))
                word, vector = parts[0], [float(i) for i in parts[1:]]
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def load_embedding_vectors_fasttext(vocabulary, filename, binary):

    embedding_vectors = load_embedding_vectors_word2vec(vocabulary, filename, binary=binary)

    return embedding_vectors


def get_datasets_twitter_large(csv_file, text_column = 'text', label_column = 'label'):


    df = pandas.read_csv(csv_file)
    #shuffeling
    df = df.sample(frac=1)
    train_size = int(len(df) * 0.8)


    datasets = dict()
    datasets['data'] = list(df['text'])
    datasets['target'] = list(df['label'])
    label_mapping = {'complaint': 0, 'request': 1, 'compliment':2}
    datasets['target'] = [label_mapping[item] for item in datasets['target']]
    datasets['target_names'] = ['complaint', 'request', 'compliment']

    dataset_train = dict()
    dataset_test = dict()
    dataset_train['data'] = datasets['data'][:train_size]
    dataset_train['target'] = datasets['target'][:train_size]
    dataset_train['target_names'] = datasets['target_names']

    dataset_test['data'] = datasets['data'][train_size:]
    dataset_test['target'] = datasets['target'][train_size:]
    dataset_test['target_names'] = datasets['target_names']

    return dataset_train, dataset_test

def get_datasets_it_ticket(csv_file, delimiter=None):
    df = pandas.read_csv(csv_file, delimiter=delimiter, encoding='utf-8')

    datasets = dict()
    datasets['data'] = list(df['Description'])
    datasets['target'] = list(df['Component'])


    le = preprocessing.LabelEncoder()
    le.fit(datasets['target'])
    datasets['target'] = le.transform(datasets['target'])



    #datasets['target'] = list(df['Component'])
    #label_mapping = {'complaint': 0, 'request': 1, 'compliment':2}
    #datasets['target'] = [label_mapping[item] for item in datasets['target']]

    datasets['target_names'] = le.classes_

    return datasets

def get_datasets_it_ticket2(csv_file, delimiter=None):
    from sklearn import preprocessing
    df = pandas.read_csv(csv_file, delimiter=delimiter, encoding='utf-8')

    datasets = dict()
    datasets['data'] = list(df['Description'])
    datasets['target'] = list(df['target'])

    out_path = os.path.join("data/it-ticket/label_mappings.txt")
    print("Saving label mapping to {0}".format(out_path))
    with open(out_path, "r") as text_file:
        datasets['target_names'] = text_file.readlines()

    return datasets



def get_datasets_ag_news(train_csv_file, test_csv_file, classes_file):

    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', header = None, names=["label", "subject", "description"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "subject", "description"])


    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes
    df_train['text'] = df_train['subject'] + ' ' + df_train['description']

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes
    df_test['text'] = df_test['subject'] + ' ' + df_test['description']

    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    file = open(classes_file, encoding='utf-8')
    dataset_train['target_names'] = [line.strip() for line in file.readlines()]
    dataset_test['target_names'] = dataset_train['target_names']
    file.close()

    return dataset_train, dataset_test

def get_datasets_dbpedia(train_csv_file, test_csv_file, classes_file):


    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', names=["label", "subject", "description"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "subject", "description"])



    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes
    df_train['text'] = df_train['subject'] + ' ' + df_train['description']

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes
    df_test['text'] = df_test['subject'] + ' ' + df_test['description']


    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    file = open(classes_file, encoding='utf-8')
    dataset_train['target_names'] = [line.strip() for line in file.readlines()]
    dataset_test['target_names'] = dataset_train['target_names']
    file.close()

    return dataset_train, dataset_test

def get_datasets_yelp_polarity(train_csv_file, test_csv_file, classes_file):

    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', names=["label", "text"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "text"])

    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes

    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    file = open(classes_file, encoding='utf-8')
    dataset_train['target_names'] = [line.strip() for line in file.readlines()]
    dataset_test['target_names'] = dataset_train['target_names']
    file.close()

    return dataset_train, dataset_test

def get_datasets_yelp_full(train_csv_file, test_csv_file, classes_file):

    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', names=["label", "text"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "text"])

    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes

    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    dataset_train['target_names'] = list(df_train['code'].unique())
    dataset_train['target_names'].sort()
    dataset_test['target_names'] = dataset_train['target_names']


    return dataset_train, dataset_test


def get_datasets_amazon_review_full(train_csv_file, test_csv_file, classes_file):


    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', names=["label", "subject", "description"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "subject", "description"])
    df_train.fillna('NaN', inplace=True)
    df_test.fillna('NaN', inplace=True)

    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes
    df_train['text'] = df_train['subject'] + ' ' + df_train['description']

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes
    df_test['text'] = df_test['subject'] + ' ' + df_test['description']

    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    dataset_train['target_names'] = list(df_train['code'].unique())
    dataset_train['target_names'].sort()

    dataset_test['target_names'] = dataset_train['target_names']


    return dataset_train, dataset_test


def get_datasets_amazon_review_polarity(train_csv_file, test_csv_file, classes_file):

    df_train = pandas.read_csv(train_csv_file, encoding='utf-8', names=["label", "subject", "description"])
    df_test = pandas.read_csv(test_csv_file, encoding='utf-8', names=["label", "subject", "description"])
    df_train.fillna('NaN', inplace=True)
    df_test.fillna('NaN', inplace=True)

    df_train.label = pandas.Categorical(df_train.label)
    df_train['code'] = df_train.label.cat.codes
    df_train['text'] = df_train['subject'] + ' ' + df_train['description']

    df_test.label = pandas.Categorical(df_test.label)
    df_test['code'] = df_test.label.cat.codes
    df_test['text'] = df_test['subject'] + ' ' + df_test['description']


    dataset_train = dict()
    dataset_test = dict()

    dataset_train['data'] = list(df_train['text'])
    dataset_train['target'] = list(df_train['code'])

    dataset_test['data'] = list(df_test['text'])
    dataset_test['target'] = list(df_test['code'])

    dataset_train['target_names'] = list(df_train['code'].unique())
    dataset_train['target_names'].sort()

    dataset_test['target_names'] = dataset_train['target_names']


    return dataset_train, dataset_test





def get_datasets_burberry(train_csv_file, test_csv_file, text_column = 'TEXT', label_column = 'REASON_DESCRIPTION__C'):
    from sklearn import preprocessing
    df_train = pandas.read_csv(train_csv_file)
    df_test = pandas.read_csv(test_csv_file)

    df = pandas.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    le.fit(df[label_column])
    df_train['target'] = le.transform(df_train[label_column])
    df_test['target'] = le.transform(df_test[label_column])



    dataset_train = dict()
    dataset_train['data'] = list(df_train[text_column])
    dataset_train['target'] = list(df_train['target'])
    dataset_train['target_names']  = le.classes_

    dataset_test = dict()
    dataset_test['data'] = list(df_test[text_column])
    dataset_test['target'] = list(df_test['target'])
    dataset_test['target_names'] = le.classes_

    return dataset_train, dataset_test



def get_datasets_csv(csv_file, label_mapping_file, delimiter=None, encoding='utf-8', shuffle = True, columns = None):
    from sklearn import preprocessing
    df = pandas.read_csv(csv_file, delimiter=delimiter, encoding=encoding)


    if shuffle:
        df = df.reindex(np.random.permutation(df.index))
        df = df.reset_index(drop=True)



    datasets = dict()
    datasets['data'] = list(df['text'])
    datasets['target'] = list(df['target'])

    out_path = os.path.join(label_mapping_file)
    print("loading label mapping from {0}".format(out_path))
    with open(out_path, "r") as text_file:
        datasets['target_names'] = text_file.readlines()

    return datasets


def read_csv_data(data_file=None, file_type='csv', columns=None, encoding='utf-8'):
    if data_file is not None:
        if file_type not in ['csv', 'xlsx']:
            print("Unsupported file type.......")
            return None, None, None
        elif file_type == 'csv':
            df = pandas.read_csv(data_file, encoding=encoding)
        elif file_type == 'xlsx':
            df = pandas.pandas.read_excel(data_file)

    print("{} Data Loaded...".format(df.shape[0]))

    if columns != None:
        df = df[columns]

    return df
