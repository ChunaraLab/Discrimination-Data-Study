"""
Class definition for basic text reading and writing files for training text classifier
"""
# Python compatibility with 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import pandas as pd


class TextReader:
    def __init__(self, train_csv_file, test_csv_file, classes_file=None):
        """
        Accepts train and test CSV files and sets their file names
        """
        assert train_csv_file is not None and path.exists(train_csv_file), "Train file not found in path!"
        assert test_csv_file is not None and path.exists(test_csv_file), "Test file not found in path!"

        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.classes_file = classes_file
        self.df_train, self.df_test = None, None

        return

    def read_ag_news_files(self):
        """
        Read AG News train and test files and return respective dataframes
        :return: [train dataframe, test dataframe]
        """
        assert self.classes_file is not None, "Please set classes file for AG News"

        file = open(self.classes_file, encoding="utf-8")
        map_dict = {i: line.strip() for i, line in enumerate(file)}
        file.close()

        def process_csv_file(csv_file, map_dict):
            df = pd.read_csv(csv_file, encoding="utf-8", names=["label", "subject", "description"])
            df.label = pd.Categorical(df.label)
            df['code'] = df["label"].cat.codes
            df['text'] = df['subject'] + ' ' + df['description']
            df["label_"] = df["code"].map(map_dict)
            df = df[["text", "code", "label_"]]

            return df

        self.df_train = process_csv_file(self.train_csv_file, map_dict)
        self.df_test = process_csv_file(self.test_csv_file, map_dict)

        return self.df_train, self.df_test, map_dict

    def read_db_pedia_files(self):
        """
        Read DBPedia train and test files and return respective dataframes
        :return: [train dataframe, test dataframe]
        """
        assert self.classes_file is not None, "Please set classes file for DBPedia"

        file = open(self.classes_file, encoding="utf-8")
        map_dict = {i: line.strip() for i, line in enumerate(file)}
        file.close()

        def process_csv_file(csv_file, map_dict):
            df = pd.read_csv(csv_file, encoding="utf-8", names=["label", "subject", "description"])
            df.label = pd.Categorical(df.label)
            df['code'] = df["label"].cat.codes
            df['text'] = df['subject'] + ' ' + df['description']
            df["label_"] = df["code"].map(map_dict)
            df = df[["text", "code", "label_"]]

            return df

        self.df_train = process_csv_file(self.train_csv_file, map_dict)
        self.df_test = process_csv_file(self.test_csv_file, map_dict)

        return self.df_train, self.df_test, map_dict

    def read_yelp_polarity_files(self):
        """
        Read Yelp polarity train and test files and return respective dataframes
        :return: [train dataframe, test dataframe]
        """
        assert self.classes_file is not None, "Please set classes file for Yelp Polarity"

        file = open(self.classes_file, encoding="utf-8")
        map_dict = {i: line.strip() for i, line in enumerate(file)}
        file.close()

        def process_csv_file(csv_file, map_dict):
            df = pd.read_csv(csv_file, encoding="utf-8", names=["label", "text"])
            df.label = pd.Categorical(df.label)
            df['code'] = df["label"].cat.codes
            df["label_"] = df["code"].map(map_dict)
            df = df[["text", "code", "label_"]]

            return df

        self.df_train = process_csv_file(self.train_csv_file, map_dict)
        self.df_test = process_csv_file(self.test_csv_file, map_dict)

        return self.df_train, self.df_test, map_dict

    def read_yahoo_answers_files(self):
        """
        Read Yahoo Answers train and test files and return respective dataframes
        :return: [train dataframe, test dataframe]
        """
        assert self.classes_file is not None, "Please set classes file for Yahoo Answers"

        file = open(self.classes_file, encoding="utf-8")
        map_dict = {i: line.strip() for i, line in enumerate(file)}
        file.close()

        def process_csv_file(csv_file, map_dict):
            df = pd.read_csv(csv_file, encoding="utf-8", names=["label", "description_1",
                                                                "description_2", "description_3"])
            df.label = pd.Categorical(df.label)
            df['code'] = df["label"].cat.codes
            df["label_"] = df["code"].map(map_dict)
            # df = df.fillna('')
            df["text"] = df["description_1"].map(str) + " " + df["description_2"].map(str) + " " + df[
                "description_3"].map(str)
            df = df[["text", "code", "label_"]]

            return df

        self.df_train = process_csv_file(self.train_csv_file, map_dict)
        self.df_test = process_csv_file(self.test_csv_file, map_dict)

        return self.df_train, self.df_test, map_dict

    def read_burberry_files(self, category=1):
        """
        Read Burberry train and test files and return respective dataframes
        :return: [train dataframe, test dataframe]
        """
        category_list = ["REASON_CODE__C", "REASON_DESCRIPTION__C", "REASON_DETAIL__C"]
        map_dict = {}

        def process_csv_file(csv_file):
            df = pd.read_csv(csv_file, encoding="utf-8", usecols=[category_list[category], "TEXT"])
            df.rename(columns={category_list[category]: 'label_', "TEXT": "text"}, inplace=True)
            df.label_ = pd.Categorical(df.label_)
            df['code'] = df["label_"].cat.codes
            df = df[["text", "code", "label_"]]

            return df

        self.df_train = process_csv_file(self.train_csv_file)
        self.df_test = process_csv_file(self.test_csv_file)

        return self.df_train, self.df_test, map_dict

    def save_to_fasttext_format(self, train_file, test_file):
        """
        Uses dataframe content to write to file
        :return: 
        """
        assert self.df_train is not None and self.df_test is not None, "Set train/test dataframes to write to file"

        train_text = list(self.df_train["text"] + " __label__" + self.df_train["label_"])
        test_text = list(self.df_test["text"] + " __label__" + self.df_test["label_"])

        def write_list_file(string_list, file_name):
            """
            Creates a file in path file_name, with one string per line from the string_list
            """
            with open(file_name, "w", encoding="utf-8") as f:
                for line in string_list:
                    f.write("%s\n" % line)

        if self.df_train is not None and self.df_test is not None:
            write_list_file(train_text, train_file)
            write_list_file(test_text, test_file)

        return
