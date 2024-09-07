import pathlib
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing


class ExtentionFileException(Exception):
    pass


class PandasPreprocessor:
    """Class to preprocessing any datasets"""

    def __init__(self, settings: Dict):
        self.settings = settings  # settings['data']
        self.__read_file()
        self.numerics_list = {
            "int16",
            "int32",
            "int",
            "float",
            "bool",
            "int64",
            "float16",
            "float32",
            "float64",
        }

    def __read_file(self):
        ext = pathlib.Path(self.settings["path"]).suffix

        if ext == ".csv":
            self.df = pd.read_csv(self.settings["path"], sep=";")
            print(self.df)
        elif ext == ".xlsx" or ext == ".xls":
            self.df = pd.read_excel(self.settings["path"])

        elif ext == ".tsv":
            self.df = pd.read_table(self.settings["path"], sep=";")

        else:
            raise ExtentionFileException

        self.df.columns = self.df.columns.astype("str")

    def preprocess(self):
        self.fillna()
        self.encoding()
        self.scale()

    def fillna(self):
        value = self.settings["preprocessing"]["fillna"]
        if value == "mean":
            for col in self.df.columns:
                if self.df[col].dtype in self.numerics_list:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode().values[0])
        elif value == "median":
            for col in self.df.columns:
                if self.df[col].dtype in self.numerics_list:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode().values[0])
        elif value == "droprows":
            self.df = self.df.dropna()

    def encoding(self):
        method = self.settings["preprocessing"]["encoding"]
        if method == "label_encoding":
            transformer = preprocessing.LabelEncoder()

        for column in self.df.select_dtypes(exclude=self.numerics_list):
            transformer.fit(self.df[column].astype(str).values)
            self.df[column] = transformer.transform(self.df[column].astype(str).values)

    def scale(self):
        method = self.settings["preprocessing"]["scaling"]
        if method:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.df)
            self.df = scaler.transform(self.df)
        else:
            pass

    def get_numeric_df(self, df):
        return df.select_dtypes(include=self.numerics_list)

    def get_categorical_df(self, df):
        return df.select_dtypes(exclude=self.numerics_list)


def get_categorical_col(data):
    cat_list = []
    for col in data.columns:
        if data[col].nunique() < 10:
            cat_list.append(col)
    return cat_list


def read_file(path):
    ext = pathlib.Path(path).suffix

    if ext == ".csv":
        df = pd.read_csv(path)

        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=";")

    elif ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path)

    elif ext == ".tcv":
        df = pd.read_excel(path, sep="\t")

    else:
        df = pd.DataFrame()
    return df


def get_confusion_matrix(true_values, pred_values):
    tp = fn = tn = fp = 0
    for i in range(len(true_values)):
        if true_values[i] == 1 and pred_values[i] == 1:
            tp += 1
        if true_values[i] == 1 and pred_values[i] == 0:
            fn += 1
        if true_values[i] == 0 and pred_values[i] == 0:
            tn += 1
        if true_values[i] == 0 and pred_values[i] == 1:
            fp += 1
    return [[tp, fn], [fp, tn]]


def get_class_names(group_var, path, data):
    init_df = read_file(path)
    init_unique_values = np.unique(init_df[group_var])
    number_class = []
    data_col = data[group_var].tolist()
    for name in init_unique_values:
        number_class.append(data_col[list(init_df[group_var]).index(name)])
    dict_classes = dict(zip(number_class, init_unique_values))
    return dict_classes
