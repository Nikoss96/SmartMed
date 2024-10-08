import pathlib
from typing import Dict

import pandas as pd

from sklearn import preprocessing

numerics_list = [
    "int16",
    "int32",
    "int",
    "float",
    "bool",
    "int64",
    "float16",
    "float32",
    "float64",
]


class PandasPreprocessor:
    def __init__(self, settings: Dict, chat_id):
        self.chat_id = chat_id
        self.settings = settings
        self.__read_file()
        self.numerics_list = numerics_list

    def __read_file(self):
        ext = pathlib.Path(self.settings["path"]).suffix

        if ext == ".xlsx" or ext == ".xls":
            self.df = pd.read_excel(self.settings["path"])

        self.df.columns = self.df.columns.astype("str")

    def preprocess(self):
        self.fillna()
        self.encoding()
        self.save_df_to_file()

    def fillna(self):
        value = self.settings["fillna"]

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

        elif value == "dropna":
            self.df = self.df.dropna()

    def encoding(self):
        method = self.settings["encoding"]

        if method == "label_encoding":
            transformer = preprocessing.LabelEncoder()

        for column in self.df.select_dtypes(exclude=self.numerics_list):
            transformer.fit(self.df[column].astype(str).values)
            self.df[column] = transformer.transform(self.df[column].astype(str).values)

    def save_df_to_file(self):
        ext = pathlib.Path(self.settings["path"]).suffix

        if ext == ".xlsx" or ext == ".xls":
            self.df.to_excel(self.settings["path"], index=False)

    def get_categorical_df(self, df):
        return df.select_dtypes(exclude=self.numerics_list)


def get_categorical_col(data):
    cat_list = []
    for col in data.columns:
        if data[col].nunique() < 5:
            cat_list.append(col)
    return cat_list


def get_numeric_df(df):
    return df.select_dtypes(include=numerics_list)
