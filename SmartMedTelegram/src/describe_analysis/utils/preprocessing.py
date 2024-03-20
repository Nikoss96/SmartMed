import pathlib
from typing import Dict

import pandas as pd
import numpy as np

from sklearn import preprocessing


class PandasPreprocessor:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.__read_file()
        self.numerics_list = [
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
        self.preprocess()

    def __read_file(self):
        ext = pathlib.Path(self.settings["path"]).suffix

        if ext == ".csv":
            self.df = pd.read_csv(self.settings["path"], sep=";")
        elif ext == ".xlsx" or ext == ".xls":
            self.df = pd.read_excel(self.settings["path"])

        elif ext == ".tsv":
            self.df = pd.read_table(self.settings["path"], sep=";")

        self.df.columns = self.df.columns.astype("str")

    def preprocess(self):
        self.fillna()
        self.encoding()

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

    def get_numeric_df(self, df):
        return df.select_dtypes(include=self.numerics_list)
