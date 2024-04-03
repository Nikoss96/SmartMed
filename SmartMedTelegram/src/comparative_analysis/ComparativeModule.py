import pathlib

import imgkit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kstest
from tabulate import tabulate

from data.paths import MEDIA_PATH, DATA_PATH, COMPARATIVE_ANALYSIS, \
    USER_DATA_PATH, KOLMOGOROVA_SMIRNOVA
from describe_analysis.functions_descriptive import get_user_file_df
from preprocessing.preprocessing import get_categorical_col
from sklearn import preprocessing


class ComparativeModule:
    def __init__(self, df, chat_id):
        self.df = df
        self.chat_id = chat_id

    def get_categorical_and_continuous_columns(self):
        categorical_columns = np.array(get_categorical_col(self.df))
        columns_list = np.array(self.df.columns)
        continuous_columns = [
            v
            for v in columns_list
            if v not in set(categorical_columns) & set(columns_list)
        ]

        return list(categorical_columns), continuous_columns

    def get_class_names(self, group_var, data):
        init_df = get_user_file_df(
            f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{USER_DATA_PATH}",
            self.chat_id,
        )
        init_unique_values = np.unique(init_df[group_var])
        number_class = []
        data_col = data[group_var].tolist()
        for name in init_unique_values:
            number_class.append(data_col[list(init_df[group_var]).index(name)])
        dict_classes = dict(zip(number_class, init_unique_values))
        return dict_classes

    def generate_test_kolmagorova_smirnova(self, categorical_column,
                                           continuous_column):
        classes = self.get_class_names(categorical_column, self.df)

        class1 = list(classes.keys())[0]
        class2 = list(classes.keys())[1]
        data1 = self.df[self.df[categorical_column] == class1][
            continuous_column]
        data2 = self.df[self.df[categorical_column] == class2][
            continuous_column]

        data1 = preprocessing.normalize([data1])
        data2 = preprocessing.normalize([data2])
        res1 = kstest(data1, "norm")
        res2 = kstest(data2, "norm")

        if res1[1] < 0.001:
            p1 = "< 0.001"
        else:
            p1 = np.round(res1[1], 3)

        if res2[1] < 0.001:
            p2 = "< 0.001"
        else:
            p2 = np.round(res2[1], 3)

        classes = self.get_class_names(categorical_column, self.df)
        df = pd.DataFrame(columns=["Группа", "Значение", "p-value"])
        df.loc[1] = [classes[class1], np.round(res1[0], 3), p1]
        df.loc[2] = [classes[class2], np.round(res2[0], 3), p2]

        # plt.figure(figsize=(10, 6))
        # plt.table(cellText=df.values,
        #           colLabels=df.columns,
        #           cellLoc='center', rowLoc='center',
        #           loc='center')
        # plt.axis('off')
        # plt.savefig(
        #     f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{KOLMOGOROVA_SMIRNOVA}/kolmogorova_smirnova_{self.chat_id}.png",
        #     bbox_inches="tight",
        # )

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            colLoc='center',
            loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        header_cells = table.get_celld()
        for (i, j), cell in header_cells.items():
            if i == 0:
                cell.set_text_props(fontweight='bold', color='w')
                cell.set_facecolor('#1d0691')

        for i, row in enumerate(table.get_children()):
            if i % 2 == 0:
                for cell in row.get_children():
                    cell.set_facecolor('#120303')

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{KOLMOGOROVA_SMIRNOVA}/kolmogorova_smirnova_{self.chat_id}.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()


def read_file(path):
    ext = pathlib.Path(path).suffix

    if ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path)

    else:
        df = pd.DataFrame()
    return df
