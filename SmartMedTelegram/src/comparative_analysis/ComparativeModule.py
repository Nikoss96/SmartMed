import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kstest, sem, t
from math import sqrt

from data.paths import MEDIA_PATH, DATA_PATH, COMPARATIVE_ANALYSIS, \
    USER_DATA_PATH, KOLMOGOROVA_SMIRNOVA, T_CRITERIA_INDEPENDENT
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

        class1_name = classes[class1]
        class2_name = classes[class2]

        df = pd.DataFrame(columns=["Группа", "Значение", "p-value"])
        df.loc[0] = [class1_name, np.round(res1[0], 3), p1]
        df.loc[1] = [class2_name, np.round(res2[0], 3), p2]

        fig, ax = plt.subplots(figsize=(6, 4))
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

        ax.text(0.5, 1.1, f"Группирующая переменная: {categorical_column}",
                transform=ax.transAxes, ha='center')
        ax.text(0.5, 1.05, f"Независимая переменная: {continuous_column}",
                transform=ax.transAxes, ha='center')

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

    def generate_t_criterion_student_independent(self, categorical_column,
                                                 continuous_column):
        def independent_ttest(x, y, alpha):
            mean1, mean2 = np.mean(x), np.mean(y)
            se1, se2 = sem(x), sem(y)
            sed = sqrt(se1 ** 2.0 + se2 ** 2.0)
            t_stat = abs(mean1 - mean2) / sed

            if se1 / se2 > 10 or se1 / se2 > 10:
                f = (len(x) + len(y) - 2) * (
                        0.5 + se1 * se2 / (se1 ** 2 + se2 ** 2))
            else:
                f = len(x) + len(y) - 2

            cv = t.ppf(1.0 - alpha, f)
            p = (1.0 - t.cdf(abs(t_stat), f)) * 2.0

            if p < 0.001:
                p = "< 0.001"
            else:
                p = str(np.round(p, 3))

            return np.round(t_stat, 3), np.round(f, 3), np.round(cv, 3), p

        result_columns = ['Доверительная вероятность', 'Эмпирическое значение',
                          'Критическое значение',
                          'Число степеней свободы']

        classes = self.get_class_names(categorical_column, self.df)

        class1 = list(classes.keys())[0]
        class2 = list(classes.keys())[1]
        data1 = self.df[self.df[categorical_column] == class1][
            continuous_column]
        data2 = self.df[self.df[categorical_column] == class2][
            continuous_column]

        results = independent_ttest(data1, data2, 0.05)
        res_list = ["alpha = 0.95", results[0], results[2], results[1]]
        df = pd.DataFrame(columns=result_columns)
        df.loc[1] = res_list

        mean_var1 = np.round(np.mean(data1), 3)
        std_var1 = np.round(np.std(data1), 3)
        mean_var2 = np.round(np.mean(data2), 3)
        std_var2 = np.round(np.std(data2), 3)

        res_list2 = [continuous_column, str(mean_var1) + " ± " + str(std_var1),
                     str(mean_var2) + " ± " + str(std_var2), results[3]]
        mean_p_columns_header = ["Характеристика",
                                 str(classes[class1]) + " (n1 = " + str(
                                     len(data1)) + ")",
                                 str(classes[class2]) + " (n2 = " + str(
                                     len(data2)) + ")", "p-value"]
        df2 = pd.DataFrame(columns=mean_p_columns_header)
        df2.loc[1] = res_list2

        fig, ax = plt.subplots()

        ax.text(0.5, 1.1, f"Группирующая переменная: {categorical_column}",
                transform=ax.transAxes, ha='center')
        ax.text(0.5, 1.05, f"Независимая переменная: {continuous_column}",
                transform=ax.transAxes, ha='center')

        ax.axis('tight')
        ax.axis('off')
        table1 = ax.table(cellText=df.values, colLabels=df.columns,
                          cellLoc='center', colLoc='center', rowLoc='center',
                          loc='upper center',
                          colWidths=[0.3, 0.3, 0.3, 0.3])
        table1.auto_set_font_size(False)
        table1.set_fontsize(7)

        table2 = ax.table(cellText=df2.values, colLabels=df2.columns,
                          cellLoc='center', colLoc='center', rowLoc='center',
                          loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{T_CRITERIA_INDEPENDENT}/t_criteria_independent_{self.chat_id}.png",
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
