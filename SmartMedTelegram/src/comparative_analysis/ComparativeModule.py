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

        def test_kolmagorova_smirnova(group_value):
            data = self.df[self.df[categorical_column] == group_value][
                continuous_column]
            data = preprocessing.normalize([data])
            res = kstest(data, "norm")

            if res[1] < 0.001:
                p = "< 0.001"
            else:
                p = np.round(res[1], 3)

            class_name = classes[group_value]

            return res, p, class_name

        df = pd.DataFrame(columns=["Группа", "Значение", "p-value"])

        classes_array = list(classes.keys())

        for i in range(len(classes_array)):
            res, p, class_name = test_kolmagorova_smirnova(classes_array[i])

            df.loc[i] = [class_name, np.round(res[0], 3), p]

        df.to_excel(
            f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{KOLMOGOROVA_SMIRNOVA}/kolmogorova_smirnova_{self.chat_id}.xlsx",
            index=False,
        )

    def generate_t_criterion_student_independent(self,
                                                 categorical_column,
                                                 continuous_column,
                                                 classes
                                                 ):
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

        with pd.ExcelWriter(
                f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{T_CRITERIA_INDEPENDENT}/t_criteria_independent_{self.chat_id}.xlsx",
                engine='xlsxwriter') as writer:

            df.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet1', startrow=len(df) + 2,
                         index=False)


def read_file(path):
    ext = pathlib.Path(path).suffix

    if ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path)

    else:
        df = pd.DataFrame()
    return df
