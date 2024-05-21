import numpy as np
import pandas as pd
from scipy import stats

from data.paths import MEDIA_PATH, DATA_PATH, VARIANCE_ANALYSIS, KRUSKAL_TEST


class VarianceModule:
    def __init__(self, df, chat_id):
        self.df = df
        self.chat_id = chat_id

    def get_all_columns(self):
        columns_list = np.array(self.df.columns)
        return columns_list

    def generate_test_kruskal_wallis(self, var_1, var_2, var_3):
        result_columns = [
            "Полученное значение",
            "P-значение",
        ]

        data1 = self.df[var_1]
        data2 = self.df[var_2]
        data3 = self.df[var_3]
        stat, p_value = stats.kruskal(data1, data2, data3)
        df = pd.DataFrame(columns=result_columns)
        df.loc[1] = [stat, p_value]

        with pd.ExcelWriter(
                f"{MEDIA_PATH}/{DATA_PATH}/{VARIANCE_ANALYSIS}/{KRUSKAL_TEST}/test_kruskal_wallis_{self.chat_id}.xlsx",
                engine="xlsxwriter",
        ) as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
