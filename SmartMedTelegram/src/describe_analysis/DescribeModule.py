from matplotlib import use
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scipy.stats import variation

from data.paths import USER_DATA_PATH, DATA_PATH, MEDIA_PATH, DESCRIBE_ANALYSIS, \
    DESCRIBE_TABLES, CORRELATION_MATRICES, PLOTS

use("agg")


class DescribeModule:
    def __init__(self, preprocessor, chat_id):
        self.preprocessor = preprocessor
        self.chat_id = chat_id
        self._generate_table()
        self.save_table_file()
        self.create_correlation_matrices()
        self.make_plots()

    def _generate_table(self):

        metrics = ['count', 'mean', 'std', 'max', 'min', '25%', '50%',
                   '75%']

        df = self.preprocessor.get_numeric_df(self.preprocessor.df)
        init_df = df

        df = df.describe().reset_index()
        df = df[df['index'].isin(metrics)]
        df = df.rename(columns={"index": "Метрики"})

        cols = df.columns
        init_describe_length = len(df)

        for col in init_df.columns:
            df.loc[init_describe_length, col] = np.exp(
                np.log(init_df[col]).mean())
            df.loc[init_describe_length + 1, col] = variation(init_df[col])

        df.loc[init_describe_length, 'Метрики'] = 'geom_mean'
        df.loc[init_describe_length + 1, 'Метрики'] = 'variation'

        for j in range(1, len(cols)):
            for i in range(len(df)):
                df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))

        self.table_df = df

    def save_table_file(self):
        self.table_df.loc[self.table_df[
                              'Метрики'] == 'count', 'Метрики'] = 'Количество наблюдений'
        self.table_df.loc[
            self.table_df['Метрики'] == 'mean', 'Метрики'] = 'Среднее значение'
        self.table_df.loc[self.table_df[
                              'Метрики'] == 'std', 'Метрики'] = 'Стандартное отклонение'
        self.table_df.loc[
            self.table_df['Метрики'] == 'max', 'Метрики'] = 'Максимум'
        self.table_df.loc[
            self.table_df['Метрики'] == 'min', 'Метрики'] = 'Минимум'
        self.table_df.loc[
            self.table_df['Метрики'] == '25%', 'Метрики'] = '1-ый квартиль'
        self.table_df.loc[
            self.table_df['Метрики'] == '50%', 'Метрики'] = '2-ой квартиль'
        self.table_df.loc[
            self.table_df['Метрики'] == '75%', 'Метрики'] = '3-ий квартиль'
        self.table_df.loc[
            self.table_df[
                'Метрики'] == 'geom_mean', 'Метрики'] = 'Среднее геометрическое'
        self.table_df.loc[
            self.table_df['Метрики'] == 'variation', 'Метрики'] = 'Разброс'

        self.table_file = self.table_df.to_excel(
            f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{DESCRIBE_TABLES}/{self.chat_id}_describe_table.xlsx",
            index=False)

    def create_correlation_matrices(
            self,
            sharey=False,
            annot=True,
            Pearson=True,
            Spearman=True,
            title="",
            cmap=sns.color_palette("viridis", as_cmap=True),
            fmt=".2f",
    ):
        FIG_WIDTH = 16
        FIG_HEIGHT = 14
        dataframe = self.preprocessor.df

        if not Pearson and not Spearman:
            return
        dataframe.drop(columns=dataframe.columns[0], axis=1, inplace=True)
        ncols = Spearman + Pearson

        f, axes = plt.subplots(
            nrows=1, ncols=ncols, sharey=sharey,
            figsize=(FIG_WIDTH * ncols, FIG_HEIGHT)
        )

        pltP = None
        pltS = None
        if Pearson:
            if Spearman:
                pltP = sns.heatmap(
                    dataframe.corr(method="pearson"),
                    annot=annot,
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                    center=0,
                    fmt=fmt,
                    ax=axes[0],
                )
            else:
                pltP = sns.heatmap(
                    dataframe.corr(method="pearson"),
                    annot=annot,
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                    center=0,
                    fmt=fmt,
                )
            pltP.xaxis.tick_top()
            pltP.set_title("Коэффициент корреляции Пирсона. " + title)
            pltP.set_xticklabels(pltP.get_xticklabels(), rotation=30)
        if Spearman:
            if Pearson:
                pltS = sns.heatmap(
                    dataframe.corr(method="spearman"),
                    annot=annot,
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                    center=0,
                    fmt=fmt,
                    ax=axes[1],
                )
            else:
                pltS = sns.heatmap(
                    dataframe.corr(method="spearman"),
                    annot=annot,
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                    center=0,
                    fmt=fmt,
                )
            pltS.xaxis.tick_top()
            pltS.set_title("Коэффициент корреляции Спирмена. " + title)
            pltS.set_xticklabels(pltS.get_xticklabels(), rotation=30)

        for ax in axes:
            ax.title.set_position([0.5, -0.2])

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{CORRELATION_MATRICES}/describe_corr_{self.chat_id}.png",
            bbox_inches="tight",
            pad_inches=0.0,
        )

    def make_plots(self):

        df = self.preprocessor.df
        chat_id = self.chat_id

        num_cols = len(df.columns)
        num_rows = (num_cols + 3) // 4

        fig, axs = plt.subplots(nrows=num_rows, ncols=4,
                                figsize=(20, num_rows * 5))
        axs = axs.flatten()

        counter = 0

        for i, col in enumerate(df.columns):
            sns.histplot(df[col], kde=True, ax=axs[i])
            axs[i].set_title(f'Распределение столбца "{col}"')
            axs[i].set_xlabel("Значение")
            axs[i].set_ylabel("Частота")
            counter += 1

        for j in range(counter, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{PLOTS}/describe_plots_{chat_id}.png"
        )
