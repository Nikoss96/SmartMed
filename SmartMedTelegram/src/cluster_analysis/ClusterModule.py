import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans

from data.paths import MEDIA_PATH, DATA_PATH, CLUSTER_ANALYSIS, USER_DATA_PATH, \
    ELBOW_METHOD


class ClusterModule:
    def __init__(self, df, chat_id):
        self.df = df
        self.chat_id = chat_id
        self.settings = {"fillna": "mean", "method": 0, "metric": 0,
                         "encoding": "label_encoding", "scaling": False}

    # def _prepare_dashboard_settings(self):
    #     settings = dict()
    #
    #     # prepare metrics as names list from str -> bool
    #     settings['metric'] = self.settings['metric']
    #     # prepare graphs as names list from str -> bool
    #     settings['method'] = self.settings['method']
    #     self.graph_to_method = {
    #         0: self._generate_kmeans,
    #         1: self._generate_hirarchy,
    #         2: self._generate_component
    #     }
    #
    #     settings['data'] = self.data
    #
    #     return settings

    def elbow_method_and_optimal_clusters(self, max_clusters):
        inertia_values = []

        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(self.df)
            inertia_values.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
        plt.xlabel('Количество Кластеров')
        plt.ylabel('Инерция')
        plt.title('Метод Локтя')

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{self.chat_id}.png",
        )

        plt.clf()
        plt.close()

        differences = [inertia_values[i] - inertia_values[i - 1] for i in
                       range(1, len(inertia_values))]
        norm_diff = differences / np.max(differences)
        elbow_cluster = np.argmax(norm_diff) + 1

        return elbow_cluster

    # Пример использования функ
    def _generate_layout(self):
        met_list = []
        metrics_method = {
            0: self._generate_kmeans(),
            1: self._generate_hirarchy(),
            2: self._generate_component(),
        }
        for metric in metrics_method:
            if metric == self.settings['method']:
                met_list.append(metrics_method[metric])

        # def _generate_kmeans(self):
        #     print(np.__version__)
        #     df = self.pp.get_numeric_df(self.settings['data'])
        #     print(self.pp.get_categorical_df(self.settings['data']))
        #     df = (df - df.mean()) / df.std()
        #     x = list(df.values.tolist())
        #     p = 2
        #     r = 1
        #     df = self.settings['data']
        #     print(df.dtypes)
        #
        #     # create function to calculate Manhattan distance
        #     def user1(a, b):
        #         ch = sum((val1 * val2) for val1, val2 in zip(a, b))
        #         z1 = sum((val1 ** 2) for val1, val2 in zip(a, b))
        #         z2 = sum((val2 ** 2) for val1, val2 in zip(a, b))
        #         return 1 - ch / (z1 * z2)
        #
        #     def user2(a, b):
        #         global p, r
        #         ch = (sum((val1 - val2) ** p for val1, val2 in zip(a, b))) ** (
        #                 1 / r)
        #         return ch
        #
        #     def k_znach(n, met):
        #         try:
        #             n = int(n)
        #             metric = int(met)
        #             mini = 10000000
        #             clusters = []
        #             final = []
        #             begin = []
        #             for i in range(len(x) * n):
        #                 initial_centers = kmeans_plusplus_initializer(x,
        #                                                               n).initialize()
        #                 if metric == 0:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.EUCLIDEAN))
        #                 elif metric == 1:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.EUCLIDEAN_SQUARE))
        #                 elif metric == 2:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.MANHATTAN))
        #                 elif metric == 3:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.CHEBYSHEV))
        #                 elif metric == 4:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.USER_DEFINED,
        #                                                  func=user1))
        #                 elif metric == 5:
        #                     kmeans_instance = kmeans(x, initial_centers,
        #                                              metric=distance_metric(
        #                                                  type_metric.USER_DEFINED,
        #                                                  func=user2))
        #
        #                 kmeans_instance.process()
        #                 if kmeans_instance.get_total_wce() < mini:
        #                     clusters = kmeans_instance.get_clusters()
        #                     final = kmeans_instance.get_centers()
        #                     begin = initial_centers
        #                     mini = kmeans_instance.get_total_wce()
        #             num = []
        #             for i in clusters:
        #                 num.append(len(i))
        #             clusters2 = []
        #             for i in clusters:
        #                 l = ""
        #                 for j in range(len(i)):
        #                     l += str(i[j]) + " "
        #                 clusters2.append(l)
        #             print(final)
        #             fin2 = []
        #             for i in range(len(final)):
        #                 l = []
        #                 for j in range(len(final[i])):
        #                     l += str(round(final[i][j], 2)) + " "
        #                 fin2.append(l)
        #
        #             beg2 = []
        #             for i in range(len(begin)):
        #                 l = []
        #                 for j in range(len(begin[i])):
        #                     l += str(round(begin[i][j], 2)) + " "
        #                 beg2.append(l)
        #
        #             d = {"Кластер": pd.Series(clusters2,
        #                                       index=[i for i in range(n)]),
        #                  "Число элементов в кластере": pd.Series(num,
        #                                                          index=[i for i
        #                                                                 in
        #                                                                 range(
        #                                                                     n)]),
        #                  "Начальные центры кластера": pd.Series(beg2,
        #                                                         index=[i for i
        #                                                                in range(
        #                                                                 n)]),
        #                  "Конечные центры кластера": pd.Series(fin2,
        #                                                        index=[i for i in
        #                                                               range(
        #                                                                   n)])}
        #             df1 = pd.DataFrame(d)
        #             print(df1)
        #             return df1
        #         except ValueError:
        #             print("")
        #
        #     metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev',
        #                'cosine', 'power']

    def _generate_kmeans(self):
        print(np.__version__)
        df = self.pp.get_numeric_df(self.settings['data'])
        print(self.pp.get_categorical_df(self.settings['data']))
        df = (df - df.mean()) / df.std()
        x = list(df.values.tolist())
        p = 2
        r = 1
        df = self.settings['data']
        print(df.dtypes)

        # create function to calculate Manhattan distance
        def user1(a, b):
            ch = sum((val1 * val2) for val1, val2 in zip(a, b))
            z1 = sum((val1 ** 2) for val1, val2 in zip(a, b))
            z2 = sum((val2 ** 2) for val1, val2 in zip(a, b))
            return 1 - ch / (z1 * z2)

        def user2(a, b):
            global p, r
            ch = (sum((val1 - val2) ** p for val1, val2 in zip(a, b))) ** (
                    1 / r)
            return ch

        def k_znach(n, met):
            try:
                n = int(n)
                metric = int(met)
                mini = 10000000
                clusters = []
                final = []
                begin = []
                for i in range(len(x) * n):
                    initial_centers = kmeans_plusplus_initializer(x,
                                                                  n).initialize()
                    if metric == 0:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.EUCLIDEAN))
                    elif metric == 1:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.EUCLIDEAN_SQUARE))
                    elif metric == 2:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.MANHATTAN))
                    elif metric == 3:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.CHEBYSHEV))
                    elif metric == 4:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.USER_DEFINED,
                                                     func=user1))
                    elif metric == 5:
                        kmeans_instance = kmeans(x, initial_centers,
                                                 metric=distance_metric(
                                                     type_metric.USER_DEFINED,
                                                     func=user2))

                    kmeans_instance.process()
                    if kmeans_instance.get_total_wce() < mini:
                        clusters = kmeans_instance.get_clusters()
                        final = kmeans_instance.get_centers()
                        begin = initial_centers
                        mini = kmeans_instance.get_total_wce()
                num = []
                for i in clusters:
                    num.append(len(i))
                clusters2 = []
                for i in clusters:
                    l = ""
                    for j in range(len(i)):
                        l += str(i[j]) + " "
                    clusters2.append(l)
                print(final)
                fin2 = []
                for i in range(len(final)):
                    l = []
                    for j in range(len(final[i])):
                        l += str(round(final[i][j], 2)) + " "
                    fin2.append(l)

                beg2 = []
                for i in range(len(begin)):
                    l = []
                    for j in range(len(begin[i])):
                        l += str(round(begin[i][j], 2)) + " "
                    beg2.append(l)

                d = {"Кластер": pd.Series(clusters2,
                                          index=[i for i in range(n)]),
                     "Число элементов в кластере": pd.Series(num,
                                                             index=[i for i
                                                                    in
                                                                    range(
                                                                        n)]),
                     "Начальные центры кластера": pd.Series(beg2,
                                                            index=[i for i
                                                                   in range(
                                                                    n)]),
                     "Конечные центры кластера": pd.Series(fin2,
                                                           index=[i for i in
                                                                  range(
                                                                      n)])}
                df1 = pd.DataFrame(d)
                print(df1)
                return df1
            except ValueError:
                print("")

        metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev',
                   'cosine', 'power']
