import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc


class BioquivalenceMathsModel:
    def get_describe(self, x) -> tuple:
        df = x.mean(axis=0).describe()
        return df

    def get_geom_mean(self, x: np.array) -> float:
        return np.exp(np.log(x).mean())

    def get_variation(self, x: np.array) -> float:
        return stats.variation(x)

    def get_auc(self, x: np.array, y: np.array) -> float:
        return auc(x, y)

    def get_log_array(self, x: np.array) -> np.array:
        return np.log(x)

    def get_kstest(self, x: np.array) -> tuple:
        x = (x - np.mean(x)) / np.std(x)
        return stats.kstest(x, "norm")

    def get_shapiro(self, x: np.array) -> tuple:
        return stats.shapiro(x)

    def get_f(self, x: np.array, y: np.array) -> tuple:
        return stats.f_oneway(x, y)

    def get_levene(self, x: np.array, y: np.array) -> tuple:
        lx = []
        for i in range(x.size):
            lx.append(float(x[i]))
        ly = []
        for i in range(y.size):
            ly.append(float(y[i]))
        return stats.levene(lx, ly)

    def get_k_el(self, x: np.array, y: np.array) -> float:
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        return np.polyfit(x[-3:], y[-3:], deg=1)[0]

    def get_anova(self, x: np.array, y: np.array, z: np.array) -> tuple:
        ssb = (
            x.size * (np.mean(x) - np.mean(z)) ** 2
            + y.size * (np.mean(y) - np.mean(z)) ** 2
        )
        sse = np.sum((x - np.mean(x)) ** 2) + np.sum((y - np.mean(y)) ** 2)
        sst = np.sum((z - np.mean(z)) ** 2)
        data = {
            "SS": [ssb, sse, sst],
            "df": [1, z.size - 2, z.size - 1],
            "MS": [ssb, sse / (z.size - 2), "-"],
            "F": [ssb / (sse / (z.size - 2)), "-", "-"],
            "F крит.": [stats.f.ppf(1 - self.alpha, 1, z.size - 2), "-", "-"],
        }
        df = pd.DataFrame(data)
        res = ssb / (sse / (z.size - 2)) < stats.f.ppf(1 - self.alpha, 1, z.size - 2)
        return df, res

    def get_oneside_eq(self, x: np.array, y: np.array, df: pd.DataFrame) -> tuple:
        dft = stats.t.ppf(1 - self.alpha, x.size + y.size - 2)
        left = float(
            np.mean(x)
            - np.mean(y)
            - dft * (4 * df.iloc[1, 2] / (x.size + y.size)) ** (1 / 2)
        )
        right = float(
            np.mean(x)
            - np.mean(y)
            + dft * (4 * df.iloc[1, 2] / (x.size + y.size)) ** (1 / 2)
        )
        return left, right

    def get_oneside_noteq(self, x: np.array, y: np.array, df: pd.DataFrame) -> tuple:
        dft = stats.t.ppf(1 - self.alpha / 2, x.size + y.size - 2)
        left = float(
            np.mean(x)
            - np.mean(y)
            - dft * (4 * df.iloc[1, 2] / (x.size + y.size)) ** (1 / 2)
        )
        right = float(
            np.mean(x)
            - np.mean(y)
            + dft * (4 * df.iloc[1, 2] / (x.size + y.size)) ** (1 / 2)
        )
        return left, right

    def create_auc(self, df: pd.DataFrame) -> np.array:
        time = np.array(df.columns)
        aucс = df.apply(lambda row: pd.Series({"auc": auc(time, row)}), axis=1)
        return np.array(aucс)

    def create_auc_infty(self, df: pd.DataFrame) -> np.array:
        time = np.array(df.columns)
        aucс = df.apply(lambda row: pd.Series({"auc": auc(time, row)}), axis=1)
        auuc = np.array(aucс)
        k_el_divided = df.apply(
            lambda row: pd.Series(
                {"k_el_divided": self.get_k_el(time, row) / row.iloc[-1]}
            ),
            axis=1,
        )
        k_el_divided = np.array(k_el_divided)
        return auuc + k_el_divided

    def log_auc(self):
        self.auc_log = True
        if self.plan == "parallel":
            self.auc_t = self.get_log_array(self.auc_t)
            self.auc_r = self.get_log_array(self.auc_r)
            self.auc = np.concatenate((self.auc_t, self.auc_r))
        else:
            self.auc_t_1 = self.get_log_array(self.auc_t_1)
            self.auc_r_1 = self.get_log_array(self.auc_r_1)
            self.auc_t_2 = self.get_log_array(self.auc_t_2)
            self.auc_r_2 = self.get_log_array(self.auc_r_2)

    def get_bartlett(self, x: np.array, y: np.array) -> tuple:
        return stats.bartlett(x, y)

    def two_factor_anova(
        self, t_1: np.array, r_1: np.array, t_2: np.array, r_2: np.array
    ) -> tuple:
        n = 4 * len(t_1)
        x_a1_mean = 2 * sum(t_1 + r_2) / n
        x_a2_mean = 2 * sum(r_1 + t_2) / n
        x_b1_mean = 2 * sum(t_1 + r_1) / n
        x_b2_mean = 2 * sum(t_2 + r_2) / n
        x_a1_b1_mean = np.mean(t_1)
        x_a2_b1_mean = np.mean(r_1)
        x_a1_b2_mean = np.mean(t_2)
        x_a2_b2_mean = np.mean(r_2)
        x = np.concatenate([t_1, r_1, t_2, r_2])
        x.ravel()
        ss = sum([(i - np.mean(x)) ** 2 for i in x])
        ss_a = (n / 2) * ((x_a1_mean - np.mean(x)) ** 2 + (x_a2_mean - np.mean(x)) ** 2)
        ss_b = (n / 2) * ((x_b1_mean - np.mean(x)) ** 2 + (x_b2_mean - np.mean(x)) ** 2)
        ss_ab = (n / 4) * (
            (x_a1_b1_mean - x_a1_mean - x_b1_mean + np.mean(x)) ** 2
            + (x_a2_b1_mean - x_a2_mean - x_b1_mean + np.mean(x)) ** 2
            + (x_a1_b2_mean - x_a1_mean - x_b2_mean + np.mean(x)) ** 2
            + (x_a2_b2_mean - x_a2_mean - x_b2_mean + np.mean(x)) ** 2
        )
        ss_e = (
            sum([(i - x_a1_b1_mean) ** 2 for i in t_1])
            + sum([(i - x_a2_b1_mean) ** 2 for i in r_1])
            + sum([(i - x_a1_b2_mean) ** 2 for i in t_2])
            + sum([(i - x_a2_b2_mean) ** 2 for i in r_2])
        )
        ms_e = ss_e / (n / 4 - 1)
        data = {
            "SS": [ss_a, ss_b, ss_ab, ss_e, ss],
            "df": [1, 1, 1, len(t_1) - 1, n - 1],
            "MS": [ss_a, ss_b, ss_ab, ms_e, "-"],
            "F": [ss_a / ms_e, ss_b / ms_e, ss_ab / ms_e, "-", "-"],
            "F крит.": [
                stats.f.ppf(1 - self.alpha, 1, 4 * (len(t_1) - 1)),
                stats.f.ppf(1 - self.alpha, 1, 4 * (len(t_1) - 1)),
                stats.f.ppf(1 - self.alpha, 1, 4 * (len(t_1) - 1)),
                "-",
                "-",
            ],
        }
        df = pd.DataFrame(data)
        return df

    def get_crossover_oneside_eq(
        self, x: np.array, y: np.array, df: pd.DataFrame
    ) -> tuple:
        left = float(
            np.mean(x)
            - np.mean(y)
            - stats.t.ppf(1 - self.alpha, df.iloc[3, 1])
            * (2 * df.iloc[3, 2] / (x.size + y.size)) ** (1 / 2)
        )
        right = float(
            np.mean(x)
            - np.mean(y)
            + stats.t.ppf(1 - self.alpha, df.iloc[3, 1])
            * (2 * df.iloc[3, 2] / (x.size + y.size)) ** (1 / 2)
        )
        return left, right

    def get_crossover_oneside_noteq(
        self, x: np.array, y: np.array, df: pd.DataFrame
    ) -> tuple:
        left = float(
            np.mean(x)
            - np.mean(y)
            - stats.t.ppf(1 - self.alpha / 2, df.iloc[3, 1])
            * (2 * df.iloc[3, 2] / (x.size + y.size)) ** (1 / 2)
        )
        right = float(
            np.mean(x)
            - np.mean(y)
            + stats.t.ppf(1 - self.alpha / 2, df.iloc[3, 1])
            * (2 * df.iloc[3, 2] / (x.size + y.size)) ** (1 / 2)
        )
        return left, right

    def __init__(self, settings: dict, data: dict):
        self.plan = settings["design"]
        self.alpha = 0.05
        if self.plan == "parallel":
            self.concentration_t = data["concentration_t"]
            self.concentration_r = data["concentration_r"]
            self.check_normal = settings["normality"]
            self.check_uniformity = settings["uniformity"]
            self.kstest_t = 0
            self.kstest_r = 0
            self.shapiro_t = 0
            self.shapiro_r = 0
            self.f = 0
            self.levene = 0
            self.anova = 0
            self.oneside_eq = 0
            self.oneside_noteq = 0
            self.auc_t = 0
            self.auc_r = 0
            self.auc_t_notlog = 0
            self.auc_r_notlog = 0
            self.auc_log = False
            self.auc = 0
            self.auc_t_infty = 0
            self.auc_r_infty = 0
            self.auc_t_infty_log = 0
            self.auc_r_infty_log = 0
            # statistics
            self.mean_t = 0
            self.mean_r = 0
            self.std_t = 0
            self.std_r = 0
            self.min_t = 0
            self.min_r = 0
            self.max_t = 0
            self.max_r = 0
            self.median_t = 0
            self.median_r = 0
            self.variation_t = 0
            self.variation_r = 0
            self.geom_mean_t = 0
            self.geom_mean_r = 0
        if self.plan == "cross":
            self.concentration_t_1 = data["concentration_t_1"]
            self.concentration_r_1 = data["concentration_r_1"]
            self.concentration_t_2 = data["concentration_t_2"]
            self.concentration_r_2 = data["concentration_r_2"]
            self.check_normal = settings["normality"]
            self.kstest_t_1 = 0
            self.kstest_r_1 = 0
            self.kstest_t_2 = 0
            self.kstest_r_2 = 0
            self.shapiro_t_1 = 0
            self.shapiro_r_1 = 0
            self.shapiro_t_2 = 0
            self.shapiro_r_2 = 0
            self.auc_t_1 = 0
            self.auc_r_1 = 0
            self.auc_t_2 = 0
            self.auc_r_2 = 0
            self.auc_t_1_notlog = 0
            self.auc_r_1_notlog = 0
            self.auc_t_2_notlog = 0
            self.auc_r_2_notlog = 0
            self.auc_t_1_infty = 0
            self.auc_r_1_infty = 0
            self.auc_t_2_infty = 0
            self.auc_r_2_infty = 0
            self.auc_t_1_infty_log = 0
            self.auc_r_1_infty_log = 0
            self.auc_t_1_infty_log = 0
            self.auc_r_1_infty_log = 0
            self.bartlett_groups = 0
            self.bartlett_period = 0
            self.auc_log = False
            self.anova = 0
            self.oneside_eq = 0
            self.oneside_noteq = 0
            # statistics
            self.mean_t_1 = 0
            self.mean_t_2 = 0
            self.mean_r_1 = 0
            self.mean_r_2 = 0
            self.std_t_1 = 0
            self.std_t_2 = 0
            self.std_r_1 = 0
            self.std_r_2 = 0
            self.min_t_1 = 0
            self.min_t_2 = 0
            self.min_r_1 = 0
            self.min_r_2 = 0
            self.max_t_1 = 0
            self.max_t_2 = 0
            self.max_r_1 = 0
            self.max_r_2 = 0
            self.median_t_1 = 0
            self.median_t_2 = 0
            self.median_r_1 = 0
            self.median_r_2 = 0
            self.variation_t_1 = 0
            self.variation_t_2 = 0
            self.variation_r_1 = 0
            self.variation_r_2 = 0
            self.geom_mean_t_1 = 0
            self.geom_mean_t_2 = 0
            self.geom_mean_r_1 = 0
            self.geom_mean_r_2 = 0

    def run_bio_model(self):
        if self.plan == "parallel":
            if type(self.concentration_t) == pd.DataFrame:
                self.auc_t = self.create_auc(self.concentration_t)
                self.auc_r = self.create_auc(self.concentration_r)
                self.auc_t_notlog = self.auc_t
                self.auc_r_notlog = self.auc_r
                self.auc = np.concatenate((self.auc_t, self.auc_r))
                self.auc_t_infty = self.create_auc_infty(self.concentration_t)
                self.auc_r_infty = self.create_auc_infty(self.concentration_r)
                self.auc_t_infty_log = self.get_log_array(self.auc_t_infty)
                self.auc_r_infty_log = self.get_log_array(self.auc_r_infty)
            if self.check_normal == "Kolmogorov":
                # колмогоров только для стандартного
                self.kstest_t = self.get_kstest(self.auc_t)
                self.kstest_r = self.get_kstest(self.auc_r)
                if self.kstest_t[1] <= self.alpha or self.kstest_r[1] <= self.alpha:
                    self.log_auc()
                    self.kstest_t = self.get_kstest(self.auc_t)
                    self.kstest_r = self.get_kstest(self.auc_r)
            elif self.check_normal == "Shapiro":
                self.shapiro_t = self.get_shapiro(self.auc_t)
                self.shapiro_r = self.get_shapiro(self.auc_r)
                if self.shapiro_t[1] <= self.alpha or self.shapiro_r[1] <= self.alpha:
                    self.log_auc()
                    self.shapiro_t = self.get_shapiro(self.auc_t)
                    self.shapiro_r = self.get_shapiro(self.auc_r)
            if self.check_uniformity == "F":
                self.f = self.get_f(self.auc_t, self.auc_r)
                if self.f[1] <= self.alpha and self.auc_log == False:
                    self.log_auc()
                    self.f = self.get_f(self.auc_t, self.auc_r)
            elif self.check_uniformity == "Leven":
                self.levene = self.get_levene(self.auc_t, self.auc_r)
                if self.levene[1] <= self.alpha and self.auc_log == False:
                    self.log_auc()
                    self.levene = self.get_levene(self.auc_t, self.auc_r)
            if self.auc_log == False:
                self.log_auc()
            # 0 - pd.DataFrame, 1 - bool
            self.anova = self.get_anova(self.auc_t, self.auc_r, self.auc)
            self.oneside_eq = self.get_oneside_eq(self.auc_t, self.auc_r, self.anova[0])
            self.oneside_noteq = self.get_oneside_noteq(
                self.auc_t, self.auc_r, self.anova[0]
            )

            self.mean_r = self.get_describe(self.concentration_r).loc["mean"]
            self.mean_t = self.get_describe(self.concentration_t).loc["mean"]

            self.std_r = self.get_describe(self.concentration_r).loc["std"]
            self.std_t = self.get_describe(self.concentration_t).loc["std"]

            self.min_r = self.get_describe(self.concentration_r).loc["min"]
            self.min_t = self.get_describe(self.concentration_t).loc["min"]

            self.median_r = self.get_describe(self.concentration_r).loc["50%"]
            self.median_t = self.get_describe(self.concentration_t).loc["50%"]

            self.max_r = self.get_describe(self.concentration_r).loc["max"]
            self.max_t = self.get_describe(self.concentration_t).loc["max"]

            self.geom_mean_r = self.get_geom_mean(self.concentration_r.mean(axis=0))
            self.geom_mean_t = self.get_geom_mean(self.concentration_t.mean(axis=0))

            self.variation_r = self.get_variation(self.concentration_r.mean(axis=0))
            self.variation_t = self.get_variation(self.concentration_t.mean(axis=0))
        else:
            self.auc_t_1 = self.create_auc(self.concentration_t_1)
            self.auc_r_1 = self.create_auc(self.concentration_r_1)
            self.auc_t_2 = self.create_auc(self.concentration_t_2)
            self.auc_r_2 = self.create_auc(self.concentration_r_2)
            self.auc_t_1_notlog = self.auc_t_1
            self.auc_r_1_notlog = self.auc_r_1
            self.auc_t_2_notlog = self.auc_t_2
            self.auc_r_2_notlog = self.auc_r_2
            self.auc_t_1_infty = self.create_auc_infty(self.concentration_t_1)
            self.auc_r_1_infty = self.create_auc_infty(self.concentration_r_1)
            self.auc_t_2_infty = self.create_auc_infty(self.concentration_t_2)
            self.auc_r_2_infty = self.create_auc_infty(self.concentration_r_2)
            self.auc_t_1_infty_log = self.get_log_array(self.auc_t_1_infty)
            self.auc_r_1_infty_log = self.get_log_array(self.auc_r_1_infty)
            self.auc_t_2_infty_log = self.get_log_array(self.auc_t_2_infty)
            self.auc_r_2_infty_log = self.get_log_array(self.auc_r_2_infty)
            self.log_auc()
            if self.check_normal == "Kolmogorov":
                self.kstest_t_1 = self.get_kstest(self.auc_t_1)
                self.kstest_r_1 = self.get_kstest(self.auc_r_1)
                self.kstest_t_2 = self.get_kstest(self.auc_t_2)
                self.kstest_r_2 = self.get_kstest(self.auc_r_2)
            elif self.check_normal == "Shapiro":
                self.shapiro_t_1 = self.get_shapiro(self.auc_t_1)
                self.shapiro_r_1 = self.get_shapiro(self.auc_r_1)
                self.shapiro_t_2 = self.get_shapiro(self.auc_t_2)
                self.shapiro_r_2 = self.get_shapiro(self.auc_r_2)
            self.bartlett_groups = self.get_bartlett(
                np.concatenate((self.auc_t_1, self.auc_r_1)).ravel(),
                np.concatenate((self.auc_t_2, self.auc_r_2)).ravel(),
            )
            self.bartlett_period = self.get_bartlett(
                np.concatenate((self.auc_t_1, self.auc_r_2)).ravel(),
                np.concatenate((self.auc_r_1, self.auc_t_2)).ravel(),
            )
            self.anova = self.two_factor_anova(
                self.auc_t_1, self.auc_r_1, self.auc_t_2, self.auc_r_2
            )
            self.oneside_eq = self.get_crossover_oneside_eq(
                np.concatenate((self.auc_t_1, self.auc_t_2)).ravel(),
                np.concatenate((self.auc_r_1, self.auc_r_2)).ravel(),
                self.anova,
            )
            self.oneside_noteq = self.get_crossover_oneside_noteq(
                np.concatenate((self.auc_t_1, self.auc_t_2)).ravel(),
                np.concatenate((self.auc_r_1, self.auc_r_2)).ravel(),
                self.anova,
            )

            self.mean_r_1 = self.get_describe(self.concentration_r_1).loc["mean"]
            self.mean_t_1 = self.get_describe(self.concentration_t_1).loc["mean"]
            self.mean_r_2 = self.get_describe(self.concentration_r_2).loc["mean"]
            self.mean_t_2 = self.get_describe(self.concentration_t_2).loc["mean"]

            self.std_r_1 = self.get_describe(self.concentration_r_1).loc["std"]
            self.std_t_1 = self.get_describe(self.concentration_t_1).loc["std"]
            self.std_r_2 = self.get_describe(self.concentration_r_2).loc["std"]
            self.std_t_2 = self.get_describe(self.concentration_t_2).loc["std"]

            self.min_r_1 = self.get_describe(self.concentration_r_1).loc["min"]
            self.min_t_1 = self.get_describe(self.concentration_t_1).loc["min"]
            self.min_r_2 = self.get_describe(self.concentration_r_2).loc["min"]
            self.min_t_2 = self.get_describe(self.concentration_t_2).loc["min"]

            self.median_r_1 = self.get_describe(self.concentration_r_1).loc["50%"]
            self.median_t_1 = self.get_describe(self.concentration_t_1).loc["50%"]
            self.median_r_2 = self.get_describe(self.concentration_r_2).loc["50%"]
            self.median_t_2 = self.get_describe(self.concentration_t_2).loc["50%"]

            self.max_r_1 = self.get_describe(self.concentration_r_1).loc["max"]
            self.max_t_1 = self.get_describe(self.concentration_t_1).loc["max"]
            self.max_r_2 = self.get_describe(self.concentration_r_2).loc["max"]
            self.max_t_2 = self.get_describe(self.concentration_t_2).loc["max"]

            self.geom_mean_r_1 = self.get_geom_mean(self.concentration_r_1.mean(axis=0))
            self.geom_mean_t_1 = self.get_geom_mean(self.concentration_t_1.mean(axis=0))
            self.geom_mean_r_2 = self.get_geom_mean(self.concentration_r_2.mean(axis=0))
            self.geom_mean_t_2 = self.get_geom_mean(self.concentration_t_2.mean(axis=0))

            self.variation_r_1 = self.get_variation(self.concentration_r_1.mean(axis=0))
            self.variation_t_1 = self.get_variation(self.concentration_t_1.mean(axis=0))
            self.variation_r_2 = self.get_variation(self.concentration_r_2.mean(axis=0))
            self.variation_t_2 = self.get_variation(self.concentration_t_2.mean(axis=0))
