import numpy as np
import pandas as pd
import sklearn.metrics as sm
from scipy import stats
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .ModelInterface import Model


class BaseModel(Model):

    def __init__(self, math_model_class, x, y, extra_param=None):

        if math_model_class == DecisionTreeClassifier:
            self.model = math_model_class(max_depth=extra_param[0], min_samples_split=extra_param[1],
                                          max_features=extra_param[2])
        else:
            self.model = math_model_class()
        self.math_model_class = math_model_class
        super().__init__(x, y)

    def score(self) -> float:
        return self.model.score(self.x, self.y)

    def get_resid(self) -> np.array:
        return self.model.coef_

    def predict(self, x: np.array) -> float:  # предсказанное значение для числа или списка
        return self.model.predict(x)

    def get_intercept(self):  # коэффициент пересечения
        return self.model.intercept_

    def get_all_coef(self):  # коэффициенты с пересечением
        return np.append(self.model.intercept_, self.model.coef_)

    def make_X(self, def_df, def_names):  # создаёт датафрейм признаков
        df1 = pd.DataFrame()
        for name in def_names:
            df1 = pd.concat([df1, def_df[name]], axis=1)
        return df1

    def make_Y(self, def_df, def_name):  # создаёт массив зависимой переменной
        return def_df[def_name]

    def get_mean(self, def_df_Y):  # среднее значение Y
        return sum(def_df_Y) / len(def_df_Y)

    def get_TSS(self, def_df_Y, def_mean_Y):  # дисперсия Y
        def_TSS = 0
        for i in range(len(def_df_Y)):
            def_TSS += (def_df_Y[i] - def_mean_Y) ** 2
        return def_TSS

    def get_RSS(self, def_predict_Y, def_mean_Y):  # доля объяснённой дисперсии
        def_RSS = 0
        for i in range(len(def_predict_Y)):
            def_RSS += (def_predict_Y[i] - def_mean_Y) ** 2
        return def_RSS

    def get_ESS(self, def_df_Y, def_predict_Y):  # доля необъяснённой дисперсии
        def_ESS = 0
        for i in range(len(def_df_Y)):
            def_ESS += (def_df_Y[i] - def_predict_Y[i]) ** 2
        return def_ESS

    def get_R(self, def_df_Y, def_predict_Y):  # коэффицент множественной корреляции
        return sm.r2_score(def_df_Y, def_predict_Y) ** 0.5

    def get_deg_fr(self, def_df_X):  # степени свободы в списке
        k1 = def_df_X.shape[1]
        k2 = def_df_X.shape[0] - def_df_X.shape[1] - 1
        return [k1, k2]

    def get_st_err(self, def_RSS, def_de_fr):  # стандартная ошибка оценки уравнения
        return (def_RSS / (def_de_fr[1] - 2)) ** 0.5

    def get_cov_matrix(self, def_df_X):  # обратная ковариационная матрица
        df2_X = def_df_X.copy()
        df2_X.insert(0, '1', np.ones((df2_X.shape[0], 1)))
        df2_X_T = df2_X.values.transpose()
        return np.linalg.pinv(np.dot(df2_X_T, df2_X))

    # обратная ковариационная матрица для расстояний Махалонобиса
    def get_cov_matrix_2(self, df_X):
        df2_X = df_X.copy()
        df2_X_T = df2_X.values.transpose()
        return np.linalg.inv(np.dot(df2_X_T, df2_X))

    def uravnenie(self, def_b, def_names, def_name):  # уравнение регрессии
        def_st = 'Y = ' + str(round(def_b[0], 3))
        for i in range(1, len(def_b)):
            if def_b[i] > 0:
                def_st += ' + ' + str(round(def_b[i], 3)) + 'X(' + str(i) + ')'
            else:
                def_st += ' - ' + \
                          str(round(abs(def_b[i]), 3)) + 'X(' + str(i) + ')'
        def_st += ', где:'  # \nX(0)-константа'
        uravlist = [def_st]
        uravlist.append('\n')
        uravlist.append('Y - ' + def_name + ';')
        for i in range(1, len(def_b)):
            uravlist.append('\n')
            uravlist.append(f'X({i}) - {def_names[i - 1]};')
        return uravlist

    def st_coef(self, def_df_X, def_TSS, b):  # стандартизованнные коэффициенты
        def_b = list(b)
        def_b.pop(0)
        b_st = []
        for i in range(len(def_b)):
            a = def_df_X.iloc[:, i]
            mean_X = self.get_mean(a)
            sx = self.get_TSS(a.tolist(), mean_X)
            b_st.append(def_b[i] * (sx / def_TSS) ** 0.5)
        return b_st

    def st_er_coef(self, def_df_Y, def_predict_Y, def_cov_mat):  # стандартные ошибки
        def_MSE = np.mean((def_df_Y - def_predict_Y.T) ** 2)
        var_est = def_MSE * np.diag(def_cov_mat)
        SE_est = np.sqrt(var_est)
        return SE_est

    def t_stat(self, def_df_X, def_df_Y, def_predict_Y, def_d_free, def_b):  # t-критерии коэффициентов
        s = np.sum((def_predict_Y - def_df_Y) ** 2) / (def_d_free[1] + 1)
        df2_X = def_df_X.copy()
        df2_X.insert(0, '1', np.ones((df2_X.shape[0], 1)))
        sd = np.sqrt(s * (np.diag(np.linalg.pinv(np.dot(df2_X.T, df2_X)))))
        def_t_stat = []
        for i in range(len(def_b)):
            def_t_stat.append(def_b[i] / sd[i])
        return def_t_stat

    def get_RMSD(self, def_df_Y, def_predict_Y):  # корень из среднеквадратичной ошибки
        return np.sqrt(sm.mean_squared_error(def_df_Y, def_predict_Y))

    def get_MSE(self, def_df_Y, def_predict_Y):  # среднеквадратичная ошибка
        return sm.mean_squared_error(def_df_Y, def_predict_Y)

    def get_MAE(self, def_df_Y, def_predict_Y):  # средняя абсолютная ошибка
        return sm.mean_absolute_error(def_df_Y, def_predict_Y)

    def get_R2_adj(self, def_df_X, def_df_Y, def_predict_Y):  # R^2 adjusted
        return 1 - (1 - sm.r2_score(def_df_Y, def_predict_Y)) * (
                (len(def_df_X) - 1) / (len(def_df_X) - def_df_X.shape[1] - 1))

    def get_Fst(self, def_df_X, def_df_Y, def_predict_Y):  # F-статистика
        r2 = sm.r2_score(def_df_Y, def_predict_Y)
        return r2 / (1 - r2) * (len(def_df_X) - def_df_X.shape[1] - 1) / def_df_X.shape[1]

    def p_values(self, def_df_X, def_t_stat):
        newX = pd.DataFrame(
            {"Constant": np.ones(def_df_X.shape[0])}).join(def_df_X)
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) -
                                                     len(newX.columns) - 1))) for i in def_t_stat]
        return p_values

    def get_classes(self):
        if self.math_model_class == DecisionTreeClassifier:
            return self.model.classes_
        else:
            return 0
