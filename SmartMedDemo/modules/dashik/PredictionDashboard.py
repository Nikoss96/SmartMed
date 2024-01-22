import re

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash import callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


from sklearn import tree
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as sps
import sklearn.metrics as sm
from scipy.sparse import issparse
# from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import KBinsDiscretizer
import statsmodels.api as smapi
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
from PIL import Image
import os

from scipy.stats import binomtest
from scipy.stats import f
from statsmodels.stats.proportion import proportion_confint
# from GUI.apps.PredictionApp.utils import read_file
from ..dataprep.PandasPreprocessor import read_file


from .text.linear_text import *
from .text.roc_text import *
from .text.tree_text import *
from .text.log import *
from .DashExceptions import ModelChoiceException
from .Dashboard import Dashboard
from .text.linear_text import *
from .text.roc_text import *
from ..models.LinearRegressionModel import *
from ..models.LogisticRegressionModel import *
from ..models.TreeModel import *
from ..dataprep.PandasPreprocessor import read_file

from ..dataprep.PandasPreprocessor import read_file

global ansy

class PredictionDashboard(Dashboard):

    def __init__(self):
        # self.settings = {}
        super().__init__()

    def _generate_layout(self):
        if self.settings['model'] == 'linreg':
            return LinearRegressionDashboard(self).get_layout()
        elif self.settings['model'] == 'logreg':
            return LogisticRegressionDashboard(self).get_layout()
        elif self.settings['model'] == 'roc':
            return ROC(self).get_layout()
        elif self.settings['model'] == 'polynomreg':
            return PolynomRegressionDashboard(self).get_layout()
        elif self.settings['model'] == 'tree':
            return TreeDashboard(self).get_layout()
        else:
            raise ModelChoiceException


class LinearRegressionDashboard(Dashboard):

    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict
        self.coord_list = []

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        global ansy
        ansy = ''
        metrics_list = []
        metrics_method = {
            'model_quality': self._generate_quality(),
            'signif': self._generate_signif(),
            'resid': self._generate_resid(),
            'equation': self._generate_equation(),
            'distrib_resid': self._generate_distrib()
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])

        # for metrics in self.predict.settings['metrics']:
        #    metrics_list.append(metrics_method[metrics])

        metrics = self.predict.pp.df.columns
        return html.Div([
            html.Div(html.H1(children='Множественная регрессия'), style={'text-align': 'center'}),
            dcc.Markdown(children="Задачей множественной линейной регрессии является построение линейной модели связи между набором независимых переменных (предикторов) и зависимой переменной", style={'text-align': 'center', 'padding': '20px'}),
            html.Div(metrics_list)])


    # графики
    def _generate_distrib(self):

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(ansy)
        df_Y = y
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        predict_Y = results.predict()
        # График распределения остатков
        fig_rasp_2 = go.Figure()
        df_ost_2 = pd.DataFrame({'Изначальный Y': df_Y, 'Предсказанный Y': predict_Y})
        fig_rasp_2 = px.scatter(df_ost_2, x="Изначальный Y", y="Предсказанный Y",
                                trendline="ols", trendline_color_override='red', labels='Данные')
        fig_rasp_2.update_traces(marker_size=20)

        fig = go.Figure(
            data=go.Histogram(
                x=df_Y - predict_Y,
                name='Остатки'
            )
        )

        fig.add_trace(
            go.Histogram(
                x=np.random.normal(0, 1, len(df_Y)),
                name='Нормальное распределение'
            )
        )
        fig.update_xaxes(title='Остатки')
        fig.update_layout(bargap=0.1)

        # специфичность

        residuals = df_Y - predict_Y
        num_divisions = residuals.shape[0] + 1
        quantiles = np.arange(1, residuals.shape[0]) / num_divisions

        qq_x_data = sps.norm.ppf(quantiles)
        qq_y_data = np.sort(residuals)

        line_x0 = sps.norm.ppf(0.25)
        line_x1 = sps.norm.ppf(0.75)
        line_y0 = np.quantile(residuals, 0.25)
        line_y1 = np.quantile(residuals, 0.75)
        slope = (line_y1 - line_y0) / (line_x1 - line_x0)
        line_intercept = line_y1 - (slope * line_x1)
        x_range_line = np.arange(-3, 3, 0.001)
        y_values_line = (slope * x_range_line) + line_intercept
        fig_qqplot = go.Figure()
        fig_qqplot.add_trace(
            go.Scatter(
                x=qq_x_data,
                y=qq_y_data,
                mode='markers',
                marker={'color': 'blue'},
                name='Остатки')
        )
        fig_qqplot.add_trace(
            go.Scatter(
                x=x_range_line,
                y=y_values_line,
                mode='lines',
                marker={'color': 'red'},
                name='Нормальное распределение'))
        fig_qqplot['layout'].update(
            xaxis={
                'title': 'Теоретические квантили',
                'zeroline': True},
            yaxis={
                'title': 'Экспериментальные квантили'},
            showlegend=True,
        )
        graph_styles = {
            'text-align': 'center',
            'width': '78%',
            'display': 'inline-block',
            'border-color': 'rgb(220, 220, 220)',
            'border-style': 'solid'
        }
        return html.Div([html.Div(html.H2(children='Графики остатков'), style={'text-align': 'center'}),
                         html.Div([
                             html.Div(
                                 html.H4(children='Гистограмма распределения остатков'),
                                 style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_1', figure=fig), style=graph_styles),
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График соответствия предсказанных значений зависимой переменной '
                                                  'и исходных значений'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_2', figure=fig_rasp_2), style=graph_styles),
                             html.Div(dcc.Markdown(markdown_graph))
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График квантиль-квантиль'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='graph_qqplot', figure=fig_qqplot), style=graph_styles),
                         ], style={'margin': '50px'}),


                         ], style={'margin': '50px'})

    # уравнение
    def _generate_equation(self):
        names = self.predict.settings['x'] + []
        name_Y = self.predict.settings['y']
        b = self.predict.model.get_all_coef()

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()

        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        b = results.params[::-1]
        uravnenie = LinearRegressionModel.uravnenie(
            self.predict.model, b, names, name_Y)
        df_X = self.predict.df_X_test
        b = self.predict.model.get_all_coef()
        b = results.params[::-1]

        def update_output(n_clicks, input1):
            number = len(self.coord_list)
            if n_clicks == 0 or input1 == 'Да':
                self.coord_list = []
                number = len(self.coord_list)
                return u'''Введите значение параметра "{}"'''.format(df_X.columns[0])
            if re.fullmatch(r'^([-+])?\d+([,.]\d+)?$', input1):
                number += 1
                if input1.find(',') > 0:
                    input1 = float(input1[0:input1.find(
                        ',')] + '.' + input1[input1.find(',') + 1:len(input1)])
                self.coord_list.append(float(input1))
                if len(self.coord_list) < len(df_X.columns):
                    return u'''Введите значение параметра  "{}".'''.format(df_X.columns[number])
                    # максимальное значение - len(df_X.columns)-1
                if len(self.coord_list) == len(df_X.columns):
                    number = -1
                    yzn = b[0]
                    for i in range(len(self.coord_list)):
                        yzn += self.coord_list[i] * b[i + 1]
                    return u'''Предсказанное значение равно {} \n Если желаете посчитать ещё для одного набор признаков
                    , напишите "Да".'''.format(round(yzn, 3))
            elif n_clicks > 0:
                return u'''Введено не число, введите значение параметра "{}" повторно.'''.format(df_X.columns[number])
            if number == -1 and input1 != 0 and input1 != 'Да' and input1 != '0':
                return u'''Если желаете посчитать ещё для {} набор признаков, напишите "Да".'''.format('одного')

        self.predict.app.callback(dash.dependencies.Output('output-state', 'children'),
                                  [dash.dependencies.Input(
                                      'submit-button-state', 'n_clicks')],
                                  [dash.dependencies.State('input-1-state', 'value')])(update_output)
        return html.Div([html.Div(html.H2(children='Уравнение множественной регрессии'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dcc.Markdown(id='Markdown', children=uravnenie)),
                                   html.Div(html.H4(children='Предсказание новых значений'),
                                            style={'text-align': 'center'}),
                                   dcc.Markdown(children='Чтобы получить значение зависимой переменной, '
                                                         'введите значение независимых признаков ниже:'),
                                   dcc.Input(id='input-1-state',
                                             type='text', value=''),
                                   html.Button(id='submit-button-state',
                                               n_clicks=0, children='Submit'),
                                   html.Div(id='output-state', children='')],
                                  style={'width': '78%', 'display': 'inline-block',
                                         'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                         'padding': '5px'})
                         ], style={'margin': '50px'})

    # качество модели
    def _generate_quality(self):
        df_result_1 = pd.DataFrame(
            columns=['Параметр', 'R', 'R2', 'R2adj', 'df', 'Fst', 'St.Error'])

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()
        print(y)
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        # df_Y = self.predict.df_Y_test
        # df_X = self.predict.df_X_test
        # predict_Y = LinearRegressionModel.predict(
        #     self.predict.model, self.predict.df_X_test)
        # mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y)
        # RSS = LinearRegressionModel.get_RSS(
        #     self.predict.model, predict_Y, mean_Y)
        # de_fr = LinearRegressionModel.get_deg_fr(
        #     self.predict.model, self.predict.df_X_test)
        df_result_1.loc[1] = ['Значение', round(results.rsquared ** 0.5, 3),
                              round(results.rsquared, 3),
                              round(results.rsquared_adj, 3),
                              results.df_resid,
                              round(results.fvalue, 3),
                              round(results.mse_resid ** 0.5, 3)
                              ]

        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table1',
                             columns=[{"name": i, "id": i}
                                      for i in df_result_1.columns],
                             data=df_result_1.to_dict('records'),
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_1.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table1))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    # таблица остатков
    def _generate_resid(self):

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()  # исходные значения
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        znach = results.predict()  # рассчитанное значение
        ost = []  # остатки
        for i in range(len(znach)):
            ost.append(y[i] - znach[i])
        influence = results.get_influence()
        standardized_residuals = influence.resid_studentized_internal  # стз остатки

        df_result_3 = pd.DataFrame({'Номер наблюдения': 0, 'Исходное значение признака': y,
                                    'Рассчитанное значение признака': znach, 'Остатки': ost,
                                    'Стандартизированные остатки': standardized_residuals})
        df_result_3.iloc[:, 0] = [i + 1 for i in range(df_result_3.shape[0])]
        return html.Div([html.Div(html.H2(children='Таблица остатков'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table3',
                             data=df_result_3.to_dict('records'),
                             columns=[{"name": i, "id": i}
                                      for i in df_result_3.columns],
                             # tooltip_header={i: i for i in df.columns}, #
                             # либо этот, либо тот что ниже
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_3.columns},
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },

                             # asdf
                             page_size=20,
                             fixed_rows={'headers': True},
                             style_table={'height': '330px',
                                          'overflowY': 'auto'},
                             tooltip_delay=0,
                             tooltip_duration=None,
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_3.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

    # таблица критериев значимости переменных
    def _generate_signif(self):
        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()
        print(y)
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        print(X)
        results = reg_m(y, X)
        print(reg_m(y, X).summary2())
        res_b = results.params
        print(res_b)
        res_errb = results.bse
        print(res_errb)
        res_tst = results.tvalues
        print(res_tst)
        res_pval = results.pvalues
        print(res_pval)
        df_column = list(self.predict.df_X_test.columns)
        df_column.append("const")
        print(df_column)
        df_result_2 = pd.DataFrame({'Название переменной': df_column,
                                    'b': res_b,
                                    'St.Error b': res_errb,
                                    't-критерий': res_tst,
                                    'p-value': res_pval})

        return html.Div([html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table2',
                             columns=[{"name": i, "id": i}
                                      for i in df_result_2.columns],
                             data=df_result_2.to_dict('records'),
                             style_table={'textOverflowX': 'ellipsis', },
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_2.columns},
                             tooltip_data=[
                                 {
                                     column: {'value': str(value), 'type': 'markdown'}
                                     for column, value in row.items()
                                 } for row in df_result_2.to_dict('records')
                             ],
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                             export_format='xlsx'

                         ), style={'width': str(len(df_result_2.columns) * 6) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table2))],  # style={'margin': '50px'},
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})


class LogisticRegressionDashboard(Dashboard):
    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict
        self.coord_list = []

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        metrics_list = [self._generate_matrix()]
        metrics_method = {
            'model_quality': self._generate_quality(),
            'signif': self._generate_signif(),
            'resid': self._generate_resid(),
            'distrib': self._generate_distrib()
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])
        metrics_list.append(self._generate_distrib())
        # for metrics in self.predict.settings['metrics']:
        #    metrics_list.append(metrics_method[metrics])
        df_X = self.predict.df_X_test
        if np.any((df_X.data if issparse(df_X) else df_X) < 0):
            return html.Div([html.Div(html.H1(children='Логистическая регрессия'), style={'text-align': 'center'}),
                             dcc.Markdown(children=oprlog, style={'text-align': 'center', 'padding': '20px'}),
                             html.Div(html.H3(children='Выбранная переменная - "{}"'.format(self.predict.settings['y']),
                                              style={'text-align': 'center'})),
                             html.Div(dcc.Markdown(markdown_error),
                                      style={'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)',
                                             'border-style': 'solid', 'padding': '5px'})],
                            style={'margin': '50px'})
        else:
            return html.Div([
                html.Div(html.H1(children='Логистическая регрессия'), style={'text-align': 'center'}),
                dcc.Markdown(children=oprlog, style={'text-align': 'center', 'padding': '20px'}),
                html.Div(html.H3(children='Выбранная переменная - "{}"'.format(self.predict.settings['y']),
                                 style={'text-align': 'center'})),
                html.Div(metrics_list)])

    def _generate_matrix(self):
        X = self.predict.df_X
        print(X)
        y = self.predict.df_Y
        y.loc[y == y.min()] = -2
        y.loc[y == y.max()] = -1
        y.loc[y == -2] = 0
        y.loc[y == -1] = 1
        X['intercept'] = 1.0
        logit = smapi.Logit(y, X)
        results = logit.fit()
        prediction = list(map(round, results.predict(X)))
        mat = confusion_matrix(y, prediction)
        df_matrix = pd.DataFrame(columns=['y_pred\y_true', 'Positive', 'Negative'])
        df_matrix.loc[1] = ['True', mat[1][1], mat[1][0]]
        df_matrix.loc[2] = ['False', mat[0][1], mat[0][0]]
        return html.Div([html.Div(html.H2(children='Матрица классификации'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table_matrix',
                             columns=[{"name": i, "id": i} for i in df_matrix.columns],
                             data=df_matrix.to_dict('records'),
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_matrix.columns},
                             export_format='csv',
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                         ), style={'width': str(len(df_matrix.columns) * 8) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(mistakes), style={'padding': '20px'})
                         ],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    # качество модели
    def _generate_quality(self):
        df_result_1 = pd.DataFrame(columns=['AIC', 'BIC', 'R^2', 'X^2', 'df', 'p'])
        y = self.predict.df_Y
        y.loc[y == y.min()] = -2
        y.loc[y == y.max()] = -1
        y.loc[y == -2] = 0
        y.loc[y == -1] = 1
        df_X = self.predict.df_X
        df_X['intercept'] = 1.0
        logit = smapi.Logit(y, df_X)
        results = logit.fit()
        df_result_1.loc[1] = [round(results.aic, 3),
                              round(results.bic, 3),
                              round(results.prsquared, 3),
                              round(results.llr, 3),
                              round(results.df_model, 3),
                              round(results.llr_pvalue, 3)
                              ]
        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table1',
                             columns=[{"name": i, "id": i} for i in df_result_1.columns],
                             data=df_result_1.to_dict('records'),
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_1.columns},
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             export_format='csv'
                         ), style={'width': str(len(df_result_1.columns) * 8) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(quality), style={'padding': '20px'})
                         ],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    # таблица остатков
    def _generate_resid(self):
        df_matrix = pd.DataFrame(columns=['Accuracy', 'Specificity', 'Sensitivity', 'AUC'])
        y = self.predict.df_Y
        y.loc[y == y.min()] = -2
        y.loc[y == y.max()] = -1
        y.loc[y == -2] = 0
        y.loc[y == -1] = 1
        X = self.predict.df_X
        X['intercept'] = 1.0
        logit = smapi.Logit(y, X)
        results = logit.fit()
        prediction = list(map(round, results.predict(X)))
        cm1 = confusion_matrix(y, prediction)
        total1 = sum(sum(cm1))
        accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        auc = sm.roc_auc_score(y, results.predict(X))
        df_matrix.loc[1] = [round(accuracy1, 3),
                            round(specificity1, 3),
                            round(sensitivity1, 3),
                            round(auc, 3)]

        return html.Div([html.Div(html.H2(children='Прогностические меры'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='new_table_matrix',
                             columns=[{"name": i, "id": i} for i in df_matrix.columns],
                             data=df_matrix.to_dict('records'),
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_matrix.columns},
                             export_format='csv',
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                         ), style={'width': str(len(df_matrix.columns) * 8) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(prognos), style={'padding': '20px'})
                         ],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    def _generate_distrib(self):

        X = self.predict.df_X
        y = self.predict.df_Y
        y.loc[y == y.min()] = -2
        y.loc[y == y.max()] = -1
        y.loc[y == -2] = 0
        y.loc[y == -1] = 1
        X['intercept'] = 1.0
        logit = smapi.Logit(y, X)
        results = logit.fit()
        y_score = results.predict(X)

        fpr, tpr, thresholds = roc_curve(y, y_score)

        fig = px.area(
            x=fpr, y=tpr,

            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=-0.04, x1=1, y0=-0.04, y1=1
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        graph_styles = {
            'text-align': 'center',
            'width': '78%',
            'display': 'inline-block',
            'border-color': 'rgb(220, 220, 220)',
            'border-style': 'solid'
        }
        return html.Div([html.Div(html.H2(children='Графики'), style={'text-align': 'center'}),
                         html.Div([
                             html.Div(
                                 html.H4(children='ROC Curve'),
                                 style={'text-align': 'center'}),
                             html.Div([dcc.Graph(id='Graph_ost_1', figure=fig),
                                       html.Div(dcc.Markdown(roc), style={'padding': '20px'})
                                       ],
                             style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
                                    'border-style': 'solid', 'padding': '5px'}),

                         ])

                         ], style={'margin': '50px'})

    # таблица критериев значимости переменных

    def _generate_signif(self):
        y = self.predict.df_Y
        y.loc[y == y.min()] = -2
        y.loc[y == y.max()] = -1
        y.loc[y == -2] = 0
        y.loc[y == -1] = 1
        df_X = self.predict.df_X
        df_X['intercept'] = 1.0

        # df_Y.loc[(df_Y == df_Y.min()), df_Y.name] = 0
        # df_Y.loc[(df_Y == df_Y.max()), df_Y.name] = 1
        logit = smapi.Logit(y, df_X)
        results = logit.fit()
        res_b = results.params
        res_errb = results.bse
        res_tst = results.tvalues
        res_pval = results.pvalues
        df_column = list(self.predict.df_X_test.columns)
        df_column.append("const")
        df_result_2 = pd.DataFrame({'Название переменной': df_column,
                                    'b': round(res_b, 3),
                                    'St.Error b': round(res_errb, 3),
                                    't-критерий': round(res_tst, 3),
                                    'p-value': round(res_pval, 3)})

        return html.Div([html.Div(html.H2(children='Критерии  переменных'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table2',
                             columns=[{"name": i, "id": i} for i in df_result_2.columns],
                             data=df_result_2.to_dict('records'),
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_2.columns},
                             tooltip_data=[
                                 {
                                     column: {'value': str(value), 'type': 'markdown'}
                                     for column, value in row.items()
                                 } for row in df_result_2.to_dict('records')
                             ],
                         ), style={'width': str(len(df_result_2.columns) * 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(crit), style={'padding': '20px'})],  # style={'margin': '50px'},
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})


class PolynomRegressionDashboard(Dashboard):
    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict
        self.coord_list = []

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        metrics_list = []
        metrics_method = {
            'model_quality': self._generate_quality(),
            'signif': self._generate_signif(),
            'resid': self._generate_resid(),
            'equation': self._generate_equation(),
            'distrib_resid': self._generate_distrib()
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])

        # for metrics in self.predict.settings['metrics']:
        #    metrics_list.append(metrics_method[metrics])

        return html.Div([
            html.Div(html.H1(children='Полиномиальная регрессия'), style={'text-align': 'center'}),
            html.Div(metrics_list)])

    def _generate_quality(self):
        df_result_1 = pd.DataFrame(
            columns=['Параметр', 'R', 'R2', 'R2adj', 'df', 'Fst', 'St.Error'])

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()
        print(y)
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        # df_Y = self.predict.df_Y_test
        # df_X = self.predict.df_X_test
        # predict_Y = LinearRegressionModel.predict(
        #     self.predict.model, self.predict.df_X_test)
        # mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y)
        # RSS = LinearRegressionModel.get_RSS(
        #     self.predict.model, predict_Y, mean_Y)
        # de_fr = LinearRegressionModel.get_deg_fr(
        #     self.predict.model, self.predict.df_X_test)
        df_result_1.loc[1] = ['Значение', round(results.rsquared ** 0.5, 3),
                              round(results.rsquared, 3),
                              round(results.rsquared_adj, 3),
                              results.df_resid,
                              round(results.fvalue, 3),
                              round(results.mse_resid ** 0.5, 3)
                              ]

        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table1',
                             columns=[{"name": i, "id": i}
                                      for i in df_result_1.columns],
                             data=df_result_1.to_dict('records'),
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_1.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table1))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    def _generate_equation(self):
        names = self.predict.settings['x'] + []
        name_Y = self.predict.settings['y']
        b = self.predict.model.get_all_coef()

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()

        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        b = results.params[::-1]
        uravnenie = LinearRegressionModel.uravnenie(
            self.predict.model, b, names, name_Y)
        df_X = self.predict.df_X_test
        b = self.predict.model.get_all_coef()
        b = results.params[::-1]

        def update_output(n_clicks, input1):
            number = len(self.coord_list)
            if n_clicks == 0 or input1 == 'Да':
                self.coord_list = []
                number = len(self.coord_list)
                return u'''Введите значение параметра "{}"'''.format(df_X.columns[0])
            if re.fullmatch(r'^([-+])?\d+([,.]\d+)?$', input1):
                number += 1
                if input1.find(',') > 0:
                    input1 = float(input1[0:input1.find(
                        ',')] + '.' + input1[input1.find(',') + 1:len(input1)])
                self.coord_list.append(float(input1))
                if len(self.coord_list) < len(df_X.columns):
                    return u'''Введите значение параметра  "{}".'''.format(df_X.columns[number])
                    # максимальное значение - len(df_X.columns)-1
                if len(self.coord_list) == len(df_X.columns):
                    number = -1
                    yzn = b[0]
                    for i in range(len(self.coord_list)):
                        yzn += self.coord_list[i] * b[i + 1]
                    return u'''Предсказанное значение равно {} \n Если желаете посчитать ещё для одного набор признаков
                       , напишите "Да".'''.format(round(yzn, 3))
            elif n_clicks > 0:
                return u'''Введено не число, введите значение параметра "{}" повторно.'''.format(df_X.columns[number])
            if number == -1 and input1 != 0 and input1 != 'Да' and input1 != '0':
                return u'''Если желаете посчитать ещё для {} набор признаков, напишите "Да".'''.format('одного')

        self.predict.app.callback(dash.dependencies.Output('output-state', 'children'),
                                  [dash.dependencies.Input(
                                      'submit-button-state', 'n_clicks')],
                                  [dash.dependencies.State('input-1-state', 'value')])(update_output)
        return html.Div([html.Div(html.H2(children='Уравнение полииномиальной регрессии'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dcc.Markdown(id='Markdown', children=uravnenie)),
                                   html.Div(html.H4(children='Предсказание новых значений'),
                                            style={'text-align': 'center'}),
                                   dcc.Markdown(children='Чтобы получить значение зависимой переменной, '
                                                         'введите значение независимых признаков ниже:'),
                                   dcc.Input(id='input-1-state',
                                             type='text', value=''),
                                   html.Button(id='submit-button-state',
                                               n_clicks=0, children='Submit'),
                                   html.Div(id='output-state', children='')],
                                  style={'width': '78%', 'display': 'inline-block',
                                         'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                         'padding': '5px'})
                         ], style={'margin': '50px'})

    # графики
    def _generate_distrib(self):

        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()
        df_Y = y
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        predict_Y = results.predict()
        # График распределения остатков
        fig_rasp_2 = go.Figure()
        df_ost_2 = pd.DataFrame({'Изначальный Y': df_Y, 'Предсказанный Y': predict_Y})
        fig_rasp_2 = px.scatter(df_ost_2, x="Изначальный Y", y="Предсказанный Y",
                                trendline="ols", trendline_color_override='red', labels='Данные')
        fig_rasp_2.update_traces(marker_size=20)

        fig = go.Figure(
            data=go.Histogram(
                x=df_Y - predict_Y,
                name='Остатки'
            )
        )

        fig.add_trace(
            go.Histogram(
                x=np.random.normal(0, 1, len(df_Y)),
                name='Нормальное распределение'
            )
        )
        fig.update_xaxes(title='Остатки')
        fig.update_layout(bargap=0.1)

        # специфичность

        residuals = df_Y - predict_Y
        num_divisions = residuals.shape[0] + 1
        quantiles = np.arange(1, residuals.shape[0]) / num_divisions

        qq_x_data = sps.norm.ppf(quantiles)
        qq_y_data = np.sort(residuals)

        line_x0 = sps.norm.ppf(0.25)
        line_x1 = sps.norm.ppf(0.75)
        line_y0 = np.quantile(residuals, 0.25)
        line_y1 = np.quantile(residuals, 0.75)
        slope = (line_y1 - line_y0) / (line_x1 - line_x0)
        line_intercept = line_y1 - (slope * line_x1)
        x_range_line = np.arange(-3, 3, 0.001)
        y_values_line = (slope * x_range_line) + line_intercept
        fig_qqplot = go.Figure()
        fig_qqplot.add_trace(
            go.Scatter(
                x=qq_x_data,
                y=qq_y_data,
                mode='markers',
                marker={'color': 'blue'},
                name='Остатки')
        )
        fig_qqplot.add_trace(
            go.Scatter(
                x=x_range_line,
                y=y_values_line,
                mode='lines',
                marker={'color': 'red'},
                name='Нормальное распределение'))
        fig_qqplot['layout'].update(
            xaxis={
                'title': 'Теоретические квантили',
                'zeroline': True},
            yaxis={
                'title': 'Экспериментальные квантили'},
            showlegend=True,
        )
        graph_styles = {
            'text-align': 'center',
            'width': '78%',
            'display': 'inline-block',
            'border-color': 'rgb(220, 220, 220)',
            'border-style': 'solid'
        }
        return html.Div([html.Div(html.H2(children='Графики остатков'), style={'text-align': 'center'}),
                         html.Div([
                             html.Div(
                                 html.H4(children='Гистограмма распределения остатков'),
                                 style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_1', figure=fig), style=graph_styles),
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График соответствия предсказанных значений зависимой переменной '
                                                  'и исходных значений'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_2', figure=fig_rasp_2), style=graph_styles),
                             html.Div(dcc.Markdown(markdown_graph))
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График квантиль-квантиль'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='graph_qqplot', figure=fig_qqplot), style=graph_styles),
                         ], style={'margin': '50px'}),



                         ], style={'margin': '50px'})



    # таблица остатков
    def _generate_resid(self):
        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = smapi.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = smapi.add_constant(np.column_stack((ele, X)))
            results = smapi.OLS(y, X).fit()
            return results

        y = self.predict.df_Y.tolist()  # исходные значения
        print(y)
        X = []
        for column in self.predict.df_X:
            li = self.predict.df_X[column].tolist()
            X.append(li)
        results = reg_m(y, X)
        znach = results.predict()  # рассчитанное значение
        ost = []  # остатки
        for i in range(len(znach)):
            ost.append(y[i] - znach[i])
        influence = results.get_influence()
        standardized_residuals = influence.resid_studentized_internal  # стз остатки

        df_result_3 = pd.DataFrame({'Номер наблюдения': 0, 'Исходное значение признака': y,
                                    'Рассчитанное значение признака': znach, 'Остатки': ost,
                                    'Стандартизированные остатки': standardized_residuals})
        df_result_3.iloc[:, 0] = [i + 1 for i in range(df_result_3.shape[0])]
        return html.Div([html.Div(html.H2(children='Таблица остатков'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table3',
                             data=df_result_3.to_dict('records'),
                             columns=[{"name": i, "id": i}
                                      for i in df_result_3.columns],
                             # tooltip_header={i: i for i in df.columns}, #
                             # либо этот, либо тот что ниже
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_3.columns},
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },

                             # asdf
                             page_size=20,
                             fixed_rows={'headers': True},
                             style_table={'height': '330px',
                                          'overflowY': 'auto'},
                             tooltip_delay=0,
                             tooltip_duration=None,
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_3.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

    # таблица критериев значимости переменных
    def _generate_signif(self):
            def reg_m(y, x):
                ones = np.ones(len(x[0]))
                X = smapi.add_constant(np.column_stack((x[0], ones)))
                for ele in x[1:]:
                    X = smapi.add_constant(np.column_stack((ele, X)))
                results = smapi.OLS(y, X).fit()
                return results

            y = self.predict.df_Y.tolist()
            print(y)
            X = []
            for column in self.predict.df_X:
                li = self.predict.df_X[column].tolist()
                X.append(li)
            print(X)
            results = reg_m(y, X)
            print(reg_m(y, X).summary2())
            res_b = results.params
            print(res_b)
            res_errb = results.bse
            print(res_errb)
            res_tst = results.tvalues
            print(res_tst)
            res_pval = results.pvalues
            print(res_pval)
            df_column = list(self.predict.df_X_test.columns)
            df_column.append("const")
            print(df_column)
            df_result_2 = pd.DataFrame({'Название переменной': df_column,
                                        'b': res_b,
                                        'St.Error b': res_errb,
                                        't-критерий': res_tst,
                                        'p-value': res_pval})

            return html.Div(
                [html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
                 html.Div([html.Div(dash_table.DataTable(
                     id='table2',
                     columns=[{"name": i, "id": i}
                              for i in df_result_2.columns],
                     data=df_result_2.to_dict('records'),
                     style_table={'textOverflowX': 'ellipsis', },
                     tooltip={i: {
                         'value': i,
                         'use_with': 'both'
                     } for i in df_result_2.columns},
                     tooltip_data=[
                         {
                             column: {'value': str(value), 'type': 'markdown'}
                             for column, value in row.items()
                         } for row in df_result_2.to_dict('records')
                     ],
                     style_header={
                         'textDecoration': 'underline',
                         'textDecorationStyle': 'dotted',
                     },
                     style_cell={
                         'overflow': 'hidden',
                         'textOverflow': 'ellipsis',
                         'maxWidth': 0,  # len(df_result_3.columns)*5,
                     },
                     export_format='xlsx'

                 ), style={'width': str(len(df_result_2.columns) * 6) + '%', 'display': 'inline-block'}),
                     html.Div(dcc.Markdown(markdown_linear_table2))],  # style={'margin': '50px'},
                     style={'width': '78%', 'display': 'inline-block',
                            'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                 ], style={'margin': '50px'})


class ROC(Dashboard):

    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict

        # тут вложенными списками будут значения для каждой переменной
        self.dx_list = []

        self.tp_list = []
        self.tn_list = []
        self.fp_list = []
        self.fn_list = []

        self.se_list = []  # чувствительность
        self.sp_list = []  # специфичность
        self.inv_sp_list = []  # 1-специфичность

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        if 'classificators_comparison' in self.predict.settings['metrics']:
            metrics_list = [
                self._generate_dashboard(),
                self._generate_comparison()]
        else:
            metrics_list = [self._generate_dashboard()]

        return html.Div([
            html.Div(html.H1(children='ROC-анализ'), style={'text-align': 'center'}),
            html.Div(metrics_list)])

    def _generate_metrics(self, ind):
        def dov_int_clopper(k, n):
            v1_lcl = 2 * (n - k + 1)
            v2_lcl = 2 * k
            v1_ucl = 2 * (k + 1)
            v2_ucl = 2 * (n - k)

            df_lcl = f.isf((1 - 0.95) / 2, v1_lcl, v2_lcl)
            df_ucl = f.isf((1 - 0.95) / 2, v1_ucl, v2_ucl)

            dov_int_l = k / (k + (n - k + 1) * df_lcl)
            dov_int_u = (k + 1) * df_ucl / (n - k + (k + 1) * df_ucl)
            return dov_int_l, dov_int_u

        def dov_int_wilson(k, n):
            p = k / n
            result = binomtest(k=k, n=n, p=p)
            dov_int = result.proportion_ci(confidence_level=0.95, method='wilson')
            return dov_int
        # metrics
        threshold = 1
        t_ind = 0
        for i in range(len(self.se_list[ind])):
            if threshold > abs(self.se_list[ind][i] - self.sp_list[ind][i]):
                threshold = abs(self.se_list[ind][i] - self.sp_list[ind][i])
                t_ind = i
        threshold = round(threshold, 3)
        TPR = round(self.tp_list[ind][
                        t_ind] / (self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind]), 3)
        PPV = round(self.tp_list[ind][
                    t_ind] / (self.tp_list[ind][t_ind] + self.fp_list[ind][t_ind]), 3)
        specificity = round(self.tn_list[ind][t_ind] / (
                self.tn_list[ind][t_ind] + self.fp_list[ind][t_ind]), 3)
        accuracy = round((self.tp_list[ind][t_ind] + self.tn_list[ind][t_ind]) / (
                self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind] + self.tn_list[ind][t_ind] + self.fp_list[ind][
            t_ind]), 3)
        f_measure = round(2 * self.tp_list[ind][t_ind] / (
                2 * self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind] + self.fp_list[ind][t_ind]), 3)
        auc = 0
        for i in range(len(self.sp_list[ind]) - 1):
            auc += (self.se_list[ind][i] + self.se_list[ind][i + 1]) * (
                    self.inv_sp_list[ind][i + 1] - self.inv_sp_list[ind][i]) / 2
        auc = round(abs(auc), 3)
        df_ost_2 = pd.DataFrame(
            columns=['Параметр', 'Threshold', 'Оптимальный порог', 'Чувствительность', 'Специфичность', 'Точность',
                     'Accuracy', 'F-мера', 'AUC'])
        df_ost_2.loc[1] = ['Значение', threshold, round(self.dx_list[ind][t_ind], 3), TPR, specificity, PPV, accuracy,
                           f_measure, auc]

        # dov int
        # Se
        di_se_clopper = dov_int_clopper(self.tp_list[ind][t_ind], self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind])
        di_se_wilson = dov_int_wilson(self.tp_list[ind][t_ind], self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind])
        # Sp
        di_sp_clopper = dov_int_clopper(self.tn_list[ind][t_ind], self.tn_list[ind][t_ind] + self.fp_list[ind][t_ind])
        di_sp_wilson = dov_int_wilson(self.tn_list[ind][t_ind], self.tn_list[ind][t_ind] + self.fp_list[ind][t_ind])
        # Precision
        di_prec_clopper = dov_int_clopper(self.tp_list[ind][t_ind], self.tp_list[ind][t_ind] + self.fp_list[ind][t_ind])
        di_prec_wilson = dov_int_wilson(self.tp_list[ind][t_ind], self.tp_list[ind][t_ind] + self.fp_list[ind][t_ind])
        # Accuracy
        di_accur_clopper = dov_int_clopper(self.tp_list[ind][t_ind] + self.tn_list[ind][t_ind], self.tp_list[ind][t_ind]
                                           + self.fn_list[ind][t_ind] + self.tn_list[ind][t_ind]
                                           + self.fp_list[ind][t_ind])
        di_accur_wilson = dov_int_wilson(self.tp_list[ind][t_ind] + self.tn_list[ind][t_ind], self.tp_list[ind][t_ind]
                                         + self.fn_list[ind][t_ind] + self.tn_list[ind][t_ind]
                                         + self.fp_list[ind][t_ind])
        # AUC
        di_auc = (np.var(self.se_list[ind]) /
                   (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
        di_auc_1 = round((self.se_list[ind][t_ind] - 1.96 * di_auc), 3)
        di_auc_2 = round((self.se_list[ind][t_ind] + 1.96 * di_auc), 3)
        df_dov_int = pd.DataFrame(
            columns=['Метод', 'Чувствительность', 'Специфичность', 'Точность', 'Accuracy', 'AUC'])
        df_dov_int.loc[1] = ['Пирсона-Клоппера',
                             str(round(di_se_clopper[0], 3)) + '; ' + str(round(di_se_clopper[1], 3)),
                             str(round(di_sp_clopper[0], 3)) + '; ' + str(round(di_sp_clopper[1], 3)),
                             str(round(di_prec_clopper[0], 3)) + '; ' + str(round(di_prec_clopper[1], 3)),
                             str(round(di_accur_clopper[0], 3)) + '; ' + str(round(di_accur_clopper[1], 3)),
                             str(di_auc_1) + '; ' + str(di_auc_2)]
        df_dov_int.loc[2] = ['Вилсона', str(round(di_se_wilson[0], 3)) + '; ' + str(round(di_se_wilson[1], 3)),
                             str(round(di_sp_wilson[0], 3)) + '; ' + str(round(di_sp_wilson[1], 3)),
                             str(round(di_prec_wilson[0], 3)) + '; ' + str(round(di_prec_wilson[1], 3)),
                             str(round(di_accur_wilson[0], 3)) + '; ' + str(round(di_accur_wilson[1], 3)),
                             str(di_auc_1) + '; ' + str(di_auc_2)]

        return df_ost_2, df_dov_int

    def _generate_graphs(self):
        # df_ost_2 = pd.DataFrame(
        #    {'dx': self.dx_list, 'tp': self.tp_list, 'fp': self.fp_list, 'fn': self.fn_list, 'tn': self.tn_list,
        #     'sp': self.sp_list, 'se': self.se_list})
        #       fig_rasp_2 = px.scatter(df_ost_2, x="sp", y="se")
        #       fig_rasp_2.update_traces(marker_size=20)
        fig_rasp_2 = go.Figure()
        # px.scatter(df_ost_2, x="dx", y="se")
        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.sp_list,
                y=self.se_list,
                #               xaxis='se',
                #               yaxis='1-sp',
                mode="lines+markers",
                line=go.scatter.Line(color="red"),
                showlegend=True)
        )
        fig_rasp_2.update_traces(marker_size=10)
        return html.Div([html.Div(html.H4(children='ROC'), style={'text-align': 'center'}),
                         html.Div(dcc.Graph(id='graph_ROC', figure=fig_rasp_2)),
                         ], style={'margin': '50px'})

    def _generate_dots(self, ind):
        df_ost_2 = pd.DataFrame(
            {'dx': [round(self.dx_list[ind][i], 3) for i in range(len(self.dx_list[ind]))], 'tp': self.tp_list[ind],
             'fp': self.fp_list[ind], 'fn': self.fn_list[ind], 'tn': self.tn_list[ind], 'sp': self.sp_list[ind],
             'se': self.se_list[ind]})
        return df_ost_2

    def _generate_interception(self, ind):
        #        sp_list = []
        #        for i in range(len(self.sp_list)):
        #            sp_list.append(1-self.sp_list[i])
        #       df_ost_2 = pd.DataFrame(
        #           {'dx': self.dx_list, 'tp': self.tp_list, 'fp': self.fp_list, 'fn': self.fn_list, 'tn': self.tn_list,
        #            'sp': sp_list, 'se': self.se_list})
        fig_rasp_2 = go.Figure()
        # px.scatter(df_ost_2, x="dx", y="se")
        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.dx_list[ind],
                y=self.sp_list[ind],
                mode="lines+markers",
                #             xaxis='dx',
                #             yaxis='sp',
                line=go.scatter.Line(color="red"),
                showlegend=True)
        )

        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.dx_list[ind],
                y=self.se_list[ind],
                #            xaxis='dx',
                #            yaxis='se',
                mode="lines+markers",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig_rasp_2.update_xaxes(
            title_text="Порог отсечения",
            title_font={"size": 20},
            title_standoff=25)
        fig_rasp_2.update_yaxes(
            title_text="Значение",
            title_font={"size": 20},
            title_standoff=25)
        fig_rasp_2.update_traces(marker_size=10)
        # fig_rasp_2.add_trace(px.scatter(df_ost_2, x="dx", y="se"))
        fig_rasp_2.update_traces(marker_size=10)

        return fig_rasp_2

    def _generate_inter_table(self, ind):
        #        sp_list = []
        #        for i in range(len(self.sp_list)):
        #            sp_list.append(1-self.sp_list[i])
        df_ost_2 = pd.DataFrame(
            {'sp': self.sp_list[ind], 'se': self.se_list[ind]})
        return df_ost_2

    def _generate_dashboard(self):
        # точки для разных ROC
        columns_list = self.predict.df_X.columns
        y_true = self.predict.df_Y
        for i in range(len(columns_list)):
            df_X = self.predict.df_X[columns_list[i]]
            dx = (max(df_X) - min(df_X)) / (len(df_X) - 1)
            dx_init = 0  # min(df_X) - 0.05 * dx
            y_pred = y_true.copy(deep=True)

            dx_list = []
            tp_list = []
            tn_list = []
            fp_list = []
            fn_list = []
            se_list = []
            sp_list = []
            inv_sp_list = []

            flag = True
            while True:
                if flag:
                    dx_init = min(df_X) - 0.05 * dx
                    flag = False
                else:
                    dx_init += dx

                for j in range(len(y_true)):
                    if df_X[j] < dx_init:
                        y_pred[j] = 0
                    else:
                        y_pred[j] = 1

                TN, FP, FN, TP = sm.confusion_matrix(y_true, y_pred).ravel()
                se = TP / (TP + FN)
                sp = TN / (TN + FP)

                if (len(tp_list) == 0) or (
                        len(tp_list) > 0 and (TP != tp_list[-1] or TN != tn_list[-1] or FP != fp_list[-1])):
                    dx_list.append(round(dx_init, 3))
                    tp_list.append(TP)
                    tn_list.append(TN)
                    fp_list.append(FP)
                    fn_list.append(FN)
                    se_list.append(round(se, 3))
                    sp_list.append(round(sp, 3))
                    inv_sp_list.append(round((1 - sp), 3))

                if not dx_init < max(df_X):
                    break

            self.dx_list.append(dx_list)
            self.tp_list.append(tp_list)
            self.tn_list.append(tn_list)
            self.fp_list.append(fp_list)
            self.fn_list.append(fn_list)
            self.sp_list.append(sp_list)
            self.se_list.append(se_list)
            self.inv_sp_list.append(inv_sp_list)

        # сама таблица точек
        df_dots = self._generate_dots(0)

        # таблица метрик
        df_metrics = self._generate_metrics(0)[0]
        df_dov_int = self._generate_metrics(0)[1]

        metric_list = self.predict.settings['metrics']
        for item in reversed(df_metrics.columns.tolist()):
            if item == 'Threshold' and 'trashhold' not in metric_list:
                df_metrics.pop(item)
            if item == 'Оптимальный порог' and 'trashhold' not in metric_list:
                df_metrics.pop(item)
            if item == 'Accuracy' and 'accuracy' not in metric_list:
                df_metrics.pop(item)
                df_dov_int.pop(item)
            if item == 'Точность' and 'precision' not in metric_list:
                df_metrics.pop(item)
                df_dov_int.pop(item)
            if item == 'F-мера' and 'F' not in metric_list:
                df_metrics.pop(item)
            if item == 'Чувствительность' and 'sensitivity' not in metric_list:
                df_metrics.pop(item)
                df_dov_int.pop(item)
            if item == 'Специфичность' and 'specificity' not in metric_list:
                df_metrics.pop(item)
                df_dov_int.pop(item)

        # ROC-кривая
        fig_roc = go.Figure()

        # график пересечения
        fig_inter = go.Figure()

        # точки для графика пересечения
        df_inter = self._generate_inter_table(0)

        def update_roc(column_name, self=self):
            fig_roc = go.Figure()
            ind = columns_list.tolist().index(column_name)
            dov_int = (np.var(self.se_list[
                                  ind]) / (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
            dov_list_1 = [self.se_list[ind][i] - 1.96 *
                          dov_int for i in range(len(self.se_list[ind]))]
            dov_list_2 = [self.se_list[ind][i] + 1.96 *
                          dov_int for i in range(len(self.se_list[ind]))]

            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=self.se_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    fill='tozeroy',
                    name='ROC-кривая',
                    showlegend=True)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=self.inv_sp_list[ind],
                    mode="lines",
                    line=go.scatter.Line(color="blue"),
                    # fill='tozeroy',
                    showlegend=False)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=dov_list_1,
                    mode="lines",
                    line=go.scatter.Line(color="gray"),
                    name='Доверительный интервал ROC-кривой',
                    showlegend=True)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=dov_list_2,
                    mode="lines",
                    line=go.scatter.Line(color="gray"),
                    showlegend=False)
            )

            fig_roc.update_xaxes(
                title_text="1-Специфичность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc.update_yaxes(
                title_text="Чувствительность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc.update_traces(marker_size=10)

            return fig_roc

        self.predict.app.callback(dash.dependencies.Output('graph_roc', 'figure'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_roc)

        def update_inter(column_name, self=self):
            fig_inter = go.Figure()
            ind = columns_list.tolist().index(column_name)
            # df = pd.DataFrame({'dx': self.dx_list[ind], 'sp': self.sp_list[ind], 'se': self.se_list[ind]})
            fig_inter.add_trace(
                go.Scatter(
                    x=self.dx_list[ind],
                    y=self.sp_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    name='Специфичность',
                    # fill='tozeroy',
                    showlegend=True)
            )
            fig_inter.add_trace(
                go.Scatter(
                    x=self.dx_list[ind],
                    y=self.se_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="blue"),
                    name='Чувствительность',
                    # fill='tozeroy',
                    showlegend=True)
            )
            fig_inter.update_xaxes(
                title_text="Порог отсечения",
                title_font={"size": 20},
                title_standoff=25)
            fig_inter.update_yaxes(
                title_text="Значения",
                title_font={"size": 20},
                title_standoff=25)
            fig_inter.update_traces(marker_size=10)
            return fig_inter

        self.predict.app.callback(dash.dependencies.Output('graph_inter', 'figure'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_inter)

        def update_table_dot(column_name, self=self):
            ind = columns_list.tolist().index(column_name)
            df = pd.DataFrame(
                {'dx': self.dx_list[ind], 'tp': self.tp_list[ind], 'fp': self.fp_list[ind],
                 'fn': self.fn_list[ind], 'tn': self.tn_list[ind],
                 'sp': self.sp_list[ind], 'se': self.se_list[ind]})

            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_dot', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_table_dot)

        def update_table_inter(column_name, self=self):
            ind = columns_list.tolist().index(column_name)
            df = self._generate_inter_table(ind)

            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_inter', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_table_inter)

        def update_metrics(column_name):
            ind = columns_list.tolist().index(column_name)
            df = self._generate_metrics(ind)[0]
            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_metrics', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_metrics)

        def update_dov_int(column_name):
            ind = columns_list.tolist().index(column_name)
            df = self._generate_metrics(ind)[1]
            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_dov_int_1', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_dov_int)

        div_markdown = html.Div([
            dcc.Markdown(children="Выберите группирующую переменную:"),
            dcc.Dropdown(
                id='metric_name',
                options=[{'label': i, 'value': i} for i in columns_list],
                value=columns_list[0]
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '5px'})

        div_roc = html.Div([
            html.Div(html.H4(children='ROC'), style={
                'text-align': 'center'}),
            html.Div([
                html.Div(dcc.Graph(id='graph_roc', figure=fig_roc),
                         style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                html.Div(dcc.Markdown(roc_roc))])
        ], style={'margin': '50px'})

        div_metrics = html.Div([
            html.Div(html.H4(children='Таблица метрик'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_metrics',
                    columns=[{"name": i, "id": i}
                             for i in df_metrics.columns],
                    data=df_metrics.to_dict('records'),
                    export_format='csv'
                ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                          'width': str(len(df_metrics.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(roc_table_metrics))])
        ], style={'margin': '50px'})

        div_dov_int = html.Div([
            html.Div(html.H4(children='Таблица доверительных интервалов'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_dov_int_1',
                    columns=[{"name": i, "id": i}
                             for i in df_dov_int.columns],
                    data=df_dov_int.to_dict('records'),
                    export_format='csv'
                ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                          'width': str(len(df_dov_int.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(" "))])
        ], style={'margin': '50px'})

        div_dot = html.Div([
            html.Div(html.H4(children='Таблица точек ROC'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_dot',
                    columns=[{"name": i, "id": i}
                             for i in df_dots.columns],
                    data=df_dots.to_dict('records'),
                    export_format='csv'
                ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                          'width': str(len(df_dots.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(roc_table))])
        ], style={'margin': '50px'})

        div_inter = html.Div([
            html.Div(html.H4(children='График пересечения'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dcc.Graph(id='graph_inter', figure=fig_inter),
                         style={'text-align': 'center', 'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                html.Div(dcc.Markdown(roc_inter_graph))])
        ], style={'margin': '50px'})

        div_list = [div_markdown, div_roc, div_metrics, div_dov_int, div_dot, div_inter]

        if len(df_metrics.columns.tolist()) == 2 or 'metrics_table' not in metric_list:
            div_list.remove(div_metrics)
        if 'spec_and_sens_table' not in metric_list:
            div_list.remove(div_inter)
        if 'points_table' not in metric_list:
            div_list.remove(div_dot)

            # html.Div([
            #    html.Div(html.H4(children='Точки для графика'), style={'text-align': 'center'}),
            #    html.Div(dash_table.DataTable(
            #        id='table_inter',
            #        columns=[{"name": i, "id": i} for i in df_inter.columns],
            #        data=df_inter.to_dict('records')
            #    ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
            #              'width': str(len(df_inter.columns) * 10 - 10) + '%', 'display': 'inline-block'})
            # ], style={'margin': '50px'}),
        # , style={'margin': '50px'})

        return html.Div(div_list, style={'margin': '50px'})

    def _generate_comparison(self):
        columns_list = self.predict.df_X.columns
        fig_roc_2 = go.Figure()

        sum_table = pd.DataFrame(
            columns=['Параметр', 'Threshold', 'Оптимальный порог', 'Чувствительность', 'Специфичность', 'Точность',
                     'Accuracy', 'F-мера', 'AUC'])

        sum_table_di = pd.DataFrame(
            columns=['Группирующая переменная', 'Метод', 'Чувствительность', 'Специфичность', 'Точность', 'Accuracy',
                     'AUC'])

        for i in range(len(columns_list)):
            temp_df = self._generate_metrics(i)[0]
            sum_table = pd.concat([sum_table, temp_df], ignore_index=True)

            temp_df_di = self._generate_metrics(i)[1]
            temp_df_di['Группирующая переменная'] = [columns_list[i], columns_list[i]]
            sum_table_di = pd.concat([sum_table_di, temp_df_di], ignore_index=True)
        sum_table.rename(
            columns={'Параметр': 'Группирующая переменная'}, inplace=True)
        sum_table['Группирующая переменная'] = [item for item in columns_list]

        def update_roc_2(param_1, param_2, self=self):
            fig_roc_2 = go.Figure()
            ind_1 = columns_list.tolist().index(param_1)
            ind_2 = columns_list.tolist().index(param_2)
            #            dov_int = (np.var(self.se_list[ind]) / (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
            #            dov_list_1 = [self.se_list[ind][i] - 1.96 * dov_int for i in range(len(self.se_list[ind]))]
            #            dov_list_2 = [self.se_list[ind][i] + 1.96 * dov_int for i in range(len(self.se_list[ind]))]
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_1],
                    y=self.se_list[ind_1],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    fill='tozeroy',
                    name=param_1,
                    showlegend=True)
            )
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_1],
                    y=self.inv_sp_list[ind_1],
                    mode="lines",
                    line=go.scatter.Line(color="green"),
                    # fill='tozeroy',
                    showlegend=False)
            )
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_2],
                    y=self.se_list[ind_2],
                    mode="lines+markers",
                    line=go.scatter.Line(color="blue"),
                    fill='tozeroy',
                    name=param_2,
                    showlegend=True)
            )

            fig_roc_2.update_xaxes(
                title_text="1-Специфичность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc_2.update_yaxes(
                title_text="Чувствительность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc_2.update_traces(marker_size=10)

            return fig_roc_2

        self.predict.app.callback(dash.dependencies.Output('graph_roc_2', 'figure'),
                                  [dash.dependencies.Input('group_param_1', 'value'),
                                   dash.dependencies.Input('group_param_2', 'value')])(update_roc_2)

        div_2_title = html.Div(html.H2(children='Блок сравнения'), style={'text-align': 'center'})

        div_2_markdown = html.Div([
            html.Div([
                dcc.Markdown(
                    children="Выберите первую группирующую переменную:"),
                dcc.Dropdown(
                    id='group_param_1',
                    options=[{'label': i, 'value': i}
                             for i in columns_list],
                    value=columns_list[0]
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Markdown(
                    children="Выберите вторую группирующую переменную:"),
                dcc.Dropdown(
                    id='group_param_2',
                    options=[{'label': i, 'value': i}
                             for i in columns_list],
                    value=columns_list[0]
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ], style={'padding': '5px'})

        div_2_roc = html.Div([
            html.Div(html.H4(children='ROC'), style={
                'text-align': 'center'}),
            html.Div([
                html.Div(dcc.Graph(id='graph_roc_2', figure=fig_roc_2),
                         style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                html.Div(dcc.Markdown(roc_comp_roc))])
        ], style={'margin': '50px'})

        div_2_metrics = html.Div([
            html.Div(html.H4(children='Таблица метрик'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_metrics_2',
                    columns=[{"name": i, "id": i}
                             for i in sum_table.columns],
                    data=sum_table.to_dict('records'),
                    export_format='csv'
                ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                          'width': str(len(sum_table.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(roc_comp_metrics))])
        ], style={'margin': '50px'})

        div_2_dov_int = html.Div([
            html.Div(html.H4(children='Таблица доверительных интервалов'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_dov_int_2',
                    columns=[{"name": i, "id": i}
                             for i in sum_table_di.columns],
                    data=sum_table_di.to_dict('records'),
                    export_format='csv'
                ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                          'width': str(len(sum_table_di.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(" "))])
        ], style={'margin': '50px'})

        div_2_list = [div_2_title, div_2_markdown, div_2_roc, div_2_metrics, div_2_dov_int]

        return html.Div(div_2_list, style={'margin': '50px'})


class TreeDashboard(Dashboard):
    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        metrics_list = []
        metrics_method = {
            'tree': self._generate_tree_graph(),
            'table': self._generate_table(),
            'indicators': self._generate_indicators(),
            'distributions': self._generate_class_distributions(),
            'prediction': self._generate_prediction_block()
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])

        return html.Div([
            html.Div(html.H1(children='Дерево классификации'),
                     style={'text-align': 'center'}),

            html.Div(html.H5(children=markdown_introduction),
                     style={'text-align': 'center'}),
            html.Div(html.H3(children='Выбранная переменная - "{}"'.format(self.predict.settings['y']),
                             style={'text-align': 'center'})),
            html.Div(metrics_list)], style={'margin': '50px'})

    def _generate_tree_graph(self):
        fig = plt.figure(figsize=(11, 11), dpi=800)
        columns = self.predict.df_X_test.columns
        # Classes
        dict_classes = self.predict.settings['classes']
        classes = list(dict_classes.values())

        tree.plot_tree(self.predict.model.model, fontsize=6, filled=True, feature_names=columns, class_names=classes)
        fig.savefig('tree.png')
        img = Image.open('tree.png')
        image = img.copy()
        os.remove('tree.png')
        return html.Div([html.Div(html.H3(children='Графическое представление дерева'), style={'text-align': 'center'}),
                         html.Div([html.Div(html.Img(src=image,
                                                     style={'width': '100%', 'display': 'inline-block'})),
                                   html.Div(dcc.Markdown(markdown_tree_graph))])],
                        style={'border-color': 'rgb(192, 192, 192)', 'border-style': 'solid',
                               'padding': '5px', 'margin': '50px'})

    def _generate_table(self):
        df_Y = self.predict.df_Y_test
        predict_Y = TreeModel.predict(self.predict.model, self.predict.df_X_test)
        df_Y_new = []
        predict_Y_new = []
        classes = self.predict.settings['classes']
        for i in range(len(df_Y)):
            df_Y_new.append(classes[df_Y[i]])
            predict_Y_new.append(classes[predict_Y[i]])

        df = pd.DataFrame(
            {'Наблюдаемые показатели': df_Y_new,
             'Предсказание': predict_Y_new})

        return html.Div([html.Div([
            html.Div(html.H3(children='Классификационная таблица'), style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_results_1',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_cell={'textAlign': 'center'},
                    data=df.to_dict('records'),
                    fixed_rows={'headers': True},
                    style_table={'overflowX': 'scroll', 'height': 450},
                    export_format='xlsx'),
                    style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                           'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                html.Div(dcc.Markdown(markdown_results_table))])
        ], style={'border-color': 'rgb(192, 192, 192)', 'border-style': 'solid', 'padding': '5px', 'margin': '50px'})
        ], style={'text-align': 'center'})

    def _generate_indicators(self):
        predict_Y = TreeModel.predict(self.predict.model, self.predict.df_X_test)
        classes = TreeModel.get_classes(self.predict.model)
        df_Y = self.predict.df_Y_test
        # Score
        accuracy = round(accuracy_score(predict_Y, df_Y), 3)
        # Энтропийный индекс неоднородности
        dict_frequency = {i: 0 for i in classes}
        for el in predict_Y:
            dict_frequency[el] += 1
        frequency = np.array(list(dict_frequency.values()), dtype=np.float64)
        frequency /= frequency.sum()
        entropy_heterogeneity = 0
        for el in frequency:
            if el != 0:
                entropy_heterogeneity += -1 * el * np.log(el)
        entropy_heterogeneity = np.round(entropy_heterogeneity, 3)
        # Gini
        gini = np.round(1 - (frequency ** 2).sum(), 3)
        # Индекс ошибочной классификации
        index_wrong_classification = np.round(1 - max(frequency), 3)
        # Table 1
        df = pd.DataFrame({'Score': accuracy, 'Энтропийный индекс неоднородности': entropy_heterogeneity,
                           'Индекс Джини': gini, 'Индекс ошибочной классификации': index_wrong_classification},
                          index=[0])
        # Table if 2 classes
        if len(classes) == 2:
            tp = fn = fp = tn = 0
            for i in range(len(df_Y)):
                if df_Y[i] == 1:
                    if predict_Y[i] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predict_Y[i] == 1:
                        fp += 1
                    else:
                        tn += 1
            recall = round(tp / (tp + fn), 3)
            precision = round(tp / (tp + fp), 3)
            f1 = round(2 * (recall * precision) / (recall + precision), 3)
            df = pd.DataFrame({'Accuracy': accuracy, 'Энтропийный индекс неоднородности': entropy_heterogeneity,
                               'Индекс Джини': gini, 'Индекс ошибочной классификации': index_wrong_classification,
                               'Полнота': recall, 'Тончость': precision, 'F1-мера': f1},
                              index=[0])
        return html.Div([html.Div([
            html.Div(html.H3(children='Показатели качества построенного дерева решений'),
                     style={'text-align': 'center'}),
            html.Div([
                html.Div(dash_table.DataTable(
                    id='table_quality',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    export_format='csv'
                )),
                html.Div(dcc.Markdown(markdown_quality))])
        ], style={'border-color': 'rgb(192, 192, 192)',
                  'border-style': 'solid', 'padding': '5px', 'margin': '50px'})
        ])

    def _generate_class_distributions(self):
        predict_Y = TreeModel.predict(self.predict.model, self.predict.df_X_test)
        df = pd.DataFrame.copy(self.predict.df_X_test)
        columns = df.columns.to_numpy()
        df['predict'] = predict_Y
        option_list = [{'label': str(i), 'value': str(i)} for i in columns]
        # Соответсиве номера класс и названия
        init_df = read_file(self.predict.settings['path'])
        init_y_values = init_df[self.predict.settings['y']].to_list()
        init_unique_y_values = np.unique(init_y_values)
        number_class = []
        for name in init_unique_y_values:
            number_class.append(self.predict.df_Y[init_y_values.index(name)])
        dict_classes = dict(zip(number_class, init_unique_y_values))
        class_names = []
        for num in predict_Y:
            class_names.append(dict_classes[num])
        df['class_names'] = class_names

        def update_graph(x_name, y_name):
            fig_graph = px.scatter(df, x=x_name, y=y_name, color="class_names")
            return fig_graph

        self.predict.app.callback(dash.dependencies.Output('graph_distributions', 'figure'),
                                  dash.dependencies.Input('x_name', 'value'),
                                  dash.dependencies.Input('y_name', 'value'))(update_graph)

        return html.Div([html.H3(children='График распределения классов', style={'text-align': 'center'}),
                         html.Div([
                            dcc.Markdown(
                                children="Выберите первую группирующую переменную:"),
                            dcc.Dropdown(
                                 id='x_name',
                                 options=option_list,
                                 value=option_list[0]['value'],
                                 clearable=False)
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Markdown(
                                children="Выберите вторую группирующую переменную:"),
                            dcc.Dropdown(
                                 id='y_name',
                                 options=option_list,
                                 value=option_list[0]['value'],
                                 clearable=False)
                        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                        html.Div([dcc.Graph(id='graph_distributions')],
                                 style={'width': '100%', 'display': 'inline-block'})
                    ], style={'border-color': 'rgb(192, 192, 192)',
                              'border-style': 'solid', 'padding': '5px', 'margin': '50px'})

    def _generate_prediction_block(self):
        df = pd.DataFrame.copy(self.predict.df_X_test)
        results_columns = ['Наблюдаемые значения', 'Предсказанные значения']

        def get_data(data, n_clicks):
            data = pd.DataFrame.from_records(data)
            columns_X = self.predict.df_X_test.columns
            data = data[columns_X]
            changed_id = [p['prop_id'] for p in callback_context.triggered][0]
            if 'btn_ok' in changed_id:
                predict_Y = TreeModel.predict(self.predict.model, data)
                df_Y = self.predict.df_Y_test
                df_Y_new = []
                predict_Y_new = []
                classes = self.predict.settings['classes']
                for i in range(len(df_Y)):
                    df_Y_new.append(classes[df_Y[i]])
                    predict_Y_new.append(classes[predict_Y[i]])
                df_res = pd.DataFrame(
                    {'Наблюдаемые значения': df_Y_new,
                     'Предсказанные значения': predict_Y_new
                     })
                return df_res.to_dict('records')
            else:
                raise PreventUpdate

        self.predict.app.callback(dash.dependencies.Output('predict_results', 'data'),
                                  dash.dependencies.Input('predict_table', 'data'),
                                  dash.dependencies.Input('btn_ok', 'n_clicks'))(get_data)

        return html.Div([html.Div(html.H3(children='Работа с исходной таблицей для получения предсказательных значений'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children='Вы можете изменить исходные данные и оценить предсказанное значение'),
                         html.Div([dash_table.DataTable(
                             id='predict_table',
                             columns=[{"name": i, "id": i} for i in df.columns],
                             data=df.to_dict('records'),
                             fixed_rows={'headers': True},
                             style_table={'overflowX': 'scroll', 'height': 450},
                             export_format='xlsx',
                             editable=True),
                         ]),
                         html.Div(html.Button('Предсказать', id='btn_ok', n_clicks=0), style={'padding': '20px'}),
                         dcc.Markdown(children='Полученное предсказание'),
                         html.Div([dash_table.DataTable(
                             id='predict_results',
                             columns=[{"name": i, "id": i} for i in results_columns],
                             style_cell={'textAlign': 'center'},
                             fixed_rows={'headers': True},
                             style_table={'overflowX': 'scroll', 'height': 450},
                             data=df.to_dict('records'),
                             export_format='xlsx')],
                             style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                    'text-align': 'center', 'display': 'inline-block', 'width': '50%'})
                         ], style={'border-color': 'rgb(192, 192, 192)', 'text-align': 'center',
                                   'border-style': 'solid', 'padding': '5px', 'margin': '50px'})

    # return html.Div([html.Div([
    #     html.Div(html.H3(children='Классификационная таблица'), style={'text-align': 'center'}),
    #     html.Div([
    #         html.Div(dash_table.DataTable(
    #             id='table_results_1',
    #             columns=[{"name": i, "id": i} for i in df.columns],
    #             style_cell={'textAlign': 'center'},
    #             data=df.to_dict('records'),
    #             export_format='xlsx'),
    #             style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
    #                    'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
    #         html.Div(dcc.Markdown(markdown_results_table))])
    # ], style={'border-color': 'rgb(192, 192, 192)', 'border-style': 'solid', 'padding': '5px', 'margin': '50px'})
    # ], style={'text-align': 'center'})


