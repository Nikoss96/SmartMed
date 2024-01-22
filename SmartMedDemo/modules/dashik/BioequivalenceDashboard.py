from math import e

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Dashboard import Dashboard
from text.markdown_bio import *


def round_df(df):
    cols = df.columns
    for j in range(0, len(cols)):
        for i in range(len(df)):
            if type(df.iloc[i, j]) != str and type(df.iloc[i, j]) != int:
                num = str(df.iloc[i, j])
                num = num.replace(']', '')
                while not num[0].isdigit():
                    num = num[1:]
                if df.iloc[i, j] < 1000 and df.iloc[i, j] >= 0.01:
                    point = num.find('.')
                    df.iloc[i, j] = num[:point + 3]
                elif df.iloc[i, j] > 1000:
                    point = num.find('.')
                    num = num[:point]
                    df.iloc[i, j] = num[0] + '.' + \
                                    num[1:3] + 'e' + str(len(num) - 1)
                elif 'e' in num:
                    epos = num.find('e')
                    df.iloc[i, j] = num[0:4] + num[epos:]
                elif df.iloc[i, j] < 0.01:
                    notnul = 2
                    while len(num) < notnul and num[notnul] == '0':
                        notnul += 1
                    if notnul == len(num):
                        df.iloc[i, j] = '0'
                    else:
                        df.iloc[i, j] = num[notnul] + '.' + \
                                        num[notnul + 1:notnul + 3] + 'e-' + str(notnul - 1)
    return df


class BioequivalenceDashboard(Dashboard):

    def _generate_layout(self):
        # metrics inludings is checked inside method
        return html.Div(self.graphs_and_lists)

    def _generate_criteria(self):
        if self.settings[0].plan == 'parallel':
            if self.settings[0].check_normal == 'Kolmogorov' and self.settings[0].check_uniformity == 'F':
                data = {'Критерий': ['Колмогорова-Смирнова',
                                     'Колмогорова-Смирнова', 'F-критерий'],
                        'Группа': ['R', 'T', 'RT'],
                        'Значение критерия': [self.settings[0].kstest_r[0],
                                              self.settings[0].kstest_t[0], self.settings[0].f[0]],
                        'p-уровень': [self.settings[0].kstest_r[1],
                                      self.settings[0].kstest_t[1], self.settings[0].f[1]]}
            elif self.settings[0].check_normal == 'Kolmogorov' and self.settings[0].check_uniformity == 'Leven':
                data = {'Критерий': ['Колмогорова-Смирнова',
                                     'Колмогорова-Смирнова', 'Левена'],
                        'Группа': ['R', 'T', 'RT'],
                        'Значение критерия': [self.settings[0].kstest_r[0],
                                              self.settings[0].kstest_t[0], self.settings[0].levene[0]],
                        'p-уровень': [self.settings[0].kstest_r[1],
                                      self.settings[0].kstest_t[1], self.settings[0].levene[1]]}
            elif self.settings[0].check_normal == 'Shapiro' and self.settings[0].check_uniformity == 'Leven':
                data = {'Критерий': ['Шапиро-Уилка', 'Шапиро-Уилка', 'Левена'],
                        'Группа': ['R', 'T', 'RT'],
                        'Значение критерия': [self.settings[0].shapiro_r[0],
                                              self.settings[0].shapiro_t[0], self.settings[0].levene[0]],
                        'p-уровень': [self.settings[0].shapiro_r[1],
                                      self.settings[0].shapiro_t[1], self.settings[0].levene[1]]}
            else:
                data = {'Критерий': ['Шапиро-Уилка', 'Шапиро-Уилка', 'F-критерий'],
                        'Группа': ['R', 'T', 'RT'],
                        'Значение критерия': [self.settings[0].shapiro_r[0],
                                              self.settings[0].shapiro_t[0], self.settings[0].f[0]],
                        'p-уровень': [self.settings[0].shapiro_r[1],
                                      self.settings[0].shapiro_t[1], self.settings[0].f[1]]}

            df = pd.DataFrame(data)
            df = round_df(df)
            return html.Div([html.Div(html.H1(children='Выполнение критериев'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([dash_table.DataTable(
                                         id='criteria',
                                         columns=[{"name": i, "id": i, "deletable": True}
                                                  for i in df.columns],
                                         data=df.to_dict('records'),
                                         style_cell_conditional=[
                                             {'if': {'column_id': 'Критерий'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'Группа'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'Значение критерия'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'p-уровень'},
                                              'width': '25%'},
                                         ],
                                         style_table={'overflowX': 'auto'},
                                         export_format='xlsx'
                                     )], style={'border-color': 'rgb(220, 220, 220)',
                                                'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                     style={'width': '78%', 'display': 'inline-block'}),
                                 html.Div(dcc.Markdown(children=markdown_text_criteria), style={
                                     'width': '18%', 'float': 'right', 'display': 'inline-block'})
                             ])
                             ], style={'margin': '50px'}
                            )
        else:
            if self.settings[0].check_normal == 'Kolmogorov':
                data = {'Критерий': ['Бартлетта', 'Бартлетта',
                                     'Колмогорова-Смирнова', 'Колмогорова-Смирнова', 'Колмогорова-Смирнова',
                                     'Колмогорова-Смирнова'],
                        'Выборки': ['Первая и вторая группы',
                                    'Период 1 и период 2', 'Первая группа тестовый препарат',
                                    'Первая группа референсный препарат',
                                    'Вторая группа тестовый препарат', 'Вторя группа референсный препарат'],
                        'Значение критерия': [self.settings[0].bartlett_groups[0],
                                              self.settings[0].bartlett_period[0], self.settings[0].kstest_t_1[0],
                                              self.settings[0].kstest_r_1[0], self.settings[0].kstest_t_2[0],
                                              self.settings[0].kstest_r_2[0]],
                        'p-уровень': [self.settings[0].bartlett_groups[1],
                                      self.settings[0].bartlett_period[1], self.settings[0].kstest_t_1[1],
                                      self.settings[0].kstest_r_1[1], self.settings[0].kstest_t_2[1],
                                      self.settings[0].kstest_r_2[1]]}
            else:
                data = {'Критерий': ['Бартлетта', 'Бартлетта', 'Шапиро-Уилка', 'Шапиро-Уилка', 'Шапиро-Уилка',
                                     'Шапиро-Уилка'],
                        'Выборки': ['Первая и вторая группы',
                                    'Период 1 и период 2', 'Первая группа тестовый препарат',
                                    'Первая группа референсный препарат',
                                    'Вторая группа тестовый препарат', 'Вторя группа референсный препарат'],
                        'Значение критерия': [self.settings[0].bartlett_groups[0],
                                              self.settings[0].bartlett_period[0], self.settings[0].shapiro_t_1[0],
                                              self.settings[0].shapiro_r_1[0], self.settings[0].shapiro_t_2[0],
                                              self.settings[0].shapiro_r_2[0]],
                        'p-уровень': [self.settings[0].bartlett_groups[1],
                                      self.settings[0].bartlett_period[1], self.settings[0].shapiro_t_1[1],
                                      self.settings[0].shapiro_r_1[1], self.settings[0].shapiro_t_2[1],
                                      self.settings[0].shapiro_r_2[1]]}

            df = pd.DataFrame(data)
            df = round_df(df)
            return html.Div([html.Div(html.H1(children='Выполнение критериев'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([dash_table.DataTable(
                                         id='criteria',
                                         columns=[{"name": i, "id": i, "deletable": True}
                                                  for i in df.columns],
                                         data=df.to_dict('records'),
                                         style_cell_conditional=[
                                             {'if': {'column_id': 'Критерий'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'Выборки'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'Значение критерия'},
                                              'width': '25%'},
                                             {'if': {'column_id': 'p-уровень'},
                                              'width': '25%'},
                                         ],
                                         style_table={'overflowX': 'auto'},
                                         export_format='xlsx'
                                     )], style={'border-color': 'rgb(220, 220, 220)',
                                                'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                     style={'width': '78%', 'display': 'inline-block'}),
                                 html.Div(dcc.Markdown(children=markdown_text_criteria), style={
                                     'width': '18%', 'float': 'right', 'display': 'inline-block'})
                             ])
                             ], style={'margin': '50px'}
                            )

    def _generate_param(self):
        data = {'Группа': ['R', 'T'],
                'AUC': [float(np.mean(self.settings[0].auc_r_notlog)),
                        float(np.mean(self.settings[0].auc_t_notlog))],
                'AUC_inf': [float(np.mean(self.settings[0].auc_r_infty)),
                            float(np.mean(self.settings[0].auc_t_infty))],
                'ln AUC': [float(np.mean(self.settings[0].auc_r)), float(np.mean(self.settings[0].auc_t))],
                'ln AUC_inf': [float(np.mean(self.settings[0].auc_r_infty_log)),
                               float(np.mean(self.settings[0].auc_t_infty_log))],
                'ln Tmax': [float(np.log(self.settings[0].concentration_r.columns.max())),
                            float(np.log(self.settings[0].concentration_t.columns.max()))],
                'ln Cmax': [float(np.log(self.settings[0].concentration_r.max().max())),
                            float(np.log(self.settings[0].concentration_t.max().max()))]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Таблица с распределением ключевых параметров по группам'),
                                  style={'text-align': 'center'}),
                         html.Div([
                             html.Div([
                                 html.Div([dash_table.DataTable(
                                     id='param',
                                     columns=[{"name": i, "id": i, "deletable": True}
                                              for i in df.columns],
                                     data=df.to_dict('records'),
                                     style_table={'overflowX': 'auto'},
                                     export_format='xlsx'
                                 )], style={'border-color': 'rgb(220, 220, 220)',
                                            'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                 style={'width': '68%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(children=markdown_text_param),
                                      style={'width': '28%', 'float': 'right', 'display': 'inline-block',
                                             'padding': '5px', 'margin': '5px'})
                         ])
                         ], style={'margin': '50px'}
                        )

    def _generate_log_auc(self):
        data = {'Группа': ['TR', 'RT'],
                'ln AUC T': [float(np.mean(self.settings[0].auc_t_1)), float(np.mean(self.settings[0].auc_t_2))],
                'ln AUC R': [float(np.mean(self.settings[0].auc_r_1)), float(np.mean(self.settings[0].auc_r_2))]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Средние площади под графиком по каждому препарату'),
                                  style={'text-align': 'center'}),
                         html.Div([
                             html.Div([
                                 html.Div([dash_table.DataTable(
                                     id='param',
                                     columns=[{"name": i, "id": i, "deletable": True}
                                              for i in df.columns],
                                     data=df.to_dict('records'),
                                     style_table={'overflowX': 'auto'},
                                     export_format='xlsx'
                                 )], style={'border-color': 'rgb(220, 220, 220)',
                                            'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                 style={'width': '78%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(children=markdown_text_log_auc), style={
                                 'width': '18%', 'float': 'right', 'display': 'inline-block'})
                         ])
                         ], style={'margin': '50px'}
                        )

    def _generate_anova(self):
        if self.settings[0].plan == 'parallel':
            df = self.settings[0].anova[0]
            mark = markdown_text_anova
            heading = 'Результаты классического дисперсионного анализа'
            marg = '250px'
        else:
            df = self.settings[0].anova
            mark = markdown_text_anova_cross
            heading = 'Результаты двухфакторного дисперсионного анализа'
            marg = '50px'
        df = round_df(df)
        return html.Div([html.Div(html.H1(children=heading), style={'text-align': 'center'}),
                         html.Div([
                             html.Div([
                                 html.Div([dash_table.DataTable(
                                     id='anova',
                                     columns=[{"name": i, "id": i, "deletable": True}
                                              for i in df.columns],
                                     data=df.to_dict('records'),
                                     style_cell_conditional=[
                                         {'if': {'column_id': 'SS'},
                                          'width': '20%'},
                                         {'if': {'column_id': 'df'},
                                          'width': '20%'},
                                         {'if': {'column_id': 'MS'},
                                          'width': '20%'},
                                         {'if': {'column_id': 'F'},
                                          'width': '20%'},
                                         {'if': {'column_id': 'F крит.'},
                                          'width': '20%'}
                                     ],
                                     style_table={'overflowX': 'auto'},
                                     export_format='xlsx'
                                 )], style={'border-color': 'rgb(220, 220, 220)',
                                            'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                 style={'width': '78%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(children=mark), style={
                                 'width': '18%', 'float': 'right', 'display': 'inline-block'})
                         ])
                         ], style={'margin': '50px', 'margin-top': marg}
                        )

    def _generate_interval(self):
        data = {'Критерий': ['Биоэквивалентности', 'Бионеэквивалентности'],
                'Нижняя граница': [100 * (e ** self.settings[0].oneside_eq[0]),
                                   100 * (e ** self.settings[0].oneside_noteq[0])],
                'Верхняя граница': [100 * (e ** self.settings[0].oneside_eq[1]),
                                    100 * (e ** self.settings[0].oneside_noteq[1])],
                'Доверительный интервал критерия': ['80.00-125.00%', '80.00-125.00%'],
                'Выполнение критерия': ['Выполнен' if (self.settings[0].oneside_eq[0] > -0.223 and
                                                       self.settings[0].oneside_eq[1] < 0.223) else 'Не выполнен',
                                        'Выполнен' if (self.settings[0].oneside_noteq[0] > 0.223 or
                                                       self.settings[0].oneside_noteq[1] < -0.223) else 'Не выполнен']}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Результаты оценки биоэквивалентности'),
                                  style={'text-align': 'center'}),
                         html.Div([
                             html.Div([
                                 html.Div([dash_table.DataTable(
                                     id='interval',
                                     columns=[{"name": i, "id": i, "deletable": True}
                                              for i in df.columns],
                                     data=df.to_dict('records'),
                                     style_table={'overflowX': 'auto'},
                                     export_format='xlsx'
                                 )], style={'border-color': 'rgb(220, 220, 220)',
                                            'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                 style={'width': '78%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(children=markdown_text_interval), style={
                                 'width': '18%', 'float': 'right', 'display': 'inline-block'})
                         ])
                         ], style={'margin': '50px', 'margin-top': '150px'}
                        )

    def _generate_concentration_time(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_r], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_r])]], y=[max(df.loc[
                                                                                                        yaxis_column_name_conc_r])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_r])]], y=[min(df.loc[
                                                                                                        yaxis_column_name_conc_r])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div(
                [html.Div(html.H1(children='График зависимости концентрации от времени группа референсного препарата'),
                          style={'text-align': 'center'}),
                 html.Div([
                     html.Div([
                         html.Div([
                             dcc.Markdown(
                                 children="Выберите показатель для оси ОY:"),
                             dcc.Dropdown(
                                 id='yaxis_column_name_conc_r',
                                 options=[{'label': i, 'value': i}
                                          for i in available_indicators],
                                 value=available_indicators[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                     ], style={'padding': '5px'}),
                     dcc.Graph(id='concentration_time_r')], style={'width': '78%',
                                                                   'display': 'inline-block',
                                                                   'border-color': 'rgb(220, 220, 220)',
                                                                   'border-style': 'solid', 'padding': '5px'}),
                 html.Div(dcc.Markdown(children=markdown_text_conc_time_r),
                          style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                style={'margin': '100px'}
                )
        else:
            df = self.settings[0].concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_t], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_t])]], y=[max(df.loc[
                                                                                                        yaxis_column_name_conc_t])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_t])]], y=[min(df.loc[
                                                                                                        yaxis_column_name_conc_t])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div(
                [html.Div(html.H1(children='График зависимости концентрации от времени группа тестового препарата'),
                          style={'text-align': 'center'}),
                 html.Div([
                     html.Div([
                         html.Div([
                             dcc.Markdown(
                                 children="Выберите показатель для оси ОY:"),
                             dcc.Dropdown(
                                 id='yaxis_column_name_conc_t',
                                 options=[{'label': i, 'value': i}
                                          for i in available_indicators],
                                 value=available_indicators[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                     ], style={'padding': '5px'}),
                     dcc.Graph(id='concentration_time_t')], style={'width': '78%',
                                                                   'display': 'inline-block',
                                                                   'border-color': 'rgb(220, 220, 220)',
                                                                   'border-style': 'solid', 'padding': '5px'}),
                 html.Div(dcc.Markdown(children=markdown_text_conc_time_t),
                          style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                style={'margin': '100px'}
                )

    def _generate_concentration_time_log(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r_log):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_r_log], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_r_log])]], y=[max(df.loc[
                                                                                                            yaxis_column_name_conc_r_log])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_r_log])]], y=[min(df.loc[
                                                                                                            yaxis_column_name_conc_r_log])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r_log', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div([html.Div(html.H1(
                children='График зависимости прологарифмированной концентрации от времени группа референсного препарата'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         dcc.Markdown(
                                             children="Выберите показатель для оси ОY:"),
                                         dcc.Dropdown(
                                             id='yaxis_column_name_conc_r_log',
                                             options=[{'label': i, 'value': i}
                                                      for i in available_indicators],
                                             value=available_indicators[0]
                                         )
                                     ], style={'width': '48%', 'display': 'inline-block'}),
                                 ], style={'padding': '5px'}),
                                 dcc.Graph(id='concentration_time_r_log')], style={'width': '78%',
                                                                                   'display': 'inline-block',
                                                                                   'border-color': 'rgb(220, 220, 220)',
                                                                                   'border-style': 'solid',
                                                                                   'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_text_conc_time_r_log),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )
        else:
            df = self.settings[0].concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t_log):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_t_log], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_t_log])]], y=[max(df.loc[
                                                                                                            yaxis_column_name_conc_t_log])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_t_log])]], y=[min(df.loc[
                                                                                                            yaxis_column_name_conc_t_log])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t_log', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div([html.Div(html.H1(
                children='График зависимости прологарифмированной концентрации от времени группа тестового препарата'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         dcc.Markdown(
                                             children="Выберите показатель для оси ОY:"),
                                         dcc.Dropdown(
                                             id='yaxis_column_name_conc_t_log',
                                             options=[{'label': i, 'value': i}
                                                      for i in available_indicators],
                                             value=available_indicators[0]
                                         )
                                     ], style={'width': '48%', 'display': 'inline-block'}),
                                 ], style={'padding': '5px'}),
                                 dcc.Graph(id='concentration_time_t_log')], style={'width': '78%',
                                                                                   'display': 'inline-block',
                                                                                   'border-color': 'rgb(220, 220, 220)',
                                                                                   'border-style': 'solid',
                                                                                   'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_text_conc_time_t_log),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )

    def _generate_concentration_time_linlog(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r_linlog, yaxis_type_conc_r_linlog):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_r_linlog], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_r_linlog])]], y=[max(df.loc[
                                                                                                               yaxis_column_name_conc_r_linlog])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_r_linlog])]], y=[min(df.loc[
                                                                                                               yaxis_column_name_conc_r_linlog])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r_linlog,
                                 type=yaxis_type_conc_r_linlog)
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r_linlog', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r_linlog', 'value'),
                               dash.dependencies.Input('yaxis_type_conc_r_linlog', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div(
                [html.Div(html.H1(children='График зависимости концентрации от времени группа референсного препарата'),
                          style={'text-align': 'center'}),
                 html.Div([
                     html.Div([
                         html.Div([
                             dcc.Markdown(
                                 children="Выберите показатель для оси ОY:"),
                             dcc.Dropdown(
                                 id='yaxis_column_name_conc_r_linlog',
                                 options=[{'label': i, 'value': i}
                                          for i in available_indicators],
                                 value=available_indicators[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                         html.Div([dcc.RadioItems(
                             id='yaxis_type_conc_r_linlog',
                             options=[{'label': i, 'value': i}
                                      for i in ['linear', 'log']],
                             value='linear'
                         )], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                     ], style={'padding': '5px'}),
                     dcc.Graph(id='concentration_time_r_linlog')], style={'width': '78%',
                                                                          'display': 'inline-block',
                                                                          'border-color': 'rgb(220, 220, 220)',
                                                                          'border-style': 'solid', 'padding': '5px'}),
                 html.Div(dcc.Markdown(children=markdown_text_conc_time_r),
                          style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                style={'margin': '100px'}
                )
        else:
            df = self.settings[0].concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t_linlog, yaxis_type_conc_t_linlog):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df.loc[
                    yaxis_column_name_conc_t_linlog], name='График'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df.loc[
                                                               yaxis_column_name_conc_t_linlog])]], y=[max(df.loc[
                                                                                                               yaxis_column_name_conc_t_linlog])],
                                         mode='markers', name='Максимум',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df.loc[
                                                               yaxis_column_name_conc_t_linlog])]], y=[min(df.loc[
                                                                                                               yaxis_column_name_conc_t_linlog])],
                                         mode='markers', name='Минимум',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t_linlog,
                                 type=yaxis_type_conc_t_linlog)
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t_linlog', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t_linlog', 'value'),
                               dash.dependencies.Input('yaxis_type_conc_t_linlog', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div(
                [html.Div(html.H1(children='График зависимости концентрации от времени группа тестового препарата'),
                          style={'text-align': 'center'}),
                 html.Div([
                     html.Div([
                         html.Div([
                             dcc.Markdown(
                                 children="Выберите показатель для оси ОY:"),
                             dcc.Dropdown(
                                 id='yaxis_column_name_conc_t_linlog',
                                 options=[{'label': i, 'value': i}
                                          for i in available_indicators],
                                 value=available_indicators[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                         html.Div([dcc.RadioItems(
                             id='yaxis_type_conc_t_linlog',
                             options=[{'label': i, 'value': i}
                                      for i in ['linear', 'log']],
                             value='linear'
                         )], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                     ], style={'padding': '5px'}),
                     dcc.Graph(id='concentration_time_t_linlog')], style={'width': '78%',
                                                                          'display': 'inline-block',
                                                                          'border-color': 'rgb(220, 220, 220)',
                                                                          'border-style': 'solid', 'padding': '5px'}),
                 html.Div(dcc.Markdown(children=markdown_text_conc_time_t),
                          style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                style={'margin': '100px'}
                )

    def _generate_concentration_time_cross(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1
            df_r = self.settings[0].concentration_r_1
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_tr, yaxis_type_conc_tr):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df_t.loc[
                    yaxis_column_name_conc_tr], name='T'))
                fig.add_trace(go.Scatter(x=time, y=df_r.loc[
                    yaxis_column_name_conc_tr], name='R'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_t.loc[
                                                               yaxis_column_name_conc_tr])]], y=[max(df_t.loc[
                                                                                                         yaxis_column_name_conc_tr])],
                                         mode='markers', name='Максимум T',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_t.loc[
                                                               yaxis_column_name_conc_tr])]], y=[min(df_t.loc[
                                                                                                         yaxis_column_name_conc_tr])],
                                         mode='markers', name='Минимум T',
                                         marker=dict(size=15, color='green')))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_r.loc[
                                                               yaxis_column_name_conc_tr])]], y=[max(df_r.loc[
                                                                                                         yaxis_column_name_conc_tr])],
                                         mode='markers', name='Максимум R',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_r.loc[
                                                               yaxis_column_name_conc_tr])]], y=[min(df_r.loc[
                                                                                                         yaxis_column_name_conc_tr])],
                                         mode='markers', name='Минимум R',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_tr,
                                 type=yaxis_type_conc_tr)
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_tr', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_tr', 'value'),
                               dash.dependencies.Input('yaxis_type_conc_tr', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Индивидуальные графики концентрации для пациентов группа TR'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         dcc.Markdown(
                                             children="Выберите показатель для оси ОY:"),
                                         dcc.Dropdown(
                                             id='yaxis_column_name_conc_tr',
                                             options=[{'label': i, 'value': i}
                                                      for i in available_indicators],
                                             value=available_indicators[0]
                                         )
                                     ], style={'width': '48%', 'display': 'inline-block'}),
                                     html.Div([dcc.RadioItems(
                                         id='yaxis_type_conc_tr',
                                         options=[{'label': i, 'value': i}
                                                  for i in ['linear', 'log']],
                                         value='linear'
                                     )], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                                 ], style={'padding': '5px'}),
                                 dcc.Graph(id='concentration_time_tr')], style={'width': '78%',
                                                                                'display': 'inline-block',
                                                                                'border-color': 'rgb(220, 220, 220)',
                                                                                'border-style': 'solid',
                                                                                'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_concentration_time_cross_tr),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )
        else:
            df_t = self.settings[0].concentration_t_2
            df_r = self.settings[0].concentration_r_2
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_rt, yaxis_type_conc_rt):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df_t.loc[
                    yaxis_column_name_conc_rt], name='T'))
                fig.add_trace(go.Scatter(x=time, y=df_r.loc[
                    yaxis_column_name_conc_rt], name='R'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_t.loc[
                                                               yaxis_column_name_conc_rt])]], y=[max(df_t.loc[
                                                                                                         yaxis_column_name_conc_rt])],
                                         mode='markers', name='Максимум T',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_t.loc[
                                                               yaxis_column_name_conc_rt])]], y=[min(df_t.loc[
                                                                                                         yaxis_column_name_conc_rt])],
                                         mode='markers', name='Минимум T',
                                         marker=dict(size=15, color='green')))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_r.loc[
                                                               yaxis_column_name_conc_rt])]], y=[max(df_r.loc[
                                                                                                         yaxis_column_name_conc_rt])],
                                         mode='markers', name='Максимум R',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_r.loc[
                                                               yaxis_column_name_conc_rt])]], y=[min(df_r.loc[
                                                                                                         yaxis_column_name_conc_rt])],
                                         mode='markers', name='Минимум R',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_rt,
                                 type=yaxis_type_conc_rt)
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_rt', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_rt', 'value'),
                               dash.dependencies.Input('yaxis_type_conc_rt', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Индивидуальные графики концентрации для пациентов группа RT'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         dcc.Markdown(
                                             children="Выберите показатель для оси ОY:"),
                                         dcc.Dropdown(
                                             id='yaxis_column_name_conc_rt',
                                             options=[{'label': i, 'value': i}
                                                      for i in available_indicators],
                                             value=available_indicators[0]
                                         )
                                     ], style={'width': '48%', 'display': 'inline-block'}),
                                     html.Div([dcc.RadioItems(
                                         id='yaxis_type_conc_rt',
                                         options=[{'label': i, 'value': i}
                                                  for i in ['linear', 'log']],
                                         value='linear'
                                     )], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                                 ], style={'padding': '5px'}),
                                 dcc.Graph(id='concentration_time_rt')], style={'width': '78%',
                                                                                'display': 'inline-block',
                                                                                'border-color': 'rgb(220, 220, 220)',
                                                                                'border-style': 'solid',
                                                                                'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_concentration_time_cross_rt),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )

    def _generate_concentration_time_mean(self):
        df_r = self.settings[0].concentration_r.mean()
        df_t = self.settings[0].concentration_t.mean()
        time = df_t.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=df_r, name='R'))
        fig.add_trace(go.Scatter(x=time, y=df_t, name='T'))
        fig.add_trace(go.Scatter(x=[time[np.argmax(df_t)]], y=[max(df_t)], mode='markers', name='Максимум T',
                                 marker=dict(size=15, color='violet')))
        fig.add_trace(go.Scatter(x=[time[np.argmin(df_t)]], y=[min(df_t)], mode='markers', name='Минимум T',
                                 marker=dict(size=15, color='green')))
        fig.add_trace(go.Scatter(x=[time[np.argmax(df_r)]], y=[max(df_r)], mode='markers', name='Максимум R',
                                 marker=dict(size=15, color='violet')))
        fig.add_trace(go.Scatter(x=[time[np.argmin(df_r)]], y=[min(df_r)], mode='markers', name='Минимум R',
                                 marker=dict(size=15, color='green')))
        fig.update_xaxes(title='Время')
        fig.update_yaxes(title='Концентрация',
                         type='linear')

        return html.Div([html.Div(html.H1(children='Обобщенный график зависимости концентрации от времени'),
                                  style={'text-align': 'center'}),
                         html.Div([
                             dcc.Graph(id='concentration_time_mean', figure=fig)],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'}),
                         html.Div(dcc.Markdown(children=markdown_text_conc_time_mean),
                                  style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                        style={'margin': '100px'}
                        )

    def _generate_concentration_time_log_mean(self):
        df_r = self.settings[0].concentration_r.mean()
        df_t = self.settings[0].concentration_t.mean()
        time = df_t.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=df_r, name='R'))
        fig.add_trace(go.Scatter(x=time, y=df_t, name='T'))
        fig.add_trace(go.Scatter(x=[time[np.argmax(df_t)]], y=[max(df_t)], mode='markers', name='Максимум T',
                                 marker=dict(size=15, color='violet')))
        fig.add_trace(go.Scatter(x=[time[np.argmin(df_t)]], y=[min(df_t)], mode='markers', name='Минимум T',
                                 marker=dict(size=15, color='green')))
        fig.add_trace(go.Scatter(x=[time[np.argmax(df_r)]], y=[max(df_r)], mode='markers', name='Максимум R',
                                 marker=dict(size=15, color='violet')))
        fig.add_trace(go.Scatter(x=[time[np.argmin(df_r)]], y=[min(df_r)], mode='markers', name='Минимум R',
                                 marker=dict(size=15, color='green')))
        fig.update_xaxes(title='Время')
        fig.update_yaxes(title='Концентрация',
                         type='log')

        return html.Div([html.Div(html.H1(
            children='Обобщенный график зависимости прологарифмированной концентрации от времени'),
            style={'text-align': 'center'}),
            html.Div([
                dcc.Graph(id='concentration_time_r_log_mean', figure=fig)],
                style={'width': '78%', 'display': 'inline-block',
                       'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'}),
            html.Div(dcc.Markdown(children=markdown_text_conc_time_log_mean),
                     style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
            style={'margin': '100px'}
        )

    def _generate_concentration_time_linlog_mean(self):
        df_r = self.settings[0].concentration_r.mean()
        df_t = self.settings[0].concentration_t.mean()
        time = df_t.index

        def update_graph(yaxis_type_conc_linlog_mean):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=df_r, name='R'))
            fig.add_trace(go.Scatter(x=time, y=df_t, name='T'))
            fig.add_trace(go.Scatter(x=[time[np.argmax(df_t)]], y=[max(df_t)], mode='markers', name='Максимум T',
                                     marker=dict(size=15, color='violet')))
            fig.add_trace(go.Scatter(x=[time[np.argmin(df_t)]], y=[min(df_t)], mode='markers', name='Минимум T',
                                     marker=dict(size=15, color='green')))
            fig.add_trace(go.Scatter(x=[time[np.argmax(df_r)]], y=[max(df_r)], mode='markers', name='Максимум R',
                                     marker=dict(size=15, color='violet')))
            fig.add_trace(go.Scatter(x=[time[np.argmin(df_r)]], y=[min(df_r)], mode='markers', name='Минимум R',
                                     marker=dict(size=15, color='green')))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(title='Концентрация',
                             type=yaxis_type_conc_linlog_mean)
            return fig

        self.app.callback(dash.dependencies.Output('concentration_time_linlog_mean', 'figure'),
                          [dash.dependencies.Input('yaxis_type_conc_linlog_mean', 'value')])(update_graph)

        return html.Div([html.Div(html.H1(children='Обобщенный график зависимости концентрации от времени'),
                                  style={'text-align': 'center'}),
                         html.Div([
                             html.Div([
                                 html.Div([dcc.RadioItems(
                                     id='yaxis_type_conc_linlog_mean',
                                     options=[{'label': i, 'value': i}
                                              for i in ['linear', 'log']],
                                     value='linear'
                                 )], style={'width': '48%', 'display': 'inline-block'})
                             ], style={'padding': '5px'}),
                             dcc.Graph(id='concentration_time_linlog_mean')],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'}),
                         html.Div(dcc.Markdown(children=markdown_text_conc_time_mean),
                                  style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                        style={'margin': '100px'}
                        )

    def _generate_group_mean(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1.mean()
            df_r = self.settings[0].concentration_r_1.mean()
            time = df_t.index

            def update_graph(group_mean_type_tr):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df_t, name='T'))
                fig.add_trace(go.Scatter(x=time, y=df_r, name='R'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_t)]], y=[max(df_t)], mode='markers', name='Максимум T',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_t)]], y=[min(df_t)], mode='markers', name='Минимум T',
                                         marker=dict(size=15, color='green')))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_r)]], y=[max(df_r)], mode='markers', name='Максимум R',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_r)]], y=[min(df_r)], mode='markers', name='Минимум R',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(type=group_mean_type_tr, title='Концентрация')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_tr_mean', 'figure'),
                              dash.dependencies.Input('group_mean_type_tr', 'value'))(update_graph)

            return html.Div([html.Div(html.H1(children='Средняя концентрация от времени группа TR'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([dcc.RadioItems(
                                     id='group_mean_type_tr',
                                     options=[{'label': i, 'value': i}
                                              for i in ['linear', 'log']],
                                     value='linear'
                                 )], style={'width': '48%', 'display': 'inline-block'}),
                                 dcc.Graph(id='concentration_time_tr_mean')],
                                 style={'width': '78%', 'display': 'inline-block',
                                        'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_group_mean_tr),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )
        else:
            df_t = self.settings[0].concentration_t_2.mean()
            df_r = self.settings[0].concentration_r_2.mean()
            time = df_t.index

            def update_graph(group_mean_type_rt):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=df_t, name='T'))
                fig.add_trace(go.Scatter(x=time, y=df_r, name='R'))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_t)]], y=[max(df_t)], mode='markers', name='Максимум T',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_t)]], y=[min(df_t)], mode='markers', name='Минимум T',
                                         marker=dict(size=15, color='green')))
                fig.add_trace(go.Scatter(x=[time[np.argmax(df_r)]], y=[max(df_r)], mode='markers', name='Максимум R',
                                         marker=dict(size=15, color='violet')))
                fig.add_trace(go.Scatter(x=[time[np.argmin(df_r)]], y=[min(df_r)], mode='markers', name='Минимум R',
                                         marker=dict(size=15, color='green')))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(type=group_mean_type_rt, title='Концентрация')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_rt_mean', 'figure'),
                              dash.dependencies.Input('group_mean_type_rt', 'value'))(update_graph)

            return html.Div([html.Div(html.H1(children='Средняя концентрация от времени группа RT'),
                                      style={'text-align': 'center'}),
                             html.Div([
                                 html.Div([dcc.RadioItems(
                                     id='group_mean_type_rt',
                                     options=[{'label': i, 'value': i}
                                              for i in ['linear', 'log']],
                                     value='linear'
                                 )], style={'width': '48%', 'display': 'inline-block'}),
                                 dcc.Graph(id='concentration_time_rt_mean')],
                                 style={'width': '78%', 'display': 'inline-block',
                                        'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'padding': '5px'}),
                             html.Div(dcc.Markdown(children=markdown_group_mean_rt),
                                      style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                            style={'margin': '100px'}
                            )

    def _generate_drug_mean(self):
        df_t_1 = self.settings[0].concentration_t_1.mean()
        df_t_2 = self.settings[0].concentration_t_2.mean()
        df_r_1 = self.settings[0].concentration_r_1.mean()
        df_r_2 = self.settings[0].concentration_r_2.mean()
        time = df_t_1.index

        def update_graph(drug_mean_type):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=(
                                                       df_t_1 + df_t_2) / 2, name='T'))
            fig.add_trace(go.Scatter(x=time, y=(
                                                       df_r_1 + df_r_2) / 2, name='R'))
            fig.add_trace(go.Scatter(x=[time[np.argmax((df_t_1 + df_t_2) / 2)]], y=[max((df_t_1 + df_t_2) / 2)],
                                     mode='markers', name='Максимум T',
                                     marker=dict(size=15, color='violet')))
            fig.add_trace(go.Scatter(x=[time[np.argmin((df_t_1 + df_t_2) / 2)]], y=[min((df_t_1 + df_t_2) / 2)],
                                     mode='markers', name='Минимум T',
                                     marker=dict(size=15, color='green')))
            fig.add_trace(
                go.Scatter(x=[time[np.argmax((df_r_1 + df_r_2) / 2)]], y=[max((df_r_1 + df_r_2) / 2)], mode='markers',
                           name='Максимум R',
                           marker=dict(size=15, color='violet')))
            fig.add_trace(
                go.Scatter(x=[time[np.argmin((df_r_1 + df_r_2) / 2)]], y=[min((df_r_1 + df_r_2) / 2)], mode='markers',
                           name='Минимум R',
                           marker=dict(size=15, color='green')))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type=drug_mean_type, title='Концентрация')
            return fig

        self.app.callback(dash.dependencies.Output('drug_mean', 'figure'),
                          dash.dependencies.Input('drug_mean_type', 'value'))(update_graph)

        return html.Div([html.Div(html.H1(children='Обобщенные данные по двум препаратам'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div([dcc.RadioItems(
                             id='drug_mean_type',
                             options=[{'label': i, 'value': i}
                                      for i in ['linear', 'log']],
                             value='linear'
                         )], style={'width': '48%', 'display': 'inline-block'}),
                             dcc.Graph(id='drug_mean')],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'}),
                         html.Div(dcc.Markdown(children=markdown_drug_mean),
                                  style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
                        style={'margin': '100px'}
                        )


    def _generate_statistics(self):
        if self.settings[0].plan == 'parallel':
            data = {'Группа': ['R', 'T'],
                    'Mean': [float(self.settings[0].mean_r), 
                    float(self.settings[0].mean_t)],
                    'Std': [float(self.settings[0].std_r), 
                    float(self.settings[0].std_t)],
                    'Min': [float(self.settings[0].min_r),
                     float(self.settings[0].min_t)],
                    'Max': [float(self.settings[0].max_r), 
                    float(self.settings[0].max_t)],
                    'Geom Mean': [float(self.settings[0].geom_mean_r), 
                    float(self.settings[0].geom_mean_t)],
                    'Variation': [float(self.settings[0].variation_r), 
                    float(self.settings[0].variation_t)]}
        else:
            data = {'Группа': ['TR', 'RT'],
                    'Mean_T': [float(self.settings[0].mean_t_1), 
                    float(self.settings[0].mean_t_2)],
                    'Mean_R': [float(self.settings[0].mean_r_1), 
                    float(self.settings[0].mean_r_2)],
                    'Std_T': [float(self.settings[0].std_t_1), 
                    float(self.settings[0].std_t_2)],
                    'Std_R': [float(self.settings[0].std_r_1), 
                    float(self.settings[0].std_r_2)],
                    'Min_T': [float(self.settings[0].min_t_1),
                     float(self.settings[0].min_t_2)],
                    'Min_R': [float(self.settings[0].min_r_1),
                     float(self.settings[0].min_r_2)],
                    'Max_T': [float(self.settings[0].max_t_1), 
                    float(self.settings[0].max_t_2)],
                    'Max_R': [float(self.settings[0].max_r_1), 
                    float(self.settings[0].max_r_2)],
                    'Geom_mean_T': [float(self.settings[0].geom_mean_t_1), 
                    float(self.settings[0].geom_mean_t_2)],
                    'Geom_mean_R': [float(self.settings[0].geom_mean_r_1), 
                    float(self.settings[0].geom_mean_r_2)],
                    'Variation_T': [float(self.settings[0].variation_t_1), 
                    float(self.settings[0].variation_t_2)],
                    'Variation_R': [float(self.settings[0].variation_r_1), 
                    float(self.settings[0].variation_r_2)]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Таблица с распределением статистики по группам'), 
            style={'text-align': 'center'}),
                            html.Div([
                                html.Div([
                                    html.Div([dash_table.DataTable(
                                        id='param_1',
                                        columns=[{"name": i, "id": i, "deletable": True}
                                                for i in df.columns],
                                        data=df.to_dict('records'),
                                        style_table={'overflowX': 'auto'},
                                        export_format='xlsx'
                                    )], style={'border-color': 'rgb(220, 220, 220)', 
                                    'border-style': 'solid', 'padding': '5px', 'margin': '5px'})],
                                    style={'width': '78%', 'display': 'inline-block'}),
                                html.Div(dcc.Markdown(children=markdown_statistics), 
                                style={'width': '18%', 'float': 'right', 'display': 'inline-block',
                                                                                            'padding': '5px', 'margin': '5px'})
                            ])
                            ], style={'margin': '50px'}
                        )