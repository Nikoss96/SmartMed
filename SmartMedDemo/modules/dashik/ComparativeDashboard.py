from .Dashboard import Dashboard
from ..dataprep.PandasPreprocessor import get_categorical_col
from ..dash.text.comparative_text import *
from sklearn import preprocessing
from ..dataprep.PandasPreprocessor import get_confusion_matrix
from ..dataprep.PandasPreprocessor import get_class_names
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import t
from math import sqrt
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import chi2_contingency
from scipy.stats import binomtest
from scipy.stats import f
from scipy.stats.distributions import chi2
from scipy.stats import kstest
from sklearn.metrics import confusion_matrix


class ComparativeDashboard(Dashboard):

    def _generate_layout(self):
        # metrics inludings is checked inside method
        method_list = []
        for method in self.settings['methods']:
            method_list.append(self.graph_to_method[method]())

        return html.Div([
            html.Div(html.H1(children='Сравнительный анализ'), style={'text-align': 'center'}),
            html.Div(html.H3(children=self.settings['type']), style={'text-align': 'center'}),
            html.Div(method_list)])

    def _generate_test_kolmagorova_smirnova(self):
        print("Колиагорова!!!!!!!!!")
        cat_columns = np.array(get_categorical_col(self.data))
        # cat_columns_full = np.insert(cat_columns, 0, 'Без группирующей')
        columns_list = np.array(self.data.columns)
        cont_columns = [v for v in columns_list if v not in set(cat_columns) & set(columns_list)]

        def update_table(var, group_var):
            if group_var is None:
                data = np.array(self.data[var])
                data = preprocessing.normalize([data])
                res = kstest(data, 'norm')
                print(res)
                if res[1] < 0.001:
                    p = '< 0.001'
                else:
                    p = np.round(res[1], 3)
                df = pd.DataFrame(columns=['Значение', 'p-value'])
                df.loc[1] = [np.round(res[0], 3), p]
            else:
                classes = get_class_names(group_var, self.settings['path'], self.data)
                class1 = list(classes.keys())[0]
                class2 = list(classes.keys())[1]
                data1 = self.data[self.data[group_var] == class1][var]
                data2 = self.data[self.data[group_var] == class2][var]

                print("это информация!!!!!!!!!!!!!!!")
                print(data2)

                data1 = preprocessing.normalize([data1])
                data2 = preprocessing.normalize([data2])
                res1 = kstest(data1, 'norm')
                res2 = kstest(data2, 'norm')
                if res1[1] < 0.001:
                    p1 = '< 0.001'
                else:
                    p1 = np.round(res1[1], 3)

                if res2[1] < 0.001:
                    p2 = '< 0.001'
                else:
                    p2 = np.round(res2[1], 3)
                classes = get_class_names(group_var, self.settings['path'], self.data)
                df = pd.DataFrame(columns=['Группы', 'Значение', 'p-value'])
                df.loc[1] = [classes[class1], np.round(res1[0], 3), p1]
                df.loc[2] = [classes[class2], np.round(res2[0], 3), p2]
            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

        self.app.callback(dash.dependencies.Output('kolm_smirn_table', 'data'),
                          dash.dependencies.Output('kolm_smirn_table', 'columns'),
                          dash.dependencies.Input('kolm_smirn_param', 'value'),
                          dash.dependencies.Input('kolm_smirn_group_param', 'value'))(update_table)

        return html.Div([html.Div(html.H4(children='Критерий Колмогорова-Смирнова'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_kolm_smirn_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             dcc.Markdown(children="Выберите независимую переменную:"),
                             dcc.Dropdown(
                                 id='kolm_smirn_param',
                                 options=[{'label': i, 'value': i}
                                          for i in cont_columns],
                                 value=columns_list[0]
                             )], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(children="Выберите группирующую переменную:"),
                                 dcc.Dropdown(
                                     id='kolm_smirn_group_param',
                                     options=[{'label': i, 'value': i}
                                              for i in cat_columns])
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_variables_with_group), style={'padding': '10px'})]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='kolm_smirn_table',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                             html.Div(dcc.Markdown(markdown_kolm_smirn), style={'padding': '20px'})])
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_t_criterion_student_independent(self):

        def independent_ttest(x, y, alpha):
            # calculate means
            mean1, mean2 = np.mean(x), np.mean(y)
            # calculate standard errors
            se1, se2 = sem(x), sem(y)
            # standard error on the difference between the samples
            sed = sqrt(se1 ** 2.0 + se2 ** 2.0)
            # calculate the t statistic
            t_stat = abs(mean1 - mean2) / sed
            # degrees of freedom
            if se1 / se2 > 10 or se1 / se2 > 10:
                f = (len(x) + len(y) - 2) * (0.5 + se1 * se2 / (se1 ** 2 + se2 ** 2))
            else:
                f = len(x) + len(y) - 2
            # calculate the critical value
            cv = t.ppf(1.0 - alpha, f)
            # calculate the p-value
            p = (1.0 - t.cdf(abs(t_stat), f)) * 2.0
            if p < 0.001:
                p = "< 0.001"
            else:
                p = str(np.round(p, 3))

            return np.round(t_stat, 3), np.round(f, 3), np.round(cv, 3), p

        cat_columns = get_categorical_col(self.data)
        columns_list = np.array(self.data.columns)
        cont_columns = [v for v in columns_list if v not in set(cat_columns) & set(columns_list)]
        result_columns = ['Доверительная вероятность', 'Эмпирическое значение', 'Критическое значение',
                          'Число степеней свободы']

        def update_student_table(group_var, var):
            classes = get_class_names(group_var, self.settings['path'], self.data)
            class1 = list(classes.keys())[0]
            class2 = list(classes.keys())[1]
            data1 = self.data[self.data[group_var] == class1][var]
            data2 = self.data[self.data[group_var] == class2][var]
            results = independent_ttest(data1, data2, 0.05)
            res_list = ["alpha = 0.95", results[0], results[2], results[1]]
            df = pd.DataFrame(columns=result_columns)
            df.loc[1] = res_list

            mean_var1 = np.round(np.mean(data1), 3)
            std_var1 = np.round(np.std(data1), 3)
            mean_var2 = np.round(np.mean(data2), 3)
            std_var2 = np.round(np.std(data2), 3)

            res_list2 = [var, str(mean_var1)+" ± "+str(std_var1), str(mean_var2)+" ± "+str(std_var2), results[3]]
            mean_p_columns_header = ["Характеристика", str(classes[class1])+" (n1 = "+str(len(data1))+")",
                                     str(classes[class2])+" (n2 = " + str(len(data2))+")", "p-value"]
            df2 = pd.DataFrame(columns=mean_p_columns_header)
            df2.loc[1] = res_list2
            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], \
                   df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns]

        self.app.callback(dash.dependencies.Output('table_t_criterion_student_ind', 'data'),
                          dash.dependencies.Output('table_t_criterion_student_ind', 'columns'),
                          dash.dependencies.Output('table_student_ind_mean_p', 'data'),
                          dash.dependencies.Output('table_student_ind_mean_p', 'columns'),
                          dash.dependencies.Input('student_ind_group_param', 'value'),
                          dash.dependencies.Input('student_ind_param', 'value'))(update_student_table)

        return html.Div([html.Div(html.H4(children='Т-критерий Стьюдента для независимых переменных'),
                                  style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_student_ind_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную:"),
                             dcc.Dropdown(
                                 id='student_ind_group_param',
                                 options=[{'label': i, 'value': i}
                                          for i in cat_columns],
                                 value=cat_columns[0]
                             )], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(children="Выберите независимую переменную:"),
                                 dcc.Dropdown(
                                     id='student_ind_param',
                                     options=[{'label': i, 'value': i}
                                              for i in cont_columns],
                                     value=cont_columns[0])
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_variables_with_group), style={'padding': '10px'})]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_t_criterion_student_ind',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(" "))]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_student_ind_mean_p',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_p_value), style={'padding': '20px'})],
                             style={'padding': '20px'})
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_t_criterion_student_dependent(self):

        def dependent_ttest(data1, data2, alpha):
            # calculate means
            mean1, mean2 = np.mean(data1), np.mean(data2)
            # number of paired samples
            n = len(data1)
            # sum squared difference between observations
            d1 = sum([(data1[i] - data2[i]) ** 2 for i in range(n)])
            # sum difference between observations
            d2 = sum([data1[i] - data2[i] for i in range(n)])
            # standard deviation of the difference between means
            sd = sqrt((d1 - (d2 ** 2 / n)) / (n - 1))
            # standard error of the difference between the means
            sed = sd / sqrt(n)
            sed += 10**-5
            # calculate the t statistic
            t_stat = abs(mean1 - mean2) / sed
            # degrees of freedom
            df = n - 1
            # calculate the critical value
            cv = t.ppf(1.0 - alpha, df)
            # calculate the p-value
            p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
            if p < 0.001:
                p = "< 0.001"
            else:
                p = str(np.round(p, 3))
            return np.round(t_stat, 3), np.round(df, 3), np.round(cv, 3), p

        cat_columns = get_categorical_col(self.data)
        columns_list = np.array(self.data.columns)
        cont_columns = [v for v in columns_list if v not in set(cat_columns) & set(columns_list)]
        result_columns = ['Доверительная вероятность', 'Эмпирическое значение', 'Критическое значение',
                          'Число степеней свободы']

        def update_student_table(var_1, var_2):
            data1 = self.data[var_1]
            data2 = self.data[var_2]
            results = dependent_ttest(data1, data2, 0.05)
            res_list = ["alpha = 0.95", results[0], results[2], results[1]]
            df = pd.DataFrame(columns=result_columns)
            df.loc[1] = res_list

            mean_var1 = np.round(np.mean(data1), 3)
            std_var1 = np.round(np.std(data1), 3)
            mean_var2 = np.round(np.mean(data2), 3)
            std_var2 = np.round(np.std(data2), 3)

            res_list2 = [str(mean_var1) + " ± " + str(std_var1), str(mean_var2) + " ± " + str(std_var2), results[3]]
            mean_p_columns_header = [var_1 + " (n1 = " + str(len(data1)) + ")", var_2 + " (n2 = " + str(len(data2)) +
                                     ")", "p-value"]
            df2 = pd.DataFrame(columns=mean_p_columns_header)
            df2.loc[1] = res_list2

            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], \
                   df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns]

        self.app.callback(dash.dependencies.Output('table_t_criterion_student_depend', 'data'),
                          dash.dependencies.Output('table_t_criterion_student_depend', 'columns'),
                          dash.dependencies.Output('table_student_depend_mean_p', 'data'),
                          dash.dependencies.Output('table_student_depend_mean_p', 'columns'),
                          dash.dependencies.Input('student_depend_param_1', 'value'),
                          dash.dependencies.Input('student_depend_param_2', 'value'))(update_student_table)

        return html.Div([html.Div(html.H4(children='Т-критерий Стьюдента для зависимых переменных'),
                                  style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_student_depend_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             dcc.Markdown(children="Выберите первую переменную:"),
                             dcc.Dropdown(
                                 id='student_depend_param_1',
                                 options=[{'label': i, 'value': i}
                                          for i in cont_columns],
                                 value=cont_columns[0]
                             )], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(children="Выберите вторую переменную:"),
                                 dcc.Dropdown(
                                     id='student_depend_param_2',
                                     options=[{'label': i, 'value': i}
                                              for i in cont_columns],
                                     value=cont_columns[0])
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_variables), style={'padding': '10px'})]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_t_criterion_student_depend',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(" "))]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_student_depend_mean_p',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_p_value), style={'padding': '20px'})],
                             style={'padding': '20px'})
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_u_criterion_mann_whitney(self):

        results_columns = ['Доверительная вероятность', 'Эмпирическое значение', 'Кол-во значений в 1 выборке',
                           'Кол-во значений во 2 выборке']
        cat_columns = get_categorical_col(self.data)
        columns_list = np.array(self.data.columns)
        cont_columns = [v for v in columns_list if v not in set(cat_columns) & set(columns_list)]

        def update_table_mann_whitney(group_var, var):
            classes = get_class_names(group_var, self.settings['path'], self.data)
            class1 = list(classes.keys())[0]
            class2 = list(classes.keys())[1]
            data1 = self.data[self.data[group_var] == class1][var]
            data2 = self.data[self.data[group_var] == class2][var]
            stat, p = mannwhitneyu(data1, data2)
            if p < 0.001:
                p = "< 0.001"
            else:
                p = str(np.round(p, 3))
            df = pd.DataFrame(columns=results_columns)
            df.loc[1] = ['alpha = 0.95', stat, "n1 = " + str(len(data1)), "n2 = " + str(len(data2))]

            classes = get_class_names(group_var, self.settings['path'], self.data)

            results_columns2 = ["Характеристика", str(classes[class1]) + " (n1 = " + str(len(data1)) + ")",
                                str(classes[class2]) + " (n2 = " + str(len(data2)) + ")", "p-value"]
            df2 = pd.DataFrame(columns=results_columns2)
            df2.loc[1] = [var, str(data1.median()) + " (" + str(np.percentile(data1, 25)) + " - " +
                          str(np.percentile(data1, 75)) + ")", str(data2.median()) + " (" +
                          str(np.percentile(data2, 25)) + " - " + str(np.percentile(data2, 75)) + ")", p]

            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],\
                   df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns]

        self.app.callback(dash.dependencies.Output('table_u_criterion_mann_whitney', 'data'),
                          dash.dependencies.Output('table_u_criterion_mann_whitney', 'columns'),
                          dash.dependencies.Output('table_median_mann_whitney', 'data'),
                          dash.dependencies.Output('table_median_mann_whitney', 'columns'),
                          dash.dependencies.Input('whitney_group_param', 'value'),
                          dash.dependencies.Input('whitney_param', 'value'))(update_table_mann_whitney)

        return html.Div([html.Div(html.H4(children='U-критерий Манна-Уитни'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_mann_whitney_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную:"),
                                             dcc.Dropdown(
                                                 id='whitney_group_param',
                                                 options=[{'label': i, 'value': i} for i in cat_columns],
                                                 value=cat_columns[0]
                                             )], style={'width': '48%', 'display': 'inline-block'}),
                                   html.Div([
                                       dcc.Markdown(
                                           children="Выберите независимую переменную:"),
                                       dcc.Dropdown(
                                           id='whitney_param',
                                           options=[{'label': i, 'value': i} for i in cont_columns],
                                           value=cont_columns[1])
                                   ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                                   html.Div(dcc.Markdown(markdown_student_variables_with_group), style={'padding': '10px'})]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_u_criterion_mann_whitney',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(" "))]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_median_mann_whitney',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_p_value), style={'padding': '20px'})],
                         style={'padding': '20px'})
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_t_criterion_wilcoxon(self):

        cat_columns = get_categorical_col(self.data)
        columns_list = np.array(self.data.columns)
        cont_columns = [v for v in columns_list if v not in set(cat_columns) & set(columns_list)]
        results_columns = ['Доверительная вероятность', 'Эмпирическое значение', 'Кол-во значений в выборке']

        def update_wilcoxon_table(var1, var2):
            data1 = self.data[var1]
            data2 = self.data[var2]
            stat, p = wilcoxon(data1, data2)
            if p < 0.001:
                p = "< 0.001"
            else:
                p = str(np.round(p, 3))
            df = pd.DataFrame(columns=results_columns)
            df.loc[1] = ['alpha = 0.95', stat, "n = " + str(len(data1))]

            results_columns2 = [var1, var2, 'p-value']
            df2 = pd.DataFrame(columns=results_columns2)
            df2.loc[1] = [str(data1.median()) + " (" + str(np.percentile(data1, 25)) + " - " +
                          str(np.percentile(data1, 75)) + ")", str(data2.median()) + " (" +
                          str(np.percentile(data2, 25)) + " - " + str(np.percentile(data2, 75)) + ")", p]

            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], \
                   df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns]

        self.app.callback(dash.dependencies.Output('table_t_criterion_wilcoxon', 'data'),
                          dash.dependencies.Output('table_t_criterion_wilcoxon', 'columns'),
                          dash.dependencies.Output('table_t_median_wilcoxon', 'data'),
                          dash.dependencies.Output('table_t_median_wilcoxon', 'columns'),
                          dash.dependencies.Input('wilcoxon_group_param_1', 'value'),
                          dash.dependencies.Input('wilcoxon_group_param_2', 'value'))(update_wilcoxon_table)

        return html.Div([html.Div(html.H4(children='T-критерий Уилкоксона'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_wilcoxon_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             dcc.Markdown(
                                 children="Выберите первую переменную:"),
                             dcc.Dropdown(
                                 id='wilcoxon_group_param_1',
                                 options=[{'label': i, 'value': i} for i in cont_columns],
                                 value=cont_columns[0])], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите вторую переменную:"),
                                 dcc.Dropdown(
                                     id='wilcoxon_group_param_2',
                                     options=[{'label': i, 'value': i} for i in cont_columns],
                                     value=cont_columns[1])
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_variables), style={'padding': '10px'})]),

                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_t_criterion_wilcoxon',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(" "))]),
                         html.Div([
                             html.Div(dash_table.DataTable(
                                 id='table_t_median_wilcoxon',
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_student_p_value), style={'padding': '20px'})],
                         style={'padding': '20px'})
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_chi2_pearson(self):

        columns_list = get_categorical_col(self.data)

        def update_conjugacy_table(var_1, var_2):
            table = pd.crosstab(self.data[var_1], self.data[var_2])
            class1 = get_class_names(var_1, self.settings['path'], self.data)
            class2 = get_class_names(var_2, self.settings['path'], self.data)
            table.columns = class2.values()
            table.insert(0, var_1 + " \ " + var_2, class1.values())
            return table.to_dict('records'), [{"name": i, "id": i} for i in table.columns]

        self.app.callback(dash.dependencies.Output('conjugacy_table', 'data'),
                          dash.dependencies.Output('conjugacy_table', 'columns'),
                          dash.dependencies.Input('group_param_1_pearson', 'value'),
                          dash.dependencies.Input('group_param_2_pearson', 'value'))(update_conjugacy_table)

        def update_expected_table(var_1, var_2):
            table = pd.crosstab(self.data[var_1], self.data[var_2])
            chi2_, p, f_, expected = chi2_contingency(table)
            expected_df = pd.DataFrame(expected)
            class1 = get_class_names(var_1, self.settings['path'], self.data)
            class2 = get_class_names(var_2, self.settings['path'], self.data)
            expected_df.columns = class2.values()
            expected_df.insert(0, var_1 + " \ " + var_2, class1.values())
            expected_df = expected_df.round(3)
            if p < 0.001:
                p_ = '< 0.001'
            else:
                p_ = np.round(p, 3)
            metrics = pd.DataFrame(columns=['Доверительная вероятность', 'Расчетное значение', 'Критическое значение',
                                            'Степень свободы', 'p-value'])
            metrics.loc[1] = ["alpha = 0.95", "X^2 = "+str(np.round(chi2_, 3)),
                              np.round(chi2.isf(0.05, f_, loc=0, scale=1), 3), f_, p_]
            return expected_df.to_dict('records'), [{"name": i, "id": i} for i in expected_df.columns],\
                   metrics.to_dict('records'), [{"name": i, "id": i} for i in metrics.columns]

        self.app.callback(dash.dependencies.Output('expected_table', 'data'),
                          dash.dependencies.Output('expected_table', 'columns'),
                          dash.dependencies.Output('metrics_table', 'data'),
                          dash.dependencies.Output('metrics_table', 'columns'),
                          dash.dependencies.Input('group_param_1_pearson', 'value'),
                          dash.dependencies.Input('group_param_2_pearson', 'value'))(update_expected_table)

        return html.Div([html.Div(html.H4(children='Хи-квадрат Пирсона'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_chi_2_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите первую группирующую переменную:"),
                                 dcc.Dropdown(
                                     id='group_param_1_pearson',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите вторую группирующую переменную:"),
                                 dcc.Dropdown(
                                     id='group_param_2_pearson',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                         ], style={'padding': '20px'})]),
                         html.Div([html.Div(html.H5(children='Таблица сопряженности'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='conjugacy_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                                   html.Div(dcc.Markdown(" "))], style={'padding': '20px'}),
                         html.Div([html.Div(html.H5(children='Ожидаемая частота'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='expected_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                                   html.Div(dcc.Markdown(" "))], style={'padding': '20px'}),
                         html.Div([html.Div(html.H5(children='Таблица метрик'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='metrics_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block'}),
                                   html.Div(dcc.Markdown(markdown_chi_2))], style={'padding': '20px'}),
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_sensitivity_specificity(self):
        columns_list = self.data.loc[:, self.data.nunique() == 2].columns

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

        def update_table(factor, result):
            result_values = self.data[result].to_list()
            factor_values = self.data[factor].to_list()

            tn, fp, fn, tp = confusion_matrix(result_values, factor_values).ravel()

            table = pd.DataFrame([[tp, fn], [fp, tn]])
            class_result = get_class_names(result, self.settings['path'], self.data)
            class_factor = get_class_names(factor, self.settings['path'], self.data)
            table.columns = class_factor.values()
            table.insert(0, result + " \ " + factor, class_result.values())

            se = tp / (tp + fp)
            sp = tn / (fn + tn)

            dov_in_se_clopper = dov_int_clopper(tp, tp+fp)
            dov_in_se_wilson = dov_int_wilson(tp, tp + fp)
            dov_int_sp_clopper = dov_int_clopper(tn, fn + tn)
            dov_int_sp_wilson = dov_int_wilson(tn, fn + tn)
            table_se_sp = pd.DataFrame(columns=[' ', 'Значение', 'Довирительный интервал Пирсона-Клоппера',
                                                'Доверительный интервал Вилсона'])
            table_se_sp.loc[1] = ['Чувствительность', np.round(se, 3), str(round(dov_in_se_clopper[0], 3)) + '; ' +
                                  str(round(dov_in_se_clopper[1], 3)), str(round(dov_in_se_wilson[0], 3)) + '; ' +
                                  str(round(dov_in_se_wilson[1], 3))]
            table_se_sp.loc[2] = ['Специфичность', np.round(sp, 3), str(round(dov_int_sp_clopper[0], 3)) + '; ' +
                                  str(round(dov_int_sp_clopper[1], 3)), str(round(dov_int_sp_wilson[0], 3)) + '; ' +
                                  str(round(dov_int_sp_wilson[1], 3))]
            return table.to_dict('records'), [{"name": i, "id": i} for i in table.columns], \
                   table_se_sp.to_dict('records'), [{"name": i, "id": i} for i in table_se_sp.columns],

        self.app.callback(dash.dependencies.Output('pn_table', 'data'),
                          dash.dependencies.Output('pn_table', 'columns'),
                          dash.dependencies.Output('se_sp_table', 'data'),
                          dash.dependencies.Output('se_sp_table', 'columns'),
                          dash.dependencies.Input('factor_se_sp', 'value'),
                          dash.dependencies.Input('result_se_sp', 'value'))(update_table)

        return html.Div([html.Div(html.H4(children='Чувствительность и Специфичность'), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите фактор риска:"),
                                 dcc.Dropdown(
                                     id='factor_se_sp',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите исход:"),
                                 dcc.Dropdown(
                                     id='result_se_sp',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                         ], style={'padding': '20px'})]),
                         html.Div([html.Div(html.H5(children='Таблица сопряженности'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='pn_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                                   html.Div(dcc.Markdown(" "))], style={'padding': '20px'}),

                         html.Div([html.Div(html.H5(children='Таблица метрик'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='se_sp_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block'}),
                                   html.Div(dcc.Markdown(markdown_se_sep), style={'padding': '20px'})],
                                  style={'padding': '20px'}),
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_odds_relations(self):
        columns_list = self.data.loc[:, self.data.nunique() == 2].columns

        def update_table(factor, result):
            result_values = self.data[result].to_list()
            factor_values = self.data[factor].to_list()
            tn, fp, fn, tp = confusion_matrix(result_values, factor_values).ravel()

            tp += 0.5 if tp == 0 else 0
            fn += 0.5 if fn == 0 else 0
            fp += 0.5 if fp == 0 else 0
            tn += 0.5 if tn == 0 else 0

            table = pd.DataFrame([[tp, fn], [fp, tn]])
            class_result = get_class_names(result, self.settings['path'], self.data)
            class_factor = get_class_names(factor, self.settings['path'], self.data)
            table.columns = class_factor.values()
            table.insert(0, result + " \ " + factor, class_result.values())

            or_ = (tp * tn) / (fp * fn)
            ub = np.exp(np.log(or_) + 1.96*np.sqrt(1/tp + 1/tn + 1/fp + 1/fn))
            lb = np.exp(np.log(or_) - 1.96*np.sqrt(1/tp + 1/tn + 1/fp + 1/fn))
            if (1 < ub) and (1 > lb):
                p = '≥ 0.05'
            else:
                p = '< 0.05'
            table_res = pd.DataFrame(columns=['Шанс OR', 'Нижняя граница доверительного интервала',
                                              'Верхняя граница доверительного интервала', 'p-value'])
            table_res.loc[1] = [np.round(or_, 3), np.round(lb, 3), np.round(ub, 3), p]
            return table.to_dict('records'), [{"name": i, "id": i} for i in table.columns], \
                   table_res.to_dict('records'), [{"name": i, "id": i} for i in table_res.columns],

        self.app.callback(dash.dependencies.Output('or_table', 'data'),
                          dash.dependencies.Output('or_table', 'columns'),
                          dash.dependencies.Output('or_res_table', 'data'),
                          dash.dependencies.Output('or_res_table', 'columns'),
                          dash.dependencies.Input('factor_odds', 'value'),
                          dash.dependencies.Input('result_odds', 'value'))(update_table)

        return html.Div([html.Div(html.H4(children='Отношение шансов'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_odds_relations_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите фактор риска:"),
                                 dcc.Dropdown(
                                     id='factor_odds',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите исход риска:"),
                                 dcc.Dropdown(
                                     id='result_odds',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                         ], style={'padding': '20px'})]),

                         html.Div([html.Div(html.H5(children='Таблица сопряженности'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='or_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                                   html.Div(dcc.Markdown(" "))], style={'padding': '20px'}),

                         html.Div([html.Div(html.H5(children='Таблица метрик'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='or_res_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block'}),
                                   html.Div(dcc.Markdown(markdown_odds_relations), style={'padding': '20px'})],
                                  style={'padding': '20px'}),
                         ], style={'margin': '50px', 'text-align': 'center'})

    def _generate_risk_relations(self):
        columns_list = self.data.loc[:, self.data.nunique() == 2].columns

        def update_table(factor, result):
            result_values = self.data[result].to_list()
            factor_values = self.data[factor].to_list()
            tn, fp, fn, tp = confusion_matrix(result_values, factor_values).ravel()

            tp += 0.5 if tp == 0 else 0
            fn += 0.5 if fn == 0 else 0
            fp += 0.5 if fp == 0 else 0
            tn += 0.5 if tn == 0 else 0

            table = pd.DataFrame([[tp, fn], [fp, tn]])
            class_result = get_class_names(result, self.settings['path'], self.data)
            class_factor = get_class_names(factor, self.settings['path'], self.data)
            table.columns = class_factor.values()
            table.insert(0, result + " \ " + factor, class_result.values())

            rr = (tp * (fp + tn)) / (fp * (tp + fn))
            ub = np.exp(np.log(rr) + 1.96*np.sqrt(fn/(tp*(tp+fn)) + tn/(fp*(fp+tn))))
            lb = np.exp(np.log(rr) - 1.96*np.sqrt(fn/(tp*(tp+fn)) + tn/(fp*(fp+tn))))
            if (1 < ub) and (1 > lb):
                p = '≥ 0.05'
            else:
                p = '< 0.05'
            table_res = pd.DataFrame(columns=['Риск RR', 'Нижняя граница доверительного интервала',
                                              'Верхняя граница доверительного интервала', 'p-value'])
            table_res.loc[1] = [np.round(rr, 3), np.round(lb, 3), np.round(ub, 3), p]
            return table.to_dict('records'), [{"name": i, "id": i} for i in table.columns], \
                   table_res.to_dict('records'), [{"name": i, "id": i} for i in table_res.columns],

        self.app.callback(dash.dependencies.Output('rr_table', 'data'),
                          dash.dependencies.Output('rr_table', 'columns'),
                          dash.dependencies.Output('rr_res_table', 'data'),
                          dash.dependencies.Output('rr_res_table', 'columns'),
                          dash.dependencies.Input('factor_rr', 'value'),
                          dash.dependencies.Input('result_rr', 'value'))(update_table)

        return html.Div([html.Div(html.H4(children='Отношение рисков'), style={'text-align': 'center'}),
                         html.Div(dcc.Markdown(markdown_risk_relations_head), style={'text-align': 'center'}),
                         html.Div([html.Div([
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите фактор риска:"),
                                 dcc.Dropdown(
                                     id='factor_rr',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'display': 'inline-block'}),
                             html.Div([
                                 dcc.Markdown(
                                     children="Выберите исход:"),
                                 dcc.Dropdown(
                                     id='result_rr',
                                     options=[{'label': i, 'value': i}
                                              for i in columns_list],
                                     value=columns_list[0]
                                 )
                             ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                         ], style={'padding': '20px'})]),

                         html.Div([html.Div(html.H5(children='Таблица сопряженности'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='rr_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block', 'width': '50%'}),
                                   html.Div(dcc.Markdown(" "))], style={'padding': '20px'}),

                         html.Div([html.Div(html.H5(children='Таблица метрик'), style={'text-align': 'center'}),
                                   html.Div(dash_table.DataTable(
                                       id='rr_res_table',
                                       export_format='xlsx',
                                       style_cell={'textAlign': 'center'}),
                                       style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                              'text-align': 'center', 'display': 'inline-block'}),
                                   html.Div(dcc.Markdown(markdown_risk_relations), style={'padding': '20px'})],
                                  style={'padding': '20px'}),
                         ], style={'margin': '50px', 'text-align': 'center'})
