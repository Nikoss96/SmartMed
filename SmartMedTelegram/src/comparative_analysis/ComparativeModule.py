import pathlib

import numpy as np
import pandas as pd
from scipy.stats import kstest

from data.paths import MEDIA_PATH, DATA_PATH, COMPARATIVE_ANALYSIS, USER_DATA_PATH
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
        print(init_unique_values)
        number_class = []
        data_col = data[group_var].tolist()
        for name in init_unique_values:
            number_class.append(data_col[list(init_df[group_var]).index(name)])
        dict_classes = dict(zip(number_class, init_unique_values))
        return dict_classes

    def generate_test_kolmagorova_smirnova(self, categorical_column, continuous_column):
        classes = self.get_class_names(categorical_column, self.df)

        class1 = list(classes.keys())[0]
        class2 = list(classes.keys())[1]
        data1 = self.df[self.df[categorical_column] == class1][continuous_column]
        data2 = self.df[self.df[categorical_column] == class2][continuous_column]

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

        classes = self.get_class_names(categorical_column, self.df)
        df = pd.DataFrame(columns=["Группы", "Значение", "p-value"])
        df.loc[1] = [classes[class1], np.round(res1[0], 3), p1]
        df.loc[2] = [classes[class2], np.round(res2[0], 3), p2]
        return df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]


def read_file(path):
    ext = pathlib.Path(path).suffix

    if ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path)

    else:
        df = pd.DataFrame()
    return df

    #
    # self.app.callback(dash.dependencies.Output('kolm_smirn_table', 'data'),
    #                   dash.dependencies.Output('kolm_smirn_table',
    #                                            'columns'),
    #                   dash.dependencies.Input('kolm_smirn_param', 'value'),
    #                   dash.dependencies.Input('kolm_smirn_group_param',
    #                                           'value'))(update_table)
    #
    # return html.Div([html.Div(
    #     html.H2(children='Критерий Колмогорова-Смирнова'),
    #     style={'text-align': 'left'}),
    #                  html.Div(dcc.Markdown(markdown_kolm_smirn_head),
    #                           style={'text-align': 'left'}),
    #                  html.Div([html.Div([
    #                      dcc.Markdown(
    #                          children="Выберите независимую переменную:"),
    #                      dcc.Dropdown(
    #                          id='kolm_smirn_param',
    #                          options=[{'label': i, 'value': i}
    #                                   for i in cont_columns],
    #                          value=columns_list[0]
    #                      )],
    #                      style={'width': '48%', 'display': 'inline-block'}),
    #                      html.Div([
    #                          dcc.Markdown(
    #                              children="Выберите группирующую переменную:"),
    #                          dcc.Dropdown(
    #                              id='kolm_smirn_group_param',
    #                              options=[{'label': i, 'value': i}
    #                                       for i in cat_columns])
    #                      ], style={'width': '48%', 'float': 'right',
    #                                'display': 'inline-block'}),
    #                      html.Div(dcc.Markdown(
    #                          markdown_student_variables_with_group),
    #                               style={'padding-bottom': '10px',
    #                                      'padding-top': '10px'})]),
    #                  html.Div([
    #                      html.Div(dash_table.DataTable(
    #                          id='kolm_smirn_table',
    #                          export_format='xlsx',
    #                          style_cell={'textAlign': 'left'}),
    #                          style={'border-color': 'rgb(220, 220, 220)',
    #                                 'border-style': 'solid',
    #                                 'text-align': 'left',
    #                                 'display': 'inline-block',
    #                                 'width': '50%', 'padding-top': '10px',
    #                                 'margin-bottom': '10px'}),
    #                      html.Div(dcc.Markdown(markdown_kolm_smirn),
    #                               style={'padding-top': '10px'})])
    #                  ], style={'margin': '50px', 'text-align': 'left'})
