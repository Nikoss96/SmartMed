import numpy as np

from preprocessing.preprocessing import get_categorical_col


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

    def generate_test_kolmagorova_smirnova(self):
        pass

        # def update_table(var, group_var):
        #     if group_var is None:
        #         data = np.array(self.data[var])
        #         data = preprocessing.normalize([data])
        #         res = kstest(data, 'norm')
        #         print(res)
        #         if res[1] < 0.001:
        #             p = '< 0.001'
        #         else:
        #             p = np.round(res[1], 3)
        #         df = pd.DataFrame(columns=['Значение', 'p-value'])
        #         df.loc[1] = [np.round(res[0], 3), p]
        #     else:
        #         classes = get_class_names(group_var, self.settings['path'],
        #                                   self.data)
        #         class1 = list(classes.keys())[0]
        #         class2 = list(classes.keys())[1]
        #         data1 = self.data[self.data[group_var] == class1][var]
        #         data2 = self.data[self.data[group_var] == class2][var]
        #
        #         print("это информация!!!!!!!!!!!!!!!")
        #         print(data2)
        #
        #         data1 = preprocessing.normalize([data1])
        #         data2 = preprocessing.normalize([data2])
        #         res1 = kstest(data1, 'norm')
        #         res2 = kstest(data2, 'norm')
        #         if res1[1] < 0.001:
        #             p1 = '< 0.001'
        #         else:
        #             p1 = np.round(res1[1], 3)
        #
        #         if res2[1] < 0.001:
        #             p2 = '< 0.001'
        #         else:
        #             p2 = np.round(res2[1], 3)
        #         classes = get_class_names(group_var, self.settings['path'],
        #                                   self.data)
        #         df = pd.DataFrame(columns=['Группы', 'Значение', 'p-value'])
        #         df.loc[1] = [classes[class1], np.round(res1[0], 3), p1]
        #         df.loc[2] = [classes[class2], np.round(res2[0], 3), p2]
        #     return df.to_dict('records'), [{"name": i, "id": i} for i in
        #                                    df.columns]
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
