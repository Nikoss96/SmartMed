import numpy as np

from scipy.stats import variation

from data.paths import USER_DATA_PATH, DATA_PATH, MEDIA_PATH


class DescribeTable:
    def __init__(self, preprocessor):

        self.metrics = ['count', 'mean', 'std', 'max', 'min', '25%', '50%', '75%']
        self.preprocessor = preprocessor
        self.table = self._generate_table()

    def _generate_table(self):
        df = self.preprocessor.get_numeric_df(self.preprocessor.df)
        init_df = df
        df = df.describe().reset_index()
        df = df[df['index'].isin(self.metrics)]
        df = df.rename(columns={"index": "metrics"})
        cols = df.columns
        init_describe_length = len(df)
        for col in init_df.columns:
            df.loc[init_describe_length, col] = np.exp(np.log(init_df[col]).mean())
            df.loc[init_describe_length + 1, col] = variation(init_df[col])
        df.loc[init_describe_length, 'metrics'] = 'geom_mean'
        df.loc[init_describe_length+1, 'metrics'] = 'variation'
        len_t = str(len(df.columns)*10)+'%'
        len_text = str(98-len(df.columns)*10)+'%'
        for j in range(1,len(cols)):
            for i in range(len(df)):
                df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))

        print(df)
        new_file = df.to_excel(f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/data.xlsx", index=False)


        # if len(df.columns) <= 5:
        #     return html.Div([
        #                      html.Div(dcc.Markdown(children=markdown_text_colontitul),
        #                               style={'text-align': 'right', 'margin-bottom': '20px'}),
        #                      html.Div(html.H1(children='Описательный анализ'),
        #                               style={'text-align': 'center'}),
        #                      html.Div(dcc.Markdown(children=markdown_text_begin),
        #                               style={'text-align': 'left', 'margin-bottom': '50px'}),
        #                      html.Div(html.H1(children='Описательная таблица'),
        #                               style={'text-align': 'left'}),
        #                      html.Div([
        #                          html.Div([
        #                              html.Div([dash_table.DataTable(
        #                                  id='table',
        #                                  columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
        #                                  data=df.to_dict('records'),
        #                                  style_table={'overflowX': 'auto'},
        #                                  export_format='xlsx'
        #                              )], style={'border-color': 'rgb(220, 220, 220)',
        #                                         'border-style': 'solid', 'padding': '5px', 'margin': '20px'})],
        #                              style={'width': len_t, 'display': 'inline-block'}),
        #                          html.Div(dcc.Markdown(children=markdown_text_table),
        #                                   style={'width': len_text, 'float': 'right', 'display': 'inline-block'})
        #                      ])
        #                      ], style={'margin': '50px'}
        #                     )
        # else:
        #     return html.Div([
        #                      html.Div(dcc.Markdown(children=markdown_text_colontitul),
        #                               style={'text-align': 'right', 'margin-bottom': '20px','font-size': '20px','color': '#e8093d','font-weight': 'bold'}),
        #                      html.Div(html.H1(children='Описательный анализ'),
        #                               style={'text-align':'center'}),
        #                      html.Div(dcc.Markdown(children=markdown_text_begin),
        #                               style={'text-align': 'left', 'margin-bottom': '50px'}),
        #                      html.Div(html.H1(children='Описательная таблица'),
        #                               style={'text-align': 'left'}),
        #
        #             html.Div([dash_table.DataTable(
        #                 id='table',
        #                 columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],