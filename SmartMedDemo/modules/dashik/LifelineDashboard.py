import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import variation
from .Dashboard import Dashboard
from dash.dependencies import Input, Output
from lifelines import KaplanMeierFitter
from datetime import datetime, date
from lifelines.utils import survival_table_from_events
import plotly.graph_objs as go
from .text.lifelines import *
import scipy.stats


def curve(self, dis, status, distance2, status2, number, about, inz):
    kmf = KaplanMeierFitter()
    kmf.fit(pd.Series(dis), pd.Series(status), alpha=(1-inz))
    strin1 = 'KM_estimate_upper_' + str(inz)
    strin2 = 'KM_estimate_lower_' + str(inz)
    fig = go.Figure()
    if number == 2:
        kmf2 = KaplanMeierFitter()
        kmf2.fit(pd.Series(distance2), pd.Series(status2), alpha=(1-inz))

    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'],
        line=dict(shape='hv', width=3, color='rgb(31, 119, 180)'),
        mode='lines',
        showlegend=False
    ))
    if number == 2:
        fig.add_trace(go.Scatter(
            x=kmf2.survival_function_.index, y=kmf2.survival_function_['KM_estimate'],
            line=dict(shape='hv', width=3, color='rgb(174, 34, 34)'),
            mode='lines',
            showlegend=False
        ))
    if about:
        fig.add_trace(go.Scatter(
            x=kmf.confidence_interval_.index,
            y=kmf.confidence_interval_[strin1],
            line=dict(shape='hv', width=0),
            mode='lines',
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=kmf.confidence_interval_.index,
            y=kmf.confidence_interval_[strin2],
            line=dict(shape='hv', width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.4)',
            mode='lines',
            showlegend=False
        ))
        if number == 2:
            fig.add_trace(go.Scatter(
                x=kmf2.confidence_interval_.index,
                y=kmf2.confidence_interval_[strin1],
                line=dict(shape='hv', width=0),
                mode='lines',
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=kmf2.confidence_interval_.index,
                y=kmf2.confidence_interval_[strin2],
                line=dict(shape='hv', width=0),
                fill='tonexty',
                fillcolor='rgba(174, 34, 34, 0.4)',
                mode='lines',
                showlegend=False
            ))
    fig.update_layout(
        yaxis_title="Выживаемость",
        xaxis_title="Продолжительность исследования",
        margin=dict(r=0, t=10, l=0),
        font_size=14,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18
    )
    return fig


def longlife_table(self, n, distance, status):
    n = n+1
    leng = max(distance) / (n - 1)
    inters = [0]
    for i in range(1, n):
        inters.append(round(inters[i - 1] + leng,3))
    kmf = KaplanMeierFitter()
    table = survival_table_from_events(distance, status, intervals=inters, collapse=False)
    learn = table['at_risk'] - table['censored'] / 2
    table['learning'] = learn
    table['died'] = round(table['observed'] / table['learning'], 3)
    table['alive'] = round(1 - table['died'], 3)
    alive = list(table['alive'])
    com = [1]

    for i in range(len(alive) - 1):
        com.append(round(alive[i] * com[i], 3))
    plot = []
    for i in range(len(com) - 1):
        plot.append(round((com[i] + com[i + 1]) / inters[1], 3))
    plot.append(0)
    table['Кум. доля выживших'] = com
    table['Плотность вероятности'] = plot
    table.insert(0, 'Начало интервала', inters[:-1])
    table.rename(columns={"at_risk": "Число в начале", "censored": "Число изъятых",
                          "learning": "Число изучаемых", "observed": "Число умерших",
                          "died": "Доля умерших", "alive": "Доля выживших",
                          "removed": "Число в конце"}, inplace=True)

    table[['Начало интервала', 'Число в начале', 'Число изучаемых',
           'Число изъятых', 'Число умерших', "Доля выживших", 'Доля умерших','Кум. доля выживших', 'Плотность вероятности']]

    return table
def criterias(self, type, df, name):
    df1 = df.groupby(by=["dis"])['observed'].sum()
    df1 = df1.to_frame()
    df2 = df.merge(df1, left_on='dis', right_index=True)
    df2 = df2[df2['observed_y'] != 0]

    group1 = df[df[name] == 0]
    group2 = df[df[name] == 1]
    T = group1['dis']
    E = group1['observed']
    T1 = group2['dis']
    E1 = group2['observed']
    print(group1)
    def longlife_table(distance, status):
        kmf = KaplanMeierFitter()
        table = survival_table_from_events(distance, status)
        return table

    tab = longlife_table(T, E)
    dro = list(((set(list((tab.index)))) - set(list((df2['dis'])))))
    ins = list(((set(list((df2['dis']))) - (set(list((tab.index)))))))
    tab = tab.drop(index=dro)
    tab2 = longlife_table(T1, E1)
    dro2 = list(((set(list((tab2.index)))) - set(list((df2['dis'])))))
    ins2 = list(((set(list((df2['dis']))) - (set(list((tab2.index)))))))
    tab2 = tab2.drop(index=dro2)
    t = sorted(list(set(df2['dis'])))
    d1 = tab['observed']
    for i in ins:
        d1.loc[i] = 0
    d1 = d1.to_frame()
    d1 = d1.sort_index()
    n1 = tab['at_risk']
    tabo = longlife_table(df['dis'], df['observed'])
    for i in ins:
        n1.loc[i] = int(tabo.loc[[i]]['at_risk'] - tab2.loc[[i]]['at_risk'])
    n1 = n1.to_frame()
    n1 = n1.sort_index()
    # второй
    d2 = tab2['observed']
    for i in ins2:
        d2.loc[i] = 0
    d2 = d2.to_frame()
    d2 = d2.sort_index()
    n2 = tab2['at_risk']

    for i in ins2:
        n2.loc[i] = int(tabo.loc[[i]]['at_risk'] - tab.loc[[i]]['at_risk'])
    n2 = n2.to_frame()
    n2 = n2.sort_index()
    d = d1 + d2
    n = n1 + n2
    e = []
    a = list(n1['at_risk'])
    b = list(n['at_risk'])
    c = list(d['observed'])
    f = list(n2['at_risk'])
    for i in range(len(list(n1['at_risk']))):
        e.append(a[i] * c[i] / b[i])
    dl = list(d1['observed'])
    ul = []
    for i in range(len(dl)):
        ul.append(dl[i] - e[i])

    summ = 0
    for i in range(len(dl)):
        summ += a[i] * f[i] * c[i] * (b[i] - c[i]) / (b[i] * b[i] * (b[i] - 1))
    znach = sum(ul)
    ztest1 = (znach) / (summ ** 0.5)
    print("!")

    z = 0
    zs = 0
    for i in range(len(ul)):
        z += ul[i] * (f[i] + a[i])
        zs += ((f[i] + a[i]) ** 2) * a[i] * f[i] * c[i] * (b[i] - c[i]) / (b[i] * b[i] * (b[i] - 1))
    ztest = z / (zs ** 0.5)
    if type == 0:
        return ztest1
    else:
        return ztest


def distan(self):
    dis = []
    df = self.settings['data']
    bdate = list(df.iloc[:, 0])
    edate = list(df.iloc[:, 1])
    if ':' in str(bdate[0]) and ':' in str(edate[0]):
        for i in range(len(bdate)):
            dis.append((edate[i] - bdate[i]).days)
    elif str(bdate[0]) == bdate[0]:
        lst = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август',
               'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        m1 = list(df['Месяц 1'])
        m2 = list(df['Месяц 2'])
        d1 = list(df['День 1'])
        d2 = list(df['День 2'])
        y1 = list(df['Год 1'])
        y2 = list(df['Год 2'])
        mo1 = []
        mo2 = []
        data1 = []
        data2 = []
        for i in m1:
            for j in range(len(lst)):
                if lst[j] == i:
                    mo1.append(j + 1)
        for i in m2:
            for j in range(len(lst)):
                if lst[j] == i:
                    mo2.append(j + 1)
        for i in range(len(y1)):
            data1.append(date(y1[i], mo1[i], d1[i]))
            data2.append(date(y2[i], mo2[i], d2[i]))
        dis = []
        for i in range(len(y1)):
            dis.append((data2[i] - data1[i]).days)
    else:
        dis = bdate
    print(dis)
    return dis


def observed(self):
    df = self.settings['data']
    pol = list(df['Полнота данных'])
    observed = []
    if 0 in pol:
        observed = pol
    else:
        for i in pol:
            if i == 'CENSORED':
                observed.append(0)
            else:
                observed.append(1)
    print(observed)
    return observed


class LifelineDashboard(Dashboard):
    def _generate_layout(self):
        if self.settings['model'] == 0:
            met_list = [self._comare()]
        else:
            met_list = []
        for graph in self.settings['methods']:
            met_list.append(self.criteria_to_method[graph]())
        return html.Div([
            html.Div(html.H1(children='Анализ Выживаемости'), style={'text-align': 'center'}),
            dcc.Markdown(children= lidea, style={'text-align': 'center', 'padding': '20px'}),
            html.Div(met_list)])

    def _comare(self):
        return html.Div([html.Div(html.H2(children='Сравнение выживаемости в двух выборках'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children=lcompare, style={'text-align': 'center', 'padding': '20px'})])

    def _generate_c1(self):
        df = self.settings['data']
        def update_output_div(colname):
            df = self.settings['data']
            df['dis'] = distan(self)
            df['observed'] = observed(self)
            df = df.sort_values(by='dis')
            stat = criterias(self, 0, df, colname)
            p = (scipy.stats.t.sf(abs(stat), df=1))
            d = {"p-value": pd.Series(p), "Значение критерия": pd.Series(stat)}
            df1 = pd.DataFrame(d)
            return html.Div([html.Div(dash_table.DataTable(
                columns=[{"name": i, "id": i}
                         for i in df1.columns],
                data=df1.to_dict('records'),
                export_format='xlsx',
                style_cell={'textAlign': 'center'}),
                style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                       'text-align': 'center', 'display': 'inline-block', 'width': '100%'}),
                html.Div(dcc.Markdown(lpvalue), style={'text-align': 'left', 'padding': '20px'})
            ])
        self.app.callback(
            Output(component_id='col', component_property='children'),
            [Input('colname', 'value')])(update_output_div)
        cols = []
        for i in df.columns:
            if len(df[i].unique()) == 2:
                cols.append(i)
        return html.Div([html.Div(html.H2(children='Логарифмический ранговый критерий'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children=llog, style={'text-align': 'center', 'padding': '20px'}),
                         html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную"),
                             dcc.Dropdown(
                                 id='colname',
                                 options=[{'label': i, 'value': i}
                                          for i in cols],
                                 value=cols[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                         html.Div(id='col')],
                        style={'margin': '50px', 'text-align': 'center'})

    def _generate_c2(self):
        df = self.settings['data']
        def update_output_div(colname1):
            df = self.settings['data']
            df['dis'] = distan(self)
            df['observed'] = observed(self)
            df = df.sort_values(by='dis')
            stat = criterias(self, 1, df, colname1)
            p = (scipy.stats.t.sf(abs(stat), df=1))
            d = {"p-value": pd.Series(p), "test": pd.Series(stat)}
            df1 = pd.DataFrame(d)
            return html.Div([html.Div(dash_table.DataTable(
                columns=[{"name": i, "id": i}
                         for i in df1.columns],
                data=df1.to_dict('records'),
                export_format='xlsx',
                style_cell={'textAlign': 'center'}),
                style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                       'text-align': 'center', 'display': 'inline-block', 'width': '100%'}),
                html.Div(dcc.Markdown(lpvalue), style={'text-align': 'left', 'padding': '20px'})
            ])
        self.app.callback(
            Output(component_id='col1', component_property='children'),
            [Input('colname1', 'value')])(update_output_div)
        cols = []
        for i in df.columns:
            if len(df[i].unique()) == 2:
                cols.append(i)
        return html.Div([html.Div(html.H2(children='Критерий Гехана-Вилкоксона'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children=lwil, style={'text-align': 'center', 'padding': '20px'}),
                         html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную"),
                             dcc.Dropdown(
                                 id='colname1',
                                 options=[{'label': i, 'value': i}
                                          for i in cols],
                                 value=cols[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                         html.Div(id='col1')],
                        style={'margin': '50px', 'text-align': 'center'})

    def _generate_table2(self):
        df = self.settings['data']

        def update_output_div(n, name1):
            n = int(n)
            df['dif'] = distan(self)
            df['observed'] = observed(self)
            group1 = df[df[name1] == 1]
            group2 = df[df[name1] == 0]
            T = group1['dif']
            E = group1['observed']
            T1 = group2['dif']
            E1 = group2['observed']
            df1 = longlife_table(self, n, T, E)
            df2 = longlife_table(self, n, T1, E1)
            return html.Div(
                [html.Div(html.H2(children='Таблица времен жизни 1 группы'), style={'text-align': 'center'}),
                 html.Div([html.Div(dash_table.DataTable(
                     columns=[{"name": i, "id": i}
                              for i in df1.columns],
                     data=df1.to_dict('records'),
                     export_format='xlsx',
                     style_cell={'textAlign': 'center'}),
                     style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                            'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
                 ]),
                 html.Div(html.H2(children='Таблица времен жизни 2 группы'), style={'text-align': 'center'}),
                 html.Div([html.Div(dash_table.DataTable(
                     columns=[{"name": i, "id": i}
                              for i in df2.columns],
                     data=df2.to_dict('records'),
                     export_format='xlsx',
                     style_cell={'textAlign': 'center'}),
                     style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                            'text-align': 'center', 'display': 'inline-block', 'width': '100%'}),
                             html.Div(dcc.Markdown(lifetable), style={'text-align': 'left','padding': '20px'})
                 ])])
        self.app.callback(
                Output('tab1', 'children'),
                [Input('n', 'value'), Input('name1', 'value')])(update_output_div)
        cols = []
        for i in df.columns:
            if len(df[i].unique()) == 2:
                cols.append(i)

        return html.Div([html.Div([html.Div(html.H1(children='Таблицы времен жизни'),
                                      style={'text-align': 'center'}),
                                   dcc.Markdown(children=ltab, style={'text-align': 'center', 'padding': '20px'}),
        "Введите количество интервалов: ",
                                   dcc.Markdown(children='\n'),
        dcc.Input(id='n', value=12, type='text')]),
                         html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную"),
                             dcc.Dropdown(
                                 id='name1',
                                 options=[{'label': i, 'value': i}
                                          for i in cols],
                                 value=cols[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div(id='tab1')], style={'margin': '50px', 'text-align': 'center'})


    def _generate_curve2(self):
        df = self.settings['data']

        def update_output_div(n, inz, name2):
            n = int(n)
            inz = float(inz)
            df['dif'] = distan(self)
            df['observed'] = observed(self)
            group1 = df[df[name2] == 1]
            group2 = df[df[name2] == 0]
            T = group1['dif']
            E = group1['observed']
            T1 = group2['dif']
            E1 = group2['observed']
            fig = curve(self, T, E, T1, E1, 2, n, inz)
            return fig

        self.app.callback(
            Output(component_id='fig', component_property='figure'),
            [Input('m', 'value'), Input('inz', 'value'), Input('name2', 'value')])(update_output_div)
        cols = []
        for i in df.columns:
            if len(df[i].unique()) == 2:
                cols.append(i)
        return html.Div([html.Div(html.H1(children='Оценки Каплана-Мейера'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children=lmarks, style={'text-align': 'center', 'padding': '20px'} ),

                         html.Div([
                             html.Div(html.H2(children='Кривые выживаемости'),
                                      style={'text-align': 'center'}),
                             dcc.Markdown(children=lcurve, style={'padding': '20px'}),
                             html.Div([
                                 "Введите 1 для включения доверительных интервалов: ",
                                 dcc.Markdown(children='\n'),
                                 dcc.Input(id='m', value=1, type='text'),
                                 dcc.Markdown(children='\n'),
                                 "Введите уровень значимости интервала: ",
                                 dcc.Markdown(children='\n'),
                                 dcc.Input(id='inz', value=0.05, type='text'),
                                 dcc.Markdown(children='\n'),
                                 html.Div([
                                     dcc.Markdown(children="Выберите группирующую переменную"),
                                     dcc.Dropdown(
                                         id='name2',
                                         options=[{'label': i, 'value': i}
                                                  for i in cols],
                                         value=cols[0]
                                     )
                                 ], style={'width': '48%', 'display': 'inline-block'}),
                             ]),
                             html.Div([dcc.Graph(id='fig')],
                                      style={'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)',
                                             'border-style': 'solid', 'padding': '5px'})])],
                        style={'margin': '50px', 'text-align': 'center'})

    def _generate_median(self):
        kmf = KaplanMeierFitter()
        ab = kmf.fit(pd.Series(distan(self)), pd.Series(observed(self)))
        med = ab._median
        d = {"Значение медианы": pd.Series(med)}
        df2 = pd.DataFrame(d)
        return html.Div([html.Div(html.H1(children='Медиана выживаемости'),
                           style={'text-align': 'center'}),
                         dcc.Markdown(children=lmedian, style={'text-align': 'center', 'padding': '20px'}),
                         html.Div([html.Div(dash_table.DataTable(
                             columns=[{"name": i, "id": i}
                                      for i in df2.columns],
                             data=df2.to_dict('records'),
                             export_format='xlsx',
                             style_cell={'textAlign': 'center'}),
                             style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                    'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
                         ])])


    def _generate_interval2(self):
        return html.Div()

    def _generate_table(self):
        def update_output_div(n):
            n = int(n)

            df1 = longlife_table(self, n, distan(self), observed(self))
            return html.Div(
                [html.Div(html.H2(children='Таблица времен жизни'), style={'text-align': 'center'}),
                 dcc.Markdown(children=ltab, style={'text-align': 'center', 'padding': '20px'}),
                 html.Div([html.Div(dash_table.DataTable(
                     columns=[{"name": i, "id": i}
                              for i in df1.columns],
                     data=df1.to_dict('records'),
                     export_format='xlsx',
                     style_cell={'textAlign': 'center'}),
                     style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                            'text-align': 'center', 'display': 'inline-block', 'width': '100%'}),
                             html.Div(dcc.Markdown(lifetable), style={'text-align': 'left', 'padding': '20px'})
                 ])])
        self.app.callback(
                Output('tab2', 'children'),
                [Input('h', 'value')])(update_output_div)
        return html.Div([html.Div([html.Div(html.H1(children='Таблицы времен жизни'),
                                      style={'text-align': 'center'}),
        "Введите количество интервалов: ",
        dcc.Input(id='h', value=12, type='text')]),
        html.Div(id='tab2')], style={'margin': '50px', 'text-align': 'center'})



    def _generate_curve(self):
        def update_output_div(n, inz2):
            n = int(n)
            inz2 = float(inz2)
            fig = curve(self, distan(self), observed(self), [], [], 1, n, inz2)
            return fig
        self.app.callback(
                    Output(component_id='fig1', component_property='figure'),
                    [Input('k', 'value'), Input('inz2', 'value')])(update_output_div)
        return html.Div([html.Div(html.H1(children='Оценки Каплана-Мейера'),
                                  style={'text-align': 'center'}),
                         dcc.Markdown(children=lmarks, style={'text-align': 'center', 'padding': '20px'}),

                         html.Div([
                             html.Div(html.H2(children='Кривая выживаемости'),
                                      style={'text-align': 'center'}),
                             dcc.Markdown(children=lcurve, style={'text-align': 'center', 'padding': '20px'}),
                             html.Div([
                                 "Введите 1 для включения доверительных интервалов: ",
                                 dcc.Input(id='k', value=1, type='text'),
                                 "Введите уровень значимости интервала: ",
                                 dcc.Input(id='inz2', value=0.95, type='text')]),
                             html.Div([dcc.Graph(id='fig1')],
                                      style={'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)',
                                             'border-style': 'solid', 'padding': '5px'})])],
                        style={'margin': '50px', 'text-align': 'center'})

    def _generate_median2(self):
        df = self.settings['data']
        def update_output_div(name12):
            df['dif'] = distan(self)
            df['observed'] = observed(self)
            group1 = df[df[name12] == 1]
            group2 = df[df[name12] == 0]
            T = group1['dif']
            E = group1['observed']
            T1 = group2['dif']
            E1 = group2['observed']
            kmf = KaplanMeierFitter()
            ab = kmf.fit(pd.Series(T), pd.Series(E))
            kmf2 = KaplanMeierFitter()
            cd = kmf2.fit(pd.Series(T1), pd.Series(E1))
            med1 = ab._median
            med2 = cd._median
            d = {"Значение медианы 1": pd.Series(med1), "Значение медианы 2": pd.Series(med2)}
            df2 = pd.DataFrame(d)
            return html.Div([html.Div(dash_table.DataTable(
                                 columns=[{"name": i, "id": i}
                                          for i in df2.columns],
                                 data=df2.to_dict('records'),
                                 export_format='xlsx',
                                 style_cell={'textAlign': 'center'}),
                                 style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                        'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
                             ])
        self.app.callback(
                    Output(component_id='name13', component_property='children'),
                    [Input('name12', 'value')])(update_output_div)
        cols = []
        for i in df.columns:
            if len(df[i].unique()) == 2:
                cols.append(i)
        return html.Div([html.Div(html.H1(children='Медиана выживаемости'),
                           style={'text-align': 'center'}),
                         html.Div([
                             dcc.Markdown(children="Выберите группирующую переменную"),
                             dcc.Dropdown(
                                 id='name12',
                                 options=[{'label': i, 'value': i}
                                          for i in cols],
                                 value=cols[0]
                             )
                         ], style={'width': '48%', 'display': 'inline-block'}),
                         dcc.Markdown(children=lmedian, style={'text-align': 'center', 'padding': '20px'}),
        html.Div(id='name13', style={'margin': '50px', 'text-align': 'center'})],
                        style={'margin': '50px', 'text-align': 'center'})






    def _generate_interval(self):
        return html.Div()
