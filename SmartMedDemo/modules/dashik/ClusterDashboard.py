import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import variation
from .Dashboard import Dashboard
from dash.dependencies import Input, Output

from scipy.cluster.hierarchy import dendrogram, linkage
from  sklearn.cluster import AgglomerativeClustering
import networkx as nx
from .text.cluster_text import *

from .text.markdown_stats import *
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

class ClusterDashboard(Dashboard):
    def _generate_layout(self):
        met_list = []
        metrics_method = {
            0: self._generate_kmeans(),
            1: self._generate_hirarchy(),
            2: self._generate_component()
        }
        for metric in metrics_method:
            if metric == self.settings['method']:
                met_list.append(metrics_method[metric])

        return html.Div([
            html.Div(html.H1(children='Кластерный анализ'), style={'text-align': 'center'}),
            dcc.Markdown(children=markdown_clus, style={'text-align': 'center', 'padding': '20px'}),
            html.Div(met_list)])
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
            ch = (sum((val1 - val2) ** p for val1, val2 in zip(a, b))) ** (1 / r)
            return ch

        def k_znach(n, met):
            try:
                    n = int(n)
                    metric = int(met)
                    mini = 10000000
                    clusters = []
                    final = []
                    begin = []
                    for i in range(len(x)*n):
                        initial_centers = kmeans_plusplus_initializer(x, n).initialize()
                        if metric == 0:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.EUCLIDEAN))
                        elif metric == 1:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.EUCLIDEAN_SQUARE))
                        elif metric == 2:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.MANHATTAN))
                        elif metric == 3:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.CHEBYSHEV))
                        elif metric == 4:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.USER_DEFINED, func=user1))
                        elif metric == 5:
                            kmeans_instance = kmeans(x, initial_centers,
                                                     metric=distance_metric(type_metric.USER_DEFINED, func=user2))

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
                            l += str(round(final[i][j], 2)) +" "
                        fin2.append(l)

                    beg2 = []
                    for i in range(len(begin)):
                        l = []
                        for j in range(len(begin[i])):
                            l += str(round(begin[i][j], 2)) +" "
                        beg2.append(l)

                    d = {"Кластер": pd.Series(clusters2, index=[i for i in range(n)]),
                         "Число элементов в кластере": pd.Series(num, index=[i for i in range(n)]),
                         "Начальные центры кластера": pd.Series(beg2, index=[i for i in range(n)]),
                         "Конечные центры кластера": pd.Series(fin2, index=[i for i in range(n)])}
                    df1 = pd.DataFrame(d)
                    print(df1)
                    return df1
            except ValueError:
                print("")

        metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'cosine', 'power']
        def update_output_div(n, met):
            for i in range(len(metrics)):
                if metrics[i] == met:
                    met = i
            print("я внутри")
            print(n, met)
            df1 = k_znach(n, met)
            print("я снаружи")
            clust = df1[['Кластер', 'Число элементов в кластере']]
            cent = df1[['Начальные центры кластера', 'Конечные центры кластера']]
            return html.Div([

                                 html.Div(html.H2(children='Исходные и конечные центры кластеров'),
                                          style={'text-align': 'center'}),
                                 dcc.Markdown(children=markdown_centers, style={'padding': '20px'}),
                                 html.Div([html.Div(dash_table.DataTable(
                                     columns=[{"name": i, "id": i}
                                              for i in cent.columns],
                                     data=cent.to_dict('records'),
                                     export_format='xlsx',
                                     style_cell={'textAlign': 'center'}),
            style = {'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                     'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
            ]),
                html.Div(html.H2(children='Принадлежность элементов кластерам'),
                         style={'text-align': 'center'}),
                dcc.Markdown(children=markdown_clusers, style={'padding': '20px'}),
                html.Div([html.Div(dash_table.DataTable(
                    columns=[{"name": i, "id": i}
                             for i in clust.columns],
                    data=clust.to_dict('records'),
                    export_format='xlsx',
                    style_cell={'textAlign': 'center'}),
                    style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                           'text-align': 'center', 'display': 'inline-block', 'width': '100%'})])
            ])
        self.app.callback(
                Output('tab1', 'children'),
                [Input('n', 'value'), Input('met', 'value')])(update_output_div)


        return html.Div([html.Div([html.Div(html.H1(children='Кластеризация k-средними'),
                                      style={'text-align': 'center'}),
                                   dcc.Markdown(children=markdown_kmeans),
                                   "Введите количество кластеров: ",
                                   dcc.Input(id='n', value=1, type='text'),
                                   dcc.Markdown(children='\n'),
                         "Выберите метрику расстояния: ",
                                       dcc.Dropdown(
                                           id='met',
                                           options=[{'label': i, 'value': i}
                                                    for i in metrics],
                                           value=metrics[self.settings['metric']]

                                       )]),
        dcc.Markdown(children=markdown_choice_k, style={'padding': '20px'}),
        html.Div(id='tab1')], style={'margin': '50px', 'text-align': 'center'})


    def _generate_hirarchy(self):
        df = self.pp.get_numeric_df(self.settings['data'])
        df = (df - df.mean()) / df.std()
        x = list(df.values.tolist())
        def update_output_div(n, met, meth):
            n = int(n)



            model = AgglomerativeClustering(n_clusters=n, affinity=met, linkage=meth).fit(x)
            labs = model.labels_
            lst = []
            s = [i for i in range(n)]
            for i in s:
                l = []
                for j in range(len(labs)):
                    if labs[j] == i:
                        l.append(j)
                lst.append(l)
            clusters2 = []
            for i in lst:
                l = ""
                for j in range(len(i)):
                    l += str(i[j]) + " "
                clusters2.append(l)
            num = []
            for i in lst:
                num.append(len(i))
            d = {"Кластер": pd.Series(clusters2, index=[i for i in range(n)]),
                 "Число элементов в кластере": pd.Series(num, index=[i for i in range(n)])}
            df1 = pd.DataFrame(d)

            di = sorted(linkage(x, method=meth, metric=met, optimal_ordering=False)[:, 2])

            fig = ff.create_dendrogram(df, linkagefun=lambda ci: linkage(df, meth, metric=met),
                                       color_threshold=di[-n + 1])
            fig.update_layout(autosize=True, hovermode='closest')
            fig.update_xaxes(mirror=False, showgrid=True, showline=False, showticklabels=False)
            fig.update_yaxes(mirror=False, showgrid=True, showline=True)


            return html.Div(
                [html.Div(html.H2(children='Принадлежность элементов кластерам'), style={'text-align': 'center'}),
                 dcc.Markdown(children=markdown_met, style={'padding': '20px'}),
                 html.Div([html.Div(dash_table.DataTable(
                     columns=[{"name": i, "id": i}
                              for i in df1.columns],
                     data=df1.to_dict('records'),
                     export_format='xlsx',
                     style_cell={'textAlign': 'center'}),
                     style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                            'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
                 ])]), fig

        self.app.callback(
            [Output(component_id='my-output', component_property='children'),
             Output(component_id='fig', component_property='figure')],
            [Input('my-input', 'value'), Input('metric', 'value'), Input('method', 'value')])(update_output_div)
        metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'cosine']
        methods = ['complete', 'single', 'average',  'ward']
        return html.Div([
    html.Div([
        html.Div(html.H1(children='Иерархическая кластеризация'),
                 style={'text-align': 'center'}),
        dcc.Markdown(children=markdown_hi),
        "Введите количество кластеров: ",
        dcc.Markdown(children='\n'),
        dcc.Input(id='my-input', value=1, type='text'),
        dcc.Markdown(children=''),
        "Выберите метрику расстояния: ",
        dcc.Dropdown(
            id='metric',
            options=[{'label': i, 'value': i}
                     for i in metrics],
            value=metrics[self.settings['metric']]
        ),
        "Выберите меру расстояния между кластерами: ",
        dcc.Dropdown(
            id='method',
            options=[{'label': i, 'value': i}
                     for i in methods],
            value=methods[0]
        ),
        dcc.Markdown(children=markdown_ch, style={'padding': '20px'}),
    ]),
    html.Div([
        html.Div(html.H2(children='Дендрограмма'), style={'text-align': 'center'}),
        dcc.Markdown(children=markdown_di, style={'padding': '20px'}),
        html.Div([dcc.Graph(id='fig')],
        style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
                                    'border-style': 'solid', 'padding': '5px'}),
        html.Div(id='my-output', style={'margin': '50px', 'text-align': 'center'})

        ])

], style={'margin': '50px', 'text-align': 'center'})


    def _generate_component(self):
        df = self.pp.get_numeric_df(self.settings['data'])
        df = (df - df.mean()) / df.std()
        x = list(df.values.tolist())

        def user4(a, b):
            ch = sum((val1 * val2) for val1, val2 in zip(a, b))
            z1 = sum((val1 ** 2) for val1, val2 in zip(a, b))
            z2 = sum((val2 ** 2) for val1, val2 in zip(a, b))
            return 1 - ch / (z1 * z2)

        def user0(a, b):
            ch = (sum((val1 - val2) ** 2 for val1, val2 in zip(a, b))) ** (1 / 2)
            return ch

        def user1(a, b):
            ch = (sum((val1 - val2) ** 2 for val1, val2 in zip(a, b)))
            return ch

        def user2(a, b):
            ch = (sum(abs(val1 - val2) for val1, val2 in zip(a, b)))
            return ch

        def user3(a, b):
            ch = max(abs(val1 - val2) for val1, val2 in zip(a, b))
            return ch

        g = nx.Graph()
        metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'cosine']
        def update_output_div(m, metr):
            for i in range(len(metrics)):
                if metrics[i] == metr:
                    metr = i
            g = nx.Graph()
            x = list(df.values.tolist())
            n = float(m)
            metric = int(metr)
            ras = []
            for i in range(len(x)):
                g.add_node(i)
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j:
                        lst = []
                        lst.append(i)
                        lst.append(j)
                        b = 0
                        if metric == 0:
                            b = user0(x[i], x[j])
                        elif metric == 1:
                            b = user1(x[i], x[j])
                        elif metric == 2:
                            b = user2(x[i], x[j])
                        elif metric == 3:
                            b = user3(x[i], x[j])
                        elif metric == 4:
                            b = user4(x[i], x[j])
                        print(n, metric, b)
                        if b < n:
                            lst.append(b)
                            ras.append(lst)

            g.add_weighted_edges_from(ras)
            pos = nx.spring_layout(g)
            edge_x = []
            edge_y = []

            nx.draw(g, pos)
            for i in ras:
                x0 = pos[i[0]][0]
                y0 = pos[i[0]][1]
                x1 = pos[i[1]][0]
                y1 = pos[i[1]][1]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []

            for i in pos.values():
                x = i[0]
                y = i[1]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color='Purple',
                    size=10
                ),
                line_width=2)
            node_text = []
            for node, adjacencies in enumerate(g.adjacency()):
                node_text.append(node)
            node_trace.text = node_text
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='',
                                titlefont_size=16,
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            fig2 = px.histogram(pd.DataFrame(ras), x=2)
            lst = []
            ls = [tuple(g._adj[i].keys()) for i in g._adj]
            ls = [list(i) for i in ls]
            b = 0
            for i in ls:
                i.append(b)
                b += 1
                i = sorted(i)
                if i == [20, 55, 83, 87]:
                    print(90)
                if len(i) != 1:
                    if lst:
                        f = 0
                        t = []
                        for j in range(len(lst)):
                            if j - 1 < len(lst):
                                if list(set(lst[j - 1]) & set(i) or set(lst[j - 1]) & set(t)):
                                    if t == []:
                                        t = list(set(lst[j - 1] + i))
                                    else:
                                        t = t + list(set(lst[j - 1] + i))
                                    lst.remove(lst[j - 1])
                                    f = 1

                            else:
                                print(i)

                        if f:
                            lst.append(t)
                        else:
                            lst.append(i)
                    else:
                        lst.append(i)
                else:
                    lst.append(i)
            p = []
            for k in lst:
                for d in lst:
                    if k != d:
                        if list(set(k) & set(d)):
                            lst.remove(k)
                            lst.remove(d)
                            lst.append(list(set(k + d)))

            clusters2 = []
            for i in lst:
                l = ""
                for j in range(len(i)):
                    l += str(i[j]) + " "
                clusters2.append(l)
            num = []
            for i in lst:
                num.append(len(i))
            d = {"Кластер": pd.Series(clusters2, index=[i for i in range(len(lst))]),
                 "Число элементов в кластере": pd.Series(num, index=[i for i in range(len(lst))])}
            df1 = pd.DataFrame(d)
            return html.Div(
                [html.Div(html.H2(children='Принадлежность элементов кластерам'), style={'text-align': 'center'}),
                 dcc.Markdown(children=markdown_cla, style={'padding': '20px'}),
                 html.Div([html.Div(dash_table.DataTable(
                     columns=[{"name": i, "id": i}
                              for i in df1.columns],
                     data=df1.to_dict('records'),
                     style_cell={'textAlign': 'center'}),
                     style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                            'text-align': 'center', 'display': 'inline-block', 'width': '100%'})
                 ])]), fig, fig2
        self.app.callback([Output('my-1', 'children'),
                           Output('fig1', 'figure'),
                           Output('fig2', 'figure')],
                          [Input('m', 'value'),
                           Input('metr', 'value')])(update_output_div)

        return html.Div([html.Div(html.H1(children='Кластеризация выделением связных компонент'),
                 style={'text-align': 'center'}),
        dcc.Markdown(children=markdown_co),
            html.Div([
                "Введите значение для расстояния: ",
                dcc.Input(id='m', value=11, type='text'),
                dcc.Markdown(children='\n'),
                "Выберите метрику расстояния: ",
                dcc.Dropdown(
                    id='metr',
                    options=[{'label': i, 'value': i}
                             for i in metrics],
                    value=metrics[self.settings['metric']]
                ),
                dcc.Markdown(children=markdown_r, style={'padding': '20px'}),
            ]),
            html.Div([

                html.Div(html.H2(children='Гистограмма попарных расстояний'),  style={'text-align': 'center'}),
                dcc.Markdown(children=markdown_gi, style={'padding': '20px'}),
                html.Div([dcc.Graph(id='fig2')],
                         style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
                                'border-style': 'solid', 'padding': '5px'}),

                html.Div(html.H2(children='Представление выборки в виде графа'), style={'text-align': 'center'}),
                dcc.Markdown(children=markdown_gra, style={'padding': '20px'}),
                html.Div([dcc.Graph(id='fig1')],
                         style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
                                'border-style': 'solid', 'padding': '5px'}),
                html.Div(id='my-1')

            ])

        ], style={'margin': '50px', 'text-align': 'center'})

