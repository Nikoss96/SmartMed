import numpy as np
import pandas as pd
import sklearn.model_selection as sm
import sklearn.preprocessing as sp
from sklearn.preprocessing import KBinsDiscretizer

from .ModelManipulator import ModelManipulator
from .ModuleInterface import Module
from .dash import PredictionDashboard
from .dataprep import PandasPreprocessor
from .dataprep.PandasPreprocessor import read_file


class PredictionModule(Module, PredictionDashboard):

    def _prepare_data(self):
        prep = {'fillna': self.settings['preprocessing'],
                'encoding': 'label_encoding',
                'scaling': False}
        dict_pp = {
            'preprocessing': prep,
            'path': self.settings['path'],
            'fillna': self.settings['preprocessing']
        }

        self.pp = PandasPreprocessor(dict_pp)
        self.pp.preprocess()

        return self.pp.df

    def _init_settings(self, model_type: str) -> dict:
        y = self.pp.df.columns[0]

        self.settings['variable'] = y
        if model_type != 'linreg' and model_type != 'logreg':
            return {}
    #if self.settings['bin'] is not None:


        dfX_train, dfX_test, dfY_train, dfY_test = sm.train_test_split(self.df_X, self.df_Y, test_size=0.3,
                                                                       random_state=42)

        self.df_X_train = dfX_train
        self.df_X_test = dfX_test
        self.df_Y_train = dfY_train
        self.df_Y_test = dfY_test
        self.model = ModelManipulator(
            x=self.df_X_train, y=self.df_Y_train, model_type=self.settings['model']).create()
        self.model.fit()
        self.mean = sum(dfY_test) / len(dfY_test)

        settings = dict()

        # prepare metrics as names list from str -> bool
        settings['path'] = []
        settings['preprocessing'] = []
        settings['model'] = []
        settings['metrics'] = []
        settings['y'] = []
        settings['x'] = self.pp.df.columns.tolist() + [i for i in range(self.settings['bin'])]

        for metric in self.settings.keys():
            if metric == 'model':
                settings['model'] = self.settings['model']
            elif metric == 'path':
                settings['path'] = self.settings['path']
            elif metric == 'preprocessing':
                settings['preprocessing'] = self.settings['preprocessing']
            elif metric == 'variable':
                settings['y'] = self.settings['variable']
                settings['x'].remove(self.settings['variable'])
            elif self.settings[metric]:
                settings['metrics'].append(metric)

        prep = {'fillna': self.settings['preprocessing'],
                'encoding': 'label_encoding',
                'scaling': False}
        dict_pp = {
            'preprocessing': prep,
            'path': self.settings['path'],
            'fillna': self.settings['preprocessing']
        }
        settings['data'] = dict_pp
        return settings

    def _prepare_dashboard_settings(self):
        y = self.pp.df.columns[0]
        self.settings['variable'] = y
        names = self.pp.df.columns.tolist()
        # this setting is totally artificial. It should be selected by user from the menu
        self.settings['bin'] = 3  # todo: remove it
        names.remove(self.settings['variable'])
        self.df_X = pd.DataFrame()
        for name in names:
            self.df_X = pd.concat([self.df_X, self.pp.df[name]], axis=1)
        self.df_X = self.pp.get_numeric_df(self.df_X)

        # похоже что работает и без этого, но на всякий случай оставлю
        df_cat = self.pp.get_categorical_df(self.df_X)
        settings = dict()
        names_cat = df_cat.columns.tolist()
        if len(names_cat) > 0:
            df_dum = pd.get_dummies(df_cat, prefix=[names_cat])
            self.df_X = pd.concat([self.df_X, df_dum], axis=1)
        if self.settings['model'] == 'linreg':
            self.df_Y = self.pp.df[self.settings['variable']]
            settings = self._init_settings('linreg')
        elif self.settings['model'] == 'logreg':
            numerics_list = {'int16', 'int32', 'int', 'float', 'bool',
                             'int64', 'float16', 'float32', 'float64'}

            # self.df_Y = self.pp.df[self.settings['variable']]
            df_Y = self.pp.df[self.settings['variable']]
            #print('first', type(df_Y), df_Y.dtype, df_Y.nunique())
            #print(df_Y)
            if df_Y.nunique() == 2:
                #print('12')
                self.df_Y = df_Y
            else:
                if df_Y.dtype not in numerics_list:
                    #print('23')
                    labelencoder = sp.LabelEncoder()
                    df_Y = labelencoder.fit_transform(df_Y)
                mean_Y = df_Y.mean()
                df_Y1 = df_Y
                #print('type', type(df_Y1))
                for i in range(len(df_Y)):
                    if df_Y[i] < mean_Y:
                        df_Y1[i] = 0
                    else:
                        df_Y1[i] = 1
                self.df_Y = pd.Series(df_Y1)
                #print('second', type(self.df_Y), self.df_Y.dtype, self.df_Y.nunique())
                #print(self.df_Y)
            settings = self._init_settings('logreg')
        elif self.settings['model'] == 'roc':
            numerics_list = {'int16', 'int32', 'int', 'float', 'bool',
                             'int64', 'float16', 'float32', 'float64'}
            df_Y = self.pp.df[self.settings['variable']]
            # print('first', type(df_Y), df_Y.dtype, df_Y.nunique())
            # print(df_Y)
            if df_Y.nunique() == 2:
                # print('12')
                self.df_Y = df_Y
            else:
                if df_Y.dtype not in numerics_list:
                    # print('23')
                    labelencoder = sp.LabelEncoder()
                    df_Y = labelencoder.fit_transform(df_Y)
                mean_Y = df_Y.mean()
                df_Y1 = df_Y
                # print('type', type(df_Y1))
                for i in range(len(df_Y)):
                    if df_Y[i] < mean_Y:
                        df_Y1[i] = 0
                    else:
                        df_Y1[i] = 1
                self.df_Y = pd.Series(df_Y1)
                # print('second', type(self.df_Y), self.df_Y.dtype, self.df_Y.nunique())
                # print(self.df_Y)

            # prepare metrics as names list from str -> bool
            settings['path'] = []
            settings['preprocessing'] = []
            settings['model'] = []
            settings['metrics'] = []
            settings['graphs'] = []
            settings['spec_and_sens'] = []
            settings['spec_and_sens_table'] = []
            settings['y'] = []
            settings['x'] = self.pp.df.columns.tolist()

            for metric in self.settings.keys():
                if metric == 'model':
                    settings['model'] = self.settings['model']
                elif metric == 'path':
                    settings['path'] = self.settings['path']
                elif metric == 'preprocessing':
                    settings['preprocessing'] = self.settings['preprocessing']
                elif metric == 'variable':
                    settings['y'] = self.settings['variable']
                    settings['x'].remove(self.settings['variable'])
                elif metric == 'auc' or metric == 'diff_graphics' or metric == 'paint':
                    settings['graphs'].append(metric)
                # elif metric == 'spec_and_sens':
                #    settings['spec_and_sens'] = self.settings['spec_and_sens']
                # elif metric == 'spec_and_sens_table':
                #    settings['spec_and_sens_table'] = self.settings[
                #        'spec_and_sens_table']
                elif self.settings[metric]:
                    settings['metrics'].append(metric)

            prep = {'fillna': self.settings['preprocessing'],
                    'encoding': 'label_encoding',
                    'scaling': False}
            dict_pp = {
                'preprocessing': prep,
                'path': self.settings['path'],
                'fillna': self.settings['preprocessing']
            }
            settings['data'] = dict_pp

        elif self.settings['model'] == 'polynomreg':
            count = len(self.df_X.columns)
            for i in range(count):
                for j in range(i, count):
                    data_list_1 = np.array(self.df_X.iloc[:, [i]])
                    data_list_2 = np.array(self.df_X.iloc[:, [j]])
                    data_list = data_list_1 * data_list_2
                    print(data_list)
                    if i == j:
                        data_name = str(self.df_X.columns[i] + '^2')
                    else:
                        data_name = str(self.df_X.columns[i] + ' * ' + str(self.df_X.columns[j]))
                    print(data_name)
                    self.df_X.insert(len(self.df_X.columns), data_name, data_list, True)
            print(self.df_X)
            self.df_Y = self.pp.df[self.settings['variable']]
            numerics_list = {'int16', 'int32', 'int', 'float', 'bool',
                             'int64', 'float16', 'float32', 'float64'}

            if self.df_Y.dtype not in numerics_list:
                labelencoder = sp.LabelEncoder()
                self.df_Y = labelencoder.fit_transform(self.df_Y)

            dfX_train, dfX_test, dfY_train, dfY_test = sm.train_test_split(self.df_X, self.df_Y, test_size=0.3,
                                                                           random_state=42)
            self.df_X_train = dfX_train
            self.df_X_test = dfX_test
            self.df_Y_train = dfY_train
            self.df_Y_test = dfY_test
            self.model = ModelManipulator(
                x=self.df_X_train, y=self.df_Y_train, model_type='polyreg').create()
            self.model.fit()
            self.mean = sum(dfY_test) / len(dfY_test)

            # prepare metrics as names list from str -> bool
            settings['path'] = []
            settings['preprocessing'] = []
            settings['model'] = []
            settings['metrics'] = []
            settings['y'] = []
            settings['x'] = self.df_X.columns.tolist()

            for metric in self.settings.keys():
                if metric == 'model':
                    settings['model'] = self.settings['model']
                elif metric == 'path':
                    settings['path'] = self.settings['path']
                elif metric == 'preprocessing':
                    settings['preprocessing'] = self.settings['preprocessing']
                elif metric == 'variable':
                    settings['y'] = self.settings['variable']
                    # settings['x'].remove(self.settings['variable'])
                elif self.settings[metric]:
                    settings['metrics'].append(metric)

            prep = {'fillna': self.settings['preprocessing'],
                    'encoding': 'label_encoding',
                    'scaling': False}
            dict_pp = {
                'preprocessing': prep,
                'path': self.settings['path'],
                'fillna': self.settings['preprocessing']
            }
            settings['data'] = dict_pp

        elif self.settings['model'] == 'tree':
            self.df_Y = self.pp.df[self.settings['variable']]

            numerics_list = {'int16', 'int32', 'int', 'float', 'bool',
                             'int64', 'float16', 'float32', 'float64'}

            if self.df_Y.dtype not in numerics_list:
                labelencoder = sp.LabelEncoder()
                self.df_Y = labelencoder.fit_transform(self.df_Y)

            init_df = read_file(self.settings['path'])
            init_unique_values = np.unique(init_df[self.settings['variable']])
            number_class = []
            for name in init_unique_values:
                number_class.append(self.df_Y[list(init_df[self.settings['variable']]).index(name)])
            dict_classes = dict(zip(number_class, init_unique_values))

            dfX_train, dfX_test, dfY_train, dfY_test = sm.train_test_split(self.df_X, self.df_Y, test_size=0.3,
                                                                           random_state=42)
            self.df_X_train = dfX_train
            self.df_X_test = dfX_test
            self.df_Y_train = dfY_train
            self.df_Y_test = dfY_test

            if self.settings['tree_depth'] == '':
                self.settings['tree_depth'] = None
            if self.settings['samples'] == '':
                self.settings['samples'] = 2
            if self.settings['features_count'] == '':
                self.settings['features_count'] = None
            extra_param = np.array([self.settings['tree_depth'], self.settings['samples'],
                                    self.settings['features_count']])
            self.model = ModelManipulator(
                x=self.df_X_train, y=self.df_Y_train, model_type='tree', extra_param=extra_param).create()
            self.model.fit()
            self.mean = sum(dfY_test) / len(dfY_test)

            settings['path'] = []
            settings['preprocessing'] = []
            settings['model'] = []
            settings['metrics'] = []
            settings['features'] = []
            settings['y'] = []
            settings['x'] = self.pp.df.columns.tolist()
            settings['classes'] = dict_classes

            for metric in self.settings.keys():
                if metric == 'model':
                    settings['model'] = self.settings['model']
                elif metric == 'path':
                    settings['path'] = self.settings['path']
                elif metric == 'preprocessing':
                    settings['preprocessing'] = self.settings['preprocessing']
                elif metric == 'variable':
                    settings['y'] = self.settings['variable']
                    settings['x'].remove(self.settings['variable'])
                elif metric == 'tree' or metric == 'table' or metric == 'indicators' or metric == 'distributions' or \
                        metric == 'prediction':
                    settings['metrics'].append(metric)
                elif self.settings[metric]:
                    settings['features'].append(metric)

            prep = {'fillna': self.settings['preprocessing'],
                    'encoding': 'label_encoding',
                    'scaling': False}
            dict_pp = {
                'preprocessing': prep,
                'path': self.settings['path'],
                'fillna': self.settings['preprocessing']
            }
            settings['data'] = dict_pp
        return settings

    def _prepare_dashboard(self):
        pass
