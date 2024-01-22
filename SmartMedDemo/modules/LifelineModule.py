from .ModuleInterface import Module
from .dash import LifelineDashboard

from .dataprep import PandasPreprocessor


class LifelineModule(Module, LifelineDashboard):

    def _prepare_data(self):
        self.pp = PandasPreprocessor(self.settings['data'])
        self.pp.fillna()
        return self.pp.df

    def _prepare_dashboard_settings(self):
        settings = dict()
        settings['model'] = self.settings['model']
        settings['methods'] = []
        if settings['model'] == 1:
            for metric in self.settings['method'].keys():
                if self.settings['method'][metric]:
                    settings['methods'].append(metric)
            print(settings['methods'])
            self.criteria_to_method = {
                '0': self._generate_table,
                '1': self._generate_curve,
                '2':  self._generate_median,
                '3': self._generate_interval,
                '4': self._generate_interval2
            }
        else:
            for graph in self.settings['criteria'].keys():
                if self.settings['criteria'][graph]:
                    settings['methods'].append(graph)
            self.criteria_to_method = {
                '3': self._generate_curve2,
                '1': self._generate_c2,
                '2': self._generate_table2,
                '0': self._generate_c1,
                '4': self._generate_median2
            }
        settings['data'] = self.data

        return settings

    def _prepare_dashboard(self):
        pass
