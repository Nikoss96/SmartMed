from .ModuleInterface import Module
from .dash import ClusterDashboard

from .dataprep import PandasPreprocessor


class ClusterModule(Module, ClusterDashboard):

    def _prepare_data(self):
        self.pp = PandasPreprocessor(self.settings['data'])
        self.pp.fillna()
        return self.pp.df

    def _prepare_dashboard_settings(self):
        settings = dict()

        # prepare metrics as names list from str -> bool
        settings['metric']  = self.settings['metric']
        # prepare graphs as names list from str -> bool
        settings['method'] = self.settings['method']
        self.graph_to_method = {
            0: self._generate_kmeans,
            1: self._generate_hirarchy,
            2: self._generate_component
        }

        settings['data'] = self.data

        return settings

    def _prepare_dashboard(self):
        pass
