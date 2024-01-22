from .ModuleInterface import Module
from .dash import ComparativeDashboard

from .dataprep import PandasPreprocessor


class ComparativeModule(Module, ComparativeDashboard):

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

    def _prepare_dashboard_settings(self):
        settings = dict()

        settings['methods'] = []
        for method in self.settings['methods'].keys():
            if self.settings['methods'][method]:
                settings['methods'].append(method)

        self.graph_to_method = {
            'kolmagorova_smirnova': self._generate_test_kolmagorova_smirnova,
            'student_independent': self._generate_t_criterion_student_independent,
            'student_dependent': self._generate_t_criterion_student_dependent,
            'mann_whitney': self._generate_u_criterion_mann_whitney,
            'wilcoxon': self._generate_t_criterion_wilcoxon,
            'pearson': self._generate_chi2_pearson,
            'se_sp': self._generate_sensitivity_specificity,
            'odds_ratio': self._generate_odds_relations,
            'risk_ratio': self._generate_risk_relations
        }
        
        settings['data'] = self.data
        settings['path'] = self.settings['path']
        if self.settings['type'] == 'continuous':
            settings['type'] = 'Непрерывные переменные'
        else:
            settings['type'] = 'Категориальные переменные'

        return settings

    def _prepare_dashboard(self):
        pass
