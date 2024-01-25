import pandas as pd
import sys

sys.path.append("C:\\Users\\nikch\\SmartMedDemo\\modules\\dashik")
from ModuleInterface import Module
import BioequivalenceDashboard
from models import BioquivalenceMathsModel


class BioequivalenceModule(Module, BioequivalenceDashboard):
    def _prepare_data(self):
        print(self.settings)
        if self.settings["design"] == "parallel":
            df_test = pd.read_excel(self.settings["path_test"], index_col=0)
            df_ref = pd.read_excel(self.settings["path_ref"], index_col=0)
            data = {}
            data["concentration_t"] = df_test
            data["concentration_r"] = df_ref
            return data
        else:
            df_1 = pd.read_excel(self.settings["path_test"], index_col=0)
            df_2 = pd.read_excel(self.settings["path_ref"], index_col=0)
            df_1_1 = df_1[df_1["Group"] == "T"]
            df_1_1.drop(columns="Group", inplace=True)
            df_1_2 = df_1[df_1["Group"] == "R"]
            df_1_2.drop(columns="Group", inplace=True)
            df_2_1 = df_2[df_2["Group"] == "T"]
            df_2_1.drop(columns="Group", inplace=True)
            df_2_2 = df_2[df_2["Group"] == "R"]
            df_2_2.drop(columns="Group", inplace=True)
            data = {
                "concentration_t_1": df_1_1,
                "concentration_r_1": df_1_2,
                "concentration_t_2": df_2_1,
                "concentration_r_2": df_2_2,
            }
            return data

    def _prepare_dashboard_settings(self):
        dicts_for_graphs_and_tables = [self.settings["graphs"], self.settings["tables"]]
        mathdata = BioquivalenceMathsModel(self.settings, self.data)
        mathdata.run_bio_model()
        return mathdata, dicts_for_graphs_and_tables

    def _prepare_dashboard(self):
        graphs = self.settings[1][0]
        tables = self.settings[1][1]
        if self.settings[0].plan == "parallel":
            if graphs["each_in_group"] and graphs["log_each_in_group"]:
                graphs["each_in_group"] = False
                graphs["log_each_in_group"] = False
                graphs["linlog_each_in_group"] = True
            if graphs["all_in_group"] and graphs["log_all_in_group"]:
                graphs["all_in_group"] = False
                graphs["log_all_in_group"] = False
                graphs["linlog_all_in_group"] = True

        self.graphs_and_lists = []
        temp_list = []
        for graph, boo in graphs.items():
            if boo:
                temp_list.append(graph)
        for table, boo in tables.items():
            if boo:
                temp_list.append(table)
        if self.settings[0].plan == "parallel":
            graph_to_method = {
                "linlog_all_in_group": [
                    self._generate_concentration_time_linlog(True),
                    self._generate_concentration_time_linlog(False),
                ],
                "linlog_each_in_group": self._generate_concentration_time_linlog_mean(),
                "all_in_group": [
                    self._generate_concentration_time(True),
                    self._generate_concentration_time(False),
                ],
                "log_all_in_group": [
                    self._generate_concentration_time_log(True),
                    self._generate_concentration_time_log(False),
                ],
                "each_in_group": self._generate_concentration_time_mean(),
                "log_each_in_group": self._generate_concentration_time_log_mean(),
                "criteria": self._generate_criteria(),
                "features": self._generate_param(),
                "var": self._generate_anova(),
                "statistics": self._generate_statistics(),
            }
            for graph in temp_list:
                if type(graph_to_method[graph]) == list:
                    self.graphs_and_lists.append((graph_to_method[graph])[0])
                    self.graphs_and_lists.append((graph_to_method[graph])[1])
                else:
                    self.graphs_and_lists.append(graph_to_method[graph])
            self.graphs_and_lists.append(self._generate_interval())
        else:
            graph_to_method = {
                "avg_concet": [
                    self._generate_concentration_time_cross(True),
                    self._generate_concentration_time_cross(False),
                ],
                "indiv_concet": [
                    self._generate_group_mean(True),
                    self._generate_group_mean(False),
                ],
                "gen_concet": self._generate_drug_mean(),
                "avg_auc": [self._generate_log_auc(), self._generate_criteria()],
                "anal_resylts": self._generate_anova(),
                "results": self._generate_interval(),
                "statistics": self._generate_statistics(),
            }
            for graph in temp_list:
                if type(graph_to_method[graph]) == list:
                    for elem in graph_to_method[graph]:
                        self.graphs_and_lists.append(elem)
                else:
                    self.graphs_and_lists.append(graph_to_method[graph])


# Тестируемая функция
def main():
    return 0


# Инит вызова
if __name__ == "__main__":
    main()
