import webbrowser
from abc import ABC, abstractmethod

import socket

import dash



def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


class Dashboard(ABC):
    '''
    Dashboard Interface
    Each ConcreteDashboard inreases port number
    and Dashboar_i is opened on localhost with port = 8000 + i
    in daemon thread
    '''
    port = 8002

    def __init__(self):
        # include general styleshits and scripts
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        external_scripts = [
            'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

        # create Dash(Flask) server
        self.app = dash.Dash(
            server=True,
            external_stylesheets=external_stylesheets,
            external_scripts=external_scripts
        )

        # increase port
        # address already in use fix
        while is_port_in_use(Dashboard.port):
            Dashboard.port += 1

    
    @abstractmethod
    def _generate_layout(self):
        '''
        abstractmethod to generate dashboard layout
        '''
        raise NotImplementedError

    def start(self, debug=False):
        # generate layout
        self.app.layout = self._generate_layout()

        # set port
        port = Dashboard.port
        # open dashboard
        webbrowser.open(f"http://127.0.0.1:" + str(port) + "/dash1/")

        # run dashboard
        self.app.run_server(port=port, dev_tools_silence_routes_logging=True, debug=False)