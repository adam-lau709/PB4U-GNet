import os
import socket

from munch import munchify

hostname = socket.gethostname()

PB4U_PROJECT = os.environ["PB4U_PROJECT"]
PB4U_DATA = os.environ["PB4U_DATA"]

DEFAULTS = dict()

DEFAULTS['server'] = 'local'
DEFAULTS['data_root'] = PB4U_DATA
DEFAULTS['experiment_root'] = os.path.join(PB4U_DATA, 'experiments')
DEFAULTS['vto_root'] = os.path.join(PB4U_DATA, 'vto_dataset')
DEFAULTS['aux_data'] = os.path.join(PB4U_DATA, 'aux_data')
DEFAULTS['project_dir'] = PB4U_PROJECT


DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
