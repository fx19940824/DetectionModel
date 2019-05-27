import os
import re
from argparse import Namespace
digit = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')


def parse_cfg(cfg):
    bools = ['True', 'False']
    cfgstr = cfg
    if os.path.isfile(cfgstr):
        cfgstr = open(cfgstr).read()
    items = cfgstr.split('\n')
    options = {}
    for item in items:
        if '=' not in item or item.strip().startswith('#'):
            continue
        key, val = item.replace(' ', '').split('#')[0].split('=')

        if ',' in val:
            val = val.split(',')
            if digit.match(val[0]):
                options[key] = list(map(lambda x: int(x) if str.isnumeric(x) else float(x), val))
            else:
                options[key] = val
        elif str.isnumeric(val):
            options[key] = int(val)
        elif digit.match(val):
            options[key] = float(val)
        elif val in bools:
            options[key] = 'True' == val
        else:
            options[key] = val
    return options