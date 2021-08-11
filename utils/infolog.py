# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:26:55 2019

@author: USER
"""

import atexit
from datetime import datetime
from urllib.request import Request, urlopen

_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None

def init(filename):
    global _file, _run_name
    _close_logfile()
    _file = open(filename, 'a')
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting..........\n')
    _file.write('-----------------------------------------------------------------\n')

def log(msg, slack=False):
    print(msg)
    if _file is not None:
        _file.write('[%s]    %s\n' % (datetime.now().strftime(_format)[:-3], msg))

def _close_logfile():
    global _file
    if _file is not None:
        _file.close()
        _file = None

atexit.register(_close_logfile)