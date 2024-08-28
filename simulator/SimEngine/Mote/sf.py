from __future__ import absolute_import
# =========================== imports =========================================

from builtins import range
from builtins import object
import random
import sys
from abc import abstractmethod

import netaddr

import SimEngine
from . import MoteDefines as d
from . import sixp
from SimEngine.Mote.scheduling_functions import SFNone,MSF


# =========================== defines =========================================

# =========================== helpers =========================================

# =========================== body ============================================

class SchedulingFunction(object):
    def __new__(cls, mote):
        settings    = SimEngine.SimSettings.SimSettings()
        class_name  = u'SchedulingFunction{0}'.format(settings.sf_class)
        module_name = 'SimEngine.Mote.scheduling_functions.' + settings.sf_class
        return getattr(sys.modules[module_name], class_name)(mote)
