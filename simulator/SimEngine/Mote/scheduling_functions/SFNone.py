from __future__ import absolute_import
# =========================== imports =========================================

import random
import sys
import netaddr
import SimEngine


from builtins import range
from builtins import object
from abc import abstractmethod

from .. import MoteDefines as d
from .. import sixp

from SimEngine.Mote.sfBase import SchedulingFunctionBase


class SchedulingFunctionSFNone(SchedulingFunctionBase):

    def __init__(self, mote):
        super(SchedulingFunctionSFNone, self).__init__(mote)

    def start(self):
        pass # do nothing

    def stop(self):
        pass # do nothing

    def indication_slotframe_window_ending(self, slotframe_period_size):
        pass # do nothing
    
    def indication_neighbor_added(self, neighbor_mac_addr):
        pass # do nothing

    def indication_tx_cell_elapsed(self, cell, sent_packet):
        pass # do nothing

    def indication_rx_cell_elapsed(self, cell, received_packet):
        pass # do nothing

    def indication_parent_change(self, old_parent, new_parent):
        pass # do nothing

    def detect_schedule_inconsistency(self, peerMac):
        pass # do nothing

    def recv_request(self, packet):
        pass # do nothing

    def clear_to_send_EBs_DATA(self):
        # always return True
        return True
