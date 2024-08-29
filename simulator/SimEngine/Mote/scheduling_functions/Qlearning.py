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


class SchedulingFunctionQlearning(SchedulingFunctionBase):

    SLOTFRAME_HANDLE = 1


    def __init__(self, mote):
        super(SchedulingFunctionQlearning, self).__init__(mote)

    def start(self):
        slotframe_0 = self.mote.tsch.get_slotframe(0)
        self.mote.tsch.add_slotframe(
            slotframe_handle = self.SLOTFRAME_HANDLE,
            length           = slotframe_0.length
        )

    def stop(self):
        self.mote.tsch.delete_slotframe(self.SLOTFRAME_HANDLE)

    def indication_neighbor_added(self, neighbor_mac_addr):
        pass # do nothing

    def indication_tx_cell_elapsed(self, cell, sent_packet):
        pass

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

    #################Private################

