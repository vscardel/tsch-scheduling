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
        self.locked_slots         = set([])

    def start(self):
        slotframe_0 = self.mote.tsch.get_slotframe(0)
        self.mote.tsch.add_slotframe(
            slotframe_handle = self.SLOTFRAME_HANDLE,
            length           = slotframe_0.length
        )

    #indicates the ending of a window of X slotframes
    #X = SimEngine.SLOTFRAME_PERIOD_SIZE
    def indication_slotframe_window_ending(self):
        pass

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

    #################6P INTERFACE CODE################

    def _lock_cells(self, cell_list):
        for cell in cell_list:
            self.locked_slots.add(cell[u'slotOffset'])
    
    def _unlock_cells(self, cell_list):
        for cell in cell_list:
            self.locked_slots.remove(cell[u'slotOffset'])

    #tsch interface to get available slots
    def _get_available_slots(self):
        return list(
            set(self.mote.tsch.get_available_slots(self.SLOTFRAME_HANDLE_NEGOTIATED_CELLS)) -
            self.locked_slots
        )

    #Of the available cells, pick randomly which cell to choose
    def _create_available_cell_list(self, cell_list_len):
        available_slots = self._get_available_slots()
        # remove slot offset 0 that is reserved for the minimal shared
        # cell
        if 0 in available_slots:
            available_slots.remove(0)

        if len(available_slots) < cell_list_len:
            # we don't have enough available cells; no cell is selected
            selected_slots = []
        else:
            selected_slots = random.sample(available_slots, cell_list_len)

        cell_list = []
        for slot_offset in selected_slots:
            channel_offset = random.randint(0, self.settings.phy_numChans - 1)
            cell_list.append(
                {
                    'slotOffset'   : slot_offset,
                    'channelOffset': channel_offset
                }
            )
        self._lock_cells(cell_list)
        return cell_list

    def _create_add_request_callback(
            self,
            neighbor,
            num_cells,
            cell_options,
            cell_list,
            num_tx_cells,
            num_rx_cells
        ):
        def callback(event, packet):
            if event == d.SIXP_CALLBACK_EVENT_PACKET_RECEPTION:
                assert packet[u'app'][u'msgType'] == d.SIXP_MSG_TYPE_RESPONSE
                if packet[u'app'][u'code'] == d.SIXP_RC_SUCCESS:
                    # add cells on success of the transaction
                    self._add_cells(
                        neighbor     = neighbor,
                        cell_list    = packet[u'app'][u'cellList'],
                        cell_options = cell_options
                    )

                    # The received CellList could be smaller than the requested
                    # NumCells; adjust num_{tx,rx}_cells
                    _num_tx_cells   = num_tx_cells
                    _num_rx_cells   = num_rx_cells
                    remaining_cells = num_cells - len(packet[u'app'][u'cellList'])
                    if remaining_cells > 0:
                        if cell_options == self.TX_CELL_OPT:
                            _num_tx_cells -= remaining_cells
                        elif cell_options == self.RX_CELL_OPT:
                            _num_rx_cells -= remaining_cells
                        else:
                            # never comes here
                            raise Exception()

                    # start another transaction
                    self.retry_count[neighbor] = 0
                    self._request_adding_cells(
                        neighbor       = neighbor,
                        num_tx_cells   = _num_tx_cells,
                        num_rx_cells   = _num_rx_cells
                    )
                else:
                    # TODO: request doesn't succeed; how should we do?
                    self.retry_count[neighbor] = -1

            elif event == d.SIXP_CALLBACK_EVENT_TIMEOUT:
                if self.retry_count[neighbor] == self.MAX_RETRY:
                    # give up this neighbor
                    if neighbor == self.mote.rpl.getPreferredParent():
                        self.mote.rpl.of.poison_rpl_parent(neighbor)
                    self.retry_count[neighbor] = -1 # done
                else:
                    # retry
                    self.retry_count[neighbor] += 1
                    if cell_options == self.TX_CELL_OPT:
                        _num_tx_cells = num_cells + num_tx_cells
                        _num_rx_cells = num_rx_cells
                    else:
                        _num_tx_cells = num_tx_cells
                        _num_rx_cells = num_cells + num_rx_cells
                    self._request_adding_cells(
                        neighbor       = neighbor,
                        num_tx_cells   = _num_tx_cells,
                        num_rx_cells   = _num_rx_cells
                    )
            else:
                # ignore other events
                pass

            # unlock the slots used in this transaction
            self._unlock_cells(cell_list)

        return callback


    def _request_6p(self,dstMac,sixp_command,cell_type,num_cells):

        #get random cells from the available ones
        cell_list = self._create_available_cell_list(num_cells)

        if len(cell_list) == 0:
            # we don't have available cells right now
            self.log(
                SimEngine.SimLog.LOG_MSF_ERROR_SCHEDULE_FULL,
                {
                    '_mote_id'    : self.mote.id
                }
            )
            return

        num_tx_cells = 0
        num_rx_cells = 0
        if cell_list == d.CELLOPTION_TX:
            num_tx_cells = num_cells
            num_rx_cells = 0
        else:
            num_tx_cells = 0
            num_rx_cells = num_cells

        callback = self._create_add_request_callback(
            dstMac,
            num_cells,
            cell_type,
            cell_list,
            num_tx_cells,
            num_rx_cells
        )

        self.mote.sixp.send_request(
            dstMac      = dstMac,
            command     = sixp_command,
            cellOptions = cell_type,
            numCells    = num_cells,
            cellList    = cell_list,
            callback    = callback
        )

    #################END OF 6P INTERFACE CODE################
