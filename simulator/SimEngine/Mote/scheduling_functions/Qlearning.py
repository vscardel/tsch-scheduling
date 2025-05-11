from __future__ import absolute_import
# =========================== imports =========================================

import copy
import itertools
import math
import netaddr
import numpy as np
import random
import SimEngine

from .. import MoteDefines as d
from math import factorial as fat
from math import e
from pprint import pprint
from SimEngine.Mote.sfBase import SchedulingFunctionBase


class SchedulingFunctionQlearning(SchedulingFunctionBase):

    SLOTFRAME_HANDLE = 1
    DEFAULT_CELL_LIST_LEN = 5
    MAX_RETRY = 3
    TX_CELL_OPT   = [d.CELLOPTION_TX]
    RX_CELL_OPT   = [d.CELLOPTION_RX]
    NUM_INITIAL_NEGOTIATED_TX_CELLS = 1
    NUM_INITIAL_NEGOTIATED_RX_CELLS = 0
    QUEUE_OVERFLOW = False

    charge = 0
    old_charge = 0

    prev_charge = 0

    last_inserted_cells_info = []
    last_removed_cells_info = []

    previous_queue_length = 0

    #poisson_computation
    num_packets_in_current_episode = 0
    curr_biggest_inst_score = 0

    #Q-learning
    current_state = ()
    EPSLON = None         
    EPISODE = 0
    Q_table = dict()
    TX_CELLS_PASSED = 0
    RX_CELLS_PASSED = 0
    MAX_EPSLON = 1

    #rewards weigths
    WQ = 0.33
    WE = 0.33
    WT = 0.33

    inserted_cells = []


    CUMULATIVE_REWARD = 0
    TD_ERROR = 0
    SUM_TD_ERROR = 0
    AVERAGE_TD_ERROR_100 = 0
    AVERAGE_CUMULATIVE_REWARD_100 = 0
    QLEARNING_STATS = {
        'AVERAGE_TD_ERROR': [],
        'AVERAGE_CUMULATIVE_REWARD': []
    }

    #traffic estimate variables
    TRAFFIC = 0
    array_rxs_acks = []
    prev_rx_ack = 0

    #charge estimate variables
    remaining_battery = 2821500
    AVERAGE_ENERGY_CONSUMED = 0
    array_energy_consumed = []

    #queue estimate variables
    AVERAGE_QUEUE_SIZE = 0
    array_queue_ratio = []

    
    def __init__(self, mote):
        super(SchedulingFunctionQlearning, self).__init__(mote)
        self.locked_slots         = set([])
        self.retry_count          = {}
        self.ALFA = self.settings.ALFA
        self.BETA = self.settings.BETA
        self.EPSLON_DECAY_RATE = self.settings.EPSLON_DECAY_RATE
        self.EPSLON_THRESHOLD = self.settings.EPSLON_THRESHOLD
        self.MIN_EPSLON = self.settings.MIN_EPSLON
        self.SLOTFRAME_INTERVAL_SIZE = self.settings.SLOTFRAME_INTERVAL_SIZE
        self.MAX_TX_CELLS_PASSED = self.settings.MAX_TX_CELLS_PASSED
        self.MAX_RX_CELLS_PASSED = self.settings.MAX_RX_CELLS_PASSED
        self.STATE_SIZE = self.settings.STATE_SIZE
        self.ACTION_STATE_SIZE = self.settings.ACTION_STATE_SIZE
        self.LAMBDA = self.settings.LAMBDA

    def start(self):
        self.name_to_compute_factor = {
            "traffic": self._compute_traffic,
            "queue": self._compute_queue_ratio,
            "charge": self._compute_charge,
            # "inst": self._compute_inst
         }
        self.name_to_discretize_function = {
            "traffic": self.discretize_traffic,
            "queue": self.discretize_queue_ratio,
            "charge": self.discretize_energy,
            # "inst": self.discretize_inst
         }
        slotframe_0 = self.mote.tsch.get_slotframe(0)
        self.mote.tsch.add_slotframe(
            slotframe_handle = self.SLOTFRAME_HANDLE,
            length           = slotframe_0.length
        )

        self.allocate_autonomous_rx_cell()

        if self.mote.dagRoot:
            # do nothing
            pass
        else:
            self.allocate_autonomous_rx_cell()
            self.initialize_q_table(self.STATE_SIZE,self.ACTION_STATE_SIZE)


    #indicates the ending of a window of X slotframes
    #X = SimEngine.SLOTFRAME_PERIOD_SIZE
    def indication_slotframe_window_ending(self,slotframe_period_size):
        pass


    def adapt_to_traffic(self, cellopt, cell, op):
        if self.mote.dagRoot:
            return
        preferred_parent = self.mote.rpl.getPreferredParent()

        if preferred_parent and self.mote.clear_to_send_EBs_DATA():

            current_state = self.compute_next_state(self.settings.factorial_combinations)
            if self.mote.id == 1:
                print(current_state)

            if hasattr(self, 'last_action'):
                self.compute_q_table(
                    self.last_state,    # s_t
                    current_state,      # s_{t+1}
                    self.last_action ,   # a_t,
                    op
                )

            # 3. Atualiza episodio e epslon
            self.EPISODE += 1
            self.EPSLON = self.MIN_EPSLON + (self.MAX_EPSLON - self.MIN_EPSLON) * np.exp(-self.EPSLON_DECAY_RATE * self.EPISODE)
            # print(self.EPSLON)
            # print('EPSLON')
            # print(self.EPSLON)

            state_number = self.map_state_to_number(current_state)

            if self.EPSLON < self.EPSLON_THRESHOLD:
                action = self.return_best_q_action(state_number)
            else:
                action = random.choice([0, 1, 2])


            discrete_state = self.discretize_variables(current_state)
            print(discrete_state)

            num_packets_to_be_generated = \
                discrete_state.get('queue', 0) + \
                discrete_state.get('charge', 0) +\
                discrete_state.get('traffic', 0)

            num_packets_to_remove = len(discrete_state) - num_packets_to_be_generated


            if action == 0 and num_packets_to_be_generated > 0:
                self.sixp_interface_add(
                    preferred_parent=preferred_parent,
                    num_cells=num_packets_to_be_generated,
                    cell_option=cellopt,
                )
            elif action == 1 and num_packets_to_remove > 0:
                self.sixp_interface_delete(
                    preferred_parent=preferred_parent,
                    num_cells=num_packets_to_remove,
                    cell_option=cellopt,
                )

            self.last_state = current_state
            self.last_action = action


    def take_random_action(self, preferred_parent, action, cellopt,ni, nd ,method = None):
        if action == 0:
            self.sixp_interface_add(
                preferred_parent = preferred_parent,
                num_cells        = ni,
                cell_option      = cellopt,
            )
        elif action == 1:
            self.sixp_interface_delete(
                preferred_parent = preferred_parent,
                num_cells        = nd,
                cell_option      = cellopt,
                method=method
            )

    def compute_next_state(self, factorial_combinations):
        state = {}
        for combination in factorial_combinations:
            my_function = self.return_function_by_name(combination, self.name_to_compute_factor)
            factor = my_function()
            state[combination] = factor
        return state
    

    def return_function_by_name(self, function_name, function_names):
        function_name = function_names.get(function_name)
        return function_name

    def save_qlearning_stats(self):
        self.QLEARNING_STATS['AVERAGE_TD_ERROR'].append(self.AVERAGE_TD_ERROR_100)
        self.QLEARNING_STATS['AVERAGE_CUMULATIVE_REWARD'].append(self.AVERAGE_CUMULATIVE_REWARD_100)

    def stop(self):
        self.mote.tsch.delete_slotframe(self.SLOTFRAME_HANDLE)
        self.EPSLON = 0
        self.EPISODE = 0
        self.CUMULATIVE_REWARD = 0
        self.QLEARNING_STATS = {
            'AVERAGE_TD_ERROR': [],
            'AVERAGE_CUMULATIVE_REWARD': []
        }
        

    def indication_neighbor_added(self, neighbor_mac_addr):
        pass # do nothing

    def indication_tx_cell_elapsed(self, cell, sent_packet):
        if self.mote.dagRoot:
            return
        # if not self._is_minimal_cell(cell):
        #     # self.TX_CELLS_PASSED = self.TX_CELLS_PASSED + 1
        #     # if bool(sent_packet): 
        #     #     self.num_packets_in_current_episode = self.num_packets_in_current_episode + 1
        #     # if (self.TX_CELLS_PASSED > 0 and \
        #     #         self.TX_CELLS_PASSED % self.MAX_TX_CELLS_PASSED == 0):
        #     #     # self.adapt_to_traffic([d.CELLOPTION_TX])
        #     #     self.TX_CELLS_PASSED = 0
        #     self.adapt_to_traffic([d.CELLOPTION_TX], cell, sent_packet)

    def indication_rx_cell_elapsed(self, cell, received_packet):
        if self.mote.dagRoot:
            return
        # if not self._is_minimal_cell(cell):
        #     # self.RX_CELLS_PASSED = self.RX_CELLS_PASSED + 1
        #     # if bool(received_packet): 
        #     #     self.num_packets_in_current_episode = self.num_packets_in_current_episode + 1
        #     # if (self.RX_CELLS_PASSED > 0 and \
        #     #         self.RX_CELLS_PASSED % self.MAX_RX_CELLS_PASSED == 0):
        #     #     # self.adapt_to_traffic([d.CELLOPTION_RX])
        #     #     self.RX_CELLS_PASSED = 0
        #     self.adapt_to_traffic([d.CELLOPTION_RX], cell, received_packet)

    def indication_queue_full(self):
        self.QUEUE_OVERFLOW = True

    def indication_parent_change(self, old_parent, new_parent):
        assert old_parent != new_parent

        # allocate the same number of cells to the new parent as it has for the
        # old parent; note that there could be three types of cells:
        # (TX=1,RX=1,SHARED=1), (TX=1), and (RX=1)
        if old_parent is None:
            num_tx_cells = self.NUM_INITIAL_NEGOTIATED_TX_CELLS
            num_rx_cells = self.NUM_INITIAL_NEGOTIATED_RX_CELLS
        else:
            dedicated_cells = self.mote.tsch.get_cells(
                mac_addr         = old_parent,
                slotframe_handle = self.SLOTFRAME_HANDLE
            )
            num_tx_cells = len(
                [cell for cell in dedicated_cells if cell.options == [d.CELLOPTION_TX]]
            )
            if num_tx_cells < self.NUM_INITIAL_NEGOTIATED_TX_CELLS:
                num_tx_cells = self.NUM_INITIAL_NEGOTIATED_TX_CELLS
            num_rx_cells = len(
                [cell for cell in dedicated_cells if cell.options == [d.CELLOPTION_RX]]
            )
            if num_rx_cells < self.NUM_INITIAL_NEGOTIATED_RX_CELLS:
                num_rx_cells = self.NUM_INITIAL_NEGOTIATED_RX_CELLS
        if new_parent:
            # reset the retry counter
            # we may better to make sure there is no outstanding
            # transaction with the same peer
            self.retry_count[new_parent] = 0
            cell_list = self._create_available_cell_list(self.DEFAULT_CELL_LIST_LEN)
            self._request_adding_cells(
                parent         = new_parent,
                cell_type      = [d.CELLOPTION_TX],
                num_cells      = num_tx_cells,
                cell_list      = cell_list
            )
            cell_list = self._create_available_cell_list(self.DEFAULT_CELL_LIST_LEN)
            self._request_adding_cells(
                parent         = new_parent,
                cell_type      = [d.CELLOPTION_RX],
                num_cells      = num_rx_cells,
                cell_list      = cell_list
            )

        # clear all the cells allocated for the old parent
        def _callback(event, packet):
            if event == d.SIXP_CALLBACK_EVENT_FAILURE:
                # optimization which is not mentioned in 6P/MSF spec: remove
                # the outstanding transaction because we're deleting all the
                # cells scheduled to the peer now. The outstanding transaction
                # should have the same transaction key as the packet we were
                # trying to send.
                self.mote.sixp.abort_transaction(
                    initiator_mac_addr=packet[u'mac'][u'srcMac'],
                    responder_mac_addr=packet[u'mac'][u'dstMac']
                )
            self._clear_cells(old_parent)

        if old_parent:
            cells = self.mote.tsch.get_cells(
                mac_addr         = old_parent,
                slotframe_handle = self.SLOTFRAME_HANDLE
            )
            if len(cells) >= 1:
                self.mote.sixp.send_request(
                    dstMac   = old_parent,
                    command  = d.SIXP_CMD_CLEAR,
                    callback = _callback
                )
            else:
                # do nothing
                pass

    def detect_schedule_inconsistency(self, peerMac):
        # send a CLEAR request to the peer
        self.mote.sixp.send_request(
            dstMac   = peerMac,
            command  = d.SIXP_CMD_CLEAR,
            callback = lambda event, packet: self._clear_cells(peerMac)
        )

    def recv_request(self, packet):
        if   packet[u'app'][u'code'] == d.SIXP_CMD_ADD:
            self._receive_add_request(packet)
        elif packet[u'app'][u'code'] == d.SIXP_CMD_DELETE:
            self._receive_delete_request(packet)
        elif packet[u'app'][u'code'] == d.SIXP_CMD_CLEAR:
            self._receive_clear_request(packet)
        elif packet[u'app'][u'code'] == d.SIXP_CMD_RELOCATE:
            pass
        else:
            # not implemented or not supported
            # ignore this request
            pass

    def clear_to_send_EBs_DATA(self):
        # True if we have a TX cell to the current parent
        slotframe = self.mote.tsch.get_slotframe(self.SLOTFRAME_HANDLE)
        parent_addr = self.mote.rpl.getPreferredParent()
        if (
                (slotframe is None)
                or
                (parent_addr is None)
            ):
            tx_cells = []
        else:
            tx_cells = [
                cell for cell in slotframe.get_cells_by_mac_addr(parent_addr)
                if cell.options == [d.CELLOPTION_TX]
            ]

        if self.mote.dagRoot:
            ret_val = True
        else:
            ret_val = bool(tx_cells)

        return ret_val
    
    def get_tx_cells(self, mac_addr):
        slotframe = self.mote.tsch.get_slotframe(
            self.SLOTFRAME_HANDLE
        )
        if slotframe:
            cells = slotframe.get_cells_by_mac_addr(mac_addr)
            autonomous_tx_cell = self.get_autonomous_tx_cell(mac_addr)
            return cells or autonomous_tx_cell
        else:
            return []
        
    #Q-Learning
    def generate_possible_states(self,num_state_variables):
        all_states = []
        for seq in itertools.product("01",repeat=num_state_variables):
            int_seq = [int(i) for i in seq]
            tuple_seq = tuple(int_seq)
            all_states.append(tuple_seq)
        return all_states
    
    def initialize_q_table(self, num_states, action_space_size):
        for state in range(num_states):
            self.Q_table[state] = [0]*action_space_size

    #utility methods#

    def _compute_poisson_packet_distribution(self,time_interval):
        MAX_NUM_PACKETS = 10
        distribution = []
        for i in range(0,MAX_NUM_PACKETS + 1):
            probability_i = (((self.LAMBDA*time_interval)**i) / fat(i))*(e**-(self.LAMBDA*time_interval))
            distribution.append(probability_i)
        return distribution


    def quantizing_unused_cells_rate(self,unused_cell_rate):
        if unused_cell_rate > self.UNUSED_CELL_RATE_MAX_THRESHOLD:
            return True 
        return False

    
    def compute_num_packets_to_be_generated(self,distribution):
        return distribution.index(max(distribution))
    
    def compute_num_packets_to_remove(self,LAMBDA,cell_option):
        preferred_parent = self.mote.rpl.getPreferredParent()
        allocated_cells = [cell for cell in self.mote.tsch.get_cells(preferred_parent, self.SLOTFRAME_HANDLE) 
                           if cell.options == cell_option]
        num_cells_to_remove = max(1,len(allocated_cells) - int(math.ceil(self.LAMBDA)))
        return num_cells_to_remove

    def _get_available_slots(self):
        available_slots = self.mote.tsch.get_available_slots(self.SLOTFRAME_HANDLE)
        if isinstance(available_slots,list):
            return list(set(available_slots) - self.locked_slots)
        return []
    
    def _get_unused_cells(self,cell_option):
        preferred_parent = self.mote.rpl.getPreferredParent()
        complete_cells = []
        if cell_option == d.CELLOPTION_TX:
            available_cells = [
                            {"channelOffset":cell.channel_offset,
                                "slotOffset":cell.slot_offset,
                                "num_tx": cell.num_tx,
                                "num_tx_ack": cell.num_tx} 
                            for cell in self.mote.tsch.get_cells(
                            preferred_parent,
                            self.SLOTFRAME_HANDLE
                        ) if cell.options == cell_option and float(cell.num_tx_ack )/ cell.num_tx >= 0.8]
            # and \
            #         float(cell.num_tx_ack )/ cell.num_tx >= 0.8
        else:
            available_cells = [
                            {"channelOffset":cell.channel_offset,
                                "slotOffset":cell.slot_offset,
                                "num_tx": cell.num_tx,
                                "num_tx_ack": cell.num_tx} 
                            for cell in self.mote.tsch.get_cells(
                            preferred_parent,
                            self.SLOTFRAME_HANDLE
                        ) if cell.options == cell_option and cell.num_rx > 0]
            # and \
            #         cell.num_rx > 0
        unused_cells = []
        for cell in available_cells:
            unused_cells.append(cell)
        return unused_cells
    
    def _compute_charge(self):
        self.charge +=  self.mote.radio.stats['idle_listen'] * d.CHARGE_IdleListen_uC
        self.charge += self.mote.radio.stats['tx_data_rx_ack'] * d.CHARGE_TxDataRxAck_uC
        self.charge += self.mote.radio.stats['rx_data_tx_ack'] * d.CHARGE_RxDataTxAck_uC
        self.charge += self.mote.radio.stats['tx_data'] * d.CHARGE_TxData_uC
        self.charge += self.mote.radio.stats['rx_data'] * d.CHARGE_RxData_uC
        self.charge += self.mote.radio.stats['sleep'] * d.CHARGE_Sleep_uC
        curr_charge = self.charge - self.old_charge
        self.old_charge = curr_charge
        return curr_charge
    
    def _compute_queue_ratio(self):
        return len(self.mote.tsch.txQueue)/float(self.settings.tsch_tx_queue_size)
    
    #returns how much of the total battery was consumed
    def _compute_charge_consumed_in_episode(self, charge):
        if not self.prev_charge:
            return 0 
        else:
            return charge - self.prev_charge 
        
        self.prev_charge = charge
        
    def _compute_traffic(self):
        current_traffic = self.mote.radio.stats["rx_data_tx_ack"] - self.prev_rx_ack
        self.prev_rx_ack = current_traffic
        return current_traffic

    def _compute_average_traffic(self, current_traffic):
        self.array_rxs_acks.append(current_traffic)
        if len(self.array_rxs_acks) == self.SLOTFRAME_INTERVAL_SIZE + 1:
            self.TRAFFIC = sum(
                self.array_rxs_acks[:self.SLOTFRAME_INTERVAL_SIZE])/float(self.SLOTFRAME_INTERVAL_SIZE)
            self.array_rxs_acks.pop(0)
            return self.TRAFFIC
        return current_traffic
    
    def _compute_queue_average_ratio(self, current_queue_ratio):
        self.array_queue_ratio.append(current_queue_ratio)
        if len(self.array_queue_ratio) == self.SLOTFRAME_INTERVAL_SIZE + 1:
            self.AVERAGE_QUEUE_SIZE = sum(
                self.array_queue_ratio[:self.SLOTFRAME_INTERVAL_SIZE])/float(self.SLOTFRAME_INTERVAL_SIZE)
            self.array_queue_ratio.pop(0)
            return self.AVERAGE_QUEUE_SIZE
        return current_queue_ratio
    

    def _compute_energy_average_ratio(self, current_energy_consumed):
        self.array_energy_consumed.append(current_energy_consumed)
        if len(self.array_energy_consumed) == self.SLOTFRAME_INTERVAL_SIZE + 1:
            self.AVERAGE_ENERGY_CONSUMED = sum(
                self.array_energy_consumed[:self.SLOTFRAME_INTERVAL_SIZE])/float(self.SLOTFRAME_INTERVAL_SIZE)
            self.array_energy_consumed.pop(0)
            return self.AVERAGE_ENERGY_CONSUMED
        return current_energy_consumed
    
    
    def compute_reward(self, next_state, action, op):

        if op == 'insertion':
            
            if self.last_inserted_cells_info:

                preferred_parent = self.mote.rpl.getPreferredParent()

                cells_in_slotframe = [
                    (cell.channel_offset, cell.slot_offset, cell.num_tx, cell.num_tx_ack) 
                    for cell in self.mote.tsch.get_cells(preferred_parent, self.SLOTFRAME_HANDLE)
                ]

                last_requested_cells = [
                    (cell['channelOffset'], cell['slotOffset'])
                    for cell in self.last_inserted_cells_info
                ]


                for cell in cells_in_slotframe:
                    if (cell[0], cell[1]) in last_requested_cells:
                        # we test to see if it is being used
                        # tx acks
                        if cell[3] > 0:
                            return 1
                # se nenhum das celulas que requisitei foram inseridas, isso eh ruim
                return -1
            return 0
        else:
            if self.last_removed_cells_info:
                preferred_parent = self.mote.rpl.getPreferredParent()

                cells_in_slotframe = [
                    (cell.channel_offset, cell.slot_offset, cell.num_tx, cell.num_tx_ack) 
                    for cell in self.mote.tsch.get_cells(preferred_parent, self.SLOTFRAME_HANDLE)
                ]

                last_requested_cells = [
                    (cell['channelOffset'], cell['slotOffset'])
                    for cell in self.last_removed_cells_info
                ]

                for cell in cells_in_slotframe:
                    if (cell[0], cell[1]) not in last_requested_cells:
                        return 1
                return -1
            return 0
        
    

    def discretize_queue_ratio(self,queue_ratio):
        average_queue_ratio = self._compute_queue_average_ratio(queue_ratio)
        if queue_ratio > average_queue_ratio:
            return 1
        return 0
    
    def discretize_energy(self,energy_consumed):
        average_energy_consumed = self._compute_energy_average_ratio(energy_consumed)
        if energy_consumed > average_energy_consumed:
            return 1
        return 0
    
    def discretize_traffic(self,traffic):
        average_traffic = self._compute_average_traffic(traffic)
        if traffic > average_traffic:
            return 1
        return 0

    def smooth_prefer_zero(self,x, sigma=1.0):
        return np.exp(- (x**2) / (2 * sigma**2))

    def smooth_away_from_zero(self, x, sigma=1):
        return 1 - np.exp(- (x**2) / (2 * sigma**2))
    
    def normalize_metric(self, metric_name, metric_value):
        if metric_name == 'traffic':
            return self.smooth_prefer_zero(metric_value)
        elif metric_name == 'queue':
            return self.smooth_prefer_zero(metric_value)
        elif metric_name == 'charge':
            return self.smooth_away_from_zero(metric_value, 33333)
        
    def discretize_variables(self, state):
        discrete_state =  {}
        for factor_name,factor in state.items():
            discretize_function = self.return_function_by_name(factor_name, self.name_to_discretize_function)
            discrete_factor = discretize_function(factor)
            discrete_state[factor_name] = discrete_factor
        return discrete_state

    def map_state_to_number(self, state):
        discrete_state = self.discretize_variables(state)
        # print(discrete_state)
        binary_number = ''
        for key,value in discrete_state.items():
            binary_number += str(value) 
        return int(binary_number,2)
    
    def return_best_q_value(self,state):
       current_vector = self.Q_table[state]
       return np.max(current_vector)
    
    def return_best_q_action(self,state):
        current_vector = self.Q_table[state]
        return np.argmax(current_vector)
    
    def compute_q_table(self, state, next_state, action, op):
        curr_state_number = self.map_state_to_number(state)
        next_state_number = self.map_state_to_number(next_state)

        # Compute reward
        reward = self.compute_reward(next_state, action, op)
        # print(reward)
        self.CUMULATIVE_REWARD += reward  # Accumulate reward
            
        # print('CUMULATIVE REWARD:', self.CUMULATIVE_REWARD)

        # Compute temporal difference error (TD Error)
        # import ipdb;
        # ipdb.set_trace()
        best_next_q = self.return_best_q_value(next_state_number)  # max Q(s', a')
        temporal_difference = reward + self.BETA * best_next_q - self.Q_table[curr_state_number][action]
        self.TD_ERROR = temporal_difference
        self.SUM_TD_ERROR += self.TD_ERROR

        if self.EPISODE % 100 == 0:
            self.AVERAGE_CUMULATIVE_REWARD_100 += self.CUMULATIVE_REWARD / float(100)
            self.AVERAGE_TD_ERROR_100 += self.SUM_TD_ERROR / float(100)
            self.CUMULATIVE_REWARD = 0
            self.SUM_TD_ERROR = 0
        # print('TD ERROR:', self.TD_ERROR)

        # Update Q-value using Q-learning update rule
        self.Q_table[curr_state_number][action] += self.ALFA * temporal_difference
        # print(self.Q_table)
        # from pprint import pprint 
        # print(reward)
        # print(self.Q_table)
    
    
    def _is_minimal_cell(self,cell):
        if cell.slot_offset == 0 and cell.channel_offset == 0:
            return True
        return False

    
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
    
    def _add_cells(self, neighbor, cell_list, cell_options):
        try:
            for cell in cell_list:
                self.mote.tsch.addCell(
                    slotOffset         = cell[u'slotOffset'],
                    channelOffset      = cell[u'channelOffset'],
                    neighbor           = neighbor,
                    cellOptions        = cell_options,
                    slotframe_handle   = self.SLOTFRAME_HANDLE
                )
        except Exception:
            # We may fail in adding cells since they could be allocated for
            # another peer. We need to have a locking or reservation mechanism
            # to avoid such a situation.
            raise

    def _housekeeping_collision(self):
        """
        Identify cells where schedule collisions occur.
        draft-chang-6tisch-msf-01:
            The key for detecting a schedule collision is that, if a node has
            several cells to the same preferred parent, all cells should exhibit
            the same PDR.  A cell which exhibits a PDR significantly lower than
            the others indicates than there are collisions on that cell.
        :return:
        """

        if self.mote.tsch.get_slotframe(self.SLOTFRAME_HANDLE) is None:
            return

        # for quick access; get preferred parent
        preferred_parent = self.mote.rpl.getPreferredParent()

        # collect TX cells which has enough numTX
        tx_cell_list = [cell for cell in self.mote.tsch.get_cells(preferred_parent, self.SLOTFRAME_HANDLE) if cell.options == [d.CELLOPTION_TX]]
        # pick up TX cells whose NumTx is larger than
        # MSF_MIN_NUM_TX. This is an implementation decision, which is
        # easier to implement than what section 5.3 of
        # draft-ietf-6tisch-msf-03.txt describes as the step-2 of the
        # house-keeping process.
        tx_cell_list = {
            cell.slot_offset: cell for cell in tx_cell_list if (
                d.MSF_MIN_NUM_TX < cell.num_tx
            )
        }
        # collect PDRs of the TX cells
        def pdr(cell):
            assert cell.num_tx > 0
            return cell.num_tx_ack / float(cell.num_tx)
        pdr_list = {
            slotOffset: pdr(cell) for slotOffset, cell in list(tx_cell_list.items())
        }

        if len(pdr_list) > 0:
            # find a cell to relocate using the highest PDR value
            highest_pdr = max(pdr_list.values())
            relocation_cell_list = [
                {
                    'slotOffset'   : slotOffset,
                    'channelOffset': tx_cell_list[slotOffset].channel_offset
                } for slotOffset, pdr in list(pdr_list.items()) if (
                    d.MSF_RELOCATE_PDRTHRES < (highest_pdr - pdr)
                )
            ]
            if (
                    len(relocation_cell_list) > 0
                    and
                    self.retry_count[preferred_parent] == -1
                ):
                # reset retry counter
                self.retry_count[preferred_parent] = 0
                self._request_relocating_cells(
                    neighbor             = preferred_parent,
                    cell_options         = self.TX_CELL_OPT,
                    num_relocating_cells = len(relocation_cell_list),
                    cell_list            = relocation_cell_list
                )
        else:
            # we don't have any TX cell whose PDR is available; do nothing
            pass

        # schedule next housekeeping
        self.engine.scheduleIn(
            delay         = d.MSF_HOUSEKEEPINGCOLLISION_PERIOD,
            cb            = self._housekeeping_collision,
            uniqueTag     = (self.mote.id, u'_housekeeping_collision'),
            intraSlotOrder= d.INTRASLOTORDER_STACKTASKS,
        )

    #################6P INTERFACE CODE################

    # ADD command related stuff
    def _request_adding_cells(
            self,
            parent,
            cell_type,
            num_cells,
            cell_list
        ):

        if num_cells == 0:
            # nothing to add
            self.retry_count[parent] = -1
            return

        if len(cell_list) == 0:
            # we don't have available cells right now
            self.log(
                SimEngine.SimLog.LOG_MSF_ERROR_SCHEDULE_FULL,
                {
                    '_mote_id'    : self.mote.id
                }
            )
            self.retry_count[parent] = -1
            return

        # prepare _callback which is passed to SixP.send_request()
        callback = self._create_add_request_callback(
            parent,
            num_cells,
            cell_type,
            cell_list
        )

        # send a request
        self.mote.sixp.send_request(
            dstMac      = parent,
            command     = d.SIXP_CMD_ADD,
            cellOptions = cell_type,
            numCells    = num_cells,
            cellList    = cell_list,
            callback    = callback
        )

    
    def _request_deleting_cells(
            self,
            parent,
            num_cells,
            cell_options
        ):

        # prepare cell_list to send
        cell_list = self._create_occupied_cell_list(
            neighbor      = parent,
            cell_options  = cell_options,
            cell_list_len = self.DEFAULT_CELL_LIST_LEN
        )
        assert len(cell_list) > 0

        # prepare callback
        callback = self._create_delete_request_callback(
            parent,
            num_cells,
            cell_options
        )

        # send a DELETE request
        self.mote.sixp.send_request(
            dstMac      = parent,
            command     = d.SIXP_CMD_DELETE,
            cellOptions = cell_options,
            numCells    = num_cells,
            cellList    = cell_list,
            callback    = callback
        )

    def _create_delete_request_callback(
            self,
            neighbor,
            num_cells,
            cell_options
        ):
        def callback(event, packet):
            if (
                    (event == d.SIXP_CALLBACK_EVENT_PACKET_RECEPTION)
                    and
                    (packet[u'app'][u'msgType'] == d.SIXP_MSG_TYPE_RESPONSE)
                ):
                self.retry_count[neighbor] = -1
                if packet[u'app'][u'code'] == d.SIXP_RC_SUCCESS:
                    self._delete_cells(
                        neighbor     = neighbor,
                        cell_list    = packet[u'app'][u'cellList'],
                        cell_options = cell_options
                    )
                else:
                    # TODO: request doesn't succeed; how should we do?
                    pass
            elif event == d.SIXP_CALLBACK_EVENT_TIMEOUT:
                if self.retry_count[neighbor] == self.MAX_RETRY:
                    # give it up
                    self.retry_count[neighbor] = -1
                    if neighbor == self.mote.rpl.getPreferredParent():
                        self.mote.rpl.of.poison_rpl_parent(neighbor)
                else:
                    # retry
                    self.retry_count[neighbor] += 1
                    self._request_deleting_cells(
                        neighbor,
                        num_cells,
                        cell_options
                    )
            else:
                # ignore other events
                pass

        return callback

        # CLEAR command related stuff
    def _receive_clear_request(self, request):

        peerMac = request[u'mac'][u'srcMac']

        def callback(event, packet):
            # remove all the cells no matter what happens
            self._clear_cells(peerMac)

        # create CLEAR response
        self.mote.sixp.send_response(
            dstMac      = peerMac,
            return_code = d.SIXP_RC_SUCCESS,
            callback    = callback
        )

    def _create_add_request_callback(
            self,
            parent,
            num_cells,
            cell_type,
            cell_list
        ):
        def callback(event, packet):
            if event == d.SIXP_CALLBACK_EVENT_PACKET_RECEPTION:
                assert packet[u'app'][u'msgType'] == d.SIXP_MSG_TYPE_RESPONSE
                if packet[u'app'][u'code'] == d.SIXP_RC_SUCCESS:
                    # add cells on success of the transaction
                    self._add_cells(
                        neighbor     = parent,
                        cell_list    = packet[u'app'][u'cellList'],
                        cell_options = cell_type
                    )

                    # The received CellList could be smaller than the requested
                    # NumCells; adjust num_{tx,rx}_cells
                    remaining_cells = num_cells - len(packet[u'app'][u'cellList'])
                    if remaining_cells > 0:
                        # start another transaction
                        self.retry_count[parent] = 0
                        self._request_adding_cells(
                            parent       = parent,
                            cell_type      = cell_type,
                            num_cells      = num_cells,
                            cell_list      = cell_list
                        )
                else:
                    # TODO: request doesn't succeed; how should we do?
                    self.retry_count[parent] = -1

            elif event == d.SIXP_CALLBACK_EVENT_TIMEOUT:
                if self.retry_count[parent] == self.MAX_RETRY:
                    # give up this neighbor
                    if parent == self.mote.rpl.getPreferredParent():
                        self.mote.rpl.of.poison_rpl_parent(parent)
                    self.retry_count[parent] = -1 # done
                else:
                    # retry
                    self.retry_count[parent] += 1
                    self._request_adding_cells(
                        parent       = parent,
                        cell_type      = cell_type,
                        num_cells      = num_cells,
                        cell_list      = cell_list
                    )
            else:
                # ignore other events
                pass

            # unlock the slots used in this transaction
            self._unlock_cells(cell_list)

        return callback
    
    def _receive_add_request(self, request):

        # for quick access
        proposed_cells = request[u'app'][u'cellList']
        peerMac        = request[u'mac'][u'srcMac']

        # find available cells in the received CellList
        slots_in_cell_list = set(
            [c[u'slotOffset'] for c in proposed_cells]
        )
        available_slots  = list(
            slots_in_cell_list.intersection(
                set(self._get_available_slots())
            )
        )

        # prepare cell_list
        candidate_cells = [
            c for c in proposed_cells if c[u'slotOffset'] in available_slots
        ]
        if len(candidate_cells) < request[u'app'][u'numCells']:
            cell_list = candidate_cells
        else:
            cell_list = random.sample(
                candidate_cells,
                request[u'app'][u'numCells']
            )

        # prepare callback
        if len(available_slots) > 0:
            code = d.SIXP_RC_SUCCESS

            self._lock_cells(candidate_cells)
            def callback(event, packet):
                if event == d.SIXP_CALLBACK_EVENT_MAC_ACK_RECEPTION:
                    # prepare cell options for this responder
                    if request[u'app'][u'cellOptions'] == self.TX_CELL_OPT:
                        # invert direction
                        cell_options = self.RX_CELL_OPT
                    elif request[u'app'][u'cellOptions'] == self.RX_CELL_OPT:
                        # invert direction
                        cell_options = self.TX_CELL_OPT
                    else:
                        # Unsupported cell options for MSF
                        raise Exception()

                    self._add_cells(
                        neighbor     = peerMac,
                        cell_list    = cell_list,
                        cell_options = cell_options
                )
                self._unlock_cells(candidate_cells)
        else:
            code      = d.SIXP_RC_ERR
            cell_list = None
            callback  = None

        # send a response
        self.mote.sixp.send_response(
            dstMac      = peerMac,
            return_code = code,
            cellList    = cell_list,
            callback    = callback
        )
    
    def _are_cells_allocated(
            self,
            peerMac,
            cell_list,
            cell_options
        ):

        # collect allocated cells
        assert cell_options in [self.TX_CELL_OPT, self.RX_CELL_OPT]
        allocated_cells = [cell for cell in self.mote.tsch.get_cells(peerMac, self.SLOTFRAME_HANDLE) if cell.options == cell_options]

        # test all the cells in the cell list against the allocated cells
        ret_val = True
        for cell in cell_list:
            slotOffset    = cell[u'slotOffset']
            channelOffset = cell[u'channelOffset']
            cell = self.mote.tsch.get_cell(
                slot_offset      = slotOffset,
                channel_offset   = channelOffset,
                mac_addr         = peerMac,
                slotframe_handle = self.SLOTFRAME_HANDLE
            )

            if cell is None:
                ret_val = False
                break

        return ret_val

    def _receive_delete_request(self, request):

        # for quick access
        num_cells           = request[u'app'][u'numCells']
        cell_options        = request[u'app'][u'cellOptions']
        candidate_cell_list = request[u'app'][u'cellList']
        peerMac             = request[u'mac'][u'srcMac']

        # confirm all the cells in the cell list are allocated for the peer
        # with the specified cell options
        #
        # invert the direction in cell_options
        assert cell_options in [self.TX_CELL_OPT, self.RX_CELL_OPT]
        if   cell_options == self.TX_CELL_OPT:
            our_cell_options = self.RX_CELL_OPT
        elif cell_options == self.RX_CELL_OPT:
            our_cell_options   = self.TX_CELL_OPT

        if (
                (
                    self._are_cells_allocated(
                        peerMac      = peerMac,
                        cell_list    = candidate_cell_list,
                        cell_options = our_cell_options
                    ) is True
                )
                and
                (num_cells <= len(candidate_cell_list))
            ):
            code = d.SIXP_RC_SUCCESS
            #decide what cells to delete
            cell_list = random.sample(candidate_cell_list, num_cells)

            def callback(event, packet):
                if event == d.SIXP_CALLBACK_EVENT_MAC_ACK_RECEPTION:
                    self._delete_cells(
                        neighbor     = peerMac,
                        cell_list    = cell_list,
                        cell_options = our_cell_options
                )
        else:
            code      = d.SIXP_RC_ERR
            cell_list = None
            callback  = None

        # send the response
        self.mote.sixp.send_response(
            dstMac      = peerMac,
            return_code = code,
            cellList    = cell_list,
            callback    = callback
        )

    def _clear_cells(self, neighbor):
        cells = self.mote.tsch.get_cells(
            neighbor,
            self.SLOTFRAME_HANDLE
        )
        for cell in cells:
            assert neighbor == cell.mac_addr
            self.mote.tsch.deleteCell(
                slotOffset       = cell.slot_offset,
                channelOffset    = cell.channel_offset,
                neighbor         = cell.mac_addr,
                cellOptions      = cell.options,
                slotframe_handle = self.SLOTFRAME_HANDLE
            )
        self.mote.sixp.reset_seqnum(neighbor)

    def _create_occupied_cell_list(
            self,
            neighbor,
            cell_options,
            cell_list_len
        ):

        occupied_cells = [cell for cell in self.mote.tsch.get_cells(neighbor, self.SLOTFRAME_HANDLE) if cell.options == cell_options]

        cell_list = [
            {
                'slotOffset'   : cell.slot_offset,
                'channelOffset': cell.channel_offset
            } for cell in occupied_cells
        ]

        if cell_list_len <= len(occupied_cells):
            cell_list = random.sample(cell_list, cell_list_len)

        return cell_list

    def _delete_cells(self, neighbor, cell_list, cell_options):
        for cell in cell_list:
            if self.mote.tsch.get_cell(
                    slot_offset      = cell[u'slotOffset'],
                    channel_offset   = cell[u'channelOffset'],
                    mac_addr         = neighbor,
                    slotframe_handle = self.SLOTFRAME_HANDLE
               ) is None:
                # the cell may have been deleted for some reason
                continue
            self.mote.tsch.deleteCell(
                slotOffset       = cell[u'slotOffset'],
                channelOffset    = cell[u'channelOffset'],
                neighbor         = neighbor,
                cellOptions      = cell_options,
                slotframe_handle = self.SLOTFRAME_HANDLE
            )

    def sixp_interface_add(
            self,
            preferred_parent,
            num_cells,
            cell_option,
        ):

        cell_list = self._create_available_cell_list(self.DEFAULT_CELL_LIST_LEN)
        self.last_inserted_cells_info = copy.deepcopy(cell_list)
        # prepare _callback which is passed to SixP.send_request()
        callback = self._create_add_request_callback(
            preferred_parent,
            num_cells,
            cell_option,
            cell_list
        )
        self.mote.sixp.send_request(
            dstMac      = preferred_parent,
            command     = d.SIXP_CMD_ADD,
            cellOptions = cell_option,
            numCells    = num_cells,
            cellList    = cell_list,
            callback    = callback
        )

    #num_cells is here only for backward compatibility
    def sixp_interface_delete(
            self,
            num_cells,
            preferred_parent,
            cell_option,
            method = None
        ):

        cells_to_delete = self._get_unused_cells(cell_option)
        self.last_removed_cells_info = copy.deepcopy(cells_to_delete)
        if num_cells > len(cells_to_delete):
            num_cells = cells_to_delete

        if len(cells_to_delete) >= 1:
            self.last_removed_cells_info = cells_to_delete
            callback = self._create_delete_request_callback(
                preferred_parent,
                len(cells_to_delete),
                cell_option
            )
            self.mote.sixp.send_request(
                dstMac      = preferred_parent,
                command     = d.SIXP_CMD_DELETE,
                cellOptions = cell_option,
                numCells    = num_cells,
                cellList    = cells_to_delete,
                callback    = callback
            )
#################################################
    # autonomous cell
    def get_autonomous_rx_cell(self):
        slotframe = self.mote.tsch.get_slotframe(
            self.SLOTFRAME_HANDLE
        )
        if slotframe:
            cells = slotframe.get_cells_by_mac_addr(None)
            if cells:
                assert len(cells) == 1
                assert cells[0].options == [d.CELLOPTION_RX]
                ret = cells[0]
            else:
                ret = None
        else:
            ret = None
        return ret

    def allocate_autonomous_rx_cell(self):
        mac_addr = self.mote.get_mac_addr()
        slot_offset, channel_offset = self._compute_autonomous_cell(mac_addr)
        self.mote.tsch.addCell(
            slotOffset       = slot_offset,
            channelOffset    = channel_offset,
            neighbor         = None,
            cellOptions      = [
                d.CELLOPTION_RX
            ],
            slotframe_handle = self.SLOTFRAME_HANDLE
        )

    def get_autonomous_tx_cell(self, mac_addr):
        slotframe = self.mote.tsch.get_slotframe(
            self.SLOTFRAME_HANDLE
        )
        if slotframe:
            cells = slotframe.get_cells_by_mac_addr(mac_addr)
            autonomous_cells = [
                cell for cell in cells
                if (
                        (d.CELLOPTION_TX in cell.options)
                        and
                        (d.CELLOPTION_SHARED in cell.options)
                )
            ]
            if autonomous_cells:
                assert len(autonomous_cells) == 1
                ret = autonomous_cells[0]
            else:
                ret = None
        else:
            ret = None
        return ret

    def allocate_autonomous_tx_cell(self, mac_addr):
        slot_offset, channel_offset = self._compute_autonomous_cell(mac_addr)
        self.mote.tsch.addCell(
            slotOffset       = slot_offset,
            channelOffset    = channel_offset,
            neighbor         = mac_addr,
            cellOptions      = [
                d.CELLOPTION_TX,
                d.CELLOPTION_SHARED
            ],
            slotframe_handle = self.SLOTFRAME_HANDLE
        )

    def deallocate_autonomous_tx_cell(self, mac_addr):
        slot_offset, channel_offset = self._compute_autonomous_cell(mac_addr)
        self.mote.tsch.deleteCell(
            slotOffset       = slot_offset,
            channelOffset    = channel_offset,
            neighbor         = mac_addr,
            cellOptions      = [
                d.CELLOPTION_TX,
                d.CELLOPTION_SHARED
            ],
            slotframe_handle = self.SLOTFRAME_HANDLE
        )

    def _compute_autonomous_cell(self, mac_addr):
        slotframe = self.mote.tsch.get_slotframe(
            self.SLOTFRAME_HANDLE
        )
        hash_value = self._sax(mac_addr)

        slot_offset = int(1 + (hash_value % (slotframe.length - 1)))
        channel_offset = int(hash_value % self.settings.phy_numChans)

        return (slot_offset, channel_offset)

    # SAX
    def _sax(self, mac_addr):
        # XXX: a concrete definition of this hash function is needed to be
        # provided by the draft

        LEFT_SHIFT_NUM = 5
        RIGHT_SHIFT_NUM = 2

        # assuming v (seed) is 0
        hash_value = 0
        for word in netaddr.EUI(mac_addr).words:
            for byte in divmod(word, 0x100):
                left_shifted = (hash_value << LEFT_SHIFT_NUM)
                right_shifted = (hash_value >> RIGHT_SHIFT_NUM)
                hash_value ^= left_shifted + right_shifted + byte

        # assuming T (table size) is 16-bit
        return hash_value & 0xFFFF