from __future__ import absolute_import

# =========================== imports =========================================

import random

from builtins import range

from .. import MoteDefines as d
from .MSF import SchedulingFunctionMSF


class SchedulingFunctionRLSF(SchedulingFunctionMSF):
    """RL-SF, from Pratama and Chung, ICEIEC 2022, doi 10.1109/ICEIEC54567.2022.9835090.

    At the start of every slotframe the node reads one number, how many
    messages are waiting in its queue, and picks from the action space how many
    negotiated TX cells it should hold towards its preferred parent for that
    slotframe. The action is the cell count itself, not a change to it, so 6P
    adds or deletes whatever the difference from the previous slotframe is.
    Over the slotframe the node then scores that choice on how full the queue
    stayed, how many of its cells went unused, and whether it dropped a packet.

    It is built on MSF because the paper runs on MSF (its Table I) and replaces
    only the decision, leaving the 6P negotiation, the autonomous cells and the
    collision housekeeping alone. Overriding _adapt_to_traffic to do nothing is
    what takes MSF's threshold rule out of the way. Everything below the
    decision is inherited, so RL-SF negotiates cells exactly as MSF and EMSF
    do, and in particular it removes cells the way the stock simulator does.
    The utilisation-aware removal is DynQ's own, and letting RL-SF borrow it
    would compare DynQ against DynQ's idea wearing someone else's state.

    Four things the paper does not state, chosen here and exposed as settings
    so the choice can be argued or changed rather than buried:

    1. The reward weights theta1, theta2 and theta3 of its Eq. 3. Equal weights
       of 1.0 are the default, which is the only neutral reading.
    2. The decay rate of Eq. 8. The paper gives the endpoints, 0.8 down to 0.1,
       and says the exponential schedule beat the constant one, but not the
       rate. The default reaches the floor in about a thousand decisions, which
       is a fifth of the 5000-slotframe run the paper reports.
    3. The size of the Q-table. Figure 1 of the paper draws eight rows and
       eight columns, so eight queue levels and eight cell counts, while its
       Table I sets the TSCH queue to fifteen. Eight is taken from the figure
       and the queue occupancy saturates there.
    4. Which drops count. The paper says "a packet drop in the current
       slotframe" and motivates the whole method by queue growth, so this
       counts drops from a full TX queue.

    One place the paper is followed in spirit rather than to the letter: its
    Eq. 1 and Eq. 2 together read Q <- (1-a)Q + a[(r + B maxQ') - Q], which
    subtracts the old value twice and is not Q-learning. The standard update is
    used instead, which is plainly what Eq. 1 and Eq. 2 are reaching for.
    Implementing the typo would have handed us a baseline that cannot learn.
    """

    # the paper's own figures, kept here rather than in the config so a run
    # without RLSF settings still reproduces the published algorithm
    DEFAULT_ALFA           = 0.1     # learning rate, section IV.B
    DEFAULT_BETA           = 0.95    # discount factor, section IV.B
    DEFAULT_EPSILON_INIT   = 0.8     # section IV.B
    DEFAULT_EPSILON_END    = 0.1     # section IV.B
    DEFAULT_EPSILON_DECAY  = 0.998   # not stated, see the note above
    DEFAULT_THETA_QUEUE    = 1.0     # theta1 of Eq. 3, not stated
    DEFAULT_THETA_UNUSED   = 1.0     # theta2 of Eq. 3, not stated
    DEFAULT_THETA_DROP     = 1.0     # theta3 of Eq. 3, not stated
    DEFAULT_NUM_QUEUE_STATES = 8     # rows of Figure 1
    DEFAULT_MAX_CELLS        = 8     # columns of Figure 1

    def __init__(self, mote):
        super(SchedulingFunctionRLSF, self).__init__(mote)

        self.ALFA = getattr(
            self.settings, 'RLSF_ALFA', self.DEFAULT_ALFA)
        self.BETA = getattr(
            self.settings, 'RLSF_BETA', self.DEFAULT_BETA)
        self.EPSILON_INIT = getattr(
            self.settings, 'RLSF_EPSILON_INIT', self.DEFAULT_EPSILON_INIT)
        self.EPSILON_END = getattr(
            self.settings, 'RLSF_EPSILON_END', self.DEFAULT_EPSILON_END)
        self.EPSILON_DECAY = getattr(
            self.settings, 'RLSF_EPSILON_DECAY', self.DEFAULT_EPSILON_DECAY)
        self.THETA_QUEUE = getattr(
            self.settings, 'RLSF_THETA_QUEUE', self.DEFAULT_THETA_QUEUE)
        self.THETA_UNUSED = getattr(
            self.settings, 'RLSF_THETA_UNUSED', self.DEFAULT_THETA_UNUSED)
        self.THETA_DROP = getattr(
            self.settings, 'RLSF_THETA_DROP', self.DEFAULT_THETA_DROP)
        self.NUM_QUEUE_STATES = getattr(
            self.settings, 'RLSF_NUM_QUEUE_STATES',
            self.DEFAULT_NUM_QUEUE_STATES)
        self.MAX_CELLS = getattr(
            self.settings, 'RLSF_MAX_CELLS', self.DEFAULT_MAX_CELLS)

        # one row per queue level, one column per reachable cell count
        self.q_table = [
            [0.0 for _ in range(self.MAX_CELLS)]
            for _ in range(self.NUM_QUEUE_STATES)
        ]

        self.epsilon = self.EPSILON_INIT
        self.last_state = None
        self.last_action = None

        # what happened since the last decision, which is what the reward of
        # that decision is made of
        self.cells_elapsed = 0
        self.cells_used = 0
        self.dropped_a_packet = False

        # for the record, so a run can be read back without re-deriving it
        self.RLSF_STATS = {
            'DECISIONS': {},
            'EPSILON': {},
        }
        self.decision_count = 0

    # ======================= public ==========================================

    def indication_slotframe_window_ending(self, slotframe_period_size):
        """A slotframe has ended, so the agent scores it and picks the next one."""
        if self.mote.dagRoot:
            return

        preferred_parent = self.mote.rpl.getPreferredParent()
        if preferred_parent is None:
            return

        state = self._observe_state()

        if self.last_state is not None:
            reward = self._compute_reward()
            self._update_q_table(
                self.last_state, self.last_action, reward, state
            )

        action = self._choose_action(state)
        self._apply_action(preferred_parent, state, action)

        self.last_state = state
        self.last_action = action
        self.epsilon = max(
            self.epsilon * self.EPSILON_DECAY, self.EPSILON_END
        )
        self._reset_slotframe_counters()

    def indication_tx_cell_elapsed(self, cell, sent_packet):
        """Count the cell for the reward, then let MSF do its own bookkeeping."""
        preferred_parent = self.mote.rpl.getPreferredParent()
        if (
                preferred_parent
                and
                (cell.mac_addr == preferred_parent)
                and
                (cell.options == [d.CELLOPTION_TX])
            ):
            self.cells_elapsed += 1
            if sent_packet:
                self.cells_used += 1
        super(SchedulingFunctionRLSF, self).indication_tx_cell_elapsed(
            cell, sent_packet
        )

    def indication_queue_full(self):
        """The TX queue overflowed, which is the drop term of Eq. 3."""
        self.dropped_a_packet = True

    def _adapt_to_traffic(self, neighbor, cell_opt):
        """MSF's threshold rule, disabled.

        The whole point of RL-SF is that this decision is learned instead. MSF
        still counts elapsed and used cells around this call, and those counters
        are left running because nothing else reads them here.
        """
        pass

    # ======================= private =========================================

    # === the agent

    def _observe_state(self):
        """The state is the number of queued messages, saturating at the last row."""
        queued = len(self.mote.tsch.txQueue)
        return min(queued, self.NUM_QUEUE_STATES - 1)

    def _choose_action(self, state):
        """Epsilon-greedy over the row, with the exponential decay of Eq. 8."""
        if random.random() < self.epsilon:
            return random.randrange(self.MAX_CELLS)
        row = self.q_table[state]
        best = max(row)
        # ties are broken at random rather than by index, otherwise action 0
        # wins every tie and the untried actions of a fresh row never run
        return random.choice(
            [i for i, value in enumerate(row) if value == best]
        )

    def _update_q_table(self, state, action, reward, next_state):
        """Q <- (1-a)Q + a(r + B max Q'), the standard update. See the note above."""
        target = reward + self.BETA * max(self.q_table[next_state])
        self.q_table[state][action] = (
            (1 - self.ALFA) * self.q_table[state][action] +
            self.ALFA * target
        )

    # === the reward, Eq. 3 to Eq. 6 of the paper

    @staticmethod
    def _normalize(x):
        """N(x) = 2x - 1, Eq. 6, which puts a ratio on [-1, 1]."""
        return 2.0 * x - 1.0

    def _queue_utilization(self):
        """QU of Eq. 4, packets in the queue over the queue's size."""
        size = self.mote.tsch.txQueueSize
        if size in (0, float('inf')):
            return 0.0
        return min(1.0, len(self.mote.tsch.txQueue) / float(size))

    def _unused_cell_ratio(self):
        """UC of Eq. 5, cells that carried nothing over cells that came round.

        With no cell elapsed there is no waste to report, so this reads zero.
        That is the reading that does not punish a node for a slotframe in
        which its cells never came up.
        """
        if self.cells_elapsed == 0:
            return 0.0
        unused = self.cells_elapsed - self.cells_used
        return unused / float(self.cells_elapsed)

    def _compute_reward(self):
        """r = t1 N(1 - QU) + t2 N(1 - UC) - t3 PD, Eq. 3."""
        queue_term = self._normalize(1.0 - self._queue_utilization())
        unused_term = self._normalize(1.0 - self._unused_cell_ratio())
        drop_term = 1.0 if self.dropped_a_packet else 0.0
        return (
            self.THETA_QUEUE * queue_term +
            self.THETA_UNUSED * unused_term -
            self.THETA_DROP * drop_term
        )

    # === acting on the schedule

    def _num_negotiated_tx_cells(self, neighbor):
        return len([
            cell for cell in self.mote.tsch.get_cells(
                neighbor, self.SLOTFRAME_HANDLE_NEGOTIATED_CELLS
            ) if cell.options == [d.CELLOPTION_TX]
        ])

    def _apply_action(self, neighbor, state, action):
        """Negotiate the difference between what we hold and what we chose.

        The action names the cell count for the coming slotframe, so it is the
        gap that goes to 6P: "add" when the new count is higher than the old
        one, "remove" when it is lower, nothing when they agree.
        """
        target = action + 1                      # actions 0..n-1 are 1..n cells
        current = self._num_negotiated_tx_cells(neighbor)

        self._record_decision(state, action, target, current)

        if target == current:
            return
        if self.retry_count.get(neighbor, -1) != -1:
            # a 6P transaction is already in flight and 6P allows one at a time
            return

        self.retry_count[neighbor] = 0
        if target > current:
            self._request_adding_cells(
                neighbor     = neighbor,
                num_tx_cells = target - current
            )
        else:
            self._request_deleting_cells(
                neighbor     = neighbor,
                num_cells    = current - target,
                cell_options = self.TX_CELL_OPT
            )

    def _record_decision(self, state, action, target, current):
        self.decision_count += 1
        self.RLSF_STATS['EPSILON'][self.decision_count] = self.epsilon
        self.RLSF_STATS['DECISIONS'][self.decision_count] = {
            'state'   : state,
            'action'  : action,
            'target'  : target,
            'current' : current,
            'asn'     : self.engine.getAsn(),
        }

    def _reset_slotframe_counters(self):
        self.cells_elapsed = 0
        self.cells_used = 0
        self.dropped_a_packet = False
