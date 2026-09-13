# -*- coding: utf-8 -*-
"""The scenario a run happens in, and the point its parameters depart from.

Both were born in runSensitivity, and both are now needed by the factorial and
by the final comparison. Importing them from there made runExperiments and
runSensitivity import each other, so they live on their own: whichever run
declares a disturbed scenario declares the same one, and whichever run reads
an anchor reads it the same way.
"""
from __future__ import division
from __future__ import print_function

import json


# The two disturbances the probe of 2026-09-09 validated, in the positions it
# measured: 40% of the run for the agent to settle before the first, and 30%
# after the last to measure the recovery. Together they cost the published
# DynQ about 0.077 of score, ten runs out of ten, and the delivery ratio
# recovers in nine of ten runs taking about 1600 slotframes.
DISTURBANCES = [
    {'asn_fraction': 0.40, 'kind': 'traffic', 'factor': 0.5},
    {'asn_fraction': 0.70, 'kind': 'link_quality', 'share': 0.2, 'pdr': 0.4},
]


def load_anchor(path):
    """Per learner, the settings a run departs from, as {learner: {KEY: value}}.

    A coordinate sweep measures sensitivity around one point, so the point has
    to be the one being defended, and the runs that come after the sweep have
    to depart from what it chose. Read as a mapping and applied before an
    arm's own overrides, so an arm that deliberately sets a core parameter
    still wins.
    """
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)
