"""A toy job for the timing-harness tests: it sleeps in marked phases, except for a short busy
loop in its prediction phase, so that its CPU time is visibly non-zero yet smaller than its
elapsed time.

    python timing_toy_job.py <marks.json> <gap-seconds>

A positive gap adds an unmarked pause between setup and training, which the accounting
must report as "other".
"""

import sys
import time

from gmv.character.research.presence_model.timing import Marks

marks = Marks(sys.argv[1])
gap = float(sys.argv[2])
with marks.phase("setup"):
    time.sleep(0.05)
time.sleep(gap)
for epoch in range(4):
    with marks.phase("epoch"):
        time.sleep(0.03 if epoch == 0 else 0.02)
with marks.phase("gate_eval"):
    time.sleep(0.01)
with marks.phase("prediction"):
    end = time.process_time() + 0.05
    while time.process_time() < end:
        pass
marks.write()
