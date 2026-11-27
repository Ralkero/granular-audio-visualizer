"""A toy job for the timing-harness tests: it only sleeps, in marked phases.

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
    time.sleep(0.02)
marks.write()
