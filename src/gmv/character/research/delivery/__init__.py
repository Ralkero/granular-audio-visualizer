"""Phase A4: can the audio tell Cleans, High Screams and Low Screams apart? Research only.

An offline experiment on the owner's labelled excerpts (Phase A3). Nothing in the
application imports it; it draws nothing and changes no setting, session or visual.

* ``data``      leakage-safe reading of the answers: the split is taken from the dataset
                before any answer is read, and a held-out answer's value is never decoded
* ``features``  a fixed set of audio measurements of a ten-second excerpt
* ``models``    a majority baseline, regularised logistic regression and a random forest,
                in numpy, with every learned transformation fitted on training data only
* ``evaluate``  leave-one-artist-group-out validation, metrics, abstention
* ``run``       the whole experiment, as ``gmv character research delivery``

Only three answers are supported: Cleans, High Screams and Low Screams. Gutturals and
Sing-Screams have too few examples from too few vocalists to be learned, and are left out,
never mapped onto a supported answer. A model that has learned three answers cannot
recognise a fourth: its "abstain" outcome means "not confident among the three", nothing
more.

Findings: docs/VOCAL_DELIVERY_RECOGNITION.md.
"""

SUPPORTED = ("cleans", "high_screams", "low_screams")
UNSUPPORTED = ("gutturals", "sing_screams")
NAMES = {"cleans": "Cleans", "high_screams": "High Screams", "low_screams": "Low Screams"}
VERSION = "a4.1"
