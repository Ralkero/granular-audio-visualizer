"""Phase A5.1: time-localised vocal-presence annotation and a leakage-resistant dataset
foundation. Research tooling; nothing in the application imports it, and no model is trained
or evaluated here.

* ``schema``    intervals (confirmed or uncertain), review states, confounder tags; how an
                excerpt's events fold into a state, its status and frame targets
* ``store``     the append-only, hash-chained annotation history of one partition
* ``identity``  conservative artist groups (credits, aliases, MusicBrainz IDs, duplicates)
* ``split``     the append-only partition registry: groups never move
* ``project``   the presence project's folders, manifests and fingerprints
* ``loaders``   protected reading: development tools cannot read final-test labels
* ``selection`` deterministic candidate batches, never chosen by a model
* ``window``    the annotation window; ``cli``: ``gmv character presence``

Presence and delivery are separate: an interval says only that a human voice is audible.
The Phase A3 vocal-delivery project is read, never written.
"""
