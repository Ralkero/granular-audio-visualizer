"""Real-world excerpt review: choose recordings from a local inventory, cut short excerpts,
run the existing model over the full tracks, and collect human labels in a review window.

Human labels and model output are kept in separate files and fields; the model never fills
in or chooses a label. All audio and labels stay in a local, Git-ignored work folder.
"""
