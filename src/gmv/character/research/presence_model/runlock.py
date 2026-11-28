"""Step 2a, S2a.8: the one exclusive lock every GMV experiment and benchmark runner takes
(protocol Amendment 3, A3.8, C1). Built here because the step 2 and 3 runners need it; step 4's
runners reuse it.

* **The same exclusive lock for everyone:** an operating-system ``flock(LOCK_EX)`` (never a shared
  or read lock) on one fixed file outside every working copy, so runners started from different
  folders still compete for it.
* **Non-blocking:** a second runner refuses at once with a fixed message; it never queues.
* **Held for the whole experiment** by the launcher, and released by the operating system when the
  holder exits, even after a crash. The lock's descriptor is close-on-exec, so no job inherits it.
* ``REGISTERED`` names every runner entry point that must take it; a test checks each one.
"""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

REGISTERED = (
    "gmv.character.research.presence_model.step2run:run",
    "gmv.character.research.presence_model.step3run:run",
)


class LockHeld(RuntimeError):
    """Another GMV experiment or benchmark holds the lock; this runner refuses to start."""


def default_path():
    """The fixed lock file: under the user's state folder, outside every working copy."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "gmv" / "experiment.lock"


@contextmanager
def exclusive(path=None):
    """Hold the experiment lock for the duration of the block, or raise ``LockHeld``."""
    path = Path(path) if path is not None else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockHeld("another GMV experiment or benchmark is running; this one refuses to start") from None
        yield fd
    finally:
        os.close(fd)  # closing the descriptor releases the lock
