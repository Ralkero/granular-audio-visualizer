"""Play a recorded session's messages onto a private EventBus for the visualizer.

Nothing is analysed again and no audio is involved: the logged messages are published
once more with their original audio timestamps and marked ``replayed``.
"""

import json
from pathlib import Path
from threading import Event, Thread

from ..bus import EventBus
from ..session import replay


class ReplaySession:
    def __init__(self, path, speed=1.0):
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / "messages.jsonl"
        if not self.path.is_file():
            raise ValueError(f"No recorded messages at {self.path}")
        self.speed = speed
        self.bus = EventBus()
        self.stop_event = Event()
        self.error = None
        self.configuration = {}
        manifest = self.path.with_name("manifest.json")
        if manifest.is_file():
            try:
                self.configuration = json.loads(manifest.read_text(encoding="utf-8")).get(
                    "configuration", {}
                )
            except (OSError, ValueError):
                pass
        self.thread = Thread(target=self._run, name="visualizer-replay", daemon=True)

    @property
    def title(self):
        return self.path.parent.name

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            replay(self.path, self.bus, self.stop_event, speed=self.speed)
        except Exception as exc:
            self.error = str(exc)

    @property
    def finished(self):
        return self.thread.ident is not None and not self.thread.is_alive()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
