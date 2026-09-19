"""
Watches an output directory and reports the newest checkpoint it has not reported yet.
"""

from __future__ import annotations

import glob
import os

from PySide6 import QtCore


class LatestCheckpointWatcher(QtCore.QObject):
    """
    Polls one directory tree on a timer and emits `found` with the newest CSV or VTK checkpoint each time that file changes.

    The watcher remembers the last path it reported, so it emits once per new file. A caller that loads a file some other way calls `note_loaded` to keep that memory in step, or the next poll reports a file the viewer already shows.
    """

    found = QtCore.Signal(str)

    def __init__(self, interval_ms: int, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._directory = ""
        self._last_reported: str | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll)

    def set_directory(self, directory: str) -> None:
        self._directory = directory

    def set_interval(self, interval_ms: int) -> None:
        self._timer.setInterval(interval_ms)

    def set_enabled(self, enabled: bool) -> None:
        """
        Start or stop the timer. Stopping does not clear the memory of the last reported file, so switching the watcher back on does not re-report a file already on screen.
        """

        if enabled:
            self._timer.start()
        else:
            self._timer.stop()

    def note_loaded(self, path: str) -> None:
        self._last_reported = path

    def poll(self) -> None:
        """
        Look once, now, whatever the timer is doing. The window calls this at startup and after the user changes a setting, so a result already on disk appears without a wait.
        """

        if not self._directory or not os.path.isdir(self._directory):
            return
        latest = self._find_latest_file(self._directory)
        if latest and latest != self._last_reported:
            self._last_reported = latest
            self.found.emit(latest)

    @staticmethod
    def _find_latest_file(directory: str) -> str | None:
        vtr_files = glob.glob(os.path.join(directory, "**", "Checkpoint_*.vtr"), recursive=True)
        csv_files = glob.glob(os.path.join(directory, "**", "Checkpoint_*.csv"), recursive=True)
        candidate_files = vtr_files + csv_files
        if not candidate_files:
            return None

        # Take the newest, but prefer a .vtr over a .csv the same run wrote within a second of it.
        # "Within a second of the newest" is not an ordering, so it selects a group first and sorts inside it.
        stamped = [(os.path.getmtime(path), path) for path in candidate_files]
        newest = max(mtime for mtime, _ in stamped)
        recent = [entry for entry in stamped if newest - entry[0] <= 1.0]
        # Prefer .vtr files (priority 0) over .csv files (priority 1)
        recent.sort(key=lambda entry: (0 if entry[1].lower().endswith('.vtr') else 1, -entry[0]))
        return recent[0][1]
