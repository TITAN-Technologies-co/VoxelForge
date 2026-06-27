from pyqtgraph.Qt import QtCore

from .config import logger


class VFWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class VFWorker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = VFWorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            logger.exception("Background task failed")
            self.signals.failed.emit(str(exc))

