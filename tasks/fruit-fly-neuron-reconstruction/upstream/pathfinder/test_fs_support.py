"""Bounded cleanup retries for test-owned directories on X temp's NFS."""
import errno
import gc
import tempfile
import time


class TemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        for attempt in range(10):
            try:
                super().cleanup()
                return
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt==9:raise
                gc.collect()
                time.sleep(.1)
