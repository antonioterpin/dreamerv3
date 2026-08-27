"""Provide limiters functionality."""

import threading
import time


def wait(predicate, message, info=None, sleep=0.01, notify=60):
    """Wait for state.

    Args:
        predicate: Predicate value.
        message: Message value.
        info: Info value.
        sleep: Sleep value.
        notify: Notify value.

    Returns:
        Result of the operation.
    """
    if predicate():
        return 0
    start = last_notify = time.time()
    while not predicate():
        now = time.time()
        if now - last_notify > notify:
            dur = now - start
            print(f"{message} {dur:.1f}s: {info}")
            last_notify = time.time()
        time.sleep(sleep)
    return time.time() - start


class SamplesPerInsert:
    """Represent samples per insert."""

    def __init__(self, samples_per_insert, tolerance, minsize):
        """Initialize the samples per insert.

        Args:
            samples_per_insert: Samples per insert value.
            tolerance: Tolerance value.
            minsize: Minsize value.
        """
        assert 1 <= minsize, "Expected 1 to be at most minsize."
        self.samples_per_insert = samples_per_insert
        self.minsize = minsize
        self.avail = -minsize
        self.min_avail = -tolerance
        self.max_avail = tolerance * samples_per_insert
        self.size = 0
        self.lock = threading.Lock()

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return {"size": self.size, "avail": self.avail}

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        self.size = data["size"]
        self.avail = data["avail"]

    def want_insert(self):
        # if self.samples_per_insert <= 0 or self.size < self.minsize:
        #   return True, 'ok'
        # if self.avail >= self.max_avail:
        #   return False, f'rate limited: {self.avail:.3f} >= {self.max_avail:.3f}'
        # return True, 'ok'
        """Handle want insert.

        Returns:
            Result of the operation.
        """
        if self.size < self.minsize:
            return True
        if self.samples_per_insert <= 0:
            return True
        if self.avail < self.max_avail:
            return True
        return False

    def want_sample(self):
        # if self.size < self.minsize:
        #   return False, f'too empty: {self.size} < {self.minsize}'
        # if self.samples_per_insert > 0 and self.avail <= self.min_avail:
        #   return False, f'rate limited: {self.avail:.3f} <= {self.min_avail:.3f}'
        # return True, 'ok'
        """Handle want sample.

        Returns:
            Result of the operation.
        """
        if self.size < self.minsize:
            return False
        if self.samples_per_insert <= 0:
            return True
        if self.min_avail < self.avail:
            return True
        return False

    def insert(self):
        """Handle insert."""
        with self.lock:
            self.size += 1
            if self.size >= self.minsize:
                self.avail += self.samples_per_insert

    # def remove(self):
    #   with self.lock:
    #     self.size -= 1

    def sample(self):
        """Sample state."""
        with self.lock:
            self.avail -= 1
