import time
import queue

class Scheduler:
    def __init__(self, max_batch_size=4, max_wait=0.01):
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait

    def batch(self, q: queue.Queue):
        if q.empty():
            time.sleep(0.0005)
            return []

        items = [q.get()]
        start = time.time()
        while len(items) < self.max_batch_size:
            if not q.empty():
                items.append(q.get())
            if time.time() - start >= self.max_wait:
                break
        return items
