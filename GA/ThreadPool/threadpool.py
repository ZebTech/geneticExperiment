# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from threading import Thread, Condition


class Worker():
    def __init__(self, pool):
        self.thread = Thread(target=self.work)
        self.pool = pool

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def work(self):
        while True:
            self.pool.lock.acquire()
            while not self.pool.tasks:
                if self.pool.exit:
                    return
                self.pool.lock.wait()
            task = self.pool.tasks.pop()
            self.pool.lock.release()
            task()


class ThreadPool():
    def __init__(self, tasks=[]):
        self.workers = [Worker(self) for i in xrange(cpu_count())]
        self.tasks = tasks
        self.lock = Condition()
        self.exit = False

    def notify(self):
        self.lock.notifyAll()

    def start(self):
        [w.start() for w in self.workers]

    def join(self):
        self.lock.acquire()
        self.exit = True
        self.notify()
        self.lock.release()

    def add(self, task):
        self.lock.acquire()
        self.tasks.append(task)
        self.notify()
        self.lock.release()

# TODO: The problem is that with multiprocessing, you need to share the values
#       accross the processes. That's a pain for now.
# class ThreadPool():
#    def __init__(self, tasks=[]):
#        self.pool = Pool(processes=cpu_count() - 1)
#        self.tasks = tasks
#        self.exit = False
#        self.lock = Condition()
#        self.thread = Thread(target=self.check_empty)
#
#    def add(self, task):
#        self.lock.acquire()
#        self.tasks.append(task)
#        self.lock.notify()
#        self.lock.release()
#
#    def start(self):
#        self.thread.start()
#
#    def check_empty(self):
#        while not self.exit:
#            self.lock.acquire()
#            if not self.tasks:
#                self.lock.wait()
#            t = self.tasks.pop()
#            self.lock.release()
#            print self.pool.apply_async(t).get()
#
#    def join(self):
#        self.lock.acquire()
#        self.exit = True
#        self.lock.notify()
#        self.lock.release()
#        self.thread.join()
