import signal

class signal_handler:

    def __init__(self):
        signal.signal(signal.SIGTERM, self.handler)
        signal.signal(signal.SIGINT, self.handler)
        self.handlerList = []

    def addHandler(self, handler):
        self.handlerList.append(handler)

    def handler(self, signal, frame):
        print 'Captured signal: %d' % signal
        for h in self.handlerList:
            h()
        exit(0)
