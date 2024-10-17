import threading
import time
from enum import Enum

class ThreadedCapture:

    class States(Enum):
        UNINITIALIZED = 1
        STARTED = 2
        STOPPED = 3

    def __init__(self, cap):

        self.cap = cap

        self.state = ThreadedCapture.States.UNINITIALIZED
        self.capture_thread = None

        self.frame = None
        self.ret = False

        self.timestamp = None

    def start(self):

        print("starting capture...")
        if self.state == ThreadedCapture.States.STARTED:
            return

        self.capture_thread = threading.Thread(target=self.capture_loop, args=())
        self.capture_thread.start()
        
        while not self.state == ThreadedCapture.States.STARTED:
            time.sleep(0.2)

    def stop(self):

        self.state = ThreadedCapture.States.STOPPED
        if not self.capture_thread is None:
            self.capture_thread.join()
            self.capture_thread = None

        # if not self.cap is None:
        #    print("releasing video...")
        #    self.cap.release()

    def capture_loop(self):

        try:
            first_time = True
            while self.state != ThreadedCapture.States.STOPPED:

                self.timestamp = time.time()
                self.ret, self.frame = self.cap.read()

                if first_time and self.frame is not None:
                    self.state = ThreadedCapture.States.STARTED
                    first_time = False
                

                time.sleep(0.01)

        except Exception as ex:

            print("error capturing: " + str(ex))

        self.state = ThreadedCapture.States.STOPPED
        self.capture_thread = None
        print("capturing end.")

    def read(self):

        return self.ret, self.frame
