from threading import Thread
import sys
import cv2
from queue import Queue
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
		
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False		    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self		
		
    def update(self):
        while True:	
            if self.stopped:
                return
				
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True



class FileVideoStream:
    def __init__(self, path, queueSize=30):
	
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)		
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)
    def read(self):
        return self.Q.get()
		# return next frame in the queue
    def more(self):
        return self.Q.qsize() > 0
		# return True if there are still frames in the queue
    def stop(self):
        self.stopped = True
		# indicate that the thread should be stopped
		
		
		
					
					
				
				
				
			
			
				
				
				
				
			
			
		
		
		
		
		
    
		
		
		
	
		
		
	
		
		