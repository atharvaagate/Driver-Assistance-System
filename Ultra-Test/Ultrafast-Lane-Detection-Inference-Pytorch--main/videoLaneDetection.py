import cv2
#import pafy
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/2CIxM7x-Clc'
#videoPafy = pafy.new(videoUrl)
#print(videoPafy.streams)
cap = cv2.VideoCapture("../../Test_Videos/Cali-road.mp4")
print("hh")
# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
print("jj")

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

while cap.isOpened():
	print("ins")
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()