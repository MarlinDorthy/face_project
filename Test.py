import cv2

print("OpenCV Version:", cv2.__version__)
try:
    print("Face module methods:", dir(cv2.face))
except AttributeError:
    print("cv2.face module is still unavailable.")
