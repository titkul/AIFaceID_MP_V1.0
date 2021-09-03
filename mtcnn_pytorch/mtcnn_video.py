from src import detect_faces, show_bboxes

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import cv2
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        bounding_boxes, landmarks = detect_faces(frame, min_face_size=10.0)
        # print(bounding_boxes)
        # print("hjsh")
        # print(landmarks)
        frame = show_bboxes(frame, bounding_boxes, landmarks)
        frame = np.asarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
