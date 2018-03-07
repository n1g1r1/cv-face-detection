#######################################################
# Face detection application
# @author: Christian Reichel
# Version: 0.1
# -----------------------------------------------------
# Detects faces in an image with pretrained
# classifiers by OpenCV and shows the result with
# bounding boxes in a window.
#######################################################

# IMPORTS
import numpy as np
import cv2 as cv

from modules.face_detector import face_detector as detector

# Some settings and parameters.
boundary_box_thickness = 2

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()

while output and cv.waitKey(1) == -1:    
    
    # Call the detector with lbp for fast recognition.
    # We also let the algorithm resize the image for faster recognition.
    faces, eyes, camera_image = detector.detect_faces(camera_image, detect_eyes = False, classifier = "lbp", draw_bounding_box = True, resize = True, resize_factor = 0.5)

    if len(faces) is not 0:
        # Show the result. 
        cv.imshow('FaceDetector', camera_image)

    # Grab next camera image.
    output, camera_image = camera.read()

# End procedures.
camera.release()
cv.destroyAllWindows()