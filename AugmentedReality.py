import cv2
import cv2.aruco as aruco
import numpy as np
import os #needed to access directory containing actual marker pictures

def main():
    webcamCapture = cv2.VideoCapture(0)

    while True:
        success, webcamFeed = webcamCapture.read()
        cv2.imshow("Webcam Feed", webcamFeed)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
