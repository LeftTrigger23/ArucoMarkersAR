import cv2
import cv2.aruco as aruco
import numpy as np
import os #needed to access directory containing actual marker pictures

# arg2 -> marker size is 6x6
def findArucoMarkers(webcamFeed, markerSize = 6, totalMarkersAvailable = 250, draw = True):
    # change webcamFeed to greyscale
    webcamFeedGrey = cv2.cvtColor(webcamFeed, cv2.COLOR_BGR2GRAY)

    # getattr uses the argument value provided in markerSize and totalMarkersAvailable to be used by arucoDict
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkersAvailable}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    boundingBoxes, foundMarkersIds, rejectedMarkers = aruco.detectMarkers(webcamFeedGrey, arucoDict, parameters = arucoParam)
    # print(foundMarkersIds)

    if draw:
        aruco.drawDetectedMarkers(webcamFeed, boundingBoxes)


def main():
    webcamCapture = cv2.VideoCapture(0)

    while True:
        success, webcamFeed = webcamCapture.read()
        findArucoMarkers(webcamFeed)
        cv2.imshow("Webcam Feed", webcamFeed)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
