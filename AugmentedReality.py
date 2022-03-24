import cv2
import cv2.aruco as aruco
import numpy as np
import os #needed to access directory containing actual marker pictures

def loadAugmentedImages(path):
    listOfImagesToAugment = os.listdir(path)
    numberOfMarkers = len(listOfImagesToAugment)
    print("Total number of markers detected - ", numberOfMarkers)

    # create a dictionary to store key -> id, and value -> image to augment for specific key
    markersAndImages = dict()
    for imgPath in listOfImagesToAugment:
        # split filename into the file name (1st element) and the extension (2nd element)
        # we are grabbing the file name which is a aruco id number
        key =  int(os.path.splitext(imgPath)[0])
        imgToAugment = cv2.imread(f"{path}/{imgPath}")
        markersAndImages[key] = imgToAugment
    
    return markersAndImages

def augmentImageOntoAruco(boundingBox, foundMarkersIds, webcamFeed, imgToAugment, drawId = True):
    
    # 4 corner points
    topLeft = boundingBox[0][0][0], boundingBox[0][0][1]
    topRight = boundingBox[0][1][0], boundingBox[0][1][1]
    bottomRight = boundingBox[0][2][0], boundingBox[0][2][1]
    bottomLeft = boundingBox[0][3][0], boundingBox[0][3][1]

    # get size of imageToAugment
    height, width, channels = imgToAugment.shape

    # -------------------------following steps are to warp image------------------------------
    points1 = np.array([topLeft, topRight, bottomRight, bottomLeft])
    # first array/element in 2d array is representation of top left
    # second array/element in 2d array is representation of top right
    # third array/element in 2d array is representation of bottom right
    # fourth array/element in 2d array is represenation of bottom left
    points2 = np.float32([[0,0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(points2, points1)
    
    augmentImg = cv2.warpPerspective(imgToAugment, matrix, (webcamFeed.shape[1], webcamFeed.shape[0]))

    # arg3 is color and 0,0,0 is black
    cv2.fillConvexPoly(webcamFeed, points1.astype(int), (0,0,0))

    augmentImg += webcamFeed

    return augmentImg



# arg2 -> marker size is 6x6
def findArucoMarkers(webcamFeed, markerSize = 6, totalMarkersAvailable = 250, draw = True):
    # change webcamFeed to greyscale
    webcamFeedGrey = cv2.cvtColor(webcamFeed, cv2.COLOR_BGR2GRAY)

    # getattr uses the argument value provided in markerSize and totalMarkersAvailable to be used by arucoDict
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkersAvailable}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    boundingBoxes, foundMarkersIds, rejectedMarkers = aruco.detectMarkers(webcamFeedGrey, arucoDict, parameters = arucoParam)

    if draw:
        aruco.drawDetectedMarkers(webcamFeed, boundingBoxes)
    
    return [boundingBoxes, foundMarkersIds]


def main():
    webcamCapture = cv2.VideoCapture(0)
    augmentedIdsAndImages = loadAugmentedImages("Images")


    while True:
        success, webcamFeed = webcamCapture.read()

        # a list is returned, 0th element is the bounding boxes, 1st element contains all of the Ids
        foundArucoMarkers = findArucoMarkers(webcamFeed)

        # loop through all the foundArucoMarkers and augment an image onto each of the marker
        # if length of the 1st element (bounding boxes) is 0, then we did not detect anything
        if len(foundArucoMarkers[0]) != 0:

            # looping through each boundingBox and markerId -> using zip function
            for boundingBox, markerId in zip(foundArucoMarkers[0], foundArucoMarkers[1]):
                if int(markerId) in augmentedIdsAndImages.keys():
                    webcamFeed = augmentImageOntoAruco(boundingBox, markerId, webcamFeed, augmentedIdsAndImages[int(markerId)])

        cv2.imshow("Webcam Feed", webcamFeed)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

# 36:00