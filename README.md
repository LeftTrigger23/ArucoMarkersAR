Background Information
--------------------------
This is an Augmented Reality application, developed using OpenCV with Aruco Markers which allows the user to augment Images onto the detected Aruco Markers (https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) to generate a real time augmented projection on a moving obect

Installation requirements
--------------------------
- Install git
- Git clone the repository -> `git clone https://github.com/SweekarShrestha/ArucoMarkersAR.git`
- Install Python, version 3.8.3 or later (https://www.python.org/downloads/)
- Install pip3, version 19.2.3 or later (https://pip.pypa.io/en/stable/installation/)
- Install opencv with aruco marker -> `pip3 install opencv-contrib-python`
- Allow Python camera usage from system preferences 
- Modify opencv camera code if configured with multiple cameras
- Run command -> `python3 AugmentedReality.py`
- Show aruco markers on the camera feed
- Selected images to agument should overlap the markers detected
