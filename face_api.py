# importing libraries
import streamlit as st
import mediapipe as mp
import pandas as pd
import numpy as np
import tempfile
import skimage.io
import math
import time
import cv2


class Config():
    def __init__(self):
        self.keypoints_mapping = {
            "left_ear": 234,
            "right_ear": 454,
            "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
                         158, 159, 160, 161, 246],
            "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                          388, 387, 386, 385, 384, 398],
            "jaw_line": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377,
                         400, 378, 379, 365, 397, 288, 361, 323, 454],
            "upper_head": [10, 8],
            "middle_head": [8, 1],
            "bottom_head": [164, 152],
            "left_ear_to_nose": [234, 5],
            "nose_to_right_ear": [5, 454],
            "left_eyebrow": 107,
            "right_eyebrow": 336,
            "left_corner_of_mouth": 61,
            "right_corner_of_mouth": 291,
            "upper_lip_to_lower_lip": [0, 17],
            "left_eye_cord": [159, 145],
            "right_eye_cord": [386, 374]
        }

    @st.cache
    def extractLocation(self, position):
        """
            This method will return coordinate or list of coordinates for
            given position.
            Args:
                Input:
                    position = position for which coordinates needed to be
                        return
                Return:
                    cooridnate = int/list: coordinates
        """
        return self.keypoints_mapping[position]


class ExtractCharacteristics():
    """
        This class will extract 8 characteristics from image and store it in
        Dictionary
    """

    def __init__(self, imageHeight=1024, imageWidth=1024,
                 circleColor=(0, 0, 255), lineColor=(0, 255, 0),
                 thickness=3):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.circleColor = circleColor
        self.lineColor = lineColor
        self.thickness = thickness

    def extractCoordinates(self, results, imageIdx, landmarkNumber):
        """
            This method will extract x and y coordinate of keypoint from
            mediapipe facemesh model.
            Args:
                Input:
                    landmark_number = int: number of keypoint which needs to be extracted
                Return:
                    [x, y] = list: list of x and y coordinates
        """
        x = int(results.multi_face_landmarks[imageIdx].landmark[landmarkNumber].x * self.imageWidth)
        y = int(results.multi_face_landmarks[imageIdx].landmark[landmarkNumber].y * self.imageHeight)
        return [x, y]

    @st.cache
    def drawCircle(self, image, location):
        """
            This method will draw circle on given image for given location.
            Args:
                Input:
                    image = array: image containing face
                    location = list: list of x and y coordinates
                Return:
                    None
        """
        cv2.circle(image, location, 5, self.circleColor, -1)

    @st.cache
    def drawPolylines(self, image, pts):
        """
            This method will draw multiple lines on given image for given points.
            Args:
                Input:
                    image = array: image containing face
                    pts = list: list of pts having x and y coordinates
                Return:
                    None
        """
        cv2.polylines(image, [pts], isClosed=False, color=self.lineColor, thickness=self.thickness)

    @st.cache
    def drawArrow(self, image, start, end):
        """
            This method will draw Arrow lines on given image for given points.
            Note: we need arrow on both side so we have used it for twice
            Args:
                Input:
                    image = array: image containing face
                    start = list: list of x and y coordinates of starting position
                    end = list: list of x and y coordinates of ending position
                Return:
                    None
        """
        cv2.arrowedLine(image, start, end, self.lineColor, self.thickness)
        cv2.arrowedLine(image, end, start, self.lineColor, self.thickness)

    @st.cache
    def findDistance(self, start, end):
        """
            This method will return Euclidean distance for given starting and
            ending points.
            Args:
                Input:
                    start = list: list of x and y coordinates of starting position
                    end = list: list of x and y coordinates of ending position
                Return:
                    distance = int: Euclidean distance
        """
        x1, y1 = start
        x2, y2 = end
        return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)

    @st.cache
    def findDistancePoly(self, pts):
        """
            This method will return Euclidean distance having multiple x and y
            coordinates
            Args:
                Input:
                    pts = list: list of pts having x and y coordinates
                Return:
                    jaw_distance = int: Euclidean distance
        """
        jawDistance = 0
        for i in range(len(pts) - 1):
            jawDistance += self.findDistance(pts[i], pts[i + 1])
        return jawDistance

    @st.cache
    def angleTrunc(self, a):
        """
            This is helper function for finding angle between two points.
            Args:
                Input:
                    a = arc tangent of (deltaY/deltaX) in radians.
                Return:
                    a = truncated angle
        """
        while a < 0.0:
            a += math.pi * 2
        return a

    @st.cache
    def findAngle(self, start, end):
        """
            This method will compute angle between two points and return in degees.
            Args:
                Input:
                    start = list: list of x and y coordinates of starting position
                    end = list: list of x and y coordinates of ending position
                Return:
                    angle = angle in degrees
        """
        x1, y1 = start
        x2, y2 = end
        return math.degrees(self.angleTrunc(math.atan2((y2 - y1), (x2 - x1))))

    @st.cache
    def process(self, results, image, imageIdx):
        """
            This method will process image and find characteristics, compute
            distance and angle and return dictionaries.
            Args:
                Input:
                    image = input image
                Return:
                    distanceDict = dictionary having all characteristics distances
                    angleDict = dictionary having all characteristics angle
        """
        characteristicsImage = image.copy()
        keypointsImage = image.copy()
        distanceDict = {}
        angleDict = {}

        configObject = Config()
        for keypoint in range(468):
            location = self.extractCoordinates(results, imageIdx, landmarkNumber=keypoint)
            self.drawCircle(keypointsImage, location)

        # ear to ear
        leftEar = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_ear"))
        rightEar = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_ear"))
        self.drawCircle(characteristicsImage, leftEar)
        self.drawCircle(characteristicsImage, rightEar)
        self.drawArrow(characteristicsImage, leftEar, rightEar)
        earEarDistance = self.findDistance(leftEar, rightEar)
        earEarAngle = self.findAngle(leftEar, rightEar)
        distanceDict["Ear_to_Ear_distance"] = earEarDistance
        angleDict["Ear_to_Ear_angle"] = earEarAngle

        # eye to eye
        leftEye1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[3])
        leftEye2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[5])
        leftEye3 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[11])
        leftEye4 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[13])
        rightEye1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[3])
        rightEye2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[5])
        rightEye3 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[11])
        rightEye4 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[13])
        leftEye = [(leftEye1[0] + leftEye2[0] + leftEye3[0] + leftEye4[0]) // 4,
                   (leftEye1[1] + leftEye2[1] + leftEye3[1] + leftEye4[1]) // 4]
        rightEye = [(rightEye1[0] + rightEye2[0] + rightEye3[0] + rightEye4[0]) // 4,
                    (rightEye1[1] + rightEye2[1] + rightEye3[1] + rightEye4[1]) // 4]
        self.drawCircle(characteristicsImage, leftEye)
        self.drawCircle(characteristicsImage, rightEye)
        self.drawArrow(characteristicsImage, leftEye, rightEye)
        eyeEyeDistance = self.findDistance(leftEye, rightEye)
        eyeEyeAngle = self.findAngle(leftEye, rightEye)
        distanceDict["eye_to_eye_distance"] = eyeEyeDistance
        angleDict["eye_to_eye_angle"] = eyeEyeAngle

        # eyebrow center to chin
        leftEyebrow = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eyebrow"))
        rightEyebrow = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eyebrow"))
        chin = self.extractCoordinates(results, imageIdx, configObject.extractLocation("bottom_head")[1])
        centerEyebrow = [(leftEyebrow[0] + rightEyebrow[0]) // 2, (leftEyebrow[1] + rightEyebrow[1]) // 2]
        self.drawCircle(characteristicsImage, centerEyebrow)
        self.drawCircle(characteristicsImage, chin)
        self.drawArrow(characteristicsImage, centerEyebrow, chin)
        eyebrowChinDistance = self.findDistance(centerEyebrow, chin)
        eyebrowChinAngle = self.findAngle(centerEyebrow, chin)
        distanceDict["eyebrow_to_chin_distance"] = eyebrowChinDistance
        angleDict["eyebrow_to_chin_angle"] = eyebrowChinAngle

        # jawline
        pts = []
        for item in configObject.extractLocation("jaw_line"):
            current = self.extractCoordinates(results, imageIdx, item)
            pts.append(current)
        pts = np.array(pts)
        self.drawPolylines(characteristicsImage, pts)
        jawlineDistance = self.findDistancePoly(pts)
        distanceDict["jawline_distance"] = jawlineDistance

        # left mouth cornet to right mouth corner
        leftMouth = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_corner_of_mouth"))
        rightMouth = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_corner_of_mouth"))
        self.drawCircle(characteristicsImage, leftMouth)
        self.drawCircle(characteristicsImage, rightMouth)
        self.drawArrow(characteristicsImage, leftMouth, rightMouth)
        leftRightMouthDistance = self.findDistance(leftMouth, rightMouth)
        left_right_mouth_angle = self.findAngle(leftMouth, rightMouth)
        distanceDict["left_right_mouth_distance"] = leftRightMouthDistance
        angleDict["left_right_mouth_angle"] = left_right_mouth_angle

        # virtual line
        yAdded = int((5 * characteristicsImage.shape[1]) / 100)
        xAdded = int((10 * characteristicsImage.shape[1]) / 100)
        leftVirtualline = [leftEar[0] - xAdded, chin[1] + yAdded]
        rightVirtualline = [rightEar[0] + xAdded, chin[1] + yAdded]
        self.drawCircle(characteristicsImage, leftVirtualline)
        self.drawCircle(characteristicsImage, rightVirtualline)
        self.drawArrow(characteristicsImage, leftVirtualline, rightVirtualline)
        virtuallineDistance = self.findDistance(leftVirtualline, rightVirtualline)
        virtuallineAngle = self.findAngle(leftVirtualline, rightVirtualline)
        distanceDict["virtualline_distance"] = virtuallineDistance
        angleDict["virtualline_angle"] = virtuallineAngle

        # left eyeball
        leftEye1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[12])
        leftEye2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[4])
        leftEyeballHorizontalDistance = self.findDistance(leftEye1, leftEye2)
        leftEyeballHorizontalAngle = self.findAngle(leftEye1, leftEye2)
        distanceDict["left_eyeball_horizontal_distance"] = leftEyeballHorizontalDistance
        angleDict["left_eyeball_horizontal_angle"] = leftEyeballHorizontalAngle
        leftEye3 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[3])
        leftEye4 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[5])
        leftEye5 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[11])
        leftEye6 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye")[13])
        centerLeftEye1 = [(leftEye3[0] + leftEye6[0]) // 2, (leftEye3[1] + leftEye6[1]) // 2]
        centerLeftEye2 = [(leftEye4[0] + leftEye5[0]) // 2, (leftEye4[1] + leftEye5[1]) // 2]
        leftEyeballVerticalDistance = self.findDistance(centerLeftEye1, centerLeftEye2)
        leftEyeballVerticalAngle = self.findAngle(centerLeftEye1, centerLeftEye2)
        distanceDict["left_eyeball_vertical_distance"] = leftEyeballVerticalDistance
        angleDict["left_eyeball_vertical_angle"] = leftEyeballVerticalAngle

        rightEye1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[12])
        rightEye2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[4])
        rightEyeballHorizontalDistance = self.findDistance(rightEye1, rightEye2)
        rightEyeballHorizontalAngle = self.findAngle(rightEye1, rightEye2)
        distanceDict["right_eyeball_horizontal_distance"] = rightEyeballHorizontalDistance
        angleDict["right_eyeball_horizontal_angle"] = rightEyeballHorizontalAngle
        rightEye3 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[3])
        rightEye4 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[5])
        rightEye5 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[11])
        rightEye6 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye")[13])
        centerRightEye1 = [(rightEye3[0] + rightEye6[0]) // 2, (rightEye3[1] + rightEye6[1]) // 2]
        centerRightEye2 = [(rightEye4[0] + rightEye5[0]) // 2, (rightEye4[1] + rightEye5[1]) // 2]
        rightEyeballVerticalDistance = self.findDistance(centerRightEye1, centerRightEye2)
        rightEyeballVerticalAngle = self.findAngle(centerRightEye1, centerRightEye2)
        distanceDict["right_eyeball_vertical_distance"] = rightEyeballVerticalDistance
        angleDict["right_eyeball_vertical_angle"] = rightEyeballVerticalAngle

        # uppperhead
        upperHead1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("upper_head")[0])
        upperHead2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("upper_head")[1])
        upperHeadDistance = self.findDistance(upperHead1, upperHead2)
        upperHeadAngle = self.findAngle(upperHead1, upperHead2)
        distanceDict["upperhead_distance"] = upperHeadDistance
        angleDict["upperhead_angle"] = upperHeadAngle

        # middlehead
        middleHead1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("middle_head")[0])
        middleHead2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("middle_head")[1])
        middleHeadDistance = self.findDistance(middleHead1, middleHead2)
        middleHeadAngle = self.findAngle(middleHead1, middleHead2)
        distanceDict["middlehead_distance"] = middleHeadDistance
        angleDict["middlehead_angle"] = middleHeadAngle

        # bottomhead
        bottomHead1 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("bottom_head")[0])
        bottomHead2 = self.extractCoordinates(results, imageIdx, configObject.extractLocation("bottom_head")[1])
        bottomHeadDistance = self.findDistance(bottomHead1, bottomHead2)
        bottomHeadAngle = self.findAngle(bottomHead1, bottomHead2)
        distanceDict["bottomhead_distance"] = bottomHeadDistance
        angleDict["bottomhead_angle"] = bottomHeadAngle

        # leftear_nose
        leftEar = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_ear_to_nose")[0])
        nose = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_ear_to_nose")[1])
        leftEarNoseDistance = self.findDistance(leftEar, nose)
        leftEarNoseAngle = self.findAngle(leftEar, nose)
        distanceDict["left_ear_nose_distance"] = leftEarNoseDistance
        angleDict["left_ear_nose_angle"] = distanceDict

        # rightear_nose
        rightEar = self.extractCoordinates(results, imageIdx, configObject.extractLocation("nose_to_right_ear")[0])
        nose = self.extractCoordinates(results, imageIdx, configObject.extractLocation("nose_to_right_ear")[1])
        rightEarNoseDistance = self.findDistance(rightEar, nose)
        rightEarNoseAngle = self.findAngle(rightEar, nose)
        distanceDict["right_ear_nose_distance"] = rightEarNoseDistance
        angleDict["right_ear_nose_angle"] = rightEarNoseAngle

        # upper lip to lower lip
        topLip = self.extractCoordinates(results, imageIdx, configObject.extractLocation("upper_lip_to_lower_lip")[0])
        bottomLip = self.extractCoordinates(results, imageIdx,
                                            configObject.extractLocation("upper_lip_to_lower_lip")[1])
        lipToLipDistance = self.findDistance(topLip, bottomLip)
        lipToLipAngle = self.findAngle(topLip, bottomLip)
        distanceDict["lip_to_lip_distance"] = lipToLipDistance
        angleDict["lip_to_lip_angle"] = lipToLipAngle

        # left eye
        leftEyeTop = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye_cord")[0])
        leftEyeBottom = self.extractCoordinates(results, imageIdx, configObject.extractLocation("left_eye_cord")[1])
        leftEyeToLeftEyeDistance = self.findDistance(leftEyeTop, leftEyeBottom)
        leftEyeToLeftEyeAngle = self.findAngle(leftEyeTop, leftEyeBottom)
        distanceDict["lefteye_to_lefteye_distance"] = leftEyeToLeftEyeDistance
        angleDict["lefteye_to_lefteye_angle"] = leftEyeToLeftEyeAngle

        # right eye
        rightEyeTop = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye_cord")[0])
        rightEyeBottom = self.extractCoordinates(results, imageIdx, configObject.extractLocation("right_eye_cord")[1])
        rightEyeToRightEyeDistance = self.findDistance(rightEyeTop, rightEyeBottom)
        rightEyeToRightEyeAngle = self.findAngle(rightEyeTop, rightEyeBottom)
        distanceDict["righteye_to_righteye_distance"] = rightEyeToRightEyeDistance
        angleDict["righteye_to_righteye_angle"] = rightEyeToRightEyeAngle

        return distanceDict, angleDict, characteristicsImage, keypointsImage


def main(maxNumberOfFace=5, minDetectionConfidence=0.7, imageHeight=1024, imageWidth=1024):
    st.title('CKM VIGIL Face API')
    st.subheader("Facial Landmark Detection")
    st.write(
        'CKM VIGIL Face API is solution that estimates 468 3D face landmarks in real-time. It only requires a simple face image.')

    with st.sidebar:
        st.warning(
            "Please upload two images of SINGLE-person. For best results, please also CENTER the person in the image.")
        app_mode = st.selectbox("Please select from the following",
                                ["Show Company Info", "Upload an image", "start live video", "Show Project Info"])

    if app_mode == "Show Company Info":
        st.write("Check our more projects on [ckmvigil.in/project](https://ckmvigil.in/projects)")

    elif app_mode == "Show Project Info":
        st.write("Coming soon!")

    elif app_mode == "start live video":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        stframe = st.empty()
        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

        fps = 0
        i = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi2, kpi3 = st.beta_columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_faces=3) as face_mesh:
            prevTime = 0

            while vid.isOpened():
                i += 1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                face_count = 0
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>",
                                unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.resize(frame, (640, 640))
                stframe.image(frame, channels='BGR', use_column_width=True)

        st.text('Video Processed')

        vid.release()
    else:
        optionsSelected = st.sidebar.multiselect('Which result you want to get',
                                                 ["Keypoints", "Characteristics", "Distance", "Angle", "Mouth Position",
                                                  "Eyes Position"])
        st.subheader("Please upload your image")
        uploadedFile = st.file_uploader("Choose Two images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if len(uploadedFile) > 0:
            cols = st.beta_columns(len(uploadedFile))
            for i, image in enumerate(uploadedFile):
                if image is not None:

                    # setting config for facemesh
                    mp_face_mesh = mp.solutions.face_mesh
                    mp_face_detection = mp.solutions.face_detection
                    with mp_face_detection.FaceDetection(
                            min_detection_confidence=0.5) as face_detection:

                        input_image = skimage.io.imread(image)
                        # checking if image is channeld properly
                        if input_image.shape[2] > 3:
                            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
                        if len(input_image.shape) == 2:
                            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
                        # resizing image to given width and height
                        input_image = cv2.resize(input_image, (imageHeight, imageWidth))
                        cols[i].image(input_image, caption="Original Image {}".format(i + 1), use_column_width=True,
                                      clamp=True)
                        faceDetectionImage = input_image.copy()
                        faceMeshImage = input_image.copy()
                        faceDetectionResults = face_detection.process(faceDetectionImage)
                        if faceDetectionResults.detections:
                            with mp_face_mesh.FaceMesh(
                                    static_image_mode=True,
                                    max_num_faces=maxNumberOfFace,
                                    min_detection_confidence=minDetectionConfidence) as face_mesh:
                                faceMeshResults = face_mesh.process(faceMeshImage)
                            characteristicsObject = ExtractCharacteristics(imageHeight=imageHeight,
                                                                           imageWidth=imageWidth)
                            for imageIdx in range(len(faceMeshResults.multi_face_landmarks)):
                                distanceDict, angleDict, characteristicsImage, keypointsImage = characteristicsObject.process(
                                    faceMeshResults, faceMeshImage, imageIdx)
                                if "Keypoints" in optionsSelected:
                                    cols[i].image(keypointsImage, caption='Image {} with Keypoints'.format(i + 1))
                                if "Characteristics" in optionsSelected:
                                    cols[i].image(characteristicsImage,
                                                  caption="Image {} with Characteristics".format(i + 1))
                                if "Distance" in optionsSelected:
                                    st.subheader("The distance between key points in image {}".format(i + 1))
                                    distanceDataframe = pd.DataFrame(distanceDict, index=["distance"])
                                    distanceDataframe.T
                                if "Angle" in optionsSelected:
                                    st.subheader("The angle between key points in image {}".format(i + 1))
                                    angleDataframe = pd.DataFrame(angleDict, index=["angle"])
                                    angleDataframe.T
                                if ("Mouth Position" in optionsSelected):
                                    if len(uploadedFile) == 1:
                                        st.subheader("Please upload the second image also, for getting results.")
                                    if len(faceMeshResults.multi_face_landmarks) > 2:
                                        st.subheader("This feature is not available for multiple face in images")

                                if ("Eyes Position" in optionsSelected):
                                    if len(uploadedFile) == 1:
                                        st.subheader("Please upload the second image also, for getting results.")
                                    if len(faceMeshResults.multi_face_landmarks) > 2:
                                        st.subheader("This feature is not available for multiple face in images")
                        else:
                            st.error("Sorry Face is not visible here please try other image")


if __name__ == "__main__":
    main()