import math
from math import cos, sin

import cv2
import mediapipe as mp
import numpy as np


def find_face(file):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        return
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        cv2.imwrite(str(file) + "_test" + '.png', annotated_image)
        return face_landmarks, annotated_image


def x_rotate(x, y, z, teta):
    new_y = y * cos(teta) - sin(teta) * z
    new_z = y * sin(teta) + z * cos(teta)
    return [x, new_y, new_z]


def x_rotate_c(x, y, z, c, s):
    new_y = y * c - s * z
    new_z = y * s + z * c
    return [x, new_y, new_z]


def y_rotate(x, y, z, teta):
    new_x = x * cos(teta) + sin(teta) * z
    new_z = - x * sin(teta) + y * cos(teta)
    return [new_x, y, new_z]


def y_rotate_c(x, y, z, c, s):
    new_x = x * c + s * z
    new_z = - x * s + y * c
    return [new_x, y, new_z]


def z_rotate(x, y, z, teta):
    new_x = x * cos(teta) - sin(teta) * y
    new_y = x * sin(teta) + y * cos(teta)
    return [new_x, new_y, z]


def z_rotate_c(x, y, z, c, s):
    new_x = x * c - s * y
    new_y = x * s + y * c
    return [new_x, new_y, z]


def find_axis(a, b):
    return [a.x - b.x, a.y - b.y, a.z - b.z]


def rotate(face_landmark, org, first_rotate):
    up = face_landmark.landmark[10]
    down = face_landmark.landmark[152]
    left = face_landmark.landmark[234]
    right = face_landmark.landmark[454]

    x_axis_train = find_axis(right, left)
    y_axis_train = find_axis(down, up)
    z_axis_train = np.cross(x_axis_train, y_axis_train)

    if first_rotate:
        cos_z = z_axis_train[2] / (math.sqrt(z_axis_train[0] ** 2 + z_axis_train[2] ** 2))
        sin_z = z_axis_train[0] / (math.sqrt(z_axis_train[0] ** 2 + z_axis_train[2] ** 2))
    else:
        cos_z = z_axis_train[2] / (math.sqrt(z_axis_train[1] ** 2 + z_axis_train[2] ** 2))
        sin_z = z_axis_train[1] / (math.sqrt(z_axis_train[1] ** 2 + z_axis_train[2] ** 2))

    new_land_mark = face_landmark.landmark
    number = 0

    for point in face_landmark.landmark:
        if first_rotate:
            result = y_rotate_c(point.x - org.x, point.y - org.y, point.z - org.z, cos_z, sin_z)
        else:
            result = x_rotate_c(point.x - org.x, point.y - org.y, point.z - org.z, cos_z, sin_z)
        new_land_mark[number].x = result[0] + org.x
        new_land_mark[number].y = result[1] + org.y
        new_land_mark[number].z = result[2] + org.z
        number += 1
    return face_landmark


def compare(train, test):
    imageHeight, imageWidth = image.shape[0], image.shape[1]
    org = train.landmark[0]
    return  rotate(train, org, True)
    # return rotate(land_train_1, org, False)
#     land_train_1 = rotate(train, org, True)


# import face_recognition
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FILES = ["stevejobs.jpg", "stevejobs_2.jpg"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    train_1, image = find_face(IMAGE_FILES[1])
    test_1, image_test = find_face(IMAGE_FILES[0])

    new_land = compare(train_1, test_1)

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=new_land,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=new_land,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

    cv2.imwrite('annotated_image' + '.png', image)
    # cv2.imshow('MediaPipe FaceMesh', annotated_image)
