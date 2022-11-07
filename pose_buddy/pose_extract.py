import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

def extract(image):
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True

        try:

            raw_points = results.pose_landmarks.landmark
            all_landmarks = mp_pose.PoseLandmark

            image_metadata = {}
            for i, feature in enumerate(all_landmarks):
                point = {}
                point["x"] = raw_points[feature.value].x
                point["y"] = raw_points[feature.value].y
                point["z"] = raw_points[feature.value].z
                point["visibility"] = raw_points[feature.value].visibility
                image_metadata[str(i)] = point



        except:
            image_metadata = None
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                         )      
        return image, image_metadata   

# # For scaling images to same size
# def generate_vectors(image_metadata):
#     vectors = {}
#     with open("./feature_mapping.vectors", "r") as vector_map:
#         for combination in vector_map:
#             combination_array = combination.replace("\n","").split(" ")
#             vectors[combination_array[0]] = calculate_vectors(image_metadata[combination_array[1]], image_metadata[combination_array[2]])
#     return vectors

# For scaling images to same size and push vectors as array
def generate_vectors(image_metadata):
    vectors = []
    with open("./feature_mapping.vectors", "r") as vector_map:
        for combination in vector_map:
            combination_array = combination.replace("\n","").split(" ")
            vectors.append(calculate_vectors(image_metadata[combination_array[1]], image_metadata[combination_array[2]]))
    return vectors


# Calculate vectors used for similariry matching
def calculate_vectors(joint1, joint2):
    # joint_11 = (x_11, y_11, z_11)
    # joint_12 = (x_12, y_12, z_12)
    # vector_1 = (x_11 - x_12, y_11 - y_12, z_11 - z_12) = (x_vector, y_vector, z_vector)
    # length_vector_1 = sqrt(x_vector^2 + y_vector^2 + z_vector^2)
    # normalized_vector_1 = vector_1/length_vector_1

    vector = {}

    x1 = joint1['x']
    y1 = joint1['y']
    z1 = joint1['z']
    v1 = joint1['visibility']
    
    x2 = joint2['x']
    y2 = joint2['y']
    z2 = joint2['z']
    v2 = joint2['visibility']
    
    v_new = (v1+v2) / 2

    x_new = (x1-x2)
    y_new = (y1-y2)
    z_new = (z1-z2)
    from_2_to_1_vector = [x_new, y_new, z_new]
    vector_length = np.sqrt((x_new ** 2) + (y_new ** 2) + (z_new ** 2))
    
    unit_vector = from_2_to_1_vector/vector_length

    vector["x"] = unit_vector[0]
    vector["y"] = unit_vector[1]
    vector["z"] = unit_vector[2]
    
    vector["visibility"] = v_new
    return vector

def paint_points(image):
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                         )   
        return image

    


