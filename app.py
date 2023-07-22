import cv2
import mediapipe as mp
import pandas as pd
import time
import tensorflow.lite as tflite
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# URL for the Parquet file
PARQUET_URL = 'https://raw.githubusercontent.com/pavannn29/ASL/main/data/captured.parquet'
# URL for train.csv
TRAIN_CSV_URL = 'https://raw.githubusercontent.com/pavannn29/ASL/main/data/train.csv'
# URL for the TFLite model
TFLITE_MODEL_URL = 'https://raw.githubusercontent.com/pavannn29/ASL/main/models/asl_model.tflite'

def create_frame_landmark_df(results, frame, xyz):
    """
    Takes the results from mediapipe and creates a dataframe of the landmarks
    """
    # for having values and rows for every landmark index
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')

    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')

    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')

    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    # So that skel will have landmarks even if they do not exist
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    # to have actual unique frames
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def do_capture_loop(xyz, duration):
    all_landmarks = []
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    return all_landmarks

def load_relevant_data_subset(pq_path):
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_prediction(prediction_fn, pq_url):
    xyz_np = load_relevant_data_subset(pq_url)
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    sign = ORD2SIGN[pred]
    pred_conf = prediction['outputs'][pred]
    return sign, pred_conf

@app.route('/capture', methods=['POST'])
def capture_signs():
    duration = int(request.form.get('duration', 5))
    xyz = pd.read_parquet(PARQUET_URL)
    landmarks = do_capture_loop(xyz, duration=duration)
    captured_file = 'captured.parquet'
    pd.concat(landmarks).reset_index(drop=True).to_parquet(captured_file)
    sign, confidence = get_prediction(prediction_fn, captured_file)
    return jsonify({'predicted_sign': sign, 'confidence': confidence})

if __name__ == "__main__":
    # Load the TFLite model
    interpreter = tflite.Interpreter(TFLITE_MODEL_URL)
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")

    # Load the training data and create dictionaries for sign encoding/decoding
    train = pd.read_csv(TRAIN_CSV_URL)
    train['sign_ord'] = train['sign'].astype('category').cat.codes
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    app.run(debug=True)
