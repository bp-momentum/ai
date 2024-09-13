from hashlib import sha256
import json
import random
from urllib import request
import socketio
from aiohttp import web
import os
from threading import Lock
import base64
import cv2
from PIL import Image
from io import BytesIO
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis
from scipy.signal import correlate
from scipy.signal import correlation_lags

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# =========== Env Data ===========
# BACKEND_URL = "http://localhost:80" // used to return feedback and fetch exercises
# BACKEND_PSK = "secret"              // used to authenticate with the backend


# ========= Session Data =========
# {
#    current_repetition: int,        // [0..n]
#    exercise_id: string,            // global id of the exercise
#    video: {                        // map of repetitions to frames
#        0: [frame1, frame2, ...],   // base64 encoded frames
#        1: [frame1, frame2, ...], 
#        ...
#    }
#    set_uuid: string,               // id for the evaluation
# }

custom_session_lock = Lock()
custom_session_data = {}

BACKEND_URL = os.environ.get("BACKEND_URL")
BACKEND_PSK = os.environ.get("BACKEND_PSK")

if BACKEND_URL is None or BACKEND_PSK is None:
    raise Exception("🛑 BACKEND_URL and BACKEND_PSK must be configured!")

RATING_URL = BACKEND_URL + "/api/internal/rate"
EXERCISE_URL = BACKEND_URL + "/api/getexercise"

@sio.event
async def connect(sid, _):
    with custom_session_lock:
        custom_session_data[sid] = {}
        custom_session_data[sid]['current_repetition'] = 0
    print(sid, 'connected')



@sio.event
async def disconnect(sid):
    # no need to lock here as this session is read-only
    session = custom_session_data[sid]

    # check that the session is valid and a set ended
    if session.get('set_uuid') is None:
        print("No set_uuid received, aborting...")
        return
    
    values = {
        "speed": 0,
        "accuracy": 0,
        "cleanliness": 0,
    }

    # TODO: fix this
    if session.get('video') is None:
        print("No video data received, aborting...")
        return

    for repetition in session['video']:

        frames = session['video'][repetition]
        
        ################################
        # setting mediapipe parameters #
        ################################
        worldmarks = False
        med_par = []
        # set static_image_mode (default: False); set to True if person detector is supposed to be done on every image (e.g. unrelated images instead of video); setting this to True leads to ignoring smooth_landmarks, smooth_segmentation and min_tracking_confidence
        med_par.append(False)
        # set model_complexity (default: 1, possible 0-2)
        med_par.append(1)
        # set smooth_landmarks (default: True); filters landmark positions; overruled by static image mode
        med_par.append(True)
        # set enable_segmentation (default: False); would also return segmentation mask additional to landmarks; Overruled, when entropy is given to process_video
        med_par.append(False)
        # set smooth_segmentation (default: True); filters segmentation mask; ignored when segmentation not enabled or static_image_mode=True
        med_par.append(True)
        # set min_detection_confidence (default: 0.5); Minimum confidence value from the person-detection model for the detection to be considered successful
        med_par.append(0.5)
        # set min_tracking_confidence (default: 0.5); Minimum confidence value from the landmark-tracking model for the pose landmarks to be considered tracked successfully
        # otherwise: person detection will be invoked on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency
        med_par.append(0.4)

        expectation_df = pd.DataFrame(session['expectation'])
        if expectation_df.shape[1] == 132:
            vis = True
        else:
            vis = False

        # print(len(session['expectation'][0])) # 146x33

        with mp_pose.Pose(static_image_mode=med_par[0],
                      model_complexity=med_par[1],
                      smooth_landmarks=med_par[2],
                      enable_segmentation=med_par[3],
                      smooth_segmentation=med_par[4],
                      min_detection_confidence=med_par[5],
                      min_tracking_confidence=med_par[6],                   
                     ) as pose:
            coordinates_list = []
            for frame in frames:
                im = Image.open(BytesIO(base64.b64decode((frame.split(','))[1])))
                image = cv2.cvtColor(np.array(im),cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                xyz_list = []
                if (results.pose_world_landmarks != None and worldmarks):
                    for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                        xyz_list.append(landmark.x)
                        xyz_list.append(landmark.y)
                        xyz_list.append(landmark.z)
                        if vis:
                            xyz_list.append(landmark.visibility)
                elif (results.pose_landmarks != None and not worldmarks):
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        xyz_list.append(landmark.x)
                        xyz_list.append(landmark.y)
                        xyz_list.append(landmark.z)
                        if vis:
                            xyz_list.append(landmark.visibility)
                coordinates_list.append(xyz_list)
                im.save('results.jpg')
        output_df = pd.DataFrame(coordinates_list)
        body_parts = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", 
        "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]
        column_names = []
        for i in body_parts:
            column_names.append(i+".x")
            column_names.append(i+".y")
            column_names.append(i+".z")
            if vis:
                column_names.append(i+".visibility")
        output_df = output_df.set_axis(column_names,axis=1)
        
        tmp = []
        cols = list(expectation_df.columns)
        for col in cols:
            tmp.append(expectation_df[col].apply(pd.Series))
        expectation_df = pd.concat(tmp,axis=1, ignore_index=True)
        expectation_df = expectation_df.set_axis(column_names,axis=1)

        ###################
        # Synchronisation #
        ###################
        signal_length = expectation_df.shape[0]
        ys_out = output_df.filter(like='.y')
        movement_out = (-ys_out.sub(ys_out.max(axis=1),axis=0)).diff().sum(axis=1)
        q75_out,q25_out = np.percentile(movement_out,[75,25]) # outlier removal
        intr_qr_out = q75_out-q25_out
        movement_out.loc[movement_out<q75_out-15*intr_qr_out] = np.nan
        movement_out.loc[movement_out>q75_out+15*intr_qr_out] = np.nan
        movement_out.fillna(method='bfill',inplace=True)
        ys_exp = expectation_df.filter(like='.y')
        movement_exp = (-ys_exp.sub(ys_exp.max(axis=1),axis=0)).diff().sum(axis=1)
        q75_exp,q25_exp = np.percentile(movement_exp,[75,25]) # outlier removal
        intr_qr_exp = q75_exp-q25_exp
        movement_exp.loc[movement_exp<q75_exp-15*intr_qr_exp] = np.nan
        movement_exp.loc[movement_out>q75_exp+15*intr_qr_exp] = np.nan
        movement_exp.fillna(method='bfill',inplace=True)
        standardized_out = (movement_out-movement_out.mean())/movement_out.std()
        standardized_exp = (movement_exp-movement_exp.mean())/movement_exp.std()
        correlation = correlate(standardized_out, standardized_exp, mode="full")
        lags = correlation_lags(standardized_out.size, standardized_exp.size, mode="full")
        offset = lags[np.argmax(correlation)]
        print(offset)
        if offset < 0:
            output_df = output_df.iloc[:output_df.shape[0]+offset][:]
            expectation_df = expectation_df.iloc[abs(offset):][:]
        else:
            output_df = output_df.iloc[offset:][:]
            expectation_df = expectation_df.iloc[:expectation_df.shape[0]-offset][:]
        # breakpoint()

        #############################
        # Actual Movement Analysis: #
        #############################
        # sets of joints so they are logical groups:
        head = list(range(0,11))
        torso = [11,12,23,24]
        left_arm = [13,15,17,19,21]
        right_arm = [14,16,18,20,22]
        left_leg = [25,27,29,31]
        right_leg = [24,26,28,30]
        limbs = [head,torso,left_arm,right_arm,left_leg,right_leg]
        # first DTW:
        # breakpoint()
        md = dtw_ndim.distance_fast(np.array(output_df, dtype=np.double),np.array(expectation_df, dtype=np.double))
        dm = dtw_ndim.warping_path(np.array(output_df, dtype=np.double),np.array(expectation_df, dtype=np.double))
        # best_mpath = dtw.best_path(mpaths)
        # dtwvis.plot_warping(output_df,expectation_df,best_mpath,filename='test.png')
        
        # with open('test.jpg', 'wb') as file_to_save:
        #     file_to_save.write(base64.b64decode(frames[0]))
        # print(type(frame))
        # cv2.imshow('test',frames[0]) # 146
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        values['speed'] += (signal_length-abs(offset))/signal_length # basically rhythm
        values['accuracy'] += md # how close are you to the gold standard
        values['cleanliness'] += sum([d[0]-d[1] for d in dm]) # how do you differ from the gold standard
        breakpoint()
    
    values['speed'] /= len(session['video'])
    values['accuracy'] /= len(session['video'])
    values['cleanliness'] /= len(session['video'])

    checksum = sha256(f"{session['set_uuid']}{BACKEND_PSK}".encode()).hexdigest()
    data = {
        "set_uuid": session['set_uuid'],
        "values": values,
        "checksum": checksum,
    }

    # send the data to the backend
    data = json.dumps(data).encode()
    req = request.Request(RATING_URL, data=data, headers={'Content-Type': 'application/json'})
    request.urlopen(req)

    print(sid, 'disconnected')


@sio.event
async def set_exercise_id(sid, data):
    with custom_session_lock:
        custom_session_data[sid]['exercise_id'] = data['exercise']
    print(f"{sid} started {data['exercise']}")
    # load exercise data from backend
    data = json.dumps({"id": data['exercise']}).encode()
    req = request.Request(EXERCISE_URL, data=data, headers={'Content-Type': 'application/json'})
    response = request.urlopen(req)
    exercise = response.read().decode()
    exercise = json.loads(exercise)
    ex_data = exercise.get('data')
    if ex_data is None:
        print("⚠️ No exercise data received, aborting...")
        return
    expectation = ex_data.get('expectation')
    # breakpoint()
    if expectation is None:
        print("⚠️ No exercise expectation received, aborting...")
        return
    with custom_session_lock:
        custom_session_data[sid]['expectation'] = expectation


@sio.event
async def end_repetition(sid, _):
    current_rep = None
    with custom_session_lock:
        custom_session_data[sid]['current_repetition'] += 1
        current_rep = custom_session_data[sid]['current_repetition']
    print(f"{sid} ended repetition {current_rep}")


@sio.event
async def end_set(sid, data):
    set_uuid = None
    with custom_session_lock:
        custom_session_data[sid]['set_uuid'] = data['set_uuid']
        set_uuid = custom_session_data[sid]['set_uuid']
    print(f"{sid} ended set {set_uuid}")


@sio.event
async def send_video(sid, data):
    # every time this get's called,
    # a new frame of the video is received
    # data is a base64 encoded string
    with custom_session_lock:
        session = custom_session_data[sid]
        if session.get('video') is None:
            session['video'] = {}
        if session['video'].get(session['current_repetition']) is None:
            session['video'][session['current_repetition']] = []
        session['video'][session['current_repetition']].append(data)

    # some live analysis is possible here, like so:
    # sio.emit('live_feedback', {'feedback': '👍'}, room=sid)
    # BUT: keep the processing time low, as this is blocking
    # TODO: figure out a nice datatype for the feedback

    if random.random() < 0.01:
        await sio.emit('live_feedback', {'feedback': '👍'}, room=sid)


if __name__ == '__main__':
    web.run_app(app, port=80)