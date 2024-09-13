from hashlib import sha256
import json
import random
from urllib import request
import socketio
from aiohttp import web
import os
from threading import Lock

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# =========== Env Data ===========
# BACKEND_URL = "http://localhost:80"  // used to return feedback and fetch exercises
# BACKEND_PSK = "secret"               // used to authenticate with the backend


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
        # TODO: analyze the video here (for now, it's just some random numbers)
        values['speed'] += random.randint(50, 100)
        values['accuracy'] += random.randint(50, 100)
        values['cleanliness'] += random.randint(50, 100)
    
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