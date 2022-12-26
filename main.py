import math
import random
import time
import socketio
from aiohttp import web

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

@sio.event
def connect(sid, _):
    print(sid, 'connected')


@sio.event
def disconnect(sid):
    print(sid, 'disconnected')


@sio.event
async def set_exercise_id(sid, data):
    session = await sio.get_session(sid)
    session['exercise_id'] = data
    print(session['exercise_id'])
    await sio.save_session(sid, session)


@sio.event
async def send_video(sid, data):
    # every time this get's called,
    # a new frame of the video is received
    # data is a base64 encoded string
    session = await sio.get_session(sid)
    if session.get('video') is None:
        session['video'] = []
    session['video'].append(data)
    if (session.get('last_analyzed') is None): 
        session['last_analyzed'] = time.time()
    await sio.save_session(sid, session)
    
    # asyncronously analyse the video here

    time_since_last = time.time() - session['last_analyzed']
    # 1 in 100 change
    if random.randint(0, 100) == 0:
        # send a random error
        error_list = [
            {
                "en": "No person could be found in livestream",
                "de": "Es konnte keine Person im video gefunden werden",
            },
            {
                "en": "It seems to be too dark",
                "de": "Es scheint zu dunkel zu sein",
            },
            {
                "en": "It is not possible recognize an execution",
                "de": "Es konnte keine Ausführung einer Übung erkannt werden",
            },
        ]
        await sio.emit('information', random.choice(error_list), to=sid)
    elif time_since_last > (random.random() * 2 + 2):  # 2 to 4 seconds
        session['video'] = []  # debatable
        session['last_analyzed'] = time.time()

        feedback = {
            "stats": {
                "intensity": math.ceil(random.random() * 10000) / 200 + 50,
                "speed": math.ceil(random.random() * 10000) / 200 + 50,
                "cleanliness": math.ceil(random.random() * 10000) / 200 + 50,
            },
            "coordinates": {
                "x": random.randint(0, 100),
                "y": random.randint(0, 100),
            },
        }
        print("send stats")
        await sio.emit('statistics', feedback, to=sid)

if __name__ == '__main__':
    web.run_app(app, port=80)