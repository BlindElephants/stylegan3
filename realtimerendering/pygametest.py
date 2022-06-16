import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pickle
import torch
from copy import deepcopy

from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread, Lock
from typing import List

from realtimerendering.realtimeconsts import CLIENT_IP_ADDRESS, CLIENT_PORT, SERVER_PORT, SERVER_IP_ADDRESS, OUTSIZE, \
    ALL_MODEL_FILES_LIST, OUT_PERF_RECORDER_FILE

import datetime

mutex = Lock()

last_img = None
last_ws = None
receiveLatentBuffer = []

oscSender = udp_client.SimpleUDPClient(CLIENT_IP_ADDRESS, CLIENT_PORT)
ORIGINAL_MODEL_STATE_DICT = None

with open(ALL_MODEL_FILES_LIST[-1], 'rb') as f:
    model = pickle.load(f)['G_ema'].cuda()
    ORIGINAL_MODEL_STATE_DICT = deepcopy(model.state_dict())



model.eval()

LAST_LOADED_MODEL = ALL_MODEL_FILES_LIST[-1]
NORMALIZE_OUTPUT_IMAGE = False
LAYER_SELECT = None
CHANNEL_SELECT = 0
NUMBER_CHANNELS_IN_OUTPUT = 3

LATENT_X_COORD = 0.0
LATENT_Y_COORD = 0.0


def get_render_layers(addr, *args):
    global model, oscSender
    oscSender.send_message('/render_layers', model.synthesis.layer_names)


def set_render_layer(addr, *args):
    global model, LAYER_SELECT, receiveLatentBuffer, last_ws
    ln = model.synthesis.layer_names

    if args[0] >= len(ln):
        print("Error, invalid index received: out of range.")
        return

    if args[0] < 0:
        LAYER_SELECT = None
    else:
        LAYER_SELECT = ln[args[0]]
        print("Layer Selected for render: {}".format(LAYER_SELECT))

    if len(receiveLatentBuffer) == 0:
        receiveLatentBuffer.append(last_ws)


def revert_state_dict(addr, *args):
    global model, ORIGINAL_MODEL_STATE_DICT
    model.load_state_dict(deepcopy(ORIGINAL_MODEL_STATE_DICT))


def get_layer_names(t_model):
    all_layer_names = []
    sd = t_model.state_dict()
    for k, v in sd.items():
        all_layer_names.append(k)
    return all_layer_names


def get_available_layers(addr, *args):
    global model
    ln = get_layer_names(model)
    oscSender.send_message('/layer_names', ln)


def degrade_layer_by_name(t_model, layer_name: str, threshold: float = 0.1):
    global receiveLatentBuffer, last_ws
    sd = t_model.state_dict()
    t_values = sd[layer_name]
    s = t_values.shape
    print(s)

    degrade_picks = np.random.rand(s[0], s[1], s[2], s[3])
    degrade_picks = torch.tensor((degrade_picks >= threshold).astype(float)).cuda()
    t_values *= degrade_picks
    sd[layer_name] = t_values
    t_model.load_state_dict(sd)

    if len(receiveLatentBuffer) == 0:
        receiveLatentBuffer.append(last_ws)


def degrade_layer_by_idx(addr, *args):
    global model
    ln = get_layer_names(model)
    print(ln[args[0]])
    degrade_layer_by_name(model, ln[args[0]], threshold=float(args[1]))


def recv_degrade_layer_by_name(addr, *args):
    global model
    degrade_layer_by_name(model, args[0], threshold=float(args[1]))


def get_number_of_available_models(addr, *args):
    oscSender.send_message('/number_models', len(ALL_MODEL_FILES_LIST))


def get_available_model_names(addr, *args):
    oscSender.send_message('/clear_available_model_names', True)
    MAX_LEN = 200
    NUM_FULL_BATCHES = len(ALL_MODEL_FILES_LIST) // MAX_LEN
    REMAINDER = len(ALL_MODEL_FILES_LIST) % MAX_LEN

    for m in range(NUM_FULL_BATCHES):
        oscSender.send_message('/available_model_names', ALL_MODEL_FILES_LIST[m*MAX_LEN:(m*MAX_LEN) + MAX_LEN])

    if REMAINDER > 0:
        oscSender.send_message('/available_model_names', ALL_MODEL_FILES_LIST[MAX_LEN*NUM_FULL_BATCHES:])


def get_model_idx_by_name(addr, *args):
    tcount = 0
    out_models = []
    for x in ALL_MODEL_FILES_LIST:
        if args[0] in x:
            out_models.append(x)
            tcount += 1

    oscSender.send_message('/available_models_by_name', [tcount] + out_models)


def load_model_by_name(addr, *args):
    global model, ORIGINAL_MODEL_STATE_DICT, LAST_LOADED_MODEL
    for x in ALL_MODEL_FILES_LIST:
        if args[0] in x:
            print("Loading model {}".format(x))
            with open(x, 'rb') as f:
                model = pickle.load(f)['G_ema'].cuda()
                ORIGINAL_MODEL_STATE_DICT = deepcopy(model.state_dict())
            LAST_LOADED_MODEL = x
            model.eval()
            return

    print("No model found by name: {}".format(args[0]))


def load_model_by_index(addr, *args):
    global model, ORIGINAL_MODEL_STATE_DICT, receiveLatentBuffer, last_ws, LAST_LOADED_MODEL
    idx = args[0]
    if not isinstance(idx, int):
        print("Error, should receive an integer value for loadModelByIndex()")
        return
    if idx < 0 or idx >= len(ALL_MODEL_FILES_LIST):
        print("Error, invalid index received: out of range.")
        return
    print("Loading model index {}\t\t{}".format(idx, ALL_MODEL_FILES_LIST[idx]))

    with open(ALL_MODEL_FILES_LIST[idx], 'rb') as f:
        model = pickle.load(f)['G_ema'].cuda()
        ORIGINAL_MODEL_STATE_DICT = deepcopy(model.state_dict())

    LAST_LOADED_MODEL = ALL_MODEL_FILES_LIST[idx]
    model.eval()

    if len(receiveLatentBuffer) == 0:
        receiveLatentBuffer.append(last_ws)


def receive_latent_vector(addr, *args):
    global receiveLatentBuffer, model
    if len(args) != 513:
        print('Invalid latent vector size received: {}'.format(len(args)))
        return

    if bool(args[0]):
        receiveLatentBuffer.clear()
    x = np.array(args[1:]).reshape(1, 512)
    ws = model.mapping(torch.tensor(x).to('cuda'), c=None)
    receiveLatentBuffer.append(ws)


def go_to_latent_vector_over_frames(addr, *args):
    #  Args for this should be [clear_buffer: bool, duration_frames: int, latent_vec: float*512]
    global receiveLatentBuffer, last_ws, LAST_LOADED_MODEL, LAYER_SELECT, CHANNEL_SELECT

    clear_buffer = bool(args[0])
    duration_frames = int(args[1])
    latent_vec = args[2:]
    if len(latent_vec) != 512:
        print("ERROR")
        return

    t_last_ws = last_ws
    if clear_buffer:
        receiveLatentBuffer.clear()
    else:
        if len(receiveLatentBuffer):
            t_last_ws = receiveLatentBuffer[-1]

    latent_vec = np.array(latent_vec).reshape(1, 512)
    latent_vec = model.mapping(torch.tensor(latent_vec).to('cuda'), c=None)

    for x in range(duration_frames):
        receiveLatentBuffer.append(torch.lerp(t_last_ws, latent_vec, 1 / (duration_frames - 1) * x))

    print(f"Added {duration_frames} to receiveLatentBuffer, current size: {len(receiveLatentBuffer)}")

    with open(OUT_PERF_RECORDER_FILE, 'a') as f:
        out_line = [str(datetime.datetime.now()), LAST_LOADED_MODEL, str(LAYER_SELECT), str(CHANNEL_SELECT)]
        out_line = out_line + [str(i) for i in args]
        f.write(', '.join(out_line))
        f.write('\n')


def set_latent_coords(addr, *args):
    global receiveLatentBuffer, model
    receiveLatentBuffer.clear()
    ws = get_ws_from_xy_coords(model, get_seed(float(args[0]), float(args[1])))
    receiveLatentBuffer.append(ws)


def get_number_channels(addr, *args):
    global NUMBER_CHANNELS_IN_OUTPUT
    oscSender.send_message('/number_channels_in_output', [NUMBER_CHANNELS_IN_OUTPUT])


def set_channel_select(addr, *args):
    global CHANNEL_SELECT, receiveLatentBuffer, last_ws
    CHANNEL_SELECT = int(args[0])

    if CHANNEL_SELECT < 0:
        CHANNEL_SELECT = 0
    elif CHANNEL_SELECT > (NUMBER_CHANNELS_IN_OUTPUT - 3):
        CHANNEL_SELECT = NUMBER_CHANNELS_IN_OUTPUT - 3

    oscSender.send_message('/current_channel', [CHANNEL_SELECT])

    if len(receiveLatentBuffer) == 0:
        receiveLatentBuffer.append(last_ws)


def set_normalize_output_image(addr, *args):
    global NORMALIZE_OUTPUT_IMAGE, receiveLatentBuffer, last_ws
    NORMALIZE_OUTPUT_IMAGE = bool(args[0])

    if len(receiveLatentBuffer) == 0:
        receiveLatentBuffer.append(last_ws)


dispatcher = dispatcher.Dispatcher()
dispatcher.map('/get_number_models', get_number_of_available_models)  # no args
dispatcher.map('/load_model_by_index', load_model_by_index)  # [int]
dispatcher.map('/get_available_model_names', get_available_model_names)  # no args
dispatcher.map('/get_model_idx_by_name', get_model_idx_by_name)  # [str]
dispatcher.map('/load_model_by_name', load_model_by_name)  # [str]

dispatcher.map('/latent', receive_latent_vector)  # [float*512]

dispatcher.map('/get_available_degrade_layer_names', get_available_layers)  # no args
dispatcher.map('/degrade_layer_by_name', recv_degrade_layer_by_name)  # [str]
dispatcher.map('/degrade_layer_by_idx', degrade_layer_by_idx)

dispatcher.map('/revert_state_dict', revert_state_dict)  # no args

dispatcher.map('/get_render_layers', get_render_layers)  # no args
dispatcher.map('/set_render_layer', set_render_layer)  # [int]

dispatcher.map('/set_latent_coords', set_latent_coords)  # [float, float]

dispatcher.map('/get_number_channels', get_number_channels)  # no args
dispatcher.map('/set_channel_select', set_channel_select)  # [int]
dispatcher.map('/set_normalize_output_image', set_normalize_output_image)  # [bool]

dispatcher.map('/go_to_latent_vec', go_to_latent_vector_over_frames)

oscListener = osc_server.ThreadingOSCUDPServer((SERVER_IP_ADDRESS, SERVER_PORT), dispatcher)
oscListenerThread = Thread(target=oscListener.serve_forever, args=())
print("Starting OSC listener thread.")
oscListenerThread.start()

receiveLatentBuffer.clear()
receiveLatentBuffer.append(model.mapping(torch.tensor(np.random.rand(1, 512)).to('cuda'), c=None))


def get_seed(x: float, y: float) -> List:
    w0_seeds = []
    for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        seed_x = np.floor(x) + ofs_x
        seed_y = np.floor(y) + ofs_y
        seed = (int(seed_x) + int(seed_y) * 100) & ((1 << 32) - 1)
        weight = (1 - abs(x - seed_x)) * (1 - abs(y - seed_y))
        if weight > 0:
            w0_seeds.append([seed, weight])
    return w0_seeds


def get_ws_from_xy_coords(model, w0_seeds):
    all_seeds = [seed for seed, _weight in w0_seeds]
    all_seeds = list(set(all_seeds))
    all_zs = np.zeros([len(all_seeds), model.z_dim], dtype=np.float32)

    for idx, seed in enumerate(all_seeds):
        rnd = np.random.RandomState(seed)
        all_zs[idx] = rnd.randn(model.z_dim)
    w_avg = model.mapping.w_avg
    all_zs = torch.from_numpy(all_zs).to('cuda')

    all_ws = model.mapping(z=all_zs, c=None, truncation_psi=1.0, truncation_cutoff=0) - w_avg
    all_ws = dict(zip(all_seeds, all_ws))

    w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
    w += w_avg
    return w


def get_next_frame_from_ws_buffer() -> np.array:
    global LAYER_SELECT, model, NORMALIZE_OUTPUT_IMAGE, CHANNEL_SELECT, NUMBER_CHANNELS_IN_OUTPUT
    global receiveLatentBuffer, last_ws
    ws = receiveLatentBuffer.pop(0)
    oscSender.send_message('/frames_left_in_buffer', len(receiveLatentBuffer))

    last_ws = deepcopy(ws)

    if LAYER_SELECT is None:
        out_img = model.synthesis(ws)
    else:
        ws = ws.to(torch.float32).unbind(dim=1)
        x = model.synthesis.input(ws[0])

        for layer_name, w in zip(model.synthesis.layer_names, ws[1:]):
            x = getattr(model.synthesis, layer_name)(x, w)
            try:
                if LAYER_SELECT and (LAYER_SELECT in layer_name):
                    break
            except TypeError:
                pass
        out_img = x.to(torch.float32)
    out_img = (out_img.permute(0, 2, 3, 1) * 0.5 + 0.5019607843137255).clamp(0.0, 1.0)

    if NUMBER_CHANNELS_IN_OUTPUT != out_img.shape[3]:
        NUMBER_CHANNELS_IN_OUTPUT = out_img.shape[3]
        oscSender.send_message('/number_channels_in_output', [NUMBER_CHANNELS_IN_OUTPUT])

    if out_img.shape[3] > 3:
        out_img = out_img[:, :, :, CHANNEL_SELECT:CHANNEL_SELECT + 3]

    if NORMALIZE_OUTPUT_IMAGE:
        img_range = out_img.max() - out_img.min()
        out_img = (out_img - out_img.min()) / img_range

    out_img = out_img[0].detach().cpu().numpy()
    return out_img


if __name__ == "__main__":
    img = None

    pygame.init()
    pygame.display.set_caption('ha-realtime-output')
    pygame.display.set_mode(OUTSIZE, DOUBLEBUF | OPENGL | pygame.NOFRAME)
    pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)

    glMatrixMode(GL_PROJECTION)
    glOrtho(0, OUTSIZE[0], OUTSIZE[1], 0, 1, -1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_TEXTURE_2D)

    senderTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, senderTexture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)


    def update():
        global img, last_img

        if len(receiveLatentBuffer) > 0:
            img = get_next_frame_from_ws_buffer()
            last_img = img.copy()

        glBindTexture(GL_TEXTURE_2D, senderTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[0], img.shape[1], 0, GL_RGB, GL_FLOAT, img)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord(0, 0)
        glVertex2f(0, 0)
        glTexCoord(1, 0)
        glVertex2f(OUTSIZE[0], 0)
        glTexCoord(1, 1)
        glVertex2f(OUTSIZE[0], OUTSIZE[1])
        glTexCoord(0, 1)
        glVertex2f(0, OUTSIZE[1])
        glEnd()
        pygame.display.flip()
        glBindTexture(GL_TEXTURE_2D, 0)


    while True:
        try:
            update()
        except AttributeError:
            pass

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    oscListener.server_close()
                    oscListener.shutdown()
                    oscListenerThread.join(timeout=2.0)
                    quit()
