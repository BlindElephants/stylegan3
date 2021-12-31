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

from realtimerendering.realtimeconsts import CLIENT_IP_ADDRESS, CLIENT_PORT, SERVER_PORT, SERVER_IP_ADDRESS, OUTSIZE, \
    ALL_MODEL_FILES_LIST


mutex = Lock()
receiveLatentDirty = False
receiveLatent = []

oscSender = udp_client.SimpleUDPClient(CLIENT_IP_ADDRESS, CLIENT_PORT)

ORIGINAL_MODEL_STATE_DICT = None

with open(ALL_MODEL_FILES_LIST[-1], 'rb') as f:
    model = pickle.load(f)['G_ema'].cuda()
    ORIGINAL_MODEL_STATE_DICT = deepcopy(model.state_dict())

model.eval()


LAYER_SELECT = None

LATENT_X_COORD = 0.0
LATENT_Y_COORD = 0.0


def get_render_layers(addr, *args):
    global model, oscSender
    oscSender.send_message('/render_layers', model.synthesis.layer_names)


def set_render_layer(addr, *args):
    global model, LAYER_SELECT, receiveLatentDirty
    ln = model.synthesis.layer_names

    if args[0] < 0:
        LAYER_SELECT = None
        receiveLatentDirty = True
    elif args[0] >= len(ln):
        print("Error, invalid index received: out of range.")
        return
    else:
        LAYER_SELECT = ln[args[0]]
        receiveLatentDirty = True
        print("Layer Selected for render: {}".format(LAYER_SELECT))


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
    sd = t_model.state_dict()
    t_values = sd[layer_name]
    s = t_values.shape
    print(t_values.shape)

    degrade_picks = np.random.rand(s[0], s[1], s[2], s[3])
    degrade_picks = torch.tensor((degrade_picks >= threshold).astype(float)).cuda()
    t_values *= degrade_picks
    sd[layer_name] = t_values
    t_model.load_state_dict(sd)


def recv_degrade_layer_by_name(addr, *args):
    global model, receiveLatentDirty
    degrade_layer_by_name(model, args[0], threshold = float(args[1]))
    receiveLatentDirty = True


def get_number_of_available_models(addr, *args):
    oscSender.send_message('/number_models', len(ALL_MODEL_FILES_LIST))


def get_available_models_by_name(addr, *args):
    tcount = 0
    out_models = []
    for x in ALL_MODEL_FILES_LIST:
        if args[0] in x:
            out_models.append(x)
            tcount += 1

    oscSender.send_message('/available_models_by_name', [tcount] + out_models)


def load_model_by_name(addr, *args):
    global model, receiveLatentDirty, ORIGINAL_MODEL_STATE_DICT
    for x in ALL_MODEL_FILES_LIST:
        if args[0] in x:
            print("Loading model {}".format(x))
            with open(x, 'rb') as f:
                model = pickle.load(f)['G_ema'].cuda()
                ORIGINAL_MODEL_STATE_DICT = deepcopy(model.state_dict())

            model.eval()
            receiveLatentDirty = True
            return

    print("No model found by name: {}".format(args[0]))


def load_model_by_index(addr, *args):
    global model, receiveLatentDirty, ORIGINAL_MODEL_STATE_DICT
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

    model.eval()
    receiveLatentDirty = True


def receive_latent_vector(addr, *args):
    global receiveLatentDirty, receiveLatent
    if len(args) != 512:
        print('Invalid latent vector size received: {}'.format(len(args)))
        return

    with mutex:
        receiveLatent = np.array(args).reshape(1, 512)
        receiveLatentDirty = True


def set_latent_coords(addr, *args):
    global LATENT_X_COORD, LATENT_Y_COORD, receiveLatentDirty
    LATENT_X_COORD = float(args[0])
    LATENT_Y_COORD = float(args[1])
    receiveLatentDirty = True


dispatcher = dispatcher.Dispatcher()
dispatcher.map('/get_number_models', get_number_of_available_models)
dispatcher.map('/latent', receive_latent_vector)
dispatcher.map('/load_model_by_index', load_model_by_index)
dispatcher.map('/get_available_layer_names', get_available_layers)
dispatcher.map('/degrade_layer_by_name', recv_degrade_layer_by_name)
dispatcher.map('/revert_state_dict', revert_state_dict)
dispatcher.map('/get_render_layers', get_render_layers)
dispatcher.map('/set_render_layer', set_render_layer)
dispatcher.map('/set_latent_coords', set_latent_coords)

oscListener = osc_server.ThreadingOSCUDPServer((SERVER_IP_ADDRESS, SERVER_PORT), dispatcher)
oscListenerThread = Thread(target=oscListener.serve_forever, args=())
print("Starting OSC listener thread.")
oscListenerThread.start()


receiveLatent = np.random.rand(1, 512)
receiveLatentDirty = True


def get_seed(x, y):
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


def get_image_from_numpy_array(latent_vector: np.array):
    global LAYER_SELECT, LATENT_X_COORD, LATENT_Y_COORD, model
    ws = get_ws_from_xy_coords(model, get_seed(LATENT_X_COORD, LATENT_Y_COORD))

    if LAYER_SELECT is None:
        out_img = model.synthesis(ws)
    else:
        ws = ws.to(torch.float32).unbind(dim=1)
        x = model.synthesis.input(ws[0])

        for layer_name, w in zip(model.synthesis.layer_names, ws[1:]):
            x = getattr(model.synthesis, layer_name)(x, w)
            if LAYER_SELECT in layer_name:
                break
        out_img = x.to(torch.float32)

    out_img = (out_img.permute(0, 2, 3, 1) * 0.5 + 0.5019607843137255).clamp(0.0, 1.0)

    if out_img.shape[3] > 3:
        out_img = out_img[:,:,:,:3]

    out_img = out_img[0].detach().cpu().numpy()
    return out_img


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
    global receiveLatentDirty, receiveLatent, img

    if receiveLatentDirty:
        img = get_image_from_numpy_array(receiveLatent)
        receiveLatentDirty = False

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
    update()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                oscListener.server_close()
                oscListener.shutdown()
                oscListenerThread.join(timeout=2.0)
                quit()
