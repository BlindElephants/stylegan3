import os
import glob

CLIENT_IP_ADDRESS = '192.168.0.128'
CLIENT_PORT = 8889

SERVER_IP_ADDRESS = '0.0.0.0'
SERVER_PORT = 8889

OUTSIZE = (1024, 1024)

MODEL_FILES_ROOT_DIR = os.path.join('/media', 'jeremystewart', 'slate', 'CURRENT_PROJECTS', 'hybrid_agencies', 'outputs')

ALL_MODEL_FILES_LIST = glob.glob(MODEL_FILES_ROOT_DIR + '/**/*.pkl', recursive=True)
ALL_MODEL_FILES_LIST.sort()

ALL_MODEL_FILES_LIST = [i for i in ALL_MODEL_FILES_LIST if 'ha-output-symbols' not in i]

for x in ALL_MODEL_FILES_LIST:
    print(x)

KEYFRAME_INPUT_DIR = os.path.join('/home', 'jeremystewart', 'Desktop', 'ha-frames-out')
KEYFRAME_INPUT_FILES = [os.path.join(KEYFRAME_INPUT_DIR, i) for i in os.listdir(KEYFRAME_INPUT_DIR)]
KEYFRAME_INPUT_FILES.sort()