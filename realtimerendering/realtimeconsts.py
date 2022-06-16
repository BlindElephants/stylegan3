import os
import glob

# CLIENT_IP_ADDRESS = '192.168.0.208'
CLIENT_IP_ADDRESS = '192.168.1.55'
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


OUT_PERF_RECORDER_FILE = os.path.join(os.path.dirname(__file__), 'out_performance_recorder.txt')
RENDER_OUT_FRAMES_DURATION_MULTIPLIER = 6
RENDER_OUT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "RENDERED_OUT_IMAGES")
RENDER_OUT_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "RENDERED_OUT_VIDEOS")

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "ALL_MODEL_FILES_LIST.txt"), 'w') as f:
        for m in ALL_MODEL_FILES_LIST:
            f.write(m)
            f.write('\n')