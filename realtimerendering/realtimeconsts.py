import os
import glob

CLIENT_IP_ADDRESS = '192.168.0.112'
CLIENT_PORT = 8889

SERVER_IP_ADDRESS = '0.0.0.0'
SERVER_PORT = 8889

OUTSIZE = (1920, 1080)

MODEL_FILES_ROOT_DIR = os.path.join('/home', 'jeremystewart', 'Desktop', 'ha-models')
ALL_MODEL_FILES_LIST = glob.glob(MODEL_FILES_ROOT_DIR + '/**/*.pkl', recursive=True)
ALL_MODEL_FILES_LIST.sort()
