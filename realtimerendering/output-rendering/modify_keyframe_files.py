from realtimerendering.realtimeconsts import ALL_MODEL_FILES_LIST, KEYFRAME_INPUT_FILES, KEYFRAME_INPUT_DIR
import os

for x in ALL_MODEL_FILES_LIST:
    print(x)

KEYFRAME_INPUT_FILES = KEYFRAME_INPUT_FILES[0]

if type(KEYFRAME_INPUT_FILES) is not list:
    KEYFRAME_INPUT_FILES = [KEYFRAME_INPUT_FILES]

MODEL_FILE = [40]
OUTPUT_LAYER = [None]  # all layers = list(range(-1, 15))
X_OFFSET = [None]
Y_OFFSET = [None]

MODIFICATIONS = []

for mf in MODEL_FILE:
    for ol in OUTPUT_LAYER:
        for xo in X_OFFSET:
            for yo in Y_OFFSET:
                m = {
                    'model_file': mf,
                    'output_layer': ol,
                    'xo': xo,
                    'yo': yo
                }
                MODIFICATIONS.append(m)

print(len(MODIFICATIONS))

in_dat = None

for input_file in KEYFRAME_INPUT_FILES:
    input_dat = None

    with open(input_file, 'r') as f:
        input_dat = f.read()

    input_dat = [i for i in input_dat.split('\n') if len(i)]
    for i in range(len(input_dat)):
        if len(input_dat[i]) == 0:
            continue
        n = input_dat[i].split('\t')
        n[0] = float(n[0])
        n[1] = float(n[1])
        n[2] = int(n[2])
        n[4] = int(n[4])
        n[5] = int(n[5])
        input_dat[i] = n

    for m in MODIFICATIONS:
        mf_offset = m['model_file']
        ol = m['output_layer']
        xo = m['xo']
        yo = m['yo']

        OUTPUT_FILE_NAME = '.'.join(os.path.basename(input_file).split('.')[:-1])
        OUTPUT_FILE_NAME = OUTPUT_FILE_NAME + "_{}_{}_{}_{}.txt".format(mf_offset, ol, xo, yo)
        OUTPUT_FILE_NAME = os.path.join(KEYFRAME_INPUT_DIR, OUTPUT_FILE_NAME)
        print(OUTPUT_FILE_NAME)

        if os.path.exists(OUTPUT_FILE_NAME):
            print("already created this file. skipping")
            continue

        OUTPUT_DAT = []

        with open(OUTPUT_FILE_NAME, 'w') as f:

            for line_idx, line in enumerate(input_dat):
                t_mf = line[3]
                t_mf_idx = ALL_MODEL_FILES_LIST.index(t_mf)

                t_x = line[0]
                t_y = line[1]
                t_ol = line[4]

                if xo is not None:
                    t_x += xo
                if yo is not None:
                    t_y += yo
                if mf_offset is not None:
                    t_mf_idx += mf_offset
                    t_mf = ALL_MODEL_FILES_LIST[t_mf_idx]

                if ol is not None:
                    t_ol = ol

                t_c = line[-1]

                t_out_line = [t_x, t_y, t_mf_idx, t_mf, t_ol, t_c]
                t_out_line = [str(i) for i in t_out_line]
                t_out_line = '\t'.join(t_out_line)
                t_out_line = t_out_line + "\n"
                f.write(t_out_line)




