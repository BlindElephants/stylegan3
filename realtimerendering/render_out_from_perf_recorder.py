import datetime
import os

from realtimerendering.realtimeconsts import ALL_MODEL_FILES_LIST, OUT_PERF_RECORDER_FILE
from realtimerendering.realtimeconsts import RENDER_OUT_FRAMES_DURATION_MULTIPLIER
from realtimerendering.realtimeconsts import RENDER_OUT_IMAGE_DIR
import numpy as np
import pickle
import torch
from typing import List, Union
from copy import deepcopy
import PIL.Image
from tqdm import tqdm


def load_model_by_name(model, LAST_LOADED_MODEL, model_name: str):
    if LAST_LOADED_MODEL == model_name:
        return model, LAST_LOADED_MODEL
    for x in ALL_MODEL_FILES_LIST:
        if model_name in x:
            print("Loading model {}".format(x))
            with open(x, 'rb') as f:
                model = pickle.load(f)['G_ema'].cuda()
            LAST_LOADED_MODEL = x
            model.eval()
            return model, LAST_LOADED_MODEL
    print("No model found by name: {}".format(model_name))
    return model, LAST_LOADED_MODEL


def latent_vec_to_ws_tensor(model, latent_vec: List[float]) -> torch.tensor:
    out = np.array(latent_vec).reshape(1, 512)
    out = model.mapping(torch.tensor(out).to('cuda'), c=None)
    return out


def gen_interp_ws(start_ws: torch.tensor, end_ws: torch.tensor, frames_duration: int) -> List[torch.tensor]:
    out_frames = []
    for x in range(frames_duration):
        out_frames.append(torch.lerp(start_ws, end_ws, 1 / (frames_duration - 1) * x))
    return out_frames


def parse_input_list_range(inp_string: str) -> Union[List[int], None]:
    if inp_string is None:
        return None

    inp = [i.strip().split('-') for i in inp_string.split(',')]
    out_list = []
    for x in range(len(inp)):
        t_val = inp[x]
        t_val = [int(i) for i in t_val]
        if len(t_val) == 1:
            out_list.append(t_val[0])
        else:
            out_list = out_list + list(range(t_val[0], t_val[1] + 1))
    return out_list


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default=None)
    args.add_argument('--render_lines', type=str, default=None)
    args.add_argument('--model_set', type=str, default=None)
    args.add_argument('--layer_set', type=str, default=None)
    args.add_argument('--tab_del', action='store_true')
    args.add_argument('--force', action='store_true')

    args.add_argument('--no_interp', action='store_true')
    args.add_argument('--custom_interp', type=int, default=None)

    args = args.parse_args()

    args.render_lines = parse_input_list_range(args.render_lines)
    args.model_set = parse_input_list_range(args.model_set)
    args.layer_set = parse_input_list_range(args.layer_set)

    if args.input_file is not None:
        OUT_PERF_RECORDER_FILE = args.input_file

    with open(OUT_PERF_RECORDER_FILE, 'r') as f:
        dat = f.read()

    if args.tab_del:
        use_delimiter = '\t'
    else:
        use_delimiter = ', '

    dat = [i.split(use_delimiter) for i in dat.split('\n') if len(i) > 0]

    if args.render_lines:
        dat = [dat[i] for i in args.render_lines]

    if args.model_set or args.layer_set:
        new_dat = []
        if args.model_set is not None:
            for mo in args.model_set:
                for d in dat:
                    m = mo % len(ALL_MODEL_FILES_LIST)
                    nd = deepcopy(d)
                    nd[0] = nd[0] + f'_mo_{mo}'
                    nd[1] = ALL_MODEL_FILES_LIST[m]
                    new_dat.append(nd)
        else:
            new_dat = deepcopy(dat)
        dat.clear()
        if args.layer_set is not None:
            for ls in args.layer_set:
                for d in new_dat:
                    nd = deepcopy(d)
                    nd[0] = nd[0] + f'_ls_{ls}'
                    nd[2] = ls
                    dat.append(nd)
        else:
            dat = deepcopy(new_dat)
        new_dat = None
        del new_dat

    print(len(dat))

    model = None
    LAST_LOADED_MODEL = None
    NUMBER_CHANNELS_IN_OUTPUT = 3

    last_img = None
    last_ws = None
    receiveLatentBuffer = []

    NORMALIZE_OUTPUT_IMAGE = False

    out_root_dir = RENDER_OUT_IMAGE_DIR
    os.makedirs(out_root_dir, exist_ok=True)

    no_interp_out_dir = None

    if args.no_interp:
        no_interp_out_dir = os.path.join(out_root_dir, '..', 'no_interp_out_dir')
        os.makedirs(no_interp_out_dir, exist_ok=True)
        no_interp_out_dir = os.path.join(no_interp_out_dir, str(datetime.datetime.now()))
        os.makedirs(no_interp_out_dir)

    pbar_1 = tqdm(total=len(dat))
    for d_idx, d in enumerate(dat):
        t_timestamp = d[0]
        t_model = d[1]
        t_layer = d[2]

        if t_layer == 15:
            t_layer = None

        if args.no_interp:
            t_out_root_dir = no_interp_out_dir
        else:
            t_out_root_dir = os.path.join(out_root_dir, t_timestamp)

            if os.path.exists(t_out_root_dir) and not args.force:
                print("Already rendered this sequence. Skipping...")
                pbar_1.update(1)
                continue
            else:
                os.makedirs(t_out_root_dir, exist_ok=True)

        t_channel = int(d[3])
        t_clear_buffer = bool(d[4])
        t_frames_duration = int(int(d[5]) * RENDER_OUT_FRAMES_DURATION_MULTIPLIER)
        t_latent_target = [float(i) for i in d[6:]]

        assert len(t_latent_target) == 512
        model, LAST_LOADED_MODEL = load_model_by_name(model, LAST_LOADED_MODEL, t_model)

        if t_layer == 'None':
            t_layer = None
        elif t_layer == None:
            pass
        else:
            try:
                t_layer = int(t_layer)
                if t_layer == -1:
                    t_layer = None
                else:
                    ln = model.synthesis.layer_names
                    t_layer = ln[t_layer]
            except ValueError:
                pass

        if last_ws == None:
            print("Generating zeros latent vec to start")
            last_ws = [0.0] * 512
            last_ws = latent_vec_to_ws_tensor(model, last_ws)

        t_ws = latent_vec_to_ws_tensor(model, t_latent_target)

        if args.no_interp:
            if args.custom_interp == None:
                receiveLatentBuffer.append(t_ws)
            else:
                receiveLatentBuffer = gen_interp_ws(last_ws, t_ws, args.custom_interp)
        else:
            receiveLatentBuffer = gen_interp_ws(last_ws, t_ws, t_frames_duration)

        FRAME_OUT_COUNTER = 0
        pbar_2 = tqdm(total=len(receiveLatentBuffer))
        while len(receiveLatentBuffer):
            ws = receiveLatentBuffer.pop(0)
            last_ws = deepcopy(ws)

            if (t_layer is None) or (t_layer == -1):
                out_img = model.synthesis(ws)
            else:
                ws = ws.to(torch.float32).unbind(dim=1)
                x = model.synthesis.input(ws[0])

                for layer_name, w in zip(model.synthesis.layer_names, ws[1:]):
                    x = getattr(model.synthesis, layer_name)(x, w)
                    try:
                        if t_layer and (t_layer in layer_name):
                            break
                    except TypeError:
                        pass
                out_img = x.to(torch.float32)

            out_img = (out_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            if NUMBER_CHANNELS_IN_OUTPUT != out_img.shape[3]:
                NUMBER_CHANNELS_IN_OUTPUT = out_img.shape[3]

            if out_img.shape[3] > 3:
                out_img = out_img[:, :, :, t_channel:t_channel + 3]

            if NORMALIZE_OUTPUT_IMAGE:
                img_range = out_img.max() - out_img.min()
                out_img = (out_img - out_img.min()) / img_range

            out_img = out_img[0].detach().cpu().numpy()

            if args.no_interp:
                if args.custom_interp == None:
                    out_img_path = os.path.join(t_out_root_dir, f'out_image_no_interp_{d_idx:05d}.png')
                else:
                    out_img_path = os.path.join(t_out_root_dir, f'out_image_no_interp_{(d_idx * args.custom_interp) + FRAME_OUT_COUNTER:05d}.png')
            else:
                out_img_path = os.path.join(t_out_root_dir, f'out_image_{FRAME_OUT_COUNTER:05d}.png')

            PIL.Image.fromarray(out_img, 'RGB').save(out_img_path)
            FRAME_OUT_COUNTER += 1
            pbar_2.update(1)
        pbar_2.close()
        pbar_1.update(1)
    pbar_1.close()
