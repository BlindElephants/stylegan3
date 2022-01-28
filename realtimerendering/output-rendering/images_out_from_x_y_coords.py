import numpy as np
import pickle
import torch
from realtimerendering.realtimeconsts import ALL_MODEL_FILES_LIST, KEYFRAME_INPUT_FILES
import PIL.Image
import os
from tqdm import tqdm


for input_file in KEYFRAME_INPUT_FILES:
    OUT_DIR = os.path.join('/media', 'jeremystewart', 'slate', 'CURRENT_PROJECTS', 'hybrid_agencies', 'generated_outs')
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_IMAGE_COUNTER = 0
    OUT_DIR = os.path.join(OUT_DIR, '.'.join(os.path.basename(input_file).split('.')[:-1]))
    if os.path.exists(OUT_DIR):
        print("already generated this file, skipping")
        continue
    os.makedirs(OUT_DIR)
    print("Generating with input file: {}".format(input_file))

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

    model = None
    LAST_LOADED_MODEL = None
    LAYER_SELECT = None


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


    def set_render_layer(idx):
        global model, LAYER_SELECT
        ln = model.synthesis.layer_names
        LAYER_SELECT = ln[idx]
        print("Layer selected for render: {}".format(LAYER_SELECT))


    def load_model(model_path: str):
        global LAST_LOADED_MODEL, model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)['G_ema'].cuda()
        model.eval()
        LAST_LOADED_MODEL = model_path


    load_model(ALL_MODEL_FILES_LIST[-1])

    for line in tqdm(input_dat):
        if line[3] != LAST_LOADED_MODEL:
            load_model(line[3])
        if line[4] == -1:
            LAYER_SELECT = None
        else:
            ls = model.synthesis.layer_names[line[4]]

        ws = get_ws_from_xy_coords(model, get_seed(line[0], line[1]))

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

        out_img = (out_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        channel_select = line[-1]

        if out_img.shape[3] > 3:
            out_img = out_img[:, :, :, channel_select:channel_select+3]

        out_img = out_img[0].detach().cpu().numpy()
        PIL.Image.fromarray(out_img, 'RGB').save(os.path.join(OUT_DIR, f'out_image_{OUT_IMAGE_COUNTER:05d}.png'))
        OUT_IMAGE_COUNTER += 1