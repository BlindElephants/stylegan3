
def convert_all_seqs_to_video():
    from realtimerendering.realtimeconsts import RENDER_OUT_IMAGE_DIR
    from realtimerendering.realtimeconsts import RENDER_OUT_VIDEO_DIR
    import os
    from tqdm import tqdm

    assert os.path.exists(RENDER_OUT_IMAGE_DIR)
    ld = [os.path.join(RENDER_OUT_IMAGE_DIR, i) for i in os.listdir(RENDER_OUT_IMAGE_DIR)]
    os.makedirs(RENDER_OUT_VIDEO_DIR, exist_ok=True)

    for i in tqdm(ld):
        bn = i.split('/')[-1]
        bn = bn.replace(' ', '_')
        bn = bn.replace('.', '_')
        bn = bn.replace(':', '_')
        out_path = os.path.join(RENDER_OUT_VIDEO_DIR, bn + ".mp4")
        in_path = i.replace(' ', '\ ')
        if os.path.exists(out_path):
            continue
        cmd = f"ffmpeg -i {in_path}/out_image_%05d.png -c:v libx264 -crf 1 -r 30  {out_path}"


        os.system(cmd)


if __name__ == "__main__":
    convert_all_seqs_to_video()