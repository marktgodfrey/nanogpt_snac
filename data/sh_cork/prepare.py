import os
import random
from tqdm import tqdm
import numpy as np
import torch
from snac import SNAC
import librosa

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().cuda()
# model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()


def flatten(codes):
    # frame_code = len(model.vq_strides) * model.codebook_size + 2
    flattened_codes = []
    for i in range(codes[0].shape[1]):
        # flattened_codes.append(frame_code)
        flatten_tree(codes, 0, i, model.codebook_size, flattened_codes)
    return flattened_codes


def flatten_tree(codes, level_index, node_index, codebook_size, result):
    if level_index >= len(codes) or node_index >= len(codes[level_index][0]):
        return

    # codebook_offset = len(result) % (2 ** len(codes) - 1)
    codebook_offset = level_index
    result.append(codes[level_index][0][node_index].item() + (codebook_offset * codebook_size))

    # Traverse left child's subtree
    flatten_tree(codes, level_index + 1, 2 * node_index, codebook_size, result)

    # Traverse right child's subtree
    flatten_tree(codes, level_index + 1, 2 * node_index + 1, codebook_size, result)


def write_dataset(data_dir, split):
    audio_paths = []
    for filename in os.listdir(data_dir):
        if os.path.splitext(filename)[-1] == '.mp3':
            audio_paths.append(os.path.join(data_dir, filename))

    print('found %d audio files' % len(audio_paths))

    # start_code = 2 ** len(model.vq_strides) * model.codebook_size
    # end_code = 2 ** len(model.vq_strides) * model.codebook_size + 1

    start_code = len(model.vq_strides) * model.codebook_size
    end_code = len(model.vq_strides) * model.codebook_size + 1

    dtype = np.uint16

    max_duration = int(180. * model.sampling_rate)

    num_augments = 4

    code_count = 0
    with torch.inference_mode():
        for audio_path in tqdm(audio_paths):
            code_path = os.path.join(os.path.dirname(__file__), f'{os.path.splitext(os.path.basename(audio_path))[0]}.bin')
            if os.path.exists(code_path):
                flattened_codes = np.memmap(code_path, dtype=dtype, mode='r')
            else:
                audio_orig, _ = librosa.load(audio_path, sr=model.sampling_rate)
                audio_orig = audio_orig[:max_duration]

                flattened_codes = []
                for _ in range(num_augments):
                    pitch_shift = np.random.uniform(-1.0, 1.0)
                    gain_db = np.random.uniform(-4.0, 4.0)
                    audio = librosa.effects.pitch_shift(audio_orig, sr=model.sampling_rate, n_steps=pitch_shift)
                    audio = np.clip(librosa.db_to_amplitude(gain_db) * audio, -1.0, 1.0)
                    audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).cuda()
                    codes = model.encode(audio)
                    flattened_codes = flattened_codes + [start_code] + flatten(codes) + [end_code]
                    torch.cuda.empty_cache()

                arr = np.memmap(code_path, dtype=dtype, mode='w+', shape=(len(flattened_codes),))
                arr[:len(flattened_codes)] = flattened_codes
                arr.flush()
            code_count += len(flattened_codes)

    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(code_count,))
    idx = 0
    for audio_path in tqdm(audio_paths):
        code_path = os.path.join(os.path.dirname(__file__), f'{os.path.splitext(os.path.basename(audio_path))[0]}.bin')
        codes = np.memmap(code_path, dtype=dtype, mode='r')
        arr[idx:idx+len(codes)] = codes
        idx += len(codes)
    arr.flush()


if __name__ == '__main__':

    random.seed(1137)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    train_dir = '/workspace/data/indianmusic/sh_cork_all/audio/train/'
    write_dataset(train_dir, 'train')

    val_dir = '/workspace/data/indianmusic/sh_cork_all/audio/test/'
    write_dataset(val_dir, 'val')


