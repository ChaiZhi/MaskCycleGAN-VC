from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
import os
import shutil

from hifigan.meldataset import MAX_WAV_VALUE
from hifigan.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_model(mel2wav_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    checkpt_path = "/Users/happyelements/Documents/GitHub/MaskCycleGAN-VC/hifigan/generator_v3"
    config_file = os.path.join(os.path.split(checkpt_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    haa = AttrDict(json_config)
    print(haa)

    torch.manual_seed(haa.seed)
    generator = Generator(haa).to(device)

    state_dict_g = load_checkpoint(checkpt_path, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    return generator

class HifiGANCoder:
    def __init__(
            self,
            path,
            device=get_default_device(),
            github=False,
            model_name="multi_speaker",
    ):
            self.mel2wav = load_model(path, device)
            self.device = device

    def inverse(self, mel):
        """
        Performs mel2audio conversion
        Args:
            mel (torch.tensor): PyTorch tensor containing log-mel spectrograms (batch_size, 80, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)
        """
        with torch.no_grad():
            return self.mel2wav(mel.to(self.device)).squeeze(1)
