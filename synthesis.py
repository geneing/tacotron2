
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pylab as plt

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 16000

checkpoint_path = "../outdir/checkpoint_17000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()


text = "Oswald demonstrated his thinking in connection with his return to the United States by preparing two sets of identical questions of the type which he might have thought. Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

v=mel_outputs_postnet.data.cpu().numpy()[0]

vout = np.zeros([55,v.shape[1]], dtype='float32')
vout[0:18,:]=v[0:18,:]
vout[36,:]=v[18,:]
vout[37,:]=v[19,:]

vout.T.tofile('features_synth.f32')

plot_data((mel_outputs.data.cpu().numpy()[0],
           mel_outputs_postnet.data.cpu().numpy()[0],
           alignments.data.cpu().numpy()[0].T))

print(mel_outputs)


#%%

