import torch
import random
import numpy as np

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()       

# padding the audio
def pad_audio(waveform, target_len):

  if waveform.shape[-1] < target_len:
    waveform = torch.nn.functional.pad(
        waveform, (0, target_len - waveform.shape[-1]), mode="constant")

  elif len(waveform) > target_len:
    rand_onset = random.randint(0, len(waveform) - target_len)
    waveform = waveform[rand_onset:rand_onset + target_len]

  return waveform


def to_mono(waveform):
  if waveform.shape[0] > 1:
    indx = np.random.randint(0, waveform.shape[0] - 1)
    waveform = waveform[indx]
    waveform = waveform.unsqueeze(0)
  return waveform
    