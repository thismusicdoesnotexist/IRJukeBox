
# coding: utf-8

# In[1]:


import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import audiospec
import matplotlib.pyplot as plt
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-sn", "--sname", required=True,
    help="name of the user")
args = vars(ap.parse_args())

wav_file = args["sname"]

print("wav_file = ", wav_file)

sys.path.append('/usr/local/lib/python2.7/dist-packages')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cqt


# In[7]:


def wav_to_spectrogram(wav_file, output, segment=None):
  '''Convert wav file to spectogram'''
  rate, data = wav.read(wav_file)
  if segment is not None:
    data = data[segment[0] * rate:segment[1] * rate]
  audiospec.plotstft(data[:, 0], rate, plotpath=output, plot_artifacts=False)
  cqt.plot_cqt(wav_file, output)


# In[8]:


wav_to_spectrogram(wav_file, "./output/1.jpg")


# In[9]:


img_path = "./output/1.jpg"
import cv2
img = cv2.imread(img_path)
r, c, ch = img.shape


# In[10]:


print(r, c, ch)


# In[11]:


count = 1
for i in range(0, c, 50):
    im = img[:,i:i+50,:]
    cv2.imwrite("./data/img_"+str(count)+".jpg", im)
    count += 1


# In[12]:


import numpy as np
import keras
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.models import load_model

from  skimage.measure import block_reduce
from PIL import Image
import pretty_midi as pm 
import os, os.path
import re


# In[17]:


image_path = "./data/"
model_path = "ckpt.h5"
import pretty_midi

def one_hot_to_pretty_midi(one_hot, fs=100, program=1,bpm=120):
    notes, frames = one_hot.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # prepend, append zeros so we can acknowledge inital and ending events
    piano_roll = np.hstack((np.zeros((notes, 1)),
                            one_hot,
                            np.zeros((notes, 1))))

    # use changes to find note on / note off events
    changes = np.nonzero(np.diff(piano_roll).T)

    # keep track of note on times and notes currently playing
    note_on_time = np.zeros(notes)
    current_notes = np.zeros(notes)

    bps = bpm / 60
    beat_interval = fs / bps
    strong_beats = beat_interval * 2 #(for 4/4 timing)

    last_beat_time = 0

    for time, note in zip(*changes):
        change = piano_roll[note, time + 1]

        if time >= last_beat_time + beat_interval:
            for note in current_notes:
                time = time / fs

        time = time / fs
        if change == 1:
            # note on
            if current_notes[note] == 0:
                # from note off
                note_on_time[note] = time
                current_notes[note] = 1
            else:
                #re-articulate (later in code)
                '''pm_note = pretty_midi.Note(
                        velocity=100, #don't care fer now
                        pitch=note,
                        start=note_on_time[note],
                        end=time)
                instrument.notes.append(pm_note)
                note_on_time[note] = time
                current_notes[note] = 1'''
        elif change == 0:
            #note off
            pm_note = pretty_midi.Note(
                    velocity=100, #don't care fer now
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            current_notes[note] = 0
            instrument.notes.append(pm_note)
    pm.instruments.append(instrument)
    return pm


# In[18]:


model = load_model(model_path)


# In[19]:


x = []

model = load_model(model_path)
model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

filenums = []
for image_file in os.listdir(image_path):
	print(image_file)
	
	im = Image.open(os.path.join(image_path, image_file))
	im = im.crop((14, 13, 594, 301))
	resize = im.resize((49, 145), Image.NEAREST)
	resize.load()
	arr = np.asarray(resize, dtype="float32")

	x.append(arr)
	filenums.append(int(re.search(r'\d+', image_file).group()))

x = np.array(x)
x /= 255.0

y_pred = model.predict(x)
print(y_pred)

notes_unsorted = [np.argmax(y_pred[n]) for n in range(len(y_pred))]

notes = [x for _,x in sorted(zip(filenums, notes_unsorted))]
print(notes)

i=0
for note in notes:
	one_hot = np.zeros((128, 25))
	one_hot[note, :] = 1
	mid = one_hot_to_pretty_midi(one_hot)
	mid.write('sample_outputs/daylight_' + str(i) + ".mid")
	i += 1


# In[25]:


def merge_midi(midis, input_dir, output, default_tempo=500000):
    pairs = [(int(x[:-4].split('_')[-1]), x) for x in midis]
    pairs = sorted(pairs, key=lambda x: x[0])
    midis = [join(input_dir, x[1]) for x in pairs]
    mid = MidiFile(midis[0])
    # identify the meta messages
    metas = []
    # tempo = default_tempo
    tempo = default_tempo // 2
    for msg in mid:
        if msg.type is 'set_tempo':
            tempo = msg.tempo
        if msg.is_meta:
            metas.append(msg)
    for meta in metas:
        meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))

    target = MidiFile()
    track = MidiTrack()
    track.extend(metas)
    target.tracks.append(track)
    for midi in midis:
        mid = MidiFile(midi)
    for msg in mid:
        if msg.is_meta:
            continue
        if msg.type is not 'end_of_track':
            msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
            track.append(msg)

    track.append(MetaMessage('end_of_track'))
    target.save(output)


# In[27]:


import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join

input_dir = "sample_outputs"
target_dir = "final_output/1.mid"
length = 9

# Get all the input midi files
midis = [x for x in listdir(input_dir) if x.endswith('.mid')]

merge_midi(midis, input_dir, target_dir)

