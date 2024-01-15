import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb")

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

audio = Audio()

def convert_to_wav(path):
    if path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
        path = 'audio.wav'
    return path    

def metadata_audio_file(path):   
    with contextlib.closing(wave.open(path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return frames, rate, duration



def segment_embedding(segment, duration, path):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])  

def embedding_batch(segments, duration, path):
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, duration, path)
    embeddings = np.nan_to_num(embeddings)
    return embeddings

def speaker_segmentation(embeddings, segments):
    num_speakers = 2
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    speaker_separation = {"speaker_1": [], "speaker_2":[]}
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1) 
        if segments[i]["speaker"] == 'SPEAKER 1' :
           speaker_separation["speaker_1"]   ==   speaker_separation["speaker_1"].append(segments[i]["text"][1:]) 
        else:
           speaker_separation["speaker_2"]   ==   speaker_separation["speaker_2"].append(segments[i]["text"][1:])    
    return segments , speaker_separation    