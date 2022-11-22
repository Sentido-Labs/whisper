import os
import re
import sys

import pyannote.audio
from pyannote.audio import Pipeline
from pydub import AudioSegment

import torchaudio
from torchaudio import transforms
import torch

from transcribe import transcribe, set_up_model_arguments
from audio import load_audio

from decoding import DecodingOptions

spacermilli = 2000


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def string_format_milli(milliseconds):
    seconds, milliseconds = divmod(milliseconds,1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    seconds = seconds + milliseconds/1000
    return (str(hours) if hours > 9 else '0'+str(hours))+':'+\
           (str(minutes) if minutes > 9 else '0'+str(minutes))+':'+\
           (str(seconds) if seconds > 9 else '0'+str(seconds))


def prepend_spacer(input_audio_dir):
    spacer = AudioSegment.silent(duration=spacermilli)

    from pydub.utils import mediainfo
    audio_format = mediainfo("./"+input_audio_dir)['format_name']

    if "mp3" in audio_format:
        audio = AudioSegment.from_mp3(input_audio_dir)
    elif "wav" in audio_format:
        audio = AudioSegment.from_wav(input_audio_dir)
    else:
        raise NotImplementedError('File Type not yet implemented!')

    audio = spacer.append(audio, crossfade=0)

    return audio  # audio.export(prepped_audio_dir, format='wav')


def diarize_input(audio_mapping):
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

    # TODO change to not need a directory but instead a "Mapping" with both "waveform" and "sample_rate" key:
    #  {"waveform": (channel, time) numpy.ndarray or torch.Tensor, "sample_rate": 44100}
    #  can't be too hard with pydub

    dzs = str(pipeline(audio_mapping)).splitlines()

    # print("pipeline output: "+str(dzs))

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)
    print(*groups, sep='\n')
    return groups


def get_audio_mapping(audio_path):
    SAMPLE_RATE = 16000
    from pydub.utils import mediainfo
    audio_format = mediainfo("./" + audio_path)['format_name']

    import pathlib
    torchaudio.set_audio_backend("sox_io")
    waveform, sample_rate = torchaudio.load(str(pathlib.Path().resolve()) + '/' + audio_path, format=audio_format)
    transform = transforms.Resample(sample_rate, SAMPLE_RATE)
    waveform = transform(waveform)
    # waveform = torch.from_numpy(load_audio(audio_path))

    audio_mapping = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
    return audio_mapping


def segment_audio(audio_mapping, audio_splits, spacer_prepended=False):
    waveform = audio_mapping['waveform']
    milli_hz = audio_mapping['sample_rate'] / 1000
    segment_info = []
    audio_segments = []

    i_audio_segments = -1
    for g in audio_splits:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        speaker = re.findall('SPEAKER_[0-9][0-9]', string=g[-1])[0]
        start = (millisec(start))
        end = (millisec(end))
        if spacer_prepended:
            start -= spacermilli
            end -= spacermilli
        print(start, end)
        i_audio_segments += 1

        start_hz = int(start * milli_hz)
        end_hz = int(end * milli_hz)

        # TODO stream instead of file hop
        audio_segments.append(waveform[:, start_hz:end_hz])
        segment_info.append((speaker, start, end))

    return audio_segments, segment_info


def transcribe_speaker_segments(audio_segments, speaker_info, audio_path):
    from __init__ import load_model
    model = load_model("small", device="cuda")
    args = DecodingOptions(language='en').__dict__

    output = []

    for i, segment_waveform in enumerate(audio_segments):
        # TODO fix streaming and pass into transcribe as ndarray
        # audio = np.frombuffer(audio.get_array_of_samples(), dtype=np.float32)
        (speaker, start_milli, end_milli) = speaker_info[i]

        result = transcribe(model, segment_waveform, **args)
        output.append(speaker+'\n'+string_format_milli(start_milli)+' --> '
                      + string_format_milli(end_milli) + '\n' + result['text'])

    with open('./'+os.path.basename(audio_path)+'.dia.txt', "w") as outfile:
        outfile.write('\n\n'.join(output))


def run():
    audio_path = str(sys.argv[1])
    # audio = prepend_spacer(audio_path)
    audio_mapping = get_audio_mapping(audio_path)
    audio_splits = diarize_input(audio_mapping)
    audio_segments, segment_speakers = segment_audio(audio_mapping, audio_splits)
    transcribe_speaker_segments(audio_segments, segment_speakers, audio_path)


run()
