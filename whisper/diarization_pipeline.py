import os
import re
import sys

import numpy as np
import pydub
from pyannote.audio import Pipeline
from pydub import AudioSegment

from transcribe import transcribe, set_up_model_arguments

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
    return str(hours)+':'+str(minutes)+':'+str(seconds)


def prepend_spacer(input_audio_dir):
    spacer = AudioSegment.silent(duration=spacermilli)

    audio = AudioSegment.from_mp3(input_audio_dir)

    audio = spacer.append(audio, crossfade=0)

    return audio  # audio.export(prepped_audio_dir, format='wav')


def diarize_input(audio: AudioSegment):
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

    # TODO change to not need a directory but instead a "Mapping" with both "waveform" and "sample_rate" key:
    #  {"waveform": (channel, time) numpy.ndarray or torch.Tensor, "sample_rate": 44100}
    #  can't be too hard with pydub
    mono = audio.split_to_mono()
    waveform = np.frombuffer(mono.get_array_of_samples(), dtype=np.float32)
    sample_rate = mono.sample_width
    audio_mapping = {"waveform": waveform, "sample_rate": sample_rate}

    dzs = str(pipeline(audio_mapping)).splitlines()

    print("pipeline output: "+str(dzs))

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


def segment_audio(audio, audio_splits, spacer_prepended=False):
    audio_segments = []
    segment_info = []

    i_audio_segments = -1
    for g in audio_splits:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        speaker = re.findall('SPEAKER_[0-9][0-9]', string=g[-1])[0]
        start = millisec(start) + 1
        end = millisec(end) + 1
        if spacer_prepended:
            start -= spacermilli
            end -= spacermilli
        print(start, end)
        i_audio_segments += 1
        # TODO stream instead of file hop
        #audio_segments.append((audio[start:end], start, end))
        segment_info.append((speaker, start, end))
        audio[start:end].export(str(i_audio_segments) + '.wav', format='wav')
    return i_audio_segments, segment_info


def transcribe_speaker_segments(i_audio_segments, speaker_info, input_audio_dir):
    from __init__ import load_model
    model = load_model("small", device="cuda")
    args = DecodingOptions(language='en').__dict__

    output = []

    for i in range(i_audio_segments):
        # TODO fix streaming and pass into transcribe as ndarray
        # audio = np.frombuffer(audio.get_array_of_samples(), dtype=np.float32)
        (speaker, start_milli, end_milli) = speaker_info[i]

        result = transcribe(model, str(i)+'.wav', **args)
        output.append(speaker+'\n'+string_format_milli(start_milli)+' --> '
                      + string_format_milli(end_milli) + '\n' + result['text'])

    with open('./'+os.path.basename(input_audio_dir)+'.dia.txt', "w") as outfile:
        outfile.write('\n\n'.join(output))


def run():
    input_audio_dir = str(sys.argv[1])
    audio = prepend_spacer(input_audio_dir)
    audio_splits = diarize_input(audio)
    audio_segments, segment_speakers = segment_audio(audio, audio_splits, True)
    transcribe_speaker_segments(audio_segments, segment_speakers, input_audio_dir)


run()
