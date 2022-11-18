import os
import re
import sys

import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment

from transcribe import transcribe, set_up_model_arguments

spacermilli = 2000


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def string_format_milli(millis):
    milliseconds = str(millis)[:-3]
    seconds = (millis / 1000) % 60
    seconds = str(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = str(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    hours = str(hours)
    return hours+':'+minutes+':'+seconds+','+milliseconds


def prepend_spacer(input_audio_dir, prepped_audio_dir):
    spacer = AudioSegment.silent(duration=spacermilli)

    audio = AudioSegment.from_mp3(input_audio_dir)

    #audio = spacer.append(audio, crossfade=0)

    audio.export(prepped_audio_dir, format='wav')


def diarize_input(prepped_audio_dir):
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

    # TODO change to not need a directory but instead a "Mapping" with both "waveform" and "sample_rate" key:
    #  {"waveform": (channel, time) numpy.ndarray or torch.Tensor, "sample_rate": 44100}
    #  can't be too hard with pydub
    dzs = str(pipeline(prepped_audio_dir)).splitlines()

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


def segment_audio(audio_splits, prepped_audio_dir, spacer_prepended=False):
    audio_segments = []
    segment_speakers = []

    audio = AudioSegment.from_wav(prepped_audio_dir)

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
        #segment_speakers.append(speaker)
        audio[start:end].export(str(i_audio_segments) + '.wav', format='wav')

    return audio_segments, segment_speakers


def transcribe_speaker_segments(audio_segments, speaker_segments, input_audio_dir):
    args, model, output_dir, temperature = set_up_model_arguments()

    output = []

    for i, (audio, start_milli, end_milli) in enumerate(audio_segments):
        print(audio)
        # TODO fix streaming and pass into transcribe as ndarray
        # audio = np.frombuffer(audio.get_array_of_samples(), dtype=np.float32)

        print(audio)
        result = transcribe(model, str(i)+'.wav', temperature=temperature, **args)
        output.append(speaker_segments[i]+'\n'+string_format_milli(start_milli)+' --> '
                      + string_format_milli(end_milli) + '\n' + result['text'])

    with open(output_dir+'/'+os.path.basename(input_audio_dir)+'.dia.txt', "w") as outfile:
        outfile.write('\n\n'.join(output))


def run():
    input_audio_dir = str(sys.argv[1])
    prepped_audio_dir = './temp.wav'
    prepend_spacer(input_audio_dir, prepped_audio_dir)
    audio_splits = diarize_input(prepped_audio_dir)
    audio_segments, segment_speakers = segment_audio(audio_splits, prepped_audio_dir)
    transcribe_speaker_segments(audio_segments, segment_speakers, input_audio_dir)


run()
