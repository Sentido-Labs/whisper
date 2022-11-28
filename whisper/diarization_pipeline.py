import os
import sys

from pydub import AudioSegment

from transcribe import transcribe

from decoding import DecodingOptions
from whisper.diarizing import diarize_input, get_audio_mapping, segment_audio

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


def transcribe_speaker_segments(audio_segments, speaker_info, audio_path):
    from __init__ import load_model
    model = load_model("medium", device="cuda")
    args = DecodingOptions(language='en').__dict__

    output = []

    for i, segment_waveform in enumerate(audio_segments):
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
