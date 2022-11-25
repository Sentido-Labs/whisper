import re

import torchaudio
from pyannote.audio import Pipeline
from torchaudio import transforms

from whisper.diarization_pipeline import millisec, spacermilli


# INSTRUCTIONS NICK
#
#  conda env create -f environment.yml
#  environment.yml in repo root
#
#  conda activate shhh
#
#
#  create huggingface account https://huggingface.co/join and
#  accept the user conditions on both
#  https://hf.co/pyannote/speaker-diarization and https://hf.co/pyannote/segmentation
#  log into hugging-face before running with
#   huggingface-cli login
#
#
#  get_audio_mapping first then run get_diarised_timestamps(audio_mapping)




def get_diarised_timestamps(audio_mapping):
    """
    Get speaker period timestamps over the course of the inputted audio

    Parameters
    ----------
    audio_mapping: dict {"waveform": Tensor (amplitude, time), "sample_rate": int}
        audio waveform mapping

    Returns
    -------
    groups: list[(str,int,int)]
        list of speakers numbered in order of first appearance along with time stamps of their utterances
    """
    audio_splits = diarize_input(audio_mapping)
    segment_info = []
    for g in audio_splits:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        speaker = re.findall('SPEAKER_[0-9][0-9]', string=g[-1])[0]
        segment_info.append((speaker, start, end))
    return segment_info


def diarize_input(audio_mapping):
    """
    Get speaker period timestamps over the course of the inputted audio

    Parameters
    ----------
    audio_mapping: dict {"waveform": Tensor (amplitude, time), "sample_rate": int}
        audio waveform mapping

    Returns
    -------
    groups: list[str]
        list of model outputs giving the time stamps for the different speakers
    """

    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

    dzs = str(pipeline(audio_mapping)).splitlines()

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

        audio_segments.append(waveform[:, start_hz:end_hz])
        segment_info.append((speaker, start, end))

    return audio_segments, segment_info
