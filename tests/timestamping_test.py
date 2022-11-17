import whisper
import sys
import json


model = whisper.load_model("base")
decoding_options = whisper.DecodingOptions(suppress_blank=False)

audio_path = str(sys.argv[0])
lang = str(sys.argv[1])

result = model.transcribe(audio_path, decoding_options, language=lang)

with open(audio_path+'whisper.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
