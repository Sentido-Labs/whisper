import whisper

model = whisper.load_model("base")
result = model.transcribe("Phil Denman - Funding and Payments - Neu 2.mp3")
print(result)