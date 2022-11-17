import os
import string
import re

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith(".txt") and "whisper" not in file:
            with open(os.getcwd() + '/' + file) as s:
                transcript = ''.join([line for line in s])
            # transcript = transcript.translate(str.maketrans('', '', string.punctuation))
            transcript = re.sub(r"[^\w\d'\s]+", '', transcript)
            transcript = transcript.replace('\n',' ')
            with open(file[:-3] + 'whisper.txt', "w") as outfile:
                outfile.write(transcript)
