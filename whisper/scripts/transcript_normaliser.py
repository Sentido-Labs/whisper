from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers.english import EnglishNumberNormalizer, EnglishSpellingNormalizer
import os
import sys
import re

pre_suffix = str(sys.argv[1])

for file in os.listdir(os.getcwd()):
    if os.path.isfile(os.path.join(os.getcwd(), file)):
        if file.endswith(pre_suffix+".txt"):
            with open(os.getcwd() + '/' + file) as s:
                transcript = ''.join([line for line in s])

            std1 = EnglishTextNormalizer()
            std2 = EnglishNumberNormalizer()

            transcript = re.sub(r"[^\w\d'\s]+", '', transcript)
            transcript = transcript.replace('\n', ' ')

            transcript = std1(std2(transcript))
            with open(file[:-3]+"norm.txt", "w") as outfile:
                outfile.write(transcript)
