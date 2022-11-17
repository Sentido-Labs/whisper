from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers.english import EnglishNumberNormalizer, EnglishSpellingNormalizer
import os
import sys

pre_suffix = str(sys.argv[1])

for file in os.listdir(os.getcwd()):
    if os.path.isfile(os.path.join(os.getcwd(), file)):
        if file.endswith(".txt") and pre_suffix not in file:
            with open(os.getcwd() + '/' + file) as s:
                transcript = ''.join([line for line in s])

            std1 = EnglishTextNormalizer()
            std2 = EnglishNumberNormalizer()

            transcript = std1(std2(transcript))
            if "wav" in file:
                file = file[:-7]
            else:
                file = file[:-3]
            with open(file + pre_suffix + '.txt', "w") as outfile:
                outfile.write(transcript)