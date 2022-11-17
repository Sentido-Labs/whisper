#!/bin/bash
for file in ./*.mp3
do
  whisper "$file" --output_dir ./transcript/whispers
done

