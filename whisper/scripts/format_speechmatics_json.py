import sys
import json

id_ = str(sys.argv[1])

with open(id_+'.json') as f:
    d = json.load(f)

data = d['results']
output = []

for result in data:
    output.append(result['alternatives'][0]['content'])

with open(id_+'.wallace.txt', 'w') as f:
    f.write(' '.join(output))
