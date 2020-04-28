import nltk
import re
nltk.download('framenet_v17')
from nltk.corpus import framenet as fn

i = 0

regex = re.compile(r'Core.*?Source.*?\n')

for frame in fn.frames():
    result = re.search(regex, str(frame))
    if result:
        i = i + 1
        print(frame.name)


print(i)
