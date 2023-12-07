#!/usr/bin/python
from PIL import Image
import os, sys

source = 'data/images/'
destiny = 'data/images2/'

dic = os.listdir(source)
print(dic)

for item in dic:
    img = Image.open(source + item)
    rbi = img.convert('RGB')
    width, height = img.size
    ratio = width/height
    new_width = 640
    new_height = 360
    imgResize = rbi.resize((new_width, new_height), Image.Resampling.LANCZOS)
    imgResize.save(destiny + item[:-4] + '.jpg', quality=100)