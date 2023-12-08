#!/usr/bin/python
from PIL import Image
import os, sys
from pathlib import Path
import os
import shutil
source = 'data/f1/'
destiny = 'data/l1/'

dic = os.listdir(source)
dic2 = os.listdir(destiny)

for item in dic:
    print(Path(item))
    for i in dic2:
        print((Path(i)))
        if Path(item).stem == Path(i).stem:
            Path("data/f1/" + str(item)).rename("data/f2/" + str(item))

