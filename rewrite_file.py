import re
import glob
import os
source="data/labels/"
for filename in glob.glob(os.path.join(source, '*.txt')):
    with open(filename, "r") as f:
        contents = f.read()
    contents = re.sub(r'^0', '1', contents, flags = re.MULTILINE)
    with open(filename, "w") as f:
        f.write(contents)