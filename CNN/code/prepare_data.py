####first arg => where to
####second arg=> from where
####third arg=> which postfix


from PIL import Image
import glob
import os
import sys

path = sys.argv[1]
folder = sys.argv[2] + r'\*.' + sys.argv[3]
num = 0
for filename in glob.glob(folder):
    im = Image.open(filename)
    im = im.resize((200, 200), Image.ANTIALIAS)
    n_path = os.path.join(path, str(num)) + '.' + sys.argv[3]
    im.save(n_path)
    num += 1
