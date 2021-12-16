from PIL import Image, ImageOps
import os

# Script to resize images

path = r'C:\Users\Aniket\Desktop\pro\Version_3.1.2\images\LR' #source path
spath = r'C:\Users\Aniket\Desktop\pro\Version_3.1.2\images\LR-1' #save path

for image in range(100):
    imagepath = os.path.join(path,f'{image}.png')
    oimage = Image.open(imagepath)
    rimage = ImageOps.fit(oimage,(65,65),Image.BICUBIC) #change 65 to whatever resolution you want
    rimage.save(os.path.join(spath,f'{image}.png'))
