from PIL import Image,ImageOps

#imagepath = r'C:\Users\Aniket\Desktop\pro\Version_3.1.2\images\HR\0.png'
imagepath =r'C:\Users\Aniket\Desktop\81YDuTWSHyL.png' # Source path
oimage = Image.open(imagepath)
rimage = ImageOps.fit(oimage, (96, 96), Image.BICUBIC)
#rimage.save(r'C:\Users\Aniket\Desktop\pro\Version_3.1.2\images\res\image3.png')
rimage.save(r'C:\Users\Aniket\Desktop\81YDuTWSHyLResize.png') # Destination path
