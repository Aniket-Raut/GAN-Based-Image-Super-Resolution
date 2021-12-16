warning: before doing training change save weight name or else pretrained weights of model are overwritten.

for training:
if you want to train manually on your own.
run trainer.py but before running change "gen.load_weights('700EpochWeights.h5')" to "gen.load_weights('anyNameyouwant.h5')".

for adding additional dataset
for training dataset you have to put training images in HR folder and make sure image diamentions are in 384x384 size
after that downscale images to 96x96 by using batch_image_resize.py script. and put downscaled image into LR folder.


if you want to run application without doing training you can run AppGUI.py directly by using described command.