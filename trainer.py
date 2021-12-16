from model2 import Generator,discriminator
from train_func import Gen,FineGen
from Data import Dataset
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

#
# warning: Before running this file please rename '1700epochweights.h5' to any name you want
# or else previously trained weight will be overwritten
#
# if you decide to train on your own please
# change 'gan_generator.h5' from generator.load_weights('gan_generator.h5') AppGUI.py File
# to whatever name you given to save weights

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# tf.compat.v1.disable_eager_execution()
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# ========================================================
# Training Generator
# ========================================================

gen = Generator()
dis = discriminator()
ds = Dataset(5,subset='train').dataset()
ds_val = Dataset(5,subset='validation').dataset()
def train_gen():
    trainer = Gen(gen, ds, ds_val, epochs=25000, loss=MeanSquaredError())
    trainer.train()
    # saving weights for later use
    gen.save_weights('1700EpochWeights.h5') # name changed to avoid overwrite. original file is gan_generator.h5

# ========================================================
# Fine Tuning Generator
# ========================================================

def finetune():
    gen.load_weights('700EpochWeights.h5') # name changed to avoid overwrite. original file is gan_generator.h5
    trainer = FineGen(generator=gen,discriminator=dis,epochs=25000)
    trainer.train(ds)

train_gen()
finetune()