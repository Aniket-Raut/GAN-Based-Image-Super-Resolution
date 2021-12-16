import tensorflow as tf
import os

class Dataset:
    def __init__(self,batch_size=2,subset='train'):
        self.batch = batch_size
        self.subset =subset

    def dataset(self):
        ds = tf.data.Dataset.zip((self.lr_cache(),self.hr_cache())).batch(self.batch).repeat().shuffle(100)
        return ds

    def load_lr(self):
        return [os.path.join('images/LR', f'{image}.png') for image in range(100)]

    def load_hr(self):
        return [os.path.join('images/HR', f'{image}.png') for image in range(100)]

    def lr_ds(self,image):
        ds = tf.data.Dataset.from_tensor_slices(image)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x:tf.image.decode_png(x,channels=3))
        return ds

    def hr_ds(self,image):
        ds = tf.data.Dataset.from_tensor_slices(image)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3))
        return ds

    def lr_cache_file(self):
        return os.path.join('.caches',f'Lr_{self.subset}.cache')

    def hr_cache_file(self):
        return os.path.join('.caches',f'Hr_{self.subset}.cache')

    def lr_cache(self):
        return self.lr_ds(self.load_lr()).cache(self.lr_cache_file())

    def hr_cache(self):
        return self.hr_ds(self.load_hr()).cache(self.hr_cache_file())