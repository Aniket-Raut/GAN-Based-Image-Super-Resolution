import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
import time
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError
from tensorflow.keras.applications.vgg19 import preprocess_input
from model2 import vgg19


class Gen():
    def __init__(self,model,tr_ds,val_ds,epochs,loss):
        self.train_ds = tr_ds
        self.val_ds = val_ds
        self.epochs = epochs
        self.model = model
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(model =model,
                                              optimizer = Adam(learning_rate=1e-4))
        self.chek_man = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                   directory='.checkpoint/',
                                                   max_to_keep=2)

    def train(self):
        train_ds = self.train_ds
        val_ds = self.val_ds
        self.time_now = time.perf_counter()
        self.epoch = 0
        for lr, hr in train_ds.take(self.epochs):
            self.epoch = self.epoch+1
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            loss = self.train_step(lr, hr)
            los = Mean()
            los(loss)
            # =======================================
            # To Evaluate Model after every 10 Epochs
            # =======================================
            if self.epoch % 10 == 0:
                l_val = los.result()
                los.reset_states()
                duration = time.perf_counter() - self.time_now
                # ======================
                # To Evaluate
                # ======================
                values = []
                for lr, hr in val_ds.take(10):
                    hr = tf.cast(hr,tf.float32)
                    lr = tf.cast(lr,tf.float32)
                    model = self.checkpoint.model
                    sr = model(lr)
                    sr = tf.clip_by_value(sr,0,255)
                    sr = tf.round(sr)
                    sr = tf.cast(sr,tf.uint8)
                    psnr_value = self.psnr(hr, sr)[0]
                    values.append(psnr_value)
                psnr_value = tf.reduce_mean(values)
                print(f'{self.epoch}/{self.epochs}| Generator Loss:{l_val.numpy():.3f} | Duration:{duration:.3f}s |PSNR:{psnr_value.numpy():.3f}')
                self.time_now = time.perf_counter()
                self.chek_man.save()

    def psnr(self,hr,sr):
        return tf.image.psnr(hr,sr,max_val=255)


    @tf.function
    def train_step(self,lr,hr):
        with tf.GradientTape() as tape:
            sr = self.checkpoint.model(lr,training=True)
            gen_loss = self.loss(hr,sr)
        gradient = tape.gradient(gen_loss,self.checkpoint.model.trainable_weights)
        self.checkpoint.optimizer.apply_gradients(zip(gradient,self.checkpoint.model.trainable_weights))
        return gen_loss

# ======================================================================================================================

class FineGen():
    def __init__(self,generator,discriminator,epochs):
        self.generator = generator
        self.discriminator = discriminator
        self.genOpt = Adam(learning_rate=1e-4)
        self.disOpt = Adam(learning_rate=1e-4)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.epochs = epochs
        self.vgg = vgg19()

    def train(self,dataset):
        self.epoch = 0
        self.time_now = time.perf_counter()
        p_mean = Mean()
        d_mean = Mean()
        for lr,hr in dataset:
            self.epoch = self.epoch +1
            p_loss,d_loss = self.train_step(lr,hr)
            p_mean(p_loss)
            d_mean(d_loss)
            self.duration = time.perf_counter()-self.time_now
            if self.epoch % 10 ==0:
                print(f'{self.epoch}/{self.epochs} | Perceptual Loss:{p_mean.result():.3f} | Discriminator Loss:{d_mean.result():.3f} | Duration:{self.duration:.3f}s')
                p_mean.reset_states()
                d_mean.reset_states()
            self.time_now = time.perf_counter()

    @tf.function
    def train_step(self,lr,hr):
        lr = tf.cast(lr,tf.float32)
        hr = tf.cast(hr,tf.float32)
        with tf.GradientTape() as gentape, tf.GradientTape() as dis_tape:
            sr = self.chekpoint.model(lr,training=True)
            _sr = self.discriminator(sr,training=True)
            _hr = self.discriminator(hr,training=True)
            dloss = self.disc_loss(_sr,_hr)
            gloss = self.gen_los(sr)
            closs = self.content_loss(sr,hr)
            ploss = closs + 0.001 * gloss
        genGrad = gentape.gradient(ploss,self.generator.trainable_weights)
        disGrad = dis_tape.gradient(dloss,self.discriminator.trainable_weights)
        self.chekpoint.optimizer.apply_gradients(zip(genGrad,self.generator.trainable_weights))
        self.disOpt.apply_gradients(zip(disGrad,self.discriminator.trainable_weights))
        return ploss,dloss

    def disc_loss(self,sr,hr):
        sr_loss = self.bce(tf.zeros_like(sr),sr)
        hr_loss = self.bce(tf.ones_like(hr),hr)
        return sr_loss+hr_loss

    @tf.function
    def content_loss(self,sr,hr):
        sr = tf.cast(sr,tf.float32)
        hr = tf.cast(hr,tf.float32)
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        srf = self.vgg(sr)/12.5
        hrf = self.vgg(hr)/12.5
        return self.mse(srf,hrf)

    def gen_los(self,sr):
        return self.bce(tf.ones_like(sr),sr)

