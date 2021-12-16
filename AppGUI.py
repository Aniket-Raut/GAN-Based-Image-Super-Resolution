import streamlit as st
from model2 import Generator
import tensorflow as tf
from PIL import Image
import numpy as np
import time

def getImage(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def _generate(image,model):
    sr = getImage(model, tf.expand_dims(image, axis=0))[0]
    return sr

def _image(im):
    image = Image.open(im)
    return image

def main():
    generator = Generator()
    generator.load_weights('gan_generator.h5') # loading pretrained weights
    im = st.file_uploader("upload Image", type=['png', 'jpg'])
    st.text("OR")
    path = st.text_input("Enter path")
    col1, col2 = st.beta_columns(2)

    if path:
        lr_img = np.array(Image.open(path))
        sr = _generate(lr_img,generator)
        sr = np.array(sr)
        col1.image(Image.open(path), use_column_width=True)
        col1.header("Original(Low Resolution)")
        time.sleep(0.5)
        col2.image(Image.fromarray(sr), use_column_width=True)
        col2.header("Generated(SR)")

    if im is not None:
        image = _image(im)
        lr = np.array(image)
        sr = _generate(lr,generator)
        sr = np.array(sr)
        col1.image(image, use_column_width=True)
        col1.header("Original(Low Resolution)")
        time.sleep(0.5)
        col2.image(Image.fromarray(sr), use_column_width=True)
        col2.header("Generated(SR)")

if __name__ == "__main__":
    main()
