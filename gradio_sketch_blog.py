# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# !pip install gradio -q 
# %tensorflow_version 1.x for colab only

import gradio as gr
import tensorflow as tf
import urllib.request

cnn_model = tf.keras.models.load_model("cnn.h5")

labels = urllib.request.urlopen("https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt")
labels = labels.read()
labels = labels.decode('utf-8').split("\n")[:-1]



def predict_shape(img):
  img = tf.math.divide(img, 255)
  preds = cnn_model.predict(img.reshape(-1, 28, 28, 1))[0]
  return {label: float(pred) for label, pred in zip(labels, preds)}


output = gr.outputs.Label(num_top_classes=2)
input = gr.inputs.Image(image_mode='L', 
                        source='canvas', 
                        shape=(28, 28), 
                        invert_colors=True, 
                        tool= 'select')



title="Sketch prediction app"
description="A Convolution Neural Network model trained on Google's QuickDraw dataset." \
            " Start by draw a smily face, or anything you can think of!"


gr.Interface(fn = predict_shape,
             inputs = input, 
             outputs = output, 
             live = True, 
             title=title,
             description = description,
             capture_session=True).launch(debug=True)
