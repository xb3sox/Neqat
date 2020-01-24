# many thanks to https://github.com/Tools4Project/4501Project/blob/master/cnn_model.py for the main code, however we maid some modification
import os
import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce
import cv2
import imageio
import time
import datetime
import os

images = []
images_video = []
index = 0
## Inputs 
file_content_image = 'caravan-67758_1280.jpg' # content image
file_style_image = ['trad2 (3).jpg','trad-palm1.jpg'] # style image   

## Parameters 
weight_style = 0.2 
weight_content= 0.1

ITERATIONS = [100, 28]

## Layers
layer_content = 'conv4_2' 
layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers_style_weights = [0.5,0.4,0.2,0.2,0.2]

## VGG19 model
path_VGG19 = 'imagenet-vgg-verydeep-19.mat'
# VGG19 mean for standardisation (RGB)
VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

path_output = 'output-15'  # directory to write checkpoint images into

### read image
def imread(path):
    return scipy.misc.imread(path).astype(np.float)   

# save image as uint8 and put it in range 0 - 255
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

# preprocess the image to be compat for VGG19 model  
def imgpreprocess(image):
    image = np.reshape(image, ((1,) + image.shape))
    return image - VGG19_mean

# 
def imgunprocess(image):
    temp = image + VGG19_mean
    return temp[0] 

# function to convertgreyscale to RGB
def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret
    

# create output directory
if not os.path.exists(path_output):
    os.mkdir(path_output)
for i in range(len(file_style_image)):
    # read in images
    img_content = imread(file_content_image) 
    img_style = imread(file_style_image[i]) 

    # convert if greyscale
    if len(img_content.shape)==2:
        img_content = to_rgb(img_content)

    if len(img_style.shape)==2:
        img_style = to_rgb(img_style)

    # resize style image to match content
    img_style = scipy.misc.imresize(img_style, img_content.shape)
    img_style = scipy.misc.imresize(img_style, img_content.shape)

    img_initial = img_content
    # preprocess each image
    img_content = imgpreprocess(img_content)
    img_style = imgpreprocess(img_style)
    img_initial = imgpreprocess(img_initial)
    

    #### BUILD VGG19 MODEL

    VGG19 = scipy.io.loadmat(path_VGG19)
    VGG19_layers = VGG19['layers'][0]

    # take the layer weight and bais to recreate the network
    def _conv2d_relu(prev_layer, n_layer, layer_name):
        # get weights for this layer:
        weights = VGG19_layers[n_layer][0][0][2][0][0]
        W = tf.constant(weights)
        bias = VGG19_layers[n_layer][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        # create a conv2d layer
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
        # add a ReLU function and return
        return tf.nn.relu(conv2d)

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Setup network
    with tf.Session() as sess:
        a, h, w, d     = img_content.shape
        net = {}
        net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
        net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1')
        net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2')
        net['avgpool1'] = _avgpool(net['conv1_2'])
        net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1')
        net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2')
        net['avgpool2'] = _avgpool(net['conv2_2'])
        net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1')
        net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2')
        net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3')
        net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4')
        net['avgpool3'] = _avgpool(net['conv3_4'])
        net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1')
        net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2')     
        net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3')
        net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4')
        net['avgpool4'] = _avgpool(net['conv4_4'])
        net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1')
        net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2')
        net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3')
        net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4')
        net['avgpool5'] = _avgpool(net['conv5_4'])


    ### CONTENT LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
    def content_layer_loss(p, x):

        loss = 0.5 * tf.reduce_sum(tf.pow((x - p), 2))
        return loss

    with tf.Session() as sess:
        sess.run(net['input'].assign(img_content)) # Assign the content image to be the input of the VGG model. 
        p = sess.run(net[layer_content])  # Select the output tensor of content layer
        x = net[layer_content] # Set x to be the content layer activation from the layer we have selected
        p = tf.convert_to_tensor(p) # converts Python objects of various types to Tensor objects
        content_loss = content_layer_loss(p, x) # Compute the content cost


    ### STYLE LOSS: FUNCTION TO CALCULATE AND INSTANTIATION

    def style_layer_loss(a, x):
        _, h, w, d = [i.value for i in a.get_shape()]
        M = h * w 
        N = d 
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        loss = 0.5 * tf.reduce_sum(tf.pow((G - A),2))
        return loss
    def compute_style():
        style_loss = 0.
        for i in range(len(layers_style)):
            a = sess.run(net[layers_style[i]])
            x = net[layers_style[i]]
            a = tf.convert_to_tensor(a)
            style_loss += (style_layer_loss(a, x)* layers_style_weights[i])
        return style_loss

    def gram_matrix(x, M, N):
        F = tf.reshape(x, (M, N))                   
        G = tf.matmul(tf.transpose(F), F)
        return G

    with tf.compat.v1.Session() as sess:
        sess.run(net['input'].assign(img_style)) # Assign the style image to be the input of the VGG model. 
        style_loss = compute_style() # style loss is calculated for each style layer and summed
            
    ### Define loss function and minimise
    with tf.compat.v1.Session() as sess:
        # loss function
        L_total  = weight_content * content_loss + weight_style * style_loss 
        
        # instantiate optimiser
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5.0)
        train_step = optimizer.minimize(L_total)

        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        sess.run(net['input'].assign(img_initial))
        size =0.
        last_image = ''
        for iteration in range(ITERATIONS[i]):
            sess.run(train_step)
            img_output = sess.run(net['input'])
            img_output = imgunprocess(img_output)
            output_file =  path_output+'/%d.png' % (index)
            images_video.append(output_file)
            height, width, layers = img_output.shape
            size = (width,height)
            imsave(output_file, img_output)
            index = index +1
                ###################### Creating video #######################
            images.append(output_file)
            last_image = output_file
        file_content_image = last_image

img_array = []
# Determine the width and height from the first image
for filename in images:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

