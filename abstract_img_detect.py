import os
import numpy as np
import scipy.misc
import scipy.io
import Image
import tensorflow as tf
from sys import stderr
from functools import reduce
import time  
import cv2 as cv2


input_noise = 0.1    
weight_style = 2e2 


layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
gram_style = ['conv1_1', 'conv2_1','conv3_1', 'conv4_1', 'conv5_1']
layers_style_weights = [0.2,0.2,0.2,0.2,0.2]


path_VGG19 = 'imagenet-vgg-verydeep-19.mat'

VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

batch_no = 1            
batch_size = 10   
path_output = 'output'  


def imread(path):
    return scipy.misc.imread(path).astype(np.float)   

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
    
def imgpreprocess(image):
    image = image[np.newaxis,:,:,:]
    return image - VGG19_mean


img_style = scipy.misc.imread('style.jpg').astype(np.float)

noise = np.random.uniform(
        img_style.mean()-img_style.std(), img_style.mean()+img_style.std(),
        (img_style.shape)).astype('float32')

img_initial = noise * input_noise + img_style * (1 - input_noise)



img_style = imgpreprocess(img_style)
img_initial = imgpreprocess(img_initial)
  
VGG19 = scipy.io.loadmat(path_VGG19)
VGG19_layers = VGG19['layers'][0]


def conv_relu(prev_layer, n_layer, layer_name):
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
    return tf.nn.relu(conv2d)

def pool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.Session() as sess:
    a, h, w, d     = img_style.shape
    nn = {}
    nn['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
    nn['conv1_1']  = conv_relu(nn['input'], 0, 'conv1_1')
    nn['conv1_2']  = conv_relu(nn['conv1_1'], 2, 'conv1_2')
    nn['avgpool1'] = pool(nn['conv1_2'])
    nn['conv2_1']  = conv_relu(nn['avgpool1'], 5, 'conv2_1')
    nn['conv2_2']  = conv_relu(nn['conv2_1'], 7, 'conv2_2')
    nn['avgpool2'] = pool(nn['conv2_2'])
    nn['conv3_1']  = conv_relu(nn['avgpool2'], 10, 'conv3_1')
    nn['conv3_2']  = conv_relu(nn['conv3_1'], 12, 'conv3_2')
    nn['conv3_3']  = conv_relu(nn['conv3_2'], 14, 'conv3_3')
    nn['conv3_4']  = conv_relu(nn['conv3_3'], 16, 'conv3_4')
    nn['avgpool3'] = pool(nn['conv3_4'])
    nn['conv4_1']  = conv_relu(nn['avgpool3'], 19, 'conv4_1')
    nn['conv4_2']  = conv_relu(nn['conv4_1'], 21, 'conv4_2')     
    nn['conv4_3']  = conv_relu(nn['conv4_2'], 23, 'conv4_3')
    nn['conv4_4']  = conv_relu(nn['conv4_3'], 25, 'conv4_4')
    nn['avgpool4'] = pool(nn['conv4_4'])
    nn['conv5_1']  = conv_relu(nn['avgpool4'], 28, 'conv5_1')
    nn['conv5_2']  = conv_relu(nn['conv5_1'], 30, 'conv5_2')
    nn['conv5_3']  = conv_relu(nn['conv5_2'], 32, 'conv5_3')
    nn['conv5_4']  = conv_relu(nn['conv5_3'], 34, 'conv5_4')
    nn['avgpool5'] = pool(nn['conv5_4'])



def style_layer_loss(a, x):
    _, h, w, d = [i.value for i in a.get_shape()]
    M = h * w 
    N = d 
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, M, N):
    F = tf.reshape(x, (M, N))                   
    G = tf.matmul(tf.transpose(F), F)
    return G

def style_gram(a):
    _, h, w, d = [i.value for i in a.get_shape()]
    M = h * w 
    N = d 
    A = gram_matrix(a, M, N)  
    return A

with tf.Session() as sess:
    sess.run(nn['input'].assign(img_style))
    B = tf.zeros([512,512],tf.float32)
    for layer in gram_style:
        a = sess.run(nn[layer])
        x = nn[layer]
        a = tf.convert_to_tensor(a) 
        c = style_gram(a).eval()
        c.resize((512,512))
        tf.convert_to_tensor(c)
        B += c

with tf.Session() as sess:
    sess.run(nn['input'].assign(img_style))
    style_loss = 0.
    for layer, weight in zip(layers_style, layers_style_weights):
        a = sess.run(nn[layer])
        x = nn[layer]
        a = tf.convert_to_tensor(a) 
        style_loss += style_layer_loss(a, x)

        
with tf.Session() as sess:
    
    L_total  =  weight_style * style_loss 
    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      L_total, method='L-BFGS-B',
      options={'maxiter': batch_size})
    
    path = os.getcwd()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    sess.run(nn['input'].assign(img_initial))
    for i in range(0,batch_no):
        optimizer.minimize(sess)
        stderr.write('Iteration %d/%d\n' % (i*batch_size+1, batch_no*batch_size))
        tf.convert_to_tensor(B,dtype=tf.float32)
        sess.run(B)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        style = B.eval()
        img = Image.fromarray(style, 'RGB') 
        img.save(path+'/output/'+timestr+'_style.png')
        np.savetxt(path+'/output/'+timestr+'_gram_matrix.txt', style)
        
       