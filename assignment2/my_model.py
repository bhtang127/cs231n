import tensorflow as tf
import numpy as np

def residual_layer(finput, depth, filter_width, n_filters,
                   down_sampling=1, res_depth=2):

    # implement of ResNet layers
    def sub_layer(finput, filter_width, n_filters, 
                  down_sampling, res_depth):
        # aux layer that combined into res layer
        channels = tf.shape(finput)[-1]
        fw, nf = filter_width, n_filters
        ds, rd = down_sampling, res_depth
        Wconv = [tf.get_variable("Weight1", shape=[fw,fw,channels,nf])]
        Wconv += [tf.get_variable("Weight"+str(i+1), shape=[fw,fw,nf,nf]) for i in range(1,res_depth)] 
        bconv = [tf.get_variable("bias"+str(i+1), shape=[nf]) for i in range(res_depth)]
        Gamma = [tf.get_variable("Gamma"+str(i+1), shape=[1,1,1,nf]) for i in range(res_depth)]
        Beta = [tf.get_variable("Beta"+str(i+1), shape=[1,1,1,nf]) for i in range(res_depth)]

        fout = finput
        for i in range(res_depth):
            h1 = tf.nn.conv2d(fout, Wconv[i], strides=[1,ds,ds,1], padding="SAME") + bconv[i]
            ds = 1
            mean, var = tf.nn.moments(h1, axes=[0,1,2], keep_dims=True)
            h2 = tf.nn.batch_normalization(h1, mean, var, Beta[i], Gamma[i], 1e-16)
            fout = tf.nn.relu(h2)
        
        if down_sampling == 1:
            fout += finput
            return fout
        else:
            projection = tf.get_variable("projection", shape=[1,1,channels,nf])
            o1 = tf.nn.conv2d(finput, projection, strides=[1,down_sampling,down_sampling,1], padding="SAME")
            return fout + o1
    
    fout = finput
    for i in range(depth):
        with tf.variable_scope("sub_layer"+str(i+1)):
            fout = sub_layer(fout, filter_width, n_filters, down_sampling, res_depth)
            down_sampling = 1
    
    return fout


def my_model(X, y, is_training, depth):

    # set some variables
    init_W = tf.get_variable("init_Weight", shape=[32,32,3,16])
    init_b = tf.get_variable("init_bias", shape=[16])
    init_Gamma = tf.get_variable("init_Gamma",shape=[1,1,1,16])
    init_Beta = tf.get_variable("init_Beta",shape=[1,1,1,16])

    softmax_W = tf.get_variable("softmax_Weight", shape=[64, 10])
    softmax_b = tf.get_variable("softmax_bias", shape=[10])


    h1 = tf.nn.conv2d(X, init_W, strides=[1,1,1,1], padding='SAME') + init_b
    mean, var = tf.nn.moments(h1, axes=[0,1,2], keep_dims=True)
    h2 = tf.nn.batch_normalization(h1,mean,var,init_Beta,init_Gamma,1e-16)
    output = tf.nn.relu(h2)

    with tf.variable_scope("residual_layer1"):
        output = residual_layer(output,depth,3,16,1,2)
    
    with tf.variable_scope("residual_layer2"):
        output = residual_layer(output,depth,3,32,2,2)

    with tf.variable_scope("residual_layer3"):
        output = residual_layer(output,depth,3,64,2,2)
    
    output = tf.reduce_mean(output, [1,2])
    out = tf.reshape(output, [-1, 64])
    y_out = tf.matmul(out, softmax_W) + softmax_b
    return y_out


    
