
# coding: utf-8

# In[1]:


import numpy as np 
import tensorflow as tf 


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


# load the data 
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


# In[4]:


# define place holder for data_input and out_put 
xs=tf.placeholder(shape=[None,28*28],dtype=tf.float32)
ys=tf.placeholder(shape=[None,10],dtype=tf.float32)
# convert input xs into 2d
x_image=tf.reshape(xs,[-1,28,28,1])


# In[5]:


# define convolution layer
def con_2d(inputs,filters):
    return tf.nn.conv2d(inputs,filters,strides=[1,1,1,1],padding='SAME')


# In[6]:


# define pooling layer 
def max_pool_2d(inputs):
    return tf.nn.max_pool(inputs,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='SAME')


# In[7]:


# define dense layer 
def weight(input_shape):
    initial = tf.truncated_normal(input_shape, stddev=0.1)
    return tf.Variable(initial)

def biases(input_shape):
    return tf.Variable(tf.constant(0.01,shape=input_shape,dtype=tf.float32))
    


# In[8]:


# build model 
con1l_weight=weight([5,5,1,32])
con1l_biases=biases([32])
con1l=tf.nn.relu(con_2d(x_image,con1l_weight)+con1l_biases)
pool_l1=max_pool_2d(con1l)   # now, the size is 14*14, and 32 channel 


# In[9]:


con2l_weight=weight([5,5,32,64])
con2l_biases=biases([64])
con2l=tf.nn.relu(con_2d(pool_l1,con2l_weight)+con2l_biases)
pool2l=max_pool_2d(con2l)     # now, the size is 7*7, and 64 channel 


# In[10]:


# define full connect layer 
f_w1=weight([7*7*64,1024])    # full connnect to 2014 NN
f_b1=biases([1024])
h_pool_flat=tf.reshape(pool2l,[-1,7*7*64])   # flat the output of pool2l
f_l1=tf.nn.relu(tf.matmul(h_pool_flat,f_w1)+f_b1)


# In[11]:


# define drop layer 
drop=tf.placeholder(dtype=tf.float32)
f_l1_drop=tf.nn.dropout(f_l1,drop)


# In[12]:


f_w2=weight([1024,10])    # full connnect to 2014 NN
f_b2=biases([10])
prediction=tf.nn.softmax(tf.matmul(f_l1_drop,f_w2)+f_b2)


# In[14]:


loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))


# In[15]:


train=tf.train.AdamOptimizer(1e-4).minimize(loss)


# In[16]:


sess=tf.Session()


# In[17]:


init=tf.global_variables_initializer()


# In[18]:


sess.run(init)


# In[19]:


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, drop: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, drop: 1})
    return result


# In[20]:


for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    # input image shape is [None,784], reshape it into [None,28,28,1]
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys,drop:0.5})
    if i %10==0:
        print(compute_accuracy(mnist.test.images[:1000],
                               mnist.test.labels[:1000]))


# In[ ]:


np.reshape()


# In[ ]:




