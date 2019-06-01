import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from helper import *
from vizdoom import *

from time import sleep

from class_code.global_net import AC_Network
from class_code.worker import Worker

#%% Hyperparameters

disp_res = '1280X800' #400X250,800X500,1280X800
#level = 'defend_the_center'
level = 'map01'
show_worker_0 = True
load_model = True
max_episode_length = 60
gamma = 1 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 9 # Agent can move Left, Right, or Fire
model_path = './models/' + level + '_' + disp_res 
RES = 'RES_'+ disp_res


#%% Training


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=5e-5)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    if show_worker_0 == True: num_workers = 1
    
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,model_path,global_episodes,show_worker_0,RES,level,max_episode_length))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(.1)
        worker_threads.append(t)
    coord.join(worker_threads)