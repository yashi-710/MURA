#!/usr/bin/env python
# coding: utf-8

# In[59]:


import tensorflow as tf
import random 
import numpy as np
import os
import cv2
from tqdm import tqdm




# In[48]:


class CNN():
    def __init__(self, num_input, num_classes, cnn_config):
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        self.X = tf.placeholder(tf.float32,
                                [None, num_input], 
                                name="input_X")
        self.Y = tf.placeholder(tf.int32, [None, num_classes], name="input_Y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dense_dropout_keep_prob")
        self.cnn_dropout_rates = tf.placeholder(tf.float32, [len(cnn), ], name="cnn_dropout_keep_prob")

        Y = self.Y
        X = tf.expand_dims(self.X, -1)
        pool_out = X
        with tf.name_scope("Conv_part"):
            for idd, filter_size in enumerate(cnn):
                with tf.name_scope("L"+str(idd)):
                    conv_out = tf.layers.conv1d(
                        pool_out,
                        filters=cnn_num_filters[idd],
                        kernel_size=(int(filter_size)),
                        strides=1,
                        padding="SAME",
                        name="conv_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer
                    )
                    pool_out = tf.layers.max_pooling1d(
                        conv_out,
                        pool_size=(int(max_pool_ksize[idd])),
                        strides=1,
                        padding='SAME',
                        name="max_pool_"+str(idd)
                    )
                    pool_out = tf.nn.dropout(pool_out, self.cnn_dropout_rates[idd])

            flatten_pred_out = tf.contrib.layers.flatten(pool_out)
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)

        self.prediction = tf.nn.softmax(self.logits, name="prediction")
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y, name="loss")
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


# In[49]:


class NetManager():
    def __init__(self, num_input, num_classes, learning_rate, training_data,test_data,
                 max_step_per_action=300,
                 bathc_size=25,
                 dropout_rate=0.85):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.training_data = training_data

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        count=0
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        cnn_drop_rate = [c[3] for c in action]
        
        with tf.Graph().as_default() as g:
            with g.container('experiment'+str(step)):
                model = CNN(self.num_input, self.num_classes, action)
                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    for step in range(self.max_step_per_action):
                        count=count+1
                        t,z=next_batch(self.bathc_size,training_data,count)
                        #batch_x, batch_y = self.mnist.train.next_batch(self.bathc_size)
                        feed = {model.X: t,
                                model.Y: z,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        _ = train_sess.run(train_op, feed_dict=feed)

                        if step % 10 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: t,
                                           model.Y: z,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                            print("Step " + str(step) +
                                  ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                  ", Current accuracy= " + "{:.3f}".format(acc))
                    X = []
                    y = []
                    for features,label in training_data:
                        X.append(features)
                        y.append(label)
                    #batch_x, batch_y = self.mnist.test.next_batch(10000)
                    loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: X,
                                           model.Y: y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                    print("!!!!!!acc:", acc, pre_acc)
                    if acc - pre_acc <= 0.01:
                        return acc, acc 
                    else:
                        return 0.01, acc
                    


# In[39]:


class Reinforce():
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 division_rate=100.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.3):
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network 
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor=discount_factor
        self.max_layers = max_layers
        self.global_step = global_step

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, state):
        return self.sess.run(self.predicted_action, {self.states: state})
        if random.random() < self.exploration:
            return np.array([[random.sample(range(1, 35), 4*self.max_layers)]])
        else:
            return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(tf.float32, [None, self.max_layers*4], name="states")

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                self.policy_outputs = self.policy_network(self.states, self.max_layers)
                print("outputs: ",self.policy_outputs)

            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")

            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name="predicted_action")
            print("action:",self.predicted_action)


        # regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network(self.states, self.max_layers)
                print("self.logprobs", self.logprobs)

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:, -1, :], labels=self.states)
            self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables]) # Regularization
            self.loss               = self.pg_loss + self.reg_param * self.reg_loss

            #compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)
            
            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        rewars = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss],
                     {self.states: states,
                      self.discounted_rewards: rewars})
        return ls


# In[40]:


#from tensorflow.examples.tutorials.mnist import input_data


# In[41]:


#input_data


# In[42]:



'''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363
    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)
'''
def policy_network(state, max_layers):
    with tf.name_scope("policy_network"):
        nas_cell = tf.contrib.rnn.NASCell(4*max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05]*4*max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        print("outputs: ",outputs[:, -1:, :])#,  tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]))
        #return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
        return outputs[:, -1:, :]      


# In[43]:


def train(training_data,test_data):
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    reinforce = Reinforce(sess, optimizer, policy_network, 50, global_step)
    net_manager = NetManager(num_input=40000,
                             num_classes=2,
                             learning_rate=0.001,
                             training_data=training_data,
                             test_data=test_data,
                             bathc_size=50)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0]*50], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0
    for i_episode in range(MAX_EPISODES):       
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.storeRollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = " episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)


# In[44]:




#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#train(training_data)


# In[45]:




DATADIR = "/Users/sidgupta/Documents/mura/train"
DATADIR1 = "/Users/sidgupta/Documents/mura/valid"


# In[46]:


CATEGORIES = ["abnormal", "normal"]


# In[13]:


IMG_SIZE=200
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        if ((class_num)==1):
            t=[0,1]
        else:
            t=[1,0]
        
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # resize to normalize data size
                new_array = np.array(new_array).reshape(40000)
                training_data.append([new_array,t ])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))


# In[14]:


test_data = []

def create_test_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR1,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                new_array = np.array(new_array).reshape(40000)


                test_data.append([new_array, [class_num]])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_test_data()

print(len(test_data))


# In[15]:


random.shuffle(training_data)


# In[16]:


import random
random.shuffle(test_data)
random.shuffle(training_data)


# In[17]:


def next_batch(batch_size,training_data,count):
    X = []
    y = []
    for sample in training_data[count*100:(batch_size+(count*100))]:
        X.append(sample[0])
        y.append(sample[1])    
    return X, y
    
    


# In[60]:


tf.reset_default_graph()


# In[28]:


train(training_data,test_data)

