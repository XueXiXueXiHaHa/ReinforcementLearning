#!/usr/bin/env python
#coding:utf8
#介绍:https://zhuanlan.zhihu.com/p/25719115
#git:https://github.com/yenchenlin/DeepLearningFlappyBird
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

#用标准偏差为0.01的正态分布随机初始化所有权重矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

#定义卷积层,这里的W就是卷积核,strides是滑动的步数
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

#定义pooling层,这里写死是2*2的
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

#初始化DQN网络
def createNetwork():
    #定义网络各层的参数,w,b
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 网络输入
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 网络结构
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2) # = 5 * 5 * 64
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3) # = 5 * 5 * 64
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    #网络的输出,这里输出的是ACTIONS个实数值,Q(s,a)值
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

#训练网络
def trainNetwork(s, readout, h_fc1, sess):
    # --------------定义网络的loss和优化-------------------
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    #这里的a 是s,a,r,s_,a_中的a. a中只有某一维是1,所以乘以网络输出然后按维度相加就是Q(s,a)了
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #实例化一个FlappyBird游戏
    game_state = game.GameState()

    #创建replay memory,deque是双向链表,两头都可以pop,这里其实可以直接限制大小,deque(maxlen=2)
    D = deque()

    # printing
    #a_file = open("logs_" + GAME + "/readout.txt", 'w')
    #h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    #acton=[1,0] 是什么也不干,action = [0,1] 是往上蹦一下
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #把env返回的state变成80*80的灰度图
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    #这里要把像素二值化,把像素值大于1的都置为255,也就是说把不是黑色0的地方都置为白色255
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    #变成80*80*4 ,由于是初始装填,所以只能拿4个相同的图像
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #------------保存和加载网络---------
    #声明保存变量
    saver = tf.train.Saver()
    #初始化tf的变量
    sess.run(tf.initialize_all_variables())
    #声明检查点
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    #---------------开始迭代-----------------
    #exploration的系数
    epsilon = INITIAL_EPSILON
    #t是每走一步+1
    t = 0
    while "flappy bird" != "angry bird":

        #-------------第1步:choose action------------------
        #传入输入s_t,获取网络输出,并进行e-greedy选取action,action就2个,点击一下,或者什么也不干
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        #a_t为actions长度的全0列表,选中那个action就在指定位置置为1.
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            #0.0001的概率随机选取动作
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                #根绝网络的输出,选取maxQ(s,a)
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        #减小exploration,OBSERVE=10w,这里始终保持0.0001
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        #-----------------第2步:take action -----------------------    
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        #处理一下返回的图像,并把图像添加到S_t,把最老的那一祯挤掉
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        #-----------------第3步:保存样本[s,a,r,s_,terminal]----------------------
        #保存s,a,r,s_到replay memory中,这里是5w
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #-----------------第4步:抽样学习----------------------    
        #在开始训练前,先一顿瞎玩,起码把replay填满.OBSERVE=10w,
        if t > OBSERVE:
            #随机抽取BATCH大小的样本
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            #获取抽样样本的输出Q(s_,a),计算R+gamma* maxQ(s_,a) ,也就是y_label
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            #反向传播训练DQN
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        #每隔10000steps保存一下sess
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        #------打印一些信息-----
        state = ""
        if t <= OBSERVE:
            state = "observe"
        #在10w-200w step之间还是有探索性的
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        #每一步都打印一下    
        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
