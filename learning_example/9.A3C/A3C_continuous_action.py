#coding:utf8
"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()     #cpu的数量
MAX_EP_STEP = 400                           #每个epsiode的最大steps数
MAX_GLOBAL_EP = 10                          #所有worker的最大epsiode之和
GLOBAL_NET_SCOPE = 'Global_Net'             #global网络的scope名
UPDATE_GLOBAL_ITER = 5                      #worker每隔多少步push/pull一次global    
GAMMA = 0.9                                 #discount reward 系数
ENTROPY_BETA = 0.01                         #给actor 的loss添加扰动时用的系数,为了增加探索性
LearningRate_Actor = 0.0001                               #actor网络学习率
LearningRate_Critic = 0.001                                #critic网络学习率
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0                               #所有worker的epsiode之和,,不能超过MAX_GLOBAL_EP

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high] #action数值的上下界

#global 和worker的同步锁
update_global_lock = threading.Lock()
#各个worker总共进行的epsiode计数的锁
global_epsiode_lock = threading.Lock()

#创建Actor-Critic网络
class ACNet(object):
    def __init__(self, scope, globalAC=None):
        #如果是global,创建网络,并定义actor网络的参数集为a_params,critic网络的参数集为c_param
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        #如果是worker,创建网络结构,并定义actor网络和critic网络的结构和loss
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                #创建AC网络,actor网络输出均值mu和方差sigma,critic网络输出V(s)
                mu, sigma, self.v = self._build_net()

                #-------定义critic网络的loss = r+gamma*V(s_) - V(s),---------
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                #-------定义actor网络的loss = dP(s) * f(s) ,其中f(s)是td-error---------
                with tf.name_scope('wrap_a_out'):
                    #将均值放大到action的值域范围,mu 是[-1,1]的.
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                #定义正态分布    
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                #for tensorflow 1.3
                #normal_dist = tf.distributions.Normal(mu, sigma)
                with tf.name_scope('a_loss'):
                    #计算action在这个正态分布中的log概率
                    log_prob = normal_dist.log_prob(self.a_his)
                    #action的loss
                    exp_v = log_prob * td
                    #用这个正态分布的信息熵来给loss加一些noise,从而增加探索性.
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                #从正态分布中采样一个A    
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
                #计算worker AC两个网络的梯度    
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            #global与worker通信        
            with tf.name_scope('sync'):
                #拉取global AC网络的参数赋值到worker的AC网络上
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                #更新global AC网络的参数
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
    #创建actor网络和critic网络                
    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        
        #actior网络有两个隐层,第2隐层是分开的,分别计算均值mu 和方差sigma, N_A是actons的数量
        with tf.variable_scope('actor'):
            #输入是s + 200隐层 分别输出mu,sigma
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        
        #critic网络有一个隐层,计算V(s),注意这里输入只有s,跟a没关系
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    #更新global网络    
    def update_global(self, feed_dict):  # run by a local
        #加锁,一个一个更新
        update_global_lock.acquire()
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        update_global_lock.release()
    #拉取global网络参数    
    def pull_global(self):  # run by a local
        #加锁,这里主要是防止有人在更新时,另一边在拉取.保证更新完了再拉取
        update_global_lock.acquire()
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])
        update_global_lock.release()

    #从自身ac网络中抽样    
    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        #--------------epsiode迭代---------------
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            #--------------steps迭代---------------
            for ep_t in range(MAX_EP_STEP):
                #只把一个worker的环境画出来
                if self.name == 'W_0':
                    self.env.render()

                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                
                #判断epsiode是否结束
                done = True if ep_t == MAX_EP_STEP - 1 else False
                r /= 10     # normalize reward

                ep_r += r
                #在这里用3个列表保存住这个spsiode的s,a,r,其中s_也在s的列表里,因为在后面s = s_
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                #每隔UPDATE_GLOBAL_ITER步(5步)或者epsiode结束,去更新Global网络,并拉取Global参数
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    #计算V(s_)
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    #计算target = r+GAMMA * V(s_) ,注意是从最后一个state开始计算的,只有后面的V(s)计算好了,前面的才好计算
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    #更新Global网络,这里更新得可能是别人更新后的网络,不一定是产生自己这个版本的网络
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    #拉取Global网络参数 ,这里拉取的可能是别人更新后的,不一定是自己更新后的.
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done and self.name == 'W_0':
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                if done:
                    global_epsiode_lock.acquire()
                    GLOBAL_EP += 1
                    global_epsiode_lock.release()
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        #优化器,actor和critic网络都是RMSProp
        OPT_A = tf.train.RMSPropOptimizer(LearningRate_Actor, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LearningRate_Critic, name='RMSPropC')
        #实例化Global网络
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        #实例化worker网络,数量是cpu的数量
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    #等待所有worker结束
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

