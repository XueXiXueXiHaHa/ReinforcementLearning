#coding:utf8
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import tflearn

np.random.seed(123)
tf.set_random_seed(123)

#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS = 1000
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.99  # reward discount
REPLACE_ITER_A = 300
REPLACE_ITER_C = 200
MEMORY_CAPACITY = 7000
BATCH_SIZE = 64
TAU=0.001
RENDER = False
ENV_NAME = 'Pendulum-v0'

#采用soft更新还是hard更新
REPLACEMENT = [
    dict(name='soft', tau=0.001),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
###############################  Actor  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, replacement,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.replacement = replacement

        #声明两个actor网络
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        #声明两个critic网络
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # 4个网络的参数,后面替换用.
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        #定义critic网络更新用的Q_target,这里就是最最普通的DQN更新方式(由于输入是s,a,所以都没有maxQ这个)就是|r+gamma*Q(s_,a_)-Q(s,a)|,
        q_target = self.R + GAMMA * q_
        #定义critic网络更新用的td-error
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        #定义critic网络的优化方式:Adam ,最小化td-error
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        #定义actor网络的loss,最大化Q(s,a),解释https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-DDPG/
        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace_actor = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
            self.hard_replace_critic = [tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)]
        else:
            self.soft_replace_actor = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.at_params, self.ae_params)]
            self.soft_replace_critic = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.ct_params, self.ce_params)]
    #用s作为输入,从actor_eval中选取action
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        #随机BATCH_SIZE个索引
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        #取出这些样本的s,a,r,s_
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        #更新actor_eval网络,里面用的a是actor网络的a.
        self.sess.run(self.atrain, {self.S: bs})
        #更新critic_eval网络
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    #更新target网络参数,soft更新或者hard更新
    def update_target_network(self):
        if self.replacement['name'] == 'soft':
            self.sess.run([self.soft_replace_actor,self.soft_replace_critic])
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run([self.hard_replace_actor,self.hard_replace_critic])
                self.t_replace_counter+=1       

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    #定义actor网络    
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            #30个隐藏层
            l1 = tf.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            a = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable, kernel_initializer=w_init) #tanh [-1,1]
            # a * a_bound , a是[-1,1]之间的值,乘以a的上界,意思是放大网络输出到action的连续值上.
            return tf.multiply(a, self.a_bound, name='scaled_a')

    #定义critic网络        
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2_dim = 300
            w1_s = tf.get_variable('w1_s', [400, l2_dim], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, l2_dim], trainable=trainable)
            b1 = tf.get_variable('b1', [1, l2_dim], trainable=trainable)
            #注意这里是将state和action都作为输入,然后输出Q(s,a)
            net = tf.nn.relu(tf.matmul(l1, w1_s) + tf.matmul(a, w1_a) + b1)
            #从别的代码学来的,这样初始化能加速收敛
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            return tf.layers.dense(net, 1, trainable=trainable, kernel_initializer=w_init)  # Q(s,a)

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(123)

s_dim = env.observation_space.shape[0] # 3
a_dim = env.action_space.shape[0]    # 1
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

#这里初始化一个方差用于增强 actor 的探索性
var = 3  

ddpg.update_target_network()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        # 增强探索性,从均值为a,方差为3的正态中抽样,设置范围不能超过-2和2,超过就置为-2或2.
        a = np.clip(np.random.normal(a, var), -2, 2)
        #另一种增强探索性的方式a = a + 1.0 / (1.0 + i)    
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r, s_)
        #记忆库投一次满了以后
        if ddpg.pointer > MEMORY_CAPACITY:
            #降低方差,降低探索性
            var *= .9995    
            ddpg.learn()
            ddpg.update_target_network()

        s = s_
        ep_reward += r
        #这个游戏没有terminal,所有到指定steps后就结束,重头来.
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.6f' % var, )
            if ep_reward > -300:RENDER = True
            break