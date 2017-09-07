#coding:utf8
"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0


关于PolicyGradient的解释看https://zhuanlan.zhihu.com/p/27699682?group_id=872490079585714176

"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        #reward衰减系数
        self.gamma = reward_decay 
        # 这是我们存储 回合信息的 list
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        # 建立 policy 神经网络
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        tf.reset_default_graph()
        #网络输入
        with tf.name_scope('inputs'):
            #这3个都是列表格式,其中obs格式是[[f1,f2,...],[f1,f2,..],...]
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        #定义第1隐层: 输入为obs,隐层神经元10个,激活函数tanh,w用正态分布初始化,b为0.1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        #定义第2隐层: 输入为layer(第一隐层的输出),输出神经元n_actions个,w用正态分布,b=0.1
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        #用softmax将输出值概率化,大小bitch_size * n_actions
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            
            #或者这么写,这个比较好理解,  -(bitch_size*n_actions) * (bitch_size*n_actions) (注意这里是*,不是矩阵乘法),
            #由于后面和这个矩阵是one-hot的,所以对于某个样本只有特定action位置才有log值,其它都是0    
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    
    #通过将observation输入policy网络,得到各个action的概率分布(求和为1),然后根据概率进行抽样        
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]}) #np.newaxis 就是把列表变矩阵
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    
    #这一部很简单, 就是将这一步的 observation, action, reward 加到列表中去. 
    #因为本回合完毕之后要清空列表, 然后存储下一回合的数据, 所以我们会在 learn() 当中进行清空列表的动作    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    
    #学习更新policy网络
    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode ,注意这里的三个输入的大小是这个epsiode的step数
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm
    
    #计算没一次action 的discount reward,并对reward 做标准化    
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        #反向计算dicounted reward,然后放到新的列表中discounted_ep_rs
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 良好的思路是要先对这些回报做“标准化”（比如减去平均值，除以标准差），然后再把它们扔进反向传播过程中。
        # 通过这样做，我们能够总是差不多对一半的动作进行鼓励，对一半的动作进行惩罚。
        # 从数学角度来说，这些做法是控制策略梯度预测器的方差的方法。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



