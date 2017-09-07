#coding:utf8
"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.3              pip install --upgrade tensorflow
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, Queue

EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4  # parallel workers
GAMMA = 0.9  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.001  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 5  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1  # state and action dimension


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        w_init = tf.random_normal_initializer(0., .1)
        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, kernel_initializer=w_init)
        self.v = tf.layers.dense(l1, 1,kernel_initializer=w_init)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER

        while not COORD.should_stop() :

            if GLOBAL_EP < EP_MAX:

                #等待worker发送update事件
                UPDATE_EVENT.wait()    # wait until get batch of data
                #将现在的actor网络参数保存到old_actor网络中
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                print QUEUE.qsize(),
                #从queue队列中取出训练数据
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                print QUEUE.qsize()
                data = np.vstack(data) #将列表转换成矩阵,n行
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]

                #计算advantage
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # 多次更新actor和critic.
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]


                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # 重置更新计数器
                ROLLING_EVENT.set()  # set roll-out available


                #print "Update Over"

    def _build_anet(self, name, trainable):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu6, trainable=trainable, kernel_initializer=w_init)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable, kernel_initializer=w_init)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable, kernel_initializer=w_init)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER

        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []


            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                #如果几个worker产生的样本已经>=MIN_BATCH_SIZE ,则把这些样本加入到Queue中,开始更新网络.
                # GLOBAL_UPDATE_COUNTER这个参数是几个worker公用的,所以push到Queue这个动作几个worker都会触发.但是有可能一个worker在push的时候,另一个还在产生数据,这样有小概率触发两次更新?
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    #把当前worker生成的样本
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue


                    #当queue里达到一定的样本量,开始更新网络.
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                         ROLLING_EVENT.clear()       # stop collecting data
                         UPDATE_EVENT.set()          # globalPPO update
                         #  等待所有的worker都将各自的样本put到Queue中去以后再update????

                    #超过EP_MAX设定的epsiode数量,停止训练
                    if GLOBAL_EP >= EP_MAX:
                        print "call traing stop!"
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print "Epsiode : %d" % GLOBAL_EP
            #print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r,)


if __name__ == '__main__':
    #创建全局PPO网络实例,worker就是用这个网络来生成epsiode.
    GLOBAL_PPO = PPO()

    #threading.Event机制类似于一个线程向其它多个线程发号施令的模式，其它线程都会持有一个threading.Event的对象，这些线程都会等待这个事件的“发生”，如果此事件一直不发生，那么这些线程将会阻塞，直至事件的“发生”。
    #clear()是清空事件, set()是触发事件 ,wait()是阻塞等待事件.    (这就相当于一个变量,clear相当于e=0,set相当于e=1,wait 相当于 while e==1:...)
    #UPDATE_EVENT 是触发更新网络的事件, ROLLING_EVENT 是继续产生数据的事件
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # 停止update
    ROLLING_EVENT.set()  # 先开始episode

    #创建worker实例
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    #全局更新次数 , 全局epsiode数.
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []

    # 设置多线程协调器
    COORD = tf.train.Coordinator()
    #存放训练数据的
    QUEUE = Queue.Queue()  # workers putting data in this queue

    #--------------------启动worker程序
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode');
    plt.ylabel('Moving reward');
    plt.ion();
    plt.show()
    plt.savefig("performence_after.jpg")

    '''
    #看一下训练好的模型的效果
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]
    '''