# coding:utf8
"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import random
import signal
import matplotlib.pyplot as plt

import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import math

# 平衡车
GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
# N_WORKERS=1

GLOBAL_NET_SCOPE = 'Global_Net'
MAX_GLOBAL_EP = 20000000
UPDATE_GLOBAL_ITER = 64
GAMMA = 0.99
ENTROPY_BETA = 0.01
LearnRate = 1e-4  # learning rate
# LearnRate_a=1e-4
# LearnRate_c=1e-4
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

N_A = 2

# global 和worker的同步锁
update_global_lock = threading.Lock()
# 各个worker总共进行的epsiode计数的锁
global_epsiode_lock = threading.Lock()

stop_requested = False


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        # self.check_op = tf.add_check_numerics_ops()

        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, 80, 80, 4], 'S')
                self._build_net(scope)
                self.ac_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor_critic')
                self.a_params = tf.get_collection(scope + "/actor_params")
                self.c_params = tf.get_collection(scope + "/critic_params")
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, 80, 80, 4], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, 1], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net(scope)

                td_error = tf.subtract(self.v_target, self.v, name='TD_error')

                self.c_loss = 0.5 * tf.reduce_mean(tf.square(td_error))

                # with tf.name_scope('a_loss'):
                # 对真实采取的action的概率求log,其他action置为0
                # avoid NaN with clipping when value in pi becomes zero
                self.a_prob = tf.clip_by_value(self.a_prob, 1e-20, 1.0)
                log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1,
                                         keep_dims=True)
                exp_v = log_prob * td_error
                entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1,
                                         keep_dims=True)  # encourage exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)

                self.total_loss = self.a_loss + self.c_loss

                self.ac_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor_critic')
                optimizer = tf.train.RMSPropOptimizer(LearnRate)
                # optimizer_a = tf.train.RMSPropOptimizer(LearnRate_a, name='Adam')
                # optimizer_c = tf.train.RMSPropOptimizer(LearnRate_c, name='Adam')
                # 采用传递梯度的方式更新Global网络a
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(scope+"/actor_params")
                    self.c_params = tf.get_collection(scope+"/critic_params")
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    #self.ac_grads = tf.gradients(self.total_loss, self.ac_params)

                with tf.name_scope('sync'):
                    with tf.name_scope('pull'):
                        self.pull_ac_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                  zip(self.ac_params, globalAC.ac_params)]
                        #self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                        #self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    with tf.name_scope('push'):
                        self.update_a_op = optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                        self.update_c_op = optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))
                        #self.update_ac_op = optimizer.apply_gradients(zip(self.ac_grads, globalAC.ac_params))
                        # pass

    def _build_net(self, scope):
        with tf.name_scope('actor_critic'):
            # 输入s,输出各个action的概率(向量)
            self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
            self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32])  # stride=2
            self.W_fc1, self.b_fc1 = self._fc_variable([2048, 256])

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._fc_variable([256, N_A])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

            tf.add_to_collection(scope + "/actor_params", self.W_conv1)
            tf.add_to_collection(scope + "/actor_params", self.b_conv1)
            tf.add_to_collection(scope + "/actor_params", self.W_conv2)
            tf.add_to_collection(scope + "/actor_params", self.b_conv2)
            tf.add_to_collection(scope + "/actor_params", self.W_fc1)
            tf.add_to_collection(scope + "/actor_params", self.b_fc1)
            tf.add_to_collection(scope + "/actor_params", self.W_fc2)
            tf.add_to_collection(scope + "/actor_params", self.b_fc2)

            tf.add_to_collection(scope + "/critic_params", self.W_conv1)
            tf.add_to_collection(scope + "/critic_params", self.b_conv1)
            tf.add_to_collection(scope + "/critic_params", self.W_conv2)
            tf.add_to_collection(scope + "/critic_params", self.b_conv2)
            tf.add_to_collection(scope + "/critic_params", self.W_fc1)
            tf.add_to_collection(scope + "/critic_params", self.b_fc1)
            tf.add_to_collection(scope + "/critic_params", self.W_fc3)
            tf.add_to_collection(scope + "/critic_params", self.b_fc3)
            # actor和critic公用前面的三层网络
            self.h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)  # 19 19 32
            # self.h_pool1 = self._max_pool_2x2(self.h_conv1) # 10 10 16

            self.h_conv2 = tf.nn.relu(self._conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)  # 4 4 32

            #  self.h_conv3 = tf.nn.relu(self._conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)  # 4 4 32

            self.h_conv3_flat = tf.reshape(self.h_conv2, [-1, 2048])

            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            a_prob = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

            # value (output)
            v_ = tf.matmul(self.h_fc1, self.W_fc3) + self.b_fc3
            v = tf.reshape(v_, [-1, 1])

        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        update_global_lock.acquire()
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        #SESS.run([self.update_ac_op], feed_dict)
        update_global_lock.release()

    def pull_global(self):  # run by a local
        # 加锁,这里主要是防止有人在更新时,另一边在拉取.保证更新完了再拉取
        update_global_lock.acquire()
        #SESS.run([self.pull_a_params_op,self.pull_c_params_op])
        SESS.run([self.pull_ac_params_op])
        update_global_lock.release()

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: [s]})
        # print prob_weights
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        # d = 0.05
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        # d = 0.05
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")


class Worker(object):
    def __init__(self, name, globalAC, rand_seed):
        # 实例化一个FlappyBird游戏
        self.env = game.GameState(rand_seed, show_score=False)
        # self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def process_image_begin(self, x_t):
        # 把env返回的state变成80*80的灰度图
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        # 这里要把像素二值化,把像素值大于1的都置为255,也就是说把不是黑色0的地方都置为白色255
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        # 变成80*80*4 ,由于是初始装填,所以只能拿4个相同的图像
        s_t1 = np.stack((x_t, x_t, x_t, x_t), axis=2)
        return s_t1

    def process_image_steps(self, x_t, s_t):
        x_t1 = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        return s_t1

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, stop_requested
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            if self.name == 'W_0':
                print "Begin"
            do_nothing = np.zeros(N_A)
            # acton=[1,0] 是什么也不干,action = [0,1] 是往上蹦一下
            do_nothing[0] = 1
            x_t, r_0, terminal = self.env.frame_step(do_nothing)
            s = self.process_image_begin(x_t)

            ep_r = 0
            action_count = [0, 0]
            passed_pipe = 0
            while not stop_requested:
                a = self.AC.choose_action(s)

                # a_t为actions长度的全0列表,选中那个action就在指定位置置为1.
                a_t = np.zeros([N_A])

                a_t[a] = 1

                action_count[a] += 1

                x_t1_colored, r, done = self.env.frame_step(a_t)
                # 处理一下返回的图像,并把图像添加到S_t,把最老的那一祯挤掉
                s_ = self.process_image_steps(x_t1_colored, s)

                ep_r += r
                if r >= 1:
                    passed_pipe += 1

                buffer_s.append([s])
                buffer_a.append([a])
                buffer_r.append(r)

                s = s_
                total_step += 1

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: [s_]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s = np.vstack(buffer_s)  # shape = n 80 80 4
                    buffer_a, buffer_v_target = np.array(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    self.AC.pull_global()

                    if done and self.name == 'W_0':
                        # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        #     GLOBAL_RUNNING_R.append(ep_r)
                        # else:
                        #     GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                        print(
                            "Thread: {}".format(self.name),
                            "Epsiode: %d" % GLOBAL_EP,
                            "Worker_steps: %d" % total_step,
                            "Epsiode_reward: %f" % ep_r,
                            "Action 0: %d" % action_count[0],
                            "Action 1: %d" % action_count[1],
                            "Passed Pipe: {}".format(passed_pipe)
                        )
                        action_count = [0, 0]
                        ep_r = 0
                        passed_pipe = 0
                    if done:
                        global_epsiode_lock.acquire()
                        GLOBAL_EP += 1
                        global_epsiode_lock.release()
                        #break


def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True


if __name__ == "__main__":
    SESS = tf.Session()
    # random.seed(1)
    # np.random.seed(1)
    # tf.set_random_seed(1)

    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        randint = random.randint(0, 100)
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC, i * randint))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    signal.signal(signal.SIGINT, signal_handler)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.savefig('flappybird_A3C.jpg')
# plt.show()


