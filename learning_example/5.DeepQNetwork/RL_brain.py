#coding:utf8
"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,    
            e_greedy=0.9,        
            replace_target_iter=300,#多少次更新以后,同步eval的参数给target网络
            memory_size=500,  #样本容器大小    
            batch_size=32,    #eval网络更新时的batch_size
            e_greedy_increment=None, #epsilon是否逐渐变大 
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 声明Deep Q网络,包括: [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        
        #实例化化网络结构参数.
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        #0 * 2,这里是网络的输入,用来接收 observation
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  #创建一个变量,只指定类型和大小,
        #用来接收 q_target 的值, 这个之后会通过计算得到       
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') 
        
        with tf.variable_scope('eval_net'): #variable_scope是给get_variable()创建的变量的名字加前缀
            # 是在更新 target_net 参数时会用到
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]#[collections_names],功能相当于一个变量集合,指定某些变量属于哪个集合,默认是get_variable新建的变量都会加到tf.GraphKeys.GLOBAL_VARIABLES中区. 
            #第一层的神经元数量
            n_l1 = 10
            #w参数初始化方法
            w_initializer = tf.random_normal_initializer(0., 0.3)
            #偏置项b的初始化方法
            b_initializer = tf.constant_initializer(0.1) 
            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                # 2 * 10
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                # 1 * 10                
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # relu (s * w1 + b1) 第一层隐层的输出 1 * 10
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                #10*4
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                #1*4
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                #估计的Q(s,a)值 l1 * w2 + b2, 大小为1*4 
                self.q_eval = tf.matmul(l1, w2) + b2
        #定义损失为均方误差         
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) # q_target外面算好传进来,它跟q_eval只有在max action上的值不同
        #定义梯度优化方式为RMSPro,学习率为self.lr,最优化目标为最小化self.loss
        with tf.variable_scope('train'):    
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
    
    #存储experience [s,a,r,s_] 用于抽样训练            
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):#判断self对象里面是否有memory_counter属性或者memory_counter方法
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_)) #hstack作用是横向拼接

        #保存样本到memory中,如果memory中满了,则把最老的替换掉.memory_counter是样本数计数器
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
    
    #e-greedy的方式选择ecal网络的最大Q(s,a)的action. 注意如果e_greedy_increment不为none,这里的epsilon是从0开始的一直加到epsilon_max,否则它=epsilon_max    
    def choose_action(self, observation):
        # 把列表变成1*n矩阵,用于网络的输入,feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    

    #将eval网络的参数更新到target网络上
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params') #从一个结合中取出全部变量，是一个列表
        e_params = tf.get_collection('eval_net_params')
        #把e的值赋给t
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)]) #zip的作用:按序号把2个列表的值拼成tuple,比如:zip([1,2],[3,4])=[(1,3),(2,4)]
    
    #从memory中抽取batch_size大小样本,更新eval网络,并在replace_target_iter次更新后,更新target网络
    def learn(self):
        #检查是否该替换target网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\r  target_params_replaced ')

        #从memory中抽样,大小为self.batch_size
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        #抽样样本    
        batch_memory = self.memory[sample_index, :]
        #训练网络,获得两个网络的输出q_next, q_eval
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        #---------------------------------------------------------------------------------------
        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.
        #---------------------------------------------------------------------------------------
        #这样做的目的是,只修改Q(S,A)列表,对应A的位置的值,其他位置相同,这样其他位置的loss(q_target-q_eval)就为0,不会反向传播
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #获取样本中action的下标 ,batch_memory 的格式是[s,s,a,r,s_,s_] ,由于state是两维的,所以它占两个位置
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        #获取reward, 1*n
        reward = batch_memory[:, self.n_features + 1]
        #把q_target中对应的Q(s,a)改成target值,r+gamma*maxQ(s,a). ,这样求loss时,用q_target - q_eval,除了最大的a那里有值,其它位置都是0.这个值就类似label和预测值之间的差值
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # 反向传播,更新eval网络参数, (这里传入输入,要重新前馈一遍??)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        #把batch_size样本的均方误差保存下来
        self.cost_his.append(self.cost)

        #增加epsilon的大小,意思是随着训练越多,exploration要收缩一下.增加到epsilon_max不再变.如果epsilon_increment=none,epsilon一直等于e_greedy
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    #打印网络的loss出来看,从这个结果看不出收敛来,没什么用.还是看greedy policy 下的mean reward
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
