#coding:utf8
"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    
    Story the data with it priority in tree and data frameworks.
    """
    #指向当前空着的数据位置.
    data_pointer = 0

    def __init__(self, capacity):
                """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
    
        Array type for storing:
        [0,1,2,3,4,5,6]
        #这个例子中capacity=4,前3个位置存储父节点,后4个位置存储叶子节点
        """

        #叶子节点数
        self.capacity = capacity    
        #由于是二叉树,所以最大节点数为2^n-1,n是层数.最后一层节点数2^(n-1),除了最后一层的节点数为2^(n-1)-1.所以最后一层永远比之前所有层的节点之和多1个.
        #当叶子节点数为capacity时,之前所有层的节点之和不会超过capacity-1,所以这里树的节点为2*capacity-1
        #前capacity-1个位置存储父节点,后capacity 个位置存储叶子节点,每个节点存储权重值p.
        self.tree = np.zeros(2*capacity - 1) 

        #存储叶子节点的值[s,a,r,s_],也就是所说的transitions
        self.data = np.zeros(capacity, dtype=object)    
    #添加新的样本
    def add_new_priority(self, p, data):
        #capacity - 1 是存储叶子节点的起始位置.data_pointer指向当前空着的数据位置
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    #更新树叶子节点的权重p
    def update(self, tree_idx, p):
        #修改父节点权重用
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    #递归修改树各父节点的权重值 
    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    #根据lower_bound进行采样,返回[叶子索引,权重,值]        
    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]
    
    #通过lower_bound来进行递归采样,返回节点的索引
    #lower_bound 是用于抽样的数
    def _retrieve(self, lower_bound, parent_idx=0):
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1
        #到达叶子节点,返回父节点索引
        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx
        #如果左右节点的值相同,随机选一个递归下去    
        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        #如果左节点的值大于等于 lower_bound,向左递归
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        #如果左节点的值小于 lower_bound,向右递归,lower_bound = lower_bound - left_value
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root


class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6     # [0~1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    
    #添加新的样本到sumTree中    
    def store(self, transition):
        #取叶子节点(当前所有样本中)最大的权重值,新来的样本用最大的p,保证优先被采样到,然后再通过网络更新它的p.
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    #抽取n个样本:https://arxiv.org/pdf/1511.05952.pdf    
    #用随机更新估计的期望值依赖于于这些更新有与其期望相同的分布,Prioritized replay引入了偏差.所以需要通过importance-sampling(IS) weights来修正这个bias
    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        #所有权重之和除以采样大小n,分成n个区间,segment是区间大小
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1
        #叶子节点中最小的抽样概率
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        #最大的Wi,后面正则化使用
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            #区间segment * i到segment * (i + 1) 之间随机一个数
            lower_bound = np.random.uniform(a, b)
            #抽取一个样本
            idx, p, data = self.tree.get_leaf(lower_bound)
            #计算此样本概率
            prob = p / self.tree.root_priority

            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)
        #把数组变成 n*1的矩阵    
        ISWeights = np.vstack(ISWeights)
        #正则化样本权重,Wi = (N * Pi)^(-beta) / maxWi
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights
    #根据更新后eval网络输出的各抽样样本的td-error,更新树中各抽样样本的权重p    
    def update(self, idx, error):
        
        p = self._get_priority(error)
        self.tree.update(idx, p)
    #根据td-error计算样本权重.(td_error + epsilon)^alpha,alpha起到缩放的功能
    #https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/   
    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        #限制一下error的上下界[0,abs_err_upper] ,小于0的置为0,大于upper的置为upper
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        #声明输入state和Q_target
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        #声明eval网络的各种参数,并声明eval网络        
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
                
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)
        #声明eval网络的loss    
        with tf.variable_scope('loss'):
            if self.prioritized:
                #绝对值td-error,用于update sumtree
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)
                #带权重的均方误差,权重是这个样本之前的td-error
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        #定义优化方法为RMSProp,最小化目标loss        
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)
    
    #保存样本        
    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            #对于新来的transition,用最高的优先级p来进行存储,这个时候是没有算它的td-error的..
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
    #e-greedy选择action        
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    #将eval网络的参数替换到target网络
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    #eval网络学习    
    def learn(self):
        #是否进行参数替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')
        #采样,返回树节点的索引,样本列表,样本权重
        #节点的索引用于更新节点的td-error,也就是p. 样本列表用于更新网络,进行训练.样本权重用于计算loss.
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
        #获取网络输出    
        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})
        #计算q_target,这里的解释见DQN代码中的注释.具体就是只eval网络中对应action的Q(s,a)改成r+gamma*maxQ(s,a).这样在网络里一减,只有这个action位置变为td-error,其他位置都是0                   
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        #如果设置了优先级采样,需要返回这些样本在当前eval网络中的td-error,然后更新tree
        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            #依次更新样本在树种的节点权重p                                                
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.memory.update(idx, abs_errors[i])
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)
        #如果设置了epsilon自增,这里会变化
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
