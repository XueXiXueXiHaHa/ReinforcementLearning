
# -*- coding: utf-8 -*-
"""
名称:单行移动游戏
难度:*
算法:Q-learning
技术点:
    e-greedy
描述:
    游戏环境为:--o--T
    其中T为终点,o是我们的agent,通过左右移动到T的位置为胜利,游戏结束.到达T奖励为1,否则为0.
    初始点在最左边:o----T,一共6个状态.
基于:
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
运行环境:
    python 2.7.11
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import time


N_STATES = 6   # 状态的数量
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy,以e的概率采用历史最好动作explortation,以1-e的概率随机采取动作exploration
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 15   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间,单位秒.用于展示交互过程用,此参数与算法无关.


#创建Q(S,A)结构体,用于存储value值
# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    return table

# 在某个 state 下, 选择行为a
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    #完全随机的方式选择,收敛慢   
    #action_name = np.random.choice(ACTIONS)
    #return action_name
    
    #e-greedy 方式,经典方式. 
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()    # #argmax 返回value最大的index,也就是action的名字
    return action_name

#获取环境奖励reward和下一个state,在这里设置为到达终点奖励1,其余都奖励0.    
def get_env_feedback(S, A):
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

#输出环境
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

#核心学习函数
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始化 q table
    
    for episode in range(MAX_EPISODES):     #回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        
        #####################
        #   核心算法部分
        #####################
        #只要没有到达terminat状态,就一直采取动作
        while not is_terminated:
            A = choose_action(S, q_table)   # 1.为当前状态S(t),选行动a
            S_, R = get_env_feedback(S, A)  # 2.实施行动并得到环境的反馈R,S(t+1)
            q_predict = q_table.ix[S, A]    # 3.获取old Q(S,A)值
            #4.计算Gt
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode
            #5.更新Q(S,A)    
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
        #print(q_table)
        #greedy policy
        #for i in range(len(q_table)):
        #    print("%d  %s" % (i,q_table.iloc[i,].argmax())) 
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)