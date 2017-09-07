# -*- coding: utf-8 -*-

"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # action list
        self.lr = learning_rate # 学习率a
        self.gamma = reward_decay #奖励衰减 gamma
        self.epsilon = e_greedy #贪婪度
        self.q_table = pd.DataFrame(columns=self.actions) #初始Q table
        
    #通过e-greedy 的方式选择action    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action选择 e-greedy
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]            
            #这一句的意义就是在Q初始都为0的时候,选max能选到不同的action,不加这句在最开始时就会老往一个方向走,更新不了Q.
            state_action = state_action.reindex(np.random.permutation(state_action.index))    #permutation作用是打乱顺序,洗牌.
            action = state_action.argmax() #argmax 返回value最大的index,也就是action的名字
        else:
            #随机选择action
            action = np.random.choice(self.actions)
        return action
    
    #更新Q-table    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
    
    #如果新的state不在Q-table中,添加.
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )