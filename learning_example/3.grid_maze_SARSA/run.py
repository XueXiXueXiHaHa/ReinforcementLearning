# -*- coding: utf-8 -*-
"""
名称:迷宫游戏
难度:*
算法:SARSA
技术点:
    1.e-greedy
    2.抽象化基本功能,通过继承方式实现.
描述:
    红色方块是可以移动的,四个action:上下左右,移动到黄色圆圈算赢,黑色方框是阻碍.
    reward:走一步奖励0,走到黑色方框-1,走到黄色圆圈+1
基于:
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-2-tabular-sarsa2/
运行环境:
    python 2.7.11
"""
from __future__ import print_function
from maze_env import Maze
from RL_brain import SarsaTable
import matplotlib.pyplot as plt
import numpy as np

NUM_EPISODE = 200
OBERSERVE_EPISODE = 5

def update():
    plt.figure(figsize=(16,6)) 
    
    total_reward = 0
    total_steps = 0
    avg_reward_list = []
    avg_steps_list = []
    for episode in range(NUM_EPISODE):
        # 初始化环境
        observation = env.reset()

        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))
        #print(" Episode:{}".format(episode),end='')
        total_steps+=1 
        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)
            
            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_
            
            total_steps+=1
            total_reward+=reward
            # 终止时跳出循环
            if done:
                if (episode+1) % OBERSERVE_EPISODE==0:
                    #print(" Episode:{} Avg reward:{} Avg steps:{}".format(episode,total_reward*1.0/OBERSERVE_EPISODE,total_steps*1.0/OBERSERVE_EPISODE),end='\n')
                    avg_reward_list.append(total_reward*1.0/OBERSERVE_EPISODE)
                    avg_steps_list.append(total_steps*1.0/OBERSERVE_EPISODE)                    
                    total_reward=0
                    total_steps=0                   
                    #print(RL.q_table)
                break
    x_ax = np.arange(5,NUM_EPISODE+1,OBERSERVE_EPISODE)
    plt.subplot(121)
    plt.xlabel('epsiode')
    plt.ylabel('avg reward per 5 epsiode')       
    plt.plot(x_ax,avg_reward_list,color="blue", linewidth=2.5, linestyle="-",label="SARSA-lambda",marker='D')
    plt.subplot(122)
    plt.xlabel('epsiode')
    plt.ylabel('avg steps per 5 epsiode')       
    plt.plot(x_ax,avg_steps_list,color="blue", linewidth=2.5, linestyle="-",label="SARSA-lambda",marker='D')
    
    plt.show()        
    # 大循环完毕
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()