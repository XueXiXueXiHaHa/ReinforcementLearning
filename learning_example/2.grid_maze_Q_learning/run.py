# -*- coding: utf-8 -*-
"""
名称:迷宫游戏
难度:*
算法:Q-learning
技术点:
    e-greedy
描述:
    红色方块是可以移动的,四个action:上下左右,移动到黄色圆圈算赢,黑色方框是阻碍.
    reward:走一步奖励0,走到黑色方框-1,走到黄色圆圈+1
基于:
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-2-tabular-q1/
运行环境:
    python 2.7.11
"""

from maze_env import Maze
from RL_brain import QLearningTable
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
        # 初始化 state 的观测值
        observation = env.reset()
       # print("episode:%d" %(episode)) 
        while True:
            # 更新可视化环境
            env.render()
   #         print "1.choose action"
            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))
  #          print "2.take action"
            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)
  #          print "3.update Q value"
            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将下一个 state 的值传到下一次循环
            observation = observation_
            
            total_steps+=1
            total_reward+=reward
            # 如果掉下地狱或者升上天堂, 这回合就结束了
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
    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()