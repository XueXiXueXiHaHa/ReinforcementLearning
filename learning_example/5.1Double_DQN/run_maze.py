# -*- coding: utf-8 -*-
"""
名称:迷宫游戏
难度:*
算法:DoubleDQN
技术点:
    1.e-greedy
    2.两个网络,eval网络训练更新参数,target网络用与输出label Q值
描述:
    红色方块是可以移动的,四个action:上下左右,移动到黄色圆圈算赢,黑色方框是阻碍.
    reward:走一步奖励0,走到黑色方框-1,走到黄色圆圈+1
基于:
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-5-double_DQN/
运行环境:
    python 2.7.11
    tensorflow 1.0
"""
from maze_env import Maze
from RL_brain import Double_DQN
import matplotlib.pyplot as plt

NUM_EPISODE = 50
OBERSERVE_EPISODE = 20

def run_maze():
    plt.figure(figsize=(16,6)) 
    
    total_reward = 0
    total_steps = 0
    avg_reward_list = []
    avg_steps_list = []
    
    step = 0    # 用来控制什么时候学习
    for episode in range(NUM_EPISODE):
        # 初始化环境,observation是一个二维向量,类似坐标[ 0.25 -0.5 ]
        observation = env.reset()
        
        while True:
            # 刷新环境
            env.render()

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习) ,先走200步,然后每隔5步更新一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state_ 变为 下次循环的 state
            observation = observation_
            
            total_steps+=1
            total_reward+=reward
            # 如果终止, 就跳出循环
            if done:
                if (episode+1) % OBERSERVE_EPISODE==0:
                    print("Episode:{} Avg reward:{} Avg steps:{}".format(episode,total_reward*1.0/OBERSERVE_EPISODE,total_steps*1.0/OBERSERVE_EPISODE))
                    avg_reward_list.append(total_reward*1.0/OBERSERVE_EPISODE)
                    avg_steps_list.append(total_steps*1.0/OBERSERVE_EPISODE)                    
                    total_reward=0
                    total_steps=0                   
                    #print(RL.q_table)
                break
            step += 1   # 总步数
            
        
    x_ax = [ i*OBERSERVE_EPISODE for i in range(len(avg_reward_list))]
    plt.subplot(121)
    plt.xlabel('epsiode')
    plt.ylabel('avg reward per 5 epsiode')       
    plt.plot(x_ax,avg_reward_list,color="blue", linewidth=2.5, linestyle="-",label="Double-DQN",marker='D')
    plt.subplot(122)
    plt.xlabel('epsiode')
    plt.ylabel('avg steps per 5 epsiode')       
    plt.plot(x_ax,avg_steps_list,color="blue", linewidth=2.5, linestyle="-",label="Double-DQN",marker='D')
    plt.savefig("Double-DQN.png",dpi=100)
    
    # end of game
    print('game over')
    plt.show()
    env.destroy()
    

if __name__ == "__main__":
    env = Maze()
    RL = Double_DQN(env.n_actions,env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=False
                      )
                      #replace_target_iter=200 每200步替换一次target_net的参数
    env.after(100, run_maze)
    env.mainloop()
    #RL.plot_cost()  # 观看神经网络的误差曲线