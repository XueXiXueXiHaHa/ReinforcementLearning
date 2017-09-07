# -*- coding: utf-8 -*-
"""
名称:汽车爬山游戏
难度:*
算法:DQN
技术点:
    1.e-greedy 自增e_greedy_increment
    2.两个网络,eval网络训练更新参数,target网络用与输出label Q值
    3.gym
描述:
   通过左右移动,使一个小车爬上山去
基于:
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/
运行环境:
    python 2.7.11
    tensorflow 1.0
"""
import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')   # 定义使用 gym 库中的那一个环境
env = env.unwrapped # 不做这个会有很多限制

print("action num : "+str(env.action_space)) # 查看这个环境中可用的 action 有多少个,这里有3个
print("observation num  : "+str(env.observation_space) )   # 查看这个环境中可用的 state 的 observation 有多少个,state的维度为2
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

# 定义使用 DQN 的算法
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, 
                  e_greedy=0.9,
                  replace_target_iter=100, 
                  memory_size=2000,
                  e_greedy_increment=0.0008,)

start_learn_steps = 0 # 记录步数

import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))

NUM_EPISODE = 100
OBERSERVE_EPISODE = 20

#这四个是用来画收敛图用的.
total_reward = 0
total_steps = 0
avg_reward_list = []
avg_steps_list = []

for episode in range(NUM_EPISODE):
     
    # 获取回合 i_episode 第一个 observation
    observation = env.reset()
    
    
    while True:
        env.render()    # 刷新环境

        action = RL.choose_action(observation)  # 选行为

        observation_, reward, done, info = env.step(action) # 获取下一个 state
        #位置,速度
        position, velocity = observation_

        # 车开得越高 reward 越大
        reward = abs(position - (-0.5))
        
        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        observation = observation_

        start_learn_steps += 1
        
        
        if start_learn_steps > 1000:
            RL.learn()  # 学习

        total_steps+=1
        total_reward+=reward
        
        if done:
            if (episode+1) % OBERSERVE_EPISODE==0:
                    print("Episode:{} Avg reward:{} Avg steps:{} Epsilon:{}".format(
                            episode,
                            round(total_reward*1.0/OBERSERVE_EPISODE,4),
                            total_steps*1.0/OBERSERVE_EPISODE,
                            round(RL.epsilon, 2)
                              )
                          )
                         
                    avg_reward_list.append(total_reward*1.0/OBERSERVE_EPISODE)
                    avg_steps_list.append(total_steps*1.0/OBERSERVE_EPISODE)                    
                    total_reward=0
                    total_steps=0             

            break
x_ax = [ i*OBERSERVE_EPISODE for i in range(len(avg_reward_list))]
plt.subplot(121)
plt.xlabel('epsiode')
plt.ylabel('avg reward per 5 epsiode')       
plt.plot(x_ax,avg_reward_list,color="blue", linewidth=2.5, linestyle="-",label="DQN-CartPole",marker='D')
plt.subplot(122)
plt.xlabel('epsiode')
plt.ylabel('avg steps per 5 epsiode')       
plt.plot(x_ax,avg_steps_list,color="blue", linewidth=2.5, linestyle="-",label="SARSA-lambda",marker='D')
#plt.savefig("DQN.png",dpi=100)
# end of game
print('game over')
plt.show()
      
# 最后输出 cost 曲线
RL.plot_cost()