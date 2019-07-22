from agent import Agent
import gym
import numpy as np
import sys

env = gym.make('Taxi-v2')
env.render()
print(env.action_space)
print(env.observation_space)

average_score = 0

for a in range(10):

    agent = Agent(env, 6, .1, .9, 20000, 1.0, .999, .05).q_learning()
    # agent = Agent(env, 6,.1,.9,20000,1.0,.999,.01).expecsarsa()
    Q_sarsamax = agent[0]
    policy_sarsamax = np.array([
        np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1
        for key in np.arange(48)
    ]).reshape(4, 12)

    # print(policy_sarsamax)

    V_sarsa = ([
        np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0
        for key in np.arange(48)
    ])
    # print(V_sarsa)
    average_score += agent[1]
    print('[INFO] ep_no: {}',a)
print('\n[INFO] {}'.format(average_score/100))


