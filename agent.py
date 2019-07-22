import numpy as np
from collections import defaultdict, deque
import sys
import random
import matplotlib.pyplot as plt
import math


class Agent:
    def __init__(self, env, nA, alpha, gamma, num_ep, eps_start, eps_decay,
                 eps_min):

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.nA = env.action_space.n
        self.eps = 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.num_ep = num_ep
        self.env = env
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def select_action(self, state):
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step_q_l(self, state, action, reward, next_state):

        current_q = self.Q[state][action]
        next_q = np.max(self.Q[next_state]) if next_state is not None else 0
        target_q = reward + (self.gamma * next_q)
        new_q = current_q + (self.alpha * (target_q - current_q))
        return new_q

    def step_expecsarsa(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.eps / self.nA
        policy_s[np.argmax(
            self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
        next_q = np.dot(self.Q[next_state], policy_s)
        target_q = reward + (self.gamma * next_q)
        new_q = current_q + (self.alpha * (target_q - current_q))
        return new_q

    def q_learning(self, plot_every=100):
        # print(plot_every)

        temp_scores = deque(maxlen=plot_every)
        avg_scores = deque(maxlen=self.num_ep)
        best_avg_reward = -math.inf
        self.eps = self.eps_start
        for i_ep in range(1, self.num_ep + 1):
            if i_ep % 100 == 0:
                # print("\rEpisode {}/{}".format(i_ep, self.num_ep), end="")
                sys.stdout.flush()
            score = 0
            state = self.env.reset()
            self.env.render()


            # self.eps = max(self.eps * self.eps_decay, self.eps_min)
            self.eps = 1.0 / i_ep

            while True:
                action = self.select_action(state)
                next_state, reward, done, infp = self.env.step(action)
                score += reward
                if not done:
                    next_action = self.select_action(state)
                    self.Q[state][action] = self.step_q_l(
                        state, action, reward, next_state)
                    state = next_state
                if done:
                    temp_scores.append(score)
                    break

            # if (i_ep % plot_every == 0):
            # avg_scores.append(np.mean(temp_scores))
            if (i_ep >= 100):
                # get average reward from last 100 episodes
                avg_score = np.mean(temp_scores)
                # append to deque
                avg_scores.append(avg_score)
                # update best average reward
                if avg_score > best_avg_reward:
                    best_avg_reward = avg_score
            print(
                "\rEpisode {}/{} || Best average reward {}".format(
                    i_ep, self.num_ep, best_avg_reward),
                end="")
            sys.stdout.flush()

        plt.plot(
            np.linspace(0, self.num_ep, len(avg_scores), endpoint=False),
            np.asarray(avg_scores))
        # plt.xlabel('Ep No')
        # plt.ylabel('Avg reward over next {} episodes'.format(plot_every))
        plt.show()

        return self.Q, best_avg_reward

    def expecsarsa(self, plot_every=100):
        temp_scores = deque(maxlen=plot_every)
        avg_scores = deque(maxlen=self.num_ep)
        best_avg_reward = -math.inf
        self.eps = self.eps_start
        for i_ep in range(1, self.num_ep + 1):
            if i_ep % 100 == 0:
                print("\rEpisode {}/{}".format(i_ep, self.num_ep), end="")
                sys.stdout.flush()
            score = 0
            state = self.env.reset()

            self.eps = max(self.eps * self.eps_decay, self.eps_min)
            # self.eps = 1.0 / i_ep

            while True:
                action = self.select_action(state)
                next_state, reward, done, infp = self.env.step(action)
                score += reward
                if not done:
                    next_action = self.select_action(state)
                    self.Q[state][action] = self.step_expecsarsa(
                        state, action, reward, next_state)
                    state = next_state
                if done:
                    temp_scores.append(score)
                    break

            # if (i_ep % plot_every == 0):
            # avg_scores.append(np.mean(temp_scores))
            if (i_ep >= 100):
                # get average reward from last 100 episodes
                avg_score = np.mean(temp_scores)
                # append to deque
                avg_scores.append(avg_score)
                # update best average reward
                if avg_score > best_avg_reward:
                    best_avg_reward = avg_score
            print(
                "\rEpisode {}/{} || Best average reward {}".format(
                    i_ep, self.num_ep, best_avg_reward),
                end="")
            sys.stdout.flush()

        plt.plot(
            np.linspace(0, self.num_ep, len(avg_scores), endpoint=False),
            np.asarray(avg_scores))
        # plt.xlabel('Ep No')
        # plt.ylabel('Avg reward over next {} episodes'.format(plot_every))
        plt.show()

        return self.Q, best_avg_reward
