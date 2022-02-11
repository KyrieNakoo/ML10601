import sys
import numpy as np
from environment import MountainCar


class QLearningAgent:
    def __init__(self, env, mode, gamma, lr, epsilon):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.w = np.zeros((3, self.env.state_space))
        self.b = 0

    def get_q(self, state):
        q = np.dot(self.w, state) + self.b
        return q

    def get_action(self, state):
        if np.random.choice([True, False], p=[self.epsilon, 1-self.epsilon]):
            return np.random.randint(0, 3)
        else:
            q = self.get_q(state)
            return np.argmax(q)

    def update_q(self, state, a, next_state, reward):
        target = reward + self.gamma * np.max(self.get_q(next_state))
        coeff = self.lr * (self.get_q(state)[a] - target)
        self.w[a] -= coeff * state
        self.b -= coeff

    def state_transform(self, state):
        s = np.zeros(self.env.state_space)
        for key, value in state.items():
            s[key] = value
        return s

    def train(self, episodes, max_iterations):
        re = np.zeros((episodes, 1))
        for i in range(episodes):
            state = self.state_transform(self.env.reset())
            for j in range(max_iterations):
                a = self.get_action(state)
                next_state, reward, done = self.env.step(a)
                next_state = self.state_transform(next_state)
                self.update_q(state, a, next_state, reward)
                state = next_state
                re[i, 0] += reward
                if done:
                    break
        return re


def file(w, b, itera, roll, w_name, re_name):
    f_w = str(w_name)
    f1 = open(f_w, "w")
    f1.write(str('%.60f' % b))
    f1.write('\n')
    for item in w:
        f1.write(str('%.60f' % item))
        f1.write('\n')
    f1.close()

    f_re = str(re_name)
    f2 = open(f_re, "w")
    for num in itera:
        f2.write(str(float(num)))
        f2.write('\n')
    f2.close()

    f_e = str('empiral')
    f3 = open(f_e, "w")
    for mean in roll:
        f3.write(str(mean))
        f3.write('\n')
    f3.close()


def chunks(arr, n):
    roll = []
    for i in range(len(arr)-25):
        x = sum(arr[i:i + n])
        roll.append(float(x)/n)
    return roll



if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    # run parser and get parameters values

    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=learning_rate)
    returns = agent.train(episodes, max_iterations)
    # print(returns)
    weight = np.array(np.reshape(agent.w, (1, -1), order='F'))[0]
    bia = agent.b
    # print(np.array(np.reshape(agent.w, (1, -1), order='F'))[0])
    roll_mean = chunks(returns, 25)
    file(weight, bia, returns, roll_mean, weight_out, returns_out)