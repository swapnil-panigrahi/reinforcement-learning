import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def epsilonGreedy(Q, epsilon, actions):
    if random.random() < epsilon:
        return random.randint(0, actions - 1)
    else:
        return np.argmax(Q)

def multiArmedBandits(k, steps, confidence):
    phiStar = np.zeros(k)
    for _ in range(k):
        phiStar[_] = np.random.normal(0, 1)
        
    Q = np.zeros(k)
    N = np.zeros(k)
    
    rewards = np.zeros(steps)
    
    for t in range(steps):
        A = np.argmax(Q + confidence * np.sqrt(np.log(t+1)/(N + 1e-5)))
            
        R = np.random.normal(phiStar[A], 1)
        N[A] += 1
        Q[A] += (R - Q[A]) / N[A]
        rewards[t] = R
    
    return rewards

def averageRuns(k, steps, runs, confidence):
    averageRewards = np.zeros(steps)
    for i in range(runs):
        x= multiArmedBandits(k, steps, confidence)
        averageRewards += x
    
    return averageRewards/runs

confidence = [2]
steps = 1000
runs = 2000
k = 10

avgRewards = []
optimalAction = []

for c in confidence:
    x = averageRuns(k, steps, runs, c)
    avgRewards.append(x) 

plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
for i, c in enumerate(confidence):
    plt.plot(avgRewards[i], label=f"c={c}")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()

plt.show()