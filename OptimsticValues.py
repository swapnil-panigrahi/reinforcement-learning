import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def epsilonGreedy(Q, epsilon, actions):
    if random.random() < epsilon:
        return random.randint(0, actions - 1)
    else:
        return np.argmax(Q)

def multiArmedBandits(k, steps, epsilon, alpha):
    phiStar = np.zeros(k)
    for _ in range(k):
        phiStar[_] = np.random.normal(0, 1)
        
    Q = np.full(k, 5.0)

    rewards = np.zeros(steps)
    optimalAction = np.zeros(steps)
    
    for t in range(steps):
        if callable(epsilon):
            A = epsilonGreedy(Q, decay(t+1), k)
        else:
            A = epsilonGreedy(Q, epsilon, k)
        
        if A==np.argmax(phiStar):
            optimalAction[t]+=1
            
        R = np.random.normal(phiStar[A], 1)

        Q[A] += (R - Q[A]) * alpha
        
        rewards[t] = R

    return rewards, optimalAction

def averageRuns(k, steps, epsilon, runs, alpha):
    averageRewards = np.zeros(steps)
    optimalActions = np.zeros(steps)
    for i in range(runs):
        x, y = multiArmedBandits(k, steps, epsilon, alpha)
        averageRewards += x
        optimalActions += y
    
    return averageRewards/runs, (optimalActions/runs) * 100

def decay(t):
    return 1/t

epsilon = [0]
steps = 1000
runs = 2000
k = 10
alpha = 0.1

avgRewards = []
optimalAction = []

for eps in epsilon:
    x, y = averageRuns(k, steps, eps, runs, alpha)
    avgRewards.append(x)
    optimalAction.append(y)

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
for i, eps in enumerate(epsilon):
    plt.plot(avgRewards[i], label=f"$\\epsilon$={eps}")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()

plt.subplot(2, 1, 2)
for i, eps in enumerate(epsilon):
    plt.plot(optimalAction[i], label=f"$\\epsilon$={eps}")
plt.xlabel("Steps")
plt.ylabel("Optimal Action")
plt.legend()

plt.show()