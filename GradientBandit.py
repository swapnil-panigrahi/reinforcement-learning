import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def epsilonGreedy(Q, epsilon, actions):
    if random.random() < epsilon:
        return random.randint(0, actions - 1)
    else:
        return np.argmax(Q)

def multiArmedBandits(k, steps, alpha, base):
    phiStar = np.zeros(k)
    for _ in range(k):
        phiStar[_] = np.random.normal(4, 1)
        
    H = np.zeros(k)
    optimalAction = np.zeros(steps)
    
    if base:
        RMean = 0
    else:
        RMean = None
    
    for t in range(steps):
        pi = np.exp(H)/np.sum(np.exp(H))
        A = np.random.choice(np.arange(k), p=pi)
        
        if A==np.argmax(phiStar):
            optimalAction[t]+=1
            
        R = np.random.normal(phiStar[A], 1)
        
        if base:
            RMean += (R - RMean)/(t + 1)
        
        baseline = RMean if base else 0
        for i in range(k):
            if A!=i:
                H[i] -= alpha * (R - baseline) * pi[i]
            else:
                H[i] += alpha * (R - baseline) * (1 - pi[i])
    
    return optimalAction

def averageRuns(k, steps, runs, alpha, base):
    optimalActions = np.zeros(steps)
    for i in range(runs):
        y = multiArmedBandits(k, steps, alpha, base)
        optimalActions += y
    
    return (optimalActions/runs) * 100

alpha = [0.1, 0.4]
steps = 1000
runs = 2000
k = 10

avgRewards = []
optimalAction = []

for a in alpha:
    y = averageRuns(k, steps, runs, a, True)
    optimalAction.append(y)
    
    y = averageRuns(k, steps, runs, a, False)
    optimalAction.append(y)

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(optimalAction[0], label=f"{alpha[0]} with baseline")
plt.plot(optimalAction[1], label=f"{alpha[0]} without baseline")

plt.plot(optimalAction[2], label=f"{alpha[1]} with baseline")
plt.plot(optimalAction[3], label=f"{alpha[1]} without baseline")

plt.xlabel("Steps")
plt.ylabel("Optimal Action %")

plt.legend()
plt.show()