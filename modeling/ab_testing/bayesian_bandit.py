import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt


class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0

    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1

def plot(bandits: list[Bandit], trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITY]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUMBER_TRIALS):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])
        # plot the posteriors
        if i in sample_points:
            plot(bandits, i)
        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        # update the rewards
        rewards[i] = x
        # update the distribution for the bandits whose arm we just pulled
        bandits[j].update(x)
    # print total reward
    print("total reward earned:", rewards.sum())
    
    
