import matplotlib.pyplot as plt
import numpy as np

def plot_running_avg(total_rewards):
    #take np array and gets the running average
    N = len(total_rewards)
    running_average = np.empty(N)
    for i in range(N):
        running_average[i] = total_rewards[max(0, i-100): i+1].mean()
    plt.plot(running_average)
    plt.title("Aveage score")
    plt.show()