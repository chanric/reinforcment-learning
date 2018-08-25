import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

def build_state(features):
    """
    convert [1,2,3] to 123
    """
    return int("".join([str(feature) for feature in features]))

def to_bin(value, bins):
    return np.digitize([value], bins)[0]

def to_bins(values, bins):
    return np.digitize(values, bins)


class FeatureTransformer:
    def __init__(self):
        self._cart_position_bins = np.linspace(-3,3,9)
        #velocity
        self._cart_velocity_bins = np.linspace(-2, 2, 9)
        self._pole_angle_bins =  np.linspace(-0.5,0.5, 9)
        self._pole_velocity_bins = np.linspace(-3.5, 3.5, 9)
        
        
    def transform(self, observation):
        #transform the observation to state
        cart_p, cart_v, pole_a, pole_v = observation
        return build_state([
                to_bin(cart_p, self._cart_position_bins),
                to_bin(cart_v, self._cart_velocity_bins),
                to_bin(pole_a, self._pole_angle_bins),
                to_bin(pole_v, self._pole_velocity_bins)]
                )
        

class Model:
    def __init__(self, env, feature_transformer):
        self._env = env
        self._feat_transformer = feature_transformer
        
        #initizlize Q
        num_state = 10**env.observation_space.shape[0]
        num_action = env.action_space.n
        self._Q =  np.random.uniform(low=-1, high=1,
                                     size=(num_state, num_action))
        
    def predict(self, o):
        s = self._feat_transformer.transform(o)
        return self._Q[s]
    
    def update(self, o, a, G):
        s = self._feat_transformer.transform(o)
        self._Q[s,a] += 1e-2*(G - self._Q[s,a])
        
    def sample_action(self, o, eps):
        if np.random.random() > eps:
            return np.argmax(self.predict(o))
        else:
            return self._env.action_space.sample()
    
def play_one(model, eps, gamma):
    # Play games once based on model and return total reward
    # Updates G for model during plat
    #add env to arg?
    observation = env.reset()
    done = False
    total_reward = 0
    episode_num = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_o = observation
        observation, reward, done, _ = env.step(action)
        
        #punish for ending early
        if done and episode_num < 199:
            reward = -300
        
        total_reward += reward
        
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_o, action, G)
        episode_num+=1
    return total_reward

def plot_running_avg(total_rewards):
    #take np array and gets the running average
    N = len(total_rewards)
    running_average = np.empty(N)
    for i in range(N):
        running_average[i] = total_rewards[max(0, i-100): i+1].mean()
    plt.plot(running_average)
    plt.title("Aveage score")
    plt.show()
        
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    f_transformer = FeatureTransformer()
    model = Model(env, f_transformer)
    gamma = 0.9
    
    N = 10000
    total_rewards = np.empty(N)
    for i in range(N):
        curr_eps = 1.0/np.sqrt(i+1)
        total_reward = play_one(model, curr_eps, gamma);
        total_rewards[i] = total_reward
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            #print("episode:", i, "total reward:", total_reward, "eps:", curr_eps)
            print("episode number: %s, last_100_reward_avg: %s, eps: %s" %( i, last_100_r, curr_eps))
    print("last 100 r is %s" %(total_rewards[-100:].mean()))
    print("total steps:", total_rewards.sum())
    
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(total_rewards)
    