import numpy as np
import gym
import torch as T
from torch.optim import lr_scheduler as ls
import lstm_qn, tools
#from ultis import plot_learning_curve

if __name__ == "__main__":

    env = gym.make("ALE/Breakout-v5", render_mode = "human")
    
    bot = lstm_qn.DuelQNet(in_features = [210, 160, 3], out_features = 4, hidden_lstm_size = 64, n_hidden_lstm_layers = 10, 
                            hidden_lin_size = 3096)
    bot.lr_decay_method(ls.ExponentialLR(bot.critic_net.optimizer, 0.999))
    env.reset()

    j = 0
    for i in range(50000):
        obs = env.reset()
        done = False

        #debugging space
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        bot.reward = 0
        actions = []

        bot.execute_lr_decay()
        #bot.alpha = bot.critic_net1.optimizer.param_groups[0]["lr"]

        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        while not done:
            action = bot.decision(obs)
            new_obs, reward, done, _ =  env.step(action)
            bot.transition(obs, action, new_obs, reward, done)
            bot.learn()
            obs = new_obs

            #debugging space
            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            actions.append(action)
            if(j % 10 == 0):
                print("Episode {}, Step {}: Action: {}, Total reward: {}., Epsilon: {:.4f}, Loss: {}".format(i, j, actions, bot.total_reward, bot.epsilon, bot.loss), ", Learning rate: ", bot.alpha)
                actions.clear()
            j += 1
            #env.render()



            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''       
    
    env.close()
