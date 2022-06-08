import numpy as np
import gym
import torch as T
from torch.optim import lr_scheduler as ls
import lin_qn, lstm_qn, conv_qn, environment, tools
#from ultis import plot_learning_curve

if __name__ == "__main__":

    env = gym.make("ALE/AirRaid-v5", render_mode = "human")
    
    bot = lstm_qn.DuelQNet(in_features = [250, 160, 3], out_features = 6, hidden_lstm_size = 16, n_hidden_lstm_layers = 1, 
                            hidden_lin_size = 16, replay_size = 3, bidirectional = True)
    bot.lr_decay_method(ls.ExponentialLR(bot.critic_net.optimizer, 0.9999))
    env.reset()

    j = 0
    for i in range(50000):
        obs = env.reset()
        done = False

        #debugging space
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        bot.reward = 0
        actions = []
        action = 0
        #bot.alpha = bot.critic_net1.optimizer.param_groups[0]["lr"]

        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        while not done:
            action = bot.decision()
            actions.append(action)
            new_obs, reward, done, _ =  env.step(action)
            bot.transition(obs, action, new_obs, reward, done)
            obs = new_obs
            bot.learn()
            bot.execute_lr_decay()

            #debugging space
            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            
            if(j % 10 == 0):
                print("Episode {}, Step {}: Action: {}, Total reward: {}., Epsilon: {:.4f}, Loss: {}".format(i, j, actions, bot.total_reward, bot.epsilon, bot.loss), ", Learning rate: ", bot.alpha)
                print(f"Average Loss: {bot.average_loss}, ")
                actions.clear()
            j += 1
            #env.render()



            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''       
    
    env.close()