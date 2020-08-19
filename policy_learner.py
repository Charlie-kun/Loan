import os       #create foldder, file path
import locale       #Using Currency string format.
import time
import logging      #Memorize learn information
import datetime
import numpy as np
import settings     #sold setting, logging setting
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

logger = logging.getLogger(__name__)



class PolicyLearner:
    def __init__(self, stock_code, chart_data, training_data=None, Min_trading_unit=1, max_trading_unit=2, Delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code     #stock code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)      #Enviroment object
        #agent object
        self.agent = Agent(self.environment, min_trading_unit = Min_trading_unit, max_trading_unit = max_trading_unit, delayed_reward_threshold = Delayed_reward_threshold)
        self.training_data = training_data    #learn data
        self.sample = None
        self.training_data_idx = -1
        #Policy neural network ; input size=learn data size + agent state size
        self.num_features = self.traning_data_shape[1] + self.agent.STATE_DIM
        self.policy_network=PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer=Visualizer()        #visual module