import os       # create folder, file path
import locale       # Using Currency string format.
import time
import logging      # Memorize learn information
import datetime
import numpy as np
import settings     # sold setting, logging setting
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code     # stock code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)      # Environment object
        # agent object
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data    # learn data
        self.sample = None
        self.training_data_idx = -1
        # Policy neural network ; input size=learn data size + agent state size
        self.num_features = self.traning_data_shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()        # visual module


    def reset(self):
        self.sample = None
        self.training_data_idx = -1


    def fit(
            self, num_epoches=1000, max_memory=60, balance=1000000,
            discount_factor=0, start_epsilon=.5, learning=True):
            logger.info("LR: {lr}, DF:{discount_factor}," 
                        "TU=[{min_trading_unit}, {max_trading_unit}], " 
                        "DRT:{delayed_reward_threshold}".format(
                lr=self.polict_network.lr,
                discount_factor=discount_factor,
                min_trading_unit=self.agent.min_trading_unit,
                max_trading_unit=self.agent.max_trading_unit,
                delayed_reward_threshold=self.agent.delayed_reward_threshold
            ))
    # Visualize ready
    # Already visualize because Chart data is not change.
    self.visualizer.prepare(self.environment.chart_data)

    # Ready for save folder of Result for visualize
    epoch_summary_dir=os.path.join(
        settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
            self.stock_code, settings.timestr))
    if not os.path.isdir(epoch_summary_dir):
        os.makedirs(epoch_summary_dir)

    # Setting for agent Seed money
    self.agent.set_balance(balance)

    # Information initialize for learn
    max_portfolio_value = 0
    epoch_win_cnt = 0