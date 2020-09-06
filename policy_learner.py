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
                lr=self.policy_network.lr,
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

    # Learn repeat
    for epoch in range(num_epoches):
        # initialize epoches
        loss = 0.
        itr_cnt = 0
        win_cnt = 0
        exploration_cnt = 0
        batch_size = 0
        pos_learning_cnt = 0
        neg_learning_cnt = 0

        # initialize memory
        memory_sample = []
        memory_action = []
        memory_reward = []
        memory_prob = []
        memory_pv = []
        memory_num_stocks = []
        memory_exp_idx = []
        memory_learning_idx = []

        # Initialize environment, agent, policy neural network
        self.enviroment.reset()
        self.agent.reset()
        self.policy_network.reset()
        self.reset()

        # Visualizer reset
        self.visualizer.clear([0, len(self.chart_data)])

        # continued learn so decrease of explore rate.
        if learning:
            epsilon = start_epsilon*(1.-float(epoch) / (num_epoches-1))
        else:
            epsilon = 0

        while True:
            # Create sample
            next_sample = self._bulid_sample()
            if next_sample is None:
                break

            # Decision by Policy neural network or explore
            action, confidence, exploration =self.agent.decide_action(
                self.policy_network, self.sample, epsilon)

            # Immediately gets bonus and delay bonus after Decision action.
            immediate_reward, delay_reward=self.agent.act(action, confidence)

            # Action and result save
            memory_sample.append(next_sample)
            memory_action.append(action)
            memory_reward.append(immediate_reward)
            memory_pv.append(self.agent.portfolio_value)
            memory_num_stocks.append(self.agent.num_stocks)
            memory=[(
                memory_sample[i],
                memory_action[i],
                memory_reward[i])
                for i in list(range(len(memory_action)))[-max_memory:]
              ]
              if exploration:
                  memory_exp_idx.append(itr_cnt)
                  memory_prob.append([np.nan]*Agent.NUM_ACTIONS)
              else:
                  memory_prob.append(self.policy_network.prob)

              # Refresh information for repeat.
              batch_size+= 1
              itr_cnt+=1
              exploration_cnt+=1 if exploration else 0
              win_cnt +=1 if delayed_reward >0 else 0

              # Learn mode and When get a delay bonus has exist, policy neural network reset.
              if delay_reward == 0 and batch_size >= max_memory:
                  delay_reward = immediate_reward
              if learning and delayed_reward !=0:
                  # batch learn data size
                  batch_size = min(batch_size, max_memory)
                  # Create batch learn data.
                  x, y=self._get_batch(
                      memory, batch_size, discount_factor, delayed_reward)
                  if len(x) > 0:
                      if delay_reward > 0:
                          pos_learning_cnt += 1
                      else:
                          neg_learning_cnt += 1
                      # Reset Policy neural network.
                      loss += self.policy_network.train_on_batch(x,y)
                      memory_learning_idx.append([itr_cnt, delay_reward])
                  batch_size = 0

        # Epoch information visualize.
        num_epoches_digit=len(str(num_epoches))
        epoch_str=str(epoch+1).rjust(num_epoches_digit,'0')    #check string length

        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
            action_list=Agent.ACTIONS, actions=memory_action,
            num_stocks=memory_num_stocks, outvals=memory_prob,
            exps=memory_exp_idx, learning=memory_learning_idx,
            initial_balance=self.agent.initial_balance, pvs=memory_pv
        )
        self.visualizer.save(os.path.join(
            epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                settings.timestr, epoch_str)))

        # epoch information log write.
        if pos_learning_cnt + neg_learning_cnt > 0:
            loss /=pos_learning_cnt + neg_learning_cnt
        logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                    "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                    "#Stocks:%d\tPV:%s\t"
                    "POS:%s\tNEG:%s\tLoss:%10.6f"% (
                        epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                        self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                        self.agent.num_stocks,
                        locale.currency(self.agent.portfolio_value, grouping=True),
                        pos_learning_cnt, neg_learning_cnt, loss))

        # learn information reset.
        max_portfolio_value=max(
            max_portfolio_value, self.agent.profitloss_value)
        if self.agent.portfolio_value > self.agent.initial_balance:
            epoch_win_cnt += 1

        # learn information log write
        logger.info("Max PV : %s, \t # Win : %d" %(
            locale.currenct(max_portfolio_value, grouping=True), epoch_win_cnt))

    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x=np.zeros((batch_size, 1, self.num_features))
        y=np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)

        for i, (sample, action, reward) in enumerate(
            reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1,1,self.num_features))
            y[i,action]=(delayed_reward+1)/2
            if discount_factor>0:
                y[i,action] *= discount_factor ** i
        return x, y

    def _build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx+1:
            self.training_data_idx += 1
            self.sample =self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def trade(slef, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)