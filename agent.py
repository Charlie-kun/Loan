import numpy as np
import utils


class Agent:
    # Agent state
    STATE_DIM = 2  # Stork Value Rate, Portfolio Value Rate

    # Trading charge and taxs
    TRADING_CHARGE = 0.00015  # Transaction fees 0.015%
    # TRADING_CHARGE = 0.00011  # Transaction fees 0.011%
    # TRADING_CHARGE = 0  # Transaction fees none
    TRADING_TAX = 0.0025  # Transaction tax 0.25%
    # TRADING_TAX = 0  # Transaction tax none

    # Action
    ACTION_BUY = 0  # Buy
    ACTION_SELL = 1  # Sell
    ACTION_HOLD = 2  # Holding
    # Find rate action in AI
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  #The number of outputs to be considered in the artificial neural network

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05):
        # Environment
        # Environment of now stock value
        self.environment = environment

        # Min, Max trading unit, Limited delay bonus
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # Limited delay bonus
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent
        self.initial_balance = 0  # Seed money
        self.balance = 0  # Balance
        self.num_stocks = 0  # number of present Stock.
        # portfolio value : balance + num_stocks * {Stock Value}
        self.portfolio_value = 0 
        self.base_portfolio_value = 0  # studied PV value
        self.num_buy = 0  # Number of buy
        self.num_sell = 0  # Number of sell
        self.num_hold = 0  # Number of holding
        self.immediate_reward = 0  # immediate_reward
        self.profitloss = 0  # loss value.
        self.base_profitloss = 0  # delay bonus to loss value at last time.
        self.exploration_base = 0  # Decide exploration action base.

        # Agent class state
        self.ratio_hold = 0  # stock rate
        self.ratio_portfolio_value = 0  # portfolio value rate

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value
        )
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # No predict value so explore.
            epsilon = 1
        else:
            # same predict value. so explore.
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # Decide explore.
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # Check buy one stock.
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # enviroment get price
        curr_price = self.environment.get_price()

        # reset reward
        self.immediate_reward = 0

        # but
        if action == Agent.ACTION_BUY:
            # decide trading unit
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # case of not enough seed money is possible maximally buying stock at seed money.
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # Calculate total buy value at regard of fee.
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # renew balance.
                self.num_stocks += trading_unit  # renew number of stock.
                self.num_buy += 1  # increase number of buy

        # sell
        elif action == Agent.ACTION_SELL:
            # Decide of sell unit
            trading_unit = self.decide_trading_unit(confidence)
            # case of not enough stock is possible maximally selling stock
            trading_unit = min(trading_unit, self.num_stocks)
            # sell
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                    * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # renew number of stock
                self.balance += invest_amount  # renew balance
                self.num_sell += 1  # increase number of buy

        # Holdding
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # increase number of holdding

        # renew portfolio value
        self.portfolio_value = self.balance + curr_price \
            * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) \
                / self.initial_balance
        )
        
        # immediate reward - Yield
        self.immediate_reward = self.profitloss

        # Delayed bonus - base of benefit or loss
        delayed_reward = 0
        self.base_profitloss = (
            (self.portfolio_value - self.base_portfolio_value) \
                / self.base_portfolio_value
        )
        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            # benefit renew portfolio value
            # loss renew portfolio value
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
