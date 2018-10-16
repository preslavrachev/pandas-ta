import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from enum import Enum
from abc import ABCMeta, abstractmethod
from pyfinance.ols import PandasRollingOLS
from typing import Optional


class TaDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, indicators=[], dtype=None,
                 copy=False, offset='1s'):
        super().__init__(data, index, columns, dtype, copy)
        # self.indicators = indicators

        self['timestamp'] = self['time']
        self['time'] = pd.to_datetime(self['time'], unit='s')
        self.set_index('time', inplace=True)

        for indicator in indicators:
            self[indicator] = TaDataFrame.parse(indicator, self)

    def get_opening_prices(self) -> pd.Series:
        return self['open']

    def get_closing_prices(self) -> pd.Series:
        return self['close']

    def get_low_prices(self) -> pd.Series:
        return self['low']

    def get_high_prices(self) -> pd.Series:
        return self['high']

    @staticmethod
    def parse(label, series):
        for indicator in Indicators:
            shortname, creator_func = indicator.value

            label_parts = label.split('_')
            label_shortname = label_parts[0]
            label_period = int(label_parts[1]) if str(
                label_parts[1]).isdigit() else str(label_parts[1])

            if shortname == label_shortname:
                # print('will create ', indicator)
                return creator_func(series, label_period)


@dataclass
class Order:
    class Decision(Enum):
        BUY = 1
        SELL = 2

    class OrderStatus(Enum):
        OPEN = 1
        FILLED = 2
        CANCELLED = 3
        REJECTED = 4

    decision: Decision.BUY
    amount: float
    status: OrderStatus = OrderStatus.OPEN


@dataclass
class OrderContext:
    """
    A wrapper around order, containing additional contextual information, such as the amount of funds,
    as well as the current balance
    """
    order: Order
    funds: float
    total_funds_over_time: float
    balance: float
    worth: float


class TradingStrategy(object, metaclass=ABCMeta):

    def replenish_funds(self, record) -> float:
        return 0

    @abstractmethod
    def generate_order(self, record, funds, balance) -> Order:
        pass


class RandomDemoTradingStrategy(TradingStrategy):
    def generate_order(self, record, funds, balance) -> Order:
        amount = 0.001 * random.randrange(1, 10)
        decision = random.choice([Order.Decision.BUY, Order.Decision.SELL])
        return Order(decision=decision, amount=amount)


class BacktestingTaDataFrame(TaDataFrame):
    def __init__(self, data, indicators, funds, min_amount, balance=0.0, **kwargs):
        super().__init__(data, indicators=indicators, **kwargs)
        # TODO: Add min_amount, fees, and percentage of orders not going through
        self.initial_funds = funds
        self.initial_balance = balance
        self.min_amount = min_amount

    def apply_strategy(self, strategy: TradingStrategy) -> pd.DataFrame:
        # TODO: Add support for filtering out rejected and canceled orders
        funds_and_balance = {'residual_funds': self.initial_funds,
                             'total_funds_over_time': self.initial_funds,
                             'residual_balance': self.initial_balance}

        order_contexts = self.apply(lambda record: BacktestingTaDataFrame._apply_strategy_on_record(record,
                                                                                                    strategy,
                                                                                                    funds_and_balance,
                                                                                                    self.min_amount),
                                    axis=1)
        decisions = order_contexts.apply(lambda oc: oc.order.decision.name if oc.order else np.NaN)
        amounts = order_contexts.apply(lambda oc: oc.order.amount if oc.order else 0.0)
        statuses = order_contexts.apply(lambda oc: oc.order.status.name if oc.order else np.NaN)
        funds = order_contexts.apply(lambda oc: oc.funds)
        total_funds_over_time = order_contexts.apply(lambda oc: oc.total_funds_over_time)
        balance = order_contexts.apply(lambda oc: oc.balance)
        worth = order_contexts.apply(lambda oc: oc.worth)

        # Used for comparing the strategy against simply buying and holding
        initial_buy_hold_balance = total_funds_over_time.iloc[-1] / self.iloc[0]['close']
        buy_hold = self['close'].apply(lambda price: price * initial_buy_hold_balance)

        result_df = pd.DataFrame({'decisions': decisions,
                                  'amounts': amounts,
                                  'statuses': statuses,
                                  'funds': funds,
                                  'balance': balance,
                                  'worth': worth,
                                  'buy_hold': buy_hold,
                                  'total_funds_over_time': total_funds_over_time})
        return result_df

    @staticmethod
    def _apply_strategy_on_record(record,
                                  strategy: TradingStrategy,
                                  funds_and_balance: dict,
                                  min_amount: float) -> Optional[OrderContext]:
        residual_funds = funds_and_balance['residual_funds']
        residual_balance = funds_and_balance['residual_balance']
        total_funds_over_time = funds_and_balance['total_funds_over_time']
        closing_price = record['close']

        # check, if funds are to be replenished
        amount_to_replenish_with = strategy.replenish_funds(record)
        residual_funds += amount_to_replenish_with
        total_funds_over_time += amount_to_replenish_with

        order = strategy.generate_order(record, residual_funds, residual_balance)

        if order:
            if order.amount < min_amount:
                order.status = Order.OrderStatus.REJECTED
            else:
                amount_in_funds_units = closing_price * order.amount

                if order.decision == Order.Decision.BUY:
                    if amount_in_funds_units <= residual_funds:
                        residual_funds -= amount_in_funds_units
                        residual_balance += order.amount
                        order.status = Order.OrderStatus.FILLED
                    else:
                        order.status = Order.OrderStatus.REJECTED
                elif order.decision == Order.Decision.SELL:
                    if order.amount <= residual_balance:
                        residual_balance -= order.amount
                        residual_funds += amount_in_funds_units
                        order.status = Order.OrderStatus.FILLED
                    else:
                        order.status = Order.OrderStatus.REJECTED

        # TODO: Add support for delaying orders by putting them on a queue
        funds_and_balance['residual_funds'] = residual_funds
        funds_and_balance['total_funds_over_time'] = total_funds_over_time
        funds_and_balance['residual_balance'] = residual_balance
        return OrderContext(order,
                            funds=residual_funds,
                            total_funds_over_time=total_funds_over_time,
                            balance=residual_balance,
                            worth=residual_funds + (residual_balance * closing_price))


class Indicator(object):
    pass


class SimpleMovingAverage(Indicator):

    @staticmethod
    def create(data: TaDataFrame, period):
        return data.get_closing_prices().rolling(period).mean()


class ExponentialMovingAverage(Indicator):

    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'
        return pd.ewma(data.get_closing_prices(), span=period, min_periods=period - 1)


class StochasticOscillatorK(Indicator):
    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'
        closing_prices = data.get_closing_prices().rolling(period).mean()
        min_price = data.get_low_prices().rolling(period).min()
        max_price = data.get_high_prices().rolling(period).max()
        return (closing_prices - min_price) / (max_price - min_price)


class HighLowPriceRatio(Indicator):

    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'
        low_prices = data.get_low_prices().rolling(period).min()
        high_prices = data.get_high_prices().rolling(period).max()
        return low_prices / high_prices


class AverageTrueRange(Indicator):
    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'

        ext_data = data.copy()
        ext_data['prev_close'] = data['close'].shift(1)

        ext_data['h_minus_l'] = ext_data['high'] - ext_data['low']
        ext_data['h_minus_pc'] = (ext_data['high'] - ext_data['prev_close']).abs()
        ext_data['l_minus_pc'] = (ext_data['low'] - ext_data['prev_close']).abs()

        tr_s = ext_data[['h_minus_l', 'h_minus_pc', 'l_minus_pc']].max(axis=1)
        return pd.ewma(tr_s, span=period, min_periods=period)


class LinearTrend(Indicator):
    @staticmethod
    def create(data: TaDataFrame, period):
        # TODO: Use a more optimal way to generate this series
        row_ids = pd.Series(np.arange(0, len(data), step=1), index=data.index)
        a = PandasRollingOLS(y=data['close'], x=row_ids, window=period)
        return a.beta.astype('float32')


class Indicators(Enum):
    SMA = ('sma', SimpleMovingAverage.create)
    EMA = ('ema', ExponentialMovingAverage.create)
    HILO = ('hilo', HighLowPriceRatio.create)
    STOCH_K = ('stochk', StochasticOscillatorK.create)
    ATR = ('atr', AverageTrueRange.create)
    TREND = ('trend', LinearTrend.create)


def main():
    data = [{
        "time": i,
        "low": max(i + random.randint(-100, -50), 1),
        "high": max(i + random.randint(50, 100), 1),
        "open": max(i + random.randint(-50, 50), 1),
        "close": max(i + random.randint(-50, 50), 1)
    } for i in range(1000)]

    df = BacktestingTaDataFrame(data,
                                funds=1000,
                                min_amount=0.001,
                                indicators=[
                                    'sma_60', 'sma_1min', 'ema_50', 'stochk_14', 'stochk_365', 'hilo_7', 'atr_14',
                                    'trend_14'])

    # print(df.apply_strategy(RandomDemoTradingStrategy()))
    print(df)


if __name__ == '__main__':
    main()
