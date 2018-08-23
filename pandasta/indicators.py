import pandas as pd
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, indicators=[], dtype=None,
                 copy=False, offset='1s'):
        super().__init__(data, index, columns, dtype, copy)
        # self.indicators = indicators

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
    balance: float


class TradingStrategy(object):

    def generate_order(self, record, funds, balance) -> Order:
        return Order(decision=Order.Decision.BUY, amount=0.001)
        # pass


class BacktestingTaDataFrame(TaDataFrame):
    def __init__(self, data, indicators, funds, balance=0.0, **kwargs):
        super().__init__(data, indicators=indicators, **kwargs)
        self.funds = funds
        self.balance = balance

    def apply_strategy(self, strategy: TradingStrategy) -> pd.DataFrame:
        order_contexts = self.apply(lambda record: self._apply_strategy_on_record(record, strategy), axis=1)
        decisions = order_contexts.apply(lambda oc: oc.order.decision)
        amounts = order_contexts.apply(lambda oc: oc.order.amount)
        statuses = order_contexts.apply(lambda oc: oc.order.status)
        funds = order_contexts.apply(lambda oc: oc.funds)
        balance = order_contexts.apply(lambda oc: oc.balance)

        result_df = pd.DataFrame({"decisions": decisions,
                                  "amounts": amounts,
                                  "statuses": statuses,
                                  "funds": funds,
                                  "balance": balance})
        return result_df

    def _apply_strategy_on_record(self, record, strategy: TradingStrategy) -> Optional[OrderContext]:
        order = strategy.generate_order(record, self.funds, self.balance)

        if order is None:
            return

        if order.decision == Order.Decision.BUY:
            closing_price = record['close']
            amount_to_buy = closing_price * order.amount

            if amount_to_buy <= self.funds:
                self.funds -= amount_to_buy
                order.status = Order.OrderStatus.FILLED
            else:
                pass
                order.status = Order.OrderStatus.REJECTED
        elif order.decision == Order.Decision.SELL:
            amount_to_sell = order.amount
            if amount_to_sell <= self.balance:
                self.balance -= amount_to_sell
                order.status = Order.OrderStatus.FILLED
            else:
                pass
                order.status = Order.OrderStatus.REJECTED

        return OrderContext(order, funds=self.funds, balance=self.balance)


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
        low_prices = data.get_low_prices().rolling(period).mean()
        high_prices = data.get_high_prices().rolling(period).mean()
        return (closing_prices - low_prices) / (high_prices - low_prices)


class HighLowPriceRatio(Indicator):

    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'
        low_prices = data.get_low_prices().rolling(period).min()
        high_prices = data.get_high_prices().rolling(period).max()
        return low_prices / high_prices


class Indicators(Enum):
    SMA = ('sma', SimpleMovingAverage.create)
    EMA = ('ema', ExponentialMovingAverage.create)
    HILO = ('hilo', HighLowPriceRatio.create)
    STOCH_K = ('stochk', StochasticOscillatorK.create)


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
                                indicators=[
                                    'sma_60', 'sma_1min', 'ema_50', 'stochk_14', 'stochk_365', 'hilo_7'])

    print(df.apply_strategy(TradingStrategy()))
    # print(df)


if __name__ == '__main__':
    main()
