import pandas as pd
import random
from enum import Enum


class TaDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, indicators=[], dtype=None,
                 copy=False, offset='1s'):
        super().__init__(data, index, columns, dtype, copy)
        #self.indicators = indicators

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

            if (shortname == label_shortname):
                # print('will create ', indicator)
                return creator_func(series, label_period)


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
        return pd.ewma(data.get_closing_prices(), span=period, min_periods=period-1)


class StochasticOscillatorK(Indicator):
    @staticmethod
    def create(data: TaDataFrame, period):
        assert type(
            period) == int, 'Only an integer number of periods is supported at the moment!'
        closing_prices = data.get_closing_prices().rolling(period).mean()
        low_prices = data.get_low_prices().rolling(period).mean()
        high_prices = data.get_high_prices().rolling(period).mean()
        return (closing_prices - low_prices) / (high_prices - low_prices)


class Indicators(Enum):
    SMA = ('sma', SimpleMovingAverage.create)
    EMA = ('ema', ExponentialMovingAverage.create)
    STOCH_K = ('stochk', StochasticOscillatorK.create)


def main():

    data = [{
        "time": i,
        "low": max(i + random.randint(-100, -50), 1),
        "high": max(i + random.randint(50, 100), 1),
        "open": max(i + random.randint(-50, 50), 1),
        "close": max(i + random.randint(-50, 50), 1)
    } for i in range(1000)]

    df = TaDataFrame(data, indicators=[
                     'sma_60', 'sma_1min', 'ema_50', 'stochk_14', 'stochk_365'])
    print(df)


if __name__ == '__main__':
    main()
