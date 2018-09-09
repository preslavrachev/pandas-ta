import random
import unittest

from pandasta.indicators import BacktestingTaDataFrame, Order, TaDataFrame
from pandasta.indicators import TradingStrategy

MIN_AMOUNT = 0.001


class TestStrategy(TradingStrategy):

    def __init__(self, min_amount: float = MIN_AMOUNT) -> None:
        super().__init__()
        self.min_amount = min_amount

    def generate_order(self, record, funds, balance) -> Order:
        return Order(Order.Decision.BUY, self.min_amount)


class TestPandasTA(unittest.TestCase):

    def test_trend_indicator(self):
        rate_of_change = 1.02
        data = [{
            "time": i,
            "close": i * rate_of_change
        } for i in range(1000)]

        df = TaDataFrame(data, indicators=['trend_14'])
        all_trends_match_expected_answer = (df['trend_14'].dropna() == rate_of_change).all()

        self.assertTrue(all_trends_match_expected_answer)

    def test_backtesting_when_all_orders_ask_for_sub_minimum_amounts(self):
        data = [{
            "time": i,
            "low": max(i + random.randint(-100, -50), 1),
            "high": max(i + random.randint(50, 100), 1),
            "open": max(i + random.randint(-50, 50), 1),
            "close": max(i + random.randint(-50, 50), 1)
        } for i in range(1000)]

        df = BacktestingTaDataFrame(data,
                                    funds=1000,
                                    min_amount=MIN_AMOUNT,
                                    indicators=[])
        strategy = TestStrategy(min_amount=MIN_AMOUNT - 0.0001)
        res = df.apply_strategy(strategy)

        all_rejected = (res['statuses'] == 'REJECTED').all()

        self.assertTrue(all_rejected, 'Not all orders seem to have been rejected. Please, re-check the logic')


if __name__ == '__main__':
    unittest.main()
