import random
import unittest

from pandasta.indicators import (
    BacktestingTaDataFrame,
    Order,
    TaDataFrame,
    TradingStrategy,
)

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
        data = [{"time": i, "close": i * rate_of_change} for i in range(1000)]

        df = TaDataFrame(data, indicators=["trend_14"])
        all_trends_match_expected_answer = (
            df["trend_14"].dropna() == rate_of_change
        ).all()

        self.assertTrue(all_trends_match_expected_answer)

    def test_backtesting_when_all_orders_ask_for_sub_minimum_amounts(self):
        data = [
            {
                "time": i,
                "low": max(i + random.randint(-100, -50), 1),
                "high": max(i + random.randint(50, 100), 1),
                "open": max(i + random.randint(-50, 50), 1),
                "close": max(i + random.randint(-50, 50), 1),
            }
            for i in range(1000)
        ]

        df = BacktestingTaDataFrame(
            data, funds=1000, min_amount=MIN_AMOUNT, indicators=[]
        )
        strategy = TestStrategy(min_amount=MIN_AMOUNT - 0.0001)
        res = df.apply_strategy(strategy)

        all_rejected = (res["statuses"] == "REJECTED").all()

        self.assertTrue(
            all_rejected,
            "Not all orders seem to have been rejected. Please, re-check the logic",
        )

    def test_that_funds_replenishment_does_not_change_overall_funds_by_default(self):
        class ReplenishmentStrategy(TradingStrategy):
            def generate_order(self, record, funds, balance) -> Order:
                return None

        num_of_records = 100
        initial_funds = 1000
        df = BacktestingTaDataFrame(
            data=[{"time": i, "close": 1} for i in range(num_of_records)],
            indicators=[],
            funds=initial_funds,
            min_amount=MIN_AMOUNT,
        )
        res = df.apply_strategy(ReplenishmentStrategy())

        self.assertTrue(res.iloc[-1]["funds"] == initial_funds)
        self.assertTrue(res.iloc[-1]["total_funds_over_time"] == initial_funds)

    def test_funds_replenishment(self):
        class ReplenishmentStrategy(TradingStrategy):
            def generate_order(self, record, funds, balance) -> Order:
                return None

            def replenish_funds(self, record) -> float:
                return 1.0

        num_of_records = 100
        df = BacktestingTaDataFrame(
            data=[{"time": i, "close": 1} for i in range(num_of_records)],
            indicators=[],
            funds=0.0,
            min_amount=MIN_AMOUNT,
        )
        res = df.apply_strategy(ReplenishmentStrategy())

        self.assertTrue(res.iloc[-1]["funds"] == num_of_records)
        self.assertTrue(res.iloc[-1]["total_funds_over_time"] == num_of_records)

    def test_partial_application(self):
        num_of_records = 10
        raw_data = [{"time": i, "close": 1} for i in range(num_of_records)]
        df = BacktestingTaDataFrame(
            data=raw_data, indicators=[], funds=0.0, min_amount=MIN_AMOUNT
        )
        pos = 6
        expected_index_key_value = df.reset_index().loc[pos, "time"]
        res = df.apply_strategy(TestStrategy(), start=expected_index_key_value)
        actual_index_key_value = res.reset_index().loc[0, "time"]
        self.assertTrue(actual_index_key_value == expected_index_key_value)
        self.assertTrue(res.shape[0] == num_of_records - pos)


if __name__ == "__main__":
    unittest.main()
