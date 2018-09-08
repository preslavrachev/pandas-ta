# Pandas TA

## Introduction
Pandas TA is a lightweight technical analysis wrapper on top of [Pandas](https://pandas.pydata.org/). Pandas TA is a convention-over-configuration library, which allows for easily attaching TA indicators to a Pandas DataFrame, by following a simple naming convention:

```python
data = fetch_raw_data() # from an online resource, file, etc
df = pdt.TaDataFrame(data, indicators=['sma_14', 'sma_30', 'stochk_14', 'atr_30'])
```

As a result, alongside the usual columns in the data frame, four new ones will appear, each one containing its respective indicator values. The naming convention of the indicators follows the standard naming patterns among technical analysts, as well as present in much of the available TA software. This should make it easier for non-programmers to start with.
