from distutils.core import setup

setup(
    name="pandas-ta",
    version="0.1",
    install_requires=[
        "dataclasses==0.6",
        "pandas==0.20.3",
        "pandas-datareader==0.7.0",
        "pyfinance==1.1.1",
    ],
    packages=["pandasta"],
)
