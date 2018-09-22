from distutils.core import setup
setup(name='pandas-ta',
      version='0.1',
      install_requires=[
          'dataclasses',
          'pandas',
          'pyfinance'
      ],
      packages=['pandasta'])
