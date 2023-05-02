# Reddit_Investment_Signals

## About this project

Welcome to the "Reddit_Investment_Signals" repository, where we use the popular subreddit r/wallstreetbets to predict stock values of the S&P500.

The S&P500 index is one of the most widely followed stock indices and includes the 500 largest companies in the United States by market capitalisation. Our approach is to use the collective intelligence of the Wall Street Bets community, which consists of private investors who share their thoughts, opinions and trades on the subreddit.

We use natural language processing (NLP) techniques to extract relevant information from the subreddit posts and comments, such as sentiments, keywords and stock market tickers. We then use machine learning algorithms to analyse this data and generate stock price forecasts for the S&P500 companies.

Our repository contains code for data collection, pre-processing, modelling and orchestrating the application. For convenience, we also provide pre-processed datasets and trained models. We welcome contributions from the community to improve the accuracy of our forecasts and extend our coverage to other indices, markets or other subreddits.

----------------------------------------------------------------------------------------

## Dataset download

1. Download the dataset from our Dropbox link: https://www.dropbox.com/scl/fo/6jr6i3lat3v9u1kjq5n1k/h?dl=0&rlkey=ofwk0kev3qtqui0edexvv912b
(size of the zipped file ~1.4 GB, decompressed ~7.6 GB)
2. Decompress the contents of the .zip file into the folder `datasets/`
3. Move the folder into the first level of the repository folder (`/Reddit_Investment_Signals/datasets`)


----------------------------------------------------------------------------------------


## How the repo works:

First time setup:
1. Step: Use setup.py to install required packages
2. Step: Adapt the parameters in `main.py` if desired
3. Step: Execute `main.py`

`main.py` covers all necessary functionality, so it is the only file you need to work with.

----------------------------------------------------------------------------------------

## How to modify the model setup
If you are experienced with Machine Learning, you can try out different algorihms. 
We have prepared alternatives in `ml_algorihms.py`. 
To execute them, feel free  to adapt `organiser.py` to your needs. 

Feel free to use your own data, tune parameters, explore machine learning algorithms and beat the market! #toTheMoon


----------------------------------------------------------------------------------------

## Disclaimer
Our predictions are based solely on historic data analysis and do not constitute financial advice. We recommend that you perform your own due diligence and investigate various information sources before making an investment decision.

----------------------------------------------------------------------------------------

## Citation 

The corresponding paper for this project is currently under submission for double blind review at the IEEE DSAA 2023 conference.
Once a publication is possible, we will add details on how to cite this project.