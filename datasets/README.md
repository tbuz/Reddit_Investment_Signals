# Reddit Investment Signals - Dataset Setup

## Dataset download

1. Download the dataset from our Dropbox link: https://www.dropbox.com/scl/fo/6jr6i3lat3v9u1kjq5n1k/h?dl=0&rlkey=ofwk0kev3qtqui0edexvv912b
(size of the zipped file ~1.4 GB, decompressed ~7.6 GB)
2. Decompress the contents of the .zip file into the folder `datasets/`
3. Move the folder into the first level of the repository folder (`/Reddit_Investment_Signals/datasets`)

----------------------------------------------------------------------------------------

## Dataset structure

* __220924_SP500_wiki.html__: List of S&P 500 companies extracted from Wikipedia on September 24, 2022
* __sp500_stock_info_2022-07-03.json__: Metadata on all S&P 500 stocks extracted from Yahoo! Finance
* __sp500_stock_prices_2022-07-03.json__: Stock price history of all S&P 500 stocks from 2018-01-01 to 2022-07-03, extracted from Yahoo! Finance 
* __sp500_stock_recommendations_2022-07-03.json__: Investment bank recommendations for all S&P 500 stocks between 2018-01-01 and 2022-07-03 extracted from Yahoo! Finance
* __stock_dfs_2022-07-03.json__: All features extracted for the S&P 500 stocks from all posts shared on WSB between January 1, 2018, and July 3, 2022, extended with stock market data.
* __stock_dfs_filtered_2022-07-03.json__: All features extracted for the S&P 500 stocks from filtered posts (excluding deleted posts and post categories identified as "reactive") shared on WSB between January 1, 2018, and July 3, 2022, extended with stock market data.
* __submissions_WSB_20220703.json__: All posts shared on WSB between January 1, 2018, and July 3, 2022.
* __submissions_df_2022_03_07.pkl__: Pickle file of a large pandas DataFrame containing all submissions combined with their extracted signals, stock market data for the most mentioned ticker per submission, and additional features created during dataset preparation. This dataset is the main input for our ML models.