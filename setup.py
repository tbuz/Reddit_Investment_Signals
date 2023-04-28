# First time installation:
import pip
import importlib
import pandas as pd

class first_installer():
    def __int__(self):
        self

    def install_first_time(self, submission_df, stock_price, sp500_wiki):
        # Define the packages to be installed and imported
        packages = ['spacytextblob', 'emoji', 'yfinance', 'nltk', 'pandas', 'numpy', 'collections', 'spacy', 'scikit-learn', 'xgboost'] #TODO: Versions

        # Install packages if they are not already installed
        for package in packages:
            try:
                importlib.import_module(package)
                print('exisiting: ', package)
            except ImportError:
                print('installing: ', package)
                pip.main(['install', package])
        # Import packages
        import numpy
        import pandas

        # Download NLTK Wordnet:
        import nltk
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        # Import packages

        # --- Data manipulation
        # import json
        import pandas as pd
        import numpy as np
        from collections import Counter
        #TODO: uncomment next line yfinance
        #import yfinance as yf

        # --- Time formatting
        import datetime
        from calendar import day_name

        # --- NLP
        import spacy
        from spacytextblob.spacytextblob import SpacyTextBlob   # sentiment analysis

        import nltk
        from nltk import ngrams
        from nltk.stem import WordNetLemmatizer
        from nltk import pos_tag, word_tokenize

        from emoji import is_emoji

        # --- ML
        from sklearn.model_selection import train_test_split
        from sklearn import metrics

        try:
        # Configure spaCy
            nlp = spacy.load("en_core_web_md") # , disable=['parser', 'ner', 'textcat'])
            # nlp = spacy.load("en_core_web_trf", disable=['parser', 'ner', 'textcat'])
            nlp.add_pipe('spacytextblob')   # required for sentiment analysis
        except:
            print('Spacy Load unsuccesful')

        #TODO: CHANGE this Part
        # Download dataset
        #!curl -O https://owncloud.hpi.de/s/EAmstS3hKgvWiPr/download #TODO: CHANGE
        #!tar -zxvf download

    def load_data_existing(self):
        # Load dataset
        # Standard data import anywhere else:
        try:
          submission_df = pd.read_pickle('/Users/moritzschneider/Downloads/submission_df_pickle')
        except:
            print('Submissions Load Fail')

        # Drop unnecessary columns:
        # Import list of columns to be removed:
        column_list = pd.read_csv('/Users/moritzschneider/Downloads/wsb_column_details.csv', sep=';')


        exclude_columns = column_list.loc[column_list['Use for Classification'] == 'NO']['Column'].apply(lambda x: x.strip()).to_list()

        exclude_columns_2 = ['is_created_from_ads_ui',
                          'is_crosspostable', 'is_meta', 'is_robot_indexable', 'link_flair_background_color',
                          'link_flair_css_class', 'link_flair_richtext',
                          'link_flair_text_color', 'parent_whitelist_status',
                          'preview', 'pwls', 'thumbnail', 'whitelist_status', 'wls']

        exclude_columns = exclude_columns + exclude_columns_2

        submission_df_sm = submission_df.drop(columns=exclude_columns)
        # Order posts ascending by date
        submission_df_sm = submission_df_sm.reindex(index=submission_df_sm.index[::-1])
        submission_df_sm.reset_index(inplace=True, drop=True)

        # Create idx for converting categorical columns to numbers
        # convert author to index
        author_idx = {}
        for a in submission_df_sm['author']:
            if a in author_idx:
                pass
            else:
                author_idx[a] = len(author_idx)

        # author_flair_text
        author_flair_idx = {}
        for a in submission_df_sm['author_flair_text']:
            if a in author_flair_idx:
                pass
            else:
                author_flair_idx[a] = len(author_flair_idx)

        # link_flair_text
        link_flair_idx = {}
        for l in submission_df_sm['link_flair_text']:
            if l in link_flair_idx:
                pass
            else:
                link_flair_idx[l] = len(link_flair_idx)

        # id
        id_idx = {}
        for i in submission_df_sm['id']:
            if i in id_idx:
                pass
            else:
                id_idx[i] = len(id_idx)

        # https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        # Table accessible via: //*[@id="constituents"] or #constituents

        #TODO:
        # sp500_wiki = pd.read_html('/Users/moritzschneider/Downloads/220924_SP500_wiki.html')[0]
        # sp500_symbols = sp500_wiki['Symbol'].tolist()
        #
        # sp500_symbols = [s.replace('.', '-') for s in sp500_symbols]
        #
        # print(len(sp500_symbols))  # contains 505 symbols in total
        # print(sp500_symbols[:10])
        #
        # stock_idx = {}
        # for i, s in enumerate(sp500_symbols):
        #     stock_idx[s] = i

        # stock_idx

        # Extract S&P 500 stock market data from Yahoo! Finance API
        # sp500_data = yf.Ticker('^GSPC')
        #
        # # Extract S&P 500 price history
        # sp500_prices = sp500_data.history(start="2018-01-01", end="2022-07-03")

        # sp500_prices = pd.read_csv('/Users/moritzschneider/Downloads/sp500_data.csv')
        # sp500_prices['SP500_change_1d'] = (sp500_prices['Close'].shift(periods=-1)*100/sp500_prices['Close'])-100
        # sp500_prices['SP500_change_3d'] = (sp500_prices['Close'].shift(periods=-3)*100/sp500_prices['Close'])-100
        # sp500_prices['SP500_change_1w'] = (sp500_prices['Close'].shift(periods=-7)*100/sp500_prices['Close'])-100
        # sp500_prices['SP500_change_1m'] = (sp500_prices['Close'].shift(periods=-30)*100/sp500_prices['Close'])-100
        # sp500_prices['SP500_change_3m'] = (sp500_prices['Close'].shift(periods=-90)*100/sp500_prices['Close'])-100
        # sp500_change = sp500_prices[['SP500_change_1d', 'SP500_change_3d', 'SP500_change_1w', 'SP500_change_1m', 'SP500_change_3m']]
        #
        # print(sp500_change)
        #
        # sp500_change['date'] = sp500_change.index.strftime('%Y-%m-%d')
        sp500_change = pd.read_csv('/Users/moritzschneider/Downloads/sp500_data.csv')

        return submission_df_sm, sp500_change