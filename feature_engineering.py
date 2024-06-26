import pandas as pd
import numpy as np
import datetime
class feature_engineering():
    """
        ### Features in data:
        #Investimentbanks - will give you information, if the stock was rated by one of these banks:
        'Morgan Stanley', 'Credit Suisse', 'Wells Fargo',
        'Citigroup', 'Barclays', 'Deutsche Bank', 'UBS', 'Raymond James',
        'JP Morgan', 'B of A Securities', 'BMO Capital', 'Keybanc',
        'RBC Capital', 'Goldman Sachs', 'Mizuho', 'Stifel', 'Piper Sandler',
        'Baird', 'Jefferies', 'Oppenheimer'

        #Stock indicators: Mean Average of {x} days
        'MA07', 'MA30', 'MA90'

        # development of stock in the last x days
        'prev_1w', 'prev_3d', 'prev_1d',
    """
    def __int__(self):
        self

    def Create_additional_potential_target_features(self, submission_df_sm, sp500_change):
        df = submission_df_sm.copy()

        # Create additional potential target features:
        # Has the post been successful over different time frames?
        # distinguish: negative = False, positive = True, NaN = None
        df['success_1d'] = df['change_1d'] > 0
        df['success_3d'] = df['change_3d'] > 0
        df['success_1w'] = df['change_1w'] > 0
        df['success_1m'] = df['change_1m'] > 0
        df['success_3m'] = df['change_3m'] > 0

        # Has the post been followed by top quartile stock price performance?
        # True if price growth in top 25%, else False
        df['top_1d'] = df['change_1d'] >= df['change_1d'].quantile(q=0.75)
        df['top_3d'] = df['change_3d'] >= df['change_3d'].quantile(q=0.75)
        df['top_1w'] = df['change_1w'] >= df['change_1w'].quantile(q=0.75)
        df['top_1m'] = df['change_1m'] >= df['change_1m'].quantile(q=0.75)
        df['top_3m'] = df['change_3m'] >= df['change_3m'].quantile(q=0.75)

        # If the price change is not available (due to missing data), the data point has to be excluded
        # Data is missing when
        # (1) the submission is less than 3 months old at the time of collection
        # (2) the submission does not contain a S&P 500 ticker, therefore no stock data
        df['valid_1d'] = df['change_1d'].notnull()
        df['valid_3d'] = df['change_3d'].notnull()
        df['valid_1w'] = df['change_1w'].notnull()
        df['valid_1m'] = df['change_1m'].notnull()
        df['valid_3m'] = df['change_3m'].notnull()

        #df['date'] = df.index
        # Date Feature
        sp500_change['date'] = sp500_change.index

        df['date'] = pd.to_datetime(df['date'])
        sp500_change['date'] = pd.to_datetime(sp500_change['date'])


        # Join df with S&P500 changes
        sp500_df = pd.merge(df[['date', 'change_3m']], sp500_change, on='date', how='left')
        sp500_df = sp500_df.drop(['date', 'change_3m'], axis=1)
        df = pd.concat([df, sp500_df], axis=1)
        return df

    def create_binary_feature(self, df):

        # Feature Engineering by Moritz

        #author_premium
        # features shows if the author of the post has a reddit premium account
        df.loc[df['author_premium'] == 1.0, 'author_premium_binary'] = True
        df.loc[df['author_premium_binary'] != True, 'author_premium_binary'] = False

        #send_replies
        # features shows if replies were send
        df.loc[df['send_replies'] == 1.0, 'send_replies_binary'] = True
        df.loc[df['send_replies_binary'] != True, 'send_replies_binary'] = False

        #total_awards_received
        # features shows if the post got a reddit award
        df.loc[df['total_awards_received'] == 0.0, 'total_awards_received_binary'] = False
        df.loc[df['total_awards_received_binary'] != False, 'total_awards_received_binary'] = True


        #is_original_content
        # features shows if content is original or from other sources
        df.loc[df['is_original_content'] == 1.0, 'is_original_content_binary'] = True
        df.loc[df['is_original_content_binary'] != True, 'is_original_content_binary'] = False


        #is_reddit_media_domain
        # features shows if post contains media
        df.loc[df['is_reddit_media_domain'] == 1.0, 'is_reddit_media_domain_binary'] = True
        df.loc[df['is_reddit_media_domain_binary'] != True, 'is_reddit_media_domain_binary'] = False

        #is_self
        # features shows if
        df.loc[df['is_self'] == 1.0, 'is_self_binary'] = True
        df.loc[df['is_self_binary'] != True, 'is_self_binary'] = False
        #df['is_self_binary'].value_counts()

        #is_video
        # features shows if post is a video
        df.loc[df['is_video'] == 1.0, 'is_video_binary'] = True
        df.loc[df['is_video_binary'] != True, 'is_video_binary'] = False

        #selftext #Extra Feld
        # features shows if post was deleted
        df['selftext_deleted'] = df['selftext'].apply(lambda x: x in ['deleted', "[removed]", ''])

        #tickers #Check ob mehrere Einträge im Array
        # features shows if post has stock tickers
        df['unique_ticker_count'] = df['tickers'].apply(lambda x: len(x))

        return df

    # Bag Of Tickers Vector
    def calculate_BoT(self,tickers_counter, stock_idx):
        BoT_vector = np.zeros([len(stock_idx)], dtype=int)
        for ticker in tickers_counter.keys():
            idx = stock_idx[ticker]
            count = tickers_counter[ticker]
            BoT_vector[idx] += count
        return BoT_vector

    def transform_cat_to_num_feature(self, df, author_idx, author_flair_idx, id_idx, link_flair_idx):
        try:
          df['author'] = df['author'].apply(lambda x: author_idx[x])
          df['author_flair_text'] = df['author_flair_text'].apply(lambda x: author_flair_idx[x])

          # Translate categorical features into numerical values
          df['id'] = df['id'].apply(lambda x: id_idx[x])
          df['link_flair_text'] = df['link_flair_text'].apply(lambda x: link_flair_idx[x])
        except KeyError:
          pass

        return df

    def create_signals(self, df):

        df['day_utc'] = df['created_utc']%86400   # round for full days

        df['total_awards_received'] = df['total_awards_received'].fillna(0)
        df['is_original_content'] = df['is_original_content'].fillna(False)
        df[['BUY_signal', 'SELL_signal', 'BUY_MA30', 'any_buy_MA30']] = \
        df[['BUY_signal', 'SELL_signal', 'BUY_MA30', 'any_buy_MA30']].fillna(False)
        return df

    def fill_numerical_cols(self, df):
        # Fill numerical columns that contain NaN values
        numerical_columns = ['any_buy', 'Morgan Stanley', 'Credit Suisse', 'Wells Fargo',
            'Citigroup', 'Barclays', 'Deutsche Bank', 'UBS', 'Raymond James',
            'JP Morgan', 'B of A Securities', 'BMO Capital', 'Keybanc',
            'RBC Capital', 'Goldman Sachs', 'Mizuho', 'Stifel', 'Piper Sandler',
            'Baird', 'Jefferies', 'Oppenheimer', 'prev_1w', 'prev_3d', 'prev_1d',
            'MA07', 'MA30', 'MA90',
            'Volatility', 'count_window']
        df[numerical_columns] = df[numerical_columns].fillna(0)

        # Replace infinite values
        df = df.replace(np.inf, 0.0)
        df = df.replace(-np.inf, 0.0)

        df['subreddit_subscribers'] = df['subreddit_subscribers'].fillna(method='ffill') #, axis=1)
        df['upvote_ratio'] = df['upvote_ratio'].fillna(1)

        return df

    def removing_unwanted_cols(self, df, time_horizon):
        # Remove unwanted columns (except for target, which is dropped later)
        # remove spacy_vector if sv_df is created and concatted to df
        unwanted_columns = ['url', 'change_1d', 'change_3d', 'change_1w', 'change_1m',
                            'change_3m',
                            'title', 'selftext', 'media_embed', 'secure_media',
                            'removed_by_category', 'domain', 'full_link', 'gildings',
                            'success_1d', 'success_3d', 'success_1w', 'success_1m',
                            'author_premium',
                            'success_3m',
                            'top_1d', 'top_3d', 'top_1w', 'top_1m',
                            'top_3m', 'valid_1d', 'valid_3d', 'valid_1w',
                            'valid_1m', 'valid_3m',
                            'no_follow', 'send_replies', # 'spacy_vector',
                            'send_replies_binary',
                            'SP500_change_1d', 'SP500_change_3d', 'SP500_change_1w',
                            'SP500_change_1m', 'SP500_change_3m']
        final_df = df.loc[df[f'valid_3m']].copy().drop(columns=unwanted_columns)

        final_df = final_df.loc[final_df['BUY_signal'] == True]
        final_df = final_df.reset_index()
        return final_df
    def cut_dataset_to_time_frame(self, final_df, train_split_end, test_split_end ):
        #### Train-Test-Split
        final_df['split_date'] = final_df['created_utc']

        train_split_end_date = datetime.datetime.strptime(train_split_end, '%d-%m-%Y')
        train_split_end_conv = int(train_split_end_date.timestamp())
        test_split_start_conv = int(train_split_end_date.timestamp())

        test_split_end_date = datetime.datetime.strptime(test_split_end, '%d-%m-%Y')
        test_split_end_conv = int(test_split_end_date.timestamp())

        train_final_df =  final_df[final_df['split_date'] < train_split_end_conv]
        test_final_df =  final_df[final_df['split_date'].between(test_split_start_conv, test_split_end_conv)]
        difference_final_df = final_df[final_df['split_date'].between(train_split_end_conv, test_split_start_conv)]

        #print(len(train_final_df))
        #print(len(test_final_df))

        return train_final_df, test_final_df, difference_final_df,  final_df
