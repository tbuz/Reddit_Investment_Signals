#### Define Target values
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
class pre_Training_exe:
    def __int__(self):
        self

    def set_target(self, df, target_name, timeframe):
        #timeframe = 3m
        ##### Target 1
        if target_name == 'target_1':
            target_1 = df[f'success_{timeframe}'].loc[df[f'valid_{timeframe}']]
            target_df = target_1
        ##### Target 2
        if target_name == 'target_2':
            df = df.fillna(0)
            target_2 = (df[f'change_{timeframe}'] > df[f'SP500_change_{timeframe}']).loc[df[f'valid_{timeframe}']]
            #print(target_2)
            target_df = target_2
        ##### Target 3
        if target_name == 'target_3':
            df = df.fillna(0)
            df[f'change_{timeframe}'].loc[df[f'valid_{timeframe}'] == True].head()
            # Create categorical target column out of numerical values using quantiles:
            amount_of_bins = 5
            df[f'change_{timeframe}_cat'] = pd.qcut(df[f'change_{timeframe}'].loc[df[f'valid_{timeframe}'] == True],
                                          amount_of_bins,
                                          labels=[str(x) for x in range(0, amount_of_bins)]
                                          )
            # Check categorical labels
            target_3 = df[f'change_{timeframe}_cat'].loc[df[f'valid_{timeframe}']]
            target_3.unique()
            target_df = target_3

        return target_df

    def label_encode_df(self, final_df):
        ###Label Encoder for xGBoost
        #Label Encoder
        # lab_enc_col=['author', 'author_flair_text', 'author_premium', 'tickers', 'date', 'top_ticker', 'spacy_vector', 'BUY_signal', 'SELL_signal', 'weekday', 'BUY_MA30', 'any_buy_MA30', 'author_premium_binary', 'send_replies_binary', 'total_awards_received_binary', 'is_original_content_binary', 'is_reddit_media_domain_binary', 'is_self_binary', 'is_video_binary', 'emoji_in_author_flair', 'emoji_in_selftext', 'BoT_vector']
        lab_enc_col = ['author_premium_binary', 'tickers', 'date', 'top_ticker', 'spacy_vector',
                       'weekday', 'total_awards_received_binary', 'is_original_content_binary', 'is_reddit_media_domain_binary', 'is_self_binary', 'is_video_binary']

        le = LabelEncoder()

        # Apply LabelEncoder on final_df
        final_df[lab_enc_col] = final_df[lab_enc_col].astype(str)
        final_df[lab_enc_col] = final_df[lab_enc_col].apply(le.fit_transform)

        final_df = final_df.drop(['created_utc', 'id', 'date', 'subreddit_subscribers', 'author', 'day_utc'], axis=1)

        return final_df, le
    def train_test_split(self, target, final_df, train_final_df, difference_final_df, test_final_df):
        #Split Dataset into
        y = target

        X = final_df#.drop(['success_3m'], axis = 1)

        #newSplit
        sum_split = len(train_final_df)+len(difference_final_df)
        sum_split_2 = len(train_final_df)+len(test_final_df)+len(difference_final_df)

        target = target.reset_index()
        final_df = final_df.reset_index()

        target = target.drop(['index'], axis=1)
        final_df = final_df.drop(['index'], axis=1)

        X_train = final_df.iloc[:len(train_final_df)]
        X_test = final_df.loc[sum_split:sum_split_2, :]

        y_train = target.iloc[:len(train_final_df)]
        y_test = target.loc[sum_split:sum_split_2]

        #TODO: OK?
        X_train = X_train.astype(str)
        X_test = X_test.astype(str)

        return X, y, X_train, X_test, y_train, y_test