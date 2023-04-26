#### Define Target values
##### Target 1
target_1 = df['success_3m'].loc[df['valid_3m']]

##### Target 2
target_2 = (df['change_3m'] > df['SP500_change_3m']).loc[df['valid_3m']]

##### Target 3
df['change_3m'].loc[df['valid_3m'] == True].head()
# Create categorical target column out of numerical values using quantiles:
amount_of_bins = 5
df['change_3m_cat'] = pd.qcut(df['change_3m'].loc[df['valid_3m'] == True], 
                              amount_of_bins, 
                              labels=[str(x) for x in range(0, amount_of_bins)]
                              )
# Check categorical labels
target_3 = df['change_3m_cat'].loc[df['valid_3m']] 
target_3.unique()

###Label Encoder for xGBoost
#Label Encoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# lab_enc_col=['author', 'author_flair_text', 'author_premium', 'tickers', 'date', 'top_ticker', 'spacy_vector', 'BUY_signal', 'SELL_signal', 'weekday', 'BUY_MA30', 'any_buy_MA30', 'author_premium_binary', 'send_replies_binary', 'total_awards_received_binary', 'is_original_content_binary', 'is_reddit_media_domain_binary', 'is_self_binary', 'is_video_binary', 'emoji_in_author_flair', 'emoji_in_selftext', 'BoT_vector']
lab_enc_col = ['author_premium_binary', 'tickers', 'date', 'top_ticker', 'spacy_vector', 
               'weekday', 'total_awards_received_binary', 'is_original_content_binary', 'is_reddit_media_domain_binary', 'is_self_binary', 'is_video_binary', 'emoji_in_author_flair', 'emoji_in_selftext']

le = LabelEncoder()

# Apply LabelEncoder on final_df
final_df[lab_enc_col] = final_df[lab_enc_col].astype(str)
final_df[lab_enc_col] = final_df[lab_enc_col].apply(le.fit_transform)

final_df = final_df.drop(['created_utc', 'id', 'date', 'subreddit_subscribers', 'author', 'day_utc'], axis=1)

#Split Dataset into
y = target_3

#variables
X = final_df # .drop(['success_3m'], axis = 1)

#newSplit
sum_split = len(train_final_df)+len(difference_final_df)
sum_split_2 = len(train_final_df)+len(test_final_df)+len(difference_final_df)

target_3 = target_3.reset_index()
final_df = final_df.reset_index()

target_3 = target_3.drop(['index'], axis=1)
final_df = final_df.drop(['index'], axis=1)

X_train = final_df.iloc[:len(train_final_df)]
X_test = final_df.loc[sum_split:sum_split_2, :]

y_train = target_3.iloc[:len(train_final_df)]
y_test = target_3.loc[sum_split:sum_split_2]
