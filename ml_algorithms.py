#Classifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

class ml_algorithms():
    def __int__(self):
        self

    def ml_xgBoost(self, df, X_train, y_train, X_test, y_test):
        #xgBoost Classifier
        xgb_cl = xgb.XGBClassifier(enable_categorical=True, use_label_encoder=True, max_depth=15, subsample=0.5)

        xgb_cl.fit(X_train,y_train)

        preds = xgb_cl.predict(X_test)

        preds_proba = xgb_cl.predict_proba(X_test)

        print(balanced_accuracy_score(y_test, preds))

        print(preds_proba)

        #list(preds).index(True)
        R = ['3','4']
        N = []

        for i in range(len(preds)):

            if preds[i] in R:
                N.append(i)

        print(N)

        values = X_test.iloc[N].index.values.tolist()

        df_validated = df.loc[df['valid_3m']].copy()
        df_validated['change_3m'] += 100
        df_validated['change_3m'].iloc[values].mean()

        values = X_test.iloc[N].index.values.tolist()

        df_validated = df.loc[df['valid_3m']].copy()
        df_validated['change_3m'] += 100
        df_validated['change_3m'].iloc[values].mean()

        df_validated['change_3m'].iloc[values]

        return df_validated