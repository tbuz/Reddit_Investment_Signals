#Classifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

class ml_algorithms():
    def __int__(self):
        self

    def ml_xgBoost(self, df, X_train, y_train, X_test, y_test):

        # xgBoost Classifier
        # Create an instance of the XGBClassifier class with some hyperparameters
        # enable_categorical=True: enable handling of categorical features
        # use_label_encoder=True: use scikit-learn's LabelEncoder to encode the target variable
        # max_depth=15: maximum depth of each tree in the ensemble
        # subsample=0.5: fraction of the training data to use for each tree
        xgb_cl = xgb.XGBClassifier(enable_categorical=True, use_label_encoder=True, max_depth=15, subsample=0.5)

        # Train the XGBoost classifier on the training data
        # X_train: the training feature matrix
        # y_train: the training target vector
        xgb_cl.fit(X_train, y_train)

        # Use the trained classifier to predict the labels of the test data
        # X_test: the test feature matrix
        preds = xgb_cl.predict(X_test)

        # Use the trained classifier to predict the class probabilities of the test data
        preds_proba = xgb_cl.predict_proba(X_test)

        # Compute the balanced accuracy score of the XGBoost classifier on the test data
        # y_test: the true labels of the test data
        print(balanced_accuracy_score(y_test, preds))

        # Print the predicted class probabilities of the test data
        print(preds_proba)


        #TODO: adapt to target

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