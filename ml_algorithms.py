#Classifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

class ml_algorithms():
    def __int__(self):
        self

    def ml_xgBoost(self, df, label_encoder, X_train, y_train, X_test, y_test, time_horizon, target):

        X_train = X_train.apply(label_encoder.fit_transform)
        X_test = X_test.apply(label_encoder.fit_transform)
        y_train = y_train.apply(label_encoder.fit_transform)
        y_test = y_test.apply(label_encoder.fit_transform)

        # xgBoost Classifier
        # Create an instance of the XGBClassifier class with some hyperparameters
        # enable_categorical=True: enable handling of categorical features
        # use_label_encoder=True: use scikit-learn's LabelEncoder to encode the target variable
        # max_depth=15: maximum depth of each tree in the ensemble
        # subsample=0.5: fraction of the training data to use for each tree
        xgb_cl = xgb.XGBClassifier(max_depth=15, subsample=0.5)

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
        print('Balanced accuracy: ', balanced_accuracy_score(y_test, preds))

        # target_names = ['false', 'true']
        print(classification_report(y_test, preds))  # , target_names=target_names))

        # Print the predicted class probabilities of the test data
        # print(preds_proba)

        #Validation if xgBoost Works
        #print(y_test.mean())
        #print(preds.mean())

    # TODO: Target abfragen:

        # #TODO: adapt to target
        if target == 'target_3':
            #list(preds).index(True)
            R = ['3','4']
            N = []

            for i in range(len(preds)):

                if preds[i] in R:
                    N.append(i)

            #print(N)

            #TODO:
            values = X_test.iloc[N].index.values.tolist()

            df_validated = df.loc[df[f'valid_{time_horizon}']].copy()
            df_validated[f'change_{time_horizon}'] += 100
            print('Stock Performance: ',df_validated[f'change_{time_horizon}'].iloc[values].mean())
            return df_validated

        if target == 'target_1' or target == 'target_2':
            #TODO: Target abfragen:
            # list(preds).index(True)
            True_value = [1]
            indexing_value = []

            for i in range(len(preds)):

                if preds[i] in True_value:
                    indexing_value.append(i)

            #print(indexing_value)
            #
            # #TODO:
            values = X_test.iloc[indexing_value].index.values.tolist()
            df_validated = df.loc[df[f'valid_{time_horizon}']].copy()
            df_validated[f'change_{time_horizon}'] += 100
            print('Stock Performance: ',df_validated[f'change_{time_horizon}'].iloc[values].mean())

            return df_validated

        else:
            print('Wrong Target')