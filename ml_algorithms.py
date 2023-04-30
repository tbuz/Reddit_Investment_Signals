#Classifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import numpy as np
class ml_algorithms():
    def __int__(self):
        self

    def ml_xgBoost(self, df, label_encoder, X_train, y_train, X_test, y_test, time_horizon, target, df_for_target):

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
        # print(y_test.mean())
        # print(preds.mean())

        if target == 'target_3':
            #list(preds).index(True)
            R = [3,4]
            N = []

            for i in range(len(preds)):

                if preds[i] in R:
                    N.append(i)

            df = df_for_target
            values = X_test.iloc[N].index.values.tolist()

            df_validated = df.loc[df[f'valid_{time_horizon}']].copy()
            df_validated[f'change_{time_horizon}'] += 100
            #print(df_validated[f'change_{time_horizon}'].iloc[values])
            print('Stock Performance: ',df_validated[f'change_{time_horizon}'].iloc[values].mean())
            return df_validated

        if target == 'target_1' or target == 'target_2':

            True_value = [1]
            indexing_value = []

            df = df_for_target

            for i in range(len(preds)):

                if preds[i] in True_value:
                    indexing_value.append(i)

            values = X_test.iloc[indexing_value].index.values.tolist()
            df_validated = df.loc[df[f'valid_{time_horizon}']].copy()
            df_validated[f'change_{time_horizon}'] += 100
            print('Stock Performance: ',df_validated[f'change_{time_horizon}'].iloc[values].mean())

            return df_validated

        else:
            print('Wrong Target')


###The Following Code is not in use, but can be easily used by calling the function and adding always:  X_train, y_train, X_test, y_test
    def ml_SGD_classifier(self, X_train, y_train, X_test, y_test):

    # Loss hinge
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2',
                                random_state=42, alpha=1e-3,
                                max_iter=2000, tol=0.001).fit(X_train.fillna(0), y_train)

        predicted = sgd_clf.predict(X_test.fillna(0))

        # Mean Accuracy
        print(f"Class Distribution: \t{np.mean(y_test == True)}")
        print(f"Accuracy: \t\t{np.mean(y_test.squeeze() == predicted)}")
        print(classification_report(y_test, predicted))

    #Loss modified huber
        sgd_clf = SGDClassifier(loss='modified_huber', penalty='elasticnet',
                                random_state=42, l1_ratio=0.05, alpha=1e-3,
                                max_iter=500, tol=0.001).fit(X_train.fillna(0), y_train)

        predicted = sgd_clf.predict(X_test.fillna(0))

        # Mean Accuracy
        print(f"Class Distribution: \t{np.mean(y_test == True)}")
        print(f"Accuracy: \t\t{np.mean(y_test.squeeze() == predicted)}")
        print(classification_report(y_test, predicted))

        combined_list_sgd = zip(sgd_clf.feature_names_in_, sgd_clf.coef_[0])

        features = []
        for feature in combined_list_sgd:
            if feature[1] > 0.0:
                features.append(feature)

        features = features.sort(key=lambda x: x[1], reverse=True)
        print(features)

    def ml_KNN(self, X_train, y_train, X_test, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import balanced_accuracy_score

        knn = KNeighborsClassifier(n_neighbors=3)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        knn.fit(X_train.fillna(0), y_train)
        y_pred = knn.predict(X_test.fillna(0))

        print("KNN:", balanced_accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def ml_Random_Forrest(self,  X_train, y_train, X_test, y_test):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=10, max_depth=10)

        clf = clf.fit(X_train.fillna(0), y_train)

        y_pred = clf.predict(X_test.fillna(0))

        print("Random Forrest:", balanced_accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        combined_list_random_forrest = zip(X_test.columns, clf.feature_importances_)

        features = []
        for feature in combined_list_random_forrest:
            if feature[1] > 0.0:
                features.append(feature)

        features = features.sort(key=lambda x: x[1], reverse=True)
        print(features)

    def ml_SVM(self, X_train, y_train, X_test, y_test):
        from sklearn import svm
        svm_clf = svm.SVC().fit(X_train.fillna(0), y_train)
        # kernel='linear' necessary for feature importance eval, but extremely slow

        predicted = svm_clf.predict(X_test.fillna(0))

        print(f"Class Distribution: \t{np.mean(y_test == True)}")
        print(f"Accuracy: \t\t{np.mean(y_test.squeeze() == predicted)}")

        print(classification_report(y_test, predicted))

    def ml_MLP(self, X_train, y_train, X_test, y_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train.fillna(0))
        # apply same transformation to test data
        X_test_scaled = scaler.transform(X_test.fillna(0))

        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10, 10, 10, 10))

        clf.fit(X_train_scaled, y_train)

        # predict for simple class label predictions
        predicted = clf.predict(X_test_scaled)

        # predict_proba for probability distributions
        predicted_proba = clf.predict_proba(X_test_scaled)
        #print(predicted_proba)

        print(classification_report(y_test, predicted))