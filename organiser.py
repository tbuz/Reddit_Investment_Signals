from pre_Training import pre_Training_exe
from ml_algorithms import ml_algorithms
from feature_engineering import feature_engineering
from setup import first_installer

class organiser_class:
    def __init__(self, time_horizon, start_date, end_date, target):
        self.time = time_horizon
        self.start_date = start_date
        self.end_date = end_date
        self.target = target

    def print_info(self):
        """
        displays parameters in command line
        """
        print(f"The time horizon is {self.time} days.")
        print(f"The start date is {self.start_date}.")
        print(f"The end date is {self.end_date}.")
        print(f"The target is {self.target}.")

    def get_setup(self):
        """
        install python packages
        """
        print('Installing')
        first = first_installer()
        first.install_first_time()

    def get_data(self):
        """
        get data from sources
        """
        submission_df_sm = ''
        sp500_change = ''
        return submission_df_sm, sp500_change

    def start_feature_engineering(self, submission_df_sm, sp500_change, train_split_end, test_split_end):
        """
        feature engineering
        """
        feature_eng = feature_engineering()
        df = feature_eng.Create_additional_potential_target_features(submission_df_sm, sp500_change)
        df = feature_eng.create_binary_feature(df)
        df = feature_eng.transform_cat_to_num_feature(df)
        df = feature_eng.create_signals(df)
        df = feature_eng.fill_numerical_cols(df)
        df = feature_eng.removing_unwanted_cols(df)
        train_final_df, test_final_df, difference_final_df,  final_df = feature_eng.cut_dataset_to_time_frame(df, train_split_end, test_split_end)
        return train_final_df, test_final_df, difference_final_df,  final_df
    def start_preTraining(self, target, final_df, df, time_horizon, target_name, train_final_df, difference_final_df, test_final_df):
        """
        start converting data into usable dataframe for ml_algorithms
        """
        pre_Training = pre_Training_exe()
        pre_Training.set_target(df, target_name, time_horizon)
        final_df = pre_Training.label_encode_df(final_df)
        X, y, x_train, x_test, y_train, y_test = pre_Training.train_test_split(target, final_df, train_final_df, difference_final_df, test_final_df)
        return X, y, x_train, x_test, y_train, y_test
    def start_xgBoost(self, df, X, y, x_train, x_test, y_train, y_test):
        """
        xg Boost to get performance
        """
        ml_algo = ml_algorithms()
        result = ml_algo.ml_xgBoost(df, x_train, y_train, x_test, y_test)
        return result