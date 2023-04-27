from pre_training import pre_Training_exe
from ml_algorithms import ml_algorithms
from feature_engineering import feature_engineering
from setup import first_installer
import os
import json
import pandas as pd
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

    def get_setup(self, submission_df, stock_price, sp500_wiki):
        """
        install python packages
        """
        print('Installing')
        first = first_installer()
        first.install_first_time(submission_df, stock_price, sp500_wiki)

    def json_to_dataframe(self, json_data):
        """
        Convert a JSON object to a pandas DataFrame.

        Args:
            json_data (str or dict): JSON data to convert.

        Returns:
            pandas.DataFrame: The converted DataFrame.
        """
        # If the input is a string, parse it into a dictionary
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        # Convert the JSON data to a DataFrame
        df = pd.DataFrame.from_dict(json_data, orient='columns')

        return df

    def get_data(self, json_file_path):
        """
        Read a JSON file and convert its contents to a pandas DataFrame.

        Args:
            json_file_path (str): The path to the JSON file.

        Returns:
            pandas.DataFrame: The converted DataFrame.
        """
        print(json_file_path)
        # Check if the file exists
        if not os.path.exists(json_file_path):
            raise ValueError(f"JSON file not found: {json_file_path}")

        # Read the file contents into a string
        with open(json_file_path, 'r') as f:
            json_str = f.read()

        # Call the existing function to convert the JSON string to a DataFrame
        df = organiser_class.json_to_dataframe(self, json_str)

        return df

    def start_feature_engineering(self, submission_df_sm, sp500_change, train_split_end, test_split_end, time_horizon):
        """
        feature engineering
        """
        feature_eng = feature_engineering()
        df = feature_eng.Create_additional_potential_target_features(submission_df_sm, sp500_change)
        df = feature_eng.create_binary_feature(df)
        df = feature_eng.transform_cat_to_num_feature(df)
        df = feature_eng.create_signals(df)
        df = feature_eng.fill_numerical_cols(df)
        df = feature_eng.removing_unwanted_cols(df, time_horizon)
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

    def convert_html(self, sp500_wiki_path):
        sp500_wiki = pd.read_html(sp500_wiki_path)[0]
        sp500_symbols = sp500_wiki['Symbol'].tolist()

        sp500_symbols = [s.replace('.', '-') for s in sp500_symbols]

        stock_idx = {}
        for i, s in enumerate(sp500_symbols):
            stock_idx[s] = i

        return sp500_symbols, stock_idx

class runner():
    def __int__(self):
        self
    def run_organizer(self, time_horizon, start_date, end_date, target, submissions_path, sp500_wiki_path, stock_price):
        instance_of_organiser = organiser_class(time_horizon, start_date, end_date, target)
        instance_of_organiser.print_info()
        #instance_of_organiser.get_setup(submissions_path, stock_price, sp500_wiki_path)

        ### run
        sp500_change = instance_of_organiser.get_data(submissions_path)
        submission_df_sm = instance_of_organiser.get_data(stock_price)
        #sp500_symbols, stock_idx = instance_of_organiser.convert_html(sp500_wiki_path)

        train_final_df, test_final_df, difference_final_df, final_df = instance_of_organiser.start_feature_engineering(
            submission_df_sm, sp500_change, start_date, end_date, time_horizon)
        # TODO
        X, y, x_train, x_test, y_train, y_test = instance_of_organiser.start_preTraining(target, final_df, dataframe,
                                                                                         time_horizon, target,
                                                                                         train_final_df,
                                                                                         difference_final_df,
                                                                                         test_final_df)
        result = instance_of_organiser.start_xgBoost(X, y, x_train, x_test, y_train, y_test)
        return result