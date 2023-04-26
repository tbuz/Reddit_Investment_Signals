#import pre_Training
#import xgBoost
#import feature_engineering
from setup import first_installer

class organiser_class:
    def __init__(self, time_horizon, start_date, end_date, target):
        self.time = time_horizon
        self.start_date = start_date
        self.end_date = end_date
        self.target = target

    def print_info(self):
        print(f"The time horizon is {self.time} days.")
        print(f"The start date is {self.start_date}.")
        print(f"The end date is {self.end_date}.")
        print(f"The target is {self.target}.")

    def get_setup(self):
        print('Installing')
        first = first_installer()
        first.install_first_time()