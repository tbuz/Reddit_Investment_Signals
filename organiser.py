#import pre_Training
#import xgBoost
#import feature_engineering

class organiser_class:
    def __init__(self, time_horizon, start_date, end_date):
        self.time = time_horizon
        self.start_date = start_date
        self.end_date = end_date

    def print_info(self):
        print(f"The time horizon is {self.time} days.")
        print(f"The start date is {self.start_date}.")
        print(f"The end date is {self.end_date}.")