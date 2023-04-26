#Hey and welcome to Reddit Investment Signals
from organiser import organiser_class

# set your PARAMETER here
# in days (1, 3, 7, 30, 90)
time_horizon = 3
# start and end-date in DD-MM-YYYY
start_date = '01-01-2021'
end_date = '01-04-2021'
# target (see Read_me or Paper)
target = 'target_1'

#if you set your parameter press play!
if __name__ == '__main__':
    instance_of_organiser = organiser_class(time_horizon, start_date, end_date, target)
    instance_of_organiser.print_info()
    instance_of_organiser.get_setup()

    ### run
    submission_df_sm, sp500_change, train_split_end, test_split_end = instance_of_organiser.get_data()
    train_final_df, test_final_df, difference_final_df,  final_df= instance_of_organiser.start_feature_engineering(submission_df_sm, sp500_change, train_split_end, test_split_end)
    #TODO
    X, y, x_train, x_test, y_train, y_test = instance_of_organiser.start_preTraining(target, final_df, df, time_horizon, target, train_final_df, difference_final_df, test_final_df)
    result = instance_of_organiser.start_xgBoost(X, y, x_train, x_test, y_train, y_test)
    print(result)