#Hey and welcome to Reddit Investment Signals
from organiser import runner

if __name__ == '__main__':
    #All packages are installed {True, False}
    installed_flag = True
    # set your PARAMETER here
    # in days {'1d', '3d', '1w', '1m', '3m'}
    time_horizon = '1m'
    # start and end-date of Timeframe you want to look at in DD-MM-YYYY
    start_date = '01-07-2021'
    end_date = '01-10-2021'
    # target {'target_1','target_2', 'target_3' } (see Read_me or Paper)
    target = 'target_1'

    #data submissions_path
    submissions_path = 'datasets/submission_df_pickle'
    config_file = 'datasets/sp500_data.csv'
    sp500_data = 'datasets/wsb_column_details.csv'

    # if you set your parameter press play!
    runner = runner()
    runner.run_organizer(time_horizon= time_horizon, start_date = start_date, end_date=end_date, target= target, submissions_path = submissions_path, sp500_data = sp500_data, config_file = config_file, flag = installed_flag)