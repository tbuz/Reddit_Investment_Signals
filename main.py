#Hey and welcome to Reddit Investment Signals
from organiser import runner

if __name__ == '__main__':
    #All packages are installed {True, False}
    installed_flag = True
    # set your PARAMETER here
    # in days {'1d', '3d', '1w', '1m', '3m'}
    time_horizon = '3d'
    # start and end-date of Timeframe you want to look at in DD-MM-YYYY
    start_date = '01-07-2021'
    end_date = '01-10-2021'
    # target {'target_1','target_2', 'target_3' } (see Read_me or Paper)
    target = 'target_2'

    #data submissions_path
    submissions_path = '/Users/moritzschneider/Downloads/submission_df_pickle' #'submission_df_pickle'
    config_file = '/Users/moritzschneider/Downloads/sp500_data.csv' #'sp500_data.csv'
    sp500_data = '/Users/moritzschneider/Downloads/wsb_column_details.csv' #'wsb_column_details.csv'

    # if you set your parameter press play!
    #TODO: in eine Klasse // Bash script // Parameter Grenzen
    runner = runner()
    runner.run_organizer(time_horizon= time_horizon, start_date = start_date, end_date=end_date, target= target, submissions_path = submissions_path, sp500_data = sp500_data, config_file = config_file, flag = installed_flag)