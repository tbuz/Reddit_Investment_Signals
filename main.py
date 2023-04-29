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
    submissions_path = '/Users/moritzschneider/Downloads/WSB_DSAA/datasets/stock_dfs_2022-07-03.json'
    sp500_wiki = '/Users/moritzschneider/Downloads/WSB_DSAA/datasets/220924_SP500_wiki'
    stock_price_path = '/Users/moritzschneider/Downloads/WSB_DSAA/datasets/sp500_stock_prices_2022-07-03.json'

    # if you set your parameter press play!
    #TODO: in eine Klasse // Bash script // Parameter Grenzen
    runner = runner()
    runner.run_organizer(time_horizon= time_horizon, start_date = start_date, end_date=end_date, target= target, submissions_path = submissions_path, sp500_wiki_path = sp500_wiki, stock_price_path = stock_price_path, flag = installed_flag)