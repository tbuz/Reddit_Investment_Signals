#Hey and welcome to Reddit Investment Signals
from organiser import runner

if __name__ == '__main__':
    # set your PARAMETER here
    # in days (1, 3, 7, 30, 90)
    time_horizon = 3
    # start and end-date in DD-MM-YYYY (min: - max:)
    start_date = '01-01-2021'
    end_date = '01-04-2021'
    # target (see Read_me or Paper)
    target = 'target_1'

    # if you set your parameter press play!

    #TODO: in eine Klasse // Bash script // Parameter Grenzen
    runner = runner()
    runner.run_organizer(time_horizon= time_horizon, start_date = start_date, end_date=end_date, target= target)