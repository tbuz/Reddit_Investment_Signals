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
    example_instance = organiser_class(time_horizon, start_date, end_date, target)
    example_instance.print_info()
    example_instance.get_setup()