#Hey and welcome to Reddit Investment Signals
from organiser import organiser_class


# set your PARAMETER here
time_horizon = 3
start_date = '01-01-2021'
end_date = '01-04-2021'


#if you set your parameter press play!
if __name__ == '__main__':
    example_instance = organiser_class(time_horizon, start_date, end_date)
    example_instance.print_info()