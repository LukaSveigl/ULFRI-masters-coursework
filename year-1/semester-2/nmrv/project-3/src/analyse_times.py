import numpy as np

def aggregate_data() -> tuple:
    init_times = []
    tracking_times = []

    with open('tracking_times.txt', 'r') as file:
        times = file.readlines()

    for i in range(len(times)):
        if times[i].startswith('Initialization time:'):
            init_times.append(float(times[i].split(': ')[1]))
        else:
            tracking_times.append(float(times[i].split(': ')[1]))

    return init_times, tracking_times

def analyse_init_time() -> float:
    """
    Calculates the average initialization time of the tracker.

    Returns:
        float: The average initialization time.
    """
    # Open the file and read the lines
    with open('init_times.txt', 'r') as file:
        lines = file.readlines()

    # Extract the initialization times
    times = [float(line.split(': ')[1]) for line in lines if line.strip()]

    # Calculate the average
    average_time = sum(times) / len(times)

    print(f'Average initialization time: {average_time}')


def analyse_tracking_time() -> float:
    """
    Calculates the average tracking time of the tracker.

    Returns:
        float: The average tracking time.
    """
    # Open the file and read the lines
    with open('tracking_times.txt', 'r') as file:
        lines = file.readlines()

    # Extract the tracking times
    times = [float(line.split(': ')[1]) for line in lines if line.strip()]

    # Calculate the average
    average_time = sum(times) / len(times)

    print(f'Average tracking time: {average_time}')


if __name__ == '__main__':
    init_times, tracking_times = aggregate_data()
    print(f'Average initialization time: {np.mean(init_times)}')
    print(f'Average tracking time: {np.mean(tracking_times)}')
    #analyse_init_time()
    #analyse_tracking_time()