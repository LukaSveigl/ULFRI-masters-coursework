import pandas as pd
import matplotlib.pyplot as plt

blocked_path = './data/blocked/'
naive_path = './data/naive/'

def load_data(name: str, naive: bool = False) -> pd.DataFrame:
    """
    Load a dataset from the data folder.

    Args:
        - name: the name of the dataset to load
        - naive: whether to load the naive or blocked dataset

    Returns:
        A pandas DataFrame with the loaded dataset.
    """
    full_path = (naive_path if naive else blocked_path) + name
    df = pd.read_csv(full_path)
    return df


if __name__ == '__main__':
    file_names = ['56-56', '224-56', '224-224', '896-224']

    for name in file_names:
        naive_df = load_data(name + '.csv', naive=True)
        blocked_df = load_data(name + '.csv')

        mc = name.split('-')[0]
        nc = name.split('-')[1]

        BW_ticks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]

        plt.title('Naive vs Blocked GEMM: mc = ' + mc + ', nc = ' + nc)
        #plt.title('Blocked GEMM: mc = ' + mc + ', nc = ' + nc)
        plt.plot(naive_df['kc'], naive_df['BW'], label='Naive')
        plt.plot(blocked_df['kc'], blocked_df['BW'], label='Blocked')
        plt.yticks(BW_ticks)
        plt.ylabel('Bandwidth (MB/s)')
        plt.xlabel('kc')
        plt.legend()
        plt.show()