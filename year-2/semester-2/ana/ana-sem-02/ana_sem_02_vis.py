import sys

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    algorithms = ['dyn', 'exh', 'greedy', 'fptas']
    algorithm = 'dyn'
    file_path = ''

    if len(sys.argv) == 3:
        algorithm = sys.argv[1]
        file_path = sys.argv[2]
        
        if algorithm not in algorithms:
            print(f"Invalid algorithm '{algorithm}'. Choose from {list(algorithms.keys())}.")
            sys.exit(1)
    else:
        print("Invalid number of arguments. Usage: python ana_sem_02.py <algorithm> <file_path>")
        sys.exit(1)

    df = pd.read_csv(file_path, sep=",")

    # Get the number of unique test types in the test_type column
    test_types = df['test_type'].unique()

    # For each test type, plot the time taken and memory consumption based on the n and k values
    # The n and k values should be sorted in ascending order
    if algorithm in ['dyn', 'exh']:
        df = df.sort_values(by=['n', 'k'])
        for test_type in test_types:
            test_df = df[df['test_type'] == test_type]

            # Plot time taken vs n
            plt.figure(figsize=(6, 4))
            plt.plot(test_df['n'], test_df['time_taken'], marker='o', label='Time Taken')
            plt.title(f'Time Taken vs n for {test_type}')
            plt.xlabel('n')
            plt.ylabel('Time Taken (seconds)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{algorithm}/{algorithm}_{test_type}_time_vs_n.pdf")
            plt.close()

            # Plot memory used vs n
            plt.figure(figsize=(6, 4))
            plt.plot(test_df['n'], test_df['memory_used'], marker='o', label='Memory Used', color='orange')
            plt.title(f'Memory Used vs n for {test_type}')
            plt.xlabel('n')
            plt.ylabel('Memory Used (bytes)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{algorithm}/{algorithm}_{test_type}_memory_vs_n.pdf")
            plt.close()

            # Plot time taken vs k
            plt.figure(figsize=(6, 4))
            plt.plot(test_df['k'], test_df['time_taken'], marker='o', label='Time Taken')
            plt.title(f'Time Taken vs k for {test_type}')
            plt.xlabel('k')
            plt.ylabel('Time Taken (seconds)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{algorithm}/{algorithm}_{test_type}_time_vs_k.pdf")
            plt.close()

            # Plot memory used vs k
            plt.figure(figsize=(6, 4))
            plt.plot(test_df['k'], test_df['memory_used'], marker='o', label='Memory Used', color='orange')
            plt.title(f'Memory Used vs k for {test_type}')
            plt.xlabel('k')
            plt.ylabel('Memory Used (bytes)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{algorithm}/{algorithm}_{test_type}_memory_vs_k.pdf")
            plt.close()
    else:
        # If algorithm one of 'greedy' or 'fptas', do not plot time and memory usage, but plot
        # the disparity between the result and the k value
        df = df.sort_values(by=['n', 'k'])
        for test_type in test_types:
            test_df = df[df['test_type'] == test_type]

            # n is the x-axis, k should be on the y-axis. Both k and result should be plotted
            # with different colors and labels
            plt.figure(figsize=(6, 4))
            plt.plot(test_df['n'], test_df['k'], marker='o', label='k')
            plt.plot(test_df['n'], test_df['result'], marker='o', label='Result')
            plt.title(f'k vs Result for {test_type}')
            plt.xlabel('n')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{algorithm}/{algorithm}_{test_type}_k_vs_result.pdf")
            plt.close()
            
            
