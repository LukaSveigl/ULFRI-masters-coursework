import matplotlib.pyplot as plt

# Data
motion_models = ['RW', 'NCV', 'NCA']

data = {
    'RW': {'N': [50, 60, 70, 80, 90, 100],
           'Failures': [105, 109, 105, 106, 107, 110],
           'Avg_overlap': [0.4, 0.39, 0.4, 0.41, 0.39, 0.4],
           'FPS': [190.71, 161.81, 139.41, 128.22, 111.89, 100.46]},
    
    'NCV': {'N': [50, 60, 70, 80, 90, 100],
            'Failures': [62, 63, 49, 58, 60, 51],
            'Avg_overlap': [0.46, 0.46, 0.47, 0.48, 0.5, 0.47],
            'FPS': [178.77, 156.13, 135.18, 115, 109.11, 97.04]},
    
    'NCA': {'N': [50, 60, 70, 80, 90, 100],
            'Failures': [308, 291, 277, 246, 233, 204],
            'Avg_overlap': [0.5, 0.51, 0.5, 0.51, 0.51, 0.51],
            'FPS': [95.89, 90.67, 84, 84.95, 80.03, 75.79]}
}

# Plotting and Exporting
for key in ['Failures', 'Avg_overlap', 'FPS']:
    plt.figure()
    for model in motion_models:
        plt.plot(data[model]['N'], data[model][key], label=model)
    plt.title(key)
    plt.xlabel('N')
    plt.ylabel(key)
    plt.legend()
    plt.savefig(f'results/particles/{key}.pdf')

plt.show()