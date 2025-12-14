import matplotlib.pyplot as plt

# Open the file and read its content
with open('schematic.txt', 'r') as file:
    lines = file.readlines()

# Create a figure and an axes
fig, ax = plt.subplots()

y_offset = 0

# Iterate over the lines and characters
for y, line in enumerate(lines):
    for x, char in enumerate(line):
        if char == '\n':
            y_offset += 10
        if char != ' ':
            # Plot a point at the corresponding position
            ax.plot(x, -(y + y_offset), marker='o', markersize=2, color='black')

# Remove axes for a clean look
ax.axis('off')

# Save the figure to a PDF file
plt.savefig('schematic.pdf')