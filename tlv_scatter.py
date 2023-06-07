import json
import matplotlib.pyplot as plt

input_file = 'tlv_data_log.json'
with open(input_file, 'r') as file:
    data = json.load(file)

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define color mapping for point colors
color_mapping = {
    1020: (0, 0, 0.5),
    1010: "red",
    "other": "green"
}
num_points = 0
# Plot all data in one plot
for frame in data:
    x = frame['x']
    y = frame['y']
    z = frame['z']
    tlv_type = frame["TLV_type"]
    num_points += len(x)
    # Get the corresponding color for the TLV type
    point_color = color_mapping.get(tlv_type, color_mapping["other"])

    ax.scatter(x, y, z, color=point_color, marker='o', s=5)

text_str = f"Number of frames: {len(data)}\n Number of points: {num_points}\n"
ax.text2D(0.05, 0.05, text_str, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')

# Create the legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0, 0.5), markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=5)
]

legend_labels = ['TLV Type 1020', 'TLV Type 1010', 'Other TLV Types']

ax.legend(legend_elements, legend_labels)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('All Radar Data Points')
plt.show()