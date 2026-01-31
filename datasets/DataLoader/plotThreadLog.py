import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Directory containing your logs
log_directory = '.'

# Color mapping for each task
label_colors = {
    'thread_sync': 'white',
    'pan/zoom': 'purple',
    'resize': 'cyan',
    'depth_processing': 'yellow',
    'segmentation_processing': 'blue',
    'augmentations': 'red',
    'keypoints': 'black',
    'saving_test': 'gray',
    'rgb_loading': 'skyblue',
    'combined_depth_segmentation_loading': 'lightgreen',
    'depth_loading': 'lightgreen',
    'segmentation_loading': 'lightcoral',
    'update_cpu_batch': 'brown',
    'update_gpu_batch': 'green'
}

# Find all thread_*.log files
log_files = glob.glob(os.path.join(log_directory, 'thread_*.log'))
log_files.sort()

# Data storage
events = []

# Read and parse files
for log_file in log_files:
    thread_name = os.path.basename(log_file).replace('.log', '')
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                timestamp, flag, label = line.split(',')
                timestamp = int(timestamp)
                flag = int(flag)
                events.append((thread_name, label, timestamp, flag))

# Sort events by timestamp
events.sort(key=lambda x: x[2])

# Organize start and end pairs
tasks = []  # Each entry: (thread_name, label, start_time, end_time)
ongoing = {}  # Key: (thread_name, label) -> start_time

for thread_name, label, timestamp, flag in events:
    key = (thread_name, label)
    if flag == 1:  # Start
        ongoing[key] = timestamp
    elif flag == 0:  # End
        start_time = ongoing.pop(key, None)
        if start_time is not None:
            tasks.append((thread_name, label, start_time, timestamp))

# Plotting
fig, ax = plt.subplots(figsize=(22, 10))

yticks = []
yticklabels = []
y = 0

# Group tasks per thread
threads = sorted(set(thread for thread, _, _, _ in tasks))

for thread in threads:
    thread_tasks = [task for task in tasks if task[0] == thread]
    for _, label, start, end in thread_tasks:
        color = label_colors.get(label, 'gray')  # fallback to gray if unknown label
        ax.barh(y, end - start, left=start, color=color,) # edgecolor='black'
    yticks.append(y)
    yticklabels.append(thread)
    y += 1

ax.set_xlabel('Timestamp')
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('Dataloader CPU Thread Timeline')
ax.grid(True)

# Add legend
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_colors.items()]
ax.legend(handles=legend_patches, loc='upper right')

plt.tight_layout()
plt.savefig('thread_jobs_timeline.png')
print("Saved plot to thread_jobs_timeline.png")

