import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
csv_file = "all_comments_emotions.csv"
try:
    data = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found. Ensure the file exists and try again.")
    exit()

# Group by 'act' and calculate the mean for emotion columns
emotions = ["Happy", "Angry", "Surprise", "Sad", "Fear"]
average_per_act = data.groupby("act")[emotions].mean()

# Display the average values for each act
print("Average Emotion Scores per Act:")
print(average_per_act)

average_per_act.plot(kind="bar", figsize=(12, 6))

# Customize the plot
plt.title("Average Emotion Scores per Act", fontsize=16)
plt.ylabel("Average Score", fontsize=12)
plt.xlabel("Acts", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Emotions", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plo
plt.show()
episodes_data = average_per_act[average_per_act.index.str.contains("Episode")]

# Plot the filtered data
episodes_data.plot(kind="line", figsize=(12, 6), marker='o')

# Customize the plot
plt.title("Average Emotion Scores per Episode", fontsize=16)
plt.ylabel("Average Score", fontsize=12)
plt.xlabel("Episodes", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Emotions", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

