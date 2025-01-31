import json
import pandas as pd
import text2emotion as te
import matplotlib.pyplot as plt

# Load the data
with open("reddit_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a list to store results for all comments
results = []

# Analyze each comment and store results
for act, comments in data.items():
    print(f"Processing comments for {act}...")
    for comment in comments:
        emotions = te.get_emotion(comment)
        emotions['act'] = act  # Add act identifier
        emotions['comment'] = comment  # Add original comment
        results.append(emotions)

# Save all results to a CSV
raw_results_df = pd.DataFrame(results)
raw_results_df.to_csv("all_comments_emotions.csv", index=False)
print("Saved raw emotion results for all comments to 'all_comments_emotions.csv'.")

# Calculate averages per act
average_results_df = raw_results_df.groupby('act').mean()
average_results_df.to_csv("average_emotions_per_act.csv")
print("Saved average emotion results per act to 'average_emotions_per_act.csv'.")

# Display the averages
print("Average Emotion Analysis for Acts:")
#print(average_results_df)

