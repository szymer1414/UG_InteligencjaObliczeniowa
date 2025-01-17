import praw

# Create a Reddit instance
reddit = praw.Reddit(
    client_id="wgLER6y-FS37eVwopxrapA",
    client_secret="uywwh4_kF8tFZWUSQ6KWvkHY35k_xw",
    user_agent="Old-Tell-7733"
)
#wgLER6y-FS37eVwopxrapA uywwh4_kF8tFZWUSQ6KWvkHY35k_xw Old-Tell-7733
# Get the most recent 100 posts from r/civ
subreddit = reddit.subreddit("civ")
posts = []
for post in subreddit.new(limit=100):
    posts.append({
        "title": post.title,
        "content": post.selftext,
        "url": post.url,
        "author": post.author.name,
        "date": post.created_utc
    })

import pandas as pd
df = pd.DataFrame(posts)
df.to_csv("reddit_civ_posts.csv", index=False)
print("Saved posts to reddit_civ_posts.csv")
