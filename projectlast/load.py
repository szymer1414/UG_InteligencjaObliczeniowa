import praw
import re
import json

reddit = praw.Reddit(
    client_id="wgLER6y-FS37eVwopxrapA",
    client_secret="uywwh4_kF8tFZWUSQ6KWvkHY35k_xw",
    user_agent="Old-Tell-7733"
)
def fetch_comments(thread_url):
    try:
        submission = reddit.submission(url=thread_url)
        submission.comments.replace_more(limit=0)
        return [comment.body for comment in submission.comments.list()]
    except Exception as e:
        print(f"Error fetching comments from {thread_url}: {e}")
        return []
def clean_text(text):
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"@\S+", "", text)  
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    return text.lower().strip()
'''
act_1_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gmy7xx/s2_act_1_spoilers_arcane_season_2_act_1_discussion/")
act_2_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gscu22/s2_act_2_spoilers_arcane_season_2_act_2_discussion/")
act_3_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gxtynf/s2_act_3_spoilers_arcane_season_2_act_3_discussion/")
episode_1_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gnouvk/s2_act_1_spoilers_arcane_2x01_heavy_is_the_crown/")
episode_2_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gnovdo/s2_act_1_spoilers_arcane_2x02_watch_it_all_burn/")
episode_3_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gnovv2/s2_act_1_spoilers_arcane_2x03_finally_got_the/")
episode_4_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gscume/s2_act_2_spoilers_arcane_2x04_paint_the_town_blue/")
episode_5_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gscutq/s2_act_2_spoilers_arcane_2x05_blisters_and/")
episode_6_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gscv3n/s2_act_2_spoilers_arcane_2x06_the_message_hidden/")
episode_7_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gxtzcy/s2_act_3_spoilers_arcane_2x07_pretend_like_its/")
episode_8_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gxtzlg/s2_act_3_spoilers_arcane_2x08_killing_is_a_cycle/")
episode_9_comments = fetch_comments("https://www.reddit.com/r/arcane/comments/1gxtzta/s2_act_3_spoilers_arcane_2x09_the_dirt_under_your/")

data = {
    "Act 1": act_1_comments,
    "Act 2": act_2_comments,
    "Act 3": act_3_comments,
    "Episode 1": episode_1_comments,
    "Episode 2": episode_2_comments,
    "Episode 3": episode_3_comments,
    "Episode 4": episode_4_comments,
    "Episode 5": episode_5_comments,
    "Episode 6": episode_6_comments,
    "Episode 7": episode_7_comments,
    "Episode 8": episode_8_comments,
    "Episode 9": episode_9_comments
}
'''
threads = {
    "Act 1": "https://www.reddit.com/r/arcane/comments/1gmy7xx/s2_act_1_spoilers_arcane_season_2_act_1_discussion/",
    "Act 2": "https://www.reddit.com/r/arcane/comments/1gscu22/s2_act_2_spoilers_arcane_season_2_act_2_discussion/",
    "Act 3": "https://www.reddit.com/r/arcane/comments/1gxtynf/s2_act_3_spoilers_arcane_season_2_act_3_discussion/",
    "Episode 1": "https://www.reddit.com/r/arcane/comments/1gnouvk/s2_act_1_spoilers_arcane_2x01_heavy_is_the_crown/",
    "Episode 2": "https://www.reddit.com/r/arcane/comments/1gnovdo/s2_act_1_spoilers_arcane_2x02_watch_it_all_burn/",
    "Episode 3": "https://www.reddit.com/r/arcane/comments/1gnovv2/s2_act_1_spoilers_arcane_2x03_finally_got_the/",
    "Episode 4": "https://www.reddit.com/r/arcane/comments/1gscume/s2_act_2_spoilers_arcane_2x04_paint_the_town_blue/",
    "Episode 5": "https://www.reddit.com/r/arcane/comments/1gscutq/s2_act_2_spoilers_arcane_2x05_blisters_and/",
    "Episode 6": "https://www.reddit.com/r/arcane/comments/1gscv3n/s2_act_2_spoilers_arcane_2x06_the_message_hidden/",
    "Episode 7": "https://www.reddit.com/r/arcane/comments/1gxtzcy/s2_act_3_spoilers_arcane_2x07_pretend_like_its/",
    "Episode 8": "https://www.reddit.com/r/arcane/comments/1gxtzlg/s2_act_3_spoilers_arcane_2x08_killing_is_a_cycle/",
    "Episode 9": "https://www.reddit.com/r/arcane/comments/1gxtzta/s2_act_3_spoilers_arcane_2x09_the_dirt_under_your/"
}
data = {}
for key, url in threads.items():
    print(f"Fetching comments for {key}...")
    comments = fetch_comments(url)
    data[key] = [clean_text(comment) for comment in comments]

with open("reddit_comments.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Comments saved to 'reddit_comments.json'.")