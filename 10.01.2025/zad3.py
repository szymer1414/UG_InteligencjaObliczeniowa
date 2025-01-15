from twscrape.api import TwitterScraper
#from twscrape import TwitterScraper

import pandas as pd
import asyncio

# Define the async scraping function
async def scrape_tweets(query, max_tweets):
    async with TwitterScraper() as scraper:
        tweets = []
        count = 0
        async for tweet in scraper.search(query):
            tweets.append({
                "id": tweet.id,
                "content": tweet.rawContent,
                "username": tweet.user.username,
                "date": tweet.date,
                "retweets": tweet.retweetCount,
                "likes": tweet.likeCount,
                "replies": tweet.replyCount,
                "url": f"https://twitter.com/{tweet.user.username}/status/{tweet.id}"
            })
            count += 1
            if count >= max_tweets:
                break
        return tweets

# Main function to run the scraper
def main():
    query = "civ6"  # Change this to your desired search query
    max_tweets = 100  # Number of tweets to scrape
    tweets = asyncio.run(scrape_tweets(query, max_tweets))
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets)
    
    # Save to CSV
    output_file = "twitter_civ6_tweets.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} tweets to '{output_file}'.")

if __name__ == "__main__":
    main()
