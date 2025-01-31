import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    # Define stop words
    custom_stop_words = ['et', 'al', 'also', 'tmn', 'fig', 'b','im','didnt','doesnt','got','make','dont','cant','couldnt','wouldnt','shouldnt','isnt','arent','wasnt','werent','havent','hasnt','hadnt','wont']
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stop_words)
    
    # Filter tokens
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# Load the data from JSON file
json_file = "reddit_comments.json"
try:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{json_file}' was not found.")
    exit()

# Create a directory for saving word clouds
os.makedirs("wordclouds", exist_ok=True)

# Process each episode/act individually
for act, comments in data.items():
    # Combine all comments for the current act
    combined_text = " ".join(comments)
    
    # Preprocess the text
    processed_text = preprocess_text(combined_text)
    
    # Count word frequencies
    word_counts = Counter(processed_text.split())
    most_common_words = word_counts.most_common(10)
    
    # Save word frequencies to a CSV file
    word_counts_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    word_counts_df.to_csv(f'wordfreq/word_frequencies_{act}.csv', index=False)
    print(f"Word frequencies for {act} saved to 'word_frequencies_{act}.csv'.")
    
    # Bar chart of most common words
    words, counts = zip(*most_common_words)
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Frequent Words - {act}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
    
    # Generate and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {act}')
    #plt.show()
    
    # Save the word cloud as an image
    wordcloud.to_file(f"wordclouds/wordcloud_{act}.png")
    print(f"Word cloud for {act} saved to 'wordclouds/wordcloud_{act}.png'.")
