#from nltk.book import *
#text1.concordance("nature.txt")'
import nltk
nltk.download("punkt_tab")
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    print("ilosc tokenow: ", len(tokens))
     # d) - custom usuwa 21 tokenow
    custom_stop_words = ['et', 'al', 'also', 'tmn','fig','b']
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stop_words) 
    #print(len(stop_words))
    #print(stop_words)
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    #print(filtered_tokens)
    print("po filtorowaniu: ", len(filtered_tokens))
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    #print(lemmatized_tokens)
    print("po lemizatorze: ", len(lemmatized_tokens))
    return processed_text

with open('nature.txt', 'r', encoding='utf-8') as f:
        text = f.read()

processed_text = preprocess_text(text)
#print(processed_text)

import matplotlib.pyplot as plt
from collections import Counter

word_counts = Counter(processed_text.split())


most_common_words = word_counts.most_common(10)
words, counts = zip(*most_common_words)

plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
from wordcloud import WordCloud

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud')
plt.show()
