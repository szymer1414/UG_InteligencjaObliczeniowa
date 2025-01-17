
#import nltk
#nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
textpos = """
I’m absolutely blown away by the staff. People are very nice helpful and considerate. I want to especially thank Marta since she was super helpful and took care of me , finding me a room that is ready and not letting me wait. I truly appreciate the care and thought she put in. Thank you again"""
textneg = """
The rooms are ridiculously small. Like barely have room to walk around bed. The bathroom was okay if you ignore the fact that the shower head has mold on it and would spray the entire bathroom every time we tried to take a shower.
Location wise it’s 30min away from the center of Rome which wouldn’t be an issue if taxis or Ubers would take the ride but absolutely no one will want to take you over there since as the 20th Uber driver that finally took the ride said “getting into that area is never worth it because of traffic”. We literally were stuck in Rome until 11pm trying to find a taxi. Room wasn’t very clean either
"""
textpos ="""This is the best McDonald I’ve been to. The stuff are so nice and kind, they smile at everyone and the food was really quick and very tasty. Very recommended ;)"""
textneg ="""I waited an age for my order and when it did arrive it was stone cold and disgusting! I wanted a refund! I asked for a refund! The manager handed me another burger and I asked what about my refund! Disgusting! I will think twice in future in eating at a McDonalds! The food is crap! The service is crap! And the managers are crap!"""


result1 = sia.polarity_scores(textpos)
result2 = sia.polarity_scores(textneg)
print(result1)  
print(result2) 

#BERT
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

result = classifier(textpos)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")

result = classifier(textneg)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")