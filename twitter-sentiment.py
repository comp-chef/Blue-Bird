from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

Tweet = "@twitter it's raining tonight 😒 https://www.metoffice.gov.uk/weather"

# Pre-process tweet
Tweet_words = []

for word in Tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'

    elif word.startswith('http'):
        word = "http"
    Tweet_words.append(word)

Tweet_proc = " ".join(Tweet_words)

# Load the model and tokenizer
roBERTa = "cardiffnlp/twitter-roberta-base-sentiment"

Model = AutoModelForSequenceClassification.from_pretrained(roBERTa)
Tokenizer = AutoTokenizer.from_pretrained(roBERTa)

# Sentiment analysis
Encoded_tweet = Tokenizer(Tweet_proc, return_tensors = 'pt')

Output = Model(Encoded_tweet['input_ids'], Encoded_tweet['attention_mask'])

Scores = Output[0][0].detach().numpy()

Scores = softmax(Scores)

Labels = ['Negative', 'Neutral', 'Positive']

for i in range(len(Scores)):

    L = Labels[i]
    S = Scores[i]
    
    print(L,S)

