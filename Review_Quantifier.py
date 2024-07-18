import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required nltk data
nltk.download('vader_lexicon')

# Function to calculate average satisfaction score
def calculate_average_satisfaction(reviews):
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    scores = []
    
    for review in reviews:
        # Get sentiment scores
        sentiment = sia.polarity_scores(review)['compound']
        # Round sentiment score to 2 decimal places
        score = round(sentiment, 2)
        scores.append(score)
    
    # Calculate average score
    average_score = round(sum(scores) / len(scores), 2) if scores else 0
    return average_score


