"""
Text preprocessing utilities for Suicide Ideation Detection
Based on methodology: Tokenization, Stop-word removal, Lemmatization
"""

import re
import string


# Comprehensive stop words list (replaces NLTK)
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
])

# Simple suffix-based lemmatization rules
def simple_lemmatize(word):
    """Basic suffix-stripping lemmatizer."""
    if len(word) <= 3:
        return word
    rules = [
        ('ying', 'y'), ('ies', 'y'), ('ied', 'y'),
        ('ing', ''), ('ness', ''), ('ment', ''),
        ('tion', ''), ('ations', ''), ('ational', ''),
        ('ed', ''), ('er', ''), ('ers', ''),
        ('ful', ''), ('less', ''), ('ly', ''),
        ('s', ''),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)] + replacement
    return word


def preprocess_text(text):
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation & special chars
    5. Remove digits
    6. Tokenize
    7. Remove stop words
    8. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stop words and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 8. Simple lemmatization
    tokens = [simple_lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def extract_linguistic_features(text):
    """
    Extract interpretable linguistic features for risk analysis.
    Returns a dict of feature scores.
    """
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    # Hopelessness indicators
    hopelessness_words = [
        'hopeless', 'hopelessness', 'worthless', 'worthlessness', 'pointless',
        'meaningless', 'useless', 'empty', 'nothing', 'nobody', 'never',
        'always', 'forever', 'trapped', 'stuck', 'burden', 'alone', 'lonely',
        'isolated', 'abandoned', 'unwanted', 'unloved'
    ]

    # Suicidal ideation keywords
    suicidal_keywords = [
        'suicide', 'suicidal', 'kill myself', 'end my life', 'end it all',
        'want to die', 'wish i was dead', 'better off dead', 'no reason to live',
        'not worth living', 'take my life', 'overdose', 'slit', 'hang myself',
        'jump off', 'death wish', 'goodbye forever', 'last day', 'final note'
    ]

    # Negative emotion words
    negative_emotions = [
        'depressed', 'depression', 'anxiety', 'panic', 'fear', 'terror',
        'miserable', 'suffering', 'pain', 'hurt', 'crying', 'tears',
        'heartbroken', 'devastated', 'destroyed', 'shattered', 'broken',
        'exhausted', 'tired', 'numb', 'empty', 'dark', 'darkness',
        'hate', 'hatred', 'angry', 'rage', 'despair', 'desperate'
    ]

    # Self-referential language (I, me, my)
    self_ref = sum(1 for w in words if w in ['i', 'me', 'my', 'myself', 'mine'])

    hopelessness_score = sum(1 for w in hopelessness_words if w in text_lower) / total_words
    suicidal_score = sum(1 for kw in suicidal_keywords if kw in text_lower)
    negative_score = sum(1 for w in negative_emotions if w in text_lower) / total_words
    self_ref_score = self_ref / total_words

    return {
        'hopelessness_score': round(hopelessness_score * 100, 2),
        'negative_emotion_score': round(negative_score * 100, 2),
        'self_referential_score': round(self_ref_score * 100, 2),
        'direct_ideation_count': suicidal_score,
        'text_length': len(words)
    }
