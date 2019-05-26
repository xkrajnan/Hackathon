import nltk
import string


PUNCTABLE = str.maketrans({key: None for key in string.punctuation})


def get_sentences(original_string):
    return nltk.sent_tokenize(original_string)


def tokenize(original_string):
    tokens = nltk.word_tokenize(original_string)
    
    cleaned_up_tokens = [] 
    for token in tokens:
        cleaned_up = str(token.lower()).translate(PUNCTABLE)
        if cleaned_up:
            cleaned_up_tokens.append(cleaned_up)
        else:
            # token consisted only of special chars, keep
            cleaned_up_tokens.append(token)

    return cleaned_up_tokens

