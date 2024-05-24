from transformers import pipeline


def classify_text(email):
    """
    Use facebook's BART model to classify an email into 'spam' or 'not spam'

    Args:
        email (str): The email to classify
    Returns:
        str: The classification of the email
    """

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-nmli")
    labels = ["spam", "not spam"]
    results = classifier(email, labels, hypothesis_template="hypothesis_template")
    return results[labels][0]


classify_text("Hi I am spam")  # spam
