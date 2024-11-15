# Module containing the functions used for the text analysis

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
import json
import time


def import_summaries_data(filename: str) -> pd.DataFrame:
    """Function to import the summaries data

    Args:
        filename (str): name of file with summaries

    Returns:
        pd.DataFrame: dataframe where the index is the film id and has the summaries
    """
    raw_data_folder = "./data/raw_data/"
    summaries = pd.read_csv(
        raw_data_folder + filename, sep="\t", index_col=0, names=["index", "summary"]
    )
    return summaries


def tokenize_summaries(summaries: pd.DataFrame) -> list:
    """Function to produce tokens from a dataframe of summaries

    Args:
        summaries (pd.DataFrame): dataframe with column with summaries of movies

    Returns:
        list: tokens for text
    """
    nltk.download()
    all_summaries = summaries["summary"].str.cat(sep=" ")
    return word_tokenize(all_summaries)


def dispersion_plot(tokens: list, words: list) -> None:
    """Plot the dispersion plot for a given set of Tokens

    Args:
        tokens (list): Tokens
        words (list): List of words to plot
    """
    nltk.draw.dispersion.dispersion_plot(tokens, words)


def lexical_diversity(text: list) -> float:
    """Calculate the lexical diversity of the text

    Args:
        text (list): List of words

    Returns:
        float: Lexical diversity measure
    """
    return len(set(text)) / len(text)


def plot_most_common(tokens: list, start_range: int, end_range: int) -> None:
    """Plot the frequency of tokens from the xth most common to the yth most common

    Args:
        tokens (list): tokens
        start_range (int): start of range to plot
        end_range (int): end of range to plot
    """
    fdist1 = FreqDist(tokens)
    most_common_tokens = fdist1.most_common()
    most_common_tokens = most_common_tokens[start_range:end_range]
    tokens, counts = zip(*most_common_tokens)
    plt.figure(figsize=(10, 6))
    plt.bar(tokens, counts)
    plt.xticks(rotation=90)  # Rotate token labels for better readability
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title(f"Top {start_range}  to {end_range} Most Common Tokens")
    plt.tight_layout()
    plt.show()


def identify_personas(filename: str):
    """Function to identify the most common characters in a movie summary

    Args:
        filename (str): file with the summary
    """
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.5.7")
    with open(filename, "r") as file:
        sentence = file.read().strip()
    named_entities = nlp.ner(sentence)
    Coref = nlp.coref(sentence)
    nlp.close()
    filtered_named_entities = [
        entity for entity in named_entities if entity[1] == "PERSON"
    ]
    # print(filtered_named_entities)
    entity_counts = Counter(filtered_named_entities)
    sorted_entities = entity_counts.most_common()
    print(
        "Sorted named entities by count:",
        [(entity[0], count) for entity, count in sorted_entities],
    )


def extract_entities_and_adjectives(text: str) -> dict:
    """Function to identify the personas and the adjectives associated with them from a text

    Args:
        text (str): the text

    Returns:
        dict: personas and the associated adjectives
    """
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.5.7")
    # Annotate the text using CoreNLP
    annotated_text = nlp.annotate(
        text,
        properties={
            "annotators": "tokenize,ssplit,pos,lemma,ner",  # POS tagging, Lemmatization, NER
            "outputFormat": "json",
        },
    )

    try:
        annotated_json = json.loads(annotated_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing the JSON response: {e}")
        return {}
    entities = {}
    # Iterate over sentences to extract named entities and adjectives
    for sentence in annotated_json.get("sentences", []):
        sentence_adjectives = set()  # To hold adjectives in the sentence
        sentence_names = set()  # To hold entities in the sentence
        for word in sentence.get("tokens", []):
            word_text = word.get("word")
            pos_tag = word.get("pos")
            ner_tag = word.get("ner")
            # If the word is a named entity, add it to the sentence names
            if ner_tag == "PERSON":
                sentence_names.add(word_text)
            # If the word is an adjective, add it to sentence adjectives
            if pos_tag and pos_tag.startswith("JJ"):  # Adjective tags like JJ, JJR, JJS
                sentence_adjectives.add(word_text)

        # Now, associate adjectives with names in this sentence
        for name in sentence_names:
            if name not in entities:
                entities[name] = set()
            # Add all adjectives found in this sentence to each name
            entities[name].update(sentence_adjectives)
    nlp.close()

    entities_adjectives = entities
    if entities_adjectives:
        for entity, adj_list in entities_adjectives.items():
            print(f"Entity: {entity}, Adjectives: {', '.join(adj_list)}")
        else:
            print("No entities with adjectives found.")

    return entities


def extract_entities_and_verbs(text):
    """Function to identify the personas and the verbs associated with them in a text

    Args:
        text (str): text to analyse

    Returns:
        dict: personas and their associated verbs
    """
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.5.7")
    # Annotate the text using CoreNLP
    annotated_text = nlp.annotate(
        text,
        properties={
            "annotators": "tokenize,ssplit,pos,lemma,ner",  # POS tagging, Lemmatization, NER
            "outputFormat": "json",
        },
    )

    try:
        annotated_json = json.loads(annotated_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing the JSON response: {e}")
        return {}

    entities = {}

    # Iterate over sentences to extract named entities and verbs
    for sentence in annotated_json.get("sentences", []):
        sentence_verbs = set()  # To hold verbs in the sentence
        sentence_names = set()  # To hold entities in the sentence

        for word in sentence.get("tokens", []):
            word_text = word.get("word")
            pos_tag = word.get("pos")
            ner_tag = word.get("ner")

            # If the word is a named entity, add it to the sentence names
            if ner_tag == "PERSON":
                sentence_names.add(word_text)

            # If the word is a verb, add it to sentence verbs
            if pos_tag and pos_tag.startswith(
                "VB"
            ):  # Verb tags like VB, VBD, VBG, VBN, VBP, VBZ
                sentence_verbs.add(word_text)

        # Now, associate verbs with names in this sentence
        for name in sentence_names:
            if name not in entities:
                entities[name] = set()
            # Add all verbs found in this sentence to each name
            entities[name].update(sentence_verbs)
    entities_verbs = entities
    if entities_verbs:
        for entity, verb_list in entities_verbs.items():
            print(f"Entity: {entity}, Verbs: {', '.join(verb_list)}")
    else:
        print("No entities with verbs found.")
    nlp.close()
    return entities


def split_text_into_chunks(text, max_length=1000):
    """Function to split the text into chunks as there is a limit to the length of the text we can use with core NLP

    Args:
        text (str): text to cut
        max_length (int, optional): max length of each chunk of text. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    # Split text into sentences and group them into chunks of a specified size
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Add sentence to current chunk if it doesn't exceed max_length
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:  # Add any remaining text as the last chunk
        chunks.append(current_chunk.strip())

    return chunks


def retry_request(text, retries=3, delay=5):
    # Retry function in case of timeout
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.5.7")

    for attempt in range(retries):
        try:
            annotated_text = nlp.annotate(
                text,
                properties={
                    "annotators": "tokenize,ssplit,pos,ner,sentiment",
                    "outputFormat": "json",
                    "timeout": 30000,  # Timeout in milliseconds
                },
            )
            return json.loads(annotated_text)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                return None
    nlp.close()


def extract_entity_sentiments_intermediate(text):
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.5.7")
    # Annotate the text using CoreNLP
    annotated_json = retry_request(text)
    if annotated_json is None:
        print("Failed to process the text.")
        return {}, {}

    entities_sentiments = {}
    overall_sentiment_counts = {"Good": 0, "Neutral": 0, "Bad": 0}

    # Iterate over sentences
    for sentence in annotated_json.get("sentences", []):
        sentiment = sentence.get("sentiment")  # Sentiment for the sentence
        sentiment_category = None

        # Map sentiment to categories
        if sentiment in ["Verypositive", "Positive"]:
            sentiment_category = "Good"
        elif sentiment in ["Neutral"]:
            sentiment_category = "Neutral"
        elif sentiment in ["Negative", "Verynegative"]:
            sentiment_category = "Bad"

        if sentiment_category:
            overall_sentiment_counts[sentiment_category] += 1

        # Extract entities (e.g., PERSON) in the sentence
        sentence_entities = {
            token["word"]
            for token in sentence.get("tokens", [])
            if token.get("ner") == "PERSON"
        }

        # Associate sentiment with each entity
        for entity in sentence_entities:
            if entity not in entities_sentiments:
                entities_sentiments[entity] = {"Good": 0, "Neutral": 0, "Bad": 0}
            entities_sentiments[entity][sentiment_category] += 1
    nlp.close()

    return overall_sentiment_counts, entities_sentiments


def extract_entity_sentiment(text):

    # Split the large document into chunks
    chunks = split_text_into_chunks(text)

    # Initialize result containers
    overall_sentiment_percentages = {"Good": 0, "Neutral": 0, "Bad": 0}
    individual_sentiment_percentages = {}

    # Process each chunk separately
    for chunk in chunks:
        chunk_overall, chunk_individual = extract_entity_sentiments_intermediate(chunk)

        # Aggregate results
        for sentiment in overall_sentiment_percentages:
            overall_sentiment_percentages[sentiment] += chunk_overall.get(sentiment, 0)

        for entity, percentages in chunk_individual.items():
            if entity not in individual_sentiment_percentages:
                individual_sentiment_percentages[entity] = {
                    "Good": 0,
                    "Neutral": 0,
                    "Bad": 0,
                }
            for sentiment, count in percentages.items():
                individual_sentiment_percentages[entity][sentiment] += count

    # After processing all chunks, calculate final percentages
    total_sentences = sum(overall_sentiment_percentages.values())
    final_overall_percentages = {
        sentiment: (count / total_sentences * 100)
        for sentiment, count in overall_sentiment_percentages.items()
    }

    # Print final results
    print("Overall Sentiment Percentages:")
    for sentiment, percentage in final_overall_percentages.items():
        print(f"{sentiment}: {percentage:.2f}%")

    print("\nIndividual Sentiment Percentages:")
    for entity, percentages in individual_sentiment_percentages.items():
        print(f"Entity: {entity}")
        total_entity_sentences = sum(percentages.values())
        for sentiment, count in percentages.items():
            percentage = (
                (count / total_entity_sentences * 100)
                if total_entity_sentences > 0
                else 0
            )
            print(f"  {sentiment}: {percentage:.2f}%")
