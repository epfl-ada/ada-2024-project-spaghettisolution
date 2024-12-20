# Module containing the functions used for the text analysis

import numpy as np
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
import json
import time

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import ast
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


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
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
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
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
    annotated_text = nlp.annotate(
        text,
        properties={
            "annotators": "tokenize,ssplit,pos,lemma,ner",
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
        sentence_adjectives = set()
        sentence_names = set()
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

        for name in sentence_names:
            if name not in entities:
                entities[name] = set()
            entities[name].update(sentence_adjectives)
    nlp.close()

    entities_adjectives = entities
    if entities_adjectives:
        for entity, adj_list in entities_adjectives.items():
            print(f"Entity: {entity}, Adjectives: {', '.join(adj_list)}")

    return entities


def extract_entities_and_verbs(text):
    """Function to identify the personas and the verbs associated with them in a text

    Args:
        text (str): text to analyse

    Returns:
        dict: personas and their associated verbs
    """
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
    annotated_text = nlp.annotate(
        text,
        properties={
            "annotators": "tokenize,ssplit,pos,lemma,ner",
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
        sentence_verbs = set()
        sentence_names = set()

        for word in sentence.get("tokens", []):
            word_text = word.get("word")
            pos_tag = word.get("pos")
            ner_tag = word.get("ner")

            if ner_tag == "PERSON":
                sentence_names.add(word_text)

            if pos_tag and pos_tag.startswith(
                "VB"
            ):  # Verb tags like VB, VBD, VBG, VBN, VBP, VBZ
                sentence_verbs.add(word_text)

        for name in sentence_names:
            if name not in entities:
                entities[name] = set()
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

    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def retry_request(text, retries=3, delay=5):
    # Retry function in case of timeout
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")

    for attempt in range(retries):
        try:
            annotated_text = nlp.annotate(
                text,
                properties={
                    "annotators": "tokenize,ssplit,pos,ner,sentiment",
                    "outputFormat": "json",
                    "timeout": 30000,
                },
            )
            return json.loads(annotated_text)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
    nlp.close()


def extract_entity_sentiments_intermediate(text):
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
    annotated_json = retry_request(text)
    if annotated_json is None:
        print("Failed to process the text.")
        return {}, {}

    entities_sentiments = {}
    overall_sentiment_counts = {"Good": 0, "Neutral": 0, "Bad": 0}

    for sentence in annotated_json.get("sentences", []):
        sentiment = sentence.get("sentiment")
        sentiment_category = None

        if sentiment in ["Verypositive", "Positive"]:
            sentiment_category = "Good"
        elif sentiment in ["Neutral"]:
            sentiment_category = "Neutral"
        elif sentiment in ["Negative", "Verynegative"]:
            sentiment_category = "Bad"

        if sentiment_category:
            overall_sentiment_counts[sentiment_category] += 1

        sentence_entities = {
            token["word"]
            for token in sentence.get("tokens", [])
            if token.get("ner") == "PERSON"
        }

        for entity in sentence_entities:
            if entity not in entities_sentiments:
                entities_sentiments[entity] = {"Good": 0, "Neutral": 0, "Bad": 0}
            entities_sentiments[entity][sentiment_category] += 1
    nlp.close()

    return overall_sentiment_counts, entities_sentiments


def extract_entity_sentiment(text):

    chunks = split_text_into_chunks(text)
    overall_sentiment_percentages = {"Good": 0, "Neutral": 0, "Bad": 0}
    individual_sentiment_percentages = {}
    for chunk in chunks:
        chunk_overall, chunk_individual = extract_entity_sentiments_intermediate(chunk)

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

    total_sentences = sum(overall_sentiment_percentages.values())
    final_overall_percentages = {
        sentiment: (count / total_sentences * 100)
        for sentiment, count in overall_sentiment_percentages.items()
    }
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


def prepare_persona_data(genres_filter=None, country_filter=None, title=None):
    movies_df = pd.read_csv(f"data\cleaned_data\movies_data.csv")

    if genres_filter:
        if country_filter:
            new_movies_df = movies_df[
                (movies_df["genres"].str.contains(genres_filter, case=False, na=False))
                & (
                    movies_df["country"].str.contains(
                        country_filter, case=False, na=False
                    )
                )
            ]
        else:
            new_movies_df = movies_df[
                movies_df["genres"].str.contains(genres_filter, case=False, na=False)
            ]
    else:
        if country_filter:
            new_movies_df = movies_df[
                movies_df["country"].str.contains(country_filter, case=False, na=False)
            ]
        else:
            new_movies_df = movies_df

    new_movies_df["Persona"] = None
    new_movies_df.to_csv(f"data\cleaned_data\{title}_movies_data.csv", index=False)
    return new_movies_df


def extract_entities_and_related_words_with_coref(text):
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
    annotated_text = nlp.annotate(
        text,
        properties={
            "annotators": "tokenize,ssplit,pos,lemma,ner,depparse,coref",
            "outputFormat": "json",
        },
    )

    try:
        annotated_json = json.loads(annotated_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing the JSON response: {e}")
        return {}

    entities = {}

    coref_persons = {}
    coreferences = annotated_json.get("corefs", {})

    for sentence_index, sentence in enumerate(annotated_json.get("sentences", [])):
        for token in sentence.get("tokens", []):
            if token["ner"] == "PERSON":
                person_name = token["word"]
                if person_name not in coref_persons:
                    coref_persons[person_name] = person_name

    for coref_id, mentions in coreferences.items():
        main_person_name = None

        for mention in mentions:
            # adapt the index to 0-based for python
            sentence_index = mention["sentNum"] - 1
            token_start = mention["startIndex"] - 1
            token_end = mention["endIndex"] - 1
            tokens = annotated_json["sentences"][sentence_index]["tokens"][
                token_start:token_end
            ]

            person_tokens = [token for token in tokens if token["ner"] == "PERSON"]
            if person_tokens:
                proper_names = [
                    token["word"]
                    for token in person_tokens
                    if token["pos"] in ["NNP", "NNPS"]
                ]
                if proper_names:
                    main_person_name = " ".join(proper_names)
                    break

        if main_person_name:
            for mention in mentions:
                sentence_index = mention["sentNum"] - 1
                token_start = mention["startIndex"] - 1
                token_end = mention["endIndex"] - 1
                tokens = annotated_json["sentences"][sentence_index]["tokens"][
                    token_start:token_end
                ]
                mention_text = " ".join(token["word"] for token in tokens)

                coref_persons[mention_text] = main_person_name

    for sentence_index, sentence in enumerate(annotated_json.get("sentences", [])):
        # Use dependency parsing to find adjective, noun, and verb modifiers
        for dep in sentence.get("enhancedPlusPlusDependencies", []):
            gov_index = dep.get("governor") - 1
            dep_index = dep.get("dependent") - 1
            dep_relation = dep.get("dep")

            governor_word = sentence["tokens"][gov_index]["word"]
            dependent_word = sentence["tokens"][dep_index]["word"]
            governor_pos = sentence["tokens"][gov_index]["pos"]
            dependent_pos = sentence["tokens"][dep_index]["pos"]

            linked_entity = None

            # Check if the dependency is 'amod' a PERSON is mentionned or coref
            if dep_relation == "amod" and governor_word in coref_persons:
                linked_entity = coref_persons[governor_word]
                if linked_entity:
                    if linked_entity not in entities:
                        entities[linked_entity] = set()
                    entities[linked_entity].add(dependent_word)

            # Check if the relation is a verb or adjective linked to a PERSON entity
            elif (
                dep_relation in ["cop", "nsubj", "xcomp", "acl"]
                and dependent_word in coref_persons
            ):
                linked_entity = coref_persons[dependent_word]
                if linked_entity:
                    if linked_entity not in entities:
                        entities[linked_entity] = set()
                    entities[linked_entity].add(governor_word)

            # Check if the relation is a noun linked to a PERSON entity
            elif dep_relation in ["nsubj", "dobj", "iobj"] and (
                governor_pos.startswith("NN") or dependent_pos.startswith("NN")
            ):
                if governor_word in coref_persons:
                    linked_entity = coref_persons[governor_word]
                elif dependent_word in coref_persons:
                    linked_entity = coref_persons[dependent_word]

                if linked_entity:
                    if linked_entity not in entities:
                        entities[linked_entity] = set()
                    entities[linked_entity].add(governor_word)
                    entities[linked_entity].add(dependent_word)
    nlp.close()
    return entities


def retry_request(text, retries=3, delay=5):

    for attempt in range(retries):
        try:
            annotated_text = nlp.annotate(
                text,
                properties={
                    "annotators": "tokenize,ssplit,pos,ner,sentiment",
                    "outputFormat": "json",
                    "timeout": 30000,
                },
            )
            return json.loads(annotated_text)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None


def extract_entity_sentiments(text):
    print("Current position in the computer: ", os.getcwd())
    nlp = StanfordCoreNLP(r"..\stanford-corenlp-4.5.7\stanford-corenlp-4.5.7")
    annotated_json = retry_request(text)
    if annotated_json is None:
        print("Failed to process the text.")
        return {}, {}

    entities_sentiments = {}
    overall_sentiment_counts = {"Good": 0, "Neutral": 0, "Bad": 0}

    for sentence in annotated_json.get("sentences", []):
        sentiment = sentence.get("sentiment")
        sentiment_category = None

        if sentiment in ["Verypositive", "Positive"]:
            sentiment_category = "Good"
        elif sentiment in ["Neutral"]:
            sentiment_category = "Neutral"
        elif sentiment in ["Negative", "Verynegative"]:
            sentiment_category = "Bad"

        if sentiment_category:
            overall_sentiment_counts[sentiment_category] += 1

        sentence_entities = {
            token["word"]
            for token in sentence.get("tokens", [])
            if token.get("ner") == "PERSON"
        }

        for entity in sentence_entities:
            if entity not in entities_sentiments:
                entities_sentiments[entity] = {"Good": 0, "Neutral": 0, "Bad": 0}
            entities_sentiments[entity][sentiment_category] += 1
    nlp.close()
    return overall_sentiment_counts, entities_sentiments


def process_personas_movies(df):
    nlp = StanfordCoreNLP(r"../stanford-corenlp-4.5.7/stanford-corenlp-4.5.7")
    total_rows = len(df)

    for idx, (index, row) in enumerate(df.iterrows(), start=1):
        print(f"Processing {idx}/{total_rows}")
        df.at[index, "Persona"] = (
            extract_entities_and_related_words_with_coref(row["plot"])
            if pd.notnull(row["plot"])
            else {}
        )

    nlp.close()
    return df


def split_personas(dataframe):
    new_rows = []
    # Iterate over each row in the DataFrame
    for index, row in dataframe.iterrows():
        persona_dict = ast.literal_eval(
            row["Persona"]
        )  # take all of the personas in a film
        for persona, actions in persona_dict.items():
            new_row = row.drop("Persona")
            persona_series = pd.Series(
                {
                    "persona": persona,  # Add the persona name
                    "actions": actions,  # Add the actions
                }
            )
            new_row = pd.concat([new_row, persona_series])
            new_rows.append(new_row)

    # Convert the list of new rows into a new DataFrame
    dataframe = pd.DataFrame(new_rows)

    dataframe["actions_clean"] = dataframe["actions"].map(
        lambda x: re.sub(r"[{}':,.!?]", "", " ".join(x))
    )
    dataframe["actions_clean"] = dataframe["actions_clean"].map(lambda x: x.lower())
    dataframe = dataframe[dataframe["actions_clean"].str.split().str.len() > 1]
    return dataframe


def run_lda(dataframe, nb_clusters):
    # LDA function from sklearn takes a list with the "documents" as input -> transform each persona in the dataframe to an entry in the document list
    documents_list = dataframe["actions_clean"].tolist()
    words = [word for string in documents_list for word in string.split()]
    word_counts = Counter(words)
    unique_words = [word for word, count in word_counts.items() if count == 1]
    updated_documents = [
        " ".join([word for word in string.split() if word not in unique_words])
        for string in documents_list
    ]
    documents_list = [string for string in updated_documents if len(string.split()) > 1]

    # tokenize and vectorize the document list
    tokenizer = RegexpTokenizer(r"\w+")
    vectorizer = CountVectorizer()
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        tokenizer=tokenizer.tokenize,
    )
    train_data = tfidf.fit_transform(documents_list)

    # Create LDA object
    model = LatentDirichletAllocation(n_components=nb_clusters)

    # Fit and Transform SVD model on data
    lda_matrix = model.fit_transform(train_data)
    # Get Components
    lda_components = model.components_
    feature_names = tfidf.get_feature_names_out()
    top_words_per_topic = {}
    for topic_idx, topic in enumerate(lda_components):
        top_words = [
            feature_names[i] for i in topic.argsort()[:-11:-1]
        ]  # Top 10 words for each topic
        top_words_per_topic[topic_idx] = top_words

    # Get Document-Topic Distribution
    doc_topic_distribution = lda_matrix  # Each document's topic distribution

    # Return results
    return {
        "document_topic_distribution": doc_topic_distribution,
        "topic_word_distribution": top_words_per_topic,
        "lda_model": model,
        "tfidf": tfidf,
    }


def match_personas_topic(dataframe, tfidf, model):

    for index, row in dataframe.iterrows():
        wordlist = dataframe.loc[index, "actions_clean"]
        wordlist = wordlist.split()
        word = wordlist[0]
        word_index = tfidf.vocabulary_.get(word)

        if word_index is not None:
            # Get the topic-word distribution from the LDA model
            topic_word_distribution = model.components_  # Shape (n_topics, n_words)

            # Get the distribution of the word across all topics
            word_topic_distribution = topic_word_distribution[:, word_index]

            # Find the topic with the highest probability for the word
            best_topic = np.argmax(word_topic_distribution)
            dataframe.loc[index, "topic"] = best_topic

def plot_topic_evolution(start_topic, end_topic,topic_evolution):
    selected_topics = list(range(start_topic, end_topic + 1))
    selected_topic_evolution = topic_evolution[selected_topics]
    
    cumulative_topic_evolution = selected_topic_evolution.cumsum()
    
    max_appearance = cumulative_topic_evolution.max()
    cumulative_topic_evolution_normalized = cumulative_topic_evolution.div(max_appearance)
    
    cumulative_topic_evolution_normalized.plot(kind='line', figsize=(15, 8))
    plt.title(f'Normalized Cumulative Appearance of Assigned Topics {start_topic} to {end_topic} Over Time')
    plt.xlabel('Release Date')
    plt.ylabel('Normalized Cumulative Proportion of Topics')
    plt.legend(title='Assigned Topics', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.show()