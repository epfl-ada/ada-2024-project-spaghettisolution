# Readme

## Title

History in Hollywood: How trends in the plots, themes and personas in American Cinema reflect political, social and economic events.

## Website

The datastory is available at <https://nvalsa.github.io/>

## Abstract

Art is shaped by current events, and  cinema is no exception. In this project classical statistical analysis along with modern machine learning techniques are used to classify the characters and events described in the CMU Movie Summary Corpus. The main goal is to highlight the links between cinema and major social, political and economic events of the second half of the XXth century and the early XXIst century. Through our analysis of the personas and plots we aim to develop a method to assess the socio-political climate through its representation in film. Our main focus is on the american film industry.

## Research Questions

Throughout this study, we will examine the relationship between movies and the following key historical periods and events:



We will focus on the following research questions:

1. Are the main political social and economic events reflected in the themes explored in movies ?  

2. How do the dominant genres of movies shift in response to major historical or cultural events (e.g., war, economic crises)?

3. Are there certain types of personas that are more prelavent in certain genres of movies or certain time periods?

4. How are both the genre of the movie and personas of characters impacted by the political climate at the time?




## Additional datasets

We supplemented the dataset with a dataset containing plot summaries of movies from wikipedia. We chose to do this to increase the number of plot summaries available for analysis. We removed duplicates between the orignial dataset and the supplemental one.
The supplemental dataset is available at  <https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots>

We also used the CoreNLP library that is trained on external data

## Timeline

In this timeline, the weeks are counted from the beginning of the semester.

Week 9 (11th November to 17th November):
The datasets are cleaned and processed to handle both linguistic and non-linguistic data. Preliminary analyses are performed to identify patterns and insights, helping to refine and narrow the research questions for further investigation.

Week 10 (18th of November to 24th November):
The linguistic analysis of the summaries will continue, and data from the movies, actors and actresses, and the results of CoreNLP processing will be merged. This unified dataset will enable the exploration of relationships between narrative elements, casting choices, and historical or cultural contexts, providing insights into how linguistic patterns in movie summaries align with actor demographics, production details, and societal influences.

Week 11-12 (25th November to 8th December):
The analysis of the data will be completed, addressing the previously formulated research questions. Relevant plots will be generated to represent the findings. Afterward, the code in the repository will be cleaned up, and the process of writing the narrative for the project will begin, with the goal of creating the webpage to present the results.

Week 13-14 (9th of December to 20th December): 
The story will be finalized, and the webpage will be set up using Jekyll and GitHub Pages. The repository cleanup will be completed, including organizing the code into a main notebook and separate Python scripts. Additionally, the README.md will be updated to provide clear documentation of the project.

## Methods

### Data Preprocessing

**Cleaning the Data**  

- Remove unnecessary fields, such as Freebase identifiers, that are irrelevant to the analysis.  
- Use regular expressions to eliminate superfluous symbols and standardize fields.  
- Quantify missing data to identify potential biases and ensure robust interpretations.  

**Dataset Understanding**  

- Evaluate key metrics within the dataset to determine which fields are comparable.  
- Avoid comparisons between unbalanced portions of the dataset to maintain validity.  

### Data Analysis

**Field Isolation and Visualization**  

- Separate movies produced in the USA from those produced in other countries due to the former's dominance in the dataset.  
- Plot the distribution of each data field (e.g., genres, revenue, runtime) over time.  

**Semantic Analysis**
- Identify key words associated to each main historical event and plot their evolution over time. 
- Use statistical tests to verify the trends identified.

**Linguistic Analysis of Summaries**  

- Use of CoreNLP to extract the adjectives and verbs linked to the different characters of each films. 
- The sentiment of each sentence with a character will be analyzed to help the differentiation.
- Use clustering on the different words extracted this way to determine different kind of personas.
- Visualize the distribution of the personas in each cluster over time and in certain genres 




## Team Organization \& Milestones

Nico Valsangiacomo: Writing the story, creation of the web page

Ismaël Gomes Almada Guillemin: Processing and analysis of movies

Zoë Evans: Processing and analysis of summaries data

Valentin Biselli: Processing and analysis of summaries data

Nils Antonovitch: Writing the story, creation of the web page

## Instructions for external libraries and the dataset

The dataset is not included in the git repository as it is too large. It should be downloaded from <https://www.cs.cmu.edu/~ark/personas/> . It should be placed in the `data/raw_data` folder of the repository.

The supplementary dataset is not included in the git repository as it is too large. It should be downloaded from <https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots>. It should be placed in the `data/raw_data/kaggleData` folder of the repository.

Two external librairies are used in this project : StanfordcoreNLP and NLTK

To install StanfordcoreNLP, please download the Java version from [Stanford CoreNLP website](https://stanfordnlp.github.io/CoreNLP/). The folder should be placed in parallel of the repository of the code. 
You should have Java installed on your computer.

Then execute the following command:
   ```bash
   pip install stanford-corenlp

For NLTK:

1. Run the following command to install the library:
   ```bash
   pip install nltk
2. After installation, open a Python environment and run the following commands:
   ```python
   import nltk
   nltk.download()