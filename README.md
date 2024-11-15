# Readme

## Title
Consequences of historical events over the plots and personas present in the USA cinematographic industry and its relationship with tendencies observed in cinematographic industries throughout the world.

History in Hollywood: How trends in the plots and personas in American Cinema reflect historical event and relate with other film industries.

## Abstract
% Just like every other form of art, cinema is influenced by the hot topics of its time. This analysis is rendered feasible not only thanks to the compilation of quantifiable data, such as the age or gender of actors and actresses playing roles in pictures, but also by more modern techniques allowing us to classify the different characters described in summaries of said pictures. Our project's main goal is to highlight the link between cinema and history using these techniques, and see which characteristics of the movies are impacted by important social and political events. By examining the evolution of these aspects in available films, one could develop a method to assess the socio-political climate of a country at the time of a film's production.

Art is shaped by current events, and  cinema is no exception. This project uses an analysis of the characteristics of actors such as their gender and ethnicity along with modern machine learning techniques to classify the characters and events described in movie summaries from the CMU Movie Summary Corpus. The main goal is to highlight the links between cinema and major social, political and economic events of the second half of the XXth century. Through our analysis of the personas and plots we aim to develop a method to assess the socio-political climate through its representation in film. Our main focus is on the american film industry, however, we aim to contrast it with other film industries for select historical periods

## Research Questions

Throughout this study, we will examine the relationship between movies and the following key historical periods and events:

- Second World War (1939-1945)
- Indian independence (1947)
- Civil rights movements (1954 to 1968)
- Vietnam war (1955-1975)
- First and subsequent waves of  feminism
- Key Events of the Cold War (1947-1991):
    - Arms Race
    - Space Race
    - Proxy wars
    - Fall of the Berlin wall (1989)
- Rwandan Genocide (1994)
- Rise of internet and digital culture (late 90's)
- 9/11 (2001)
- Financial Crisis (2008)

We will focus on the following research questions:

1. How can important social movements influence the casting of actors and actresses?

2. How are both the genre of the movie and personas of characters impacted by the political climate at the time?

3. Are the reactions to social and political events seen in movies produced in the USA also observable in film industries around the world?

4. How do the archetypes of villains evolve to reflect contemporary fears (e.g., Nazis in WWII-era films, terrorists post-9/11, corporations post-financial crisis)?

5. Are conflicts in movies during the Cold War more focused on ideological struggles compared to personal or interpersonal conflicts in post-1991 films?

6. How do the dominant genres of movies shift in response to major historical or cultural events (e.g., war, economic crises)?

7. How does the portrayal of technology in movies change before and after the rise of the internet (1990s)?

## Additional datasets
We do not use any additional datasets. However, we do make use of the coreNLP python library, which uses external data.

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

**Linguistic Analysis of Summaries**  
- Use of CoreNLP to extract the adjectives and verbs linked to the different characters of each films. 
- The sentiment of each sentence with a character will be analyzed to help the differentiation.
- Use clustering on the different words extracted this way to determine different kind of personas.
%- Utilize CoreNLP to extract occurrences of characters and analyze their attributes, such as personas, names, and nationalities.  
- Visualize these characteristics over time to identify trends.  
- Further refine the linguistic methodology to ensure robust analysis of narrative elements.  

### Answering Research Questions
**Linking Trends to Historical Events**  
- Identify statistically significant changes in data fields over time and relate them to historical and social events.

**Analyzing Shifts in Narrative and Character Representation**  
- Observe shifts in plot themes, character archetypes, and casting trends to determine their alignment with societal movements and global events.   

### Final Steps
**Data Integration**  
- Merge the movie data, actor data, and CoreNLP linguistic analysis into a unified dataset for comprehensive examination.  

**Result Presentation**  
- Generate plots and statistical summaries to represent findings and provide clear answers to the research questions.  



## Team Organization \& Milestones
Nico Valsangiacomo: Writing the story, creation of the web page

Ismaël Gomes Almada Guillemin: Processing and analysis of movies and actors data

Zoë Evans: Processing and analysis of summaries data

Valentin Biselli: Processing and analysis of summaries data

Nils Antonovitch: Writing the story, creation of the web page

## Instructions for externals librairies 

Two external librairies are used in this project : StanfordcoreNLP and NLTK

To install the first one, please download the Java version from [Stanford CoreNLP website](https://stanfordnlp.github.io/CoreNLP/). The folder should be placed in parallel of the repository of the code. 

For NLTK:

1. Run the following command to install the library:
   ```bash
   pip install nltk
2. After installation, open a Python environment and run the following commands:
   ```python
   import nltk
   nltk.download()



## Questions for TAs

Are we permitted to use a pretrained LLM to reformulate the plot summaries into simpler sentences? This would improve the quality of our results using CoreNLP as it struggles to identify which adjective to associate with each persona in complex sentences. This would not significantly change our pipeline, but rather just introduce an extra step to simplify the texts.