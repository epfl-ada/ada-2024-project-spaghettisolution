import pandas as pd

#clean movies dataframe 
def clean_movies_df(movies_df, plots_df):
    #remove useless columns
    movies_clean_df = movies_df.copy()
    movies_clean_df['freebase_id']
    movies_clean_df.drop('freebase_id', axis= 1, inplace= True)

    #remove all freebase code such as "/m/02h40lc"
    movies_clean_df['languages'] = movies_clean_df['languages'].str.replace('"/m/\w*": ', '', regex= True)
    movies_clean_df['country'] = movies_clean_df['country'].str.replace('"/m/\w*": ', '', regex= True)
    movies_clean_df['genres'] = movies_clean_df['genres'].str.replace('"/m/\w*": ', '', regex= True)

    #remove ""
    movies_clean_df['languages'] = movies_clean_df['languages'].str.replace('"', '', regex= False)
    movies_clean_df['country'] = movies_clean_df['country'].str.replace('"', '', regex= False)
    movies_clean_df['genres'] = movies_clean_df['genres'].str.replace('"', '', regex= False)


    #remove {}
    #before to remove all the curly brackets, let's fill with None items which have empty {}
    movies_clean_df['languages'] = movies_clean_df['languages'].replace('{}', None , regex= False)
    movies_clean_df['country'] = movies_clean_df['country'].replace('{}', None , regex= False)
    movies_clean_df['genres'] = movies_clean_df['genres'].replace('{}', None , regex= False)


    #now we can remove the brackets {}
    movies_clean_df['languages'] = movies_clean_df['languages'].str.replace('{', '', regex= False)
    movies_clean_df['languages'] = movies_clean_df['languages'].str.replace('}', '', regex= False)

    movies_clean_df['country'] = movies_clean_df['country'].str.replace('{', '', regex= False)
    movies_clean_df['country'] = movies_clean_df['country'].str.replace('}', '', regex= False)

    movies_clean_df['genres'] = movies_clean_df['genres'].str.replace('{', '', regex= False)
    movies_clean_df['genres'] = movies_clean_df['genres'].str.replace('}', '', regex= False)


    #remove Language in languages columns 
    movies_clean_df['languages'] = movies_clean_df['languages'].str.replace('Language', '', regex= False, case = False)

    #merge movies_df and plots_df
    movies_clean_df =movies_clean_df.merge(plots_df, on = 'wiki_id', how= 'outer') #from 81741 to 81840

    #save only th the year in released date column and convert it to int object
    movies_clean_df['release_date'] = movies_clean_df['release_date'].str.replace('-.*', '', regex= True)
    movies_clean_df['release_date'] = pd.to_numeric(movies_clean_df['release_date'], errors= 'coerce', downcast= 'integer') # it stil convert into float... 
    return movies_clean_df

# check for missing data in movies df
def missing_movies_data_check(movies_clean_df):
    for columns in movies_clean_df.columns:
        missing_data = movies_clean_df[columns].isna().sum()
        print(f"{missing_data} out of {len(movies_clean_df[columns])} movies have no {columns} data, which means {missing_data/len(movies_clean_df[columns])*100}% of the data is missing")


# clean characters dataframe
def clean_characters_df(characters_df):
    characters_clean_df = characters_df.copy()

    #remove useless columns
    characters_clean_df.drop('freebase_movie_id', axis= 1, inplace= True)
    characters_clean_df.drop('movie_release_date', axis= 1, inplace= True)
    characters_clean_df.drop('freebase_character_map_id', axis= 1, inplace= True)
    characters_clean_df.drop('freebase_character_id', axis= 1, inplace= True)
    return characters_clean_df

# check for missing data in movies df
def missing_characters_data_check(characters_clean_df):
    for columns in characters_clean_df.columns:
        missing_data = characters_clean_df[columns].isna().sum()
        print(f"{missing_data} out of {len(characters_clean_df[columns])} characters have no {columns} data, which means {missing_data/len(characters_clean_df[columns])*100}% of the data is missing")


# save clean version of dataframes 
def save_cleaned_datas(movies_clean_df, characters_clean_df):
    movies_clean_df.to_csv('./data/cleaned_data/movies_data.csv')
    characters_clean_df.to_csv('./data/cleaned_data/characters_data.csv')
