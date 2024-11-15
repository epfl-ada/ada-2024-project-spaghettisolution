import os
import pandas as pd

def read_datas():
    raw_data_folder = './data/raw_data/' 
    characters_header_name = ['movie_wiki_id','freebase_movie_id','movie_release_date', 'character_name', 'birth', 'gender', 'height', 'ethnicity', 'name', 'release_age', 'freebase_character_map_id', 'freebase_character_id', 'freebase_actor_id']
    movies_header_name = ['wiki_id', 'freebase_id', 'name', 'release_date', 'revenue', 'runtime', 'languages', 'country', 'genres']
    plots_header_name =['wiki_id', 'plot']

    characters_df = pd.read_csv( raw_data_folder + 'character.metadata.tsv', sep = '\t', names = characters_header_name)
    movies_df = pd.read_csv( raw_data_folder + 'movie.metadata.tsv', sep = '\t', names= movies_header_name, index_col= 'wiki_id' )
    plots_df = pd.read_csv(raw_data_folder + 'plot_summaries.txt', sep= '\t', names = plots_header_name, index_col= 'wiki_id' )
    return characters_df, movies_df, plots_df