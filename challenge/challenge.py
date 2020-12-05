#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing dependencies

import json
import pandas as pd
import numpy as np
import re
import os 
import pathlib
from sqlalchemy import create_engine
import time
from config import db_password


# In[ ]:


# creating a function that takes all 3 data tables, loads it, and performs extraction onto a new table

new_file_dir = './'

kaggle_metadata_file = 'movies_metadata.csv'

ratings_file = 'ratings.csv'

wiki_json_file_load = 'wikipedia.movies.json'

alt_titles = {}

def auto_pipeline_etl(ratings_file, wiki_json_file_load, kaggle_metadata_file):

    meta_kaggle_movies_df = pd.read_csv(f'{new_file_dir}{kaggle_metadata_file}', low_memory=False)

    meta_kaggle_movies_df = meta_kaggle_movies_df[meta_kaggle_movies_df['adult'] == 'False'].drop('adult',axis='columns')

    meta_kaggle_movies_df['video'] = meta_kaggle_movies_df['video'] == 'True'

    meta_kaggle_movies_df['budget'] = meta_kaggle_movies_df['budget'].astype(int)

    meta_kaggle_movies_df['id'] = pd.to_numeric(meta_kaggle_movies_df['id'], errors='raise')

    meta_kaggle_movies_df['popularity'] = pd.to_numeric(meta_kaggle_movies_df['popularity'], errors='raise')

    meta_kaggle_movies_df['release_date'] = pd.to_datetime(meta_kaggle_movies_df['release_date'])

    # return (meta_kaggle_movies_df) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ratings_data = pd.read_csv(f'{new_file_dir}{ratings_file}')

    ratings_data['timestamp'] = pd.to_datetime(ratings_data['timestamp'], unit='s')

    rating_counts = ratings_data.groupby(['movieId','rating'], as_index=False).count()                     .rename({'userId':'count'}, axis=1)                     .pivot(index='movieId',columns='rating', values='count')

    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    # return (ratings_data) ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with open(f'{new_file_dir}{wiki_json_file_load}', mode = 'r') as file:
    
        wiki_movies_list = json.load(file)

        wiki_movies_df = pd.DataFrame(wiki_movies_list)

        wiki_movies_list = [movie for movie in wiki_movies_list if ('Director' in movie or 'Directed by' in movie) and 'imdb_link' in movie and 'No. of episodes' not in movie]

        wiki_movies_df = pd.DataFrame(wiki_movies_list)

        def clean_movies(yard):

            movie = dict(yard)

            # alt_titles = {}

            for key in ['Also known as','Arabic','Cantonese','Chinese','French',
            'Hangul','Hebrew','Hepburn','Japanese','Literally',
            'Mandarin','McCune–Reischauer','Original title','Polish',
            'Revised Romanization','Romanized','Russian',
            'Simplified','Traditional','Yiddish']:
    
                if key in movie:
    
                    alt_titles[key] = movie[key]
                    movie.pop(key)


            if len(alt_titles) > 0:

                movie['alt_titles'] = alt_titles
        
            def change_column_name(old_name, new_name):

                if old_name in movie:
        
                    movie[new_name] = movie.pop(old_name)
        
            change_column_name('Adaptation by', 'Writer(s)')
            change_column_name('Country of origin', 'Country')
            change_column_name('Directed by', 'Director')
            change_column_name('Distributed by', 'Distributor')
            change_column_name('Edited by', 'Editor(s)')
            change_column_name('Length', 'Running time')
            change_column_name('Original release', 'Release date')
            change_column_name('Music by', 'Composer(s)')
            change_column_name('Produced by', 'Producer(s)')
            change_column_name('Producer', 'Producer(s)')
            change_column_name('Productioncompanies ', 'Production company(s)')
            change_column_name('Productioncompany ', 'Production company(s)')
            change_column_name('Released', 'Release Date')
            change_column_name('Release Date', 'Release date')
            change_column_name('Screen story by', 'Writer(s)')
            change_column_name('Screenplay by', 'Writer(s)')
            change_column_name('Story by', 'Writer(s)')
            change_column_name('Theme music composer', 'Composer(s)')
            change_column_name('Written by', 'Writer(s)')
                
            return movie

        clean_movies = [clean_movies(movie) for movie in wiki_movies_list]

        wiki_movies_df = pd.DataFrame(clean_movies)

        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')

        wiki_movies_df.drop_duplicates(subset = 'imdb_id', inplace = True)

        selected_columns = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]

        wiki_movies_df = wiki_movies_df[selected_columns]

        wiki_movies_df = wiki_movies_df.drop('alt_titles',axis='columns')

        box_office_col = wiki_movies_df['Box office'].dropna()

        budget = wiki_movies_df['Budget'].dropna()

        form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
        
        form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

        def parse_dollars(s):

            if type(s) != str:
            
                return np.nan

            if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

                s = re.sub('\$|\s|[a-zA-Z]','', s)

                value = float(s) * 10**6

                return value

            elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

                s = re.sub('\$|\s|[a-zA-Z]','', s)

                value = float(s) * 10**9

                return value

            elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

                s = re.sub('\$|,','', s)

                value = float(s)

                return value

            else:
                return np.nan


        wiki_movies_df['box_office_col'] = box_office_col.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

        wiki_movies_df.drop('Box office', axis=1, inplace=True)

        budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x).str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

        matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
        
        matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
        
        budget[~matches_form_one & ~matches_form_two]

        budget = budget.str.replace(r'\[\d+\]\s*', '')
        
        budget[~matches_form_one & ~matches_form_two]

        wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

        wiki_movies_df.drop('Budget', axis=1, inplace=True)

        release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

        date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
        
        date_form_two = r'\d{4}.[01]\d.[123]\d'
        
        date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
        
        date_form_four = r'\d{4}'

        wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(
            f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


        running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

        running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

        running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

        wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

        wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # return wiki_movies_df ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    movies_df = pd.merge(wiki_movies_df, meta_kaggle_movies_df, on='imdb_id', suffixes=['_wiki','_kaggle'])

    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)
        
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')

    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')

    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office_col')

    movies_df.drop('video', axis=1, inplace=True)

    movies_df = movies_df.loc[:, [
        'imdb_id',
        'id',
        'title_kaggle',
        'original_title',
        'tagline',
        'belongs_to_collection',
        'url',
        'imdb_link',
        'runtime',
        'budget_kaggle',
        'revenue',
        'release_date_kaggle',
        'popularity',
        'vote_average',
        'vote_count',
        'genres',
        'original_language',
        'overview',
        'spoken_languages',
        'Country',
        'production_companies',
        'production_countries',
        'Distributor',
        'Producer(s)',
        'Director',
        'Starring',
        'Cinematography',
        'Editor(s)',
        'Writer(s)',
        'Composer(s)',
        'Based on']]

    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True) 

    # return movies_df ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

#   return movies_with_ratings_df.columns.tolist() ---------------------------------------------------------------------------------------------------------------------------------------  

    db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data_ssingh"

    engine = create_engine(db_string)
    
    rows_imported = 0

    start_time = time.time()

    for data in pd.read_csv(f'./ratings.csv', chunksize=1000000):
        
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end = '')

        data.to_sql(name='rating', con=engine, if_exists='append')
    
        rows_imported += len(data)
    
        print(f'Done. {time.time() - start_time} total seconds elapsed')

auto_pipeline_etl(ratings_file, wiki_json_file_load, kaggle_metadata_file)


# In[ ]:




