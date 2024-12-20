import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_movies_distribution(movies_clean_df):
    oldest_year = int(movies_clean_df['release_date'].min())
    youngest_year = int(movies_clean_df['release_date'].max())
    year_difference = youngest_year - oldest_year

    plt.figure(figsize=(16, 6))
    sns.histplot(movies_clean_df['release_date'], binwidth=1)
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies')
    plt.title('Distribution of Movies Over Time')
    plt.show()
#distribution of number of film realized by country
def distributiion_per_country(movies_clean_df):
    #first we focus on film realized by one unique country and not collaborate with other
    dist_per_country1_df = movies_clean_df.copy(deep= True)

    #drop row which containts None in column country
    dist_per_country1_df.dropna(subset = ['country'], inplace=True)

    #drop row which containts no plots 
    dist_per_country1_df.dropna(subset= ['plot'], inplace= True) 

    # now remove all film that come from a collaboration (i.e contain "," )
    dist_per_country1_df = dist_per_country1_df[~ dist_per_country1_df['country'].str.contains(',', regex= False).values]

    # get nbr movies released by each country
    freq_per_country1_df = dist_per_country1_df['country'].value_counts()


    #do a cut off 
    freq_per_country1_df = freq_per_country1_df[freq_per_country1_df.values> 100]

    #plot result
    plt.figure(figsize=(15, 5))
    sns.barplot(x= freq_per_country1_df.index, y= freq_per_country1_df.values)
    plt.xticks(rotation = 90)
    plt.title(" Distribution of movies realesed per country ")
    plt.xlabel("Country")
    plt.ylabel(" # movies released")

    return dist_per_country1_df


# distribution: focus on USA, Soviet Union and Russia
def distributiion_per_sub_country(dist_per_country1_df):
    dist_per_sub_country1_df = dist_per_country1_df[(dist_per_country1_df['country'] == 'United States of America') | (dist_per_country1_df['country'] == 'Soviet Union') | (dist_per_country1_df['country'] == 'Russia')]

    freq_per_sub_country1_df = dist_per_sub_country1_df['country'].value_counts()
    sns.barplot(x= freq_per_sub_country1_df.index, y= freq_per_sub_country1_df.values)
    plt.title(" Distribution of movies realesed by USA, Soviet Union and Russia  ")
    plt.xlabel("Country")
    plt.ylabel(" # movies released")
    return dist_per_sub_country1_df

# filter df to remove None value in realese_date column and convert it to int
def filter_date(dist_per_sub_country1_df):
    dist_per_country1_date_df= dist_per_sub_country1_df.dropna(subset = ['release_date'],inplace=False)
    dist_per_country1_date_df['release_date'] = dist_per_country1_date_df['release_date'].astype(int, copy= True)
    return dist_per_country1_date_df

#plot distribution of movies released by USA over time
def plot_USA_date_distribution(dist_per_country1_date_df):
    # get only USA data
    dist_USA_date_df= dist_per_country1_date_df[dist_per_country1_date_df['country'] == 'United States of America']

    #plot
    sns.histplot(x ='release_date', data= dist_USA_date_df); 
    plt.xticks(rotation = 90)
    plt.title("USA")
    plt.xlabel("")
    plt.ylabel("")

#plot distribution of movies released by Soviet Union over time
def plot_Soviet_date_distribution(dist_per_country1_date_df):
    # get only Soviet Union  data
    dist_Soviet_date_df= dist_per_country1_date_df[dist_per_country1_date_df['country'] == 'Soviet Union']

    #plot
    sns.histplot(x ='release_date', data= dist_Soviet_date_df); 
    plt.xticks(rotation = 90)
    plt.title("Soviet Union")
    plt.xlabel('')
    plt.ylabel("")

#plot distribution of movies released by Russia over time
def plot_Russia_date_distribution(dist_per_country1_date_df):
    # get only Russia  data
    dist_Russia_date_df= dist_per_country1_date_df[dist_per_country1_date_df['country'] == 'Russia']

    #plot
    sns.histplot(x ='release_date', data= dist_Russia_date_df); 
    plt.xticks(rotation = 90)
    plt.title("Russia")
    plt.xlabel("")
    plt.ylabel("")


#same analysis but now per region

def add_region_column(dist_per_country1_df):
    #first lets define the different region with corresponding country in it
    north_america_list = ["United States of America", "Canada", "Mexico", "Puerto Rico", "Bahamas"]
    south_america_list = ["Argentina", "Brazil", "Chile", "Venezuela", "Colombia", "Uruguay", "Bolivia", "Peru", "Costa Rica", "Cuba", "Haiti", "Jamaica"]
    west_europe = ["United Kingdom", "France", "Italy", "Spain", "Netherlands", "Germany", "West Germany", "Austria", "Belgium", "Switzerland", "Ireland", "Portugal", "Luxembourg", "Malta", "Weimar Republic", "England", "Scotland", "Wales", "Kingdom of Great Britain", "Nazi Germany"]
    east_central_europe =["Soviet Union", "Czechoslovakia", "German Democratic Republic", "Poland", "Hungary", "Yugoslavia", "Russia", "Czech Republic", "Croatia", "Romania", "Bulgaria", "Albania", "Estonia", "Georgia", "Slovenia", "Ukraine", "Serbia", "Republic of Macedonia", "Armenia", "Georgian SSR", "Serbia and Montenegro", "Lithuania", "Slovakia", "Azerbaijan", "Federal Republic of Yugoslavia", "Socialist Federal Republic of Yugoslavia", "Uzbek SSR", "Uzbekistan", "Soviet occupation zone", "Crime", "Bosnia and Herzegovina", "Greece"]
    scandinavia_north_europe = ["Sweden", "Denmark", "Norway", "Finland", "Iceland", "Northern Ireland"]
    africa = ["Nigeria", "South Africa", "Egypt", "Morocco", "Tunisia", "Senegal", "Ethiopia", "Burkina Faso", "Democratic Republic of the Congo", "Cameroon", "Mali", "Algeria", "Guinea-Bissau", "Kenya", "Libya", "Zimbabwe"]
    middle_east =["Israel", "Lebanon", "Jordan", "Iraq", "Iran", "United Arab Emirates", "Palestinian territories", "Kuwait", "Bahrain", "Cyprus", "Turkey"]
    east_asia = ["Japan", "China", "Hong Kong", "South Korea", "Korea", "Taiwan", "Mongolia", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Cambodia", "Singapore", "Philippines", "Sri Lanka", "Bhutan", "Burma"]
    south_asia_india= ["India", "Pakistan", "Bangladesh", "Nepal", "Afghanistan"]
    oceania = ["Australia", "New Zealand"]

    #then define a function in order to sort country in corresponding region
    def choice_region(country):
        if country in north_america_list:
            return 'North America'
        elif country in south_america_list:
            return 'South America'
        elif country in west_europe:
            return 'West Europe'
        elif country in east_central_europe:
            return 'East/Central Europe'
        elif country in scandinavia_north_europe:
            return 'Scandinavia/North Europe'
        elif country in africa:
            return 'Africa'
        elif country in middle_east:
            return 'Middle East'
        elif country in east_asia:
            return 'East Asia'
        elif country in south_asia_india:
            return 'South Asia/India'
        elif country in oceania:
            return 'Ociana'
        else: 
            return None
        
    #finally add new column to dataset with "region" as label fill the new column
    dist_per_region_df = dist_per_country1_df.assign(region= lambda x: x['country'].apply(choice_region)) #only one got None (Malayalam Language)
    return dist_per_region_df

#distribution of number of film realized by region
def plot_distribution_per_region(dist_per_region_df):
    # get nbr movies released by each region
    freq_per_region_df = dist_per_region_df['region'].value_counts()

    #plot result
    plt.figure(figsize=(15, 5))
    sns.barplot(x= freq_per_region_df.index, y= freq_per_region_df.values)
    plt.xticks(rotation = 90)
    plt.title(" Distribution of movies realesed per region")
    plt.xlabel("Region")
    plt.ylabel(" # Movies Released")

#North America : distribution of film realized over time
def north_america_plot(dist_per_region_date_df):
    dist_north_america_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'North America']

    #plot
    sns.histplot(x ='release_date', data= dist_north_america_date_df); 
    plt.xticks(rotation = 90)
    plt.title("North America")
    plt.xlabel("")
    plt.ylabel("")

#West Europe: distribution of film realized over time
def west_europe_plot(dist_per_region_date_df):
    dist_west_europe_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'West Europe']

    #plot
    sns.histplot(x ='release_date', data= dist_west_europe_date_df); 
    plt.xticks(rotation = 90)
    plt.title("West Europe ")
    plt.xlabel("")
    plt.ylabel("")

#South Asia/India: distribution of film realized over time
def south_asia_india_plot(dist_per_region_date_df):
    dist_south_asia_india_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'South Asia/India']

    #plot
    sns.histplot(x ='release_date', data= dist_south_asia_india_date_df); 
    plt.xticks(rotation = 90)
    plt.title("South Asia/India")
    plt.xlabel("")
    plt.ylabel("")

#East Asia: distribution of film realized over time
def east_asia_plot(dist_per_region_date_df):
    dist_east_asia_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'East Asia']

    #plot
    sns.histplot(x ='release_date', data= dist_east_asia_date_df); 
    plt.xticks(rotation = 90)
    plt.title("East Asia")
    plt.xlabel("")
    plt.ylabel("")

# East/Central Europe: distribution of film realized over time
def east_central_europe_plot(dist_per_region_date_df):
    dist_east_central_europe_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'East/Central Europe']

    #plot
    sns.histplot(x ='release_date', data= dist_east_central_europe_date_df); 
    plt.xticks(rotation = 90)
    plt.title("East/Central Europe")
    plt.xlabel("")
    plt.ylabel("")

# South America: distribution of film realized over time 
def south_america_plot(dist_per_region_date_df):
    dist_south_america_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'South America']

    #plot
    sns.histplot(x ='release_date', data= dist_south_america_date_df); 
    plt.xticks(rotation = 90)
    plt.title("South America")
    plt.xlabel("")
    plt.ylabel("")

# Scandinavia/North Europe: distribution of film realized over time 
def scandinave_north_europe_plot(dist_per_region_date_df):
    dist_scandinave_north_europe_date_df= dist_per_region_date_df[dist_per_region_date_df['region'] == 'Scandinavia/North Europe']

    #plot
    sns.histplot(x ='release_date', data= dist_scandinave_north_europe_date_df); 
    plt.xticks(rotation = 90)
    plt.title("Scandinave/North Europe")
    plt.xlabel("")
    plt.ylabel("")

# filter to obtain only film that come from a collaboration between countries 
def filter_collaboration(movies_clean_df):
    collaboration_movies_df = movies_clean_df.copy(deep= True)

    # remove movies that have no country entry
    collaboration_movies_df.dropna(subset =['country'], inplace = True)

    # drop movies that have no plot
    #collaboration_movies_df.dropna(subset= ['plot'], inplace= True) 

    # transform the entries of columns 'country' into a List
    collaboration_movies_df['country']= collaboration_movies_df['country'].str.split(', ')

    # transform the Lists [] into Set {} since we don't care of the order (relevent for value.count())
    collaboration_movies_df['country']= collaboration_movies_df['country'].apply(lambda x: set(x))


    # count number of country per movie
    collaboration_movies_df['country_count'] = collaboration_movies_df['country'].apply(lambda x: len(x))

    # Filter for movies that are associated with more than one country
    collaboration_movies_df = collaboration_movies_df[collaboration_movies_df['country_count'] > 1]
    print(f'Number of movies made by more than one country: {len(collaboration_movies_df)}')
    return collaboration_movies_df

# filter to only get collaboration with USA and plot 
def filter_collaboration_USA(collaboration_movies_df, cut_off =5):
    collaborationUSA_movies_df = collaboration_movies_df.copy(deep=True) #carefull, it doesn't really copy object in df...
    #keep only collaboration with USA
    collaborationUSA_movies_df= collaborationUSA_movies_df[collaborationUSA_movies_df['country'].apply(lambda x: 'United States of America' in x)]

    #remove USA in the set and change column name
    collaborationUSA_movies_df.rename(columns= {'country' : 'collaboration'}, inplace= True)
    collaborationUSA_movies_df['collaboration'].apply(lambda x: x.remove('United States of America'))

    #change country_count into collaboration_count and remove 1
    collaborationUSA_movies_df.rename(columns= {'country_count': 'collaboration_count'}, inplace= True)
    collaborationUSA_movies_df['collaboration_count'] = collaborationUSA_movies_df['collaboration_count'] - 1

    #explode column  collaboration
    collaborationUSA_movies_df = collaborationUSA_movies_df.explode('collaboration')

    #distribution of collaboration with US
    dist_collab_USA_df= collaborationUSA_movies_df['collaboration'].value_counts()

    #do a cut off 
    dist_collab_USA_df = dist_collab_USA_df[dist_collab_USA_df.values> cut_off]

    #plot 
    plt.figure(figsize=(15, 5))
    sns.barplot(x= dist_collab_USA_df.index, y= dist_collab_USA_df.values)
    plt.xticks(rotation = 90)
    plt.title("Main collaborators with USA")
    plt.xlabel("Country")
    plt.ylabel(" # movies released")
    return collaborationUSA_movies_df

# get distribution of collaboration with USA over time
def collab_USA_over_time(collaborationUSA_movies_df):
    # remove movies that have no release date and convert date into int
    collabUSA_movies_date_df= collaborationUSA_movies_df[~ collaborationUSA_movies_df['release_date'].isna()]
    collabUSA_movies_date_df['release_date'] = collabUSA_movies_date_df['release_date'].astype(int, copy= True)


    # get distribution movies over time and by collaboration 
    collab_date_df= pd.crosstab(collabUSA_movies_date_df['collaboration'], collabUSA_movies_date_df['release_date'])

    # dates go from 1910 to 2013, but some dates are skipped, lets make the date continious (in year)
    #print(collab_date_df.columns)
    #create a similar df but with continous dates and fill this table with NaN
    continous_dates = np.arange(1910, 2014)
    cont_collab_date_df = pd.DataFrame( index= collab_date_df.index, columns= continous_dates )
    cont_collab_date_df.columns.name = 'release_date'

    #combine this two dataframes and replace NaN by zero
    collab_date_df = collab_date_df.combine_first(cont_collab_date_df)
    collab_date_df.fillna(0, inplace= True)
    collab_date_df = collab_date_df.astype(int)
    return collab_date_df


# merge dates to see better the results and consider only some country
country_list = ['Canada', 'China', 'England', 'France', 'German Democratic Republic', 'Germany', 'Hong Kong', 'India', 'Iraq', 'Japan', 'Kingdom of Great Britain', 'Russia', 'Soviet Union', 'United Kingdom', 'Weimar Republic', 'West Germany']

def merge_dates_column(collab_date_df, time_bin = 10, country_list = country_list):
    # we will fill this table dates per dates 
    merge_collab_date_df = pd.DataFrame( index= collab_date_df.index )
    merge_collab_date_df.columns.name = 'release_date'


    # combine dates with a bin time
    nbr_dates= len(collab_date_df.columns)
    date_index = 0
    begin_date_index = date_index
    if nbr_dates%time_bin == 0:
        nbr_bins = nbr_dates//time_bin -1
    else:
        nbr_bins = nbr_dates//time_bin
    for i in range(nbr_bins): #iteration entre bin
        current_sum = 0
        for j in range(time_bin): #iteration dans le bin
            # perform the sum within the bin
            current_sum += collab_date_df.iloc[:,date_index]
            date_index += 1
        #fill the table 
        end_date_index = date_index -1
        column_name = f'[{collab_date_df.columns[begin_date_index]} - {collab_date_df.columns[end_date_index]}]'
        merge_collab_date_df[column_name] = current_sum.values

        #actualise index
        begin_date_index = date_index

    #for the last date 
    current_sum = 0
    if nbr_dates%time_bin == 0:
        nbr_dates_last_bin = time_bin
    else:
        nbr_dates_last_bin = nbr_dates%time_bin
    for i in range(nbr_dates_last_bin):
        current_sum += collab_date_df.iloc[:,date_index]
        date_index += 1

    #update last column 
    end_date_index = date_index -1 
    column_name = f'[{collab_date_df.columns[begin_date_index]} - {collab_date_df.columns[end_date_index]}]'
    merge_collab_date_df[column_name] = current_sum.values
    if isinstance(country_list, list):
        return merge_collab_date_df.loc[country_list, :]
    else:
        return merge_collab_date_df

# build heatmap to see better the results by normalizing also each row
def normelize_collab_USA_heatmap(merge_collab_date_df):
    # we are not only interested in absolut value but in the variation of number of movies produced also, so let's just normalize each row
    normelize_merge_collab_date_df = merge_collab_date_df.div(merge_collab_date_df.sum(axis=0), axis=1)
    plt.figure(figsize=(7, 7))
    sns.heatmap(normelize_merge_collab_date_df, fmt= '.0f', annot = merge_collab_date_df, cbar_kws={'label': 'Normalized # Movies per Country'})
    plt.title("Collaborators with USA : Distribution of Movies per Released Date")
    plt.xlabel("Released Date")
    plt.xticks(rotation = 90)
    plt.ylabel("Collaborators")

def plot_heat_map( table_df, title, ylabel, xlabel = 'Released Date', rotation = 90, normelize_mapping = True, cbar_label = 'Normalized # Movies', figsize=(7, 7)):
    if normelize_mapping:
        mapping_df = table_df.div(table_df.sum(axis=0), axis=1)
    else:
        mapping_df = table_df

    plt.figure(figsize=figsize)
    sns.heatmap(mapping_df, fmt= '.0f', annot = table_df, cbar_kws={'label': cbar_label})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation = rotation)
    plt.ylabel(ylabel)

def plot_genres_heat_map(dominent_genre_date_df):
        genre_date_cross_df = pd.crosstab(dominent_genre_date_df['genres'], dominent_genre_date_df['release_date']) # time is already continious

        # regroup released_date to make the table more concise
        merge_genre_date_df= merge_dates_column(genre_date_cross_df, time_bin=10, country_list= None)
        

        plot_heat_map(table_df= merge_genre_date_df,
                      title= 'USA movies: Distribution of mains genres per Released Date',
                      ylabel= 'Main Genres',
                      figsize= (8.5,7))
        
def cross_tab_cont_time(df, column1, column2, filtered_column = None, filter= None):
    
    # get distribution movies over time and by collaboration 
    cross_df= pd.crosstab(df[column1], df[column2])

    # lets make the date continious (in year)
    discont_date = cross_df.columns
    #create a similar df but with continous dates and fill this table with NaN
    continous_dates = np.arange(discont_date[0], discont_date[-1] + 1)
    cont_cross_df = pd.DataFrame( index= cross_df.index, columns= continous_dates )
    cont_cross_df.columns.name = 'release_date'

    #combine this two dataframes and replace NaN by zero
    cross_df = cross_df.combine_first(cont_cross_df)
    cross_df.fillna(0, inplace= True)
    cross_df = cross_df.astype(int)
    
    return cross_df

def filter_date(df):
    date_df= df[~ df['release_date'].isna()]
    date_df['release_date'] =date_df['release_date'].astype(int)
    return date_df



def plot_spec_genres_heat_map(df):
    # make a cross table 
    cross_df = cross_tab_cont_time(df, 'genres', 'release_date')

    # regroup released_date to make the table more concise
    merge_cross_df= merge_dates_column(cross_df, time_bin=10, country_list= None)


    plot_heat_map(table_df= merge_cross_df,
                      title= 'USA movies: Distribution of Specific Genres per Released Date',
                      ylabel= 'Main Genres',
                      figsize= (8.5,7))
    

# Action: distribution of film realized over time 
def hist_sub_plot(df, x, filtered_column, filter, title = None, weights = None):
    if isinstance(filter, str):
        filter_df= df[df[filtered_column] == filter]
        title = filter
    elif isinstance(filter, list):
        filter_df =  list_filter(df, filtered_column, filter)
        title = title

    #plot
    sns.histplot(x = x, data= filter_df, weights = weights, binwidth =5 ); 
    plt.xticks(rotation = 90)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    # # del
    # plt.xlim(1900, 2021)

def hist_dom_genres_plots(df, normelized = False, dict_weight = None):
    weight_label = None
    title_normelized = ''

    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '
        #  ########## mettre la fonction add_weitgh et juste appelé les weight après dasn hist_subplot
        

    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized + 'Distribution of USA movies realesed per dominent genres over time', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized +'# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # Action: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Action', weights = weight_label) 


    plt.subplot(2,3,2)
    # Black and white: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Black-and-white', weights = weight_label ) 

    plt.subplot(2,3,3)
    # World Cinmea: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Silent film', weights = weight_label ) 

    plt.subplot(2,3,4)
    #Drama: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Drama', weights = weight_label ) 

    plt.subplot(2,3,5)
    # Crime Fiction: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Crime Fiction', weights = weight_label ) 


    plt.subplot(2,3,6)
    # Short Film: distribution of film realized over time 
    hist_sub_plot(df,'release_date', 'genres', 'Short Film', weights = weight_label ) 



def hist_spec_genres_plots(df, normelized = False, dict_weight = None):
    weight_label = None
    title_normelized = ''

    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '
    
    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized + 'Distribution of USA movies realesed per specific genres over time', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized +'# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # 'Anti-war: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Anti-war', weights = weight_label ) 


    plt.subplot(2,3,2)
    # Documentary: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Documentary', weights = weight_label ) 

    plt.subplot(2,3,3)
    # Political drama: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Political drama', weights = weight_label ) 

    plt.subplot(2,3,4)
    #Political cinema: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Political cinema', weights = weight_label ) 

    plt.subplot(2,3,5)
    # Political satire: distribution of film realized over time
    hist_sub_plot(df,'release_date', 'genres', 'Political satire', weights = weight_label ) 


    plt.subplot(2,3,6)
    # Political thriller: distribution of film realized over time 
    hist_sub_plot(df,'release_date', 'genres', 'Political thriller', weights = weight_label ) 


def hist_gen_event_date_plots(df, normelized = False, dict_weight = None):
    weight_label = None
    title_normelized = ''

    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '

    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized +'Movie Distribution by Release Date and Country with War/Politics Themes', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized +'# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # Germany: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'Germany', weights = weight_label )

    plt.subplot(2,3,2)
    # United States of America: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'United States of America', weights = weight_label )

    plt.subplot(2,3,3)
    # France: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'France', weights = weight_label )

    plt.subplot(2,3,4)
    # United Kingdom: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'United Kingdom', weights = weight_label ) 

    plt.subplot(2,3,5)
    # Italie: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'Italy', weights = weight_label )


    plt.subplot(2,3,6)
    # Russia: distribution of film realized over time with War/Plotics Themes
    hist_sub_plot(df, 'release_date', 'country', 'Russia', weights = weight_label )


def plot_gen_event_date_country_heatmap(df):
    # make cross table 
    cross_df =cross_tab_cont_time(df, 'country', 'release_date')

    # regroup released_date to make the table more concise
    merge_cross_df= merge_dates_column(cross_df, time_bin=10, country_list= ['Germany', 'United States of America','Italy', 'United Kingdom', 'German Democratic Republic', 'Russia',  'Soviet Union', 'West Germany', 'Weimar Republic' ] )
    plot_heat_map(table_df= merge_cross_df,
                      title= 'Movie Distribution by Release Date and Country with War/Politics Themes',
                      ylabel= 'Country',
                      figsize= (8.5,7))
    

def list_filter(df, filtered_column, list_items):
    # transform this list into a regular expression
    list_re = '|'.join(list_items)

    # filter dataframe to obtain only movies that have one of this word in the plot 
    filtered_df = df[df[filtered_column].str.contains(list_re, case= False, regex= True)]
    return filtered_df

# def ww2_filter(df):
#     # considler only movies related to ww2
#     ww2_df = list_filter(df, ['world war II', 'world war 2', 'nazis', 'allied forces', 'axis power', 'Holocaust', 'gestapo', 'pearl harbor', 'Concentration camps', 'third Reich', 'hitler'])
#     return ww2_df


def hist_event_USA_plots(df, normelized = False, dict_weight = None):
    # filter to obtain only movies with USA
    country_df = df.dropna(subset= ['country'], inplace = False)
    USA_df= country_df[country_df['country'].apply(lambda x: 'United States of America' in x)]

    weight_label = None
    title_normelized = ''
    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(USA_df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '

    
    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized+'USA Movie Distribution by Release Date with Different Topics', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized+'# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # 'ww2: distribution of film realized over time
    hist_sub_plot(USA_df,'release_date', 'plot', ['world war II', 'world war 2', 'nazis', 'allied forces', 'axis power', 'Holocaust', 'gestapo', 'pearl harbor', 'Concentration camp', 'third Reich', 'hitler'], 'world War II', weights = weight_label) 

    plt.subplot(2,3,2)
    # Civil rights movements : distribution of film realized over time
    hist_sub_plot(USA_df,'release_date', 'plot', ['Malcom X', 'Rosa Parks', 'Martin Luther', 'Stokely Carmichael', 'March on Washington', 'Montgomery Bus ', 'Birmingham Campaign',  'Civil Rights','Voting Rights Act of 1965', 'NAACP', 'SCLC', 'National Association for the Advancement of Colored People', 'Southern Christian Leadership Conference', 'black panthers', 'Segregation', 'Racial equality', 'Human rights', 'Jim Crow laws'], 'Civil Rights Movements', weights = weight_label)

    plt.subplot(2,3,3)
    #Vietnam war: distribution of film realized over time
    hist_sub_plot(USA_df,'release_date', 'plot', ['Vietnam War', 'Vietnam', 'Viet', 'Cong', 'Vietnamization', 'fall off Saigon', 'Mekong Delta'], 'Vietnam War', weights = weight_label)

    plt.subplot(2,3,4)
    #Cold war: distribution of film realized over time
    hist_sub_plot(USA_df,'release_date', 'plot',['Cold war', 'Iron Curtain', 'Arms Race', 'Space race', 'proxy wars', 'Berlin wall'], 'Cold War', weights = weight_label)

    plt.subplot(2,3,5)
    # Financial Crisis (2008): distribution of film realized over time
    hist_sub_plot(USA_df,'release_date', 'plot', ["Financial Crisis 2008" "Subprime Mortgages", "Lehman Brothers", "Mortgage-Backed Securities","MBS", "Toxic Assets", "Bailout", "Financial Meltdown", "Foreclosure", "Bankruptcy", "Economic Recession"], 'Financial Crisis (2008)', weights = weight_label)


    plt.subplot(2,3,6)
    # 11/09/01 : distribution of film realized over time 
    hist_sub_plot(USA_df,'release_date', 'plot', ['twin towers', 'world trade center', 'al-qaeda', 'terrorism', 'September 11', 'flight 93'], 'September 11', weights = weight_label)

def filter_per_country(df):
    country_df = df.copy(deep=True)
    country_df.dropna(subset= ['country'], inplace= True)
    country_df['country'] = country_df['country'].str.split(', ')
    country_df = country_df.explode('country')
    return country_df

def filter_per_plot_date(df):
    plot_date_df = df.copy(deep= True)
    plot_date_df.dropna(subset= ['plot'], inplace= True)
    plot_date_df.dropna(subset= ['release_date'], inplace= True)
    plot_date_df['release_date'] = plot_date_df['release_date'].astype(int, copy= True)
    return plot_date_df


def hist_gen_event_date_plot(df, normelized = False, dict_weight = None):
    weight_label = None
    title_normelized = ''

    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '

    general_event_list = ["Political", "Anti-war", 'politic', 'war', 'revolution', 'Propaganda', 'Ideology', 'Military']

    # filter dataframe to obtain only movies that have one of this word in the plot 
    gen_event_date_df = list_filter(df, 'plot',general_event_list)


    # distribution over time
    plt.figure(figsize=(4, 4))
    sns.histplot(x = 'release_date', data= gen_event_date_df, weights= weight_label, binwidth=5); 
    plt.xticks(rotation = 90)
    plt.title(title_normelized+ 'Distribution of Movies per release date containing words relative to war and politic in plot')
    plt.xlabel("Released Date")
    plt.ylabel(title_normelized + "# Movie Released Date")

    return gen_event_date_df



def dist_spec_genre_plot(genre_df):
    
    specific_genre = ['Anti-war','Documentary', 'Political drama','Political cinema', 'Political thriller', 'Political satire', 'Political Documetary', 'Dystopia']
    specific_genre_df = genre_df[genre_df['genres'].isin(specific_genre)]

    # plot distribution 
    dist_specific_genre = specific_genre_df['genres'].value_counts()
    
    plt.figure(figsize=(15, 5))
    sns.barplot(x= dist_specific_genre.index, y= dist_specific_genre.values)
    plt.xticks(rotation = 90)
    plt.title("Specific Genres in USA Movies")
    plt.xlabel("Genres")
    plt.ylabel(" # movies released")
    return specific_genre_df

def dist_USA_dom_genre_plot(movies_clean_df, threshold =10):
    genre_df = movies_clean_df.copy(deep= True)

    # drop na value in columns genres 
    genre_df.dropna(subset= ['genres'], inplace = True)

    #remove film with no country
    country_genre_df = genre_df.dropna(subset= ['country'], inplace = False)
    # keep only movies that are made by USA
    USA_genre_df= country_genre_df[country_genre_df['country'].apply(lambda x: 'United States of America' in x)]

    # transform unique str into list of strings in column 'genres'
    USA_genre_df['genres'] = USA_genre_df['genres'].str.split(', ')

    # explode genre column 
    USA_genre_df = USA_genre_df.explode('genres')

    # there is a lot of different genre, let's consider only the most dominents 
    threshold = threshold
    USA_dominent_genre = USA_genre_df['genres'].value_counts().index[:threshold]
    USA_dominent_genre_df = USA_genre_df[USA_genre_df['genres'].isin(USA_dominent_genre)]

    # plot distribution 
    dist_USA_dominent_genre = USA_dominent_genre_df['genres'].value_counts()
    
    plt.figure(figsize=(15, 5))
    sns.barplot(x= dist_USA_dominent_genre.index, y= dist_USA_dominent_genre.values)
    plt.xticks(rotation = 90)
    plt.title("Main genres USA Movies")
    plt.xlabel("genres")
    plt.ylabel(" # movies released")

    return USA_dominent_genre_df, USA_genre_df

def hist_economic_event_date_plots(event_date_df, normelized = False, dict_weight = None):

    # filter to take only movies that containes econmics related words in the plot
    economic_event_list = ["Business", "Finance", "Investing",  'poverty', 'Economic collapse', 'Unemployment', 'jobless','Inflation','Stock market','Bankruptcy','Debt']
        # filter dataframe to obtain only movies that have one of this words in the plot 
    economic_event_date_df = list_filter(event_date_df, 'plot',economic_event_list)


    # remove None and explode column country 
    economic_country_event_date_df = economic_event_date_df[~ economic_event_date_df['country'].isna()]
    economic_country_event_date_df['country'] = economic_country_event_date_df['country'].str.split(', ')
    economic_country_event_date_df = economic_country_event_date_df.explode('country')

    weight_label = None
    title_normelized = ''
    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(economic_country_event_date_df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '

    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized +'Movie Distribution by Release Date and Country with Economic Struggle Theme', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized +'# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # Germany: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'Germany',  weights = weight_label)

    plt.subplot(2,3,2)
    # United States of America: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'United States of America',  weights = weight_label)

    plt.subplot(2,3,3)
    # France: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'France',  weights = weight_label)

    plt.subplot(2,3,4)
    # United Kingdom: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'United Kingdom',  weights = weight_label) 

    plt.subplot(2,3,5)
    # Italy: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'Italy',  weights = weight_label)


    plt.subplot(2,3,6)
    # Japan: distribution of film realized over time with economic Themes
    hist_sub_plot(economic_country_event_date_df, 'release_date', 'country', 'Japan',  weights = weight_label)


def hist_social_event_date_plots(event_date_df, normelized = False, dict_weight = None):

    # filter to take only movies that containes econmics related words in the plot
    social_event_list = ["rights", "oppression", "protest", "equality", "revolution", "racism", "activis", "civil rights", "discrimination", 'reform', 'social justice'," Feminism", 'feminist',  "Racism",'racist', "Homophobia",'homophobic', "Gender Equality", "Sexism", 'sexist', "Intersectionality", "LGBTQ+","LGBT", 'coming out', 'hate crime', 'misogyny', 'hatred of women', 'misogynist', 'sexual harassment', 'rape','raped', 'rapist', 'Segregation', 'Antisemitism','Anti-semitism','anti-semitic', 'antisemitic', 'jewish', 'black people',"LGBT", "Gay", "Feminist", "Social problem", "Gay Interest", "Social issues", 'Law & Crime', 'racist',  'rights', 'oppression', 'protest', 'equality', 'revolution']
        # filter dataframe to obtain only movies that have one of this words in the plot 
    social_event_date_df = list_filter(event_date_df, 'plot',social_event_list)


    # remove None and explode column country 
    social_country_event_date_df = social_event_date_df[~ social_event_date_df['country'].isna()]
    social_country_event_date_df['country'] = social_country_event_date_df['country'].str.split(', ')
    social_country_event_date_df = social_country_event_date_df.explode('country')

    weight_label = None
    title_normelized = ''
    if normelized:
        # add a column weight for the normalization plot
        add_column_weight(social_country_event_date_df, dict_weight)
        weight_label = 'weight'
        title_normelized = 'Normelized '

    fig = plt.figure(figsize=(15, 10.4))
    fig.suptitle(title_normelized +'Movie Distribution by Release Date and Country with Social Changes Theme', size = 'x-large', y =0.95)
    fig.supylabel(title_normelized + '# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    plt.subplot(2,3,1)
    # Germany: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'Germany', weights = weight_label)

    plt.subplot(2,3,2)
    # United States of America: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'United States of America', weights = weight_label)

    plt.subplot(2,3,3)
    # France: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'France', weights = weight_label)

    plt.subplot(2,3,4)
    # United Kingdom: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'United Kingdom', weights = weight_label) 

    plt.subplot(2,3,5)
    # Italy: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'Italy', weights = weight_label)


    plt.subplot(2,3,6)
    # West Germany: distribution of film realized over time with social Themes
    hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'West Germany', weights = weight_label)

    # est aussi pas mal 
    # plt.subplot(2,3,6)
    # # West Germany: distribution of film realized over time with social Themes
    # hist_sub_plot(social_country_event_date_df, 'release_date', 'country', 'Soviet Union') 


def hist_USA_regrouped_genre_date_plot(event_date_df):
    #remove film with no country
    event_date_df.dropna(subset= ['country'], inplace = True)
    # keep only movies that are made by USA
    USA_event_date_df= event_date_df[event_date_df['country'].apply(lambda x: 'United States of America' in x)]
    
    war_theme = ["War", "Political", "Anti-war", 'politic', 'war', 'revolution', 'Propaganda', 'Ideology', 'Military']
    social_theme = ["LGBT", "Gay", "Feminist", "Social problem", "Gay Interest", "Social issues", 'Law & Crime', 'racist',  'rights', 'oppression', 'protest', 'equality', 'revolution']
    economic_theme =  ["Business", "Finance", "Investing",  'poverty', 'Economic collapse', 'Unemployment', 'jobless','Inflation','Stock market','Bankruptcy','Debt']

    USA_war_date_df = list_filter(USA_event_date_df, 'plot',war_theme)
    USA_social__date_df = list_filter(USA_event_date_df, 'plot',social_theme)
    USA_economic__date_df = list_filter(USA_event_date_df, 'plot',economic_theme)

    # plot distribution 
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('USA Movie Distribution by Release Date and Thematic', size = 'x-large', y =0.95)
    fig.supylabel('# Movies Released', x = 0.05)
    fig.supxlabel('Released Date')

    # war/politic plot
    plt.subplot(1,3,1)
    sns.histplot(x = 'release_date', data= USA_war_date_df, weights = None, binwidth =5 ); 
    plt.xticks(rotation = 90)
    plt.title('War/Politics')
    plt.xlabel("")
    plt.ylabel("")

    # Social Changes plot
    plt.subplot(1,3,2)
    sns.histplot(x = 'release_date', data= USA_social__date_df, weights = None, binwidth =5 ); 
    plt.xticks(rotation = 90)
    plt.title('Social Changes')
    plt.xlabel("")
    plt.ylabel("")

    # Economic struggle plot
    plt.subplot(1,3,3)
    sns.histplot(x = 'release_date', data= USA_economic__date_df, weights = None, binwidth =5 ); 
    plt.xticks(rotation = 90)
    plt.title('Economic Struggle')
    plt.xlabel("")
    plt.ylabel("")

def plot_woman_rights_movies(movies_clean_df):
    woman_right_themes = [
        "Feminist", "gender equality", "pay gap", "patriarchy", 
        "domestic violence", "sexual harassment", "suffragette", "intersectional"
    ]
    pattern_woman_right = '|'.join(woman_right_themes)
    df_movies = movies_clean_df[movies_clean_df['country'].str.contains('United States of America', na=False)]
    df_woman_right_movies = df_movies[df_movies['plot'].str.contains(pattern_woman_right, case=False, na=False)]
    total_movies_per_year = df_movies[df_movies["country"] == "United States of America"]['release_date'].value_counts().sort_index()
    woman_rights_movies_per_year = df_woman_right_movies['release_date'].value_counts().sort_index()

    normalized_woman_rights_movies_per_year = woman_rights_movies_per_year / total_movies_per_year

    woman_rights_movies_per_year.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.title('Frequency of Woman rights Related Movies per Year')
    plt.show()

    normalized_woman_rights_movies_per_year = normalized_woman_rights_movies_per_year[normalized_woman_rights_movies_per_year > 0]

    normalized_woman_rights_movies_per_year.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Year')
    plt.ylabel('Normalized Number of Movies')
    plt.title('Normalized Frequency of Woman rights Related Movies per Year')
    plt.show()
def plot_event_movies(df_movies, event_words, event_name, country=None):
    
    pattern_event = '|'.join(event_words)

    df_event_movies = df_movies[df_movies['plot'].str.lower().str.contains(pattern_event, case=False, na=False)]

    if country:
        total_movies_per_year = (
            df_movies[df_movies["country"] == country]['release_date']
            .value_counts()
            .sort_index()
        )
    else:
        total_movies_per_year = (
            df_movies['release_date']
            .value_counts()
            .sort_index()
        )

    event_movies_per_year = (
        df_event_movies['release_date']
        .value_counts()
        .sort_index()
    )

    normalized_event_movies_per_year = event_movies_per_year / total_movies_per_year

    plt.figure(figsize=(12, 6))
    event_movies_per_year.plot(kind='bar', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.title(f'Frequency of {event_name}-Related Movies per Year')
    if country:
        plt.title(f'Frequency of {event_name}-Related Movies per Year in {country}')
    plt.show()

    normalized_event_movies_per_year = normalized_event_movies_per_year[normalized_event_movies_per_year > 0]

    plt.figure(figsize=(12, 6))
    normalized_event_movies_per_year.plot(kind='bar', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Normalized Number of Movies')
    plt.title(f'Normalized Frequency of {event_name}-Related Movies per Year')
    if country:
        plt.title(f'Normalized Frequency of {event_name}-Related Movies per Year in {country}')
    plt.show()
   
def add_weight(df, column_weighted, apply_to_same_df = True, df_to_weight = None):
    dict_weight = df[column_weighted].value_counts().to_dict()
    dict_inv_weight = {key: 1/value for key, value in dict_weight.items()}
    if apply_to_same_df:
        df['weight'] = df['release_date'].map(dict_inv_weight)
        return df
    else: 
        df_to_weight['weight'] = df_to_weight['release_date'].map(dict_inv_weight)
        return df_to_weight
    

def create_dict_weight(df, column_weighted):
    dict_weight = df[column_weighted].value_counts().to_dict()
    dict_inv_weight = {key: 1/value for key, value in dict_weight.items()}
    return dict_inv_weight 

def add_column_weight(df, dict_weight):
    df['weight'] = df['release_date'].map(dict_weight)
    return df

def filter_USA_date(df):
    date_df = filter_date(df)
    country_date_df = date_df.dropna(subset= ['country'], inplace = False)
    USA_date_df = country_date_df[country_date_df['country'].apply(lambda x: 'United States of America' in x)]
    return USA_date_df

def dict_USA_date_weight(movies_clean_df):
    USA_date_df = filter_USA_date(movies_clean_df)
    USA_date_weight_dict = create_dict_weight(USA_date_df, 'release_date')
    return USA_date_weight_dict

def dict_date_plot_weight(movies_clean_df):
    plot_date_df = filter_per_plot_date(movies_clean_df)
    plot_date_weight_dict = create_dict_weight(plot_date_df, 'release_date')
    return plot_date_weight_dict

def dict_date_plot_country_weight(movies_clean_df):
    plot_date_df = filter_per_plot_date(movies_clean_df)
    country_plot_date_df = plot_date_df.dropna(subset= ['country'], inplace = False)
    country_plot_date_weight_dict = create_dict_weight(country_plot_date_df, 'release_date')
    return country_plot_date_weight_dict

def dict_USA_date_plot_weight(movies_clean_df):
    USA_date_df = filter_USA_date(movies_clean_df)
    USA_date_plot_df =USA_date_df.dropna(subset= ['plot'], inplace= False) 
    USA_date_plot_weight_dict = create_dict_weight(USA_date_plot_df, 'release_date')
    return USA_date_plot_weight_dict