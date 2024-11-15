import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    for i in range(nbr_dates//time_bin): #iteration entre bin
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
    for i in range(nbr_dates%time_bin):
        current_sum += collab_date_df.iloc[:,date_index]
        date_index += 1

    #update last column 
    end_date_index = date_index -1 
    column_name = f'[{collab_date_df.columns[begin_date_index]} - {collab_date_df.columns[end_date_index]}]'
    merge_collab_date_df[column_name] = current_sum.values
    return merge_collab_date_df.loc[country_list, :]

# build heatmap to see better the results by normalizing also each row
def normelize_collab_USA_heatmap(merge_collab_date_df):
    # we are not only interested in absolut value but in the variation of number of movies produced also, so let's just normalize each row
    normelize_merge_collab_date_df = merge_collab_date_df.div(merge_collab_date_df.sum(axis=1), axis=0)
    plt.figure(figsize=(7, 7))
    sns.heatmap(normelize_merge_collab_date_df, fmt= '.0f', annot = merge_collab_date_df, cbar_kws={'label': 'Normalized # Movies per Country'})
    plt.title("Collaborators with USA : Distribution of Movies per Released Date")
    plt.xlabel("Released Date")
    plt.xticks(rotation = 90)
    plt.ylabel("Collaborators")
