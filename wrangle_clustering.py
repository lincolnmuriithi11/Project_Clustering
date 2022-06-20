import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn

from sklearn.model_selection import train_test_split
# imputer from sklearn
from math import sqrt
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire and Clean Data ######################
#sql credentials 
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# ----------------------------------------------------------------- #
#acquiring zillow data from SQL data base 
def get_zillow_sql():

    ''' this function calls a sql file from the codeup database and creates a data frame from the zillow db.
    '''
    query ='''
     SELECT  
    *
    FROM properties_2017 P_2017
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN
    predictions_2017 USING (parcelid)
        LEFT JOIN
    propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN
    airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN
    architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN
    buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN
    heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN
    storytype USING (storytypeid)
        LEFT JOIN
    typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN
    unique_properties USING (parcelid)
    WHERE
    propertylandusedesc = 'Single Family Residential'
        '''
    df = pd.read_sql(query, get_connection('zillow'))
    #creating a csv for easy access 
    return df
        
def get_zillow_df():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_cluster.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_cluster.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_zillow_sql()
        
        # Cache data
        df.to_csv('zillow_cluster.csv', index = 0)
        
    return df


# # outlier handling to remove quant_cols with >3.5 z-score (std dev)
# # this is another way of removing outliers
# def remove_outliers(df, calculated, columns):

#     z = np.abs((stats.zscore(df[quant_cols])))
#     df_without_outliers=  df[(z < threshold).all(axis=1)]
#     print(df.shape)
#     print(df_without_outliers.shape)
#     return df_without_outliers

# this code handle outliers with a 20 cutoff on missing data 
def handle_nulls(df):
    pct_null = df.isnull().sum() / len(df)
    missing_features = pct_null[pct_null > 0.20].index
    df.drop(missing_features, axis=1, inplace=True)
    df.dropna(inplace= True)
    return df


###function is for acquiring the zillow data, dropping nulls/ nan, removing outliers,
###converting the fips data to categorical, changing fips data to counties, renaming columns and saving the clean data in clean_zillow_data
def wrangle_zillow():
    """this function is for acquiring the zillow data, dropping nulls/ nan, removing outliers,converting the fips data to categorical, changing fips data to counties, renaming columns and saving the clean data in clean_zillow_data"""
   
    df = get_zillow_df()# this function calls the zillow df into the wrangle function for trh cleaning process
    df = handle_nulls(df)
    df = df.replace({"regionidzip":{399675: 99675}})#fixing the 8 rows of zipcodes with error
    df['regionidzip'] = df['regionidzip'].astype("category") #converting zipcodes from numerical to categorical 
    df = df[df.calculatedfinishedsquarefeet <= 8000] #removed outliers by cutting houses over 60000 sqfeet, below 70feet
    df = df[df.calculatedfinishedsquarefeet>70] #removes outliers below 70 sqfeet
    df = df[df.taxvaluedollarcnt<=1_800_000]#removed houses over 1.8 million in dollar amount to remove outliers
    df = df[df.bedroomcnt <=8]# removed houses above 8 bedrooms/
    df = df[df.bathroomcnt <=8] # and 8baths  in order to have a normal distribution
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)] #take out any bathrooms and bedroomcnt equal to 0
    df = df[df.logerror <= 3]#capped the logerror to 3
    df = df[df.logerror>= -3]# bottom cap logerror to-3
    df = df[['parcelid','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','fips','longitude','latitude','regionidzip','yearbuilt','taxvaluedollarcnt','max_transactiondate','logerror','propertylandusedesc']] #columns to keep
    df['month'] = pd.DatetimeIndex(df['max_transactiondate']).month #converted max tranaction date to months sold
    df["fips"] = pd.Categorical(df.fips) #converted county data to categorical and changed the names to county names for readability purposes
    df['fips'] = df['fips'].astype(str).apply(lambda x: x.replace('.0',''))#stripped fips to later convert to counties
    df = df.rename(columns = {"fips":"county"})#renamed columns for easy readability as well
    df["county"].replace("6111",'Ventura', inplace=True)#converting fips data to counties
    df["county"].replace("6059",'Orange', inplace=True)#converting fips data to counties
    df["county"].replace("6037",'Los_Angeles', inplace=True)#converting fips data to counties
    dummy_df = pd.get_dummies(df[['county']], dummy_na =False)#created dummies for counties
    df = pd.concat([df, dummy_df], axis = 1)

    return df
# this function is splitting data to train, validate, and test to avoid data leakage
# finding the best features using kbest 
def features_kbest(X, y, n):
    """this function takes in a data frame with numeric features , uses kbest algorithm and brings back the top n features according to the target"""
    f_selector = sklearn.feature_selection.SelectKBest(f_regression, k = n)
    f_selector.fit(X, y)
    f_support = f_selector.get_support()
    #f_feature = X.loc[:,f_support].columns.tolist()
    f_feature = X.columns[f_support]
    return f_feature

# finding the best features using RFE
def features_rfe(X, y, n):
    """this function takes in a data frame with numeric features , uses rfe algorithm and brings back the top n features according to the target"""
    lm = LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n)
    X_rfe = rfe.fit(X,y)
    mask = rfe.support_
    rfe_features = X.columns[mask]
    return rfe_features
    