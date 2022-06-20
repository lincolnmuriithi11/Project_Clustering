################ Libraries and documents needed for this project ################
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from env import host, user, password

import warnings
warnings.filterwarnings("ignore")

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

###################### Acquire and Clean Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# ----------------------------------------------------------------- #
#
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
# ----------------------------------------------------------------- #

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
############################## Engineered Features###########################
#this function takes in a data frame and engineers features that give a better overview of the data 
def engineered_features(df):
    df["price_sqft"] = df.taxvaluedollarcnt/df.calculatedfinishedsquarefeet.astype(int) #adding a column for price per sqft
    df["age"] =2017-df.yearbuilt # converting year built into age by subtracting by 2017, year of the transaction
    df["price_bath"] = (df.taxvaluedollarcnt/ df.bathroomcnt).astype(int)#price per bathroom is a feature that is price of the house per every bath added
    df = df.join(df.groupby("regionidzip").taxvaluedollarcnt.mean(), on="regionidzip", rsuffix = "_zone") #grouping by average prices per zone using zipcode
    df["taxvaluedollarcnt_zone"] = df.taxvaluedollarcnt_zone.astype(int)# this code creates an integer feature for the avg price per zip
    df["regionidzip"] = df.regionidzip.astype(int) #converts zipcode to an integer
    df = df.drop(columns = ["county","propertylandusedesc","max_transactiondate","yearbuilt"]) #dropping any redundant columns after feature engineering 
    df['age_bin'] = pd.cut(df.age, bins = [0, 15, 30, 45, 60, 70, 120], labels = ['0 to 15', '15 to 30', '30 to 45','45 to 60', '60 to 70', '70 to 120']) # creating age bins 
    df["price_region_bins"] = pd.cut(df.taxvaluedollarcnt_zone, bins = [0, 100000, 250000, 350000, 500000, 650000, 850000,1000000, 1400000], labels = ['0 to 100k', '100k to 250k', '250k to 350k','350k to 500k', '500k to 650k', '650k to 850k','850k to 1mil','1mil to 1.4mil'])
    #creating bins for the zipcode mean prices 
    df = df.dropna()
    return df

# ----------------------------------------------------------------- #
    
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
    
# splitting dadta to train, validate, and test to avoid data leakage
def split_data(df):
    '''
    This function performs split on zillow data, stratify taxvaluedollarcnt.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
                                            # stratify = df.logerror) in regression its not neccesary to scale your target variable
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
                                    #    stratify=train_validate.logerror)
    return train, validate, test




#this function takes the clean and split data, makes copies, uses the standard scaler to scale the data for modeling purposes. 
def scaling_standard(train, validate, test, columns_to_scale):

    '''
    This function takes in a data set that is split , makes a copy and uses the  standard scaler to scale all three data sets. additionally it adds the columns names on the scaled data and returns trainedscaled data, validate scaled data and test scale
    '''
    #copying the dataframes for distinguishing between scaled and unscaled data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # defining the standard scaler 
    scaler = StandardScaler()
    
    #scaling the trained data and giving the scaled data column names 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    #scaling the validate data and giving the scaled data column names 
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    
    #scaling the test data and giving the scaled data column names 
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #returns three dataframes; train_scaled, validate_scaled, test_scaled
    return train_scaled, validate_scaled, test_scaled


# code for mean max scaling that takes in split dataframes and columns intended to be scaled and returns scaled data
def scaling_minmax(train, validate, test, columns_to_scale):

    '''
    This function takes in a data set that is split , makes a copy and uses the min max scaler to scale all three data sets. additionally it adds the columns names on the scaled data and returns trainedscaled data, validate scaled data and test scale
    '''
    #copying the dataframes for distinguishing between scaled and unscaled data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # defining the minmax scaler 
    scaler = MinMaxScaler()
    
    #scaling the trained data and giving the scaled data column names 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    #scaling the validate data and giving the scaled data column names 
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    
    #scaling the test data and giving the scaled data column names 
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #returns three dataframes; train_scaled, validate_scaled, test_scaled
    return train_scaled, validate_scaled, test_scaled