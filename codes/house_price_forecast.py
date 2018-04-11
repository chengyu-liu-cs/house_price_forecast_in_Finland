import logging
import math
import numpy as np
import os
import pandas as pd

from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

def combine_houseprice_family_data(house_price, family):
    '''Combine house price and family data.

    Args:
        house_price, family (DataFrame):
    Return:
        A DataFrame
    '''
    logging.debug('Combining house price and family data')
    df_house = house_price.copy()
    df_family = family.copy()
    df_family = df_family.set_index(['information', 'area']).stack().reset_index()
    df_family.columns = ['information', 'area', 'year', 'value']
    df_family = pd.pivot_table(df_family, index=['area', 'year'],
            values='value', columns=['information'], aggfunc=np.sum).reset_index()
    df_family['year'] = df_family['year'].astype(int)
    # combine family and house price data
    df = pd.merge(df_house, df_family,  how='inner', on=['area', 'year'])
    # reorder columns and sort values
    df = df.sort_values(['area', 'information', 'year'])
    return df

def clean_house_price_data(df, threshold=3):
    ''' Clean house price data.
    In the house price data, not all the counties contain sale observations or house price
    over last 9 years. Some of them even do not have any records, which should be removed.
    For example, 009_Alavieska does not have any information about price or observations
    for block_of_flats. Here a threshold is used to filter out county or types of houses
    that do not have enough information.

    Args:
        house_price (DataFrame): can be dataframe that is from load_house_price_data() or
            from combine_houseprice_family_data().
        threshold (int): a min number of records. Default is 3
    Return:
        a DataFrame
    '''
    logging.debug('Cleaning house price data')
    agg = df.groupby(['area','information']).count().reset_index()
    agg = agg[agg['price_per_square_meter'] >= threshold][['area', 'information']]
    rtn = pd.merge(df, agg,  how='inner', on=['area', 'information'])
    return rtn

def add_response_column(df):
    ''' Add a new column after shifting the house price column by one. Values of the new
    column indicate the price of next year and will be the response (forecasting) variable
    for training model.

    Args:
        df (DataFrame): can be dataframe that is from load_house_price_data() or
            from combine_houseprice_family_data().
    Return:
        DataFrame with an extra column 'response' that is the next year price and
            is used to train models.
    '''
    logging.debug('Creating a response column')
    df = df.copy()
    df['response'] = df['price_per_square_meter'].shift(-1)
    df.loc[df['year'] == 2016, 'response'] = np.nan
    return df

def impute_missing_value(df):
    '''Missing values are not desired when machine learning methods are applied.
    Here a strategy is used to impute missing value. The idea is to propagate last
    valid observation forward to next valid. There are also other options such as backfill.
    See details
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
    '''
    logging.debug('imputing missing values')
    df = df.fillna(method='ffill') # backfill, ffill
    return df

def load_family_data(filename, sep=';'):
    '''Load family data which was from Statistics Finland. It is hard to understand the
    factors that affect the house price using the house price table alone. Additional
    information will help understand the behind reason.

    This table contains the number of of families and average family size in each area
    (county) from 2008 to 2016. This information is used to provide better insights into
    reason behind house price changes in Finland.

    Args:
        filename (string):
        sep (string): Default ';'
    '''
    logging.debug('Loading family data')
    family = pd.read_csv(filename, sep)
    return family

def load_house_price_data(filename, sep=';'):
    '''Read housing price file
        This table contains

    Args:
        filename (string):
        sep (string): Default ';'
    '''
    logging.info('Loading house price data')
    house_price = pd.read_csv(filename, sep)
    return house_price

def load_other_data(filename, sep):
    '''
    '''
    logging.info('Loading other data from Statistics of Finland')
    data = pd.read_csv(filename, sep)
    return data

def  load_population_data(filename, sep):
    ''' Read population density file
    Not sure whether the errors in the area and population density columns are intended.
    For example, the calculations were not correct. e.g., 9899 / 1008.79 = 9.812745962985359
    (but got 0,43125). Another example is that the columns have values like '29.40.00' and
    '55.49.00', which were manually corrected from source file.

    The population density column was recalculated.

    Args:
        filename (string):
        sep (string): Default ';'
    '''
    logging.info('Loading population data')
    population = pd.read_csv(filename, sep)
    population['population_area_ratio'] = population['population'] / population['area']
    return population

def encode_categorical_variables(data):
    ''' Convert categorical variables into numerical variables by applying one hot
    encoding method.
    See http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    Args:
        data (DataFrame)
    Return:
        A DataFrame with one hot encoded variables
    '''
    def _get_ordered_county(counties):
        '''categorical values for variables
        '''
        output = []
        for x in counties:
            if x not in output:
                output.append(x)
        return output
    X = data.select_dtypes(include=[object])
    # encode labels with value between 0 and n_classes-1.
    le = preprocessing.LabelEncoder()
    X_2 = X.apply(le.fit_transform)
    # Create a OneHotEncoder object, and fit it to all of X and Transform it to one hot encoding.
    enc = preprocessing.OneHotEncoder()#(sparse=False)
    enc.fit(X_2)
    onehotlabels = enc.transform(X_2)
    # create one hot encoding dataframe
    columns = _get_ordered_county(X['area'].tolist() + X['information'].tolist())
    encoded_vars = pd.DataFrame(onehotlabels.toarray(), index=data.index, columns=columns)
    return encoded_vars

def data_split(data):
    '''Split datafram into training, testing and new predicting datasets.

    Data from year 2008 to year 2014 are used for training and data from 2015 are used
    for testing.
    Data from year 2016 is used for new predicting datasets which are not able to validate
    but can be used to predict house price for 2017.

    Args:
        data (DataFrame):
    Return:
        Five DataFrame objects: X_train, Y_train, X_test, Y_test, data_2017
    '''
    # data for predicting house price of 2017
    data_2017 = data[data['year'] == 2016]
    # data from training and testing datasets.
    train_test_data = data[data['year'] != 2016]
    # train data
    X_train = train_test_data[train_test_data['year'] != 2015].drop(columns=['response',
                    'area', 'information'])
    Y_train = train_test_data[train_test_data['year'] != 2015][['response']]
    # test data
    X_test = train_test_data[train_test_data['year'] == 2015].drop(columns=['response',
                    'area', 'information'])
    Y_test = train_test_data[train_test_data['year'] == 2015][['response']]
    return X_train, Y_train, X_test, Y_test, data_2017

def get_country_level_agg(data):
    '''Get country level aggregation information such as price, sale observation,
    family size, family numbers.

    Args:
        data (DataFrame):
    Return:
        A DataFrame object with columns: 'year', 'area', 'information',
                'price_per_square_meter', 'observations',
                'avg_size_of_family', 'num_of_families'
    '''
    country = data.groupby(['year', 'information']).agg({'price_per_square_meter': 'mean',
                                            'observations': 'sum',
                                            'avg_size_of_family': 'mean',
                                            'num_of_families': 'sum'}).reset_index()
    country['area'] = '000_Finland'
    # reorder
    country = country[['year', 'area', 'information', 'price_per_square_meter',
            'observations', 'avg_size_of_family', 'num_of_families']]
    return country

def evaluate_rmse(y_predict, Y_test):
    # evaluate performance using root mean squared error
    regr_mse = mean_squared_error(y_predict, Y_test)
    regr_rmse = math.sqrt(abs(regr_mse))
    return regr_rmse

def lr_train(x_train, y_train):
    # linear regression model fit
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train.values.ravel())
    return model

def lr_predict(model, df):
    # prediction on testing data
    y_predict = model.predict(df)
    return y_predict

def rf_train(x_train, y_train):
    # train random forest
    model = RandomForestRegressor(n_estimators=50,max_depth=11,min_samples_split=2,
                        random_state=0)
    model.fit(x_train, y_train.values.ravel())
    return model

def predict(model, df):
    # prediction
    y_predict = model.predict(df)
    return y_predict

def adaboost_train(x_train, y_train):
    model = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=0.001,
                        loss='linear', random_state=None)
    model.fit(x_train, y_train.values.ravel())
    return model

def adaboost_predict(model, df):
    y_predict = model.predict(df)
    return y_predict


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                format='[%(asctime)s][%(name)s:%(filename)s:%(lineno)d]%(levelname)s: %(message)s')
    # filenames
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_family = os.path.join(dir_path, '../data/family_data.csv')
    file_house_price = os.path.join(dir_path,'../data/housing_prices.csv')
    file_other_features = os.path.join(dir_path, '../data/all_data.csv')
    file_prediction = os.path.join(dir_path, '../data/prediction.csv')
    # load Finnish family size and number, house price and population data
    family = load_family_data(file_family, sep=';')
    house_price = load_house_price_data(file_house_price, sep=';')
    otherdata = load_other_data(file_other_features, sep=';')

    # Clean house_price data so that counties or types of houses that do not have enough records
    #  are filtered out
    house_price = clean_house_price_data(house_price, threshold=3)

    # combine house price and other data (31 variables)
    data = combine_houseprice_family_data(house_price, otherdata)

    # add a column of response by shifting the price column by one.
    new_df = add_response_column(data)
    imputed_data = impute_missing_value(new_df)
    encoded_vars = encode_categorical_variables(imputed_data)
    encoded_data = pd.concat([imputed_data, encoded_vars], axis=1)

    # split data into, training, testing and predicting data for 2017
    X_train, Y_train, X_test, Y_test, predict_2017 = data_split(encoded_data)

    # Linear regression training and testing
    model = lr_train(X_train, Y_train)
    y_predict = predict(model, X_test)
    # random forest training and testing
    model_rf = rf_train(X_train, Y_train)
    y_predict_rf = predict(model_rf, X_test)
    # AdaBoost training and testing
    model_adaboost = adaboost_train(X_train, Y_train)
    y_predict_boost = predict(model_adaboost, X_test)
    ## evaluate performance using root mean squared error
    # linear regression
    print "Root Mean Square Error (lr): {}".format(evaluate_rmse(y_predict, Y_test))
    print "R^2 regression score (lr): {}".format(r2_score(y_predict, Y_test))
    # lr: 0.942048017149 (Higher better, 1 is the best and 0 is the least desired)
    # 119.190581072,  So we are an average of 119.19 Euro away from the ground truth
    #   price when making predictions on our test set.
    # Random Forest
    print "Root Mean Square Error (rf): {}".format(evaluate_rmse(y_predict_rf, Y_test))
    print "R^2 regression score (rf): {}".format(r2_score(y_predict_rf, Y_test))
    # AdaBoost
    print "Root Mean Square Error (Adaboost): {}".format(evaluate_rmse(y_predict_boost, Y_test))
    print "R^2 regression score (Adaboost): {}".format(r2_score(y_predict_boost, Y_test))

    # predict house price for 2017
    y_pred_hat = predict(model, predict_2017.drop(columns=['response', 'area', 'information']))
    predict_2017.loc[:, 'price_per_square_meter'] = y_pred_hat
    predict_2017.loc[:, 'year'] = 2017
    predict_2017.loc[:, 'observations'] = np.nan
    df = pd.concat([data, predict_2017[data.columns]])
    df = df.sort_values(['area', 'information', 'year'])
    df.to_csv(file_prediction)
