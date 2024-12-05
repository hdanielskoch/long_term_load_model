import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display
from datetime import datetime
import seaborn as sns
import warnings

client = bigquery.Client(project='ebce-models')

def add_territory_feature(df, expansion_date):
    """
    Add binary feature for pre/post territory addition
    """
    territory_change_date = pd.Timestamp(expansion_date)
    df['post_territory_change'] = (df['datetime'] >= territory_change_date).astype(int)
    return df

def get_load_features(table_name='ebce-models.hdk_temp.weathersource_load_hourly_actuals'):
    """
    Retrieves load features data from BigQuery and returns basic information about the dataset.
    
    Args:
        table_name (str): Full path to BigQuery table (project.dataset.table)
                         Default: 'ebce-models.hdk_temp.weathersource_load_hourly_actuals'
    
    Returns:
        DataFrame: Load features data including datetime, load, and weather information
    """
    warnings.filterwarnings('ignore', message='Unable to represent RANGE schema as struct using pandas ArrowDtype')

    query = f"""
        -- 73MB
        SELECT *
        FROM `{table_name}`
        ORDER BY datetime
    """

    # Execute the query
    query_job = client.query(query)
    load_features = query_job.to_dataframe()

    # Print dimensions
    display(f"Load dimensions: {load_features.shape}")

    min_time = load_features['datetime'].min()
    max_time = load_features['datetime'].max()
    print(f"The range of time for weather data is from {min_time} to {max_time}.")

    # Display sample data
    display(load_features.head())
    
    return load_features


def plot_customers_and_load(df):
    """
    Plot number of customers and total system load over time.
    
    Args:
        df (DataFrame): DataFrame containing datetime, total_system_load, and total_system_customers columns
    """
    # Create the figure
    fig = go.Figure()

    # Add trace for Total System Load
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df['total_system_load'], 
        mode='lines', 
        name='Total System Load', 
        line=dict(color='blue')
    ))

    # Add trace for Number of Customers
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df['total_system_customers'], 
        mode='lines', 
        name='Number of Customers', 
        line=dict(color='green'),
        yaxis='y2'  # Specify secondary y-axis
    ))

    # Update layout for two y-axes
    fig.update_layout(
        title='Number of Customers and Total System Load Over Time',
        xaxis_title='Datetime',
        yaxis=dict(
            title='Total System Load',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Number of Customers',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        
        legend=dict(x=0, y=1),
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white'
    )

    # Show the plot
    fig.show()
    return


def remove_anomalous_data(df, anomaly_datetimes=['2023-03-06 23:00:00']):
    """
    Remove anomalous data from the dataframe for specific datetimes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        anomaly_datetimes (list): List of datetime strings of anomalies to remove
        
    Returns:
        pd.DataFrame: Dataframe with anomalies removed
    """
    df_clean = df.copy()
    for anomaly_dt in anomaly_datetimes:
        df_clean = df_clean[df_clean['datetime'] != anomaly_dt]
    print(f"Load dimensions before transformation: {df.shape}")
    print(f"Load dimensions after transformation: {df_clean.shape}")
    return df_clean

def handle_nulls(df, columns_to_check=None, verbose=True):
   """
   Remove rows with null values and report statistics
   
   Args:
       df: DataFrame to clean
       columns_to_check: List of columns to check for nulls. If None, uses default temperature columns
       verbose: Whether to print diagnostic information
   
   Returns:
       Cleaned DataFrame
   """
   if columns_to_check is None:
       columns_to_check = ['temp_3h_ago', 'temp_2h_ago', 'temp_1h_ago', 'temperature_avg']
   
   # Create copy of data
   result_df = df.copy()
   
   if verbose:
       print(f"Initial number of rows: {len(result_df)}")
       # Check for nulls and display counts
       null_counts = result_df.isnull().sum()
       print("\nNull counts before cleaning:")
       print(null_counts[null_counts > 0])
    
   
   # Remove rows with nulls in specified columns
   for col in columns_to_check:
       result_df = result_df[result_df[col].notnull()]
   
   if verbose:
       # Print final statistics
       print(f"\nFinal number of rows: {len(result_df)}")
       print("\nNull counts after cleaning:")
       null_counts = result_df.isnull().sum()
       print(null_counts[null_counts > 0] if any(null_counts > 0) else "No nulls remaining")
       
       # Calculate percentage of data retained
       retention_rate = (len(result_df) / len(df)) * 100
       print(f"\nPercentage of data retained: {retention_rate:.3f}%")
       
       # Print info about removed rows
       rows_removed = len(df) - len(result_df)
       print(f"\nTotal rows removed: {rows_removed}")
   
   return result_df

def handle_holidays(df):
    """
    Modifies day_of_week based on holidays and surrounding days.
    """
    result_df = df.copy()
    result_df['day_of_week'] = result_df['datetime'].dt.weekday
    result_df['day_of_week_modified'] = result_df['day_of_week']
    
    # Define holidays
    holidays = {
        'new_years_day': {'month': 1, 'day': 1, 'before': (12, 31), 'friday_to_saturday': True},
        'memorial_day': {'month': 5, 'day_last_monday': True, 'after': (5, 'last_monday')},
        'independence_day': {'month': 7, 'day': 4, 'before': (7, 3)},
        'labor_day': {'month': 9, 'day_first_monday': True, 'after': (9, 'first_monday')},
        'thanksgiving_day': {'month': 11, 'day_fourth_thursday': True, 'before': (11, 'fourth_thursday'), 'after': (11, 'fourth_thursday')},
        'christmas_day': {'month': 12, 'day': 25, 'before': (12, 24), 'after': (12, 26)}
    }
    
    def is_holiday(date, holiday_info):
        if 'day' in holiday_info and date.month == holiday_info['month'] and date.day == holiday_info['day']:
            return True
        if 'before' in holiday_info and date.month == holiday_info['before'][0] and date.day == holiday_info['before'][1]:
            return True
        if 'after' in holiday_info and date.month == holiday_info['after'][0] and date.day == holiday_info['after'][1]:
            return True
        # Memorial Day
        if 'day_last_monday' in holiday_info and date.month == holiday_info['month'] and date.weekday() == 0:
            last_monday = max([d for d in pd.date_range(start=f'{date.year}-{date.month}-01', 
                             end=pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(0)) if d.weekday() == 0])
            return date.date() == last_monday.date()
        # Day after memorial day
        if 'after' in holiday_info and date.month == holiday_info['month'] and holiday_info['after'][1] == 'last_monday' and date.weekday() == 1:
            last_monday = max([d for d in pd.date_range(start=f'{date.year}-{date.month}-01', 
                             end=pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(0)) if d.weekday() == 0])
            return date.date() == (last_monday + pd.DateOffset(days=1)).date()
        # Labor Day
        if 'day_first_monday' in holiday_info and date.month == holiday_info['month'] and date.weekday() == 0:
            first_monday = min([d for d in pd.date_range(start=f'{date.year}-{date.month}-01', 
                              end=f'{date.year}-{date.month}-07') if d.weekday() == 0])
            return date.date() == first_monday.date()
        # Day after labor day
        if 'after' in holiday_info and date.month == holiday_info['month'] and holiday_info['after'][1] == 'first_monday' and date.weekday() == 1:
            first_monday = min([d for d in pd.date_range(start=f'{date.year}-{date.month}-01', 
                              end=f'{date.year}-{date.month}-07') if d.weekday() == 0])
            return date.date() == (first_monday + pd.DateOffset(days=1)).date()
        # Thanksgiving related days
        if date.month == 11:
            fourth_thursday = [d for d in pd.date_range(start=f'{date.year}-11-01', 
                             end=pd.Timestamp(date.year, 11, 1) + pd.offsets.MonthEnd(0)) if d.weekday() == 3][3]
            # Day before Thanksgiving
            if date.weekday() == 2 and date.date() == (fourth_thursday - pd.DateOffset(days=1)).date():
                return True
            # Thanksgiving Day
            if date.weekday() == 3 and date.date() == fourth_thursday.date():
                return True
            # Day after Thanksgiving
            if date.weekday() == 4 and date.date() == (fourth_thursday + pd.DateOffset(days=1)).date():
                return True
        return False

    # Modify the 'day_of_week_modified' based on holidays
    for index, row in result_df.iterrows():
        date = row['datetime']
        for holiday, info in holidays.items():
            if is_holiday(date, info):
                if holiday in ('new_years_day', 'independence_day', 'christmas_day'):
                    if date.day == info.get('day', None):  # Exact Holiday
                        result_df.at[index, 'day_of_week_modified'] = 5 if date.weekday() == 4 else 6
                    elif holiday == 'independence_day' and 'before' in info:
                        result_df.at[index, 'day_of_week_modified'] = 4  # Friday
                    else:
                        result_df.at[index, 'day_of_week_modified'] = 5  # Saturday
                elif holiday in ['memorial_day', 'labor_day', 'thanksgiving_day']:
                    if holiday == 'memorial_day':
                        if date.weekday() == 0:
                            result_df.at[index, 'day_of_week_modified'] = 5  # Saturday
                        elif date.weekday() == 1:
                            result_df.at[index, 'day_of_week_modified'] = 0  # Monday
                    elif holiday == 'labor_day':
                        if date.weekday() == 0:
                            result_df.at[index, 'day_of_week_modified'] = 5  # Saturday
                        elif date.weekday() == 1:
                            result_df.at[index, 'day_of_week_modified'] = 3  # Thursday
                    elif holiday == 'thanksgiving_day':
                        if date.weekday() == 2:  # Day before
                            result_df.at[index, 'day_of_week_modified'] = 0  # Monday
                        elif date.weekday() in [3, 4]:  # Thanksgiving and day after
                            result_df.at[index, 'day_of_week_modified'] = 5  # Saturday
    
    # Convert Wednesdays to Tuesdays
    result_df['day_of_week_modified'] = result_df['day_of_week_modified'].replace({2: 1})
    
    return result_df

def show_holidays(df):
    # Filter rows where the date matches the holiday list (ignoring the year)
    holiday_examples = df[df['day_of_week'] != df['day_of_week_modified']]

    # Further filter to only show rows where the hour is 0 and not Tuesday
    holiday_examples = holiday_examples[(holiday_examples['datetime'].dt.hour == 0) & 
                                     (holiday_examples['day_of_week_modified'] != 1)]

    # Display one example hour for each holiday
    pd.set_option('display.max_rows', None)  
    print(holiday_examples[['datetime', 'day_of_week', 'day_of_week_modified']].tail(25))
    pd.set_option('display.max_rows', 10)
    return


## Feature Engineering
def update_customer_counts(df, reference_start='2023-11-01', reference_end='2024-11-01'):
    """
    Updates the total_system_customers column using reference period values for all years.
    
    Args:
        df: DataFrame containing datetime and total_system_customers columns
        reference_start: Start date of reference period (str or Timestamp)
        reference_end: End date of reference period (str or Timestamp)
    
    Returns:
        DataFrame with updated customer counts
    """
    # Create copy of input dataframe
    result_df = df.copy()
    
    # Get reference period customer counts
    reference_customers = df[
        (df['datetime'] >= pd.Timestamp(reference_start)) &
        (df['datetime'] < pd.Timestamp(reference_end))
    ][['datetime', 'total_system_customers']].copy()
    
    # Create temporary matching columns in both dataframes
    for df_ in [reference_customers, result_df]:
        df_['temp_month'] = df_['datetime'].dt.month
        df_['temp_day'] = df_['datetime'].dt.day
        df_['temp_hour'] = df_['datetime'].dt.hour
        
    # Drop duplicates during DST from reference data (keep first occurrence)
    reference_customers = reference_customers.drop_duplicates(
        subset=['temp_month', 'temp_day', 'temp_hour'], 
        keep='first'
    )
        
    
    # Merge reference customer counts
    result_df = pd.merge(
        result_df.drop(columns=['total_system_customers']),
        reference_customers[['temp_month', 'temp_day', 'temp_hour', 'total_system_customers']],
        on=['temp_month', 'temp_day', 'temp_hour'],
        how='left'  # Changed to left join to keep all rows
    )
    
    print(f"Rows before customer merge: {len(df)}")
    print(f"Rows after customer merge: {len(result_df)}")
    
    # Clean up temporary columns
    result_df = result_df.drop(columns=['temp_month', 'temp_day', 'temp_hour'])
    
    # Sort by datetime
    result_df = result_df.sort_values('datetime')
    
    # Verify no nulls were created
    null_count = result_df['total_system_customers'].isnull().sum()
    if null_count > 0:
        print(f"Warning: {null_count} null values created during customer count update")
    
    return result_df

def add_cross_terms(df, indicator_columns, numeric_columns):
    """Create cross terms between indicator and numeric variables"""
    cross_terms = {}
    for ind_col in indicator_columns:
        for num_col in numeric_columns:
            col_name = f'{ind_col}*{num_col}'
            cross_terms[col_name] = df[ind_col].to_numpy() * df[num_col].to_numpy()
    
    return pd.concat([df, pd.DataFrame(cross_terms, index=df.index)], axis=1)

def add_polynomial_terms(df, base_columns):
    """Create polynomial terms for given columns"""
    poly_terms = {}
    for col in base_columns:
        poly_terms[f'{col}^2'] = df[col] ** 2
        poly_terms[f'{col}^3'] = df[col] ** 3
    
    return pd.concat([df, pd.DataFrame(poly_terms, index=df.index)], axis=1)

def create_time_features(df, reference_start=None, reference_end=None):
    """
    Create time-based dummy variables with optional reference period
    
    Args:
        df: DataFrame containing time columns (hour, month, day_of_week_modified)
        reference_start: Optional start date of reference period
        reference_end: Optional end date of reference period
    
    Returns:
        DataFrame with time dummy variables
    """
    result_df = df.copy()

    
    if reference_start is not None and reference_end is not None:
        # Reference period logic
        reference_data = df[
            (df['datetime'] >= pd.Timestamp(reference_start)) &
            (df['datetime'] < pd.Timestamp(reference_end))
        ].copy()
        
        # Create temporary matching columns
        for df_ in [reference_data, result_df]:
            df_['temp_day'] = df_['datetime'].dt.day
            
        # Drop duplicates during DST from reference data
        reference_data = reference_data.drop_duplicates(
            subset=['month', 'temp_day', 'hour'], 
            keep='first'
        )
        
        # Drop day_of_week_modified from result_df before merge to avoid duplicates
        if 'day_of_week_modified' in result_df.columns:
            result_df = result_df.drop(columns=['day_of_week_modified'])

        # Merge time features from reference period
        result_df = pd.merge(
            result_df,
            reference_data[['temp_day', 'hour', 'month', 'day_of_week_modified']],
            on=['month', 'temp_day', 'hour'],
            how='left'
        )

        # Remove rows with null customer counts from the join above
        print(f"Removing Rows with null customer counts: {result_df[result_df['total_system_customers'].isnull()].datetime.values}")
        null_count = result_df['day_of_week_modified'].isnull().sum()
        if null_count > 0:
            print(f"Removed {null_count} rows with null day of week modified values")
        # Remove nulls  
        result_df = result_df[result_df['total_system_customers'].notnull()]

        #With nulls in the join, this casts int32 to float64 to hanlde nulls. Convert back.
        result_df['day_of_week_modified'] = result_df['day_of_week_modified'].astype('int32')  # Force int32 type

        
        result_df = result_df.drop(columns=['temp_day'])

    
    # Create dummy variables (works for both cases)
    hour_dummies = pd.get_dummies(result_df['hour'], prefix='hour', dtype=int)
    month_dummies = pd.get_dummies(result_df['month'], prefix='month', dtype=int)
    day_of_week_dummies = pd.get_dummies(result_df['day_of_week_modified'], 
                                        prefix='weekday', dtype=int)
    
    # Combine features
    result_df = pd.concat([
        result_df,
        hour_dummies,
        month_dummies,
        day_of_week_dummies
    ], axis=1)
    
    # Drop original time columns
    result_df = result_df.drop(columns=['hour', 'month', 'day_of_week_modified'])
    
    return result_df

def process_features_linear(df, use_reference_period=False, reference_start=None, reference_end=None):
    """
    Master function to process features for both training/testing and scenario analysis.
    
    Args:
        df: DataFrame containing base features
        use_reference_period: Boolean indicating whether to use reference period for customer counts and time features
        reference_start: Start date of reference period (only used if use_reference_period=True)
        reference_end: End date of reference period (only used if use_reference_period=True)
    
    Returns:
        DataFrame with processed features
    """
    # Start with only base features
    base_features = ['datetime', 'total_system_load', 'total_system_customers',
                    'temperature_avg', 'temp_1h_ago', 'temp_2h_ago', 'temp_3h_ago', 
                    'avg_temp_last_24h', 'day_of_week_modified', 'post_territory_change']
    
    # Create copy with only base features
    result_df = df[base_features].copy()

    # Add hour and month
    result_df['hour'] = result_df['datetime'].dt.hour
    result_df['month'] = result_df['datetime'].dt.month
    
    # Step 1: Update customer counts if using reference period
    if use_reference_period:
        if reference_start is None or reference_end is None:
            raise ValueError("Reference start and end dates required when use_reference_period=True")
        result_df = update_customer_counts(result_df, reference_start, reference_end)
    
    # Step 2: Add polynomial terms for temperature
    temp_cols = ['temperature_avg', 'temp_1h_ago', 'temp_2h_ago', 
                'temp_3h_ago', 'avg_temp_last_24h']
    result_df = add_polynomial_terms(result_df, temp_cols)
    
    # Step 3: Create time features
    reference_start_arg = reference_start if use_reference_period else None
    reference_end_arg = reference_end if use_reference_period else None
    result_df = create_time_features(result_df, reference_start_arg, reference_end_arg)
    
    # Get column names for cross terms
    weekday_cols = [col for col in result_df.columns if col.startswith('weekday_')]
    month_cols = [col for col in result_df.columns if col.startswith('month_')]
    hour_cols = [col for col in result_df.columns if col.startswith('hour_')]
    
    # Step 4: Add cross terms
    # Temperature cross terms
    indicator_columns = month_cols + hour_cols
    numeric_columns = ['temperature_avg', 'temperature_avg^2', 'temperature_avg^3', 
                   'temp_1h_ago', 'temp_1h_ago^2', 'temp_1h_ago^3',
                   'temp_2h_ago', 'temp_2h_ago^2', 'temp_2h_ago^3',
                   'temp_3h_ago', 'temp_3h_ago^2', 'temp_3h_ago^3',
                   'avg_temp_last_24h', 'avg_temp_last_24h^2', 'avg_temp_last_24h^3'
    ]
    
    result_df = add_cross_terms(result_df, indicator_columns, numeric_columns)
    
    # Weekday-month cross terms
    result_df = add_cross_terms(result_df, weekday_cols, month_cols)
    
    # Print feature information
    time_features = [c for c in result_df.columns if c.startswith(('month_', 'hour_', 'weekday_'))]
    weather_features = [c for c in result_df.columns if any(x in c for x in ['temperature', 'temp_'])]
    cross_features = [c for c in result_df.columns if '*' in c]
    
    print(f"Total columns: {len(result_df.columns)}")
    print(f"Total features: {len(result_df.drop(columns=['datetime', 'total_system_load']).columns)}")
    print(f"Time features: {len(time_features)}")
    print(f"Weather features: {len(weather_features)}")
    print(f"Cross features: {len(cross_features)}")
    
    return result_df


def add_periodic_time_features(df, day_of_week_col='day_of_week_modified'):
    """
    Add periodic time features (sin/cos transformations) for month, day of year, day of week, and hour.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a datetime column
    day_of_week_col : str, optional
        Name of the day of week column to use (default: 'day_of_week_modified')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added periodic time features
    """
    result = df.copy()
    
    # Extract time components
    result['month'] = result['datetime'].dt.month
    result['hour'] = result['datetime'].dt.hour 
    result['day_of_year'] = result['datetime'].dt.dayofyear

    # Add periodic features
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    result['day_of_year_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result['day_of_year_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
    
    result['day_of_week_sin'] = np.sin(2 * np.pi * result[day_of_week_col] / 7)
    result['day_of_week_cos'] = np.cos(2 * np.pi * result[day_of_week_col] / 7)
    
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    
    return result

def select_features(df, features):
    """
    Select core features for neural network model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing all features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features only
    """
    result = df[features]
    
    return result

def slice_by_date(df, start_date, end_date, datetime_col='datetime'):
    """
    Slice a DataFrame based on a date range (inclusive start, exclusive end).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the datetime column
    start_date : str or pandas.Timestamp
        Start date (inclusive)
    end_date : str or pandas.Timestamp
        End date (exclusive)
    datetime_col : str, optional
        Name of the datetime column (default: 'datetime')
        
    Returns:
    --------
    pandas.DataFrame
        Sliced DataFrame containing only rows within the specified date range
    """
    
    # Slice the DataFrame
    sliced_df = df[(df[datetime_col] >= start_date) & 
                   (df[datetime_col] < end_date)].copy()
    
    # Print info about the slice
    print(f"Date range: {start_date} to {end_date}")
    
    return sliced_df

def plot_avg_load_per_customers(df):
    """
    Plot number of customers, total system load and kwh per customer vs datetime
    
    Args:
        df: DataFrame containing datetime, total_system_load and total_system_customers columns
    """
    # Create the figure
    fig = go.Figure()

    # Add trace for Number of Customers
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df['total_system_customers'], 
        mode='lines', 
        name='Number of Customers', 
        line=dict(color='green'),
        yaxis='y2'  # Specify secondary y-axis
    ))

    # Add trace for kwh per customer
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df['total_system_load'] / df['total_system_customers'], 
        mode='lines', 
        name='kwh / customer', 
        line=dict(color='red', dash='dash'),
        yaxis='y3'  # Specify secondary y-axis
    ))

    # Add trace for Total System Load
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df['total_system_load'], 
        mode='lines', 
        name='Total System Load', 
        line=dict(color='blue')
    ))

    # Update layout for three y-axes
    fig.update_layout(
        title='Number of Customers and Total System Load Over Time',
        xaxis_title='Datetime',
        yaxis=dict(
            title='Total System Load',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Number of Customers',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='kwh/Customer',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0, y=1),
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white'
    )

    # Show the plot
    fig.show()
    return 

def avg_kwh_per_customer(df):
    df['kwh_per_customer'] = df['total_system_load'] / df['total_system_customers']

    df = df.drop(columns=['total_system_load'])
    return df

def preprocess_features():
    load_features = get_load_features()
    load_features_tx = remove_anomalous_data(load_features, anomaly_datetimes=['2023-03-06 23:00:00'])
    
    base_features = ['datetime', 'total_system_load', 'total_system_customers',
                    'temperature_avg', 'temp_1h_ago', 'temp_2h_ago', 'temp_3h_ago', 
                    'avg_temp_last_24h']
    load_features_tx = load_features_tx[base_features]
    
    load_features_tx = handle_nulls(
       load_features_tx,
       columns_to_check=['temp_3h_ago', 'temp_2h_ago', 'temp_1h_ago', 'temperature_avg'],
       verbose=True
    )
    
    load_features_tx = handle_holidays(load_features_tx)
    load_features_tx = load_features_tx.drop(columns=['day_of_week'], errors='ignore')
    
    load_features_nn = add_periodic_time_features(load_features_tx)
    
    load_features_per_cust = avg_kwh_per_customer(load_features_nn)
    
    features = ['datetime',
        'total_system_customers',
        'day_of_year_sin', 'day_of_year_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'hour_sin', 'hour_cos',
        'temperature_avg',
        'kwh_per_customer'
    ]
    load_features_per_cust_reduced = select_features(load_features_per_cust, features)
    
    start_date = pd.Timestamp('2018-12-01')
    end_date = pd.Timestamp('2024-11-01')
    load_features_timeboxed = slice_by_date(load_features_per_cust_reduced, start_date, end_date)
    return load_features_timeboxed

def plot_correlation_matrix(df):
    """
    Create and display a correlation matrix heatmap for the given DataFrame.
    
    Args:
        df: pandas DataFrame containing the features to analyze
    """
    corr_matrix = df.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True,  # Show correlation values
                cmap='coolwarm',  # Color scheme
                vmin=-1, vmax=1,  # Correlation range
                center=0,
                xticklabels=df.columns,
                yticklabels=df.columns)

    plt.title('Feature Correlation Matrix')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def create_scalers(df, target_col, datetime_col, train_start, train_end):
    """Create and fit scalers on training data only"""
    train_data = df[(df[datetime_col] >= train_start) & (df[datetime_col] < train_end)]
    
    # Prepare features and target
    X_train = train_data.drop(columns=[datetime_col, target_col])
    y_train = train_data[target_col]
    
    # Create and fit scalers
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
    
    return scaler_X, scaler_y

def prepare_and_scale_data(df, scaler_X, scaler_y, target_col, datetime_col, start_date, end_date):
    # Get the exact features used in fitting
    feature_names = scaler_X.feature_names_in_
    
    # Select data within date range
    mask = (df[datetime_col] >= start_date) & (df[datetime_col] <= end_date)
    data = df[mask].copy()
    
    # Ensure we have exactly the same features in the same order
    X = data[feature_names]  # This will raise a clear error if any features are missing
    X_scaled = scaler_X.transform(X)
    
    # Scale target
    y = data[target_col].values.reshape(-1, 1)
    y_scaled = scaler_y.transform(y)
    
    return X_scaled, y_scaled, data.index



def make_predictions(model, X_scaled, scaler_y):
    """Make predictions and inverse transform them"""
    predictions_scaled = model.predict(X_scaled)
    return scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

def create_predictions_df(df, predictions, datetime_col, target_col, is_scenario=False):
    results_df = pd.DataFrame({
        datetime_col: df[datetime_col],
        'predicted': predictions.flatten()
    })
    
    # Only try to access actuals if this isn't a scenario
    if not is_scenario and target_col in df.columns:
        results_df['actual'] = df[target_col]

    results_df['MAPE'] = np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual']) * 100
    results_df['RMSE'] = np.sqrt((results_df['actual'] - results_df['predicted'])**2)
    results_df['MSE'] = (results_df['actual'] - results_df['predicted'])**2
    
    return results_df


def calculate_peak_day_mape(df, actual, predicted):
    """Calculate MAPE for peak load days in each year-month"""
    # Create DataFrame with datetime and values
    df = df.copy()
    df['actual'] = actual
    df['predicted'] = predicted
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    
    # Find peak load day for each year-month
    peak_days = (df.groupby(['year', 'month', 'date'])['actual']
                  .mean()  # Average load for each day
                  .reset_index()
                  .sort_values('actual', ascending=False)
                  .groupby(['year', 'month'])
                  .first()  # Get highest day for each year-month
                  .reset_index())
    
    # Calculate MAPE for each peak day
    peak_day_mapes = []
    peak_day_details = []
    
    for _, row in peak_days.iterrows():
        day_data = df[df['date'] == row['date']]
        
        # Calculate hourly MAPEs for this day
        hourly_mapes = np.abs((day_data['actual'] - day_data['predicted']) / day_data['actual']) * 100
        
        peak_day_details.append({
            'year': row['year'],
            'month': row['month'],
            'date': row['date'],
            'daily_mape': np.mean(hourly_mapes),
            'max_hourly_mape': np.max(hourly_mapes),
            'min_hourly_mape': np.min(hourly_mapes),
            'peak_load': row['actual'],
            'num_hours': len(day_data)
        })
        
        peak_day_mapes.append(np.mean(hourly_mapes))
    
    # Convert details to DataFrame for analysis
    details_df = pd.DataFrame(peak_day_details)
    
    # print("\nPeak Day Details:")
    # print(details_df.to_string())
    
    # print(f"\nOverall Peak Day MAPE: {np.mean(peak_day_mapes):.2f}%")
    # print(f"Best Peak Day MAPE: {np.min(peak_day_mapes):.2f}%")
    # print(f"Worst Peak Day MAPE: {np.max(peak_day_mapes):.2f}%")
    
    return np.mean(peak_day_mapes), details_df

def calculate_metrics(actual, predicted):
    """Calculate regression metrics"""
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    metrics = {
        'MSE': mse,
        'R2': r2,
        'MAPE': mape
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}" + ("%" if name == 'MAPE' else ""))

    # Calculate MAPE curve
    sorted_df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'abs_error': np.abs(actual - predicted),
        'mape': np.abs((actual - predicted) / actual) * 100
    }).sort_values('actual', ascending=False)
    
    mape_values = np.abs((sorted_df['actual'] - sorted_df['predicted']) / 
                        sorted_df['actual']) * 100
    rolling_mape = pd.Series(mape_values).expanding().mean()
    
    print("\nPerformance on Highest Load Hours:")
    print(f"Top 10 Hours MAPE: {sorted_df['mape'].head(10).mean():.2f}%")
    print(f"Top 50 Hours MAPE: {sorted_df['mape'].head(50).mean():.2f}%") 
    print(f"Top 100 Hours MAPE: {sorted_df['mape'].head(100).mean():.2f}%")
    print(f"Top 500 Hours MAPE: {sorted_df['mape'].head(500).mean():.2f}%")
    
    return metrics

def plot_predictions(results_df, title="Predicted vs Actual Load Over Time"):
    """Create interactive plotly plot of predictions vs actuals"""
    fig = go.Figure()
    
    # Plot actual values if they exist
    if 'actual' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['actual'],
            mode='lines',
            name='Actual Load',
            line=dict(color='blue')
        ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=results_df['datetime'],
        y=results_df['predicted'],
        mode='lines',
        name='Predicted Load',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Datetime',
        yaxis_title='kWh/Customer',
        legend=dict(x=0, y=1),
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white'
    )
    
    return fig

def analyze_high_load_cases(error_df):
    """
    Analyze error metrics for high load cases and create visualizations.
    
    Args:
        error_df: DataFrame containing 'actual', 'predicted', 'MAPE' and 'datetime' columns
    """
    error_df['MAPE'] = np.abs((error_df['actual'] - error_df['predicted']) / error_df['actual']) * 100
    error_df['RMSE'] = np.sqrt((error_df['actual'] - error_df['predicted'])**2)
    error_df['MSE'] = (error_df['actual'] - error_df['predicted'])**2

    # Avg MAPE of top 100 highest load cases
    top_100_load = error_df.nlargest(100, 'actual')
    avg_mape = top_100_load['MAPE'].mean()
    print(f"Average MAPE of top 100 highest load cases: {avg_mape:.2f}%")

    #Peak Hour Mape (17-22)
    error_df['hour'] = error_df['datetime'].dt.hour
    peak_hour_mape = error_df[(error_df['hour'] >= 17) & (error_df['hour'] < 22)]['MAPE'].mean()
    print(f"Average MAPE of peak hours (17-22): {peak_hour_mape:.2f}%")

    

    # Calculate average MAPE for top 10 highest load hours per month
    error_df['month'] = error_df['datetime'].dt.month
    
    # Group by month and get top 10 highest load hours
    monthly_peaks = (error_df.groupby('month')
                    .apply(lambda x: x.nlargest(10, 'actual'))
                    .reset_index(drop=True))
    
    # Calculate average MAPE for peak hours by month
    peak_hours_mape = (monthly_peaks.groupby('month')['MAPE']
                      .mean()
                      .reset_index())
    
    print(f"Average MAPE of top 10 peak load hours per month: {peak_hours_mape['MAPE'].mean():.2f}%")

    #plot avg mape per hour 
    plt.figure(figsize=(10, 6))
    sns.barplot(data=error_df, x='hour', y='MAPE')
    plt.title('Avg MAPE(%) per Hour')
    plt.xlabel('Hour')
    plt.ylabel('MAPE (%)')
    plt.show()

    # Plot MAPE for peak hours by month
    plt.figure(figsize=(10, 6))
    sns.barplot(data=peak_hours_mape, x='month', y='MAPE', errorbar=('ci', 68))
    plt.title('Avg MAPE(%) for Top 10 Highest Load Hours per Month')
    plt.xlabel('Month')
    plt.ylabel('MAPE (%)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()





    # Sort by actual load in descending order
    sorted_df = error_df.sort_values('actual', ascending=False).reset_index(drop=True)

    # Calculate rolling average MAPE and RMSE
    rolling_mape = sorted_df['MAPE'].expanding().mean()
    rolling_rmse = sorted_df['RMSE'].expanding().mean() / 1000000

    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot MAPE on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('N (Number of Highest Load Cases)')
    ax1.set_ylabel('Average MAPE (%)', color=color)
    ax1.plot(range(1, len(rolling_mape)+1), rolling_mape, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create secondary y-axis and plot RMSE
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Average RMSE', color=color)
    ax2.plot(range(1, len(rolling_rmse)+1), rolling_rmse, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Average MAPE(%) and RMSE(kWh/Customer) for Top N Highest Load Cases')
    plt.grid(True)
    plt.show()

    # Print extreme cases
    print("\nTop 100 highest Load cases:")
    pd.set_option('display.max_rows', None)
    print(error_df.nlargest(100, 'actual')[['datetime', 'actual', 'predicted', 'MAPE']])
    pd.set_option('display.max_rows', 10)

#Plot MAPE and Load Over Time
def plot_error_metrics_over_time(results_df, title_prefix=''):
    """
    Create separate plots for MAPE vs Load and RMSE vs Load over time
    
    Args:
        results_df: DataFrame containing 'datetime', 'actual', 'predicted' columns
        title_prefix: Optional prefix for plot titles
    """
    # Calculate error metrics
    results_df['MAPE'] = np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual']) * 100
    results_df['RMSE'] = np.sqrt((results_df['actual'] - results_df['predicted'])**2)
    
    # Plot 1: MAPE vs Load
    fig_mape = go.Figure()
    
    fig_mape.add_trace(
        go.Scatter(x=results_df['datetime'], y=results_df['MAPE'],
                  name='MAPE', line=dict(color='red'))
    )
    fig_mape.add_trace(
        go.Scatter(x=results_df['datetime'], y=results_df['actual'],
                  name='kWh/Customer', line=dict(color='blue'),
                  yaxis='y2')
    )
    
    fig_mape.update_layout(
        title=f'{title_prefix}MAPE vs kWh/Customer Over Time',
        yaxis=dict(title='MAPE (%)', side='left'),
        yaxis2=dict(title='kWh/Customer', side='right', overlaying='y'),
        height=400,
        showlegend=True
    )
    
    # Plot 2: RMSE vs Load
    fig_rmse = go.Figure()
    
    fig_rmse.add_trace(
        go.Scatter(x=results_df['datetime'], y=results_df['RMSE'],
                  name='RMSE', line=dict(color='red'))
    )
    fig_rmse.add_trace(
        go.Scatter(x=results_df['datetime'], y=results_df['actual'],
                  name='kWh/Customer', line=dict(color='blue'),
                  yaxis='y2')
    )
    
    fig_rmse.update_layout(
        title=f'{title_prefix}RMSE vs kWh/Customer Over Time',
        yaxis=dict(title='RMSE', side='left'),
        yaxis2=dict(title='Load', side='right', overlaying='y'),
        height=400,
        showlegend=True
    )
    
    return fig_mape, fig_rmse

def analyze_error_correlation(error_df):
    """
    Analyze correlation between actual load and MAPE, create visualization plots,
    and print extreme cases.
    
    Args:
        error_df: DataFrame containing 'Actual', 'Predicted', 'MAPE' and 'datetime' columns
    """
    import seaborn as sns
    
    # Calculate correlation
    correlation = error_df['actual'].corr(error_df['MAPE'])
    print(f"Correlation between Load and MAPE: {correlation:.3f}")

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=error_df, x='actual', y='MAPE')
    plt.title('MAPE vs kWh/Customer')
    plt.xlabel('kWh/Customer')
    plt.ylabel('MAPE (%)')

    # Add trend line
    sns.regplot(data=error_df, x='actual', y='MAPE', scatter=False, color='red')
    plt.show()