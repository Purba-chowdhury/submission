import pandas as pd
import numpy as np


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df = pd.read_csv("datasets/dataset-1.csv")
    car_matrix = df.pivot_table(values="car", index="id_1", columns="id_2")
    car_matrix.fillna(0, inplace=True)
    for i in range(len(car_matrix)):
       car_matrix.iloc[i, i] = 0
    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    category = ['low', 'medium', 'high']
    df['car_type'] = np.select(conditions, category, default='Unknown')

    type_counts = df['car_type'].value_counts().to_dict()

    dict = {key: type_counts[key] for key in sorted(type_counts.keys())}

    return dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    mean_bus = df["bus"].mean()
    bus_condition = df["bus"] > 2 * mean_bus
    bus_indexes = df[bus_condition].index.to_list()
    bus_indexes.sort()

    return list()


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_condition = df.groupby("route")["truck"].mean() > 7
    filtered_routes = df.loc[route_condition]["route"].to_list()
    filtered_routes.sort()

    return list()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    matrix = matrix.copy()
    matrix.loc[matrix > 20] *= 0.75
    matrix.loc[matrix <= 20] *= 1.25
    matrix.round(1)

    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df2["start_time"] = pd.to_datetime(df2[["startDay", "startTime"]].apply(lambda x: " ".join(x), axis=1))
    df2["end_time"] = pd.to_datetime(df2[["endDay", "endTime"]].apply(lambda x: " ".join(x), axis=1))
    df2["time_diff"] = df2["end_time"] - df2["start_time"]
    full_24_hour = df2['duration'].dt.total_seconds() >= 86400
    all_7_days = df2['start_datetime'].dt.dayofweek.nunique() == 7
    df2 = df2.groupby(['id', 'id_2']).apply(lambda x: all(full_24_hour.loc[x.index]) and all_7_days.loc[x.index[0]])

    return pd.Series()


#read the dataset
df = pd.read_csv("datasets/dataset-1.csv")
df2= pd.read_csv("datasets/dataset-2.csv")
result = generate_car_matrix(df)
print("Question 1:\n",result)

result = get_type_count(df)
print("Question 2:\n",result)

result = get_bus_indexes(df)
print("Question 3:\n",result)

result = filter_routes(df)
print("Question 4:\n",result)

result = multiply_matrix(result)
print("Question 5:\n",result)

result=time_check(df2)
print("Question 6:",result)