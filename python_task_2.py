#import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_distance_matrix(df):
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance
    for k in unique_ids:
        distance_matrix += np.where((distance_matrix == 0) & (distance_matrix.T != 0) & (distance_matrix[k] != 0),
                                    distance_matrix[:, k:k+1] + distance_matrix[k:k+1, :], 0)

    np.fill_diagonal(distance_matrix.values, 0)

    return df


def unroll_distance_matrix(df):
    unique_ids = distance_matrix.index
    unrolled_data = []

    for i, id_start in enumerate(unique_ids):
        for id_end in unique_ids[i+1:]:
            distance = distance_matrix.at[id_start, id_end]
            unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})


    return pd.DataFrame(unrolled_data)


def find_ids_within_ten_percentage_threshold(df, reference_id):
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    threshold_min, threshold_max = reference_avg_distance - (reference_avg_distance * 0.1), reference_avg_distance + (reference_avg_distance * 0.1)

    filtered_ids = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = filtered_ids[(filtered_ids['distance'] >= threshold_min) & (filtered_ids['distance'] <= threshold_max)]

    return filtered_ids.sort_values(by='id_start')

def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df.drop(columns='distance')


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.
    Args:
        df (pandas.DataFrame)
        df (pandas.DataFrame): Input dataframe with columns 'id_start', 'id_end', 'distance', 'start_time', 'end_time', 'start_day', 'end_day'.
    Returns:
        pandas.DataFrame
        pandas.DataFrame: Updated DataFrame with added columns 'start_day', 'start_time', 'end_day', 'end_time'.
    """
    # Write your logic here
    # Define time ranges and discount factors
    time_ranges = [
        {'start': time(0, 0, 0), 'end': time(10, 0, 0), 'weekday_factor': 0.8, 'weekend_factor': 0.7},
        {'start': time(10, 0, 0), 'end': time(18, 0, 0), 'weekday_factor': 1.2, 'weekend_factor': 0.7},
        {'start': time(18, 0, 0), 'end': time(23, 59, 59), 'weekday_factor': 0.8, 'weekend_factor': 0.7}
    ]

    # Apply discount factors based on time ranges and weekdays/weekends
    for time_range in time_ranges:
        mask = (df['start_time'] >= time_range['start']) & (df['end_time'] <= time_range['end'])
        df.loc[mask & ~df['start_day'].isin(['Saturday', 'Sunday']), 'distance'] *= time_range['weekday_factor']
        df.loc[mask & df['start_day'].isin(['Saturday', 'Sunday']), 'distance'] *= time_range['weekend_factor']

    return df


# Read the dataset
data = pd.read_csv('datasets\dataset-3.csv')

# Calculate distance matrix
resulting_distance_matrix = calculate_distance_matrix(data)
print("Distance Matrix result:\n", resulting_distance_matrix)

# Unroll the distance matrix
unrolled_data = unroll_distance_matrix(resulting_distance_matrix)
print("Unroll Distance Matrix result:\n", unrolled_data)


reference_value = 1001404 
resulting_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_data, reference_value)
print("Finding IDs within Percentage Threshold:\n", resulting_ids_within_threshold)

result_with_toll_rates = calculate_toll_rate(unrolled_data)
print("Calculate Toll Rate:\n", result_with_toll_rates)
 
