from library_ubidots import Ubidots
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sys import exit
import config as cfg

import os
from dotenv import load_dotenv
load_dotenv()

# Base configuration -----------------------------------------------------------

# Ubidots API
API_URL = cfg.API_URL
_TOKEN: str = os.environ["_token"]

LST_VAR_FIELDS = cfg.LST_VAR_FIELDS
LST_HEADERS = cfg.LST_HEADERS

# Date and Time
DATETIME_FORMAT = cfg.DATETIME_FORMAT
DATE_FORMAT = cfg.DATE_FORMAT
LOCAL_TIMEZONE = cfg.LOCAL_TIMEZONE

# Plotting parameters
confidence_interval = cfg.CONFIDENCE_INTERVAL

# General parameters
ALLOWED_DATE_OVERLAP = cfg.ALLOWED_DATE_OVERLAP
DAYS_PER_MONTH = cfg.DAYS_PER_MONTH

dct_dow = cfg.dct_dow

LST_NIGHTTIME_HOURS = cfg.lst_nighttime_hours

# Client level configuration ---------------------------------------------------

# Ubidots data parameters
device_group_label = cfg.DEVICE_GROUP_LABEL
DATA_FREQUENCY = cfg.DATA_FREQUENCY


# Plotting parameters
wide_figure_size = cfg.WIDE_FIGURE_SIZE
save_figures = cfg.SAVE_FIGURES
show_optional_figures = cfg.SHOW_OPTIONAL_FIGURES

# Cleaning parameters
clean_data = cfg.CLEAN_DATA
validate_cleaning = cfg.VALIDATE_CLEANING

SHORT_WINDOW = cfg.SHORT_WINDOW
SHORT_CONFIDENCE_INTERVAL = cfg.SHORT_CONFIDENCE_INTERVAL

LONG_WINDOW = cfg.LONG_WINDOW
LONG_CONFIDENCE_INTERVAL = cfg.LONG_CONFIDENCE_INTERVAL

SMOOTHING_METHOD = cfg.SMOOTHING_METHOD
SMOOTHING_WINDOW = cfg.SMOOTHING_WINDOW

# General parameters
cop_per_kwh = cfg.COP_PER_KWH

ACTIVE_ENERGY_LABELS = cfg.ACTIVE_ENERGY_LABELS
ACTIVE_POWER_LABELS = cfg.ACTIVE_POWER_LABELS
ACTIVE_ENERGY_LUMP_LABEL = cfg.ACTIVE_ENERGY_LUMP_LABEL
ACTIVE_POWER_LUMP_LABEL = cfg.ACTIVE_POWER_LUMP_LABEL
TOTAL_ACTIVE_ENERGY_LABEL = cfg.TOTAL_ACTIVE_ENERGY_LABEL
TOTAL_ACTIVE_POWER_LABEL = cfg.TOTAL_ACTIVE_POWER_LABEL
TOTAL_REACTIVE_ENERGY_LABEL = cfg.TOTAL_REACTIVE_ENERGY_LABEL


SUB_STR = cfg.SUB_STR

# ------------------------------------------------------------------------------


def calculate_interval_duration_days(interval):
    dt_start = pd.to_datetime(interval['start'])
    dt_end = pd.to_datetime(interval['end'])
    return (dt_end - dt_start).days + 1


def count_days(df):
    return len(np.unique(df.index.date))


def find_date_overlap(baseline, study):
    latest_start = max(
        pd.to_datetime(baseline['start']),
        pd.to_datetime(study['start'])
    )
    earliest_end = min(
        pd.to_datetime(baseline['end']),
        pd.to_datetime(study['end'])
    )
    delta = (earliest_end - latest_start).days + 1
    return int(max(0, delta))


def check_intervals(baseline, study, max_overlap=0):
    overlap = find_date_overlap(baseline, study)
    if (overlap > max_overlap):
        print(
            "Error: The baseline and study intervals must not overlap. "
            f"Current overlap is {overlap} days"
        )
        exit()


# def get_available_devices(device_group_label):
#     # The API requires a tilde leading the device group label
#     # but the user shouldn't be expected to know this
#     tilde_device_group_label = '~' + device_group_label
#     r_devices = Ubidots.get_device_group_devices(
#         _TOKEN, tilde_device_group_label)
#     dct_available_devices = dict(zip(r_devices['id'], r_devices['label']))
#     return dct_available_devices

def get_available_devices(device_group_label):
    # The API requires a tilde leading the device group label
    # but the user shouldn't be expected to know this
    tilde_device_group_label = '~' + device_group_label
    r_devices = Ubidots.get_device_group_devices(
        _TOKEN, tilde_device_group_label)
    return pd.DataFrame(r_devices)


def get_available_variables(device_id):
    if not isinstance(device_id, list):
        if isinstance(device_id, str):
            device_id = [device_id]
        else:
            device_id = list(device_id)

    return Ubidots.get_var_id_for_multiple_devices(device_id, _TOKEN)


def show_available_devices(dct_available_devices):
    print("Available devices in group:")
    print(json.dumps(dct_available_devices, sort_keys=True, indent=4))
    return dct_available_devices


def show_available_variables(obj_variables):
    print("Available variables:")
    print(json.dumps(set(obj_variables), sort_keys=True, indent=4))


def show_response_contents(df):
    print("The response contains:")
    print(json.dumps(list(df['variable'].unique()), sort_keys=True, indent=4))
    print(json.dumps(list(df['device'].unique()), sort_keys=True, indent=4))


def show_variable_labels_to_request(LST_VAR_LABELS):
    lst1 = list(set(LST_VAR_LABELS))
    lst1.sort()
    print("Variable labels to request: ")
    print(json.dumps(lst1, sort_keys=False, indent=4))


def show_outlier_counts(df):
    is_selection = (
        (df['outlier'] == True)
    )
    print("Outlier counts:")
    print(df.loc[is_selection, 'variable'].value_counts())


def ceildiv(a, b):
    return -(a // -b)

    
def make_request(VAR_IDS_TO_REQUEST, date_interval):
    # the request must be made in millisecond timestamps
    start_timestamp = str_date_to_int_timestamp_ms(
        date_interval['start'], cfg.DATE_FORMAT)
    end_timestamp = str_date_to_int_timestamp_ms(
        date_interval['end'], cfg.DATE_FORMAT)

    # The request is made with an array of all variable IDs.
    response = Ubidots.get_raw_data(
        VAR_IDS_TO_REQUEST,
        cfg.LST_VAR_FIELDS,
        start_timestamp,
        end_timestamp,
        _TOKEN,
        join=False
    )

    # The connection is left open by default
    response.close()
    return response


def str_date_to_int_timestamp_ms(date_string, date_format):
    element = datetime.strptime(date_string, date_format)
    return int(datetime.timestamp(element)) * 1000


def parse_response(lst_response, DCT_VAR_ID_TO_LABEL):
    lst_df = []
    for res in lst_response:
        df_temp = flatten_bulk_raw_response(
            res.json()['results'], cfg.LST_HEADERS)
        df_temp = parse_flat_data(df_temp, DCT_VAR_ID_TO_LABEL)

        lst_df.append(df_temp)

    return pd.concat(lst_df)


def flatten_bulk_raw_response(r_json_data, headers):
    lst_df_idx = []
    for idx in range(len(r_json_data)):
        df_idx = pd.DataFrame(r_json_data[idx], columns=headers)
        lst_df_idx.append(df_idx)

    return pd.concat(lst_df_idx).reset_index(drop=True)


def parse_flat_data(df, DCT_VAR_ID_TO_LABEL):
    # The Ubidots API does not return a variable-label field
    # and naming is inconsistent, so labels must be mapped from ids.
    df['variable'] = df['variable'].map(DCT_VAR_ID_TO_LABEL)

    # datetimes are human readable
    df["datetime"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = convert_timezone(df)

    df.drop_duplicates(
        subset=['timestamp', 'variable', 'device'], inplace=True)
    return df.drop(columns='timestamp')


def convert_timezone(obj, from_tz='utc', to_tz='America/Bogota'):
    if isinstance(obj, str):
        obj = pd.to_datetime(obj).tz_localize(from_tz).tz_convert(to_tz)
    elif isinstance(obj, datetime):
        obj.tz_localize(from_tz).tz_convert(to_tz)
    elif isinstance(obj, pd.DataFrame):
        # A DatetimeIndex must be set to allow for easy
        # timezone conversion
        obj.set_index('datetime', inplace=True)
        obj = obj.tz_localize(from_tz).tz_convert(to_tz)

    return obj


def post_process_data(df, DCT_INTERVALS_TO_DISCARD):
    # Discard first entry before cleaning as it might not belong to interval
    # sort by datetime to guarantee chronological order when dropping rows
    df.sort_values(by=['datetime', 'device', 'variable'], inplace=True)
    df = subset_drop_first_n_rows(
        df, subset_a='device', subset_b='variable', n_rows=1)

    if len(DCT_INTERVALS_TO_DISCARD) > 0:
        df = subset_discard_date_intervals(df, DCT_INTERVALS_TO_DISCARD)

    if (clean_data is True):
        # TODO: fix this line. As a temp measure we are disallowing negative active
        # power and active energy. This is not actually valid for some systems
        # which may contain solar panels, for instance.
        is_ea_or_pa = (df['variable'].isin(
            ACTIVE_ENERGY_LABELS + ACTIVE_POWER_LABELS))
        is_negative = (df['value'] < 0)
        df = df[~(is_ea_or_pa & is_negative)]

        df = double_subset_rolling_clean(
            df,
            subset_1='device',
            subset_2='variable',
            clean_on='value'
        )

    # plotting requires day of week and hour of day labels
    apply_datetime_transformations(df)
    return df


def apply_datetime_transformations(df):
    df.loc[:, 'dow'] = pd.to_datetime(df.index).dayofweek
    df.loc[:, 'dow'] = df['dow'].map(dct_dow)

    df.loc[:, 'hour'] = pd.to_datetime(df.index).hour
    df.loc[:, 'year'] = pd.to_datetime(df.index).year
    df.loc[:, 'month'] = pd.to_datetime(df.index).month
    df.loc[:, 'day'] = pd.to_datetime(df.index).day

    df.sort_values(by=['datetime', 'variable'], inplace=True)
    return df


def double_subset_rolling_clean(df, subset_a=None, subset_b=None, clean_on=None):
    lst_df = []
    for item_a in set(df[subset_a]):
        df_subset_a = df[df[subset_a] == item_a]

        # Find unique subset_B items from df_subset_a
        # to reduce the amount of empty sets
        for item_b in set(df_subset_a[subset_b]):
            df_subset = df_subset_a[df_subset_a[subset_b] == item_b].copy()

            # A primary key allows more reliable concatenation later
            df_subset['primary_key'] = range(len(df_subset))

            # long window high percentile
            is_long_outlier = rolling_percentile_outlier(
                df_subset[clean_on],
                LONG_WINDOW,
                LONG_CONFIDENCE_INTERVAL
            )
            df_subset['outlier'] = is_long_outlier

            df_long_outliers = df_subset[is_long_outlier].copy()

            # short window low percentile
            df_long_clean = df_subset[~is_long_outlier].copy()

            is_short_outlier = rolling_percentile_outlier(
                df_long_clean[clean_on],
                SHORT_WINDOW,
                SHORT_CONFIDENCE_INTERVAL
            )

            df_long_clean['outlier'] = is_short_outlier

            df_clean = pd.concat([df_long_outliers, df_long_clean])
            df_clean.sort_values(
                by=[df_clean.index.name, subset_a, subset_b], inplace=True)
            df_clean.drop(columns='primary_key', inplace=True)

            is_outlier = (df_clean['outlier'] == True)

            if (SMOOTHING_METHOD == 'median'):
                df_clean.loc[~is_outlier, clean_on] = df_clean.loc[~is_outlier, clean_on].rolling(
                    window=SMOOTHING_WINDOW, center=True).median()
            elif (SMOOTHING_METHOD == 'mean'):
                df_clean.loc[~is_outlier, clean_on] = df_clean.loc[~is_outlier, clean_on].rolling(
                    window=SMOOTHING_WINDOW, center=True).mean()

            lst_df.append(df_clean)

    return pd.concat(lst_df)


def rolling_percentile_outlier(series, window, confidence_interval):
    upper_quantile = (1 + confidence_interval/100)/2
    lower_quantile = 1 - upper_quantile

    s_upper_percentile = series.rolling(window=window, center=True).quantile(
        quantile=upper_quantile,
        axis=0,
        numeric_only=True,
        interpolation='linear'
    )

    s_lower_percentile = series.rolling(window=window, center=True).quantile(
        quantile=lower_quantile,
        axis=0,
        numeric_only=True,
        interpolation='linear'
    )

    is_outlier = (
        (series > s_upper_percentile)
        | (series < s_lower_percentile)
    )

    return is_outlier


def subset_discard_date_intervals(df, DCT_INTERVALS):
    for device in DCT_INTERVALS.keys():
        lst_intervals_dev = DCT_INTERVALS[device]
        if len(lst_intervals_dev) > 0:
            for interval in lst_intervals_dev:
                is_outside_range = (
                    (df.index < interval[0])
                    | (df.index > interval[1])
                )
                df = df[is_outside_range]

    return df

# def run_cleaning_analysis(df, variable=None, start_date=None, end_date=None, bins=None, wide_figsize=(30,10), square_figsize=(10,10)):
#     device_name = df['device_name'][0]

#     is_sel_1 = (
#         (df['variable']==variable)
#         & (df.index>start_date)
#         & (df.index<end_date)
#     )

#     is_sel_2 = (
#         (df['outlier']==False)
#         & (df['variable']==variable)
#         & (df.index>start_date)
#         & (df.index<end_date)
#     )

#     s_1 = df.loc[is_sel_1, 'value']
#     s_2 = df.loc[is_sel_2, 'value']

#     s_res = s_1 - s_2
#     lst_series = [
#         s_1,
#         s_2
#     ]

#     plot_list_series(lst_series, device_name, wide_figsize, draw_markers=False)
#     plot_list_series([s_res], device_name, wide_figsize, draw_markers=False)

#     plt.figure(figsize=square_figsize)
#     plt.scatter(s_res.index, s_res)
#     plt.show()

#     s_res.hist(bins=bins, figsize=square_figsize)


def split_into_baseline_and_study(df, baseline_interval, study_interval):
    # Slicing on non-monotonic indexes is deprecated so sorting is a must
    if not df.index.is_monotonic_increasing:
        df.sort_values(by=[df.index.name, 'device', 'variable'], inplace=True)

    df_baseline = df[baseline_interval['start']:baseline_interval['end']]
    df_study = df[study_interval['start']:study_interval['end']]
    return df_baseline, df_study


def split_date_intervals_by_device(df, df_intervals):
    # Slicing on non-monotonic indexes is deprecated so sorting is a must
    if not df.index.is_monotonic_increasing:
        df.sort_values(by=[df.index.name, 'device', 'variable'], inplace=True)

    lst_df_bl = []
    lst_df_st = []
    for device in df['device'].unique():
        df_device = df[df['device'] == device]

        bl_start = df_intervals.loc[device, 'bl_start']
        bl_end = df_intervals.loc[device, 'bl_end']
        st_start = df_intervals.loc[device, 'st_start']
        st_end = df_intervals.loc[device, 'st_end']

        lst_df_bl.append(df_device[bl_start:bl_end].copy())
        lst_df_st.append(df_device[st_start:st_end].copy())

    return pd.concat(lst_df_bl), pd.concat(lst_df_st)

# def discard_date_intervals(df, discard_date_interval):
#     for interval in discard_date_interval:
#         is_outside_range = (
#             (df.index < interval[0])
#             | (df.index > interval[1])
#         )
#     return df[is_outside_range].copy()


def pareto(df, new_label=None, method='elbow'):
    """
    pick method='elbow' to maximize info/complexity ratio
    pick method='hierarchical' to guarantee lumped is smaller than smallest primary variable
    """
    s_total = df['value'].groupby(df['variable']).sum()
    # Identify main variables
    if (method == 'elbow'):
        s_total.sort_values(ascending=False, inplace=True)
        s_delta = (s_total / s_total.sum()) * 100
        elbow = s_delta.diff().astype(float).idxmin()
        idx = s_delta.index.get_loc(elbow)
        lst_main_labels = list(s_delta.iloc[:idx+1].index)
    elif (method == 'hierarchical'):
        None

    # make hue order before changing names
    hue_order = make_hue_order(s_delta, lst_main_labels, new_label)
    return s_total, hue_order, lst_main_labels


def new_pareto(df, new_label=None, method='elbow'):
    df_pareto = df.groupby(by='variable').agg({'value': np.sum})


def plot_pareto(series):
    s_2 = series.cumsum() / series.sum()*100

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(series.index, series, color="C0")

    ax2.plot(series.index, s_2, color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(ymin=0)

    ax1.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    plt.show()


def make_hue_order(s_delta, lst_main_labels, new_label):
    is_main_var = s_delta.index.isin(lst_main_labels)
    s_sorter = s_delta[is_main_var]
    s_sorter[new_label] = s_delta[~is_main_var].sum()
    s_sorter.sort_values(ascending=False, inplace=True)
    return list(s_sorter.index)


def assemble_aggregator(lst_non_value_cols, value_method, bulk_method):
    # make sure keys start with 'value'
    lst_non_value_cols.remove('value')

    lst_keys = ['value']
    lst_keys.extend(lst_non_value_cols)

    # assign value_method to value, bulk_method to rest
    lst_values = [value_method]
    lst_values.extend([bulk_method]*len(lst_non_value_cols))

    return dict(zip(lst_keys, lst_values))


def subset_group_by(df, agg_func=None, subset=None, group_by=None):
    lst_df = []
    for label in df[subset].unique():
        is_selection = (df[subset] == label)
        df_temp = df[is_selection].groupby(group_by).agg(agg_func)
        df_temp.drop(columns=group_by, inplace=True)
        lst_df.append(df_temp)

    return pd.concat(lst_df, ignore_index=False)


def subset_resample(df, agg_func=None, subset=None, freq=None):
    lst_df = []
    for label in df[subset].unique():
        is_selection = (df[subset] == label)
        df_temp = df[is_selection].resample(freq).agg(agg_func)
        lst_df.append(df_temp)

    return pd.concat(lst_df, ignore_index=False).dropna(how='all')


def subset_drop_first_n_rows(df, subset_a=None, subset_b=None, n_rows=None):
    lst_df = []
    set_a = set(df[subset_a])
    set_b = set(df[subset_b])
    for item_1 in set_a:
        for item_2 in set_b:
            is_selection = (
                (df[subset_a] == item_1)
                & (df[subset_b] == item_2)
            )
            lst_df.append(df[is_selection].iloc[n_rows:, :])
    return pd.concat(lst_df, ignore_index=False)


def lump_secondary_variables(df, lst_main_labels, agg_func=None, new_label=None, by=None):
    """"
    To avoid using the term "grouped" which might imply a groupby operation.
    Lumped variables are termed "others" and represent a smaller quatity than main variables.
    """
    # TODO: use by= to loop through devices
    # replace secondary variable labels
    is_main_var = df['variable'].isin(lst_main_labels)
    df.loc[~is_main_var, 'variable'] = new_label

    lst_df = []
    for variable in df['variable'].unique():
        df_sel = df[df['variable'] == variable]

        # index must be unique per variable
        df_output = df_sel.groupby(df_sel.index).agg(agg_func)
        lst_df.append(df_output)

    return pd.concat(lst_df, ignore_index=False)

# def find_consumption_delta_per_variable(df_bl, df_st, merge_on=None):
#     """
#     Negative means a decrease in energy consumption
#     """
#     df = pd.merge(df_bl, df_st, on=merge_on)
#     df['delta'] = df['value_y'] - df['value_x']
#     return df[['delta', 'variable']]

# def subplots_stack(df1, df2, figsize):
#     df1_wide = df1.pivot(index=None, columns='variable', values='value')
#     df2_wide = df2.pivot(index=None, columns='variable', values='value')

#     f, (ax, bx) = plt.subplots(1,2,sharey=True) # like mine
#     ax.stackplot(df1_wide.index, df1_wide.T, labels=list(df1_wide.columns))
#     bx.stackplot(df2_wide.index, df2_wide.T, labels=list(df2_wide.columns))
#     plt.legend()
#     plt.show()


def get_coords_dataframe(LST_DEVICE_ID_TO_REQUEST):
    # TODO: add device label as "device"
    df_coords = Ubidots.get_gps_for_multiple_device_id(
        LST_DEVICE_ID_TO_REQUEST, _TOKEN)
    # df_coords.set_index('device_name', drop=True, inplace=True)
    return df_coords


def center_colombia():
    golden_ratio = 1.618

    northmost_point = [12.458838209852894, -71.6681525371038]
    southmost_point = [-4.21322948919576, -69.94291862660366]
    eastmost_point = [1.1900937452541394, -66.83834477921218]
    westmost_point = [1.7843244901198274, -78.7878992714576]

    np_corner_lat = np.array(
        [
            northmost_point[0],
            northmost_point[0],
            southmost_point[0],
            southmost_point[0],
        ]
    )

    np_corner_lon = np.array(
        [
            eastmost_point[1],
            westmost_point[1],
            westmost_point[1],
            eastmost_point[1],
        ]
    )

    mean_lat = np.mean(np_corner_lat)
    mean_lon = np.mean(np_corner_lon)

    np_centered_lat = np_corner_lat - mean_lat
    np_centered_lon = np_corner_lon - mean_lon

    np_shifted_lat = np_centered_lat * golden_ratio + mean_lat
    np_shifted_lon = np_centered_lon * golden_ratio + mean_lon

    lataxis_range = [np_shifted_lat[-1], np_shifted_lat[0]]
    lonaxis_range = [np_shifted_lon[0], np_shifted_lon[1]]
    return lataxis_range, lonaxis_range


############################################################
# Our utilization metric will be a duty cycle
# , except it will be expressed as hours at a
# reference power; where the reference power
# is the 95th percentile of recorded power
# for the given period.
# This metric is not meant to be absolute
# but relative; to compare relative utilizaton
# between periods.


def reference_power_per_device(df, variable=None):
    is_variable = (df['variable'] == variable)
    return df[is_variable].groupby(by=['device', 'device_name']).agg({'value': lambda x: np.percentile(x, q=cfg.REFERENCE_POWER_PERCENTILE)}).reset_index()


def daily_consumption_per_device(df, variable=None):
    # if datetime index and date doesn't exist
    if (isinstance(df.index, pd.DatetimeIndex) & ('date' not in df.columns)):
        df['date'] = df.index.date

    is_variable = (df['variable'] == variable)
    return df[is_variable].groupby(by=['device', 'device_name', 'date']).agg({'value': np.sum}).reset_index()


def utilization_per_device(df_daily_cons, df_ref_power):
    df_kpi = df_daily_cons.copy()
    df_kpi['reference_power'] = df_kpi['device'].map(
        dict(zip(df_ref_power['device'], df_ref_power['value'])))
    df_kpi['utilization_hours'] = df_kpi['value'] / df_kpi['reference_power']
    df_kpi['duty_cycle'] = df_kpi['utilization_hours'] / 24
    df_kpi.rename(columns={'value': 'consumption'}, inplace=True)
    return df_kpi


def aggregate_all_active_consumption_by_hour(df):
    """
    For use on single site with circuit-per-device arrangements.
    """
    # Consumption can be summed across devices and across time
    # So no need to use device or variable subsets.
    is_variable = (df['variable'].isin(ACTIVE_ENERGY_LABELS))
    return df[is_variable].resample('1h').agg({'value': np.sum})


def aggregate_all_active_consumption(df):
    """
    For use on single site with circuit-per-device arrangements.
    """
    # Consumption can be summed across devices and across time
    # So no need to use device or variable subsets.
    is_variable = (df['variable'].isin(ACTIVE_ENERGY_LABELS))
    agg_func = {'value': np.mean, 'outlier': np.sum}
    return df[is_variable].resample(DATA_FREQUENCY).agg(agg_func)


def aggregate_all_active_power(df):
    """
    For use on single site with circuit-per-device arrangements.
    """
    # Power is instantaneous, not cummulative, so it can only
    # be summed across variables and devices where times coincide.
    # Because reported power is likely an interval average
    # and not an near-instantaneous reading, it is OK to
    # shift it a little so the datetimes align. Resampling with
    # the data frequency is an easy way to do this.

    # Make sure only power variables are being aggregated.
    # Include outliers since this method will mainly be used
    # to calculate the reference power; outliers will be dealt
    # with further down the processing path.
    is_variable = (df['variable'].isin(ACTIVE_POWER_LABELS))

    # Don't mix devices and variables while resampling
    group_cols = ['device', 'device_name', 'variable']

    # Resampling must be done with a mean, not a sum
    # (see explanation about power aggregation above)
    agg_func_1 = {'value': np.mean, 'outlier': np.sum}

    df_resampled = df[is_variable].groupby(by=group_cols).resample(
        DATA_FREQUENCY).agg(agg_func_1).reset_index()

    # One resampled we can sum all (roughly) simultaneous
    # power readings acrross devices and variables
    agg_func_2 = {'value': np.sum, 'outlier': np.sum}
    return df_resampled.groupby('datetime').agg(agg_func_2)


def find_data_period_mode(df):
    """unfinished"""
    df_mode = df.reset_index().groupby(by=['device', 'variable']).agg(
        {'datetime': lambda x: stats.mode(np.diff(x))}
    ).reset_index()

    return None


def differentiate_single_variable(df, new_var_name, remove_gap_data=False):
    lst_df = []
    for device in set(df['device']):
        df_sel = df.query("device == @device").copy()
        if (~df_sel.index.is_monotonic_increasing):
            df_sel = df_sel.sort_values(by='datetime')

        hour_deltas = df_sel.index.to_series().diff() / np.timedelta64(1, 'h')
        interval_mean_power = df_sel['value'].diff() / hour_deltas

        df_mono = pd.DataFrame()
        if (remove_gap_data is True):
            data_rate = hour_deltas.mode()[0]
            is_indeterminate = (hour_deltas > data_rate)
            df_mono['value'] = interval_mean_power[~is_indeterminate]
        else:
            df_mono['value'] = interval_mean_power

        df_mono['variable'] = new_var_name
        df_mono['device'] = device

        lst_df.append(df_mono)

    return pd.concat(lst_df)


def mean_power_from_energy(s_energy_id):
    s_delta_kWh = s_energy_id.diff()
    s_delta_hours = s_energy_id.index.to_series().diff() / np.timedelta64(1, 'h')
    return s_delta_kWh / s_delta_hours


def find_period_mean_power(df, target_vars):
    df_output = pd.DataFrame()
    idx = 0
    for device in set(df['device']):
        for variable in set(target_vars):
            for month in set(df['month']):
                is_sel = (
                    (df['device'] == device)
                    & (df['variable'] == variable)
                    & (df['month'] == month)
                )

                df_sel = df[is_sel].copy()

                start_date = df_sel.index.min()
                end_date = df_sel.index.max()
                delta_time = (end_date - start_date) / np.timedelta64(1, 'h')
                delta_energy = df_sel['value'].max() - df_sel['value'].min()
                mean_power = delta_energy / delta_time

                # assumes device names are correctly mapped to labels
                # and are unique per label
                df_output.loc[idx, 'device_name'] = df_sel['device_name'][0]
                df_output.loc[idx, 'device'] = device
                df_output.loc[idx, 'month'] = month
                df_output.loc[idx, 'start_date'] = start_date
                df_output.loc[idx, 'end_date'] = end_date
                df_output.loc[idx, 'mean_power'] = mean_power

                idx += 1

    return df_output
