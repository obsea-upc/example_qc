"""
This file contains useful operations with datasats, such as open, convert, purge, etc...

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
"""

import sys
import gc
from m3.qc import qc_flags
import time
import pandas as pd
import numpy as np
from datetime import datetime
from rich.progress import Progress
import rich
from .common import get_dataframe_precision, multiprocess, log_to_lin, lin_to_log
from matplotlib import pyplot as plt


def open_dataset(csv_file, time_range=[], format=True):
    """
    Opens a CSV datasets and arranges it to be processed and inserted
    :param csv_file: CSV file to process
    :param time_format: format of the timestamp
    :param time_range: list of two timestamps used to slice the input dataset
    :return: dataframe with the dataset
    """
    init = time.time()
    pinit = init
    print("Loading dataset %s..." % csv_file)
    df = pd.read_csv(csv_file)
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})  # rename first column to timestamp

    print("done (%.02f seconds)" % (time.time() - pinit))
    print("Indexing and formatting timestamp...")
    pinit = time.time()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    if format:
        df = df.sort_index(ascending=True)
        print("done (%.02f seconds)" % (time.time() - pinit))
        print("Dropping NaNs...")
        df = df.dropna(how="all")
        print("Keeping only observations from 2000-01-01T00:00:00z onwards...")
        try:
            df = df["2000-01-01":]
        except Exception as e:
            print("Got exception %s" % e)
            pass
        print("done (%.02f seconds)" % (time.time() - pinit))

    if type(time_range) is str and time_range:
        time_range = time_range.split("/")

    if time_range:
        print("Selecting time range", time_range[0], time_range[1])
        df = df[time_range[0]:time_range[1]]
    print("Loading dataset took %.02f seconds" % (time.time() - init))

    for var in df.columns:
        if var.endswith("_qc"):
            df[var] = df[var].replace(np.nan, -1)
            df[var] = df[var].astype(np.int8)  # set to integer
            df[var] = df[var].replace(-1, np.nan)

    return df


def delete_vars_from_df(df, vars: list):
    # ignore (delete) variables in list
    for var in vars:
        # use it as a column name
        found = False
        if var in df.columns:
            found = True
            rich.print(f"[yellow]Variable {var} won't be resampled")
            del df[var]

        if not found:  # maybe it's a prefix, delete everything starting like var
            for v in [col for col in df.columns if col.startswith(var)]:
                rich.print(f"[yellow]Ignoring variable {v}")
                del df[v]  # delete it
    return df


def resample_dataframe(df, average_period="30min", std_column=True, log_vars=[], ignore=[]):
    """
    Resamples incoming dataframe, performing the arithmetic mean. If QC control is applied select only values with
    "good data" flag (1). If no values with good data found in a data segment, try to average suspicious data. Data
    segments with only one value will be ignored (no std can be computed).
    :param df: input dataframe
    :param average_period: average period (e.g. 1h, 15 min, etc.)
    :param std_colum: if True creates a columns with the standard deviation for each variable
    :return: averaged dataframe
    """
    df = df.copy()
    df = delete_vars_from_df(df, ignore)

    # Get the current precision column by column
    precisions = get_dataframe_precision(df)
    if log_vars:
        print("converting the following variables from log to lin:", log_vars)
        df = log_to_lin(df, log_vars)

    resampled_dataframes = []
    # print("Splitting input dataframe to a dataframe for each variable")
    for var in df.columns:
        if var.endswith("_qc"):  # ignore qc variables
            continue

        var_qc = var + "_qc"
        df_var = df[var].to_frame()
        df_var[var_qc] = df[var_qc]

        # Check if QC has been applied to this variable
        if max(df[var_qc].values) == min(df[var_qc].values) == 2:
            rich.print(f"[yellow]QC not applied to {var}")
            rdf = df_var.resample(average_period).mean().dropna(how="any")
            rdf[var_qc] = rdf[var_qc].fillna(0).astype(np.int8)

        else:
            var_qc = var + "_qc"
            # Generate a dataframe for good, suspicious and bad data
            df_good = df_var[df_var[var_qc] == qc_flags["good"]]
            df_na = df_var[df_var[var_qc] == qc_flags["not_applied"]]
            df_suspicious = df_var[df_var[var_qc] == qc_flags["suspicious"]]

            # Resample all dataframes independently to avoid averaging good data with bad data and drop all n/a records
            rdf = df_good.resample(average_period).mean().dropna(how="any")
            rdf_na = df_na.resample(average_period).mean().dropna(how="any")
            #  rdf["good_count"] = df_good[var_qc].resample(average_period).count().dropna(how="any")
            rdf[var_qc] = rdf[var_qc].astype(np.int8)

            rdf_suspicious = df_suspicious.resample(average_period).mean().dropna(how="any")
            rdf_suspicious[var_qc] = rdf_suspicious[var_qc].astype(np.int8)

            if std_column:
                rdf_std = df_good.resample(average_period).std().dropna(how="any")
                rdf_suspicious_std = df_suspicious.resample(average_period).std().dropna(how="any")
                rdf_na_std = rdf_na.resample(average_period).std().dropna(how="any")

            # Join good and suspicious data
            rdf = rdf.join(rdf_na, how="outer", rsuffix="_na")
            rdf = rdf.join(rdf_suspicious, how="outer", rsuffix="_suspicious")

            # Fill n/a in QC variable with 0s and convert it to int8 (after .mean() it was float)
            rdf[var_qc] = rdf[var_qc].fillna(0).astype(np.int8)

            # Select all lines where there wasn't any good data (missing good data -> qc = 0)
            i = 0
            for index, row in rdf.loc[rdf[var_qc] == 0].iterrows():
                i += 1
                if not np.isnan(row[var + "_na"]):
                    # modify values at resampled dataframe (rdf)
                    rdf.at[index, var] = row[var + "_na"]
                    rdf.at[index, var_qc] = qc_flags["not_applied"]

                if not np.isnan(row[var + "_suspicious"]):
                    # modify values at resampled dataframe (rdf)
                    rdf.at[index, var] = row[var + "_suspicious"]
                    rdf.at[index, var_qc] = qc_flags["suspicious"]

            # Calculate standard deviations
            if std_column:
                var_std = var + "_std"
                rdf[var_std] = 0

                # assign good data stdev where qc = 1
                rdf.loc[rdf[var_qc] == qc_flags["good"], var_std] = rdf_std[var]
                # assign data with QC not applied
                if not rdf.loc[rdf[var_qc] == qc_flags["not_applied"]].empty:
                    rdf.loc[rdf[var_qc] == qc_flags["not_applied"], var_std] = rdf_na_std[var]
                # assign suspicious stdev where qc = 3
                if not rdf.loc[rdf[var_qc] == qc_flags["suspicious"]].empty:
                    rdf.loc[rdf[var_qc] == qc_flags["suspicious"], var_std] = rdf_suspicious_std[var]

                del rdf_std
                del rdf_suspicious_std
                del rdf_na_std
                gc.collect()

            # delete suspicious and bad columns
            del rdf[var + "_na"]
            del rdf[var + "_suspicious"]
            del rdf[var + "_qc_na"]
            del rdf[var + "_qc_suspicious"]

            # delete all unused dataframes
            del df_var
            del rdf_na
            del rdf_suspicious
            del df_good
            del df_suspicious

            gc.collect()  # try to free some memory with garbage collector
        rdf = rdf.dropna(how="any")  # make sure that each data point has an associated qc and stdev

        # append dataframe
        resampled_dataframes.append(rdf)

    # Join all dataframes
    df_out = resampled_dataframes[0]
    for i in range(1, len(resampled_dataframes)):
        merge_df = resampled_dataframes[i]
        df_out = df_out.join(merge_df, how="outer")

    df_out.dropna(how="all", inplace=True)

    if log_vars:  # Convert back to linear
        print("converting back to log", log_vars)
        df_out = lin_to_log(df_out, log_vars)

    # Apply the precision
    for colname, precision in precisions.items():
        df_out[colname] = df_out[colname].round(decimals=precision)

    # Set the precisions for the standard deviations (precision + 2)
    for colname, precision in precisions.items():
        std_var = colname + "_std"
        if std_var in df_out.keys():
            df_out[std_var] = df_out[std_var].round(decimals=(precision+2))

        if colname in log_vars:  # If the variable is logarithmic it makes no sense calculating the standard deviation
            del df_out[std_var]

    return df_out


def resample_polar_dataframe(df, magnitude_label, angle_label, units="degrees", average_period="30min", log_vars=[],
                             ignore=[]):
    """
    Resamples a polar dataset. First converts from polar to cartesian, then the dataset is resampled using the
    resample_dataset function, then the resampled dataset is converted back to polar coordinates.
    :param df: input dataframe
    :param magnitude_label: magnitude column label
    :param angle_label: angle column label
    :param units: angle units 'degrees' or 'radians'
    :param average_period:
    :return: average period (e.g. 1h, 15 min, etc.)
    """
    df = df.copy()
    df = delete_vars_from_df(df, ignore)

    # column_order = df.columns  # keep the column order

    columns = [col for col in df.columns if not col.endswith("_qc")]  # get columns names (no qc vars)
    column_order = []
    for col in columns:
        column_order += [col, col + "_qc", col + "_std"]

    precisions = get_dataframe_precision(df)

    # replace 0s with a tiny number to avoid 0 angle when module is 0
    almost_zero = 10 **(-int((precisions[magnitude_label] + 2)))  # a number 100 smaller than the variable precision
    df[magnitude_label] = df[magnitude_label].replace(0, almost_zero)

    # Calculate sin and cos of the angle to allow resampling
    expanded_df = expand_angle_mean(df, magnitude_label, angle_label, units=units)
    # Convert from polar to cartesian
    cartesian_df = polar_to_cartesian(expanded_df, magnitude_label, angle_label, units=units, x_label="x", y_label="y",
                                      delete=False)
    # Resample polar dataframe
    resampled_cartesian_df = resample_dataframe(cartesian_df, average_period=average_period, std_column=True,
                                                log_vars=log_vars, ignore=ignore)
    # re-calculate the angle (use previously calculated sin / cos values)
    resampled_cartesian_df = calculate_angle_mean(resampled_cartesian_df, angle_label, units=units)
    # convert to polar
    resampled_df = cartesian_to_polar(resampled_cartesian_df, "x", "y", units=units, magnitude_label=magnitude_label,
                                      angle_label=angle_label)

    resampled_df = resampled_df.reindex(columns=column_order)  # reindex columns

    # Convert all QC flags to int8
    for c in resampled_df.columns:
        if c.endswith("_qc"):
            resampled_df[c] = resampled_df[c].replace(np.nan, -1)
            resampled_df[c] = resampled_df[c].astype(np.int8)
            resampled_df[c] = resampled_df[c].replace(-1, np.nan)

    # apply precision
    for c, precision in precisions.items():
        resampled_df[c] = resampled_df[c].round(decimals=precision)

    # delete STD for angle column
    if angle_label + "_std" in resampled_df.columns:
        del resampled_df[angle_label + "_std"]

    return resampled_df


def polar_to_cartesian(df, magnitude_label, angle_label, units="degrees", x_label="X", y_label="y", delete=True):
    """
    Converts a dataframe from polar (magnitude, angle) to cartesian (X, Y)
    :param df: input dataframe
    :param magnitude_label: magnitude's name in dataframe
    :param angle_label: angle's name in dataframe
    :param units: angle units, it must be "degrees" or "radians"
    :param x_label: label for the cartesian x values (defaults to X)
    :param y_label: label for the cartesian y values ( defaults to Y)
    :param delete: if True (default) deletes old polar columns
    :return: dataframe expanded with north and east vectors
    """
    magnitudes = df[magnitude_label].values
    angles = df[angle_label].values
    if units == "degrees":
        angles = np.deg2rad(angles)

    x = abs(magnitudes) * np.sin(angles)
    y = abs(magnitudes) * np.cos(angles)
    df[x_label] = x
    df[y_label] = y

    # Delete old polar columns
    if delete:
        del df[magnitude_label]
        del df[angle_label]

    if magnitude_label + "_qc" in df.columns:
        # If quality control has been applied to the dataframe, add also qc to the cartesian dataframe
        # Get greater QC flag, so both x and y have the most restrictive quality flag in magnitudes / angles
        df[x_label + "_qc"] = np.maximum(df[magnitude_label + "_qc"].values, df[angle_label + "_qc"].values)
        df[y_label + "_qc"] = df[x_label + "_qc"].values
        # Delete old qc flags
        if delete:
            del df[magnitude_label + "_qc"]
            del df[angle_label + "_qc"]

    return df


def cartesian_to_polar(df, x_label, y_label, units="degrees", magnitude_label="magnitude", angle_label="angle",
                       delete=True):
    """
    Converts a dataframe from cartesian (X, Y) to polar (magnitud, angle)
    :param df: input dataframe
    :param x_label: X column name (input)
    :param y_label: Y column name (input)
    :param magnitude_label: magnitude column name (output)
    :param angle_label: angle column name (output)
    :param units: angle units, it must be "degrees" or "radians"
    :param delete: if True (default) deletes old cartesian columns
    :return: dataframe expanded with north and east vectors
    """
    x = df[x_label].values
    y = df[y_label].values
    magnitudes = np.sqrt((np.power(x, 2) + np.power(y, 2)))  # magnitude is the module of the x,y vector
    angles = np.arctan2(x, y)
    angles[angles < 0] += 2 * np.pi  # change the range from -pi:pi to 0:2pi

    if units == "degrees":
        angles = np.rad2deg(angles)

    df[magnitude_label] = magnitudes
    df[angle_label] = angles

    if delete:
        del df[x_label]
        del df[y_label]
    if x_label + "_qc" in df.columns:
        # If quality control has been applied to the dataframe, add also qc to the cartesian dataframe
        # Get greater QC flag, so both x and y have the most restrictive quality flag in magnitudes / angles
        df[magnitude_label + "_qc"] = np.maximum(df[y_label + "_qc"].values, df[x_label + "_qc"].values)
        df[angle_label + "_qc"] = df[x_label + "_qc"].values
        # Delete old qc flags
        del df[x_label + "_qc"]
        del df[y_label + "_qc"]
    return df


def average_angles(magnitude, angle):
    """
    Averages
    :param wind:
    :param angle:
    :return:
    """
    u = []
    v = []
    for i in range(len(magnitude)):
        u.append(np.sin(angle[i] * np.pi / 180.) * magnitude[i])
        v.append(np.cos(angle[i] * np.pi / 180.) * magnitude[i])
    u_mean = np.nanmean(u)
    v_mean = np.nanmean(v)
    angle_mean = np.arctan2(u_mean, v_mean)
    angle_mean_deg = angle_mean * 180 / np.pi  # angle final en degreee
    return angle_mean_deg


def expand_angle_mean(df, magnitude,  angle, units="degrees"):
    """
    Expands a dataframe with <angle>_sin and <angle>_cos. This
    :param df: input dataframe
    :param magnitude: magnitude columns name
    :param angle: angle column name
    :param units: "degrees" or "radians"
    :return: expanded dataframe
    """
    if units in ["degrees", "deg"]:
        angle_rad = np.deg2rad(df[angle].values)
    elif units in ["radians" or "rad"]:
        angle_rad = df[angle].values
    else:
        raise ValueError("Unkonwn units %s" % units)

    df[angle + "_sin"] = np.sin(angle_rad)*df[magnitude].values
    df[angle + "_cos"] = np.cos(angle_rad)*df[magnitude].values

    if magnitude + "_qc" in df.columns:
        # genearte a qc column with the MAX from magnitude qc and angle qc
        df[angle + "_sin" + "_qc"] = np.maximum(df[angle + "_qc"].values, df[magnitude + "_qc"].values)
        df[angle + "_cos" + "_qc"] = df[angle + "_sin" + "_qc"].values

    return df


def calculate_angle_mean(df, angle, units="degrees"):
    """
    This function calculates the mean of an angle (expand_angle_mean should be called first).
    :param df: input df (expanded with expand angle mean)
    :param angle: angle column name
    :param units: "degrees" or "radians"
    :return:
    """
    sin = df[angle + "_sin"].values
    cos = df[angle + "_cos"].values
    df[angle] = np.arctan2(sin, cos)
    if units in ["degrees", "deg"]:
        df[angle] = np.rad2deg(df[angle].values)

    del df[angle + "_sin"]
    del df[angle + "_cos"]
    return df


def purge_dataframe(df, deployment_history: list):
    """
    Takes the deployment history of a sensor and drops all observations outside the deployment periods
    :param df: dataframe
    :param deployment_history: list of dicts with the info of each deployment
    :return: purged dataset
    """

    print("Erasing data acquired outside deployment periods")
    __deployed_status = ["now", ""]  # keywords to detect that a instrument is already deployed (empty end time or "")
    purge_start = ["1980-01-01T00:00:00"]  # purge all measurements before a deployment
    purge_end = []

    for deployment in deployment_history:
        deployment_start = deployment["timePeriod"][0]
        deployment_end = deployment["timePeriod"][1]

        # erase timezone (if any)
        deployment_start = deployment_start.replace("Z", "").replace("z", "")
        deployment_end = deployment_end.replace("Z", "").replace("z", "")

        purge_start.append(deployment_end)
        purge_end.append(deployment_start)

    # if the sensor us currenty deployed
    if purge_start[-1] in __deployed_status:
        # ignore the last timeperiod
        purge_start = purge_start[:-1]
    else:
        # if it is not deployed, purge until now
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # string with current time
        purge_end.append(now)

    with Progress() as progress:
        task = progress.add_task("[cyan]Purging dataframes...", total=len(purge_start))
        for i in range(0, len(purge_start)):
            start = purge_start[i]
            end = purge_end[i]
            start_datetime = pd.to_datetime(start)
            end_datetime = pd.to_datetime(end)
            if end_datetime < df.index.values[0] or start_datetime > df.index.values[-1]:
                rich.print("    [blue]Ignoring interval %s and %s..." % (start, end))
                pass  # period outside dataset, skip it
            else:
                rich.print("    [cyan]Drop data between %s and %s..." % (start, end))
                df = df.drop(df[start:end].index)
            progress.update(task, advance=1)
    return df


def drop_duplicated_indexes(df, store_dup=""):
    """
    Drops duplicated data points. If an index
    :param df:
    :param save: if set, the duplicated indexes will be stored in this file
    :return:
    """
    dup_idx = df[df.index.duplicated(keep=False)]
    total_idx = len(df.index.values)
    if len(dup_idx.index) > 0:
        print("Found %d duplicated entries (%.04f %%)" % (len(dup_idx.index), 100 * len(dup_idx) / total_idx))
        if store_dup:
            rich.print("[cyan]Storing a copy of duplicated indexes at %s" % store_dup)
            dup_idx.to_csv(store_dup)
        print("Dropping duplicate entries...")
        df = df.drop(dup_idx.index)
    return df


def slice_dataframes(df, max_rows=-1, frequency=""):
    """
    Slices input dataframe into multiple dataframe, making sure than every dataframe has at most "max_rows"
    :param df: input dataframe
    :param max_rows: max rows en every dataframe
    :param frequency: M for month, W for week, etc.
    :return: list with dataframes
    """
    if max_rows < 0 and not frequency:
        raise ValueError("Specify max rows or a frequency")

    if max_rows > 0:
        if len(df.index.values) > max_rows:
            length = len(df.index.values)
            i = 0
            dataframes = []
            with Progress() as progress:
                task = progress.add_task("slicing dataframes...", total=length)
                while i + max_rows < length:
                    init = i
                    end = i + max_rows
                    newdf = df.iloc[init:end]
                    dataframes.append(newdf)
                    i += max_rows
                    progress.update(task, advance=max_rows)
                newdf = df.iloc[i:]
                dataframes.append(newdf)
                progress.update(task, advance=(length-i))
        else:
            dataframes = [df]
    else:  # split by frequency
        dataframes = [g for n, g in df.groupby(pd.Grouper(freq=frequency))]

    # ensure that we do not have empty dataframes
    dataframes = [df for df in dataframes if not df.empty]

    return dataframes






