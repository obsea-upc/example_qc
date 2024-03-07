#!/usr/bin/env python3

import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import rich
import time
from rich.progress import Progress
import multiprocessing as mp
from concurrent import futures
import numpy as np

# Color codes
GRN = "\x1B[32m"
RST = "\033[0m"
BLU = "\x1B[34m"
YEL = "\x1B[33m"
RED = "\x1B[31m"
MAG = "\x1B[35m"
CYN = "\x1B[36m"
WHT = "\x1B[37m"
NRM = "\x1B[0m"
RST = "\033[0m"


def setup_log(name, path="log", log_level="debug"):
    """
    Setups the logging module
    :param name: log name (.log will be appended)
    :param path: where the logs will be stored
    :param log_level: log level as string, it can be "debug, "info", "warning" and "error"
    """

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Check arguments
    if len(name) < 1 or len(path) < 1:
        raise ValueError("name \"%s\" not valid", name)
    elif len(path) < 1:
        raise ValueError("name \"%s\" not valid", name)

    # Convert to logging level
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    elif log_level == 'error':
        level = logging.ERROR
    else:
        raise ValueError("log level \"%s\" not valid" % log_level)

    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, name)
    if not filename.endswith(".log"):
        filename += ".log"
    print("Creating log", filename)
    print("name", name)

    logger = logging.getLogger()
    logger.setLevel(level)
    log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-7s: %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')
    handler = TimedRotatingFileHandler(filename, when="midnight", interval=1, backupCount=7)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(log_formatter)
    logger.addHandler(consoleHandler)

    logger.info("")
    logger.info(f"===== {name} =====")

    return logger


def collect_from_dict(dictfrom, dictto):
    """
    This function collects all key-value pairs from one dictionary and stores it to another. Only key-value pairs with
    str values are collected.
    :param dictfrom: dictionary to collect from
    :param dictto: dict to collect to
    """
    for key, value in dictfrom.items():
        if type(value) == str:
            if key in dictto.keys():
                raise ValueError("Key %s already exists in dict" % key)
            else:
                dictto[key] = value


def normalize_string(instring, lower_case=False):
    """
    This function takes a string and normalizes by replacing forbidden chars by underscores.The following chars
    will be replaced: : @ $ % & / + , ; and whitespace
    :param instring: input string
    :return: normalized string
    """
    # forbidden_chars = [":", "@", "$", "%", "&", "/", "+", ",", ";", " ", "_"]
    forbidden_chars = [":", "@", "$", "%", "&", "/", "+", ",", ";", " ", "-"]
    outstring = instring
    for char in forbidden_chars:
        outstring = outstring.replace(char, "_")
    if lower_case:
        outstring = outstring.lower()
    return outstring


def camel_case_to_underscore(instring):
    """
    This function converts from camelCase to camel_case with underscores
    :param instring: e.g. camelCase
    :return: formatted string camel_case
    """
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', instring).lower()


def __threadify_index_handler(index, handler, args):
    """
    This function adds the index to the return of the handler function. Useful to sort the results of a
    multi-threaded operation
    :param index: index to be returned
    :param handler: function handler to be called
    :param args: list with arguments of the function handler
    :return: tuple with (index, xxx) where xxx is whatever the handler function returned
    """
    result = handler(*args)  # call the handler
    return index, result  # add index to the result


def threadify(arg_list, handler, max_threads=10, text: str = "progress..."):
    """
    Splits a repetitive task into several threads
    :param arg_list: each element in the list will crate a thread and its contents passed to the handler
    :param handler: function to be invoked by every thread
    :param max_threads: Max threads to be launched at once
    :return: a list with the results (ordered as arg_list)
    """
    index = 0  # thread index
    with futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        threads = []  # empty thread list
        results = []  # empty list of thread results
        for args in arg_list:
            # submit tasks to the executor and append the tasks to the thread list
            threads.append(executor.submit(__threadify_index_handler, index, handler, args))
            index += 1

        # wait for all threads to end
        with Progress() as progress:  # Use Progress() to show a nice progress bar
            task = progress.add_task(text, total=index)
            for future in futures.as_completed(threads):
                future_result = future.result()  # result of the handler
                results.append(future_result)
                progress.update(task, advance=1)

        # sort the results by the index added by __threadify_index_handler
        sorted_results = sorted(results, key=lambda a: a[0])

        final_results = []  # create a new array without indexes
        for result in sorted_results:
            final_results.append(result[1])
        return final_results


def multiprocess(arg_list, handler, max_workers=20, text: str = "progress..."):
    """
    Splits a repetitive task into several processes
    :param arg_list: each element in the list will crate a thread and its contents passed to the handler
    :param handler: function to be invoked by every thread
    :param max_threads: Max threads to be launched at once
    :return: a list with the results (ordered as arg_list)
    :param text: text to be displayed in the progress bar
    """
    index = 0  # thread index
    ctx = mp.get_context('spawn')
    with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        processes = []  # empty thread list
        results = []  # empty list of thread results
        for args in arg_list:
            # submit tasks to the executor and append the tasks to the thread list
            processes.append(executor.submit(__threadify_index_handler, index, handler, args))
            index += 1

        with Progress() as progress:  # Use Progress() to show a nice progress bar
            task = progress.add_task(text, total=index)
            for future in futures.as_completed(processes):
                future_result = future.result()  # result of the handler
                results.append(future_result)
                progress.update(task, advance=1)

        # sort the results by the index added by __threadify_index_handler
        sorted_results = sorted(results, key=lambda a: a[0])

        final_results = []  # create a new array without indexes
        for result in sorted_results:
            final_results.append(result[1])
        return final_results


def get_dataframe_precision(df, check_values=1000, min_precision=-1):
    """
    Retunrs a dict with the precision for each column in the dataframe
    :param df: input dataframes
    :param check_values: instead of checking the precision of all values, the first N values will be checked
    :param min_precision: If precision is less than min_precision, use this value instead
    """
    column_precision = {}
    length = min(check_values, len(df.index.values))
    for c in df.columns:
        if "_qc" in c:
            continue  # avoid QC columns
        precisions = np.zeros(length, dtype=int)
        for i in range(0, length):
            value = df[c].values[i]
            try:
                _, float_precision = str(value).split(".")
            except ValueError:
                float_precision = ""  # if error precision is 0 (point not found)
            precisions[i] = len(float_precision)  # store the precision
        column_precision[c] = max(precisions.max(), min_precision)
    return column_precision


def time_me(handler, args, text=""):
    """
    Times the execution time of a function
    :param handler: function to be exectued
    :param args: arguments to be passed
    :param text: label to be displayed
    :return: handler's return value
    """
    init = time.time()
    result = handler(args)
    rich.print("[blue]%s took %0.02f seconds " % (text, time.time() - init))
    return result


def log_to_lin(df: pd.DataFrame, vars: list):
    """
    Converts some columns of a dataframe from logarithmic to linear
    :param df: dataframe
    :param vars: list of variables to convert
    :returns: dataframe with convert columns
    """

    for var in df.columns:
        if var in vars:
            df[var] = 10 ** (df[var] / 10)
    return df


def lin_to_log(df: pd.DataFrame, vars: list):
    """
    Converts some columns of a dataframe from linear to logarithmic
    :param df: dataframe
    :param vars: list of variables to convert
    :returns: dataframe with convert columns
    """
    for var in df.columns:
        if var in vars:
            df[var] = 10 * np.log10(df[var])
    return df


def delete_duplicate_values(df, timestamp="timestamp"):
    """
    useful for datasets that have duplicated values with consecutive timestamps (e.g. data is generated minutely, but
    inserted into a database every 20 secs). So the following dataframe:

                                col1      col2    col3
        timestamp
        2020-01-01 00:00:00    13.45    475.45    12.7
        2020-01-01 00:00:20    13.45    475.45    12.7
        2020-01-01 00:00:40    13.45    475.45    12.7
        2020-01-01 00:01:00    12.89    324.12    78.8
        2020-01-01 00:01:20    12.89    324.12    78.8
        2020-01-01 00:01:40    12.89    324.12    78.8
        ...

    will be simplified to:

                                col1      col2    col3
        timestamp
        2020-01-01 00:00:00    13.45    475.45    12.7
        2020-01-01 00:01:00    12.89    324.12    78.8

    :param df: input dataframe
    :return: simplified dataframe
    """
    if df.empty:
        rich.print("[yellow]WARNING empty dataframe")
        return df
    df = df.reset_index()
    columns = [col for col in df.columns if col != timestamp]
    del_array = np.zeros(len(df))  # create an empty array
    duplicates = 0
    with Progress() as progress:  # Use Progress() to show a nice progress bar
        task = progress.add_task("Detecting duplicates", total=len(df))
        init = True
        for index, row in df.iterrows():
            progress.update(task, advance=1)
            if init:
                init = False
                last_valid_row = row
                continue

            diff = False  # flag to indicate if the current column is different from the last valid
            for column in columns:  # compare value by value
                if row[column] != last_valid_row[column]:
                    # column is different
                    last_valid_row = row
                    diff = True

                    break
            if not diff:  # there's no difference between columns, so this one needs to be deleted
                del_array[duplicates] = index
                duplicates += 1

    print(f"Duplicated lines {duplicates} from {len(df)}, ({100*duplicates/len(df):.02f} %)")
    del_array = del_array[:duplicates]  # keep only the part of the array that has been filled
    rich.print("dropping rows...")
    df.drop(del_array, inplace=True)
    df = df.set_index(timestamp)
    return df


def find_in_dict_list(elements: list, name: str, name_key="name"):
    """
    Loops through a list of dicts and finds an element with matching a speficic name. By default "name" key is used, but
    can be changed with the parameter name_key
    :param elements: list of elements
    :param name: name of the element
    :param name_key: key for name, defaults to "name"
    :return: element
    """
    for element in elements:
        if type(element) != dict:
            continue
        elif element[name_key] == name:
            return element

    raise LookupError(f"element {name} not found!")


def file_list(dir_name):
    """ create a list of file and sub directories names in the given directory"""
    listOfFile = os.listdir(dir_name)
    allFiles = list()
    for entry in listOfFile:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            allFiles = allFiles + file_list(full_path)
        else:
            allFiles.append(full_path)
    return allFiles


def assert_dict(conf: dict, required_keys: dict):
    """
    Checks if all the expected keys in a dictionary are there. The lists __required_keys and the dict __type are
    required
    :param conf: dict with configuration to be checked
    :param required_keys: dictionary with required keys
    :raises: AssertionError if the input does not match required_keys
    """
    for key, expected_type in required_keys.items():
        if key not in conf.keys():
            raise AssertionError(f"Required key \"{key}\" not found")

        value = conf[key]
        if type(value) != expected_type:
            raise AssertionError(f"Value for key \"{key}\" wring type, expected type {expected_type}, but got "
                                 f"{type(value)}")


qc_flags = {
    "good": 1,
    "not_applied": 2,
    "suspicious": 3,
    "bad": 4,
    "missing": 9
}


def setup_log(name, path="log", logger_name="QualityControl", log_level="debug"):
    """
    Setups the logging module
    :param name: log name (.log will be appended)
    :param path: where the logs will be stored
    :param log_level: log level as string, it can be "debug, "info", "warning" and "error"
    """

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Check arguments
    if len(name) < 1 or len(path) < 1:
        raise ValueError("name \"%s\" not valid", name)
    elif len(path) < 1:
        raise ValueError("name \"%s\" not valid", name)

    # Convert to logging level
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    elif log_level == 'error':
        level = logging.ERROR
    else:
        raise ValueError("log level \"%s\" not valid" % log_level)

    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, name)
    if not filename.endswith(".log"):
        filename += ".log"
    print("Creating log", filename)
    print("name", name)

    logger = logging.getLogger()
    logger.setLevel(level)
    log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-7s: %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')
    handler = TimedRotatingFileHandler(filename, when="midnight", backupCount=7)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(log_formatter)
    logger.addHandler(consoleHandler)

    logger.info("")
    logger.info(f"===== {logger_name} =====")

    return logger


def dataframe_to_dict(df, key, value):
    """
    Takes two columns of a dataframe and converts it to a dictionary
    :param df: input dataframe
    :param key: column name that will be the key
    :param value: column name that will be the value
    :return: dict
    """

    keys = df[key]
    values = df[value]
    d = {}
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d


def varname_from_datastream(ds_name):
    """
    Extracts a variable name from a datastream name. The datastream name must follow the following pattern:
        <station>:<sensor>:<VARNAME>:<data_type>
    :param ds_name:
    :raises: SyntaxError if the patter doesn't match
    :return: variable name
    """
    splitted = ds_name.split(":")
    if len(splitted) != 4:
        raise SyntaxError(f"Datastream name {ds_name} doesn't have the expected format!")
    varname = splitted[2]
    return varname


def assert_dict(conf: dict, required_keys: dict):
    """
    Checks if all the expected keys in a dictionary are there. The lists __required_keys and the dict __type are
    required
    :param conf: dict with configuration to be checked
    :param required_keys: dictionary with required keys
    :raises: AssertionError if the input does not match required_keys
    """
    for key, expected_type in required_keys.items():
        if key not in conf.keys():
            raise AssertionError(f"Required key \"{key}\" not found")

        value = conf[key]
        if type(value) != expected_type:
            raise AssertionError(f"Value for key \"{key}\" wring type, expected type {expected_type}, but got "
                                 f"{type(value)}")


def reverse_dictionary(data):
    """
    Takes a dictionary and reverses key-value pairs
    :param data: any dict
    :return: reversed dictionary
    """
    return {value: key for key, value in data.items()}


def normalize_string(instring, lower_case=False):
    """
    This function takes a string and normalizes by replacing forbidden chars by underscores.The following chars
    will be replaced: : @ $ % & / + , ; and whitespace
    :param instring: input string
    :return: normalized string
    """
    forbidden_chars = [":", "@", "$", "%", "&", "/", "+", ",", ";", " ", "-"]
    outstring = instring
    for char in forbidden_chars:
        outstring = outstring.replace(char, "_")
    if lower_case:
        outstring = outstring.lower()
    return outstring


class LoggerSuperclass:
    def __init__(self, logger: logging.Logger, name: str, colour=NRM):
        """
        SuperClass that defines logging as class methods adding a heading name
        """
        self.__logger_name = name
        self.__logger = logger
        if not logger:
            self.__logger = logging  # if not assign the generic module
        self.__log_colour = colour

    def warning(self, *args):
        mystr = YEL + "[%s] " % self.__logger_name + str(*args) + RST
        self.__logger.warning(mystr)

    def error(self, *args, exception=False):
        mystr = "[%s] " % self.__logger_name + str(*args)
        self.__logger.error(RED + mystr + RST)
        if exception:
            raise ValueError(mystr)

    def debug(self, *args):
        mystr = self.__log_colour + "[%s] " % self.__logger_name + str(*args) + RST
        self.__logger.debug(mystr)

    def info(self, *args):
        mystr = self.__log_colour + "[%s] " % self.__logger_name + str(*args) + RST
        self.__logger.info(mystr)
