#!/usr/bin/env python3
"""
This file provides a user-friendly interface for actions such as data curator, apply qc, resample dataset and more

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
created: 23/3/21
"""
import os
import pandas as pd
from rich.progress import Progress
import concurrent.futures as futures
import multiprocessing as mp


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



def merge_dataframes(df_list, sort=False):
    """
    Appends together several dataframes into a big dataframe
    :param df_list: list of dataframes to be appended together
    :param sort: If True, the resulting dataframe will be sorted based on its index
    :return:
    """
    df = df_list[0]
    for new_df in df_list[1:]:
        df = df.append(new_df)

    if sort:
        df = df.sort_index(ascending=True)
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



if __name__ == "__main__":
    pass
