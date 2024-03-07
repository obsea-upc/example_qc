#!/usr/bin/env python3
"""

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
"""

from argparse import ArgumentParser
import pandas as pd
from m3 import open_dataset
import os
import json
import rich
import time
from m3.utils import merge_dataframes, slice_dataframes, multiprocess
import gc
from m3.qc.qc import apply_qc

def dataset_qc(df: pd.DataFrame, qc_config: dict, show=False, save="", varlist=[], limits: dict={}, tests=[], paralell=False):
    """
    Applies QualityControl to a dataset using DatasetRegistry class
    :param mc: MetadataCollector object
    :param sensor_id: Sensor ID
    :param show: if True, results will be plotted
    :param save: directory name where the test results will be stored (optional)
    :param varlist: list of variables no show/save (optional)
    :param tests: list of tests to show/save (optional)
    :return: DatasetRegistry
    """
    init = time.time()

    if "all" in varlist or not varlist:  # if empty or explicitly all
        varlist = list(df.columns)

    rich.print("[cyan]Applying Quality Control (multiprocessing)...")
    dataframes = slice_dataframes(df, frequency="M")  # generate a dataframe for each month
    argument_list = []
    for df in dataframes:
        argument_list.append((df, qc_config, show, save, varlist, limits, tests))

    t = time.time()
    qc_dataframes = multiprocess(argument_list, apply_qc, max_workers=20)
    rich.print("Multiprocess QC %.02f s" % (time.time() - t))
    t = time.time()
    df = merge_dataframes(qc_dataframes, sort=True)
    rich.print("Merge dataframes %.02f s" % (time.time() - t))
    gc.collect()

    rich.print("[cyan]QC took %.02f seconds" % (time.time() - init))
    return df


if __name__ == "__main__":
    # Adding command line options #
    argparser = ArgumentParser()
    argparser.add_argument("-o", "--output", type=str, required=False, help="Output file to store the curated dataset")
    argparser.add_argument("-i", "--input", required=True, help="CSV dataset to be processed", default="")
    argparser.add_argument("-l", "--limits", help="set the limits as a dict: -l {\"temperature\": [12,40]}", default="{}")
    argparser.add_argument("-t", "--tests", help="comma-separated list of tests", default="")
    # argparser.add_argument("-p", "--parallel", help="Applies multiprocessing to QC to speed it up", action="store_true")
    argparser.add_argument("-V", "--variables", default="all", help="comma-separated list of variables (empty=all)")
    argparser.add_argument("-A", "--qc-all", default=False, action="store_true", help="Prints all QC tests")
    argparser.add_argument("-S", "--save-qc", default="", type=str, help="Store the result of the QC tests as images")
    argparser.add_argument("-X", "--show-qc", default=False, action="store_true", help="Plots all QC tests")
    argparser.add_argument("-q", "--qc-config", type=str, help="QC config file", default="", required=True)
    args = argparser.parse_args()

    df = open_dataset(args.input)


    rich.print("Removing old QC columns...")
    for c in list(df.columns):
        if c.lower().endswith("_qc"):
            del df[c]
    variables = args.variables.split(",")
    if not variables:
        variables = list(df.columns)
    limits = json.loads(args.limits)

    for v in list(df.columns):
        if " (" in v:
            rich.print(f"replacing {v}")
            df = df.rename(columns={v: v.split(" (")[0]})
    rich.print(df)
    rich.print(f"Processing variables {variables}")

    with open(args.qc_config) as f:
        qc_config = json.load(f)

    if args.save_qc:
        filename = os.path.basename(args.input).split(".")[0]
        savedir = os.path.join(args.save_qc, filename)
        os.makedirs(savedir, exist_ok=True)

    if not args.tests:
        tests = []
    else:
        tests = args.tests.split(",")

    df = dataset_qc(df, qc_config, show=args.show_qc, save=savedir, varlist=variables, limits=limits, tests=args.tests, paralell=True)
    rich.print(df)

    if args.output:
        rich.print(f"saving to {args.output}...", end="")
        df.to_csv(args.output)
        rich.print(f"[green]ok!")
    else:
        rich.print(f"[yellow]Not saving results!")
