"""
tools for plotting per-step means and standard
deviations of metrics from batch runs of MobilityModel
the data structure (riffing off MESA's batchrunner module) is of a list containing of nested dicts, with structure
[models in run order
    {steps per model
        {metrics per step
            {submetrics (one each for actor and vacancy, or per level of the hierarchy)
                {means and stdevs for submetric
}}}}]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from string import punctuation
from copy import deepcopy


def get_data_of_models_in_run_order(batchrun):
    """returns all per-step model data (as a pd.DataFrame) for each run, in the order of the runs"""
    tuples_in_run_order = [(int(str(k).translate(str.maketrans('', '', punctuation))),
                            v["Data Collector"]) for k, v in batchrun.model_vars.items()]
    tuples_in_run_order.sort()
    models_in_run_order = [i[1].get_model_vars_dataframe() for i in tuples_in_run_order]
    return models_in_run_order


def get_metrics_timeseries_dataframes(batchrun):
    """take a batchrun and return a dict of pd.DataFrames, one for each metric
    each dataframe shows metric timeseries across steps, for each run; example output below
                    METRIC 1
    eg.         step 1  step 2
        run 1     4       5
        run 2    4.5      6
        run 3    2.2      3
    """
    models_in_run_order = get_data_of_models_in_run_order(batchrun)
    metric_names = models_in_run_order[0].columns.values
    metric_dataframes = {m: {} for m in metric_names}
    for one_run in models_in_run_order:
        for metric in one_run.columns.values:
            values_across_steps = one_run.loc[:, metric]
            flat_values_across_steps = flatten_dict(values_across_steps)
            flatten_dicts_into_df(flat_values_across_steps, metric_dataframes, metric)
    return metric_dataframes


# helper function for get_metrics_timeseries_dataframes
def flatten_dict(pandas_series):
    """
    flatten a pandas series of dicts, where each dict has the same keys, into a dict whose keys are a list
    of values across (former) subdicts
    e.g. pd.Series({"me": you, "her": him}, {"me": Thou, "her": jim}) => {"me": you, Thou, "her": him, jim}
    """
    # TODO this might be messing up order, need to look into it again
    keys = pandas_series[0].keys()
    values_across_steps = {k: [] for k in keys}
    for step in pandas_series:
        for base_value in step.items():
            values_across_steps[base_value[0]].append(base_value[1])
    return values_across_steps


# helper function for get_metrics_timeseries_dataframes
def flatten_dicts_into_df(input_dict, output_dict, output_dict_key):
    """
    for a set of dicts, each with the same keys and whose values are lists of equal length,
    stacks the lists in a pd.DataFrame named after the key, and insert the new key:values into some dict.
    e.g. {"me": [you, Thou], "her": [him, jim]}
         {"me": [Pradeep, King], "her": [without, slim]}"
         => some_dict{
         "me" : pd.DataFrame(
            you         thou
            Pradeep     King),
         "her" : pd.DataFrame(
            him          jim
            without      slim) }
    """
    # if output_dict empty at certain key, make its value a dict, full of input_dict_key : pd.DataFrame
    if output_dict[output_dict_key] == {}:
        input_dict_keys = list(input_dict.keys())
        output_dict[output_dict_key] = {k: pd.DataFrame() for k in input_dict_keys}
    # turn the lists into pd.Series and append them to the dataframe
    for i in input_dict.items():
        output_dict[output_dict_key][i[0]] = output_dict[output_dict_key][i[0]].append(pd.Series(i[1]),
                                                                                       ignore_index=True)


def get_means_std(batchrun):
    """for each step, get a submetric's mean and starndard deviations across all model runs"""
    metrics = get_metrics_timeseries_dataframes(batchrun)
    per_step_stats = deepcopy(metrics)  # keep nested dict structure but replace dataframes with means and stdevs
    for k in metrics.keys():  # look into the metrics
        for l in metrics[k].items():  # look into the submetrics
            # calculate means and standard deviations, across all model runs
            mean = l[1].mean(axis=0)
            stdev = l[1].std(axis=0)
            if isinstance(per_step_stats[k][l[0]], pd.DataFrame):
                per_step_stats[k][l[0]] = {}
            # put into appropriate place
            per_step_stats[k][l[0]]["Mean Across Runs"] = mean
            per_step_stats[k][l[0]]["StDev Across Runs"] = stdev
    return per_step_stats


def make_time_series_figures(batchrun):
    """
    makes time series figures and saves them to disk
    if input is more than one batchrun (must be with same parameter types and number of steps) then
    overlays the output of the two batchruns in the same figure, so you can see difference betweem
    batchrun metrics
    """
    per_step_stats = get_means_std(batchrun)
    for k in per_step_stats.keys():  # look into the metrics
        colour_counter = 0
        for l in per_step_stats[k].keys():  # look into the submetrics
            mean_line = None
            stdev_line = None
            for m in per_step_stats[k][l].items():  # look into means and stdevs
                if m[0] == "Mean Across Runs":
                    mean_line = m[1]
                else:
                    stdev_line = m[1]
            # plot the line
            plot_mean_line(l, mean_line, stdev_line, colour_counter, linestyle="-")
            colour_counter += 1
        # name the plot, save the figure to disk
        title_name = k.replace('_', ' ')
        save_figure(title_name, batchrun)


def plot_mean_line(line_name, mean_line, stdev_line, colour_counter, linestyle):
    """given line name, mean and associated stdev values, and a colour counter, plots a line"""
    colours = ['r-', 'b-', 'k-', 'g-', 'c-', 'm-', 'y-']
    colour_counter = colour_counter
    x = np.linspace(0, len(mean_line) - 1, len(mean_line))
    plt.plot(x, mean_line, colours[colour_counter], linestyle=linestyle, label=line_name)
    # make sure lower stdev doesn't go below zero
    stdev_lowbound = mean_line - stdev_line * 2
    stdev_lowbound[stdev_lowbound < 0] = 0
    plt.fill_between(x, stdev_lowbound, mean_line + stdev_line * 2, alpha=0.2)


def save_figure(metric_name, batchrun, close=True):
    """given a title for the figure and a batchrun, completes figure, saves it to .png, and closes open plots"""
    title_name = metric_name.replace('_', ' ')
    plt.title(title_name.title())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5,  fontsize='medium')
    figure_filename = "figures/" + metric_name + "_" + str(batchrun.iterations) + \
                      "runs_" + str(batchrun.max_steps) + "steps.png"
    plt.savefig(figure_filename)
    plt.close()


def overaly_time_series_figures(batchruns):
    """
    makes time series figures and saves them to disk
    if input is more than one batchrun (must be with same parameter types and number of steps) then
    overlays the output of the two batchruns in the same figure, so you can see difference betweem
    batchrun metrics
    """

    br1_per_step_stats = get_means_std(batchruns[0])
    br2_per_step_stats = get_means_std(batchruns[1])
    # look into the metrics
    # NB: all batchruns have identical nesting and key:value structure, only bottom level data differ
    for k in br1_per_step_stats.keys():
        colour_counter = 0
        for l in br1_per_step_stats[k].keys():  # look into the submetrics
            br1_mean_line = None
            br1_stdev_line = None
            br2_mean_line = None
            br2_stdev_line = None
            for m in br1_per_step_stats[k][l].keys():  # look into means and stdevs
                if m == "Mean Across Runs":
                    br1_mean_line = br1_per_step_stats[k][l][m]
                    br2_mean_line = br2_per_step_stats[k][l][m]
                else:
                    br1_stdev_line = br1_per_step_stats[k][l][m]
                    br2_stdev_line = br2_per_step_stats[k][l][m]
            # plot the line
            plot_mean_line(l, br1_mean_line, br1_stdev_line, colour_counter, "-")
            plot_mean_line("with firings", br2_mean_line, br2_stdev_line, colour_counter, "--")
            colour_counter += 1
        # name the plot, save the figure to disk
        save_figure(k, batchruns[0])



