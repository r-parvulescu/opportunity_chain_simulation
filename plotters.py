import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from copy import deepcopy

#TODO add proper documentation


def time_series_metrics(batchrun):
    """Take a batchrun and return a dict of dataframes, one for each metric the model computes per step.
    Each dataframe is a timeseries for that metrics: each row is a run, each column is a step.
                    METRIC 1
    eg.         step 1  step 2
        run 1     4       5
        run 2    4.5      6
        run 3    2.2      3
    """
    models_in_run_order = data_of_models_in_run_order(batchrun)
    metric_names = models_in_run_order[0].columns.values
    metrics = {m: {} for m in metric_names}
    for one_run in models_in_run_order:
        columns = one_run.columns.values
        for c in columns:
            values_across_steps = one_run.loc[:, c]
            list_of_values_across_steps = turn_base_values_to_list(values_across_steps)
            stack_list_on_df(metrics, c, list_of_values_across_steps)
    return metrics


def data_of_models_in_run_order(batchrun):
    """Returns all the per-step model data (as dataframe) for each run, in a list of tuples
    that maintains the order of the runs."""
    tuples_in_run_order = [(int(str(k).translate(str.maketrans('', '', string.punctuation))),
                            v["Data Collector"]) for k, v in batchrun.model_vars.items()]
    tuples_in_run_order.sort()
    models_in_run_order = [i[1].get_model_vars_dataframe() for i in tuples_in_run_order]
    return models_in_run_order


def turn_base_values_to_list(pandas_series):
    """Turns a pandas series where each entry contains a dict (same key across all entries)
    into a dict with the key and a list of the values across all entries, in order."""
    keys = pandas_series[0].keys()
    lists_of_values_across_steps = {k: [] for k in keys}
    for step in pandas_series:
        for base_value in step.items():
            lists_of_values_across_steps[base_value[0]].append(base_value[1])
    return lists_of_values_across_steps


def stack_list_on_df(dictio, dict_key, lst):
    """As values (in lists) come in for different dict entries, stacks these values as new rows
        in pandas dataframe."""
    if dictio[dict_key] == {}:
        keys = list(lst.keys())
        dictio[dict_key] = {k: pd.DataFrame() for k in keys}
    for i in lst.items():
        dictio[dict_key][i[0]] = dictio[dict_key][i[0]].append(pd.Series(i[1]), ignore_index=True)


def means_std(batchrun):
    """Get means and starndard deviations across all runs, one mean and stdev per model step."""
    metrics = time_series_metrics(batchrun)
    per_step_stats = deepcopy(metrics)  # just replace current entries with what will go instead
    for k in metrics.keys():  # look into the metrics
        for l in metrics[k].items():  # look into the submetrics
            # calculate means and standard deviations per set, across all model runs
            mean = l[1].mean(axis=0)
            stdev = l[1].std(axis=0)
            if isinstance(per_step_stats[k][l[0]], pd.DataFrame):
                per_step_stats[k][l[0]] = {}
            # dump into appropriate place
            per_step_stats[k][l[0]]["Mean Across Runs"] = mean
            per_step_stats[k][l[0]]["StDev Across Runs"] = stdev
    return per_step_stats


def plot_lines(batchrun):
    """Given mean and standard deviation for each step, plot lines with shaded regions 2*stdev."""

    colours = ['r-', 'b-', 'g-']
    per_step_stats = means_std(batchrun)
    for k in per_step_stats.keys():  # look into the metrics
        colour_counter = 0
        for l in per_step_stats[k].keys():  # look into the submetrics
            mean_line = None
            stdev_line = None
            for m in per_step_stats[k][l].items():  # look into means and stdevs
                # calculate means and standard deviations per set, across all model runs
                if m[0] == "Mean Across Runs":
                    mean_line = m[1]
                else:
                    stdev_line = m[1]

            x = np.linspace(0, len(mean_line)-1, len(mean_line))
            plt.plot(x, mean_line, colours[colour_counter], label=l)
            stdev_lowbound = mean_line - stdev_line * 2
            stdev_lowbound[stdev_lowbound < 0] = 0
            plt.fill_between(x, stdev_lowbound, mean_line + stdev_line * 2, alpha=0.2)
            colour_counter += 1

        title_name = k.replace('_', ' ')
        plt.title(title_name.title())
        plt.legend(loc='upper left', fontsize='medium')
        figure_filename = "figures/" + title_name + "_" + str(batchrun.iterations) + \
                          "runs_" + str(batchrun.max_steps) + "steps.png"
        plt.savefig(figure_filename)
        plt.close()
