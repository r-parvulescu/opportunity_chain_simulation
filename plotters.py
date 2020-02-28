import matplotlib.pyplot as plt
from model import MobilityModel
import numpy as np
import pandas as pd


def tuple_visualiser(data_desired, runs, steps, params):
    ppl, mp, ivf, fs = params[0], params[1], params[2], params[3]
    a_df = pd.DataFrame()
    v_df = pd.DataFrame()
    for run in range(runs):
        model = MobilityModel(ppl, mp, ivf, fs)
        for s in range(steps):
            model.step()
        # get the data from the data collector and store it: remember this is in order of steps
        data = model.datacollector.model_vars[data_desired]
        actor_entries_per_step = []
        vacancy_entries_per_step = []
        for d in data:
            actor_entries_per_step.append(d[0])
            vacancy_entries_per_step.append(d[1])

        a = pd.Series(actor_entries_per_step)
        v = pd.Series(vacancy_entries_per_step)
        a_df = a_df.append(a, ignore_index=True)
        v_df = v_df.append(v, ignore_index=True)

    a_means = a_df.mean(axis=0)
    a_stdevs = a_df.std(axis=0)
    v_means = v_df.std(axis=0)
    v_stdevs = v_df.std(axis=0)

    x = np.linspace(0, steps - 1, steps)

    plt.plot(x, a_means, 'r-', label="Actors")
    a_low_bound = a_means - a_stdevs * 2
    a_low_bound[a_low_bound < 0] = 0
    plt.fill_between(x, a_low_bound, a_means + a_stdevs * 2, alpha=0.2)

    plt.plot(x, v_means, 'b-', label="Vacancies")
    v_low_bound = v_means - v_stdevs * 2
    v_low_bound[v_low_bound < 0] = 0
    plt.fill_between(x, v_low_bound, v_means + v_stdevs * 2, alpha=0.2)

    plt.title(data_desired.replace('_', ' ').title())
    plt.legend(loc='upper left', fontsize='medium')
    plt.show()
