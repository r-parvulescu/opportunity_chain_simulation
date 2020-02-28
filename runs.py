from plotters import tuple_visualiser



positions_per_level = [100, 200, 300]
move_probability = 0.5
initial_vacancy_fraction = 0.2
firing_schedule = None
params = positions_per_level, move_probability, initial_vacancy_fraction, firing_schedule
# firing_schedule = {"steps": [10], "level-retire probability": [(1, 1.0)]}

#tuple_visualiser("agent_counts", 10, 40, params)
tuple_visualiser("mean_lengths", 100, 100, params)
#tuple_visualiser("lengths_std", 10, 40, params)
#tuple_visualiser("mean_spells", 10, 40, params)
#tuple_visualiser("spells_std", 10, 40, params)
#tuple_visualiser("percent_vacant_per_level", 10, 40, params)



'''
df = pd.DataFrame()
for run in range(100):
    #initiate model
    model = MobilityModel(positions_per_level, move_probability, initial_vacancy_fraction, firing_schedule)
    for steps in range(40):
        model.step()
    # get the data from the data collector and store it: remember this is in order of steps
    basic_data = model.datacollector.model_vars["Counts, Means, Standard Deviations"]
    actor_counts_per_step = []
    for a in basic_data:
        actor_counts_per_step.append(a[1])
    # now store in the dataframe
    s = pd.Series(actor_counts_per_step)
    #print(s)
    df = df.append(s, ignore_index=True)

#print(df)

means = df.mean(axis=0)
stdevs = df.std(axis=0)
#print(means)
#print(stdevs)
#print(len(means))

x = np.linspace(0, 39, 40)
ax1 = plt.subplot(2, 1, 1)
plt.plot(x, means, 'r--', label="Actor Counts")

chart_box1 = ax1.get_position()
ax1.fill_between(x, means - stdevs*2, means + stdevs*2, alpha=0.2)
ax1.set_position([chart_box1.x0, chart_box1.y0, chart_box1.width * 0.6, chart_box1.height])
ax1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.show()

#basic_data_show(model)
#percent_vacant_per_level(model)
#spell_show(model)


fixed_params = {
    "positions_per_level": [10, 20, 30],
     "initial_vacancy_fraction": 0.2,
    "firing_schedule": None
}

variable_params = {"move_probability": [0.5]}

# The variables parameters will be invoked along with the fixed parameters allowing for either or both to be honored.
batch_run = BatchRunner(
    MobilityModel,
    variable_params,
    fixed_params,
    iterations=2,
    max_steps=10,
    model_reporters={"all_data": DataCollector}
)

batch_run.run_all()

run_data = batch_run.get_model_vars_dataframe()
run_data.to_csv("text.csv")
#v = run_data.loc[:, "vacancy_percent_per_level"]
#for s in v:
#    print(v)
#print(v)
#print(type(v))
#print(type)
#run_data.head()
#plt.plot(run_data.move_probability, run_data.vacancy_percent_per_level)
#plt.show()
'''


"""
def vacancy_percent_per_level_visualiser(data_desired, runs, steps, params):
    ppl, mp, ivf, fs = params[0], params[1], params[2], params[3]
    dfs = [pd.DataFrame() for i in range(len(ppl))]
        for run in range(runs):
        model = MobilityModel(ppl, mp, ivf, fs)
        for s in range(steps):
            model.step()
        # get the data from the data collector and store it: remember this is in order of steps
        data = model.datacollector.model_vars[data_desired]
        entries_per_step = [[] for i in range(len(ppl))]
        for i in range(len(ppl))
            entries_per_step[i].append(data[i])

        e = [pd.Series(entries_per_step[i]) for in range(len(ppl))]
        e = pd.Series(entries_per_step)
        for i in range(len(ppl)):
            dfs[i] = dfs[i].append(e[i], ignore_index=True)

    e_means = e_df.mean(axis=0)
    e_stdevs = e_df.std(axis=0)

    x = np.linspace(0, steps - 1, steps)

    plt.plot(x, e_means, 'r-', label="Actors")
    e_low_bound = e_means - e_stdevs * 2
    e_low_bound[e_low_bound < 0] = 0
    plt.fill_between(x, e_low_bound, e_means + e_stdevs * 2, alpha=0.2)

    plt.title(data_desired.replace('_', ' ').title())
    plt.legend(loc='upper left', fontsize='medium')
    plt.show()
"""