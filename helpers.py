import numpy as np
import random
from itertools import groupby

#TODO add proper documentation

# mere helper
def fraction_of_list(fraction, list_length):
    """Returns a list of bools split according to a float [0,1]"""
    fraction_trues = int(list_length * fraction)
    list_of_bools = fraction_trues * [True] + (list_length - fraction_trues) * [False]
    random.shuffle(list_of_bools)
    return list_of_bools


def percent_vacancy_per_level(model):
    """Find the percent of vacancies for each level."""
    actor_level_counts = [0 for i in range(model.num_levels)]
    vacancy_level_counts = [0 for i in range(model.num_levels)]
    for e in model.schedule.agents:
        e_level = int(e.position[0])
        if e.type == "actor":
            actor_level_counts[e_level - 1] += 1
        else:
            vacancy_level_counts[e_level - 1] += 1
    vacancy_percentages = {"Level " + str(i+1): 0 for i in range(model.num_levels)}
    for i in range(model.num_levels):
        key = "Level " + str(i+1)
        vacancy_percentages[key] = vacancy_level_counts[i] / (vacancy_level_counts[i] + actor_level_counts[i])
    return vacancy_percentages


# basic statistics
def agent_counts(model):
    """Compute the number of actors and vacancies."""
    act_cntr, vac_cntr = 0, 0
    for a in model.schedule.agents:
        if a.type == "vacancy":
            vac_cntr += 1
        if a.type == "actor":
            act_cntr += 1
    if act_cntr + vac_cntr != sum(model.positions_per_level):
        raise ValueError('Some positions have no dual at all. Impossible!')
    return {"Actor Count": act_cntr, "Vacancy Count": vac_cntr}


# chain/sequence statistics
def sequence_and_chain_lengths(model):
    """Find the lengths of actor sequences and vacancy chains."""
    chain_lens, seq_lens = [], []
    for a in model.schedule.agents:
        chain_lens.append(len(a.log)) if a.type == "vacancy" else seq_lens.append(len(a.log))
    return seq_lens, chain_lens


def mean_lengths(model):
    """Compute the average length of actor sequences and vacancy chains."""
    lengths = sequence_and_chain_lengths(model)
    return {"Actor Sequence": np.mean(lengths[0]), "Vacancy Chain": np.mean(lengths[1])}


def length_std(model):
    """Compute the standard deviations of actors sequences and vacancy chains."""
    lengths = sequence_and_chain_lengths(model)
    return {"Actor Sequence": np.std(lengths[0]), "Vacanacy Chain": np.std(lengths[1])}


# spell statistics
def mean_spell_length_per_agent(lst):
    """Given a list of strings, compute mean spell length, where spell is a run of consecutive, identical
    entries, at least two long."""
    spell_lengths = [sum(1 for i in g) for k, g in groupby(lst)]
    spell_lengths = [i for i in spell_lengths if i != 1]
    if not spell_lengths:
        return np.mean(spell_lengths)


def list_of_spell_length_per_agent_type(model):
    """Compute the average length of spells for actor sequences and vacancy chains."""
    actor_mean_spell_lengths = []
    vacancy_mean_spell_lengths = []
    for a in model.schedule.agents:
        if len(a.log) > 1:
            mean_spell_length = mean_spell_length_per_agent(a.log)
            if mean_spell_length is not None:
                actor_mean_spell_lengths.append(mean_spell_length) if a.type == "actor" \
                    else vacancy_mean_spell_lengths.append(mean_spell_length)
    return actor_mean_spell_lengths, vacancy_mean_spell_lengths


def mean_spell_lengths(model):
    """Compute the mean length of spells for actor sequences and vacancy chains."""
    lengths = list_of_spell_length_per_agent_type(model)
    return {"Actor Sequence": np.mean(lengths[0]), "Vacancy Chain": np.mean(lengths[1])}


def std_spell_lengths(model):
    """Compute the standard deviation of spells for actor sequences and vacancy chains."""
    lengths = list_of_spell_length_per_agent_type(model)
    return {"Actor Sequence": np.std(lengths[0]), "Vacancy Chain": np.std(lengths[1])}
