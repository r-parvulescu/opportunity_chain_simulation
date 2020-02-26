from mesa import Model
from agent import Actor, Position, Vacancy
from random_simultaneous import SimultaneousActivation
from mesa.datacollection import DataCollector
import numpy as np
import uuid
import random
from itertools import groupby


class MobilityModel(Model):
    """A hierarchical mobility model."""

    def __init__(self, positions_per_level, move_probability, initial_vacancy_fraction, firing_schedule):
        self.levels = len(positions_per_level)  # int: number of levels in system
        self.positions_per_level = positions_per_level  # list of ints: number of positions per level

        self.move_probability = move_probability  # the probability [0,1] with which agents decide to move

        # fraction [0,1] of positions per level that are initialised with a vacancy
        self.vacancy_fraction = initial_vacancy_fraction

        # dict giving steps at which to replace actors with vacancies,
        # and probability [0,1] of any given actor actors in that level being kicked out
        # e.g. {"fire steps":5,10, "fire orders": (1,0.4), (2,0.4), (3,0.6)}
        self.firing_schedule = firing_schedule

        self.schedule = SimultaneousActivation(self)
        self.running = True

        # dict of positions per level
        self.positions = {i: {} for i in range(1, self.levels + 1)}

        # make positions and populate them
        for i in range(self.levels):
            vacancies = self.fraction_of_list(initial_vacancy_fraction, self.positions_per_level[i])
            for j in range(self.positions_per_level[i]):
                position_id = str(i + 1) + '-' + str(j + 1)  # position ID = level-position number
                p = Position(position_id, self)
                self.positions[i + 1][position_id] = p
                # make entity and put in position
                agent = Vacancy(uuid.uuid4(), self) if vacancies[j] else Actor(uuid.uuid4(), self)
                self.schedule.add(agent)
                agent.position = p.unique_id
                p.dual = [agent.unique_id, agent.type]
                # update logs
                agent.log.append(p.unique_id)
                p.log.append(agent.unique_id)

        self.retiree_spots = set()
        self.desired_positions = []
        self.retirees = {"actor": {}, "vacancy": {}}

        self.datacollector = DataCollector(
            model_reporters={"Counts, Means, Standard Deviations": cnt_avrg_std,
                             "Means and Standard Deviations of Spells": spell_stats})

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        # update all position logs
        for lvl in self.positions.values():
            for p in lvl.values():
                p.log.append(p.dual[0])

        if self.firing_schedule is not None:
            for step in self.firing_schedule["fire steps"]:
                if step == self.schedule.steps:
                    for orders in self.firing_schedule["fire orders"]:
                        lvl = orders[0]
                        prob_of_vacating = orders[1]
                        positions_to_be_vacated = {p.unique_id for p in self.positions[lvl].values()}
                        for a in self.schedule.agents:
                            if (a.position in positions_to_be_vacated) and (a.type == "actor"):
                                if bool(np.random.binomial(1, prob_of_vacating)):
                                    v = Vacancy(uuid.uuid4(), self)
                                    a.retire(v)
        # reset with every step
        self.retiree_spots = set()
        self.desired_positions = []

    @staticmethod
    def fraction_of_list(fraction, list_length):
        """Method that returns a list of bools split according to a float [0,1]"""
        fraction_trues = int(list_length * fraction)
        list_of_bools = fraction_trues * [True] + (list_length - fraction_trues) * [False]
        random.shuffle(list_of_bools)
        return list_of_bools


def cnt_avrg_std(model):
    """Compute numbers of vacancies and actors, means of actor sequences and vacancy chains,
    and standard variation of actor sequences and vacancy chains,
    for actors and vacancies currently in system."""
    chain_lens, seq_lens = [], []
    act_cntr, vac_cntr = 0, 0
    for a in model.schedule.agents:
        if a.type == "vacancy":
            chain_lens.append(len(a.log))
            vac_cntr += 1
        if a.type == "actor":
            seq_lens.append(len(a.log))
            act_cntr += 1
    mean_chain_len, mean_seq_len = np.mean(chain_lens), np.mean(seq_lens)
    sd_chain_len, sd_seq_len = np.std(chain_lens), np.std(seq_lens)

    return [vac_cntr, act_cntr, mean_chain_len, mean_seq_len, sd_chain_len, sd_seq_len]


def spell_stats(model):
    """Compute average and standard deviation of average per agent spell length."""
    vacancy_mean_spell_lengths = []
    actor_mean_spell_lengths = []
    for a in model.schedule.agents:
        if len(a.log) > 1:
            # print(a.log)
            if a.type == "vacancy":
                spells = [sum(1 for i in g) for k, g in groupby(a.log)]
                spells = [i for i in spells if i != 1]
                if spells is not []:
                    vacancy_mean_spell_lengths.append(np.mean(spells))
            if a.type == "actor":
                spells = [sum(1 for i in g) for k, g in groupby(a.log)]
                spells = [i for i in spells if i != 1]
                if spells is not []:
                    actor_mean_spell_lengths.append(np.mean(spells))

    return [np.mean(vacancy_mean_spell_lengths), np.std(vacancy_mean_spell_lengths),
            np.mean(actor_mean_spell_lengths), np.std(actor_mean_spell_lengths)]
