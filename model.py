"""
The following code implements an agent-based model of mobility in hierarchical organisations,
inspired by Harrison White's "Chains of Opportunity" (1970) and social sequence analysis,
distilled in "Social Sequence Analysis: Methods and Application" (2015) by Benjamin Cornwell.
Author of Python Code:
    Pârvulescu, Radu Andrei (2020)
    rap348@cornell.edu
"""

from mesa import Model
from agent import Actor, Position, Vacancy
from random_simultaneous import SimultaneousActivation
from mesa.datacollection import DataCollector
from uuid import uuid4
from numpy import mean, std
from itertools import groupby
from random import shuffle


# start of datacollector functions


def get_percent_vacancy_per_level(model):
    """return the percentage of vacancies for each level of the mobility system"""
    actor_counts = [0 for i in range(model.num_levels)]
    vacancy_counts = [0 for i in range(model.num_levels)]
    for e in model.schedule.agents:
        e_level = int(e.position[0])
        if e.type == "actor":
            actor_counts[e_level - 1] += 1
        else:
            vacancy_counts[e_level - 1] += 1
    vacancy_percentages = {"Level " + str(i + 1): 0 for i in range(model.num_levels)}
    for i in range(model.num_levels):
        key = "Level " + str(i + 1)
        vacancy_percentages[key] = vacancy_counts[i] / (vacancy_counts[i] + actor_counts[i])
    return vacancy_percentages


# TODO the functions above and below overlap, can split off another function and simplify

def get_agent_counts(model):
    """return the total number of actors and vacancies currently in the mobility system"""
    actor_count, vacancy_count = 0, 0
    for e in model.schedule.agents:
        if e.type == "vacancy":
            vacancy_count += 1
        if e.type == "actor":
            actor_count += 1
    if actor_count + vacancy_count != sum(model.positions_per_level):
        raise ValueError('Some positions have no dual: PROBLEM!')
    return {"Actor Count": actor_count, "Vacancy Count": vacancy_count}


def get_sequence_and_chain_lengths(model):
    """return the lengths of actor sequences and vacancy chains for agents currently in the system"""
    chain_lens, seq_lens = [], []
    for a in model.schedule.agents:
        chain_lens.append(len(a.log)) if a.type == "vacancy" else seq_lens.append(len(a.log))
    return seq_lens, chain_lens


def get_sequence_and_vacancy_mean_lengths(model):
    """return the average length of actor sequences and vacancy chains for agents currently in the system"""
    lengths = get_sequence_and_chain_lengths(model)
    return {"Actor Sequence": mean(lengths[0]), "Vacancy Chain": mean(lengths[1])}


def get_sequence_and_vacancy_length_stdev(model):
    """return the standard deviations of actors sequences and vacancy chains for agents current in the system"""
    lengths = get_sequence_and_chain_lengths(model)
    return {"Actor Sequence": std(lengths[0]), "Vacanacy Chain": std(lengths[1])}


def get_mean_spell_length(some_list):
    """
    return mean spell length; "spell" == a run (at least two) of consecutive, identical entries in a list
    e.g. in [1,2,2,3,4,5,5,5] the spells are [2,2] and [5,5,5]
    """
    spell_lengths = [sum(1 for i in g) for k, g in groupby(some_list)]
    spell_lengths = [i for i in spell_lengths if i != 1]
    if not spell_lengths:
        return mean(spell_lengths)


def get_list_of_mean_spell_lengths_per_agent_type(model):
    """return lists of mean spell lengths of agent logs, one list per type of agent currently in the system"""
    actor_mean_spell_lengths = []
    vacancy_mean_spell_lengths = []
    for a in model.schedule.agents:
        if len(a.log) > 1:
            mean_spell_length = get_mean_spell_length(a.log)
            if mean_spell_length is not None:
                actor_mean_spell_lengths.append(mean_spell_length) if a.type == "actor" \
                    else vacancy_mean_spell_lengths.append(mean_spell_length)
    return actor_mean_spell_lengths, vacancy_mean_spell_lengths


# TODO can probably do away with two functions below if I'm smart about it

def get_mean_spell_lengths(model):
    """return the mean of spell length means of logs of actors and vacancies currently in the system"""
    lengths = get_list_of_mean_spell_lengths_per_agent_type(model)
    return {"Actor Sequence": mean(lengths[0]), "Vacancy Chain": mean(lengths[1])}


def get_stdev_spell_lengths(model):
    """
    return the standard deviation of spell length means of logs of actors and vacancies currently
    in the system
    """
    lengths = get_list_of_mean_spell_lengths_per_agent_type(model)
    return {"Actor Sequence": std(lengths[0]), "Vacancy Chain": std(lengths[1])}


# for the position intialiser TODO there's probably a more elegant solution that doesn't use this
def fraction_of_list(fraction, list_length):
    """Returns a list of bools split according to a float [0,1]"""
    fraction_trues = int(list_length * fraction)
    list_of_bools = fraction_trues * [True] + (list_length - fraction_trues) * [False]
    shuffle(list_of_bools)
    return list_of_bools


class MobilityModel(Model):
    """
    This model implements abstract occupational mobility in an equally abstract hierarchical organisation.
    The organisation is populated by two types of agents, vacancies and actors, each moving across positions.
    Only one actor OR one vacancy can be the dual (i.e. occupy) a position at any given point. No position
    can be devoid of duals. Actors can only choose to move when they're at the top level of the hierarchy: in
    this case they retire. When actors retire, they call a vacancy from outside the system and put it in their
    former place. Vacancies can only move down the hierarchy. When moving vacancies pick a position, and if this
    position is not already desired by another (vacancy) then our vacancy swaps places with the actor occupying
    the desired position. In this way, an actor moves up the hierarchy whenever a vacancy moves down. Finally,
    if a vacancy is in the bottom level and moves down it "retires", calling in an actor from outside the system
    and putting them in its former position.
    """

    # TODO give agents the choice to move laterally
    # TODO also need to introduce retirement probabilities for second leve, in case of firing

    def __init__(self, positions_per_level, move_probabilities, initial_vacancy_fraction, firing_schedule):
        """
        :param positions_per_level: list of positions per level ;list of ints
                                    e.g. [10,20,30] == 10 positions in level 1, 20 in level 2, etc.
        :param move_probabilities: dict of move probabilities for agents, specific format below
                                    e.g. {"actor retirement prob": 0.1, "vacancy move prob": 0.5,
                                         "vacancy retire prob": 0.2}
        :param initial_vacancy_fraction: float [0,1] telling us what percentage of positions in each level
                                         should be vacant at model initialisation
        :param firing_schedule: dict indicating what retirement probabilities should be at given steps (form below)
                                this facilitates one-off changes where portions of levels are emptied of actors
                                e.g. {"steps":5,10, "level-retire probability": (1,0.4), (2,0.4), (3,0.6)}
        """
        super().__init__()
        # set parameters
        self.num_levels = len(positions_per_level)
        self.positions_per_level = positions_per_level
        self.move_probabilities = move_probabilities
        self.vacancy_fraction = initial_vacancy_fraction
        self.firing_schedule = firing_schedule

        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={"agent_counts": get_agent_counts,
                             "percent_vacant_per_level": get_percent_vacancy_per_level,
                             "mean_lengths": get_sequence_and_vacancy_mean_lengths,
                             "lengths_std": get_sequence_and_vacancy_length_stdev,
                             "mean_spells": get_mean_spell_lengths,
                             "spells_std": get_stdev_spell_lengths})

        # TODO if I get rid of that "-" in the position IDs will simpify str to int moves throughout
        # make positions and populate them with agents
        self.positions = {i: {} for i in range(1, self.num_levels + 1)}
        for i in range(self.num_levels):
            vacancies = fraction_of_list(initial_vacancy_fraction, self.positions_per_level[i])
            for j in range(self.positions_per_level[i]):
                position_id = str(i + 1) + '-' + str(j + 1)  # position ID = level-position number
                p = Position(position_id, self)
                self.positions[i + 1][position_id] = p
                # make entity
                agent = Vacancy(uuid4(), self) if vacancies[j] else Actor(uuid4(), self)
                self.schedule.add(agent)
                # associate it with position
                agent.position = p.unique_id
                p.dual = [agent.unique_id, agent.type]
                # update logs
                agent.log.append(p.unique_id)
                p.log.append(agent.unique_id)
        self.retiree_spots = set()
        self.desired_positions = []
        self.retirees = {"actor": {}, "vacancy": {}}

    def step(self):
        # collect data before anything moves
        self.datacollector.collect(self)
        # if there are firing orders, carry them out
        if self.firing_schedule is not None:
            self.fire()
        # tell agents to step
        self.schedule.step()
        # update position logs
        for lvl in self.positions.values():
            for p in lvl.values():
                p.log.append(p.dual[0])
        # reset the sets that agents use to coordinate movement
        self.retiree_spots = set()
        self.desired_positions = []

    # TODO if I make "steps" a set I can dispense with the first for-loop
    # part of step
    def fire(self):
        """at specified step change the retirement probability of actors in specified level"""
        for step in self.firing_schedule["steps"]:
            if step - 1 == self.schedule.steps:
                for orders in self.firing_schedule["level-retire probability"]:
                    level = orders[0]
                    probability_of_retiring = orders[1]
                    for a in self.schedule.agents:
                        if (a.type == "actor") and (int(a.position[0]) == level):
                            a.move_probability = probability_of_retiring
