from mesa import Model
from agent import Actor, Position, Vacancy
from helpers import fraction_of_list, agent_counts, mean_lengths, length_std, percent_vacancy_per_level
from helpers import mean_spell_lengths, std_spell_lengths
from random_simultaneous import SimultaneousActivation
from mesa.datacollection import DataCollector
import uuid

#TODO add proper documentation


class MobilityModel(Model):
    """A hierarchical mobility model.
    Initialisation Parameters
    :param: positions_per_level: list of ints
    :param move_probabilities: float [0,1]
    :param initial_vacancy_fraction: float [0,1]
    :param firing_schedule: dict of form {"steps":5,10, "level-retire probability": (1,0.4), (2,0.4), (3,0.6)}
    """

    def __init__(self, positions_per_level, move_probabilities, initial_vacancy_fraction, firing_schedule):
        self.num_levels = len(positions_per_level)
        self.positions_per_level = positions_per_level
        self.move_probabilities = move_probabilities
        self.vacancy_fraction = initial_vacancy_fraction
        self.firing_schedule = firing_schedule
        self.percent_vacancy_per_level = []

        self.schedule = SimultaneousActivation(self)
        self.scheduled_actor = set()
        self.scheduled_vacancies = set()
        self.running = True

        # dict of positions per level
        self.positions = {i: {} for i in range(1, self.num_levels + 1)}

        # make positions and populate them
        for i in range(self.num_levels):
            vacancies = fraction_of_list(initial_vacancy_fraction, self.positions_per_level[i])
            for j in range(self.positions_per_level[i]):
                position_id = str(i + 1) + '-' + str(j + 1)  # position ID = level-position number
                p = Position(position_id, self)
                self.positions[i + 1][position_id] = p
                # make entity
                agent = Vacancy(uuid.uuid4(), self) if vacancies[j] else Actor(uuid.uuid4(), self)
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

        # all data results but last are tuples, by convention actor always first entry, vacancy second
        self.datacollector = DataCollector(
            model_reporters={"agent_counts": agent_counts,
                             "mean_lengths": mean_lengths,
                             "lengths_std": length_std,
                             "mean_spells": mean_spell_lengths,
                             "spells_std": std_spell_lengths,
                             "percent_vacant_per_level": percent_vacancy_per_level})

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        if self.firing_schedule is not None:
            self.fire()
        self.schedule.step()
        # update all position logs
        for lvl in self.positions.values():
            for p in lvl.values():
                p.log.append(p.dual[0])
        # reset with every step
        self.retiree_spots = set()
        self.desired_positions = []

    def fire(self):
        """
        At desired step change the retirement probability of actors in specified level.
        eg. firing schedule: {"steps": (5, 10), "level-retire probability": (1, 1.0), (2, 0.4), (3, 0.6)}
        """
        for step in self.firing_schedule["steps"]:
            if step - 1 == self.schedule.steps:
                for orders in self.firing_schedule["level-retire probability"]:
                    level = orders[0]
                    probability_of_retiring = orders[1]
                    for a in self.schedule.agents:
                        if (a.type == "actor") and (int(a.position[0]) == level):
                            a.move_probability = probability_of_retiring


