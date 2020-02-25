from mesa import Model
from random_simultaneous import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import Entity
import numpy as np
import uuid
import random


class MobilityModel(Model):
    """
    A hierarchical mobility model with some number of positions per level.
    :param trans_mat: a numpy array of (n x n+1), where n = number of levels, indicating transition
    probabilities among levels, and to the outside (hence the extra column). Each entry is a float
    eg. [ [0.4, 0.6, 0.0],
          [0.3, 0.4, 0.3] ]
    means that we have a two-level system, and that vacancies in the top level have a 0.4 change of moving to
    another top level position, 0.6 chance of moving down to level two, and no chance of leaving the system.
    :param rprobs: a list of retirement probabilities for actors at each level
    eg. [0.3, 0.2, 0.4] means that actors in the top level have a 0.3 chance of leaving the system, actors
    the next level down have a 0.2 chance of retiring, and actors at the bottom level have a 0.4 chance
    :param pos_per_lvl: list: number of positions per level
     ex. for three levels, [4,6,10] means "4 positions for level 0, 6 positions for level 1,
    10 positions for level 2.
    """

    def __init__(self, transmat, rprobs, pos_per_lvl):
        self.transmat = transmat
        self.rprobs = rprobs
        self.lvls = transmat.shape[0]  # number of levels in system
        self.pos_per_lvl = pos_per_lvl
        self.running = True
        # dict of positions per level, level is key and value is dict, each with key-value = unique_id : Entity
        self.positions = {i: {} for i in range(1, self.lvls + 1)}
        # make the positions
        for i in range(self.lvls):
            for j in range(pos_per_lvl[i]):
                # assign ID of form "unit nr - level"
                position_id = str(j + 1) + '-' + str(i + 1)
                # make position
                p = Entity(position_id, self, "position")
                p.lvl = i + 1
                self.positions[i + 1][position_id] = p

        # create dict of all actors and vacancies, key = ID, value = Entity object
        self.act_vac = {}

        # at each step agents move simultaneously and in random order
        self.schedule = SimultaneousActivation(self)

        # three sets of positions to which vacancies or actors want to go
        self.retire_pos = set()
        self.desired_pos = []

        self.datacollector = DataCollector(
            model_reporters={"Counts, Means, Standard Deviations": cnt_avrg_std})

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        self.retire_pos = set()
        self.desired_pos = []

    def populate(self, frac_vac):
        """
        Populate positions with entities.
        :param : float, [0,1]. fraction of total number of positions vacant; the rest are occupied by actors
                    e.g. 0.2 means there are, at the limit, 20% vacancies and 80% actors
                    NB: This fraction is per level, i.e. 20% of each level is vacant, NOT 20% of all positions
        """

        for lvl in self.positions.values():
            fraction = int(len(lvl) * frac_vac)
            vacancies = fraction * [True] + (len(lvl) - fraction) * [False]
            random.shuffle(vacancies)
            cntr = 0
            for p in lvl.values():
                # assign only if position unoccupied
                if p.dual == ['', '']:
                    # make entity
                    agent = Entity(uuid.uuid4(), self, "vacancy") if vacancies[cntr] \
                        else Entity(uuid.uuid4(), self, "actor")
                    self.schedule.add(agent)
                    self.act_vac[agent.unique_id] = agent
                    # make actor and position dual
                    agent.dual, agent.lvl = [p.unique_id, p.e_type], p.lvl
                    p.dual = [agent.unique_id, agent.e_type]
                    # mark position in actor's log, and vice versa
                    agent.log.append(p.unique_id)
                    p.log.append(agent.unique_id)
                    cntr += 1

    def fire_level(self, level):
        """
        For some level, replace all actors with vacancies.
        :param level: int representing some level in the hierarchy. 1 indexed, 1 is highest level.
        """
        lvl = self.positions[level]
        fired = [v.dual for v in lvl.values()]
        for f in fired:
            if f[1] == "actor":
                for o in self.schedule.agent_buffer():  # and swap with whoever's in it
                    if f[0] == o.unique_id:
                        o.retire()


def cnt_avrg_std(model):
    """Compute number of actor sequences and vacancy chains, plus mean length and variance of
        sequences and chains."""
    chain_lens, seq_lens = [], []
    act_cntr, vac_cntr = 0, 0
    for a in model.act_vac.values():
        if a.e_type == "vacancy":
            chain_lens.append(len(a.log))
            vac_cntr += 1
        if a.e_type == "actor":
            seq_lens.append(len(a.log))
            act_cntr += 1
    mean_chain_len, mean_seq_len = np.mean(chain_lens), np.mean(seq_lens)
    sd_chain_len, sd_seq_len = np.std(chain_lens), np.std(seq_lens)

    return [vac_cntr, act_cntr, mean_chain_len, mean_seq_len, sd_chain_len, sd_seq_len]
