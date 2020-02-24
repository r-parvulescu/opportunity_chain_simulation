# mobility_wave.py
# simulates occupational mobility in a hierarchical structure, with special
# focus on movements of people and opportunities

from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
import uuid
import random


class Entity(Agent):
    """
    A position, vacancy, or person, each with distinct traits
    :param unique_id: int
    :param model: model you're in
    :param e_type: string denoting entity type: position, vacancy, or actor
    """

    def __init__(self, unique_id, model, e_type):
        super().__init__(unique_id, model)
        # type of entity: position, vacancy, or actor
        self.e_type = e_type
        # currently dual, its unique ID and type
        self.dual = ['', '']
        # entity's occupancy log, list of unique IDs
        self.log = []
        # entity's level, empty if outside system
        self.lvl = None

    def step(self):
        """Vacancies and actors move."""
        print(self.lvl)
        print(self.dual)
        print(self.e_type)
        if self.e_type == "actor":  # actors may retire
            if bool(np.random.binomial(1, self.model.rprobs[self.lvl-1])):
                self.retire()
                return
        if bool(np.random.binomial(1, 0.5)):  # vacancies decide to move based on a fair coin toss
            nxt_lvl = self.draw_multinomial() + 1  # pick level where you're moving
            if nxt_lvl == self.model.transmat.shape[1]:  # in this case you're retiring
                #self.retire()
                return
            else:  # pick a position in the next level that is occupied by a dissimilar agent and swap
                for p in self.model.positions[nxt_lvl].values():  # NB: iteration over dict is random
                    if p.dual[1] != self.e_type:
                        other = self.model.act_vac[p.dual[0]]
                        self.swap(other)
                        return



    def draw_multinomial(self):
        """
        Draw from multinomial distribution by cutting line into segments, each of length
        given by probability vector, and uniform-random throwing darts at segmented line.
        :param prob_vector: a list of floats that sum to one
        :return: the draw, an int
        """
        prob_vector = self.model.transmat[self.lvl-1, :]
        cum_sum = np.cumsum(prob_vector)
        cum_sum = np.insert(cum_sum, 0, 0)
        # throw random dart
        rd = np.random.uniform(0.0, 1.0)
        # see where dart hit
        m = np.asarray(cum_sum < rd).nonzero()[0]
        mn_draw = m[len(m) - 1]
        return mn_draw

    def swap(self, other):
        """
        Swap positions with another in-system entity
        :param other: an Entity-class object
        :return: None
        """
        # mark where you're going
        your_new_dual, your_new_lvl = other.dual, other.lvl
        # put swapee in your position, updating its log
        other.dual, other.lvl = self.dual, self.lvl
        other.log.append(self.dual[0])
        # and you take its old position, updating your log
        self.dual, self.lvl = your_new_dual, your_new_lvl
        self.log.append(self.dual[0])


    def retire(self):
        """Call a dissimilar agent from outside the system and swap with it."""
        dissimilar = "actor" if self.e_type == "vacancy" else "vacancy"
        e = Entity(uuid.uuid4(), self.model, dissimilar)
        # put new entity into the set of ever-existed entities and swap
        self.model.act_vac[e.unique_id] = e
        self.swap(e)
        # put the new agent into the scheduler and take yourself out of it
        #self.model.schedule.add(e)
        #self.model.schedule.remove(self)

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

        # create dict of actors and vacancies, key = ID, value = Entity object
        self.act_vac = {}

        # at each step agents move in random order
        self.schedule = RandomActivation(self)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()

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


transmat = np.array(([0.5, 0.1, 0.1, 0.3], [0.3, 0.3, 0.3, 0.1], [0.1, 0.3, 0.3, 0.3]))
#transmat = np.array(([0.5, 0.2, 0.3, 0.0], [0.3, 0.4, 0.3, 0.10], [0.1, 0.5, 0.4, 0.0]))
rprobs = [0.1, 0.1, 0.1]
pos_per_lvl = [10, 20, 30]
model = MobilityModel(transmat, rprobs, pos_per_lvl)
model.populate((0.23))
for i in range(40):
    model.step()
