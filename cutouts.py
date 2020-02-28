# the entity class when it moved according to transition matrices

from mesa import Model
from random_simultaneous import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import Entity, Position
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
        # currently position, its unique ID and type
        self.position = ['', '']
        # entity's occupancy log, list of unique IDs
        self.log = []
        # entity's level, empty if outside system
        self.lvl = None
        # the next position an in-system actor or vacancy would like to go to
        self.next = ''
        # entity you're going to swap with
        self.swapee = ''

    def step(self):
        """Vacancies and actors decide where they want to move."""
        if self.e_type == "actor":  # actors may retire
            if bool(np.random.binomial(1, self.model.rprobs[self.lvl - 1])):
                # mark yourself as retiring and put your position in the set of retiree positions
                self.next = "retire"
                self.model.retire_pos.add(self.position[0])
                return
        if bool(np.random.binomial(1, 0.5)):  # vacancies decide to move based on a fair coin toss
            nxt_lvl = self.draw_multinomial() + 1  # pick level where you're moving
            if nxt_lvl == self.model.transmat.shape[1]:  # in this case you're retiring
                # mark yourself as retiring and put your position in the set of retiree positions
                self.next = "retire"
                self.model.retire_pos.add(self.position[0])
                return
            else:  # pick a random position in the next level
                next_positions = list(self.model.positions[nxt_lvl].values())
                random.shuffle(next_positions)
                for p in next_positions:
                    if p.dual[1] != self.e_type:  # make sure it's occupied by a dissimilar agent
                        # mark where you want to go, add that to the model list of desired positions
                        # and mark down the identity of who you want to swap with
                        self.next = p.unique_id
                        self.swapee = p.dual[0]
                        self.model.desired_pos.append(self.next)
                        return

    def advance(self):
        """Vacancies and actors actually move."""
        # if retiring, just do it
        if self.next == "retire":
            self.retire()
            self.next = ''
            return
        # if your next position is that of a retiring person. bow out
        if self.next in self.model.retire_pos:
            self.next = ''
            return
        # if the non-retiree position you want is oversubscribed
        if self.next in {k for k, v in Counter(self.model.desired_pos).items() if v > 1}:
            # renounce your claim to it and bow out
            self.model.desired_pos.remove(self.next)
            self.next = ''
            return
        else:  # move to that position
            for o in self.model.schedule.agent_buffer():  # and swap with whoever's in it
                if self.swapee == o.unique_id:
                    self.swap(o)
                    self.next = ''
                    return

    def draw_multinomial(self):
        """
        Draw from multinomial distribution by cutting line into segments, each of length
        given by probability vector, and uniform-random throwing darts at segmented line.
        :return: the draw, an int
        """
        prob_vector = self.model.transmat[self.lvl - 1, :]
        cum_sum = np.cumsum(prob_vector)
        cum_sum = np.insert(cum_sum, 0, 0)
        # throw random dart
        rd = np.random.uniform(0.0, 1.0)
        # see where dart hit
        m = np.asarray(cum_sum < rd).nonzero()[0]
        mn_draw = m[len(m) - 1]
        return mn_draw


# the model with transition matrixes


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
                # assign ID of form "level - position nr"
                position_id = str(i + 1) + '-' + str(j + 1)
                # make position
                p = Entity(position_id, self, "position")
                p.lvl = i + 1
                self.positions[i + 1][position_id] = p

        # create dict of all actors and vacancies, key = ID, value = Entity object
        self.act_vac = {}

        # at each step agents move simultaneously and in random order
        self.schedule = SimultaneousActivation(self)

        # three sets of positions to which vacancies or actors want to go
        self.retiree_spots = set()
        self.desired_positions = []

        # dict of dicts for retired agents and vacancies, entities keyd by time of retirement
        self.retired = {"actors": {}, "vacancies": {}}

        self.datacollector = DataCollector(
            model_reporters={"Counts, Means, Standard Deviations": cnt_avrg_std})

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        self.retiree_spots = set()
        self.desired_positions = []

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
                    agent.position, agent.lvl = [p.unique_id, p.e_type], p.lvl
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



### original plotters

def percent_vacant_per_level(model):
    percent_data = model.datacollector.model_vars["vacancy_percent_per_level"]
    level_1_vacancy_pecent = []
    level_2_vacancy_percentage = []
    level_3_vacancy_pecentage = []
    for r in percent_data:
        level_1_vacancy_pecent.append(r[0])
        level_2_vacancy_percentage.append(r[1])
        level_3_vacancy_pecentage.append(r[2])

    ax = plt.subplot(1, 1, 1)
    plt.plot(level_1_vacancy_pecent, "r-", label="Level 1")
    plt.plot(level_2_vacancy_percentage, "b-", label="Level 2")
    plt.plot(level_3_vacancy_pecentage, "g-", label="Level 3")

    chart_box1 = ax.get_position()
    ax.set_position([chart_box1.x0, chart_box1.y0, chart_box1.width * 0.9, chart_box1.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.8), shadow=True, ncol=1, fontsize="small")
    plt.show()


def spell_show(model):
    spell_data = model.datacollector.model_vars["Means and Standard Deviations of Spells"]
    vacancy_spell_means = []
    vacancy_spell_sds = []
    actor_spell_means = []
    actor_spell_sds = []
    for s in spell_data:
        vacancy_spell_means.append(s[0])
        vacancy_spell_sds.append(s[1])
        actor_spell_means.append(s[2])
        actor_spell_sds.append(s[3])

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(actor_spell_means, 'r--', label="Actor Spell Means")
    plt.plot(vacancy_spell_means, 'b-', label="Vacancy Spell Means")

    chart_box1 = ax1.get_position()
    ax1.set_position([chart_box1.x0, chart_box1.y0, chart_box1.width * 0.6, chart_box1.height])
    ax1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    ax2 = plt.subplot(2, 1, 2)
    plt.plot(actor_spell_sds, 'r--', label="Actor Spell StdDev")
    plt.plot(vacancy_spell_sds, 'b-', label="Vacancy Spell StdDev")

    chart_box2 = ax2.get_position()
    ax2.set_position([chart_box2.x0, chart_box2.y0, chart_box2.width * 0.6, chart_box2.height])
    ax2.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.show()


def basic_data_show(model):
    basic_data = model.datacollector.model_vars["Counts, Means, Standard Deviations"]
    actor_counts = []
    vacancy_counts = []
    avgs_seq_lens = []
    avgs_chain_lens = []
    sds_seq_lens = []
    sds_chain_lens = []
    for a in basic_data:
        vacancy_counts.append(a[0])
        actor_counts.append(a[1])
        avgs_chain_lens.append(a[2])
        avgs_seq_lens.append(a[3])
        sds_chain_lens.append(a[4])
        sds_seq_lens.append(a[5])

    plt.plot(sds_seq_lens, 'r--', sds_chain_lens, 'b-')

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(vacancy_counts, 'b-', label="Vacancies in System")
    plt.plot(actor_counts, 'r-', label="Actors in System")

    chart_box1 = ax1.get_position()
    ax1.set_position([chart_box1.x0, chart_box1.y0, chart_box1.width * 0.6, chart_box1.height])
    ax1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, fontsize="medium")

    ax2 = plt.subplot(3, 1, 2)
    plt.plot(avgs_chain_lens, 'b-', label="Average Chain Lengths")
    plt.plot(avgs_seq_lens, 'r-', label="Average Sequence Lengths")

    chart_box2 = ax2.get_position()
    ax2.set_position([chart_box2.x0, chart_box2.y0, chart_box2.width * 0.6, chart_box2.height])
    ax2.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, fontsize="medium")

    ax3 = plt.subplot(3, 1, 3)
    plt.plot(sds_chain_lens, 'b-', label="StDev Chain Lengths")
    plt.plot(sds_seq_lens, 'r-', label="StDev Sequence Lengths")

    chart_box3 = ax3.get_position()
    ax3.set_position([chart_box3.x0, chart_box3.y0, chart_box3.width * 0.6, chart_box3.height])
    ax3.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, fontsize="medium")

    plt.show()