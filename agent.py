from mesa import Agent
import numpy as np
import uuid
import random
from collections import Counter


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
                self.model.retire_pos.add(self.dual[0])
                return
        if bool(np.random.binomial(1, 0.5)):  # vacancies decide to move based on a fair coin toss
            nxt_lvl = self.draw_multinomial() + 1  # pick level where you're moving
            if nxt_lvl == self.model.transmat.shape[1]:  # in this case you're retiring
                # mark yourself as retiring and put your position in the set of retiree positions
                self.next = "retire"
                self.model.retire_pos.add(self.dual[0])
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
        self.log.append(your_new_dual[0])

    def retire(self):
        """Call a dissimilar agent from outside the system and swap with it."""
        dissimilar = "actor" if self.e_type == "vacancy" else "vacancy"
        e = Entity(uuid.uuid4(), self.model, dissimilar)
        # put new entity into the set of ever-existed entities and swap
        self.model.act_vac[e.unique_id] = e
        self.swap(e)
        # put the new agent into the scheduler and take yourself out of it
        self.model.schedule.add(e)
        self.model.schedule.remove(self)
