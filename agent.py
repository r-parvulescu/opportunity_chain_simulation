"""
classes defining position, vacancy, and actor behaviour
"""

from mesa import Agent
from entity import Entity
from numpy import random
from uuid import uuid4
from collections import Counter
import numpy as np


class Position(Agent):
    """a position that can be occupied by vacancies and actors"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.dual = ['', '']  # the ID and type of current occupant
        self.log = []  # log of occupants


class Actor(Entity):
    """an agent that can retire or be moved around by vacancies"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "actor"
        self.move_probability = self.model.move_probabilities["actor retirement probs"]

    def step(self):
        """may retire"""
        self.retire_probability = self.move_probability[int(self.position[0]) - 1]
        if bool(random.binomial(1, self.retire_probability)):
            self.model.retiree_spots.add(self.position)  # mark your position as that of a retiree
            self._next_state = "retire"

    def advance(self):
        """if retiring, call an outside vacancy to take your place, else update your log"""
        if self._next_state == "retire":
            v = Vacancy(uuid4(), self.model)
            self.retire(v)
        else:
            self.unmoving_update_log()


class Vacancy(Entity):
    """an entity that can change positions or retire"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "vacancy"
        self.move_probability = self.model.move_probabilities["vacancy move probs"]

    def step(self):
        """vacancies stay put, move in level, move down, or retire"""
        next_move = self.pick_move()
        if next_move == 1:
            self.model.retiree_spots.add(self.position)
            self._next_state = "retire"
        elif next_move == 2:  # move in same level
            self._next_state = self.get_next_position(self.position[0])
        elif next_move == 3:  # move down one level
            # if you're at bottom level already, stay puy
            if int(self.position[0]) + 1 <= self.model.num_levels:
                self._next_state = self.get_next_position(int(self.position[0]) + 1)

    def advance(self):
        """coordinate to make sure all position are occupied and no position is oversubscribed"""
        if self._next_state is None:  # the ones that don't move
            self.unmoving_update_log()
            return
        if self._next_state == "retire":  # the retirees
            a = Actor(uuid4(), self.model)
            self.retire(a)
            return
        if self._next_state in self.model.retiree_spots:  # those that want retiree spots; bow out
            self.unmoving_update_log()
            return
        # those who want oversubscribed positions; renounce your claim and bow out
        if self._next_state[0] in {k for k, v in Counter(self.model.desired_positions).items() if v > 1}:
            self.model.desired_positions.remove(self._next_state[0])
            self.unmoving_update_log()
            return
        else:  # swap with the agent in that position
            for o in self.model.schedule.agent_buffer():
                if self._next_state[1] == o.unique_id:
                    self.swap(o)
                    return


