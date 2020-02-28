from mesa import Agent
from entity import Entity
import numpy as np
import uuid
from collections import Counter

#TODO add proper documentation


class Position(Agent):
    """Initiates positions, conceived as duals to vacancies and actors."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.dual = ['', '']  # the ID and type of current occupant
        self.log = []  # log of all previous occupants


class Actor(Entity):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "actor"
        self.retire_probability = self.model.move_probabilities["actor retirement prob"]

    def step(self):
        """"Actors only move to retire from the top level, and only if it won't leave too many vacancies."""
        if self.position[0] == "1":
            if bool(np.random.binomial(1, self.retire_probability)):
                self.model.retiree_spots.add(self.position)  # mark your position as that of retiree
                self._next_state = "retire"

    def advance(self):
        if self._next_state == "retire":
            v = Vacancy(uuid.uuid4(), self.model)
            self.retire(v)
        else:
            self.unmoving_update_log()


class Vacancy(Entity):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "vacancy"
        self.move_probability = self.model.move_probabilities["vacancy move prob"]
        self.retire_probability = self.model.move_probabilities["vacancy retire prob"]


    def step(self):
        """Vacancies only move down. When they hit bottom level, they retire."""
        if int(self.position[0]) == self.model.num_levels:  # if at bottom level,
            if bool(np.random.binomial(1, self.retire_probability)):
                self.model.retiree_spots.add(self.position)
                self._next_state = "retire"
        else:  # move down
            if bool(np.random.binomial(1, self.move_probability)):
                self._next_state = self.next_position(int(self.position[0]) + 1)

    def advance(self):
        """Move to spot."""
        if self._next_state is None:  # the ones that don't move
            self.unmoving_update_log()
            return
        if self._next_state == "retire":
            a = Actor(uuid.uuid4(), self.model)
            self.retire(a)
            return
        # if your next position is that of retiree. bow out
        if self._next_state in self.model.retiree_spots:
            self.unmoving_update_log()
            return
        # if the non-retiree position you want is oversubscribed, renounce your claim to it and bow out
        if self._next_state[0] in {k for k, v in Counter(self.model.desired_positions).items() if v > 1}:
            self.model.desired_positions.remove(self._next_state[0])
            self.unmoving_update_log()
            return
        else:  # swap with whatever's in that position
            for o in self.model.schedule.agent_buffer():
                if self._next_state[1] == o.unique_id:
                    self.swap(o)
                    return
