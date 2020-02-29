"""
generalised behaviour for actors and vacancies
"""

from mesa import Agent
from random import shuffle


class Entity(Agent):
    """
    superclass for vacancy and actor agents
    not intended to be used on its own, but to inherit its methods to multiple other agents
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = ''  # type of entity: vacancy, or actor
        self.position = ''  # ID of current position
        self.log = []  # log of moves
        self.move_probability = None  # the probability with which an agent moves, float [0,1]
        self._next_state = None

    def get_next_position(self, next_level):
        """
        randomly pick a position in some level and return its ID and the ID of its current occupant.
        :param next_level: int
        """
        next_positions = list(self.model.positions[next_level].values())
        shuffle(next_positions)
        for p in next_positions:
            if p.dual[1] != self.type:  # vacancies only pick positions occupied by actors, and vice versa
                self.model.desired_positions.append(p.unique_id)  # mark position as desired
                return p.unique_id, p.dual[0]  # return positions ID and ID of current dual/occupant

    def retire(self, other):
        """
        swap with an agent and mark yourself as retired
        :param other: an Entity-class object
        """
        self.swap(other)
        self.model.schedule.add(other)  # put new entity into scheduler
        self.model.schedule.remove(self)  # take yourself out of it
        self.model.retirees[self.type][self.model.schedule.steps] = self  # mark yourself as retiree

    def swap(self, other):
        """
        swap positions with an entity
        :param other: an Entity-class object
        """
        new_position = other.position  # mark where you're going
        other.position = self.position  # put swapee in your position
        other.log.append(other.position)  # update swapee's log
        # update your old position's dual
        your_old_level = int(self.position[0])
        self.model.positions[your_old_level][self.position].dual = [other.unique_id, other.type]

        self.position = new_position  # take your new position
        self.log.append(self.position)  # update your log
        # if you have a new position, update its dual
        if self.position != '':
            your_new_level = int(self.position[0])
            self.model.positions[your_new_level][self.position].dual = [self.unique_id, self.type]

    def unmoving_update_log(self):
        """update own log if not moving."""
        self.log.append(self.log[-1])
