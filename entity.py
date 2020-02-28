from mesa import Agent
import random


class Entity(Agent):
    """Superclass for vacancy and actor agents."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = ''  # type of entity: vacancy, or actor
        self.position = ''  # ID of current position
        self.log = []  # log of moves
        self.move_probability = None  # the probability with which an agent moves, float [0,1]
        self._next_state = None

    def next_position(self, next_level):
        """
        Given a level, pick a random position to which you want to advance.
        :param next_level: int
        """
        next_positions = list(self.model.positions[next_level].values())
        random.shuffle(next_positions)
        for p in next_positions:
            if p.dual[1] != self.type:  # only pick if position is occupied by a dissimilar agent
                # mark position as desired
                self.model.desired_positions.append(p.unique_id)
                # and return its ID and who's in it
                return p.unique_id, p.dual[0]

    def retire(self, other):
        """
        Swap with an agent from outside the system, and leave the system.
        :param other: an Entity-class object
        """
        # swap with a dissimilar entity from the outside
        self.swap(other)
        self.model.schedule.add(other)  # put new entity into scheduler
        self.model.schedule.remove(self)  # take yourself out of it
        self.model.retirees[self.type][self.model.schedule.steps] = self  # mark yourself as retiree

    def swap(self, other):
        """
        Swap positions with another in-system entity
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
        """Update own log if not moving."""
        self.log.append(self.log[-1])
