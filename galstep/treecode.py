import numpy as np
from collections import defaultdict


G = 44920.0
# Predefined offsets for each octant
offsets = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
], dtype=np.float64)

class oct_tree():
    __slots__ = ('COM', 'mass', 'center', 'side', 'branches', 'childless')

    def __init__(self, side, center=np.zeros(3), COM=np.zeros(3), mass=0.0):
        self.COM = np.array(COM, dtype=np.float64)
        self.mass = mass
        self.center = np.array(center, dtype=np.float64)
        self.side = side
        self.branches = [None] * 8
        self.childless = True

    def find_place(self, pos):
        dx = int(pos[0] > self.center[0])
        dy = int(pos[1] > self.center[1])
        dz = int(pos[2] > self.center[2])
        return (dx << 2) | (dy << 1) | dz  # value between 0 and 7

    def insert(self, pos, mass):
        insertion_list = [[pos, mass]]
        if self.childless and self.mass > 0:
            # Needs to reinsert the existing particle
            insertion_list.append([self.COM.copy(), self.mass])

        if self.mass > 0:
            self.COM = (self.mass * self.COM + mass * pos) / (self.mass + mass)
        else:
            self.COM = pos.copy()
        self.mass += mass

        self.childless = False

        for p, m in insertion_list:
            node = self
            while True:
                index = node.find_place(p)

                if node.branches[index] is None:
                    new_center = node.center + node.side / 4.0 * offsets[index]
                    node.branches[index] = oct_tree(
                        node.side / 2.0,
                        center=new_center,
                        COM=p,
                        mass=m
                    )
                    break
                else:
                    child = node.branches[index]
                    if child.mass > 0:
                        child.COM = (child.mass * child.COM + m * p) / (child.mass + m)
                    else:
                        child.COM = p.copy()
                    child.mass += m

                    if child.childless:
                        child.childless = False
                        node = child
                        break
                    else:
                        node = child


def potential(pos, tree):
    d = np.linalg.norm(tree.COM-pos)
    if tree.childless:
        return -G*tree.mass/d
    theta = tree.side / d
    if theta < 0.5:
        return -G*tree.mass / d
    else:
        sum_ = 0.0
        for branch in tree.branches:
            if branch != None:
                sum_ += potential(pos, branch)
        return sum_
