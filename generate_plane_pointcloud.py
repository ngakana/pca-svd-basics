import random

import numpy as np


def get_points():
    """
    :return: point cloud of points that lie within 10% of the plane: x + 4y - 30z + 60 = 0
    """

    with open('points.txt') as f:
        lines = [line.strip('\n').split('\t') for line in f.readlines()]

    nums = np.array(lines).flatten()
    nums = [int(num) for num in nums]
    points_xyz = []

    for i in range(10000):
        x = nums[random.randint(0, 9999)]
        y = nums[random.randint(0, 9999)]
        points_xyz.append([x, y, x + y + 60])

    return np.array(points_xyz)
