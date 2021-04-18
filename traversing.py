import math
import os
import time
from tools.airsim_env import AirSimUAV
from tools.dem_handling import DemHandler
import tools.cython_files.thetastar as search
import numpy as np
import functools

PI = math.pi

# data_path = os.getcwd() + '\\..\\maze_cv_data\\'
# gt_dsm = data_path + 'GTsampled22.tif'

def load_path(path_file, alt=0.2):
    pts = []
    with open(path_file, 'r') as f:
        for line in f.readlines():
            valstr = line.split(',')
            vals = [float(x) for x in valstr]
            vals[2] += alt
            pts.append(vals)
    return pts

data = '.\\navigation_system\\output'
navigator = '.\\navigation_system\\navigate.py'

demfile = os.path.join(data, 'heightmap.tif')
grownfile = os.path.join(data, 'grown_heightmap.tif')
occfile = os.path.join(data, 'occupancy_grid.tif')
path_path = os.path.join(data, 'generated_path_coordinates.txt')


def main():

    res = 0.3
    grid_origin = [-35.44869906, -97.41504192]
    h, w = 592, 1352
    handler = DemHandler(res, grid_origin, h, w)

    alt = 1.2

    hm = handler.load_heightmap(grownfile)
    occ = handler.load_heightmap(occfile)
    Planner = search.PathFinder(occ, hm)

    points = np.where(occ == 0)

    env = AirSimUAV()
    env.increase_buffer_size()

    flights = 400
    collisions = 0

    for i in range(flights):
        grid_path = None
        while grid_path is None:
            x = np.random.randint(0, len(points[0]))
            start = (points[0][x], points[1][x])

            while True:
                y = np.random.randint(0, len(points[0]))
                goal = (points[0][y], points[1][y])
                dist = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
                if 80 < dist < 250:  # at least 80 meters apart
                    break
            grid_path = Planner.thetastar(start, goal, 0.2)
            if grid_path is not None:
                path = [handler.image2world_z(x, y, hm, alt) for x, y in grid_path]

        env.reset()
        env.set_xyz(path[0])
        env.takeOff(wait=False)
        time.sleep(2)
        env.hover()

        env.draw_lines(path, duration=60)
        env.move_on_path(path, 3, wait=False)
        goal = np.array(path[-1])
        while True:
            pos = np.array(env.get_xyz())
            dist = np.linalg.norm(pos - goal)
            if env.get_collision_info(): # Collided
                collisions += 1
                break
            if dist < 2:  # Reached the goal
                break
        print('Flight %d done, %.2f %% collided' % (i, 100 * collisions / (i+1)))
    print('Collided %d out of %d' % (collisions, flights))


if __name__ == '__main__':
    main()
