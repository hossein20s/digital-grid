#!/usr/bin/env python

from pathfinder import seek
import csv
from math import ceil
from pprint import PrettyPrinter
import numpy as np
from random import randint
import ipdb
from shapely.geometry import Point, LineString
from skimage.io import imsave
from simplekml import Kml
from functools import reduce
from sklearn.cluster import DBSCAN
from geopy.distance import vincenty as distance
import json
import os
import sys
import math
from sympy.geometry import line as gl, point as gp
from argparse import ArgumentParser


# DEFAULTS

DEFAULT_ZOOM_SCALE = 500000
DEFAULT_CELL_WIDTH = 100
DEFAULT_MAX_STACK_SIZE = 10000

# Globals
pp = PrettyPrinter(indent=4, width= 250)
cluster = None
reference = None
scale = None

def point_to_latlong(point, point_to_grid_map):
    if point in point_to_grid_map:
        num_towers = len(point_to_grid_map[point])
        coordinates_sum = list(map(float, \
            reduce(lambda x, y: (float(x[0]) + float(y[0]), float(x[1]) + float(y[1])), \
                map(\
                    lambda tower: (cluster[tower]['center'][1], cluster[tower]['center'][0]), \
                    point_to_grid_map[point]) \
            ) \
        ))
        return (coordinates_sum[1] * 1.0 / num_towers, coordinates_sum[0] * 1.0 / num_towers)
    else:
        y = point[0]
        x = point[1]

        lt = reference[0] + y * scale[0]
        lg = reference[1] + x * scale[1]

        return (lt, lg)

def convert_pixel_latlong(points, point_to_grid_map):
    return (
        point_to_latlong(points[0], point_to_grid_map),
        point_to_latlong(points[1], point_to_grid_map)
    )
    
def is_equal_point(point1, point2):
    return point1[0] == point2[0] and point1[1] == point2[1]

def find_path_pixel(path_matrix):
    for i, row in enumerate(path_matrix):
        for j, cell in enumerate(row):
            if cell > 0:
                return (i, j)

def get_path_neighbors(cell, path_matrix):
    neighbors = []
    max_y, max_x = path_matrix.shape
    X = [-1, 0, 1]
    Y = [-1, 0, 1]
    if cell[0] == 0:
        Y = Y[1:]
    if cell[0] == path_matrix.shape[0]-1:
        Y = Y[:-1]
    if cell[1] == 0:
        X = X[1:]
    if cell[1] == path_matrix.shape[1]-1:
        X = X[:-1]

    for diff_x in X:
        for diff_y in Y:
            loc = (cell[0] + diff_y, cell[1] + diff_x)
            if not (diff_x == 0 and diff_y == 0) and path_matrix[loc] == 1:
                neighbors.append(loc)
    return neighbors

def save_edges_to_kml(edges, paths_save_file_path):
    line_count = 0
    max_path_length = 0
    kml = Kml()
    
    for idx, edge in enumerate(edges):
        coordinates_for_line = edge
        coordinates_for_line = tuple(map(lambda row: (row[1], row[0]), coordinates_for_line))
        kml.newlinestring(name='Transmisssion Line %d' % idx, description='', coords=coordinates_for_line)
        max_path_length = max(max_path_length, len(coordinates_for_line))
        line_count += 1
    
    max_path_length = max(max_path_length, len(coordinates_for_line))

    pp.pprint(kml)
    kml.save(paths_save_file_path, format=True)
    pp.pprint('paths_save_file_path: %s' % os.path.realpath(paths_save_file_path))
    pp.pprint('max_path_length: %d' % max_path_length)
    pp.pprint('line_count: %d' % line_count)

def sort_key(p):
    return p[2]

def get_farthest_line(ref_line, src_points, dst_points):
    final_ref_line = ref_line
    max_distance_till_now = 0

    for ref_point in src_points + dst_points:
        new_ref_line = ref_line.parallel_line(gp.Point(*ref_point))
        max_distance_src = max(map(lambda src_point: float(new_ref_line.distance(gp.Point(*src_point))), src_points))
        max_distance_dst = max(map(lambda dst_point: float(new_ref_line.distance(gp.Point(*dst_point))), dst_points))
        max_distance = max(max_distance_src, max_distance_dst)

        if max_distance > max_distance_till_now:
            final_ref_line = new_ref_line
            max_distance_till_now = max_distance

    return final_ref_line

def connect_edges(src, dst, edges, point_to_grid_map):
    if not src in point_to_grid_map:
        return

    src_cluster_point = cluster[point_to_grid_map[src][0]]
    dst_cluster_point = cluster[point_to_grid_map[dst][0]]
    src_points = src_cluster_point['points']
    dst_points = dst_cluster_point['points']
    edge = []

    if len(src_points) > 1 or len(dst_points) > 1:
        src_center = list(src_cluster_point['center'])
        dst_center = list(dst_cluster_point['center'])
        ref_line = gl.Line(gp.Point(*src_center), gp.Point(*dst_center))
        # src_center.reverse()
        # dst_center.reverse()
        if len(src_points) == len(dst_points):
            new_ref_line = get_farthest_line(ref_line, src_points, dst_points)

            src_points = list(map(lambda p: p + (new_ref_line.distance(gp.Point(*p)),), src_points))
            dst_points = list(map(lambda p: p + (new_ref_line.distance(gp.Point(*p)),), dst_points))

            src_points.sort(key=sort_key)
            dst_points.sort(key=sort_key)
            
            for idx, src_point in enumerate(src_points):
                edge.append((src_point[:-1], dst_points[idx][:-1]))

        if edge == []:
            edge = [(tuple(src_center), tuple(dst_center))]
    else:
        edge = [convert_pixel_latlong((src, dst), point_to_grid_map)]

    for e in edge:
        edges.append(e)
        # if len(edges) == 1438:
        #     print(src, dst)

def graph_walker(current_cell, tower_to_connect, path_matrix, edges: list, point_to_grid_map, level=0):
    # print('level=%d' % level)
    # if level > 2000:
    #     print('level=%d' % level)
    #     return level
    if current_cell is None:
        return level

    path_matrix[current_cell] = 0
    if tower_to_connect is None :
        tower_to_connect = current_cell

    for neighbor in get_path_neighbors(current_cell, path_matrix):
        if neighbor in point_to_grid_map:
            connect_edges(tower_to_connect, neighbor, edges, point_to_grid_map)
            
        next_to_connect = neighbor if neighbor in point_to_grid_map else tower_to_connect
        level = max(level, graph_walker(neighbor, next_to_connect, path_matrix, edges, point_to_grid_map, level + 1))

    return level

def run(input_file_path, zoom_scale, cell_width, max_stack_limit, output_filepath):
    global cluster
    global reference
    global scale
    # Constants
    ZOOM_SCALE = zoom_scale
    iMaxStackSize = max_stack_limit
    rows = None
    with open(os.path.realpath(input_file_path)) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    coordinates = [(float(row['latitude']), float(row['longitude'])) for row in rows]

    # Create cluster oof points
    def distance_in_m(p1, p2):
        return distance(p1, p2).m
    coordinate_cluster_output = DBSCAN(eps=45.0, min_samples=1, metric=distance_in_m, n_jobs=-1).fit(coordinates)

    number_of_clusters = max(coordinate_cluster_output.labels_) + 1

    cluster = np.full(number_of_clusters, {
            'center': (0, 0),
            'points': []
        })

    for idx, point in enumerate(coordinates):
        label = coordinate_cluster_output.labels_[idx]
        current_points = cluster[label]['points'] + [point]

        center = tuple(sum(map(np.array, current_points)) / len(current_points))

        cluster[label] = {
            'center': center,
            'points': current_points
        }

    cluster_coordinates = np.array(list(map(lambda c: (np.array(c['center']) + 90) * ZOOM_SCALE, cluster)))

    cluster_points = cluster_coordinates // cell_width

    # print(points)
    X = np.array(list(map(lambda point: point[0], cluster_points)))
    Y = np.array(list(map(lambda point: point[1], cluster_points)))

    # Top, Right, Botttom, Left
    boundaries = {
        'top': min(Y),
        'right': max(X),
        'bottom': max(Y),
        'left': min(X)
    }

    reference = (
        min(map(lambda location: location[0], cluster_coordinates)),
        min(map(lambda location: location[1], cluster_coordinates)),
        max(map(lambda location: location[0], cluster_coordinates)),
        max(map(lambda location: location[1], cluster_coordinates))
    )

    n_rows = ceil(boundaries['bottom'] - boundaries['top']) + 1
    n_columns = ceil(boundaries['right'] - boundaries['left']) + 1

    scale = (
        (reference[2] - reference[0]) / n_rows,
        (reference[3] - reference[1]) / n_columns
    )

    grid = np.full((n_columns, n_rows), 0)
    targets = np.full((n_columns, n_rows), 0)

    point_to_grid_map = {}

    for idx, point in enumerate(cluster_points):
        x, y = point
        row = int(y - boundaries['top'])
        col = n_columns - int(x - boundaries['left']) - 1
        targets[(col, row)] = 1
        # print((col, row))
        if (idx % 1000) == 0:
            grid[(col, row)] = 1

        if (col, row) in point_to_grid_map:
            point_to_grid_map[(col, row)].append(idx)
        else:
            point_to_grid_map[(col, row)] = [idx]

    origins = grid

    imsave('targets.png', targets)
    imsave('origins.png', origins)

    # Run path finder algorithm
    path_finder_results = seek(
        origins,
        targets=targets,
        weights=None,
        path_handlings='link',
        debug=False,
        film=False
    )
    paths = path_finder_results['paths']
    # Save paths pixels to png image
    imsave('output_v2.png', paths)

    # edges = path_finder_results['edges']

    all_edges = []

    # Run edge finder
    current_cell = find_path_pixel(paths)
    current_stack_size = sys.getrecursionlimit()
    sys.setrecursionlimit(iMaxStackSize)
    while not current_cell is None:
        max_level = graph_walker(current_cell, None, paths, all_edges, point_to_grid_map)
        print('max_level', max_level)
        current_cell = find_path_pixel(paths)
    sys.setrecursionlimit(current_stack_size)

    paths_save_file_path = output_filepath

    # Save all found edges to kml file
    save_edges_to_kml(all_edges, paths_save_file_path)

    # Save paths pixels to png image
    imsave('output-after_v2.png', paths)

def main(args):
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input_file', metavar='I', type=str,
                    # default='new_towers_combined.csv',
                    default='florida_poles.csv',
                    help='Input csv file path')
    parser.add_argument('-z', '--zoom_scale', metavar='Z', type=int,
                    default=DEFAULT_ZOOM_SCALE,
                    help='Zoom Scale')
    parser.add_argument('-w', '--cell_width', metavar='W', type=int,
                    default=DEFAULT_CELL_WIDTH,
                    help='Cell Width')
    parser.add_argument('-s', '--max_stack_limit', metavar='S', type=int,
                    default=DEFAULT_MAX_STACK_SIZE,
                    help='Max Recursion limit')
    parser.add_argument('-o', '--output_file', metavar='O', type=str,
                    default='paths_v2_%d.kml',
                    help='Output file path')
    
    arguments = parser.parse_args()
    arguments.output_file = arguments.output_file % arguments.cell_width
    pp.pprint(arguments)
    input_file_path = arguments.input_file
    output_file_path = arguments.output_file
    cell_width = arguments.cell_width
    max_stack_limit = arguments.max_stack_limit
    zoom_scale = arguments.zoom_scale
    run(input_file_path, zoom_scale, cell_width, max_stack_limit, output_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
    pass