"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import traceback
import datetime
import pandas as pd
import numpy as np
from geo.sphere import distance, bearing, destination as destination_func
from math import cos, sin, radians, degrees, sqrt, copysign
from pyproj import Transformer
from os import listdir
from os.path import isfile, join
import argparse
import time
from numba import njit, prange, jit
import itertools
from numba.core import types
from numba.typed import Dict
import json
from pathlib import Path

@njit
def angle_from_velocity(v):
    """
    Given velocity (x,y) computes the counter-clockwise angle (whole angle)
    and the 0-180 degrees angle (angle)
    with the minus sign denoting counter-clockwise and the plus sign denoting clockwise
    between the vector v and the x axis.
    :param v: velocity (x,y)
    :return: counter clockwise angle (whole angle), angle (angle) with the minus sign denoting counter-clockwise
             and the plus sign denoting clockwise
    """
    angle = degrees(np.arctan2(v[1], v[0]))
    whole_angle = (360 + angle) % 360

    return np.float64(whole_angle), np.float64(angle)

@njit
def velocity_from_angle(angle, speed_magnitude):
    x_speed_component = (cos(radians(angle))) * speed_magnitude
    y_speed_component = (sin(radians(angle))) * speed_magnitude

    return np.array([x_speed_component,y_speed_component])


def intersection_point(line1_points, line2_points):
    """
    Computes the intersection point between two lines in 2D Euclidean space.
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    :param line1_points: (x1,y1),(x2,y2)
    :param line2_points: (x3,y3),(x4,y4)
    :return: intersection point (x,y)
    """
    x1 = line1_points[0][0]
    y1 = line1_points[0][1]
    x2 = line1_points[1][0]
    y2 = line1_points[1][1]
    x3 = line2_points[0][0]
    y3 = line2_points[0][1]
    x4 = line2_points[1][0]
    y4 = line2_points[1][1]

    x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    y = ((x1 * y2 - y1 * x2)*(y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return x, y

@njit
def line_segments(A):
    segments = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            segments.append([A[i], A[j]])

    return segments

@njit
def line_segments_share_point(s1,s2):
    for p1 in s1:
        for p2 in s2:
            if p1[0] == p2[0] and p1[1] == p2[1]:
                return True, p1[:2]

    return False, None

@njit
def line_segments_intersect2(s1, s2):
    l_passing_s1_interesects_s2 = False
    l_passing_s2_interesects_s1 = True
    intersects = False
    intersection_p = np.zeros((2)).astype(np.float64)

    x1 = s1[0][0]
    y1 = s1[0][1]
    x2 = s1[1][0]
    y2 = s1[1][1]
    x3 = s2[0][0]
    y3 = s2[0][1]
    x4 = s2[1][0]
    y4 = s2[1][1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t_numerator = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))
    u_numerator = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3))
    ######################################
    # if 0.0 ≤ t ≤ 1.0 intersection in t #
    # if 0.0 ≤ u ≤ 1.0 intersection in u #
    ######################################
    collinear = False
    if denominator > 0:
        if t_numerator < 0 or t_numerator > denominator:
            l_passing_s2_interesects_s1 = False
        if u_numerator > 0 and u_numerator < denominator:
            l_passing_s1_interesects_s2 = True
    elif denominator < 0:
        if t_numerator > 0 or t_numerator < denominator:
            l_passing_s2_interesects_s1 = False
        if u_numerator < 0 and u_numerator > denominator:
            l_passing_s1_interesects_s2 = True
    else:
        collinear = True
        if s2[0][0] <= s2[1][0]:
            min_x = s2[0][0]
            max_x = s2[1][0]
        else:
            min_x = s2[1][0]
            max_x = s2[0][0]

        for p in s1:
            if p[0] > min_x and p[0] < max_x:
                intersection_p = p[:2]
                intersects = True
                break
        if not intersects:
            if s1[0][0] <= s1[1][0]:
                min_x = s1[0][0]
                max_x = s1[1][0]
            else:
                min_x = s1[1][0]
                max_x = s1[0][0]

            for p in s2:
                if p[0] > min_x and p[0] < max_x:
                    intersection_p = p[:2]
                    intersects = True
                    break

        l_passing_s1_interesects_s2 = True

    if l_passing_s2_interesects_s1 and l_passing_s1_interesects_s2 and not collinear:
        t = t_numerator/denominator
        intersects = True
        intersection_p[0] = float(x1 + t*(x2-x1))
        intersection_p[1] = float(y1 + t*(y2-y1))

    if not intersects:
        intersects, intersection_p = line_segments_share_point(s1, s2)

    return l_passing_s1_interesects_s2, intersects, intersection_p

@njit
def line_segments_intersect(s1, s2):
    x1 = s1[0][0]
    y1 = s1[0][1]
    x2 = s1[1][0]
    y2 = s1[1][1]
    x3 = s2[0][0]
    y3 = s2[0][1]
    x4 = s2[1][0]
    y4 = s2[1][1]

    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    t_numerator = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))
    u_numerator = ((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))

    if denominator > 0:
        if t_numerator < 0 or t_numerator > denominator:
            return False
        if u_numerator < 0 or u_numerator > denominator:
            return False
    else:
        if t_numerator > 0 or t_numerator < denominator:
            return False
        if u_numerator > 0 or u_numerator < denominator:
            return False

    return True

@njit
def check_triangle_intersection(r1, t2):

    r1_line_segments = poly_line_segments(r1)
    t2_line_segments = line_segments(t2)

    for r1_segment in r1_line_segments:
        for t2_segment in t2_line_segments[0:2]:
            if line_segments_intersect(r1_segment, t2_segment):
                return True

    return False

def line_segments_to_points(l_segments, convex):

    points = l_segments[:, 0, :]
    if not convex:
        points = np.concatenate([points, [l_segments[-1, 1, :]]], axis=0)

    return points

@njit
def poly_line_segments(r, dim2=2, convex=True):
    if convex:
        dim0 = r.shape[0]
    else:
        dim0 = r.shape[0]-1
    segments = np.zeros((dim0,2,dim2))

    for i in range(len(r)-1):
        segments[i, 0, :] = r[i]
        segments[i, 1, :] = r[i+1]

    if convex:
        segments[-1, 0, :] = r[-1]
        segments[-1, 1, :] = r[0]

    return segments

@njit
def ownship_possible_path_points(ship_p, ship_velocity, dangles, dspeeds, t_w, h_sep_minimum):
    ship_angle, _ = angle_from_velocity(ship_velocity)
    max_angle = ship_angle+dangles[-1]
    min_angle = ship_angle+dangles[1]

    angle_bisector = (max_angle+min_angle)/2
    ship_speed = np.sqrt(np.sum(ship_velocity**2))
    ship_max_speed = ship_speed+dspeeds[-1]
    circle_radius = ship_max_speed*t_w+h_sep_minimum

    circle_point_at_bisector = ship_p+np.array([circle_radius*cos(radians(angle_bisector)),
                                                circle_radius*sin(radians(angle_bisector))])
    tangent_point_circle_center_diff = circle_point_at_bisector-ship_p
    a = tangent_at_cpab_a = -tangent_point_circle_center_diff[0]/tangent_point_circle_center_diff[1]
    c = tangent_at_cpab_c = circle_point_at_bisector[0]*tangent_point_circle_center_diff[0]/\
                            tangent_point_circle_center_diff[1] + circle_point_at_bisector[1]

    ship_max_angle_velocity = velocity_from_angle(max_angle, ship_max_speed)
    b = velocity_a = ship_max_angle_velocity[1]/ship_max_angle_velocity[0]
    d = velocity_b = ship_p[1]-velocity_a*ship_p[0]
    perpendicular_point_max_angle_x = \
        ship_p[0]-copysign(1, ship_max_angle_velocity[1])*sqrt(h_sep_minimum**2/(1+1/b**2))
    perpendicular_point_max_angle_y = ship_p[1]-(1/b)*(perpendicular_point_max_angle_x-ship_p[0])
    perpendicular_point_max_angle = [perpendicular_point_max_angle_x, perpendicular_point_max_angle_y]
    d2 = velocity_b = perpendicular_point_max_angle[1]-velocity_a*perpendicular_point_max_angle[0]

    max_angle_intersection_p = [(d2 - c) / (a - b), a * (d2 - c) / (a - b) + c]

    ship_min_angle_velocity = velocity_from_angle(min_angle, ship_max_speed)
    b = velocity_a = ship_min_angle_velocity[1]/ship_min_angle_velocity[0]
    d = velocity_b = ship_p[1]-velocity_a*ship_p[0]
    perpendicular_point_min_angle_x = \
        ship_p[0]+copysign(1, ship_max_angle_velocity[1])*sqrt(h_sep_minimum**2/(1+1/b**2))
    perpendicular_point_min_angle_y = ship_p[1]-(1/b)*(perpendicular_point_min_angle_x-ship_p[0])
    perpendicular_point_min_angle = [perpendicular_point_min_angle_x, perpendicular_point_min_angle_y]
    d2 = velocity_b = perpendicular_point_min_angle[1]-velocity_a*perpendicular_point_min_angle[0]

    min_angle_intersection_p = [(d2-c)/(a-b), a*(d2-c)/(a-b)+c]

    return np.array([perpendicular_point_max_angle,
                     perpendicular_point_min_angle,
                     min_angle_intersection_p,
                     max_angle_intersection_p])

@njit
def triangle_points(ship_p, ship_velocity, dangles, dspeeds, t_w):

    ship_angle, _ = angle_from_velocity(ship_velocity)
    max_angle = ship_angle+dangles[-1]
    min_angle = ship_angle+dangles[1]
    angle_bisector = (max_angle+min_angle)/2
    ship_speed = np.sqrt(np.sum(ship_velocity**2))
    ship_max_speed = ship_speed+dspeeds[-1]
    circle_radius = ship_max_speed*t_w
    circle_point_at_bisector = ship_p+np.array([circle_radius*cos(radians(angle_bisector)),
                                                circle_radius*sin(radians(angle_bisector))])
    tangent_point_circle_center_diff = circle_point_at_bisector-ship_p
    a = tangent_at_cpab_a = -tangent_point_circle_center_diff[0]/tangent_point_circle_center_diff[1]
    c = tangent_at_cpab_c = circle_point_at_bisector[0]*tangent_point_circle_center_diff[0]/\
                            tangent_point_circle_center_diff[1] + circle_point_at_bisector[1]
    ship_max_angle_velocity = velocity_from_angle(max_angle, ship_max_speed)
    b = velocity_a = ship_max_angle_velocity[1]/ship_max_angle_velocity[0]
    d = velocity_b = ship_p[1]-velocity_a*ship_p[0]
    max_angle_intersection_p = [(d-c)/(a-b), a*(d-c)/(a-b)+c]
    ship_min_angle_velocity = velocity_from_angle(min_angle, ship_max_speed)
    b = velocity_a = ship_min_angle_velocity[1]/ship_min_angle_velocity[0]
    d = velocity_b = ship_p[1]-velocity_a*ship_p[0]
    min_angle_intersection_p = [(d-c)/(a-b), a*(d-c)/(a-b)+c]

    return np.array([list(ship_p.astype(np.float64)), min_angle_intersection_p, max_angle_intersection_p])

@njit
def ownship_possible_path_points(own_poly_area, neighbour_triangular_area):
    """
    :param ownship_p: 2D (x,y) point
    :param neighbour_p: 2D (x,y) point
    :param ownship_velocity: 2D (x,y) velocity
    :param neighbour_velocity: 2D (x,y) velocity
    :param dangles: uncertainty angles
    :param dspeeds: uncertainty speeds
    :param t_w: time window
    :return: boolean
    """

    intersect = check_triangle_intersection(own_poly_area,neighbour_triangular_area)

    if not intersect:
        if point_in_poly(own_poly_area, neighbour_triangular_area[0]) or \
                point_in_poly(neighbour_triangular_area, own_poly_area[0]):
            return True

        return False

    return intersect

@njit
def flights_passed_intersection(A0,A1,u,v):
    """
    A0: position of ownship
    A1: position of neighbour
    u: speed vector of ownship
    v: speed vector of neighbour
    tcpa = -W0(v-u)/norm(v-u)**2
    where W0 = vec(A0A1)
    """


    A01 = A0+u*5
    A11 = A1 + v * 5

    x1 = A0[0]
    y1 = A0[1]
    x2 = A01[0]
    y2 = A01[1]

    x3 = A1[0]
    y3 = A1[1]
    x4 = A11[0]
    y4 = A11[1]

    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denominator == 0:
        return False, np.array([np.inf, np.inf])
    nominator_x = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    nominator_y = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)

    intersection_p = np.array([nominator_x / denominator, nominator_y / denominator])

    if np.all((intersection_p-A0)*u > 0) and np.all((intersection_p-A1)*v > 0):
        return False, intersection_p

    return True, intersection_p

@njit
def distance_at_crossing_point(A0, A1, V0, V1):
    passed, intersection_p = flights_passed_intersection(A0, A1, V0, V1)

    if intersection_p[0] == np.inf and intersection_p[1] == np.inf:
        return np.inf, np.inf, intersection_p

    d0 = np.sqrt(np.sum(V0 ** 2))
    d1 = np.sqrt(np.sum(V1 ** 2))
    t0 = np.sqrt(np.sum((intersection_p-A0)**2))/d0
    t1 = np.sqrt(np.sum((intersection_p - A1) ** 2)) / d1

    t = min(t0, t1)

    D = np.sqrt(np.sum(((A1-A0)+(V1-V0)*t)**2))

    return D, t, intersection_p

def distance_at_crossing_point_unit_testing():
    D, t = distance_at_crossing_point(np.array([1, 1]), np.array([14.2, 0]), np.array([3, 4]), np.array([-4, 5]))
    assert round(D) == 1

def possible_paths_intersect_unit_testing():
    d_angles = np.array([0, -1, 2])
    d_speeds = np.array([0, -3, 5])
    print("dangles: ", d_angles)
    print("dspeeds: ", d_speeds)

    assert possible_paths_intersect(np.array([1., 1.]), np.array([0, 15]),
                                    np.array([3, 4]), np.array([3, 4]),
                                    dangles=d_angles, dspeeds=d_speeds, t_w=3) == False

    assert possible_paths_intersect(np.array([1., 1.]), np.array([0, 15]),
                                    np.array([3, 4]), np.array([3, -4]),
                                    dangles=d_angles, dspeeds=d_speeds, t_w=3) == True

    assert possible_paths_intersect(np.array([1., 1.]), np.array([0, 15]),
                                    np.array([3, 4]), np.array([-3, -4]),
                                    dangles=d_angles, dspeeds=d_speeds, t_w=3) == False

    assert possible_paths_intersect(np.array([1., 1.]), np.array([5, 5]),
                                    np.array([3, 4]), np.array([1.5, 2]),
                                    dangles=d_angles, dspeeds=d_speeds, t_w=3) == True

@njit
def point_in_poly(poly, point):
    """
    from https://stackoverflow.com/questions/1119627/how-to-test-if-a-point-is-inside-of-a-convex-polygon-in-2d-integer-coordinates
    also similar https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
    :param poly: 2D polygon
    :param point: 2D point
    :return: boolean
    """
    def get_side(a, b):
        x = a[0] * b[1] - a[1] * b[0] #cosine_sign(affine_segment, affine_point)
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return None

    previous_side = None
    n_vertices = len(poly)
    for n in range(n_vertices):
        a, b = poly[n], poly[(n+1)%n_vertices]
        affine_segment = b - a
        affine_point = point - a
        current_side = get_side(affine_segment,affine_point)

        if current_side is None:
            return False #outside or over an edge
        elif previous_side is None: #first segment
            previous_side = current_side
        elif previous_side != current_side:
            return False
    return True

def line_segments_intersection_unit_testing():
    s1 = np.array([[1, 1.6], [4, 3.1]])
    s2 = np.array([[1, 0.6], [4, -0.9]])

    print(line_segments_intersect(s1, s2))

    s2 = np.array([[4, 1], [1, 2.5]])
    print(line_segments_intersect(s1, s2))

def check_triangle_intersection_unit_testing():
    r1 = np.array([[1, 1], [2.5, 4], [3.5, 4], [5, 1]])
    t2 = np.array([[1, 3], [7, 4], [7, 2]])

    print(check_triangle_intersection(r1, t2))

    t2 = np.array([[1, 3], [0, 6], [2, 6]])

    print(check_triangle_intersection(r1, t2))

    t2 = np.array([[1, 1], [3, 4], [5, 1]])

    print(check_triangle_intersection(r1, t2))

    t2 = np.array([[1, 1], [0, 6], [2, 6]])
    print(check_triangle_intersection(r1, t2))

    t2 = np.array([[2, 1.5], [3, 3], [4, 1.5]])
    print(check_triangle_intersection(r1, t2))

def poly_line_segments_unit_testing():
    r = np.array([[1, 1], [4, 1], [4, 3], [1, 2]])
    print(r)
    print(poly_line_segments(r))

def point_in_poly_unit_testing():
    poly = np.array([[1, 1], [2.5, 4], [3.5, 4], [5, 1]])
    point = np.array([2, 2])
    assert point_in_poly(poly, point) == True
    point = np.array([0, 2])
    assert point_in_poly(poly, point) == False
    point = np.array([0.5, 2.5])
    assert point_in_poly(poly, point) == False
    point = np.array([2.5, 2.5])
    assert point_in_poly(poly, point) == True

def line_segments_intersect2_unit_testing():

    line_from_s1_intersects_s2, intersect, int_p = \
        line_segments_intersect2(np.array([[1., 1.], [2., 2.]]), np.array([[4., 6.], [6., 4.]]))
    assert line_from_s1_intersects_s2 and not intersect

    line_from_s1_intersects_s2, intersect, int_p = \
        line_segments_intersect2(np.array([[4, 6], [6, 4]]), np.array([[1, 1], [2, 2]]))

    assert not (line_from_s1_intersects_s2 or intersect)

    line_from_s1_intersects_s2, intersect, _ = \
        line_segments_intersect2(np.array([[2, 2], [1, 1]]), np.array([[4, 6], [6, 4]]))
    assert line_from_s1_intersects_s2 and not intersect

    line_from_s1_intersects_s2, intersect, int_p = \
        line_segments_intersect2(np.array([[4, 4], [6, 6]]), np.array([[4, 6], [6, 4]]))
    assert line_from_s1_intersects_s2 and intersect
    print(int_p)


if __name__ == "__main__":

    line_segments_intersect2_unit_testing()
