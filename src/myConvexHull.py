"""
Provide convex hull vertices for numpy 2d-array visualisation. The library is built
using a divide-and-conquer approach called the QuickHull, first described by
William F. Eddy in 1977.

Reference: http://www.cse.yorku.ca/~aaw/legacy/Hang/quick_hull/Algorithm.html
"""

import numpy as np

hull_points = []
hull_points_copy = []
hull_point_count = 0
center_weight = [0, 0]
center_weight_calculated = False


def convex_hull(points):
    global hull_points, hull_points_copy, hull_point_count, center_weight, center_weight_calculated

    hull_points = []
    center_weight = [0, 0]
    center_weight_calculated = False

    points.sort(axis=0)
    leftmost_point = points[0]
    rightmost_point = points[-1]
    np.delete(points, 0)
    np.delete(points, -1)

    hull_points.append(leftmost_point)
    hull_points.append(rightmost_point)

    left_point = []
    right_point = []

    for point in points:
        point_position = point_position_from_line(leftmost_point, rightmost_point, point)
        if point_position == "left":
            left_point.append(point)
        elif point_position == "right":
            right_point.append(point)
        else:
            continue

    find_hull(leftmost_point, rightmost_point, left_point)
    find_hull(leftmost_point, rightmost_point, right_point, inverted=True)

    hull_point_count = len(hull_points)
    hull_points_copy = [_ for _ in hull_points]
    hull_points.sort(key=cyclic_order_hull_points)

    return hull_points


def point_position_from_line(left_point, right_point, point_checked):
    triangle_matrix = [[left_point[0], left_point[1], 1],
                       [right_point[0], right_point[1], 1],
                       [point_checked[0], point_checked[1], 1]]

    triangle_area = np.linalg.det(triangle_matrix)
    if triangle_area > 0:
        return "left"
    elif triangle_area < 0:
        return "right"
    else:
        return "collinear"


def find_hull(left_point, right_point, points, inverted=False):
    if inverted:
        left_point, right_point = right_point, left_point

    if not points:
        return
    else:
        left_point = np.asarray(left_point)
        right_point = np.asarray(right_point)
        points = np.asarray(points)

        furthest_point = points[0]
        max_distance = (np.abs(np.cross(right_point - left_point, furthest_point - left_point))
                        / (np.linalg.norm(right_point - left_point)))

        for i in range(1, len(points)):
            point_distance = (np.abs(np.cross(right_point - left_point, points[i] - left_point))
                              / np.linalg.norm(right_point - left_point))
            if point_distance > max_distance:
                furthest_point = points[i]
                max_distance = point_distance

        hull_points.append(furthest_point)

        left_triangle_points = []
        right_triangle_points = []

        for point in points:
            if (point_position_from_line(left_point, furthest_point, point) == "left"
                    and point_position_from_line(furthest_point, right_point, point) == "right"):
                left_triangle_points.append(point)
            elif (point_position_from_line(left_point, furthest_point, point) == "right"
                    and point_position_from_line(furthest_point, right_point, point) == "left"):
                right_triangle_points.append(point)
            else:
                continue

        find_hull(left_point, furthest_point, left_triangle_points)
        find_hull(furthest_point, right_point, right_triangle_points)


def cyclic_order_hull_points(point):
    global center_weight_calculated

    if not center_weight_calculated:
        for _point in hull_points_copy:
            center_weight[0] += _point[0]
            center_weight[1] += _point[1]
        center_weight[0] /= hull_point_count
        center_weight[1] /= hull_point_count
        center_weight_calculated = True

    angle = np.arctan2([point[1] - center_weight[1]], [point[0] - center_weight[0]])

    return angle[0]
