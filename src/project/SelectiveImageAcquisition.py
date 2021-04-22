import numpy as np
import cv2

##
# Objective: Simulate different types of acquisition patterns by implementing the
# following functions.
##

def cartesianPattern(mask_size, percent):
    mask = np.zeros(mask_size)
    height = mask_size[0]
    number_of_lines = height * percent

    if percent > 50:
        mask = np.ones(mask_size)
        return mask
    # initial scan line
    for v in range(mask_size[1]):
        mask[0, v] = 1

    lines_remaining = number_of_lines - 1
    step = int(height / number_of_lines)
    line_index = step
    while lines_remaining > 0:
        for v in range(mask_size[1]):
            mask[line_index, v] = 1

        line_index += step
        lines_remaining -= 1
    return mask


def circlePattern(mask_size, radius):
    mask = np.zeros(mask_size)
    for u in range(0, mask_size[0]):
        for v in range(0, mask_size[1]):
            M = pow((u - (mask_size[0] / 2)), 2)
            N = pow((v - (mask_size[1] / 2)), 2)

            Duv = np.sqrt(M + N)
            if Duv <= radius:
                mask[u, v] = 1
            else:
                mask[u, v] = 0

    return mask


def ellipsePattern(mask_size, major_axis, minor_axis, angle):
    mask = np.zeros(mask_size)
    rx = minor_axis
    ry = major_axis
    cx = mask_size[0] / 2
    cy = mask_size[1] / 2

    for x in range(mask_size[0]):
        for y in range(mask_size[1]):
            a1 = (x - cx) * np.cos(angle)
            b1 = (y - cy) * np.sin(angle)
            res1 = pow(a1 + b1, 2) / pow(rx, 2)

            a2 = (x - cx) * np.sin(angle)
            b2 = (y - cy) * np.cos(angle)
            res2 = pow(a2 - b2, 2) / pow(ry, 2)

            if res1 + res2 <= 1:
                mask[x, y] = 1

    return mask


def bandPattern(mask_size, width, length, angle):

    cx = mask_size[0] / 2
    cy = mask_size[1] / 2

    mask = np.zeros(mask_size)

    y0 = cx - (length / 2)
    y1 = cx + (length / 2)

    x0 = cy - (width / 2)
    x1 = cy + (width / 2)

    corners = [[x0, y0], [x0, y1], [x1, y0], [x1, y1]]
    corners_rotated = []
    x_rotated_points = []
    y_rotated_points = []

    for x, y in corners:
        x_rotated = (x - cx) * np.cos(np.deg2rad(angle)) - (y - cy) * np.sin(np.deg2rad(angle))
        y_rotated = (x - cx) * np.sin(np.deg2rad(angle)) + (y - cy) * np.cos(np.deg2rad(angle))

        corners_rotated.append([x_rotated + cx, y_rotated + cy])
        x_rotated_points.append(x_rotated + cx)
        y_rotated_points.append(y_rotated + cy)

    x0 = round((min(x_rotated_points)))
    x1 = round(max(x_rotated_points))
    y0 = round(min(y_rotated_points))
    y1 = round(max(y_rotated_points))

    A = [x1, y0]
    B = [x0, y0]
    C = [x0, y1]

    for x in range(mask_size[0]):
        for y in range(mask_size[1]):
            M = [x, y]
            AB = [B[0] - A[0], B[1] - A[1]]
            AM = [M[0] - A[0], M[1] - A[1]]
            BC = [C[0] - B[0], C[1] - B[1]]
            BM = [M[0] - B[0], M[1] - B[1]]
            if 0 <= np.dot(AB, AM) <= np.dot(AB, AB) and 0 <= np.dot(BC, BM) <= np.dot(BC, BC):
                mask[x, y] = 1

    return mask


def radialPattern(mask_size, ray_count):

    mask = np.zeros(mask_size)
    cy = mask_size[0] / 2
    cx = mask_size[1] / 2

    radius = mask_size[0] / 2

    for x in range(mask_size[1]):
        mask[int(cy), x] = 1

    equidistance = np.pi / ray_count
    angle = equidistance

    ray_count -= 1
    while ray_count > 0:
        print(ray_count)
        x1 = cx + (radius * np.cos(angle))

        y1 = cy + (radius * np.sin(angle))
        if cx - round(x1) == 0: # vertical line
            for y in range(mask_size[0]):
                mask[y, int(x1)] = 1
        else:
            slope = (cy - y1) / (cx - x1)
            for y in range(mask_size[1]):
                for x in range(mask_size[0]):
                    if round(y - y1) == round(slope * (x - x1)):
                        distance = pow(radius, 2) - (pow(cx - x, 2) + pow(cy - y, 2))
                        if distance  >= 0:
                            mask[x, y] = 1

        ray_count -= 1
        angle += equidistance

    return mask


def spiralPattern(mask_size, sparsity):
    mask = np.zeros(mask_size)

    cx = mask_size[1] / 2
    cy = mask_size[0] / 2

    radius = sparsity

    while radius < mask_size[1] / 2:
        for x in range(mask_size[1]):
            for y in range(mask_size[0]):

                if abs(x - cx) + abs(y - cy) == radius:
                    mask[x, y] = 1

        radius += sparsity

    return mask
