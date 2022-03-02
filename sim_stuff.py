from typing import List, Tuple
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from circlify import circlify, Circle

def safe_to_list(input) -> list:
    if isinstance(input, np.ndarray):
        return input.tolist()
    elif isinstance(input, tuple):
        return list(input)
    else:
        return input


def scale_circle_pack(input: List[Circle], scale:float) -> List[Circle]:
    c_pack = []
    for c in input:
        x = c.x * scale
        y = c.y * scale
        r = c.r * scale
        c_pack.append(Circle(x, y, r))
    return c_pack


def update_equal_inner_radius(input: List[Circle], radius:float) -> List[Circle]:
    scale = radius / input[0].r
    c_pack = []
    for c in input:
        x = c.x * scale
        y = c.y * scale
        r = c.r * scale
        c_pack.append(Circle(x, y, r))
    return c_pack

def move_circles(input, x=0.0, y=0.0):
    c_pack = []
    for c in input:
        c_x = c.x + x
        c_y = c.y + y
        c_pack.append(Circle(c_x, c_y, c.r))
    return c_pack


def weighted_circles(input: List[list], enclosure_radius:float = 1.0) -> Tuple[list, list]:
    circels = []
    labels = []
    for elem in input:
        c_pack = circlify(safe_to_list(elem))
        circels.append(scale_circle_pack(c_pack, enclosure_radius))
        labels.append(np.argsort(elem))
    return circels, labels


def equal_circles(input: List[int], enclosure_radius: float = 1.0) -> list:
    circles = []
    for elem in input:
        c_pack = circlify(np.ones(elem).tolist())
        circles.append(scale_circle_pack(c_pack, enclosure_radius))
    return circles


def plane_circles(outer_circles, inner_circles):
    plane = []
    for outer_c, inner_c in zip(outer_circles, inner_circles):
        plane.append(move_circles(inner_c, outer_c.x, outer_c.y))
    return plane


def plot_circle_pack(circles, labels=None):
    # Create just a figure and only one subplot
    fig, ax = plt.subplots(figsize=(10,10))

    # Remove axes
    ax.axis('off')

    # Find axis boundaries
    lim = max(
        max(
            abs(circle.x) + circle.r,
            abs(circle.y) + circle.r,
        )
        for circle in circles
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    if labels is not None:
        labels = labels.astype(str)
        labels = np.append(labels, 'Enclosure')
        # print circles
        for circle, label in zip(circles, labels):
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2))
            plt.annotate(
                label,
                (x, y),
                va='center',
               ha='center'
            )
            print(label, r)
    else:
        # print circles
        for circle in circles:
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2))#
            print(r)

    plt.show()


def connect_planes(weighted_plane, equal_plane, labels):
    circles = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            idx = labels[i][j]
            c_connect = [
                weighted_plane[i][j],
                equal_plane[idx][i]
            ]
            circles.append(c_connect)
    return circles


def layer_planes(input: List[list] = None, weighted_radius=0.5, equal_radius=0.5):
    input = softmax(np.random.random((7, 7)), axis=-1).tolist() if input is None else input
    weighted_c, labels = weighted_circles(input, enclosure_radius=weighted_radius)
    plane_weighted = equal_circles([len(input)])[0]
    plane_weighted = update_equal_inner_radius(plane_weighted, radius=weighted_radius)

    n_connections_list = [len(elem) for elem in input]
    equal_c = equal_circles(n_connections_list, enclosure_radius=equal_radius)
    plane_equal = equal_circles([np.max(n_connections_list)])[0]
    plane_equal = update_equal_inner_radius(plane_equal, radius=equal_radius)

    weighted_plane = plane_circles(plane_weighted, weighted_c)
    equal_plane = plane_circles(plane_equal, equal_c)

    return connect_planes(weighted_plane, equal_plane, labels)

#layer_planes()
#circles = equal_circles([7,7])
#labels = np.arange(7)
#print(circles)
#plot_circle_pack(update_equal_inner_radius(circles[0], 0.5), labels)
