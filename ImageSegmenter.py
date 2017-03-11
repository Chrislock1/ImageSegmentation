import numpy as np
import scipy.signal as spsgn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class EdgeDetector:

    def __init__(self):
        self.gauss_filter = []

        self.sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.sobel_y = self.sobel_x.transpose()

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def gauss_filter_create(self, var, aperture):
        gauss_filter = np.zeros((aperture, aperture))
        center = aperture / float(2) + 0.5
        for x in range(aperture):
            for y in range(aperture):
                gauss_filter[x][y] = 1 / (2 * np.pi * var) * np.exp(-((y+1 - center) ** 2 + (x+1 - center) ** 2) / (2 * var))
        self.gauss_filter = gauss_filter

    def canny(self, img, var, aperture, low_thresh, high_thresh):
        self.gauss_filter_create(var, aperture)
        gray = self.rgb2gray(img)
        kern_x = spsgn.convolve2d(self.sobel_x, self.gauss_filter)
        kern_y = spsgn.convolve2d(self.sobel_y, self.gauss_filter)
        grad_x = spsgn.convolve2d(gray, kern_x)
        grad_y = spsgn.convolve2d(gray, kern_y)
        grad_abs = np.hypot(grad_x, grad_y)
        grad_arg = np.arctan2(grad_x, grad_y)
        grad_arg = np.round(grad_arg/float(np.pi/4))*45
        for x in np.nditer(grad_arg, op_flags=['readwrite']):
            if x < 0.0:
                x += 180
        edges = np.zeros_like(grad_abs)
        it = np.nditer(edges, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            try:
                if grad_arg[i[0]][i[1]] <= 0 and grad_abs[i[0]][i[1]] > grad_abs[i[0]][i[1]+1] and grad_abs[i[0]][i[1]] > grad_abs[i[0]][i[1]-1]:
                    edges[i[0]][i[1]] = grad_abs[i[0]][i[1]]
                elif grad_arg[i[0]][i[1]] == 45 and grad_abs[i[0]][i[1]] > grad_abs[i[0]+1][i[1]+1] and grad_abs[i[0]][i[1]] > grad_abs[i[0]-1][i[1]-1]:
                    edges[i[0]][i[1]] = grad_abs[i[0]][i[1]]
                elif grad_arg[i[0]][i[1]] == 90 and grad_abs[i[0]][i[1]] > grad_abs[i[0]+1][i[1]] and grad_abs[i[0]][i[1]] > grad_abs[i[0]-1][i[1]]:
                    edges[i[0]][i[1]] = grad_abs[i[0]][i[1]]
                elif grad_arg[i[0]][i[1]] == 135 and grad_abs[i[0]][i[1]] > grad_abs[i[0]+1][i[1]-1] and grad_abs[i[0]][i[1]] > grad_abs[i[0]-1][i[1]+1]:
                    edges[i[0]][i[1]] = grad_abs[i[0]][i[1]]
                it.iternext()
            except IndexError:
                it.iternext()
                pass
        top = np.max(edges)
        it = np.nditer(edges, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            if edges[i[0]][i[1]] < top*low_thresh:
                edges[i[0]][i[1]] = 0
            elif edges[i[0]][i[1]] < top*high_thresh:
                edges[i[0]][i[1]] = 1
            else:
                edges[i[0]][i[1]] = 2
            it.iternext()

        it = np.nditer(edges, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            if edges[i[0]][i[1]] == 2 or (edges[i[0]][i[1]] == 1 and (edges[i[0]-1][i[1]] == 2 or edges[i[0]-1][i[1]-1] == 2 or edges[i[0]][i[1]-1] == 2
                                           or edges[i[0]+1][i[1]-1] == 2 or edges[i[0]+1][i[1]] == 2 or edges[i[0]+1][i[1]+1] == 2
                                           or edges[i[0]][i[1]+1] == 2 or edges[i[0]-1][i[1]+1] == 2)):
                edges[i[0]][i[1]] = 1
            else:
                edges[i[0]][i[1]] = 0
            it.iternext()
        return edges


# Initialize a searcher circle spanning a radius from 2 to x
def circle(rad):
    search_array = []
    for radius in range(2, rad+1):
        f = 1 - radius
        ddf_x = 1
        ddf_y = -2 * radius
        x = 0
        y = radius
        search_array.append((0, +radius))
        search_array.append((0, -radius))
        search_array.append((+radius, 0))
        search_array.append((-radius, 0))

        while x < y:
            if f >= 0:
                y -= 1
                ddf_y += 2
                f += ddf_y
            x += 1
            ddf_x += 2
            f += ddf_x
            search_array.append((+ x, + y))
            search_array.append((- x, + y))
            search_array.append((+ x, - y))
            search_array.append((- x, - y))
            search_array.append((+ y, + x))
            search_array.append((- y, + x))
            search_array.append((+ y, - x))
            search_array.append((- y, - x))
    return search_array
search_array = circle(5)


def draw_line(x0, y0, x1, y1):
    line_array = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            line_array.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            line_array.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        line_array.append((x, y))
    return line_array

img = mpimg.imread('digging_machine.jpg')
edge = EdgeDetector()
edges = edge.canny(img, 1, 3, 0.05, 0.1)
# Find all endpoints of edges

it = np.nditer(edges, flags=['multi_index'])
n = 1
while not it.finished:
    i = it.multi_index
    if edges[i[0], i[1]] == 1:
        n += 1
        done = False
        start = i
        prev = i
        while not done:
            edges[i[0]][i[1]] = 2
            try:
                if not (i[0], i[1] + 1) == prev and edges[i[0]][i[1] + 1] == 1:
                    i = (i[0], i[1] + 1)

                elif not (i[0] + 1, i[1] + 1) == prev and edges[i[0] + 1][i[1] + 1] == 1:
                    i = (i[0] + 1, i[1] + 1)

                elif not (i[0] - 1, i[1] + 1) == prev and edges[i[0] - 1][i[1] + 1] == 1:
                    i = (i[0] - 1, i[1] + 1)

                elif not (i[0] + 1, i[1]) == prev and edges[i[0] + 1][i[1]] == 1:
                    i = (i[0] + 1, i[1])

                elif not (i[0] - 1, i[1]) == prev and edges[i[0] - 1][i[1]] == 1:
                    i = (i[0] - 1, i[1])

                elif not (i[0], i[1] - 1) == prev and edges[i[0]][i[1] - 1] == 1:
                    i = (i[0], i[1] - 1)

                elif not (i[0] - 1, i[1] - 1) == prev and edges[i[0] - 1][i[1] - 1] == 1:
                    i = (i[0] - 1, i[1] - 1)

                elif not (i[0] + 1, i[1] - 1) == prev and edges[i[0] + 1][i[1] - 1] == 1:
                    i = (i[0] + 1, i[1] - 1)
                else:
                    edges[i[0]][i[1]] = 3
                    if start:
                        i = start
                        start = False
                    else:
                        done = True
            except IndexError:
                edges[i[0]][i[1]] = 2
    it.iternext()

it = np.nditer(edges, flags=['multi_index'])
while not it.finished:
    i = it.multi_index
    if edges[i] == 3:
        for index in search_array:
            p = tuple(np.add(i, index))
            if p[0] < 0 or p[1] < 0:
                continue
            try:
                if not edges[tuple(p)] == 0:
                    line = draw_line(i[0], p[0], i[1], p[1])
                    for l in line:
                        edges[l] = 2
                    break
            except:
                continue
    it.iternext()

plt.imshow(edges)
plt.show()

