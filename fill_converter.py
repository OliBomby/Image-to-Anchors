import sys
import traceback

from converter import *
import cv2
import networkx as nx
from TSP_Solver import Graph_TSP


class Line:
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.i = -1

    def get_hashable(self):
        return self.x1, self.x2, self.y, self.i

    def __str__(self):
        return '%s:%s,%s,%s' % (self.x1, self.x2, self.y, self.i)


def line_distance(line1, line2):
    return abs(line2[2] - line1[2]) if line2[0] <= line1[1] and line2[1] >= line1[0] else \
                            min(abs(line2[0] - line1[1]) + abs(line2[2] - line1[2]),
                                abs(line2[1] - line1[0]) + abs(line2[2] - line1[2]))


def draw_line(line, anchors):
    anchors.append(np.array((line[0], line[2])))
    anchors.append(np.array((line[1], line[2])))


def draw_path_to_line(start_line, end_line, path, anchors):
    pos = (start_line[1], start_line[2])
    lbound = None
    rbound = None
    prev = None
    last_dir_up = None
    for line in path:
        dir_up = prev[0][2] < line[2] if prev is not None else None

        # Interpolate and check if its inside the line
        if lbound is None or rbound is None:
            lbound = line[0]
            rbound = line[1]
        elif pos[1] != line[2]:
            lbi = (line[0] - pos[0]) / abs(line[2] - pos[1]) + pos[0]
            rbi = (line[1] - pos[0]) / abs(line[2] - pos[1]) + pos[0]

            lbound = max(lbi, lbound)
            rbound = min(rbi, rbound)

            # If this is the last line then it will add a line straight to the start of the end line
            # we do this check to make sure this doesn't pass through any black area
            if line == end_line and lbi < lbound:
                lbound = rbound + 1

        if lbound > rbound or dir_up != last_dir_up:
            if prev[0] != start_line and prev[0] != end_line:
                nx = prev[2] if (line[0]+line[1])/2 > pos[0] else prev[1]
                pos2 = (round((nx - pos[0]) * abs(prev[0][2] - pos[1]) + pos[0]), prev[0][2])
                if pos2 != pos and not (pos2[0] == end_line[0] and pos2[1] == end_line[2])\
                        and not (pos2[0] == start_line[1] and pos2[1] == start_line[2]):
                    anchors.append(np.array(pos2))
                    pos = pos2
            lbound = line[0]
            rbound = line[1]

        # Make sure component connections go through the same two points every time
        # so only one connection line is visible (all overlap)
        if prev is not None and line_distance(prev[0], line) > 1:
            # Add an anchor at the closest point
            nx = prev[0][1] if line[0] > prev[0][1] else prev[0][0] if line[1] < prev[0][0] else max(line[0], prev[0][0])
            pos2 = (nx, prev[0][2])
            if pos2 != pos and not (pos2[0] == end_line[0] and pos2[1] == end_line[2])\
                    and not (pos2[0] == start_line[1] and pos2[1] == start_line[2]):
                anchors.append(np.array(pos2))
                pos = pos2

            nx = line[0] if line[0] > prev[0][1] else line[1] if line[1] < prev[0][0] else max(line[0], prev[0][0])
            pos2 = (nx, line[2])
            if pos2 != pos and not (pos2[0] == end_line[0] and pos2[1] == end_line[2])\
                    and not (pos2[0] == start_line[1] and pos2[1] == start_line[2]):
                anchors.append(np.array(pos2))
                pos = pos2

        last_dir_up = dir_up
        prev = (line, lbound, rbound)


def triangle_area(a, b, c):
    return abs(0.5 * (a[0] * (b[1] - c[1]) +
                  b[0] * (c[1] - a[1]) +
                  c[0] * (a[1] - b[1])))


def collinear(a, b, c):
    return abs((a[1] - b[1]) * (a[0] - c[0]) - (a[1] - c[1]) * (a[0] - b[0])) <= 1e-6


# Basic image converter for multi-colour images
class FillConverter(Converter):
    def __init__(self, config_file):
        super().__init__(config_file)

    def convert(self, path, time=0, start_pos=None):
        self.load_image(path)
        self.prepare_image()

        if self.VERBOSE:
            print("Image resolution: ", self.imgshape)
            print("Slider resolution:", self.shape)
            print("Slider size: ", self.shape * self.PIXEL_SPACING)

        try:
            return self.process_image(time)
        except Exception:
            print("Exception in user code:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)

    def process_image(self, time):
        SCALE = float(self.CONFIG['SETTINGS']['SCALE'])
        WINDOW_WIDTH = 70 * 4 / SCALE

        windowsize = int(np.ceil(WINDOW_WIDTH * 0.05))
        radius = int(np.ceil(windowsize / 2))

        circle_window = np.zeros((windowsize, windowsize), np.uint8)
        cv2.circle(circle_window, (radius, radius), windowsize, (255, 255, 255), -1)

        circle_window2 = np.zeros((windowsize // 3, windowsize // 3), np.uint8)
        cv2.circle(circle_window2, (radius // 3, radius // 3), windowsize // 3, (255, 255, 255), -1)

        img = self.data[:, :, 0].astype(np.uint8)
        #img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Erode the image
        img = cv2.erode(img, circle_window)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        hierarchy = hierarchy[0]
        contours = [contour.reshape(-1, 2) for contour in contours]

        # Erode again for the scan lines
        img = cv2.erode(img, circle_window2)
        line_width = int(windowsize * 2 / 3)

        # Create horizontal scan lines for the graph
        if self.VERBOSE:
            print("Creating scan lines...")

        lines_dict = {}
        g = nx.Graph()
        counter = 0
        prev_line_lines = []
        for y in range(int(time % line_width), img.shape[0], line_width):
            x1 = -1
            line_lines = []
            for x in range(img.shape[1]):
                pixel = img[y, x]

                # Omit black pixels
                if pixel < 128:
                    if x - x1 > 1:
                        line_lines.append(Line(x1 + 1, x - 1, y))
                    x1 = x
                elif x == img.shape[1] - 1:
                    line_lines.append(Line(x1 + 1, x, y))

            for line in line_lines:
                line.i = counter
                lines_dict[counter] = line
                g.add_node(line.get_hashable(), weight=1)
                counter += 1

                # Add direct neighbors
                for prev_line in prev_line_lines:
                    if prev_line.x1 <= line.x2 and prev_line.x2 >= line.x1:
                        # Lines are touching
                        g.add_edge(line.get_hashable(), prev_line.get_hashable())

            prev_line_lines = line_lines

        if self.VERBOSE:
            print("counter: %s" % counter)

        components = [g.subgraph(c).copy() for c in nx.connected_components(g)]

        fillings = []
        for c in components:
            counter = 0
            lines_dict2 = {}
            for n in c:
                lines_dict2[counter] = lines_dict[n[3]]
                counter += 1

            if counter == 0:
                continue

            # Use dijkstra's algorithm to calculate distance matrix with pathing
            # Each node stores for each other node the distance (int) and the previous node in the path (int)
            # This is enough to know the nearest nodes and how to get there
            # This can be stored in a 3D int array
            if self.VERBOSE:
                print("Creating distance matrix...")
            len_path = dict(nx.all_pairs_dijkstra(c))

            # Make fully connected adjacency matrix
            adj = np.empty((counter, counter))
            ni = 0
            for n in len_path:
                total_dist = len_path[n][0]
                di = 0
                for d in total_dist:
                    adj[ni, di] = total_dist[d]
                    di += 1
                ni += 1

            # Solve traveling salesman
            if self.VERBOSE:
                print("Solving TSP...")
            instance_graph = Graph_TSP(lines_dict2, adj, "frame", -1)
            christofides = instance_graph.christofides()

            result = christofides

            # Get the part of the cycle which is in the top right corner
            min_y = 99999999999
            min_x = 99999999999
            min_index = 0
            for i in range(len(result)):
                line = lines_dict2[result[i][0]]
                if line.y < min_y:
                    min_y = line.y
                    min_x = line.x1
                    min_index = i
                elif line.y == min_y and line.x1 < min_x:
                    min_x = line.x1
                    min_index = i

            result = result[min_index:] + result[:min_index]

            # Draw anchors
            if self.VERBOSE:
                print("Drawing anchors...")

            anchors = []
            if len(result) > 0:
                prev_line = None
                for t in result:
                    line = lines_dict2[t[0]].get_hashable()

                    if prev_line is not None:
                        paths = len_path[prev_line][1]
                        draw_path_to_line(prev_line, line, paths[line], anchors)

                    draw_line(line, anchors)

                    prev_line = line

                # Add path to the start again
                line = lines_dict2[result[0][0]].get_hashable()
                if prev_line is not None:
                    paths = len_path[prev_line][1]
                    draw_path_to_line(prev_line, line, paths[line], anchors)
                anchors.append(np.array((line[0], line[2])))
            else:
                line = lines_dict2[0].get_hashable()
                draw_line(line, anchors)
                anchors.append(np.array((line[0], line[2])))

            fillings.append(anchors)

        # Make sliders
        if self.VERBOSE:
            print("Making sliders...")

        used_fillings = []
        slidercode = ""
        # Loop through all outer contours using the hierarchy
        next_outer = 0
        while next_outer != -1:
            current_outer = next_outer
            contour = contours[current_outer]

            # Get the index of the next outer contour from the hierarchy
            next_outer = hierarchy[current_outer][0]

            if len(contour) < 2:
                continue

            # Get all the contours that are child of this contour
            child_contours = []
            next_child = hierarchy[current_outer][2]
            while next_child != -1:
                child_contours.append(contours[next_child])
                next_child = hierarchy[next_child][0]

            # Get the filling that fits inside this contour and the place to connect them
            c_fillings = []
            max_dist = windowsize * 1.5
            c_start_indices = []
            for i in range(len(contour)):
                coord = contour[i]

                # Find the filling for this contour start pos
                for filling in fillings:
                    pos = filling[0]
                    dist = np.linalg.norm(pos - coord, ord=np.inf)
                    if dist <= max_dist and list(pos) not in used_fillings:
                        c_fillings.append(filling)
                        c_start_indices.append(i)
                        used_fillings.append(list(pos))

            # Find the bet way to connect the child contours to the outer contour
            # or add child contours to other child contours
            # First we add any children to other children
            added_children = []
            for k, child in enumerate(child_contours):
                # This is the top left point of the child contour
                pos = child[0]
                # Find the closest point on the outer contour that is to the top left of this point
                best_dist = np.inf
                best_i = -1
                for i, coord in enumerate(contour):
                    dist = np.linalg.norm(pos - coord)
                    if dist <= best_dist and coord[0] <= pos[0] and coord[1] <= pos[1]:
                        best_dist = dist
                        best_i = i
                # Check if any other child is closer than the outer contour
                best_child = -1
                for j, other_child in enumerate(child_contours):
                    if k == j:
                        continue
                    for i, coord in enumerate(other_child):
                        dist = np.linalg.norm(pos - coord)
                        if dist <= best_dist and coord[0] <= pos[0] and coord[1] <= pos[1]:
                            best_dist = dist
                            best_i = i
                            best_child = j
                if best_child != -1:
                    child_contours[best_child] = np.vstack((child_contours[best_child][0:best_i], child, child[0], child_contours[best_child][best_i:-1]))
                    added_children.append(k)

            # Add the remaining children to the outer contour
            for k, child in enumerate(child_contours):
                if k in added_children:
                    continue
                # This is the top left point of the child contour
                pos = child[0]
                # Find the closest point on the outer contour that is to the top left of this point
                best_dist = np.inf
                best_i = -1
                for i, coord in enumerate(contour):
                    dist = np.linalg.norm(pos - coord)
                    if dist <= best_dist and coord[0] <= pos[0] and coord[1] <= pos[1]:
                        best_dist = dist
                        best_i = i

                # Add the index of the best connection point to the list of start indices
                # and add the child contour to the list of fillings so it gets added
                c_fillings.append(list(child) + [child[0]])
                c_start_indices.append(best_i)

            # Combine everything into a list of anchors
            anchors = []
            for i, coord in enumerate(contour):
                anchors.append(coord)

                while i in c_start_indices:
                    f_index = c_start_indices.index(i)
                    anchors += c_fillings.pop(f_index)
                    c_start_indices.pop(f_index)
                    anchors.append(coord)

            if len(contour) > 0:
                anchors.append(contour[0])

            anchor1 = anchors.pop(0)
            anchor1_rounded = np.round(anchors.pop(0) * SCALE)
            slidercode += "%s,%s,%s,6,0,L" % (int(anchor1_rounded[0]), int(anchor1_rounded[1]), int(time))

            total_dist = 0
            last_anchor_rounded = anchor1_rounded
            last_anchor = anchor1
            lastlast_anchor = anchor1
            for i, anchor in enumerate(anchors):
                round_anchor = np.round(anchor * SCALE)

                # We skip some anchors if they are ugly
                if 0 < i < len(anchors) - 1:
                    next_anchor = anchors[i + 1]
                    nextnext_anchor = anchors[i + 2] if i + 2 < len(anchors) else next_anchor
                    big_space = np.linalg.norm(next_anchor - anchor) > 10
                    next_big_space = np.linalg.norm(next_anchor - nextnext_anchor) > 10
                    if ((np.linalg.norm(last_anchor - lastlast_anchor) > np.linalg.norm(anchor - next_anchor) or not big_space) and
                            np.linalg.norm(last_anchor - anchor, ord=np.inf) <= 1) or\
                        (np.linalg.norm(last_anchor - anchor) < np.linalg.norm(nextnext_anchor - next_anchor) and next_big_space and
                            np.linalg.norm(next_anchor - anchor, ord=np.inf) <= 1) or\
                        (triangle_area(last_anchor, anchor, next_anchor) < 1e-6 and
                            np.linalg.norm(last_anchor - anchor) <= np.linalg.norm(next_anchor - last_anchor)):
                        continue

                if last_anchor_rounded is not None:
                    dist = np.linalg.norm(round_anchor - last_anchor_rounded)
                    total_dist += dist

                anchor_string = "|%s:%s" % (int(round_anchor[0]), int(round_anchor[1]))
                slidercode += anchor_string

                lastlast_anchor = last_anchor
                last_anchor_rounded = round_anchor
                last_anchor = anchor

            slidercode += ",1,%s\n" % total_dist

        if self.VERBOSE:
            print("num fillings: %s" % len(fillings))
            print("num used fillings: %s" % len(used_fillings))
            print("num contours: %s" % len(contours))

        return slidercode

