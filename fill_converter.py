from converter import *
import cv2
import networkx as nx
import Graph_TSP as Graph


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

        slidercode = self.process_image(time)

        return slidercode

    def process_image(self, time):
        WIDTH = 74 * 4
        HEIGHT = 55 * 4

        windowsize = int(np.ceil(WIDTH * 0.05))
        radius = int(np.ceil(windowsize / 2))

        circle_window = np.zeros((windowsize, windowsize), np.uint8)
        cv2.circle(circle_window, (radius, radius), windowsize, (255, 255, 255), -1)

        circle_window2 = np.zeros((windowsize // 2, windowsize // 2), np.uint8)
        cv2.circle(circle_window2, (radius, radius // 2), windowsize // 2, (255, 255, 255), -1)

        circle_window_line = np.zeros((3, 3), np.uint8)
        cv2.circle(circle_window_line, (1, 1), windowsize, (255, 255, 255), -1)

        img = self.data[:, :, 0].astype(np.uint8)
        #img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        img = cv2.erode(img, circle_window)
        #dilated = cv2.dilate(dilated, circle_window)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Erode again for the scan lines
        img = cv2.erode(img, circle_window2)

        # Create horizontal scan lines for the graph
        if self.VERBOSE:
            print("Creating scan lines...")

        lines_dict = {}
        g = nx.Graph()
        counter = 0
        prev_line_lines = []
        for y in range(0, img.shape[0], windowsize):
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
                dist = len_path[n][0]
                di = 0
                for d in dist:
                    adj[ni, di] = dist[d]
                    di += 1
                ni += 1

            # Solve traveling salesman
            if self.VERBOSE:
                print("Solving TSP...")
            instance_graph = Graph.Graph_TSP(lines_dict2, adj, "frame", -1)
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
        slidercode = ""
        for contour in contours:
            if len(contour) < 2:
                continue

            anchors = []

            dist = 0
            last_coord = None
            for coord in contour:
                if last_coord is not None:
                    dist += np.linalg.norm(coord - last_coord)
                anchors.append(coord.reshape(2))
                last_coord = coord
            if len(contour) > 0:
                if last_coord is not None:
                    dist += np.linalg.norm(contour[0] - last_coord)
                anchors.append(contour[0].reshape(2))

            # Get the part of the contour which is in the top right corner
            min_y = windowsize * 2
            min_x = windowsize * 2
            c_start_pos = contour[0]
            for coord in contour:
                if coord[0, 1] < min_y:
                    min_y = coord[0, 1]
                    min_x = coord[0, 0]
                    c_start_pos = coord
                elif coord[0, 1] == min_y and coord[0, 0] < min_x:
                    min_x = coord[0, 0]
                    c_start_pos = coord
            c_start_pos = c_start_pos.reshape(2)

            # Find the filling for this contour
            closest = None
            closest_dist = 999999999999
            for filling in fillings:
                pos = filling[0]
                dist = np.linalg.norm(pos - c_start_pos)
                if dist <= closest_dist and pos[0] >= c_start_pos[0] and pos[1] >= c_start_pos[1]:
                    closest_dist = dist
                    closest = filling

            if closest is not None:
                anchors += closest

            anchor1 = anchors.pop(0)
            slidercode += "%s,%s,%s,6,0,L" % (anchor1[0], anchor1[1], int(time))

            for anchor in anchors:
                anchor_string = "|%s:%s" % (anchor[0], anchor[1])
                slidercode += anchor_string

            slidercode += ",1,%s\n" % dist

        return slidercode

