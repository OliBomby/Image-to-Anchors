import networkx as nx
from converter import *
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


# Image converter specifically made for Bad Apple!!
# Only places white anchors on bright pixels and leaves the rest as black as possible
class SmartConverter(Converter):
    def add_layer(self, anchors, layer, reverse):
        # - Calculate horizontal lines of adjacent anchors.
        # - Calculate adjacencies and distances between the lines.
        # - Solve the travelling salesman problem on all the points.
        # - Traverse the solution cycle and place anchors for each traversed node.

        if layer > 1:
            return

        # Create horizontal scan lines for the graph
        if self.VERBOSE:
            print("Creating scan lines...")

        lines_dict = {}
        g = nx.Graph()
        counter = 0
        prev_line_lines = []
        for y in range(self.shape[0]):
            x1 = -1
            line_lines = []
            for x in range(self.shape[1]):
                osucoord = np.array([x, y]) * self.PIXEL_SPACING
                pixel = self.osu_to_pixel(osucoord)

                # Omit transparent pixels
                if pixel[0] < 128:
                    if x - x1 > 1:
                        line_lines.append(Line(x1 + 1, x - 1, y))
                    x1 = x
                elif x == self.shape[1] - 1:
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

        if counter == 0:
            return

        # Detect disconnected fragments and connect them
        if self.VERBOSE:
            print("Connecting components...")

        components = list(nx.connected_components(g))
        while len(components) > 1:
            if self.VERBOSE:
                print("number of components: %s" % len(components))
            # d contains disconnected subgraphs
            for component in components:
                # Find the closest connection between this component and another component
                closest = None
                closest_dist = 99999999999999
                for node in component:
                    for other_component in components:
                        if other_component == component:
                            continue
                        for other_node in other_component:
                            dist = line_distance(node, other_node)
                            if node[2] == other_node[2] == 0:
                                dist = min(dist, 2)
                            if node[0] == other_node[0] == 0:
                                dist = min(dist, 2)
                            if dist <= closest_dist:
                                closest_dist = dist
                                closest = (node, other_node)
                # Add the connection
                if closest is not None:
                    g.add_edge(closest[0], closest[1])

            components = list(nx.connected_components(g))

        # Use dijkstra's algorithm to calculate distance matrix with pathing
        # Each node stores for each other node the distance (int) and the previous node in the path (int)
        # This is enough to know the nearest nodes and how to get there
        # This can be stored in a 3D int array
        if self.VERBOSE:
            print("Creating distance matrix...")
        len_path = dict(nx.all_pairs_dijkstra(g))

        # Make fully connected adjacency matrix
        adj = np.empty((counter, counter))
        for n in len_path:
            dist = len_path[n][0]
            for d in dist:
                adj[n[3], d[3]] = dist[d]

        # Solve traveling salesman
        if self.VERBOSE:
            print("Solving TSP...")
        instance_graph = Graph_TSP(lines_dict, adj, "frame", -1)
        christofides = instance_graph.christofides()

        result = christofides

        # Get the part of the cycle which is in the top right corner
        min_y = 99999999999
        min_x = 99999999999
        min_index = 0
        for i in range(len(result)):
            line = lines_dict[result[i][0]]
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
        prev_line = None
        for t in result:
            line = lines_dict[t[0]].get_hashable()

            if prev_line is not None:
                paths = len_path[prev_line][1]
                self.draw_path_to_line(prev_line, line, paths[line], anchors)

            self.draw_line(line, anchors)

            prev_line = line

    def draw_line(self, line, anchors):
        for x in range(line[0], line[1] + 1):
            anchors.append(np.array((x, line[2])) * self.PIXEL_SPACING)

    def draw_path_to_line(self, start_line, end_line, path, anchors):
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
                        anchors.append(np.array(pos2) * self.PIXEL_SPACING)
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
                    anchors.append(np.array(pos2) * self.PIXEL_SPACING)
                    pos = pos2

                nx = line[0] if line[0] > prev[0][1] else line[1] if line[1] < prev[0][0] else max(line[0], prev[0][0])
                pos2 = (nx, line[2])
                if pos2 != pos and not (pos2[0] == end_line[0] and pos2[1] == end_line[2])\
                        and not (pos2[0] == start_line[1] and pos2[1] == start_line[2]):
                    anchors.append(np.array(pos2) * self.PIXEL_SPACING)
                    pos = pos2

            last_dir_up = dir_up
            prev = (line, lbound, rbound)

