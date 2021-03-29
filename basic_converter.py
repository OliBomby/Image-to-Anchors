from converter import *


# Basic image converter for multi-colour images
class BasicConverter(Converter):
    def __init__(self, config_file):
        super().__init__(config_file)

    def add_layer(self, anchors, layer, reverse):
        y_range = range(self.shape[0], 0, -1) if reverse else range(self.shape[0])

        for y in y_range:
            line_first_anchors_on_pixel = []
            for x in range(self.shape[1]):
                # If not in E mode, every other line reverse the scanning direction to create a zigzag pattern
                new_x = x
                direction = np.array([1, 0])
                if not self.E_MODE and y % 2 == 1:
                    new_x = self.shape[1] - x - 1
                    direction = np.array([-1, 0])

                osucoord = np.array([new_x, y]) * self.PIXEL_SPACING

                # Handle offset and rotate
                if layer == 2 and self.LAYER_2_OFFSET:
                    osucoord += np.array([4, 4])
                    if self.ROTATE:
                        osucoord -= np.array([1, 1])

                pixel = self.osu_to_pixel(osucoord)

                next_pixel = self.osu_to_pixel(osucoord + direction * self.PIXEL_SPACING)
                next_colour = self.get_anchor_code(next_pixel, layer)
                next_white = ((next_pixel[1] < 128 or len(next_colour) == 0) and self.BRIGHT_BG) or next_colour == 'W'

                # Omit transparent pixels
                if pixel[1] < 128:
                    continue

                # Find the corresponding anchor code for this luminosity level
                code = self.get_anchor_code(pixel, layer)

                if len(code) == 0:
                    continue

                # Add anchors for this pixel based on the code
                anchors_on_pixel = get_anchors_from_code(code, osucoord, direction, next_white)

                if len(line_first_anchors_on_pixel) == 0 and len(anchors_on_pixel) > 0:
                    line_first_anchors_on_pixel = anchors_on_pixel

                anchors += anchors_on_pixel

            # If in E mode, move back to the start of the line by repeating the line's first anchors
            if self.E_MODE:
                anchors += reversed(line_first_anchors_on_pixel)
