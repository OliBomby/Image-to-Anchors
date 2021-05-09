from PIL import Image
import numpy as np


def get_anchors_from_code(code, osucoord, direction, next_white):
    anchors_on_pixel = []
    for i in range(len(code)):
        c = code[i]
        if c == 'R':
            if '_' not in code[i:] and '_' in code and next_white:
                anchors_on_pixel.append(osucoord.copy())
                break

            anchors_on_pixel.append(osucoord.copy())
            anchors_on_pixel.append(osucoord.copy())
        elif c == 'W':
            anchors_on_pixel.append(osucoord.copy())
        elif c == '_':
            osucoord += direction
    return anchors_on_pixel


class Converter:
    def __init__(self, config):
        self.data = None
        self.imgshape = None
        self.downscale_factor = None
        self.shape = None

        self.CONFIG = config
        self.PIXEL_SPACING = int(config['SETTINGS']['PIXEL_SPACING'])
        self.ROTATE = config['SETTINGS'].getboolean('ROTATE')
        self.LAYER_2_OFFSET = config['SETTINGS'].getboolean('LAYER_2_OFFSET')
        self.BRIGHT_BG = config['SETTINGS'].getboolean('BRIGHT_BG')
        self.E_MODE = config['SETTINGS'].getboolean('E_MODE')
        self.VERBOSE = config['SETTINGS'].getboolean('VERBOSE')

        self.SLIDER_MAX_WIDTH = int(config['SETTINGS']['SLIDER_MAX_WIDTH'])
        self.SLIDER_MAX_HEIGHT = int(config['SETTINGS']['SLIDER_MAX_HEIGHT'])

        self.LAYER_1_LEVELS = list(map(int, config['LAYER 1'].keys()))
        self.LAYER_1_CODES = list(config['LAYER 1'].values())
        self.LAYER_2_LEVELS = list(map(int, config['LAYER 2'].keys()))
        self.LAYER_2_CODES = list(config['LAYER 2'].values())

    def convert(self, path, time=0, start_pos=None):
        self.load_image(path)
        self.prepare_image()

        if self.VERBOSE:
            print("Image resolution: ", self.imgshape)
            print("Slider resolution:", self.shape)
            print("Slider size: ", self.shape * self.PIXEL_SPACING)

        anchors = []

        self.add_layer(anchors, 1, False)
        self.add_layer(anchors, 2, True)

        if len(anchors) < 2:
            print("Insufficient anchors generated for slider.")
            return

        if start_pos is not None:
            if not np.equal(start_pos, anchors[0]).all():
                anchors.insert(0, start_pos)

        anchor1 = anchors.pop(0)
        slidercode = "%s,%s,%s,6,0,L" % (anchor1[0], anchor1[1], int(time))

        for anchor in anchors:
            anchor_string = "|%s:%s" % (anchor[0], anchor[1])
            slidercode += anchor_string

        slidercode += ",1,1\n"

        return slidercode

    def load_image(self, infilename):
        img = Image.open(infilename)
        img.load()
        self.data = np.asarray(img, dtype="int32")

    def prepare_image(self):
        # Handle grayscale and colour images
        if len(self.data.shape) > 2:
            data_color = self.data[:, :, :3]
            # Convert colour to gray scale with relative luminance
            data_gray = np.average(data_color, axis=2, weights=[0.2126, 0.7152, 0.0722])

            # If there is no alpha channel, just make everything maximum opacity
            if self.data.shape[2] > 3:
                data_a = self.data[:, :, 3]
            else:
                data_a = np.full(data_gray.shape[:2], 255)
        else:
            data_gray = self.data
            data_a = np.full(data_gray.shape[:2], 255)

        self.data = np.dstack([data_gray, data_a])
        self.imgshape = self.data.shape

        # Calculate a downscale factor such that the resulting slider will fit in the boundaries defined in the config
        self.downscale_factor = np.max([data_gray.shape[0] / self.SLIDER_MAX_WIDTH,
                                        data_gray.shape[1] / self.SLIDER_MAX_HEIGHT]) * self.PIXEL_SPACING if self.ROTATE else \
            np.max([data_gray.shape[1] / self.SLIDER_MAX_WIDTH,
                    data_gray.shape[0] / self.SLIDER_MAX_HEIGHT]) * self.PIXEL_SPACING

        self.shape = np.ceil(np.divide(data_gray.shape, self.downscale_factor)).astype(np.int32)

    def osu_to_pixel(self, coord):
        scaled = np.round(np.divide(coord, self.PIXEL_SPACING) * self.downscale_factor).astype(np.int32)
        try:
            return self.data[scaled[1], scaled[0]]
        except IndexError:
            return np.array([0, 0])

    def get_anchor_code(self, pixel, layer):
        levels = self.LAYER_1_LEVELS if layer == 1 else self.LAYER_1_LEVELS
        codes = self.LAYER_1_CODES if layer == 1 else self.LAYER_1_CODES

        level = 0
        while level + 1 < len(levels) and levels[level + 1] <= pixel[0]:
            level += 1

        colour = codes[level]
        return colour

    def add_layer(self, anchors, layer, reverse):
        pass
