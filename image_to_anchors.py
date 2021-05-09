from basic_converter import BasicConverter
from smart_converter import SmartConverter
from fill_converter import FillConverter
import configparser


def main():
    image_path = "images\\250.bmp"

    config = configparser.ConfigParser()
    config.read('config_animation.ini')

    converter = FillConverter(config)
    slidercode = converter.convert(image_path, 0)

    with open("output.txt", "w+") as f:
        f.write(slidercode)

    print("Done!")
    # input("Press enter to continue...")

if __name__ == "__main__":
    main()
