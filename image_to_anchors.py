from basic_converter import BasicConverter
from smart_converter import SmartConverter
import configparser


def main():
    image_path = "images\\urdead.png"

    config = configparser.ConfigParser()
    config.read('config2.ini')

    converter = BasicConverter(config)
    slidercode = converter.convert(image_path, 0)

    with open("output.txt", "w+") as f:
        f.write(slidercode)

    print("Done!")
    # input("Press enter to continue...")

if __name__ == "__main__":
    main()
