# Image to Anchors
Tool for converting images to slider anchors for view in the osu! editor.

## Usage
To use, install the libraries from requirements.txt, open image_to_anchors.py, input the path to your image and run it.
The .osu code of the slider will be in output.txt

For animations use images_to_animation.py. Input the path to the folder with all the images and give it the
start time and frame duration in milliseconds for timing the frames. Multiple slidercodes will be generated in output.txt

## Config
You can tweak how the conversion works by editing the config files or reading a different config file. 

The file config.ini has settings for (what I think) the best quality image in the editor and
it's meant to be viewed at 800x504 custom resolution (which you can set by manually editing your osu! config file). 

The file config2.ini has settings for a decent result which can be viewed at 1080p and most other resolutions.

### Explanation of all config settings
- PIXEL_SPACING: The distance between every anchor in osu! pixels.
- ROTATE: To adjust some offsets and rotate the bounding box, so the result looks good after rotating it 90 degrees.
- LAYER_2_OFFSET: To offset the second layer by 4 osu! pixels which results in a seamingly higher resolution in the result.
- BRIGHT_BG: To make the result look better on bright backgrounds by adding some white anchors to hide red anchors outside the bounds of the image.
- E_MODE: To generate the anchors in an E pattern instead of the usual back-and-forth lines. Might look better on some edges.
- VERBOSE: To print extra debug information.
- SLIDER_MAX_WIDTH: Maximum width in osu! pixels for the result.
- SLIDER_MAX_HEIGHT: Maximum height in osu! pixels for the result.

### Colour programming
There are two anchor layers and for both layers you can define entirely in 
the config file how to generate anchors for each luminosity level. 

The luminosity ranges from 0 to 255. 
You configure the anchors by adding key-value pairs in the [Layer 1] or [Layer 2] categories. 
The key is the luminosity level and the value is a string of 'R' 'W' and '\_' which defines a pattern of anchors. 
'R' adds a red anchor, 'W' adds a white anchor, and '\_' moves one osu! pixel further.

For example "0: RR_RR_RR_RR_RR_RR_RR" generates 7 double stacked red anchors 1 pixel apart for any luminosity from 0 to the next.
