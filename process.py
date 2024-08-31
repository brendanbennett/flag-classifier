from PIL import Image
from pathlib import Path

flag_source_dir = Path("country-flags-main/png1000px/")
flag_output_dir = Path("flags/")
target_size = 128

for im_filename in flag_source_dir.iterdir():
    im = Image.open(im_filename)
    im = im.convert("RGBA")
    if im.height > im.width:
        sf = target_size / im.height
        im = im.resize(size=(round(sf * im.width), target_size))
    else:
        sf = target_size / im.width
        im = im.resize(size=(target_size, round(sf * im.height)))
    im = im.crop(box=(0, 0, target_size, target_size), )
    im.save(flag_output_dir / im_filename.name)