# LEXT Stitcher

## Description
This program was created to enhance the stitching capabilities of an Olympus LEXT laser confocal microscope.

When stitching, the LEXT software creates a final image that shows dark borders in between the individual images (see Results/LEXT_comparison.jpeg in this repository). 

This program makes a fitting across the surface of the image and makes it flat, resulting in a uniformly illuminated image. Afterwards, the program makes a stitching of the images, resulting in a much cleaner final result (see Results/Result.jpeg in this repository)

## Installation
Please install the libraries shown in the requirements.txt


## Usage
Two files can be used: LEXT_stitcher_BW.py (black & white result) & LEXT_stitcher_RGB.py (color result)

Currently, results from the black & white version are better.

At the bottom of the file, write the path to the folder where the separate images from the LEXT microscope are in "folder_path".

The program will then fit the images, make an average of all the fits, and subtract that matrix to all the pictures.

For best results, modify the "diagonal" parameter in line 61, so that it contains picture number the most uniform images. This step is important because in a microscopic level, analyzed surfaces usually contain plenty of defects, and by choosing the most uniform images we make sure that program is only fitting the lighting inhomogeneities and not the sample defects.

## Support
For help, contact enandayapa@gmail.com

## Roadmap
In the future, we plan to have more automatized process includying a graphical interface.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors and acknowledgment
All programming was done by Edgar Nandayapa B.
Field testing has been done by several members of Prof. Dr. Emil List-Kratochwil group at Humbold University of Berlin.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Known issues
No known issues, it just requires a lot of manual fine tunning.