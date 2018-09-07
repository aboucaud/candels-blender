# HST galaxy blender

**Initial dataset**: a galaxy catalog and stamps from the public HST CANDELS dataset, plus the associated segmentation masks obtained with SExtractor.

The code uses this dataset to produce pairs of manually blended galaxies in the form of 128x128, 32 bits images and their associated 128x128x2, binary masks.

## Usage

1. Clone the repository
   ```
   git clone https://github.com/aboucaud/candels-blender.git
   cd candels-blender
   ```

2. Install the dependancies
  - with [conda](https://www.anaconda.com/download/)
  ```
  conda env create                  # Use environment.yml to create the 'candel-blender' env
  source activate candels-blender   # Activate the virtual env
  ```
  - without `conda` (**needs Python 3.6**)
  ```
  python -m pip install -r requirements.txt
  ```

3. Download the data (contact me for now)
   ```
   mkdir data
   # put the three files in this directory
   ```

4. Use the first script to create the blends and their mask
   ```
   python produce_blends.py <number_of_desired_images>

   # 20 000 blends of magnitude above 23.5 without irregular galaxies
   python produce_blends.py 20000 -e irr --mag_high 23.5

   # Check the full options with
   python produce_blends.py --help
   ```
   All the images will be placed in a specific directory.

5. Use the second script to merge the images into a single object and create the labels
   ```
   python concatenate_blends.py output-s_XX-nXXXXX --method overlap_galaxies

   # Check the full options with
   python concatenate_blends.py --help
   ```
   This step will produce two files `images.npy` and `labels.npy` in the same directory as the individual images.

