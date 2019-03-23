CANDELS galaxy blender
======================


Context
-------

**Initial dataset**: a galaxy catalog and stamps from the public HST CANDELS dataset, plus the associated segmentation masks obtained with SExtractor.  

The code uses this dataset to produce blended galaxies pairs in the form of 128x128, 32 bits images and their association binary segmentation masks.  
The neighboring objects that could be initially on the stamps are replaced with noise realizations from the background (this may affect the quality of the dataset).  

Usage
-----

1. Clone the repository
   ```
   git clone https://github.com/aboucaud/candels-blender.git
   cd candels-blender
   ```

2. Install the dependencies
   - with [conda](https://www.anaconda.com/download/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
   ```
   conda update conda                # Update conda
   conda env create                  # Use environment.yml to create the 'candels-blender' env
   conda activate candels-blender    # Activate the virtual env
   ```
   - without `conda` (**needs Python 3.6+**)
   ```
   python3 -m pip install -r requirements.txt
   ```

3. Install the `candels-blender` package and download the data
   ```
   python3 -m pip install .
   python3 download_data.py
   ```

4. Use the `candels-blender` command-line interface (CLI) to create your own blend dataset
   ```
   candels-blender <action>
   ```
   There are three actions currently accessible via the `candels-blender` CLI:
     - `produce`
     - `concatenate`
     - `convert`
   
   For each action, all the available options are accessible via
   ```
   candels-blender <action> --help
   ```

Example
-------

The three available actions are to be used sequentially.

1. Create the blends
   ```
   candels-blender produce -n 20000 --exclude irr --mag_high 23.5 --seed 42 --use_clean_galaxies
   ```
   will create 20 000 blends of magnitude above 23.5 without irregular galaxies into a directory called `output-s_42-n_20000` along with 20 000 accompanying segmentation masks and two catalogues `train/test_catalogue.csv`.

2. Format the images, masks and catalogues into three distinct files
   ```
   candels-blender concatenate output-s_42-n_20000 --method ogg_masks --delete
   ```
   will format the blend stamps and masks into `train/test_blends.npy`, `train/test_ogg_masks.npy` and delete the individual files.

3. Obtain an array of the flux of both individual galaxies
   ```
   candels-blender convert output-s_42-n_20000 --zeropoint=25.5
   ```
   will use the magnitude of each galaxy, stored in the catalogues, to create the arrays of corresponding flux `train/test_flux.npy`, depending on the zero-point value.


## Example notebooks

You will find a directory with two static notebooks that can help you understand the blending process.  
They are **not meant to be runnable**.


## Authors

- Alexandre Boucaud - _aboucaud_ at _apc_ dot _in2p3.fr_
