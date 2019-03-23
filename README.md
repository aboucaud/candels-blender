CANDELS galaxy blender
======================


Context
-------

**Initial dataset**: a galaxy catalog and stamps from the public HST CANDELS dataset, plus the associated segmentation masks obtained with SExtractor.  

The code uses this dataset to produce blended galaxies pairs in the form of 128x128, 32 bits images and their association binary segmentation masks.  
The neighboring objects that could be initially on the stamps are replaced with noise realizations from the background (this may affect the quality of the dataset).  

Install
-------

1. Clone the repository
   ```bash
   git clone https://github.com/aboucaud/candels-blender.git
   cd candels-blender
   ```

2. Install the dependencies and the module
   - with [conda](https://www.anaconda.com/download/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
   ```bash
   conda update conda                # Update conda
   conda env create                  # Use environment.yml to create the 'candels-blender' env
   conda activate candels-blender    # Activate the virtual env
   pip install .
   ```
   - without `conda` (**needs Python 3.6+**)
   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m pip install .
   ```

3. Download the CANDELS data
   ```bash
   python3 download_data.py
   ```

Usage
-----

The `candels-blender` command-line interface (CLI) can be used to create a custom dataset of realistic blended galaxies
```bash
candels-blender <action>
```

Three actions are currently available via the CLI:
  - `produce`
  - `concatenate`
  - `convert`

For each action, the available options are accessible via
```bash
candels-blender <action> --help
```

Example
-------

The three available actions are to be used sequentially.

### 1) Create the blends and catalogue
```bash
candels-blender produce -n 20000 --exclude irr --mag_high 23.5 --seed 42 --use_clean_galaxies
```
will create 20 000 blends of magnitude above 23.5 without irregular galaxies into a directory called `output-s_42-n_20000` along with 20 000 accompanying segmentation masks and two catalogues `train/test_catalogue.csv`.

### 2) Format the images and masks into distinct files
```bash
candels-blender concatenate -d output-s_42-n_20000 --method ogg_masks --delete
```
will format the blend stamps and masks into `train/test_blends.npy`, `train/test_ogg_masks.npy` and delete the individual files.

### 3) Obtain an array of the flux of both individual galaxies
```bash
candels-blender convert -d output-s_42-n_20000 --zeropoint=25.5
```
will use the magnitude of each galaxy, stored in the catalogues, to create the arrays of corresponding flux `train/test_flux.npy`, depending on the zero-point value.


## Example notebooks

You will find a directory with two static notebooks that can help you understand the blending process.  
They are **not meant to be runnable**.


## Authors

- Alexandre Boucaud - _aboucaud_ at _apc_ dot _in2p3.fr_
