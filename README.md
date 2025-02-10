# p-brain (mini)
![banner](https://github.com/user-attachments/assets/4f8bdcba-dbe6-41c7-b644-c0cdaf031fe9)
This script automates the processing and analysis of DCE-MRI data, including:
- Segmentation with FastSurfer
- Mask generation and visualization
- Concentration-time curve (CTC) computation
- Patlak analysis for Ki estimation
- Per-voxel computation of perfusion parameters

## Installation

### 1. Clone the repository
First, clone the repository to your local machine:

```sh
git clone https://github.com/edtireli/p-brain-mini.git
cd p-brain-mini
```

### 2. Install dependencies
Ensure you have Python 3 installed, then run:

```sh
pip install -r requirements.txt
```

### 3. Install FreeSurfer & FastSurfer (if needed)
This script integrates with **FastSurfer** for segmentation. Install it following their [official guide](https://github.com/Deep-MI/FastSurfer).

FreeSurfer (needed for certain utilities like `mri_convert`) can be installed via:

```sh
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

## Running the Script

Launch the script using:

```sh
python p-brain-mini.py
```

A GUI will open, where you can:

- Load T1 & M0 maps (.mat format)
- Load the Arterial Input Function (AIF) (.mat)
- Load the DCE-MRI scan (.nii.gz)
- Load the anatomical T1-weighted MRI (.nii)

Once all files are selected, click **Analyze** to run the full pipeline.

## Updating Dependencies
To update all dependencies, run:

```sh
pip install --upgrade -r requirements.txt
```

## Notes
- The script performs **coregistration**, segmentation, and perfusion analysis.
- If FastSurfer segmentation is missing, it will be computed automatically.
- Outputs (Ki maps, segmentation masks, etc.) are saved in the `analysis/` directory.
