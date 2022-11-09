# Advanced Installation
## Installing Drake's nightly build
If you want to use the latest drake features:
```
conda env create -f environment_drake_nightly.yaml
conda activate drake_nightly
```

## Installation with Drake built from source
First build drake from [source](https://drake.mit.edu/from_source.html), then in this repo do:
```
conda env create -f environment_drake_source.yaml
conda activate drake_source
conda develop ~/drake-build/install/lib/python3.8/site-packages/
```
The source installation has the benefit that the `ur3e` model is included. However, building from source takes approximately 30 minutes.
