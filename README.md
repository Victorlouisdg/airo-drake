# AIRO-Drake

Scripts and notebooks to test the [Drake](https://drake.mit.edu/) robotics toolbox for our use cases at AIRO.

# Installation

A little bit quirky. I built from source, but after that there is no package to be pip-installed. 
So I created a venv and added this to its `activate` script:
```
PYTHONPATH="/home/idlab185/drake-build/install/lib/python3.8/site-packages:$PYTHONPATH"
export PYTHONPATH
```

When simply pip installing drake, not all models are included due to their large size. 