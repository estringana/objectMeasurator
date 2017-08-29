# Measurator

This project provides a way of measuring objects.

The way it works is by having reference object (first item on the left) which its measures are known. From there, we iterates over any other object on the image and get their measures.

## Install environment
This repot has been provided with a conda environemnt. Install conda and then run:

```
conda env create -f environment.yml
```
That commnad will create an environment. In order to use it, you need to activate it with

```
source activate Measurator
```

## Input images
All input images must be `PNG` or `JPG` and they need to be placed on the the folder called `input`

## Output images
All the output processed are left on the folder called `output`