Uncertainty quantification in non-rigid image registration via stochastic gradient Markov chain Monte Carlo
============

We provide the source code used in the research published in the MELBA Special Issue: UNSURE 2020.

## Usage

### Set-up

* NiBabel
* matplotlib
* numpy
* pandas
* PyTorch
* scikit-learn
* SimpleITK
* tvtk

### Registration

To align images use the following command:
```
python run.py -vi 1 -mcmc 1 -d device_id -c config.json
```

`config.json` specifies the configuration to use for training, incl. the path to input images and the values of hyperparameters. The input images must have a `.nii.gz` extension and will be automatically resized to dimensions specified in the configuration file. The directory with the input images must contain subdirectories `seg` with the segmentations and `masks` with the image masks.

To resume registration:
```
python train.py -r path/to/last/checkpoint.pth
```

## Citation

If you use this code, please cite our paper.

Daniel Grzech, Mohammad Farid Azampour, Huaqi Qiu, Ben Glocker, Bernhard Kainz, and Loïc Le Folgoc. **Uncertainty quantification in non-rigid image registration via stochastic gradient Markov chain Monte Carlo.** MELBA 2021, Special Issue: UNSURE 2020, 1–25.

```
@article{Grzech2021,
    author = {Grzech, Daniel and Azampour, Mohammad Farid and Qiu, Huaqi and Glocker, Ben and Kainz, Bernhard and {Le Folgoc}, Lo{\"{i}}c},
    title = {{Uncertainty quantification in non-rigid image registration via stochastic gradient Markov chain Monte Carlo}},
    year = {2021},
    journal = {MELBA},
    number = {Special Issue: UNSURE 2020},
    pages = {1--25}
}
```