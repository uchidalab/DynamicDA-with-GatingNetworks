# Dynamic Data Augmentation with Gating Networks
This is an official PyTorch implementation of the paper *Dynamic Data Augmentation with Gating Networks* which is submitted to **ICASSP2022** (under reviewing).  

## Usage

### Environment

#### Dataset
In experiments, we used [2018 UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).  
You can download our modified UCR dataset from [here](https://drive.google.com/file/d/1w1--ckigeF-PPzbwa7_6ghJdPqo7P3_P/view?usp=sharing).  
Please extract this file at ```/dataset```.  

### Guidance

#### Models
* No Augmentation --- refer to ```no_augmentation.py```.  
* Concatenate --- refer to ```concat.py```.  
* Proposed --- refer to ```proposed.py```.  

For execution, you just need to run ```experiment.sh```.  

#### Data Augmentation methods
Each DA method implementation is based on [Keras codes](https://github.com/uchidalab/time_series_augmentation) from [our preceeding journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841).
* Identity --- the original time series with no augmentation.  
* Jittering --- adds Gaussian noise to the time series.  
* Magnitude Warping --- multiply the time series by a smooth curve defined by cublic spline.  
* Time Warping --- similar to Magnitude Warping, except the warping is done in the time domain.  
* Window Warping --- selects a random window of 10% of the original time series length and warps the window by 0.5 to 2 times.  

## Citation
D. Oba, S. Matsuo and B. K. Iwana, "Dynamic Data Augmentation with Gating Networks," arXiv, 2021.  
```
@article{oba2021,
  title={Dynamic Data Augmentation with Gating Networks},
  author={Daisuke Oba, Shinnosuke Matsuo and Iwana Brian Kenji},
  journal={arXiv preprint arXiv:????.?????},
  year={2021}
}
```
