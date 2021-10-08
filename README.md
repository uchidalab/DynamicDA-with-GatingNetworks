# Dynamic Data Augmentation with Gating Networks
This is an official PyTorch implementations of the paper *Dynamic Data Augmentation with Gating Networks* which is submitted to **ICASSP2022** (under reviewing).  

## Results

## Usage

### Environment

#### Dataset
In experiments, we used [2018 UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).  

### Guidance

#### Data Augmentation methods
Each DA method implementation is based on [Keras codes](https://github.com/uchidalab/time_series_augmentation) from [a preceeding journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841) published by our group.
* Identity --- the original time series with no augmentation.  
* Jittering --- adds Gaussian noise to the time series.  
* Magnitude Warping --- multiply the time series by a smooth curve defined by cublic spline.  
* Time Warping --- similar to Magnitude Warping, except the warping is done in the time domain.  
* Window Warping --- selects a random window of 10% of the original time series length and warps the window by 0.5 to 2 times.  

#### Hyperparameters

## Citation
