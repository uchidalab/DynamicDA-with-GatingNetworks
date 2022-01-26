# Dynamic Data Augmentation with Gating Networks
This is an official PyTorch implementation of the paper *Dynamic Data Augmentation with Gating Networks* which is submitted to **ICPR2022** (under review).  

## Usage

### Environment

#### Dependencies
```pip3 install -r requirements.txt```

#### Dataset
In experiments, we used [2018 UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).  
Please be cautious that we modified these datasets in the way mentioned in the paper.  
Put dataset folders inside ```/dataset```.  
Plus, set the dataset path in a function named ```data_generator``` in the script of ```/utils/dataload.py```   

### Components

#### Models
* No Augmentation --- refer to ```no_augmentation.py```.  
* Proposed --- refer to ```proposed.py```. You can change lambda value in the paper by ```consis_lambda``` argument.  
* Feature Combination with Equal Weights -- refer to ```equal_weights.py```.  This is listed as "w/o GateNet & feature consistency loss" on the table.  
<!--* Ensemble -- refer to ```ensemble.py```.
* Concatenate --- refer to ```concat.py```.  -->  

For execution, run ```experiment.sh```.  
You will get csv file which save every 25 epoch's result and saved model parameters for the final epoch.  
You can test your saved parameters by enabling ```test_model()``` under ```if __name__ == "__main__":``` in each python file above.  

#### Data Augmentation methods
Each DA method implementation is based on [our preceeding journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841).
Please take a look on ```/utils/augmentation.py``` for the codes. Adopted methods are the following:  
* Identity --- the original time series with no augmentation.  
* Jittering --- adds Gaussian noise to the time series.  
* Magnitude Warping --- multiply the time series by a smooth curve defined by cublic spline.  
* Time Warping --- similar to Magnitude Warping, except the warping is done in the time domain.  
* Window Warping --- selects a random window of 10% of the original time series length and warps the window by 0.5 to 2 times.  

## Citation
[D. Oba, S. Matsuo and B. K. Iwana, "Dynamic Data Augmentation with Gating Networks," arXiv, 2021.](https://arxiv.org/abs/2111.03253)  
```
@article{oba2021dynamic,
  title={Dynamic Data Augmentation with Gating Networks},
  author={Daisuke Oba, Shinnosuke Matsuo and Brian Kenji Iwana},
  journal={arXiv preprint arXiv:2111.03253},
  year={2021}
}
```
