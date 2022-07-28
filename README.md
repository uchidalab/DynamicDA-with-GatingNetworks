# Dynamic Data Augmentation with Gating Networks for Time Series Recognition
This is an official PyTorch implementation of the paper *Dynamic Data Augmentation with Gating Networks for Time Series Recognition*, which is accepted to **ICPR2022**.  

## Usage

### Environment

#### Dependencies
```pip3 install -r requirements.txt```

#### Dataset
In experiments, we used [2018 UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).  
Please be cautious that we modified these datasets in the way mentioned in the paper.  
Put datasets inside ```/dataset```, and set the dataset path in the function named ```data_generator``` in the script of ```/utils/dataload.py```   

### Execution scripts

#### Models
* **No Augmentation** : refer to ```no_aug.py```.  
* **Proposed** : refer to ```proposed.py```. You can change lambda value in the paper by ```--consis_lambda```.  
* **Feature Combination with Equal Weights** : refer to ```equal_weights.py```.  This is listed as "w/o GateNet & feature consistency loss" in the paper.  
<!--* Ensemble -- refer to ```ensemble.py```.
* Concatenate --- refer to ```concat.py```.  -->  

For a training example, run ```experiment.sh```.  
You will get a csv file that contains every 25 epoch's result and a frozen model at the final epoch, which can be tested by enabling a flag ```--frozen```.  

#### Data Augmentation methods
Each DA method implementation is based on [our preceeding journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841).
Please take a look on ```/utils/augmentation.py``` for the codes. Adopted methods are the following:  
* **Identity** : original time series with no augmentation.  
* **Jittering** : adds Gaussian noise to time series.  
* **Magnitude Warping** : multiply time series by a smooth curve defined by cubic spline.  
* **Time Warping** : similar to Magnitude Warping, except the warping is done in time domain.  
* **Window Warping** : selects a random window of 10% of the original time series length and warps the window by 0.5 to 2 times.  

## Citation
[D. Oba, S. Matsuo and B. K. Iwana, "Dynamic Data Augmentation with Gating Networks for Time Series Recognition," ICPR, 2022.](https://arxiv.org/abs/2111.03253)  
```
@inproceedings{oba2022,
	year = 2022,
	author = {Daisuke Oba, Shinnosuke Matsuo and Brian Kenji Iwana},
	title = {Dynamic Data Augmentation with Gating Networks for Time Series Recognition},
	booktitle = {ICPR},
}
```
