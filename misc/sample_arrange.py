import os, glob, shutil

def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def sample_arrange(target_dir):
	sample_list = glob.glob(target_dir+'/*.pdf')
	
	if sample_list == []:
		exit()
	
	my_makedirs(target_dir+'/correct')
	my_makedirs(target_dir+'/incorrect')
	
	for sample in sample_list:
	    sample_name = sample.split('/')[-1]
	    splitted = sample_name.split('_')
	    
	    idx = splitted[1]
	    target_class = splitted[-2]
	    pred_class = splitted[-1][:-4]
	    
	    if int(pred_class)==int(target_class):
	        shutil.copyfile(sample, target_dir+'/correct/'+sample_name)
	    else:
	        my_makedirs(target_dir+'/incorrect/'+target_class)
	        shutil.copyfile(sample, target_dir+'/incorrect/'+target_class+'/'+sample_name)
    
        
if __name__=="__main__":
	dataset_list = ['FordA', 'NonInvasiveFetalECGThorax1', 'ElectricDevices', 'StarLightCurves', 'FordB', 'NonInvasiveFetalECGThorax2', 'TwoPatterns', 'PhalangesOutlinesCorrect', 'Wafer', 'Crop', 'MelbournePedestrian_adjusted', 'HandOutlines']
	
	for dataset in dataset_list:	
		target_dir = './Sample_visualize_{}'.format(dataset)
		try:
			sample_arrange(target_dir)
		except:
			pass
