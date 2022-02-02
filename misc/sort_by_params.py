import os, glob

def sort_by_params(target_dir, first=1):
    sample_list = glob.glob(target_dir+'/*.pdf')
    
    for sample in sample_list:
        sample_name = sample.split('/')[-1]
        splitted = sample_name.split('_')
        
        modified_name = ''
        for i in range(len(splitted)):
            if i!=first:
                modified_name+='_'+splitted[i]
        
        come_first = splitted[first]
        modified_name = come_first+modified_name
        print(modified_name)
        os.rename(target_dir+'/'+sample_name, target_dir+'/'+modified_name)
        
        if sample==sample_list[-1]:
            print('#################################################')
            print(os.path.basename(sample),'|', splitted[first], '|', first)

        
if __name__=="__main__":
	dataset = 'PhalangesOutlinesCorrect'
	epochs = '712'
	target_dir = '../{}/alpha_{}_identity-vs-others_1.0/{}-test'.format(dataset, dataset, epochs)
	sort_by_params(target_dir)
