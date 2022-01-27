import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def vis_pentagon(dataset, d1, d2, d3, d4, d5, target, nclass, ntest):
	c_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	da1_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	da2_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	da3_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	da4_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	da5_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	
	c1 = [[], [], [], [], []]
	c2_da1, c2_da2, c2_da3, c2_da4, c2_da5 = [], [], [], [], [] 
	c3_da1, c3_da2, c3_da3, c3_da4, c3_da5 = [], [], [], [], [] 
	c4_da1, c4_da2, c4_da3, c4_da4, c4_da5 = [], [], [], [], [] 
	c5_da1, c5_da2, c5_da3, c5_da4, c5_da5 = [], [], [], [], [] 
	c6_da1, c6_da2, c6_da3, c6_da4, c6_da5 = [], [], [], [], [] 
	c7_da1, c7_da2, c7_da3, c7_da4, c7_da5 = [], [], [], [], []
	
	for i in range(ntest):
		class_idx = int(target[i])

		da1_list[class_idx]+=d1[i]
		da2_list[class_idx]+=d2[i]
		da3_list[class_idx]+=d3[i]
		da4_list[class_idx]+=d4[i]
		da5_list[class_idx]+=d5[i]
		c_list[class_idx]+=1
	
	for class_idx in range(nclass):
		da1_list[class_idx]='{:.5f}'.format(da1_list[class_idx]/c_list[class_idx])
		da2_list[class_idx]='{:.5f}'.format(da2_list[class_idx]/c_list[class_idx])
		da3_list[class_idx]='{:.5f}'.format(da3_list[class_idx]/c_list[class_idx])
		da4_list[class_idx]='{:.5f}'.format(da4_list[class_idx]/c_list[class_idx])
		da5_list[class_idx]='{:.5f}'.format(da5_list[class_idx]/c_list[class_idx])

	#categories = ['Identity','Jitter','Window Warp', 'Magnitude Warp', 'Time Warp']
	categories = ['Identity','Jitter', 'Magnitude Warp', 'Window Warp', 'Time Warp']

	fig = go.Figure()

	for class_idx in range(nclass):
		fig.add_trace(go.Scatterpolar(
			r=[float(da1_list[class_idx]), float(da2_list[class_idx]), float(da3_list[class_idx]), float(da4_list[class_idx]), float(da5_list[class_idx])],
			theta=categories,
			fill='toself',
			opacity=0.9,
			line=dict(width=4),
			name='Class {}'.format(class_idx+1)))

	fig.update_layout(
		template=None,
		polar=dict(
		radialaxis=dict(
			tickfont=dict(size=18),
			showline=False,
			nticks= 10,
			showgrid=False,
			visible=True,
			
			range=[0.0, 1.0])),
		legend_title="Class",
		font=dict(
			family="Courier New, monospace",
			size=20,
			color="RebeccaPurple"),
		showlegend=True)

	fig.write_image("./pentagon_{}.pdf".format(dataset))
	

if __name__=="__main__":

	dataset_id = 9

	if dataset_id == 1:
		dataset = 'Crop'
		ntest = 16800
		nclass = 24
	if dataset_id == 2:
		dataset = 'ElectricDevices'
		ntest = 7711
		nclass = 7
	if dataset_id == 3:
		dataset = 'FordA'
		ntest = 1320
		nclass = 2
	if dataset_id == 4:
		dataset = 'FordB'
		ntest = 810
		nclass = 2
	if dataset_id == 5:
		dataset = 'HandOutlines'
		ntest = 370
		nclass = 2
	if dataset_id == 6:
		dataset = 'MelbournePedestrian'
		ntest = 2439
		nclass = 10
	if dataset_id == 7:
		dataset = 'NonInvasiveFetalECGThorax1'
		ntest = 1965
		nclass = 42
	if dataset_id == 8:
		dataset = 'NonInvasiveFetalECGThorax1'
		ntest = 1965
		nclass = 42
	if dataset_id == 9:
		dataset = 'PhalangesOutlinesCorrect'
		ntest = 858
		nclass = 2
	if dataset_id == 10:
		dataset = 'StarLightCurves'
		ntest = 8236
		nclass = 3
	if dataset_id == 11:
		dataset = 'TwoPatterns'
		ntest = 4000
		nclass = 4
	if dataset_id == 12:
		dataset = 'Wafer'
		ntest = 6164
		nclass = 2
	
	df = pd.read_csv('{}.csv'.format(dataset), index_col=0)
	print(df.mean())
	print(df.std())
	vis_pentagon(dataset, list(df['0']),list(df['1']),list(df['2']),list(df['3']),list(df['4']), list(df['5']), nclass, ntest)
	
