#!/usr/bin/env python

""" Basic statistical methods for the anlaysis of features """

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pingouin as pg
import os


class Analysis:


	def __init__(self, metadata, features):

		self.metadata = pd.read_csv(metadata)
		self.features = pd.read_csv(features)
		self.combined_data = pd.merge(self.metadata, self.features, on='file_id')
		self.features = self.features.drop('file_id', axis=1)  # don't want to pick up as a feature


	def plot_two_dimensions(self, df, hue, size, show=False, save_as=None):
		""" Plots in two dimensions """
		
		# make into a dataframe for plotting
		proj = pd.DataFrame(df).iloc[:,:2]
		proj.columns = ['1', '2']
		proj.index = self.features.index
		proj[hue] = self.metadata[hue]
		proj[size] = self.metadata[size]

		# plot
		plt.figure(figsize=(10,10))
		sns.scatterplot(data=proj, x='1', y='2', hue=hue, size=size, legend='full')
		if save_as:	plt.savefig(save_as)
		if show: plt.show()
		plt.clf()


	def run_tsne(self, use_PCA=False, n_components=2, perplexity=15, hue='treatment', size='day', show=False, save_as='tsne.png'):
		""" Run tSNE and plot """
		
		if use_PCA:
			features = self.PCs
		else:
			features = scale(self.features)

		# run tSNE
		tsne = TSNE(n_components, perplexity)
		tsne_features = tsne.fit_transform(features)

		self.plot_two_dimensions(tsne_features, hue, size, show, save_as)


	def run_pca(self, n_components=2, hue='treatment', size='day', show=False, save_as='pca.png'):
		""" Run PCA and plot """

		# scale data
		scaled = scale(self.features)

		# run PCA
		pca = PCA(n_components)
		Xpca = pca.fit_transform(scaled)

		self.PCs = Xpca
		self.plot_two_dimensions(Xpca, hue, size, show, save_as)


	def run_ttest(self, corr='Bonferroni'):
		"""t-test"""

		# split by treatment
		treatments = self.metadata['treatment'].unique()
		treatment1 = self.features.loc[self.metadata.treatment == treatments[0]]
		treatment2 = self.features.loc[self.metadata.treatment == treatments[1]]

		# run t-test for each feature
		ttests = []
		for feat in self.features:
			test = pg.ttest(treatment1[feat], treatment2[feat])
			test.index = [feat]
			ttests.append(test)
		ttests = pd.concat(ttests)
		ttests = ttests.sort_values(by=['p-val'])
		
		# multiple testing correction
		if corr == 'Bonferroni':
			ttests['p-corr'] = ttests['p-val']*len(self.features.columns)
		
		return ttests


	def run_anova(self, measure, within='day', between='treatment', subject='mouse_number'):
		
		# run ANOVA for each feature
		aovs = {}
		for feat in tqdm(self.features.columns):
			aov = pg.mixed_anova(dv=feat, within=within, between=between, subject=subject, data=self.combined_data)
			aovs[feat] = aov.loc[aov['Source']==measure].squeeze()

		aovs = pd.DataFrame(aovs).T
		aovs = aovs.sort_values(by=['p-unc'])
		
		return aovs


	def make_boxplots(self, path, swarm=False, selected_features=None):

		if not os.path.exists(path):
			os.mkdir(os.path.join(path))

		# allow user to select only specific features to plot
		features = selected_features if selected_features else self.features

		for feat in tqdm(features):
			sns.boxplot(data=self.combined_data, x='day', y=feat, hue='treatment')
			if swarm:
				sns.swarmplot(data=self.combined_data, x='day', y=feat, color=".2")
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(os.path.join(path, '%s.png' % feat), bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
