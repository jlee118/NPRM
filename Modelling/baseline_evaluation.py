import pandas as pd
import pickle
import numpy as np
import os
import sys
from sklearn.metrics import dcg_score, ndcg_score, mean_absolute_error,\
							mean_squared_error, accuracy_score, f1_score,\
							recall_score, precision_score
from scipy.stats import spearmanr, kendalltau

def generate_avg_rank_baseline(df, feature, metric_name, metric_fn):
	""" 
	'df' is a Dataframe with features aggregated as arrays.  
	'feature' is a String, the selected readability assessment metric. 
	'metric' is a Function chosen for computing the baseline.
	"""

	# Ground Truth is already sorted from hardest reading level to easiest
	ground_truth = df['grade_level']

	# The feature-ranked grades are just the ground truth arrangement but sorted according to the feature
	if feature == 'f_ease':
		# Flesch Reading Ease is smallest to largest in respect to reading level
		# feature_grade = ground_truth.apply(lambda x: sorted(x))
		feature_grade = df['f_ease'].apply(lambda x: 1 / np.asarray(x))
	else:
		# All other features are sorted from largest to smallest
		# feature_grade = ground_truth.apply(lambda x: sorted(x, reverse=True))
		feature_grade = df[feature]
	# Binary relevance for DCG and NDCG, comparison on Series types
	# Will iterate through this, since not all samples have the same size

	if metric_name == 'ranking_accuracy':
		ground_truth = df[feature]
		feature_grade = ground_truth.apply(lambda x: sorted(x, reverse=True))

	metric_agg = []

	for i in range(len(ground_truth)):
		curr_truth = ground_truth[i]
		curr_feat_grade = feature_grade[i]
		if np.all(curr_feat_grade == curr_feat_grade[0]):
			metric_agg.append(0)
			continue
		if np.isnan(curr_feat_grade).any():
			metric_agg.append(0)
			continue

		# If DCG or NDCG, create and use the relevancy scores
		if metric_name == 'dcg' or metric_name == 'ndcg':
			# relevance = np.equal(curr_truth, curr_feat_grade).astype(int)
			# rel = np.asarray([relevance])
			rel = np.asarray([curr_truth])
			feat = np.asarray([curr_feat_grade])
			if len(curr_truth) == 1:
				continue
			metric_agg.append(metric_fn(rel, feat))


		# Other metrics should use the ground truth and predicted rankings as input
		else:
			try:
				score = metric_fn(curr_truth, curr_feat_grade)
				if isinstance(score, tuple):
					score = score[0]
				metric_agg.append(score)
			except:
				pass

	return (np.nanmean(metric_agg))

def generate_ml_baseline(df, feature, metric_name, metric_fn):
	"""
	df is a Dataframe of predicted ML scores.
	feature is a string of the method name.
	metric_name is a string referring to the name of the evaluation metric
	metric_fn is the metric function
	"""
	truth = df['grade_level']
	preds = df[feature]
	try:
		if 'f1' in metric_name or 'precision' in metric_name or'recall' in metric_name:
			score = metric_fn(truth, preds, average='weighted')
		else:
			score = metric_fn(truth, preds)
	except:
		score = np.nan
	return score 


def readability_ndcg_score(relevance_scores, scores):
	""" 
	'relevance_scores' and 'scores' are nested numpy arrays
	"""
	# IDCG here is the maximum score if all results are relevant, which in this case means that the 
	# rankings are all correct.  The relevance score should be a vector of 1's.
	idcg = np.sum((2 ** np.ones(scores.shape[1]) - 1) / np.log2(np.asarray(range(1, scores.shape[1]+1)) + 1))
	return dcg_score(relevance_scores, scores) / idcg


def readability_mrr(ground_truth, feature_rank):
	"""
	'ground_truth' is an array of the original ranking, 'feature_rank' is an array of ranking corresponding 
	to the chosen feature.  None of these arrays are nested.
	"""
	relevance = np.equal(ground_truth, feature_rank).astype(int)
	return (1 / np.where(relevance==1)[0])

def rank_accuracy(ground_truth, predicted_rank):
	return int(np.array_equal(ground_truth, predicted_rank))


def create_results_table(feature_list, df, os_eng=False):
	"""
	'feature_list' is an array of Strings that represent a feature to perform the ranking with.
	'df' is a DataFrame with its features aggregated as an array
	"""
	results = {}
	# Hard coding the score names functions here for now
	metric_list = [('dcg',dcg_score), ('ndcg', ndcg_score),
				   ('spearman_rank', spearmanr), ('kendalltau', kendalltau),
				   ('ml_mse', mean_squared_error), ('ml_mae', mean_absolute_error),
				   ('ml_accuracy', accuracy_score), ('ml_f1', f1_score),
				   ('ml_recall', recall_score), ('ml_precision', precision_score), ('ranking_accuracy', rank_accuracy)]

	df_array_form = df.groupby(['slug']).agg(list)
	df_array_form.reset_index(inplace=True)

	for f in feature_list:
		metrics = []
		for m in metric_list:
			print(m[0])
			if (m[0] == 'accuracy') and ('class' not in f):
				metrics.append(np.nan)
				continue
			if 'ml' in m[0]:
				# If evaluating for a non-ranking metric, use traditional ML style evaluation
				metrics.append(generate_ml_baseline(df, f, m[0], m[1]))
			else:
				# If evaluating using a ranking metric, evaluate on within-topic rankings
				metrics.append(generate_avg_rank_baseline(df_array_form, f, m[0], m[1]))
		results[f] = metrics	
	results = pd.DataFrame.from_dict(results, columns=['dcg', 'ndcg','spearman_rank',\
														'kendalltau','ml_mse', 'ml_mae','ml_accuracy',\
														'ml_f1', 'ml_recall', 'ml_precision', 'ranking_accuracy'],
													   orient='index')
	return results


def evaluate_data(dataname, df, outputpath, ml_feature_list):
	print('Running {}'.format(dataname))
	# Note: Newsela and OneStopEnglish both have their reading levels ordered by hardest to easiest in the .csv files
	df_feats = [f for f in ml_feature_list if f in df.columns]
	df_results = create_results_table(df_feats, df)
	df_results.insert(0, 'Features_Models', value=df_results.index)
	df_results.sort_values(by=['Features_Models'], inplace=True)
	df_output_path = os.path.join(curr_path, 'Datasets', '{}_baselines.csv'.format(dataname))
	df_results.to_csv(df_output_path, index=False)



if __name__ == '__main__':
	curr_path = os.path.abspath('..')
	feature_list = ['f_grade', 'f_ease', 'ari']
	ml_feature_fp = os.path.join(curr_path, 'Datasets', 'ml_feature_names.pkl')

	with open(ml_feature_fp, 'rb') as f:
		ml_feature_list = pickle.load(f)

	feature_list += ml_feature_list

	
	newsela_es_filepath = os.path.join(curr_path, 'Datasets','newsela_es_rank_features.csv')
	newsela_es_output_path = os.path.join(curr_path, 'Datasets', 'newsela_es_baselines.csv')
	newsela_es = pd.read_csv(newsela_es_filepath)

	newsela_filepath = os.path.join(curr_path, 'Datasets','newsela_en_rank_features.csv')
	newsela_output_path = os.path.join(curr_path, 'Datasets', 'newsela_baselines.csv')
	newsela = pd.read_csv(newsela_filepath)

	os_eng_filepath = os.path.join(curr_path, 'Datasets','os_eng_rank_features.csv')
	os_eng_output_path = os.path.join(curr_path, 'Datasets', 'os_eng_baselines.csv')
	os_eng = pd.read_csv(os_eng_filepath)

	transread_en_filepath = os.path.join(curr_path, 'Datasets','tr_english_rank_features.csv')
	transread_en_output_path = os.path.join(curr_path, 'Datasets', 'transread_en_baselines.csv')
	tr_en = pd.read_csv(transread_en_filepath)

	transread_fr_filepath = os.path.join(curr_path, 'Datasets', 'tr_fr_rank_features.csv')
	transread_fr_output_path = os.path.join(curr_path, 'Datasets', 'tr_fr_baselines.csv')
	tr_fr = pd.read_csv(transread_fr_filepath)

	test_commoncore_filepath = os.path.join(curr_path, 'Datasets', 'commoncore_rank_features.csv')
	test_commoncore_output_path = os.path.join(curr_path, 'Datasets', 'commoncore_baselines.csv')
	test_commoncore = pd.read_csv(test_commoncore_filepath)




	data = [('newsela', newsela, newsela_output_path), ('os_eng', os_eng, os_output_path),
			('transread_en', tr_en, transread_en_output_path)]


	for i in range(len(data)):
		dataname = data[i][0]
		df = data[i][1]
		df_output_path = data[i][2]
		evaluate_data(dataname, df, df_output_path, ml_feature_list)






