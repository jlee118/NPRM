from ml_scores import get_text, WORD_EMBED_NAMES
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from rank_neural_networks import BertRank, BertRank2
from transformers import BertTokenizer, TFBertForSequenceClassification
import gensim.downloader
import numpy as np
import pandas as pd
import pickle
import os
import re 
import tensorflow as tf
import tensorflow_ranking as tfr

def pairwise_transform(X, Y, min_perm_size=3):
	"""
	This function takes as input a 2-D array 'X' and a 1-D array 'Y'.  
	If the first dimension is greater than 2, the function slices the first and last vectors out and samples 'min_perm_size' middle vectors.
	Permutations are formed (size 2) and differences between pairs of vectors are calculated.
	'X_new', vectors of the differences, and 'Y_new', the sign of the difference in reading levels, are returned.

	X: 2-D numpy array
	Y: 1-D numpy array
	min_perm_size: Number of permutations to form

	Returns X_new and Y_new, processed features and target for pairwise modelling.
	"""
	# Loop over slugs to make permutations at the reading level.
	X_new = []
	Y_new = []

	# If the number of sub-vectors are less than min_perm_size, simply form permutation pairs
	if X.shape[0] < min_perm_size:
		X_new.append(X[[0]] - X[[1]])
		Y_new.append(np.sign(Y[[0]] - Y[[1]]))

		X_new.append(X[[1]] - X[[0]])
		Y_new.append(np.sign(Y[[1]] - Y[[0]]))

		return X_new, Y_new

	# Keep first and last vectors, sample middle ranking vectors
	middle_idx = np.random.choice(range(1, X.shape[0] - 1),size=min_perm_size-2, replace=False)
	middle_X, middle_Y = X[middle_idx], Y[middle_idx]

	# Note: if min_perm_size is greater than 3, can adjust in the sampling
	result_X = np.concatenate((X[[0]], middle_X, X[[-1]]), axis=0)
	result_Y = np.concatenate((Y[[0]], middle_Y, Y[[-1]]), axis=0)

	for j in range(min_perm_size):
		# Make pairing for rank(a) > rank(b)
		for k in range(min_perm_size):
			if np.array_equal(result_X[j], result_X[k]):
				pass
			else:
				X_new.append(result_X[j] - result_X[k])
				Y_new.append(np.sign(result_Y[j] - result_Y[k]))
	return X_new, Y_new


def pairwise_bert_transform(X_text, grade_level, min_perm_size=3):
	"""
	This function forms pairwise permutations suitable for BERT models

	X_text: Array of texts aggregated by slug
	grade_level: Array of grade levels
	min_perm_size: Number of permutations to form

	Return first_texts: texts in the first position of a pairwise permutation, second texts: texts in the second position, and Y: the pairwise binary label
	"""
	first_texts = []
	second_texts = []
	Y  = []

	for i in range(len(X_text)):
		slug_text = X_text[i]
		curr_grades = grade_level[i]

		if len(slug_text) < min_perm_size:
			first_texts.extend([slug_text[0], slug_text[1]])
			second_texts.extend([slug_text[1], slug_text[0]])
			Y.append([float(curr_grades[0]), float(curr_grades[1])])
			Y.append([float(curr_grades[1], float(curr_grades[0]))])
			continue
		else:
			middle_idx = np.random.choice(range(1, len(slug_text) - 1), size=min_perm_size-2, replace=False)
			# NOTE: This part currently only works if len(middle_idx) == 1
			middle_X, middle_Y = slug_text[middle_idx[0]], curr_grades[middle_idx[0]]
			perm_texts = [slug_text[0], middle_X, slug_text[-1]]
			perm_grades = [curr_grades[0], middle_Y, curr_grades[-1]]

			for j in range(3):
				# Make pairing for readability level (j) > readability level (k)
				for k in range(3):
					if perm_texts[j] != perm_texts[k]:
						first_texts.append(perm_texts[j])
						second_texts.append(perm_texts[k])
						Y.append([float(perm_grades[j]), float(perm_grades[k])])

	return first_texts, second_texts, Y
						

def get_vecs(filenames, datapath, model):
	"""
	Retrieves word embeddings for a text file.  Returns the averaged representation of word embeddings

	filenames: String filepaths of text files
	datapath: Directory containing the text files
	model: Word Embedding Model

	Returns the averaged representation of word embeddings for the input text.  
	"""
	X = []
	for f in filenames:
		text = get_text(f, datapath)
		tokens = [w for w in re.split('\.|\\n|\s', text) if w != '']
		vec_sum = np.zeros(300)
		counter = 0
		for t in tokens:
			try:
				vec_sum += model[t]
				counter += 1
			except:
				pass
		X.append((vec_sum / counter))

	return np.asarray(X)


def pairwise_predict(model, slugs):
	"""
	This method predicts on test data using a pairwise ranking model

	model: Pairwise prediction model
	slugs: Slug groupings of text used for prediction  

	Returns the predicted scores for the pairwise test set.
	"""
	all_scores = []
	for s in slugs:
		# Form permutations and predict across reading levels
		for i in range(s.shape[0]):
			rank_score = 0
			for j in range(s.shape[0]):
				if np.array_equal(s[i],s[j]):
					continue
				pred = model.predict([s[i] - s[j]])
				if not np.isnan(pred):	
					rank_score += model.predict([s[i] - s[j]])
			all_scores.append(rank_score)
	return np.asarray(all_scores)


def pairwise_bert_predict(model, text_slugs, testing_index, bert_model_name, text_limit=512):
	"""
	Function for prediction and recording using a pairwise BERT model

	model: Pairwise BERT model used for prediction
	text_slugs: Text data aggregated by slugs
	testing_index: Indices of data to be ranked and recorded
	bert_model_name: String name of the Pairwise BERT method
	text_limit: Token limit on BERT method

	Returns max_scores: pairwise aggregate scores for BERT and their original indices
	"""
	first_texts = []
	second_texts = []
	repped_indices = []
	for s in range(len(text_slugs)):
		curr_slug = text_slugs[s]
		curr_indices = testing_index[s]
		for i in range(len(curr_slug)):
			for j in range(len(curr_slug)):
				if curr_indices[i] != curr_indices[j]:
					first_texts.append(curr_slug[i])
					second_texts.append(curr_slug[j])
					repped_indices.append(curr_indices[i])

	tokenizer = BertTokenizer.from_pretrained(bert_model_name)
	tokenized_input = tokenizer(first_texts, second_texts, max_length=text_limit, padding='max_length', truncation=True, return_tensors='np')
	test_data = tf.data.Dataset.from_tensor_slices((dict(tokenized_input))).batch(16)
	# test_data = dict(tokenized_input)
	unsummed_scores = model.predict(test_data)['logits'] 
	unsummed_scores = tf.nn.softmax(unsummed_scores)
	zipped_scores = list(zip(repped_indices, unsummed_scores))
	scores_dict = {}
	for k, v in zipped_scores:
		val = scores_dict.get(k, np.asarray([0,0]))
		scores_dict[k] = val + v
	scores = np.asarray(list(scores_dict.values()))
	max_scores = scores[:,0]
	return max_scores, list(scores_dict.keys())


def construct_training_data(slug_df, we_model, datapath):
	"""
	Arranges training data for pairwise ranking methods, aggregating by slugs and forming pariwise permutations.

	slug_df: Dataframe of text data aggregated by slugs
	we_model: Word Embedding model
	datapath: Filepath of data

	Returns X_train: pairwise  training inputs and Y_train: binary label for pairwise inputs
	"""
	X_train = []
	Y_train = []

	for index, row in slug_df.iterrows():
		train_vecs = get_vecs(row['filename'], datapath, we_model)

		if train_vecs.shape[0] < 2:
			# Skip if slug only has 1 reading level
			continue

		train_reading_lvl = np.asarray(row['grade_level'])
		X_diff, Y_diff = pairwise_transform(train_vecs, train_reading_lvl)
		X_train.append(X_diff) 
		Y_train.append(Y_diff)
	
	X_train = np.asarray(X_train).reshape(-1, 300)
	Y_train = np.asarray(Y_train).reshape(-1)

	return X_train, Y_train

def construct_bert_training_data(slug_df, datapath, bert_model_name, text_limit=512):
	"""
	Arranges training data for a pairwise BERT model.  


	slug_df: Dataframe of text data aggregated by slugs
	datapath: Filepath of data
	bert_model_name: Name of the BERT model
	text_limit: Token limit of the BERT model

	Returns a tuple of train_data: pairwise BERT training inputs and Y: pairwise BERT binary labels
	"""
	X_text = []
	grade_levels = []

	for index, row in slug_df.iterrows():
		slug_texts = []
		for f in row['filename']:
			slug_texts.append(get_text(f, datapath))
		if len(slug_texts) > 1:
			X_text.append(slug_texts)
			grade_levels.append(row['grade_level'])

	first_texts, second_texts, Y = pairwise_bert_transform(X_text, grade_levels)
	tokenizer = BertTokenizer.from_pretrained(bert_model_name)
	tokenized_input = tokenizer(first_texts, second_texts, max_length=text_limit, padding='max_length', truncation=True, return_tensors='np')
	train_data = tf.data.Dataset.from_tensor_slices((dict(tokenized_input), Y))
	train_data = train_data.shuffle(len(train_data)).batch(5)
	# train_data = dict(tokenized_input)
	return (train_data, Y)

def construct_testing_data(slug_df, we_model, datapath):
	"""
	slug_df: Dataframe of text data aggregated by slugs
	we_model: Word Embedding model
	datapath: Filepath of data

	Returns X_test: testing data for a pairwise model and testing_index: the intended indices for the data
	"""
	X_test = []
	testing_index = []
	for index, row in slug_df.iterrows():
		test_vecs = get_vecs(row['filename'], datapath, we_model)

		# Skip slugs and indices that only have 1 reading level
		if test_vecs.shape[0] == 1:
			continue

		test_reading_lvl = np.asarray(row['grade_level'])
		X_test.append(test_vecs)
		testing_index.extend(row['Original_Index'])
	return X_test, testing_index


def construct_bert_testing_data(slug_df, datapath):
	"""
	Predicts on testing data using a non-pairwise BERT model
	
	slug_df: Dataframe of text data aggregated by slugs
	datapath: Filepath of data

	Returns X_test: testing data for a pairwise BERT model and testing_index: the original indices for the testing data
	"""
	X_text_test = []
	testing_index = []
	for index, row in slug_df.iterrows():
		slug_texts = []
		for f in row['filename']:
			slug_texts.append(get_text(f, datapath))
		if len(slug_texts)  == 1:
			continue
		X_text_test.append(slug_texts)
		testing_index.append(row['Original_Index'])

	return X_text_test, testing_index

def construct_sample_lists(df):
	"""
	Constructs a sampled lists from text according to their original grade levels

	df: Dataframe of text data

	Returns df
	"""
	grade_levels = pd.unique(df['grade_level'])
	np.random.shuffle(grade_levels)

	for g in grade_levels:
		only_grade_df = df[df['grade_level'] == g].sample(frac=1)
		list_indices = list(range(len(only_grade_df)))
		df.loc[only_grade_df.index, 'slug'] = list_indices

	return df



def main(transfer=False, bert=False):
	"""
	This script performs 5-Fold Cross Validation on a Dataframe where one of the columns consists of filenames for the corresponding text data of that row.  
	The prediction results in each of the folds is mapped back to the same index of the rows that are sampled for testing.  
	unctions for construction pairwise training sets and pairwise scoring are used to construct a pairwise ranking scheme.
	Features used for this process are the same word-level embedding models from the classification and regression pipelines in 'ml_scores.py'.

	In 'transfer' mode, the pairwise training scheme is applied to one dataset, and tested on the rest of the other datasets.

	In 'bert' mode, the pairwise BERT models are employed to generate baselines.  If 'bert' ==False, word embedding methods will be evalutaed

	In both modes, the pairwise scoring is appended as column in the original feature DataFrame.  Each piece of text has its own score.

	Running this script at once will train, predict, and form predicted ranking scores for pairwise BERT or word embedding methods.  
	""" 
	curr_path = os.path.abspath('..')
	newsela_es_folderpath = os.path.join(curr_path, 'Datasets', 'newsela_article_corpus_2016-01-29', 'articles')
	newsela_es_filepath = os.path.join(curr_path, 'Datasets','newsela_es_rank_features.csv')
	newsela_es = pd.read_csv(newsela_es_filepath)

	newsela_folderpath = os.path.join(curr_path, 'Datasets', 'newsela_article_corpus_2016-01-29', 'articles')
	newsela_filepath = os.path.join(curr_path, 'Datasets','newsela_en_rank_features.csv')
	newsela = pd.read_csv(newsela_filepath)

	os_eng_folderpath = os.path.join(curr_path, 'Datasets/OneStopEnglish', 'All-Text-Files-Separated-by-ReadingLevel')
	os_eng_filepath = os.path.join(curr_path, 'Datasets', 'os_eng_rank_features.csv')
	os_eng = pd.read_csv(os_eng_filepath)

	tr_en_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_en')
	tr_en_filepath = os.path.join(curr_path, 'Datasets', 'tr_english_rank_features.csv')
	tr_en = pd.read_csv(tr_en_filepath)

	tr_fr_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_fr')
	tr_fr_filepath = os.path.join(curr_path, 'Datasets', 'tr_fr_rank_features.csv')
	tr_fr = pd.read_csv(tr_fr_filepath)


	test_commoncore_folderpath = os.path.join(curr_path, 'Datasets', 'comcore', 'data')
	test_commoncore_filepath = os.path.join(curr_path, 'Datasets', 'commoncore_rank_features.csv')
	test_commoncore = pd.read_csv(test_commoncore_filepath)

	output_folderpath = os.path.join(curr_path, 'Datasets')

	ml_features_filepath = os.path.join(output_folderpath, 'ml_feature_names.pkl')
	ml_feature_names = Path(ml_features_filepath)
	
	if ml_feature_names.is_file():
		with open(ml_features_filepath, 'rb') as f:
			ml_feature_names = pickle.load(f)
	else:
		ml_feature_names = []

	data = [('newsela', newsela_folderpath, newsela_filepath, newsela),('os_eng', os_eng_folderpath, os_eng_filepath, os_eng),('transread_en', tr_en_folderpath,  tr_en_filepath, tr_en)]
	if bert:
		if transfer:
			for i in range(len(data)):
				train_data_name = data[i][0]
				if train_data_name != 'newsela':
					break
				print('Transfer Training on {}'.format(train_data_name))

				if os.path.isdir('newsela_english_bert_pairwise'):
					# Check if the BERT model is already trained
					print("Model is already trained")
					model = TFBertForSequenceClassification.from_pretrained('newsela_english_bert_pairwise')
				else:
					train_data_folderpath = data[i][1]
					train_df = data[i][3]
					training_slugs = train_df.groupby(['slug'])[['filename', 'grade_level']].agg(list)
					training_slugs.reset_index(inplace=True)
					train_data, Y = construct_bert_training_data(training_slugs, train_data_folderpath,'bert-base-multilingual-uncased')
					validation = train_data.take(16)
					train_data = train_data.skip(16)
					model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
					model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tfr.keras.losses.PairwiseLogisticLoss())

					model.fit(train_data, validation_data=train_data, epochs=3)
					model.save_pretrained('newsela_english_bert_pairwise')

				for j in range(len(data)):
					if i == j:
						continue
					test_data_name = data[j][0]
					test_data_folderpath = data[j][1]
					test_data_filepath = data[j][2]
					test_df = data[j][3]


					feature_name = train_data_name + "_mbertRank"
					print("Predicting on {}".format(test_data_name))

					if ('commoncore' in test_data_name):
						testing_slugs = test_df.groupby(['Genre'])[['filename', 'grade_level', 'Original_Index']].agg(list)
					else:
						testing_slugs = test_df.groupby(['slug'])[['filename', 'grade_level', 'Original_Index']].agg(list)
					testing_slugs.reset_index(inplace=True)
					test_data, testing_index = construct_bert_testing_data(testing_slugs, test_data_folderpath)
					
					# If testing on transread, way too big for tokenizing in one batch
					batch_size = 1000
					for i in range(0, len(test_data), batch_size):
						print("Round {}".format(i))
						batch_test_data, batch_testing_index = test_data[i:i + batch_size], testing_index[i:i + batch_size]
						print(batch_testing_index)
						preds, pred_index = pairwise_bert_predict(model, batch_test_data, batch_testing_index, 'bert-base-multilingual-uncased',)
						test_df.loc[pred_index, feature_name] = preds

					if feature_name not in ml_feature_names:
						ml_feature_names.append(feature_name)
					test_df.to_csv(test_data_filepath, index=False)

		else:
 
			for i in range(len(data)):
				data_name = data[i][0]
				print('Training on {}'.format(data_name))
				data_folderpath = data[i][1]
				data_filepath = data[i][2]
				df = data[i][3]


				feature_name = "bertRank"

				slugs = pd.unique(df['slug'])
				kf = KFold(n_splits=5)
				counter = 1

				for slug_train_index, slug_test_index in kf.split(slugs):
					print("Training Fold:{}".format(counter))
					counter += 1
					# Indices of testing and training data

					# Aggregate the texts according to slugs for the pairwise construction, since we only rank within slugs
					training_slugs = df[df['slug'].isin(slugs[slug_train_index])].groupby(['slug'])[['filename', 'grade_level']].agg(list)
					# Keeping track of the testing data indexes for re-insertion
					testing_slugs = df[df['slug'].isin(slugs[slug_test_index])].groupby(['slug'])[['filename', 'grade_level', 'Original_Index']].agg(list)

					training_slugs.reset_index(inplace=True)
					testing_slugs.reset_index(inplace=True)
					# print(training_slugs)
					train_data, Y = construct_bert_training_data(training_slugs, data_folderpath, 'bert-base-uncased')
					# train_data, Y = construct_bert_training_data(training_slugs, data_folderpath,'bert-base-multilingual-uncased')
					test_data, testing_index = construct_bert_testing_data(testing_slugs, data_folderpath)
					print("LENGTH_TESTING_INDEX{}".format(len(testing_index)))



					validation = train_data.take(16)
					train_data = train_data.skip(16)
					model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
					model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tfr.keras.losses.PairwiseLogisticLoss())
					model.fit(train_data, validation_data=validation, epochs=3)
					
					preds, testing_inds = pairwise_bert_predict(model, test_data, testing_index, 'bert-base-uncased',)
					df.loc[testing_inds, feature_name] = preds

					if feature_name not in ml_feature_names:
						ml_feature_names.append(feature_name)
				df.to_csv(data_filepath, index=False)

	else:

		we_models = []

		for we_m in WORD_EMBED_NAMES:
			print('Loading {}'.format(we_m))
			we_model = gensim.downloader.load(we_m)
			we_models.append(we_model)

		we_models = list(zip(WORD_EMBED_NAMES, we_models))

		if transfer:
			for i in range(len(data)):
				train_data_name = data[i][0]
				print('Training on {}'.format(train_data_name))
				train_data_folderpath = data[i][1]
				train_df = data[i][3]

				if train_data_name == 'newsela_es':
					continue

				for j in range(len(data)):
					test_data_name = data[j][0]
					if train_data_name == test_data_name:
						continue


					print('Predicting on {}'.format(test_data_name))
					test_data_folderpath = data[j][1]
					test_data_filepath = data[j][2]
					test_df = data[j][3]
					if 'Original_Index' not in test_df.columns:
						test_df['Original_Index'] = test_df.index
					
					for embed_name, we_model in we_models:
						print("Loading Model {}".format(embed_name))

						training_slugs = train_df.groupby(['slug'])[['filename', 'grade_level']].agg(list)
						testing_slugs = test_df.groupby(['slug'])[['filename', 'grade_level', 'Original_Index']].agg(list)

						training_slugs.reset_index(inplace=True)
						testing_slugs.reset_index(inplace=True)

						X_train, Y_train = construct_training_data(training_slugs, we_model, train_data_folderpath)
						X_test, testing_index = construct_testing_data(testing_slugs,we_model, test_data_folderpath)

						model = LinearSVC()
						model.fit(X_train, Y_train)
						preds = pairwise_predict(model, X_test)
						preds = np.ndarray.flatten(preds)

						feature_name = '{}_{}_pairwiseSVM'.format(train_data_name, embed_name)
						test_df.loc[testing_index, feature_name] = preds

					if feature_name not in ml_feature_names:
						ml_feature_names.append(feature_name)

					test_df.to_csv(test_data_filepath, index=False)
			return

		1st Level Loop: loop through datasets
		2md Level Loop: Loop through word embedding models
		3rd Level Loop: Loop through each train/test split in the 5-Fold CV

		for i in range(len(data)):
			data_name = data[i][0]
			data_folder_path = data[i][1]
			df = data[i][2]

			print("Dataset: {}".format(data_name))
			if 'Original_Index' not in df.columns:
				df['Original_Index'] = df.index

			for embed_name, we_model in we_models:
				print("Loading Model {}".format(embed_name))
				feature_name = "{}_pairwiseSVM".format(embed_name)

				slugs = pd.unique(df['slug'])
				kf = KFold(n_splits=5)

				for slug_train_index, slug_test_index in kf.split(slugs):
					# Indices of testing and training data

					# Aggregate the texts according to slugs for the pairwise construction, since we only rank within slugs
					training_slugs = df[df['slug'].isin(slugs[slug_train_index])].groupby(['slug'])[['filename', 'grade_level']].agg(list)

					# Keeping track of the testing data indexes for re-insertion
					testing_slugs = df[df['slug'].isin(slugs[slug_test_index])].groupby(['slug'])[['filename', 'grade_level', 'Original_Index']].agg(list)

					training_slugs.reset_index(inplace=True)
					testing_slugs.reset_index(inplace=True)

					X_train, Y_train = construct_training_data(training_slugs, we_model, data_folder_path)
					X_test, testing_index = construct_testing_data(testing_slugs, we_model, data_folder_path)

					model = LinearSVC()
					model.fit(X_train, Y_train)
					preds = pairwise_predict(model, X_test)
					preds = np.ndarray.flatten(preds)

					# Each text (at the reading level) will have its own score, which is a pairwise summation of the prediction against 
					# other pairs in the same slug.  
					df.loc[testing_index, feature_name] = preds

				if feature_name not in ml_feature_names:
					ml_feature_names.append(feature_name)

		for i in range(len(data)):
			data[i][3].to_csv(data[i][2], index=False)

		with open(os.path.join(output_folderpath, 'ml_feature_names.pkl'), 'wb') as f:
			pickle.dump(ml_feature_names, f)
	
	return 

if __name__ == '__main__':
	main(bert=True, transfer=False)
