from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from linguistic_features_for_readability import HAN_SVM, BERT_SVM
from nltk.tokenize import sent_tokenize
from mord import LogisticIT
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from transformers import BertTokenizer
import os
import gensim.downloader 
import numpy as np
import pandas as pd
import pickle
import re
import tensorflow as tf

WORD_EMBED_NAMES = ['fasttext-wiki-news-subwords-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-300']
# WORD_EMBED_NAMES = ['word2vec-google-news-300']
# ML_MODELS = [("linear_reg", LinearRegression()), ("class_logistic_reg", LogisticRegression(max_iter=1000)),
# 			("class_svm", SVC()), ("ordinal_logistic_reg", LogisticIT()), ("han_svm", HAN_SVM())]
ML_MODELS = [("han_only_cat", HAN_SVM())]
HAN_MAX_WORDS_PER_SENT = 60
HAN_MAX_SENT = 60
HAN_MAX_VOC_SIZE = 20000
HAN_EMBED_DIM = 300

def get_text(filename, dp):
	"""
	filename: filepath of the text file
	dp: filepath of the folder containing 'filename'

	Returns text extracted from the filename
	"""
	fp = os.path.join(dp, filename)
	with open(fp, 'r', encoding='ISO-8859-1') as file:
		text = file.read()
	return text

def build_training_data(df, datapath, model_name, model):
	"""
	df: DataFrame with the metadata containing filenames.  
	datapath: Filepath of the folder containing text files
	model_name: String, name of the pretrained models from gensim.

	Returns a tuple, where the first element is a Numpy matrix of vectors normalized over the 
	length of the tokens, the second element is a Numpy matrix of un-normalized vectors.
	"""
	text_vecs_sum = []
	text_vecs_avg = []

	for index, row in df.iterrows():
		text = get_text(row['filename'], datapath)

		tokens = [w for w in re.split('\.|\\n|\s', text) if w != '']
		
		# Loop and sum to avoid accumlating an inner huge matrix
		vec_sum = np.zeros(300)
		counter = 0
		for t in tokens:
			try:
				vec_sum += model[t]
				counter += 1
			except:
				pass
		
		text_vecs_sum.append(vec_sum)
		text_vecs_avg.append(vec_sum / counter)

	return (np.asarray(text_vecs_avg), np.asarray(text_vecs_sum))

def cross_fold_prediction(df, X, pool, we_model_name, ml_model_name, ml_model_meth):
	"""
	This function takes as input a DataFrame of text features (consisting of the 'Y' label 'grade_level'),
	X, a set of word embeddings for training machine learning method. 'pool' an aggregation method for the embeddings,
	'we_model_name' the name of the word embeddings used, 'ml_model_name' the name of the machine learning method used,
	and 'ml_model_meth' the sklearn class for the machine learning method.

	This function will perform 5-fold cross validation on datapoints in the dataframe.  The machine learning method will be
	trained using the text embeddings corresponding to 4 splits of the data and predict on the last split.  The predictions on
	the last split will be tracked and attributed to the appropriate index in the dataframe, added as a text feature.


	df: Dataframe containing text metadata
	X: Word embeddings corresponding to data in 'df'
	pool: Pooling method, sum or avg.
	we_model_name: Name of the word embedding method
	ml_model_meth: Word embedding model
	"""
	# If feature already exists, exit function
	feature_name = pool + '_' + we_model_name + '_' + ml_model_name
	if feature_name in df.columns:
		print('Feature already present')
		return

	print('ML Model: {} {}'.format(pool, ml_model_name))
	print('Starting cross fold validation...')

	# This function assumes the fact that the order of 'X' is the same as the order of 'df'
	# Need to keep document consistent, don't split across different reading levels of the same document
	slugs = pd.unique(df['slug'])
	kf = KFold(n_splits=5)
	df[feature_name] = np.nan
	counter = 0

	if 'class' in ml_model_name or 'ordinal' in ml_model_name:
		unique_grade_levels = pd.unique(df['grade_level'])

		unique_grade_levels = np.sort(unique_grade_levels)
		grade_encoder = dict(zip(unique_grade_levels, range(len(unique_grade_levels))))
		grade_decoder = {v:k for k,v in grade_encoder.items()}

	# Calculating splits across slugs, finding appropriate training and testing data from slug splits
	for train_index, test_index in kf.split(slugs):		
		counter += 1
		print("Fold:{}".format(counter))

		training_inds = df[df['slug'].isin(slugs[train_index])].index
		testing_inds = df[df['slug'].isin(slugs[test_index])].index

		X_train = X[training_inds, :]
		Y_train = df['grade_level'][training_inds]

		X_test = X[testing_inds, :]
		Y_test = df['grade_level'][testing_inds]

		# If Newsela, use raw reading grade scale for linear regression
		# Convert the reading scale to an equidistance scale (0,1,2,..,8) to treat as classes for 
		# classification and ordinal classification.

		if "class" in ml_model_name or "ordinal" in ml_model_name:
			Y_train = Y_train.apply(lambda x: grade_encoder[x])
			Y_test = Y_test.apply(lambda x: grade_encoder[x])

		ml_model_meth.fit(X_train, Y_train)

		if 'logistic_reg' in ml_model_name:
			preds = ml_model_meth.predict(X_test)
		else:
			preds = ml_model_meth.predict(X_test)

		if "class" in ml_model_name or "ordinal" in ml_model_name:
			preds = [grade_decoder[p] for p in preds]

		df.loc[testing_inds, feature_name] = preds

	return

def build_bert_data(df, datapath):
	"""
	df: Dataframe  of text metadata
	datapath: Filepath of directory of text  files

	Returns X_tokenized: BERT Tokenized text representations

	"""
	all_text = []
	X = []
	for index, row in df.iterrows():
		all_text.append(get_text(row['filename'], datapath))
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	X_tokenized = tokenizer(all_text, max_length=128, padding='max_length', truncation=True, return_tensors='np')
	return X_tokenized


def slice_bert_features(original_tokenized, indices):
	"""
	Returns sliced tensors corresponding to BERT model inputs, such as input_ids, token_ids and attention_mask ids

	original_tokenized: is a dictionary of tensors with the above features.
	indices: are an array of integer indices

	Returns dictionary of sliced features
	"""
	# Make a copy of the original tokenized input for separate slicing after
	tokenized_dict = original_tokenized.copy()
	for feature in tokenized_dict.keys():
		tokenized_dict[feature] = tokenized_dict[feature][indices]
	return tokenized_dict


def build_han_data(df, datapath, max_sent, max_word_per_sent, we_model):
	"""
	df: Dataframe of text metadata
	datapath: Filepath corresponding to the directory containing text files
	max_sent: Max number of sentences
	max_word_per_sent: Max number of tokens per sentence
	we_model: Word Embedding model

	Returns a tuple of X: input features for HAN and embedding_matrix: arrays for accessing embeddings
	"""
	# Build tokenized
	all_text = []
	X = []
	for index, row in df.iterrows():
		all_text.append(get_text(row['filename'], datapath))

	word_tokenizer = Tokenizer(num_words=HAN_MAX_VOC_SIZE)
	word_tokenizer.fit_on_texts(all_text)

	for i, text in enumerate(all_text):
		sentences = sent_tokenize(text)
		tokenized_sentences = word_tokenizer.texts_to_sequences(sentences)
		tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=HAN_MAX_WORDS_PER_SENT)

		sent_pad_size = HAN_MAX_SENT - tokenized_sentences.shape[0]

		if sent_pad_size < 0:
			tokenized_sentences = tokenized_sentences[:HAN_MAX_SENT]
		else:
			tokenized_sentences = np.pad(tokenized_sentences, ((0, sent_pad_size), (0, 0)), mode='constant', constant_values=0)
		X.append(tokenized_sentences)
	X = np.asarray(X)

	embedding_matrix = np.random.random((len(word_tokenizer.word_index) + 1, HAN_EMBED_DIM))
	embedding_matrix[0] = 0
	for word, index in word_tokenizer.word_index.items():
		try:
			embedding_matrix[index] = we_model[word]
		except:
			pass

	return (X, embedding_matrix)

def nn_train_predict(df, folderpath, nn_model_name, nn_model_method, we_name, we_model, transfer=False, **kwargs):
	"""
	df: Dataframe of text metadata
	folderpath: Filepath for directory of text files
	nn_model_name: String name of NN model
	nn_model_method: NN model
	we_name: String name of word embedding method
	we_model: Word embedding model
	transfer: True or False for transfer learning

	5-Fold Cross validation for a NN method.  Options for using input features from an SVM or straight from the text.  Predicted results are recorded as a column in 'df'
	"""

	print('NN Model: {}'.format( nn_model_name))
	print('Starting cross fold validation...')

	# If feature already exists, exit function
	if 'bert' in nn_model_name:
		feature_name = nn_model_name
	else:
		feature_name = we_name + '_' + nn_model_name

	unique_grade_levels = pd.unique(df['grade_level'])
	unique_grade_levels = np.sort(unique_grade_levels)
	grade_encoder = dict(zip(unique_grade_levels, range(len(unique_grade_levels))))
	grade_decoder = {v:k for k,v in grade_encoder.items()}
	
	Y = df['grade_level'].apply(lambda x: grade_encoder[x])
	Y = to_categorical(Y)

	if 'han' in nn_model_name:
		X, embedding_matrix = build_han_data(df, folderpath, HAN_MAX_SENT, HAN_MAX_WORDS_PER_SENT, we_model)
		if 'svm_reg' in nn_model_name:
			nn_model_method.initialize_params(HAN_MAX_WORDS_PER_SENT, HAN_MAX_SENT, Y.shape[1], embedding_matrix, 'mean_squared_error', with_svm=True)
		if 'svm_cat' in nn_model_name:
			nn_model_method.initialize_params(HAN_MAX_WORDS_PER_SENT, HAN_MAX_SENT, Y.shape[1], embedding_matrix, 'categorical_crossentropy', with_svm=True)
		if 'only_cat' in nn_model_name:
			nn_model_method.initialize_params(HAN_MAX_WORDS_PER_SENT, HAN_MAX_SENT, Y.shape[1], embedding_matrix, 'categorical_crossentropy')
		if 'only_reg' in nn_model_name:
			nn_model_method.initialize_params(HAN_MAX_WORDS_PER_SENT, HAN_MAX_SENT, Y.shape[1], embedding_matrix, 'mean_squared_error')

	if 'bert' in nn_model_name:
		X = build_bert_data(df, folderpath)
		nn_model_method.initialize_params(Y.shape[1])

	slugs = pd.unique(df['slug'])
	kf = KFold(n_splits=5)
	counter = 0
	df[feature_name] = np.nan
	for train_index, test_index in kf.split(slugs):
		counter += 1
		print("Fold:{}".format(counter))
		nn_model_method.re_initialize_params()

		training_inds = df[df['slug'].isin(slugs[train_index])].index
		testing_inds = df[df['slug'].isin(slugs[test_index])].index

		if 'bert' in nn_model_name:
			X_train, Y_train = slice_bert_features(X, training_inds), Y[training_inds]
			X_test, Y_test = slice_bert_features(X, testing_inds),  Y[testing_inds]

		if 'han' in nn_model_name:
			X_train, Y_train = X[training_inds], Y[training_inds]
			X_test, Y_test = X[testing_inds], Y[testing_inds]

		nn_model_method.fit(X_train, Y_train)
		preds = nn_model_method.predict(X_test)
		preds = [grade_decoder[p] for p in preds]
		df.loc[testing_inds, feature_name] = preds
	return 


def build_features(we_name, we_model, data_name, df, folderpath, output_folderpath, feature_names):
	"""
	we_name: Mame of the word embedding model
	we_model: Model method of the word embedding model
	data_name: Name of the dataset
	df: Dataframe consisting of the metadata for the text
	folderpath: Filepath of directory containing the text files
	output_folderpath: Output filepath
	feature_names: Names of current machine learning model features used for ranking

	This function serves as a loop for iterative usage of the 'cross_fold_prediction' function.
	This function also operates on both the 'avg' and 'sum' pooling methods for aggregating the vectors
	"""
	print('Running {}'.format(data_name))
	we_avg, we_sum = build_training_data(df, folderpath, we_name, we_model)
	# Use both summed and averaged

	for ml_m in ML_MODELS:
		if 'han' in ml_m[0]:
			nn_feat_name = we_name + '_' + ml_m[0]
			nn_train_predict(df, folderpath, ml_m[0], ml_m[1], we_name, we_model, feature_names)
			if nn_feat_name not in feature_names:
				feature_names.append(nn_feat_name)
		elif 'bert' in ml_m[0]:
			nn_feat_name = ml_m[0]
			nn_train_predict(df, folderpath, ml_m[0], ml_m[1], None, None, feature_names)
			if nn_feat_name not in feature_names:
				feature_names.append(nn_feat_name)

		else:
			f1_name = 'sum'+ '_' + we_name + '_' + ml_m[0]
			f2_name = 'avg' + '_' + we_name + '_' + ml_m[0]

			# if (f1_name not in df.columns) or (f2_name not in df.columns):
			cross_fold_prediction(df, we_sum, 'sum', we_name, ml_m[0], ml_m[1])
			cross_fold_prediction(df, we_avg, 'avg', we_name, ml_m[0], ml_m[1])
			if (f1_name not in feature_names) or (f2_name not in feature_names):
				feature_names.extend([f1_name, f2_name])
	return 


def build_transfer_features(we_name, we_model, train_data_name, test_data_name, train_data_folderpath,
							test_data_folderpath, train_df, train_avg, train_sum, test_df, ml_feature_names):
	"""
	we_name: Name of word_embedding model
	we_model: Word_embedding method
	train_data_name: Name of the dataset to be trained on
	test_data_name: Name of the dataset being tested on
	train_data_folderpath: Filepath of the training text files
	test_datat_folderpath: Filepath of the test text files
	train_df: Dataframe consisting of the metadata for training
	train_avg: Avged word embeddings of training data
	train_sum: Summed word embeddings of the training data
	test_df: Dataframe consisting of the metadata for testing
	ml_feature_names: Array of names of machine learning methods

	
	This function performs cross-corpus evaluation of word embeddings and various machine learning methods.
	This function iterates over the list of available machine learning methods, trains the methods on the 'train_avg'
	and 'train_sum' word embeddings provided in input, and predicts on the word embeddings generated from 'test_df'.
	New machine learning features are then added to the master list of ML features ('ml_feature_names') if they are absent
	"""
	for ml_m in ML_MODELS:
		print('Model: {}'.format(ml_m[0]))
		target = train_df['grade_level']
		if 'class' or 'ordinal' in ml_m[0]:
			unique_grade_levels = pd.unique(train_df['grade_level'])
			unique_grade_levels = np.sort(unique_grade_levels)
			grade_encoder = dict(zip(unique_grade_levels, range(len(unique_grade_levels))))
			grade_decoder = {v:k for k,v in grade_encoder.items()}
			target = train_df['grade_level'].apply(lambda x: grade_encoder[x])

		summed_model = ml_m[1].fit(train_sum, target)
		avged_model = ml_m[1].fit(train_avg, target)

		print('Training on {}'.format(train_data_name))
		print(['Predicting on {}'.format(test_data_name)])

		test_avg, test_sum = build_training_data(test_df, test_data_folderpath, we_name, we_model)
		avg_feat_name = '{}_avg{}_{}'.format(train_data_name, we_name, ml_m[0])
		sum_feat_name = '{}_sum_{}_{}'.format(train_data_name, we_name, ml_m[0])
		test_df[avg_feat_name] = avged_model.predict(test_avg)
		test_df[sum_feat_name] = summed_model.predict(test_sum)

		if (avg_feat_name not in ml_feature_names) or (sum_feat_name not in ml_feature_names):
			ml_feature_names.extend([avg_feat_name, sum_feat_name])
	return


def main(transfer=False):
	"""
	This function provides the main script for generating machine learning scores and transfer-machine learning scores for 
	readability prediction on all the available datasets.  This script can perform inter or intra corpus evaluation with the combination
	of word embedding models and machine learning models specified.  

	Intra and inter corpus evaluation can be done sequentially (intra first), however the 2 separate loops can be commented out to
	perform either one of the evaluation modes exclusively.  Word embedding models can be added/removed by editing the 
	'WORD_EMBED_NAMES' variable at the top of the file, and machine learning models  can be added/removed by editing the
	'ML_MODELS' variable.
	"""

	we_models = []

	for we_m in WORD_EMBED_NAMES:
		print('Loading {}'.format(we_m))
		we_model = gensim.downloader.load(we_m)
		we_models.append(we_model)

	we_models = list(zip(WORD_EMBED_NAMES, we_models))

	curr_path = os.path.abspath('..')

	newsela_folderpath = os.path.join(curr_path, 'Datasets', 'newsela_article_corpus_2016-01-29', 'articles')
	newsela_filepath = os.path.join(curr_path, 'Datasets','newsela_en_rank_features.csv')
	newsela = pd.read_csv(newsela_filepath)

	os_eng_folderpath = os.path.join(curr_path, 'Datasets/OneStopEnglish', 'All-Text-Files-Separated-by-ReadingLevel')
	os_eng_filepath = os.path.join(curr_path, 'Datasets', 'os_eng_rank_features.csv')
	os_eng = pd.read_csv(os_eng_filepath)

	tr_en_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_en')
	tr_en_filepath = os.path.join(curr_path, 'Datasets', 'tr_english_rank_features.csv')
	tr_en = pd.read_csv(tr_en_filepath)

	# Master set of feature names
	output_folderpath = os.path.join(curr_path, 'Datasets')

	# Check if the ml_feature_names file exists
	ml_features_filepath = os.path.join(output_folderpath, 'ml_feature_names.pkl')
	ml_feature_names = Path(ml_features_filepath)
	
	if ml_feature_names.is_file():
		with open(ml_features_filepath, 'rb') as f:
			ml_feature_names = pickle.load(f)
	else:
		ml_feature_names = []

	data = [('os', os_eng_folderpath, os_eng, os_eng_filepath),
			('newsela', newsela_folderpath , newsela, newsela_filepath),
			('transread_en', tr_en_folderpath, tr_en, tr_en_filepath)]

	for i in range(len(data)):
		data_name = data[i][0]
		data_folderpath = data[i][1]
		df = data[i][2]

		for we_name, we_model, in we_models:
			build_features(we_name, we_model, data_name, df, data_folderpath, output_folderpath, ml_feature_names)

	if transfer:
		for i in range(len(data)):
			train_data_name = data[i][0]
			train_data_folderpath = data[i][1]
			train_df = data[i][2]

			target = train_df['grade_level']

			for we_m in we_models:
				print('Loading {}'.format(we_m))
				we_name = we_m[0]
				we_model = we_m[1]
				train_avg, train_sum = build_training_data(train_df, train_data_folderpath, we_name, we_model)

				for j in range(len(data)):
					test_data_name = data[j][0]
					test_data_folderpath = data[j][1]
					test_df = data[j][2]

					if train_data_name == test_data_name:
						continue
					if ((train_data_name == 'newsela') and (test_data_name == 'os')) or\
					((train_data_name == 'os') and (test_data_name == 'newsela')):
						continue

					build_transfer_features(we_name, we_model, train_data_name, test_data_name, train_data_folderpath,
	 										test_data_folderpath, train_df, train_avg, train_sum, test_df, ml_feature_names)
	
	for i in range(len(data)):
		data[i][2].to_csv(data[i][3], index=False)

	output_filepath_ml_featnames = os.path.join(output_folderpath, 'ml_feature_names.pkl')
	with open(output_filepath_ml_featnames, 'wb') as f:
		pickle.dump(ml_feature_names, f)


if __name__ == '__main__':
	main(transfer=False)



