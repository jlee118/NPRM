from pathlib import Path
from ml_scores import get_text
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, TFBertForSequenceClassification


import os
import gensim.downloader 
import numpy as np
import pandas as pd
import pickle
import re
import tensorflow as tf

def build_bert_train_data(df, datapath, model_name,grade_encoder=None):
	all_text = []
	for index, row in df.iterrows():
		all_text.append(get_text(row['filename'], datapath))
	tokenizer = BertTokenizer.from_pretrained(model_name)
	X_tokenized = tokenizer(all_text, max_length=512, padding='max_length', truncation=True, return_tensors='np')
	
	if grade_encoder:
		Y = df['grade_level'].apply(lambda x: grade_encoder[x])
	else:
		Y = df['grade_level']
	
	train_data = tf.data.Dataset.from_tensor_slices((dict(X_tokenized), Y))
	train_data = train_data.shuffle(len(train_data)).batch(6)
	return train_data

def build_bert_test_data(df, datapath, model_name):
	all_text = []
	X = []
	for index, row in df.iterrows():
		all_text.append(get_text(row['filename'], datapath))

	tokenizer = BertTokenizer.from_pretrained(model_name)
	test_tokenized = tokenizer(all_text, max_length=512, padding='max_length', truncation=True, return_tensors='np')
	test_data = tf.data.Dataset.from_tensor_slices(dict(test_tokenized)).batch(16)
	return test_data

def main(transfer=False):
	curr_path = os.path.abspath('..')
	newsela_es_folderpath = os.path.join(curr_path, 'Datasets', 'newsela_article_corpus_2016-01-29', 'articles')
	newsela_es_filepath = os.path.join(curr_path, 'Datasets','newsela_es_rank_features.csv')
	newsela_es = pd.read_csv(newsela_es_filepath)

	newsela_folderpath = os.path.join(curr_path, 'Datasets', 'newsela_article_corpus_2016-01-29', 'articles')
	newsela_filepath = os.path.join(curr_path, 'Datasets','newsela_en_rank_features.csv')
	newsela = pd.read_csv(newsela_filepath)

	tr_en_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_en')
	tr_en_filepath = os.path.join(curr_path, 'Datasets', 'tr_english_rank_features.csv')
	tr_en = pd.read_csv(tr_en_filepath)

	tr_fr_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_fr')
	tr_fr_filepath = os.path.join(curr_path, 'Datasets', 'tr_fr_rank_features.csv')
	tr_fr = pd.read_csv(tr_fr_filepath)

	tr_pairs_en_folderpath = os.path.join(curr_path, 'Datasets/TransReadData_en')
	tr_pairs_en_filepath = os.path.join(curr_path, 'Datasets', 'tr_english_pairs_rank_features.csv')
	tr_pairs_en = pd.read_csv(tr_en_filepath)

	os_eng_folderpath = os.path.join(curr_path, 'Datasets/OneStopEnglish', 'All-Text-Files-Separated-by-ReadingLevel')
	os_eng_filepath = os.path.join(curr_path, 'Datasets', 'os_eng_rank_features.csv')
	os_eng = pd.read_csv(os_eng_filepath)

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

	data = [('newsela', newsela_folderpath, newsela_filepath, newsela)]

	if transfer:

		for i in range(len(data)):
			train_data_name = data[i][0]
			if train_data_name != 'newsela':
				break
			print('Transfer Training on {}'.format(train_data_name))
			train_data_folderpath = data[i][1]
			train_df = data[i][3]
			# feature_name = train_data_name + '_bert'
			feature_name = train_data_name + '_mbert_regression'

			if feature_name not in ml_feature_names:
				ml_feature_names.append(feature_name)

			# unique_grade_levels = pd.unique(train_df['grade_level'])
			# unique_grade_levels = np.sort(unique_grade_levels)
			# grade_encoder = dict(zip(unique_grade_levels, range(len(unique_grade_levels))))
			# grade_decoder = {v:k for k,v in grade_encoder.items()}
			if os.path.isdir('newsela_english_mbert_regression'):
				print("Model is already trained")
				model = TFBertForSequenceClassification.from_pretrained('newsela_english_mbert_regression')

			else:
				# train_data = build_bert_train_data(train_df, train_data_folderpath, grade_encoder, 'bert-base-multilingual-uncased')
				train_data = build_bert_train_data(train_df, train_data_folderpath, 'bert-base-multilingual-uncased')
				validation_data = train_data.take(16)
				train_data = train_data.skip(16)
				print(train_data)

				# if os.path.isfile('newsela_english_bert_classification'):
				# 	model = TFBertForSequenceClassification.from_pretrained('newsela_english_bert_classification')
				# else:
				# model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(grade_encoder.keys()))
				# model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
				model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=1)
				model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.MeanSquaredError())
				model.fit(train_data, validation_data=validation_data, epochs=5)
				# model.save_pretrained('newsela_english_bert_classification')
				model.save_pretrained('newsela_english_mbert_regression')

			for j in range(len(data)):
				if i == j:
					continue

				test_data_name = data[j][0]
				print('Testing on {}'.format(test_data_name))
				test_data_folderpath = data[j][1]
				test_data_filepath = data[j][2]
				test_df = data[j][3]


				batch_size = 1000
				for i in range(0, test_df.shape[0], batch_size):
					print("Round {}".format(i))
					batch_test_df = test_df.iloc[i:i+batch_size]
					batch_indices = batch_test_df['Original_Index']
					test_data = build_bert_test_data(batch_test_df, test_data_folderpath, 'bert-base-multilingual-uncased')
					preds = model.predict(test_data)[0]
					# preds = np.argmax(tf.nn.softmax(preds), axis=1)
					# pred_labels = [grade_decoder[p] for p in preds]
					test_df.loc[batch_indices, feature_name] = preds

				test_df.to_csv(test_data_filepath, index=False)

	else:
		for i in range(len(data)):
			data_name = data[i][0]

			if data_name == 'newsela_es':
				continue

			data_folderpath = data[i][1]
			data_filepath = data[i][2]
			df = data[i][3]
			feature_name = 'bert_classification'
			# feature_name = 'bert_regression'

			unique_grade_levels = pd.unique(df['grade_level'])
			unique_grade_levels = np.sort(unique_grade_levels)
			grade_encoder = dict(zip(unique_grade_levels, range(len(unique_grade_levels))))
			grade_decoder = {v:k for k,v in grade_encoder.items()}

			print("Training on {}".format(data_name))

			# slugs = pd.unique(df['slug'])
			

			kf = KFold(n_splits=5, shuffle=True)
			counter = 1
			for train_index, test_index in kf.split(df): 
				print('Fold Number {}'.format(counter))
				counter += 1

				# train_df = df[df['slug'].isin(slugs[train_index])][['filename', 'grade_level']]
				train_df = df.iloc[train_index][['filename', 'grade_level']]
				test_df = df.iloc[test_index][['filename', 'grade_level', 'Original_Index']]

				train_data = build_bert_train_data(train_df, data_folderpath, 'bert-base-uncased', grade_encoder)
				# train_data = build_bert_train_data(train_df, data_folderpath, 'bert-base-uncased')

				validation_data = train_data.take(16)
				train_data = train_data.skip(16)
				model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(grade_encoder.keys()))
				# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
				model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
				# model.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.MeanSquaredError())

				model.fit(train_data, validation_data=validation_data, epochs=5)
				# model.save_pretrained('newsela_english_bert_classification')
				# model.save_pretrained('newsela_english_bert_regression')


				test_df = df.iloc[test_index][['filename', 'grade_level', 'Original_Index']]
				test_data = build_bert_test_data(test_df, data_folderpath, 'bert-base-uncased')
				preds = model.predict(test_data)[0]
				preds = np.argmax(tf.nn.softmax(preds), axis=1)
				pred_labels = [grade_decoder[p] for p in preds]
				df.loc[test_index, feature_name] = pred_labels
				# df.loc[test_index, feature_name] = preds

			df.to_csv(data_filepath, index=False)

	with open(os.path.join(output_folderpath, 'ml_feature_names.pkl'), 'wb') as f:
		pickle.dump(ml_feature_names, f)
	
	return
if __name__ == '__main__':
	main(transfer=False)



