from keras_han.model import HAN
from keras_han.layers import AttentionLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from sklearn.svm import SVC
from tqdm.keras import TqdmCallback
from transformers import TFBertForSequenceClassification

import numpy as np
import tensorflow as tf

class HAN_SVM():
	"""
	Creating a model class that combines the HAN architecture along with an SVM.  The training procedure will entail
	trainig the HAN on a dataset, predicting on the same  dataset, then feeding the predictions to train the SVM.
	This class aims to replicate the process done in "Linguistic-Features-for-Readability" by Tovley Deutsch
	"""

	def __init__(self):
		"""
		"""
		return

	def initialize_params(self, max_words_per_sent, max_sentences, output_size, embedding_matrix,loss, word_encoding_dim=100, sentence_encoding_dim=100, with_svm=False):
		self.max_words_per_sent = max_words_per_sent
		self.max_sentences = max_sentences
		self.output_size = output_size
		self.loss = loss
		self.embedding_matrix = embedding_matrix
		self.word_encoding_dim = word_encoding_dim
		self.sentence_encoding_dim = sentence_encoding_dim
		self.han = HAN(max_words_per_sent, max_sentences, output_size, embedding_matrix,\
			word_encoding_dim, sentence_encoding_dim)
		self.han.compile(optimizer=(optimizers.Adam(learning_rate=0.0001)), loss=loss, metrics=['acc'])
		self.loss = loss
		self.with_svm = None
		if with_svm:
			self.with_svm = with_svm
			self.svm = SVC()
	
		return

	def re_initialize_params(self):
		self.han = HAN(self.max_words_per_sent, self.max_sentences, self.output_size, self.embedding_matrix,\
			self.word_encoding_dim, self.sentence_encoding_dim)
		self.han.compile(optimizer=(optimizers.Adam(learning_rate=0.00001)), loss=self.loss, metrics=['acc'])

	def fit(self, X, Y, batch_size=64, epochs=30):
		"""
		Takes as input word embeddings X and labels Y (reading levels in the readability case).
		Trains a HAN model then an SVM model, using the HAN predictions on the training data as training data
		"""
		if self.loss == 'mean_squared_error':
			Y = np.argmax(Y, axis=1)

		self.han.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[TqdmCallback(verbose=2)], verbose=0)

		if self.with_svm:
			if len(Y.shape) > 1:
				Y = np.argmax(Y, axis=1)
			svm_features = self.han.predict(X)
			self.svm.fit(svm_features, Y)
		return

	def predict(self, X):
		if self.with_svm:
			svm_features = self.han.predict(X)
			preds = self.svm.predict(svm_features)
		else:
			preds = self.han.predict(X)
			preds = np.argmax(preds, axis=1)
		return preds

class BERT_SVM():
	def __init__(self):
		self.svm = SVC()
		return

	def initialize_params(self, num_labels):
		self.bert = TFBertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
		return

	def fit(self, X, Y, batch_size=8, epochs=50):
		train_data = tf.data.Dataset.from_tensor_slices((X, Y))
		train_data = train_data.shuffle(len(train_data)).batch(8)
		early_stopping = EarlyStopping(monitor='loss', patience=3)
		self.bert.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
		self.bert.fit(train_data ,epochs=epochs, callbacks=[TqdmCallback(verbose=2), early_stopping], verbose=0)
		svm_features = self.bert.predict(X)
		Y = np.argmax(Y, axis=1)
		self.svm.fit(svm_features, Y)
		return












  