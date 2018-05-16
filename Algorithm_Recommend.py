from flask import Flask
from flask import request
from collections import namedtuple
import json
from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import linear_model
# from __future__ import print_function 
from sklearn.metrics.pairwise import cosine_similarity 
from scipy import sparse
import math





app = Flask(__name__)


url_db = ""
cols_genders = ""
num_gender = ""
user_test=""





@app.route("/")
def main():
	return "Welcome!"


#Content-Based
class content_Based:    

	def get_items_rated_by_user(self,rate_matrix, user_id):

		y = rate_matrix[:,0] # all users
		# item indices rated by user_id
		# we need to +1 to user_id since in the rate_matrix, id starts from 1 
		# while index in python starts from 0
		ids = np.where(y == user_id +1)[0] 
		item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
		scores = rate_matrix[ids, 2]
		return (item_ids, scores)
	def Run_algorithm(self,url_db,cols_genders,num_gender,user_test) :
		#Config data
		url_user = url_db + '/u.user' # link file users
		url_item = url_db + '/u.item' #link file item
		url_rate = url_db + '/ua.base' #link file rate
		#Load user
		cols_users =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
		users = pd.read_csv(url_user, sep='|', names=cols_users,encoding='latin-1')
		n_users = users.shape[0]
		#Load inf rate
		cols_rate = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
		ratings_base = pd.read_csv(url_rate, sep='\t', names=cols_rate, encoding='latin-1')
		rate_train = ratings_base.as_matrix()
		#Load inf items
		items = pd.read_csv(url_item, sep='|', names=cols_genders,encoding='latin-1')
		n_items = items.shape[0]
		#Test user_test
		if (user_test < 1) or (user_test > n_users) :
			return False
		# Convert items into matrix 
		X0 = items.as_matrix()
		X_train_counts = X0[:, -int(num_gender):]#only need 19 user reviews for items
		#tfidf
		transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
		tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
		d = tfidf.shape[1] # data dimension
		W = np.zeros((d, n_users))
		b = np.zeros((1, n_users))
		for n in range(n_users):    
			ids, scores = self.get_items_rated_by_user(rate_train, n)
			clf = Ridge(alpha=0.01, fit_intercept  = True)
			Xhat = tfidf[ids, :]    
			clf.fit(Xhat, scores) 
			W[:, n] = clf.coef_
			b[0, n] = clf.intercept_
		# predicted scores
		Yhat = tfidf.dot(W) + b
		x= Yhat
		recommend_item_3start = []
		recommend_item_4start = []
		recommend_item_5start = []
		five_start = 0;
		four_start = 0;
		three_start = 0;
		for i in range(n_items):
			if Yhat[i,int(user_test) -1] >= 5:
				recommend_item_5start.append(items.iloc[i]['movie title'])
				five_start += 1
			elif (Yhat[i,int(user_test) -1] >= 4 and Yhat[i,int(user_test) -1] < 5):
				four_start += 1
				recommend_item_4start.append(items.iloc[i]['movie title'])
			elif (Yhat[i,int(user_test) -1] >= 3) and (Yhat[i,int(user_test) -1] < 4):
				three_start += 1
				recommend_item_3start.append(items.iloc[i]['movie title'])

		return   ({"five_star" :five_start,"four_star":four_start,"three_star" :three_start,"recommend_item_5star": recommend_item_5start})




# Neighborhood-Based

class neighborhood_Based(object): 	
	# Neighborhood-Based
	def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
		self.uuCF = uuCF # user-user (1) or item-item (0) CF
		self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
		self.k = k # number of neighbor points
		self.dist_func = dist_func
		self.Ybar_data = None
		# number of users and items. Remember to add 1 since id starts from 0
		self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
		self.n_items = int(np.max(self.Y_data[:, 1])) + 1
	def add(self, new_data):
		"""
		Update Y_data matrix when new ratings come.
		For simplicity, suppose that there is no new user or item.
		"""
		self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
	def normalize_Y(self):
		users = self.Y_data[:, 0] # all users - first col of the Y_data
		self.Ybar_data = self.Y_data.copy()
		self.mu = np.zeros((self.n_users,))
		for n in range(self.n_users):
			# row indices of rating done by user n
			# since indices need to be integers, we need to convert
			ids = np.where(users == n)[0].astype(np.int32)
			# indices of all ratings associated with user n
			item_ids = self.Y_data[ids, 1] 
			# and the corresponding ratings 
			ratings = self.Y_data[ids, 2]
			# take mean
			m = np.mean(ratings)
			self.mu[n] = m
			if np.isnan(m):
				m = 0 # to avoid empty array and nan value
			# normalize
			self.Ybar_data[ids, 2] = ratings - self.mu[n]

		################################################
		# form the rating matrix as a sparse matrix. Sparsity is important 
		# for both memory and computing efficiency. For example, if #user = 1M, 
		# #item = 100k, then shape of the rating matrix would be (100k, 1M), 
		# you may not have enough memory to store this. Then, instead, we store 
		# nonzeros only, and, of course, their locations.
		self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
			(self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
		self.Ybar = self.Ybar.tocsr()

	def similarity(self):
		self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
	def refresh(self):
		"""
		Normalize data and calculate similarity matrix again (after
		some few ratings added)
		"""
		self.normalize_Y()
		self.similarity() 
		
	def fit(self):
		self.refresh()
	def __pred(self, u, i, normalized = 1):
		""" 
		predict the rating of user u for item i (normalized)
		if you need the un
		"""
		# Step 1: find all users who rated i
		ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
		# Step 2: 
		users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
		# Step 3: find similarity btw the current user and others 
		# who already rated i
		sim = self.S[u, users_rated_i]
		# Step 4: find the k most similarity users
		a = np.argsort(sim)[-self.k:] 
		# and the corresponding similarity levels
		nearest_s = sim[a]
		# How did each of 'near' users rated item i
		r = self.Ybar[i, users_rated_i[a]]
		
		return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
	
	
	def pred(self, u, i, normalized = 1):
		""" 
		predict the rating of user u for item i (normalized)
		if you need the un
		"""
		if self.uuCF: return self.__pred(u, i, normalize)
		return self.__pred(i, u, normalize)
	def recommend(self, items,user_test, normalized = 1):
		"""
		Determine all items should be recommended for user u. (uuCF =1)
		or all users who might have interest on item u (uuCF = 0)
		The decision is made based on all i such that:
		self.pred(u, i) > 0. Suppose we are considering items which 
		have not been rated by u yet. 
		"""
		recommend_item_3start = []
		recommend_item_4start = []
		recommend_item_5start = []
		five_start = 0;
		four_start = 0;
		three_start = 0;
		ids = np.where(self.Y_data[:, 0] == int(user_test)-1)[0]
		items_rated_by_u = self.Y_data[ids, 1].tolist()              
		for i in range(self.n_items):
			if i not in items_rated_by_u:
				rating = self.__pred(int(user_test)-1, i)
				if rating >= 4.5 :
					recommend_item_5start.append(items.iloc[i]['movie title'])
					five_start += 1
				elif (rating >= 3.5 and rating < 4.5):
					four_start += 1
					recommend_item_4start.append(items.iloc[i]['movie title'])
				elif (rating >= 2.5) and (rating < 3.5):
					three_start += 1
					recommend_item_3start.append(items.iloc[i]['movie title'])        
		

		return   ({"five_star" :five_start,"four_star":four_start,"three_star" :three_start,"recommend_item_5star": recommend_item_5start}) 











#Matrix Factorization
class MF(object):
	"""docstring for CF"""
	def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
				 learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 0):
		self.Y_raw = Y_data.copy()
		self.Y_data = Y_data.copy()
		self.K = K
		self.lam = lam
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.print_every = print_every
		self.user_based = user_based
		# number of users and items. Remember to add 1 since id starts from 0
		self.n_users = int(np.max(Y_data[:, 0])) + 1 
		self.n_items = int(np.max(Y_data[:, 1])) + 1
		
		if Xinit is None: 
			self.X = np.random.randn(self.n_items, K)
		else:
			self.X = Xinit 
		
		if Winit is None: 
			self.W = np.random.randn(K, self.n_users)
		else: 
			self.W = Winit
		
		# item biases
		self.b = np.random.randn(self.n_items)
		self.d = np.random.randn(self.n_users)
		#self.all_users = self.Y_data[:,0] # all users (may be duplicated)
		self.n_ratings = Y_data.shape[0]
		self.mu = 0
 

	def normalize_Y(self):
		if self.user_based:
			user_col = 0
			item_col = 1
			n_objects = self.n_users
		else:
			user_col = 1
			item_col = 0 
			n_objects = self.n_items

		users = self.Y_data[:, user_col] 
		self.muu = np.zeros((n_objects,))
		for n in range(n_objects):
			# row indices of rating done by user n
			# since indices need to be integers, we need to convert
			ids = np.where(users == n)[0].astype(np.int32)
			# indices of all ratings associated with user n
			item_ids = self.Y_data[ids, item_col] 
			# and the corresponding ratings 
			ratings = self.Y_data[ids, 2]
			# take mean
			m = np.mean(ratings) 
#             print m
			if np.isnan(m):
				m = 0 # to avoid empty array and nan value
			self.muu[n] = m
			# normalize
			self.Y_data[ids, 2] = ratings - m
			
			
	def loss(self):
		L = 0 
		for i in range(self.n_ratings):
			# user, item, rating
			n, m, rate = int(self.Y_data[i, 0]), int(self.Y_data[i, 1]), self.Y_data[i, 2]
			L += 0.5*(self.X[m, :].dot(self.W[:, n]) + self.b[m] + self.d[n] + self.mu - rate)**2
			
		# regularization, don't ever forget this 
		L /= self.n_ratings
		L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') + \
						  np.linalg.norm(self.b) + np.linalg.norm(self.d))
		return L 

	
	def get_items_rated_by_user(self, user_id):
		"""
		get all items which are rated by user n, and the corresponding ratings
		"""
		# y = self.Y_data_n[:,0] # all users (may be duplicated)
		# item indices rated by user_id
		# we need to +1 to user_id since in the rate_matrix, id starts from 1 
		# while index in python starts from 0
		ids = np.where(self.Y_data[:,0] == user_id)[0] 
		item_ids = self.Y_data[ids, 1].astype(np.int32) # index starts from 0 
		ratings = self.Y_data[ids, 2]
		return (item_ids, ratings)
		
		
	def get_users_who_rate_item(self, item_id):
		"""
		get all users who rated item m and get the corresponding ratings
		"""
		ids = np.where(self.Y_data[:,1] == item_id)[0] 
		user_ids = self.Y_data[ids, 0].astype(np.int32)
		ratings = self.Y_data[ids, 2]
		return (user_ids, ratings)
		
	def updateX(self):
		for m in range(self.n_items):
			user_ids, ratings = self.get_users_who_rate_item(m)
			
			Wm = self.W[:, user_ids]
			dm = self.d[user_ids]
			xm = self.X[m, :]
			
			error = xm.dot(Wm) + self.b[m] + dm + self.mu - ratings 
			
			grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
			grad_bm = np.sum(error)/self.n_ratings + self.lam*self.b[m]
			self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
			self.b[m]    -= self.learning_rate*grad_bm
	
	def updateW(self):
		for n in range(self.n_users):
			item_ids, ratings = self.get_items_rated_by_user(n)
			Xn = self.X[item_ids, :]
			bn = self.b[item_ids]
			wn = self.W[:, n]
			
			error = Xn.dot(wn) + bn + self.mu + self.d[n] - ratings
			grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
			grad_dn = np.sum(error)/self.n_ratings + self.lam*self.d[n]
			self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
			self.d[n]    -= self.learning_rate*grad_dn
	
	def fit(self):
		self.normalize_Y()
		for it in range(self.max_iter):
			self.updateX()
			self.updateW()

	
	
	def pred(self, u, i):
		""" 
		predict the rating of user u for item i 
		if you need the un
		"""
		u = int(u)
		i = int(i)
		if self.user_based == 1:
			bias = self.muu[u]
		else:
			bias = self.muu[i]
		
		pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u] + bias 
		return max(0, min(5, pred))
		
	
	def pred_for_user(self, user_test, items):
		recommend_item_3start = []
		recommend_item_4start = []
		recommend_item_5start = []
		five_start = 0;
		four_start = 0;
		three_start = 0;
		ids = np.where(self.Y_data[:, 0] == int(user_test)-1)[0]
		items_rated_by_u = self.Y_data[ids, 1].tolist()              
		for i in range(self.n_items):
			if i not in items_rated_by_u:
				rating = self.pred(int(user_test)-1,i)
				if rating >= 4.5:
					recommend_item_5start.append(items.iloc[i]['movie title'])
					five_start += 1
				elif (rating >= 3.5 and rating < 4.5):
					four_start += 1
					recommend_item_4start.append(items.iloc[i]['movie title'])
				elif (rating >= 2.5) and (rating < 3.5):
					three_start += 1
					recommend_item_3start.append(items.iloc[i]['movie title'])        
		

		return ({"five_star" :five_start,"four_star":four_start,"three_star" :three_start,"recommend_item_5star": recommend_item_5start})










# REST config data
@app.route('/admin/config', methods=['POST'])
def configDb():
	data = request.get_json()
	url_db = str(data["url_db"])
	cols_genders = str((data ["cols_genders"]))
	num_gender = str(data["num_gender"])
	user_test = str(data["user_test"])
	# print(cols_genders)
	info = str(url_db+"|"+cols_genders+"|"+num_gender+"|"+user_test)
	f = open("C:/Users/Public/Downloads/Save_config_db.txt", 'w') 
	f.write(info)
	f.close()	
	return jsonify("Scucessful !")











# REST Content-Based
@app.route('/algorithm/content-based', methods = ['POST'])
def Content_Based():
	col_data = ["url_db", "cols_genders","num_gender","user_test"]
	data = pd.read_csv("C:/Users/Public/Downloads/Save_config_db.txt",sep='|', names=col_data,encoding='latin-1' )
	url_db = data.iloc[0]['url_db']
	cols_genders = str(data.iloc[0]['cols_genders'])
	num_gender = data.iloc[0]['num_gender']
	user_test = data.iloc[0]['user_test']
	#config data
	b = cols_genders.split('\"')
	b.pop(-1)
	cols_genders = ["movie id", "movie title" ,"release date","video release date", "IMDb URL"]
	for n in range(len(b)):
		id = b[0].split(',')
		b[0] = id[0]
		if (b[n] != ',') and (b[n] != ', ') and (b[n] != ' ,'):
			cols_genders.append(b[n])
	# print(cols_genders)
	thuattoan = content_Based()
	a = thuattoan.Run_algorithm(url_db,cols_genders,num_gender,user_test)
	if a != False :
		return jsonify({"success" : True, "data" :a})
	else:
		return jsonify({"success" : False})











#REST User-Neighborhood-Based
@app.route('/algorithm/user-neighborhood-based', methods = ['POST'])
def User_Neighborhood_Based():
	col_data = ["url_db", "cols_genders","num_gender","user_test"]
	data = pd.read_csv("C:/Users/Public/Downloads/Save_config_db.txt",sep='|', names=col_data,encoding='latin-1' )
	url_db = data.iloc[0]['url_db']
	cols_genders = str(data.iloc[0]['cols_genders'])
	num_gender = data.iloc[0]['num_gender']
	user_test = data.iloc[0]['user_test']
	#config data
	b = cols_genders.split('\"')
	b.pop(-1)
	cols_genders = ["movie id", "movie title" ,"release date","video release date", "IMDb URL"]
	for n in range(len(b)):
		id = b[0].split(',')
		b[0] = id[0]
		if (b[n] != ',') and (b[n] != ', ') and (b[n] != ' ,'):
			cols_genders.append(b[n])
	# print(cols_genders)
	a = Run_algorithm1(url_db,cols_genders,user_test)
	if a != False :
		return jsonify({"success" : True, "data" :a})
	else:
		return jsonify({"success" : False})

def Run_algorithm1(url_db,cols_genders,user_test):
	#Load inf users
	cols_users =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
	users = pd.read_csv(url_db + '/u.user', sep='|', names=cols_users,encoding='latin-1')
	n_users = users.shape[0]
	#Load inf rate
	cols_rate = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings_base = pd.read_csv(url_db + '/ua.base', sep='\t', names=cols_rate, encoding='latin-1')
	rate_train = ratings_base.as_matrix()
	#Load inf items
	items = pd.read_csv(url_db + '/u.item', sep='|', names=cols_genders,encoding='latin-1')
	n_items = items.shape[0]
	#test user_test
	if (user_test < 1) or (user_test > n_users) :
			return False
	#run   
	rate_train = ratings_base.as_matrix()
	# indices start from 0
	rate_train[:, :2] -= 1
	rs = neighborhood_Based(rate_train, k = 30, uuCF = 1)
	rs.fit()
	return rs.recommend(items,user_test)








#REST Item_Neighborhood-Based

@app.route('/algorithm/item-neighborhood-based', methods = ['POST'])
def Item_Neighborhood_Based():
	col_data = ["url_db", "cols_genders","num_gender","user_test"]
	data = pd.read_csv("C:/Users/Public/Downloads/Save_config_db.txt",sep='|', names=col_data,encoding='latin-1' )
	url_db = data.iloc[0]['url_db']
	cols_genders = str(data.iloc[0]['cols_genders'])
	num_gender = data.iloc[0]['num_gender']
	user_test = data.iloc[0]['user_test']
	#config data
	b = cols_genders.split('\"')
	b.pop(-1)
	cols_genders = ["movie id", "movie title" ,"release date","video release date", "IMDb URL"]
	for n in range(len(b)):
		id = b[0].split(',')
		b[0] = id[0]
		if (b[n] != ',') and (b[n] != ', ') and (b[n] != ' ,'):
			cols_genders.append(b[n])
	# print(cols_genders)
	a = Run_algorithm2(url_db,cols_genders,user_test)
	if a != False :
		return jsonify({"success" : True, "data" :a})
	else:
		return jsonify({"success" : False})

def Run_algorithm2(url_db,cols_genders,user_test):
	#Load inf users
	cols_users =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
	users = pd.read_csv(url_db + '/u.user', sep='|', names=cols_users,encoding='latin-1')
	n_users = users.shape[0]
	#Load inf rate
	cols_rate = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings_base = pd.read_csv(url_db + '/ua.base', sep='\t', names=cols_rate, encoding='latin-1')
	rate_train = ratings_base.as_matrix()
	#Load inf items
	items = pd.read_csv(url_db + '/u.item', sep='|', names=cols_genders,encoding='latin-1')
	n_items = items.shape[0]
	#test user_test
	if (user_test < 1) or (user_test > n_users) :
			return False
	#run   
	rate_train = ratings_base.as_matrix()
	# indices start from 0
	rate_train[:, :2] -= 1
	rs = neighborhood_Based(rate_train, k = 30, uuCF = 0)
	rs.fit()
	return rs.recommend(items,user_test)











#REST Matrix-Factorization
@app.route('/algorithm/matrix-factorization', methods = ['POST'])
def Matrix_Factorization():
	col_data = ["url_db", "cols_genders","num_gender","user_test"]
	data = pd.read_csv("C:/Users/Public/Downloads/Save_config_db.txt",sep='|', names=col_data,encoding='latin-1' )
	url_db = data.iloc[0]['url_db']
	cols_genders = str(data.iloc[0]['cols_genders'])
	num_gender = data.iloc[0]['num_gender']
	user_test = data.iloc[0]['user_test']
	print(cols_genders)
	print(url_db)
	#config data
	b = cols_genders.split('\"')
	b.pop(-1)
	cols_genders = ["movie id", "movie title" ,"release date","video release date", "IMDb URL"]
	for n in range(len(b)):
		id = b[0].split(',')
		b[0] = id[0]
		if (b[n] != ',') and (b[n] != ', ') and (b[n] != ' ,'):
			cols_genders.append(b[n])
	# print(cols_genders)
	a = Run_algorithm3(url_db,cols_genders,user_test)
	if a != False :
		return jsonify({"success" : True, "data" :a})
	else:
		return jsonify({"success" : False})

def Run_algorithm3(url_db,cols_genders,user_test):
	#Load inf users
	cols_users =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
	users = pd.read_csv(url_db + '/u.user', sep='|', names=cols_users,encoding='latin-1')
	n_users = users.shape[0]
	#Load inf rate
	cols_rate = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings_base = pd.read_csv(url_db + '/ua.base', sep='\t', names=cols_rate, encoding='latin-1')
	rate_train = ratings_base.as_matrix()
	#Load inf items
	items = pd.read_csv(url_db + '/u.item', sep='|', names=cols_genders,encoding='latin-1')
	n_items = items.shape[0]
	#test user_test
	if (user_test < 1) or (user_test > n_users) :
			return False
	#run   
	# indices start from 0 
	rate_train[:, :2] -= 1 
	rs = MF(rate_train, K = 50, lam = .01, print_every = 5, learning_rate = 50, max_iter = 30)
	rs.fit() 
	rs.pred_for_user(user_test,items)
	return rs.pred_for_user(user_test,items)





if __name__ == "__main__":
	app.run()
