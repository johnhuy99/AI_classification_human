
import numpy as np 
# import warnings
# warnings.filterwarnings('ignore') # suppress warnings
# import matplotlib.pyplot as plt 


class NeuralNet:
	def __init__(self, layers=[4096,50,40,1], learning_rate=0.001, iterations=1000):
		self.params={}
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.loss = []
		self.sample_size = None
		self.layers = layers
		self.X = None
		self.y = None


	def init_weights(self):

		'''
		Initialize weights from a random normal distribution
		'''
		np.random.seed(1)
		self.params['w1'] = np.random.randn(self.layers[0], self.layers[1])
		self.params['b1'] = np.random.randn(self.layers[1],)
		self.params['w2'] = np.random.randn(self.layers[1],self.layers[2])
		self.params['b2'] = np.random.randn(self.layers[2],)
		self.params['w3'] = np.random.randn(self.layers[2],self.layers[3])
		self.params['b3'] = np.random.randn(self.layers[3])

	def relu(self,Z):
		return np.maximum(0,Z)

	def sigmoid(self,Z):
		return 1./(1.+np.exp(-Z))

	def entropy_loss(self,y,yhat):
		nsample = len(y)
		loss = -1/nsample*(np.sum(np.multiply(np.log(yhat),y)+np.multiply((1-y),np.log(1-yhat))))
		return loss

	def forward_prop(self):
		z1 = self.X.dot(self.params['w1']) +self.params['b1']
		a1 = self.relu(z1)
		z2 = a1.dot(self.params['w2']) + self.params['b2']
		a2 = self.sigmoid(z2)
		z3 = a2.dot(self.params['w3']) + self.params['b3']
		yhat = self.sigmoid(z3)
		loss = self.entropy_loss(self.y,yhat)

		# save calculated parameters
		self.params['z1'] = z1
		self.params['z2'] = z2
		self.params['z3'] = z3
		self.params['a1'] = a1
		self.params['a2'] = a2

		return yhat,loss

	def back_prop(self,yhat):
		def dRelu(x):
			x[x<=0] = 0
			x[x>0] = 1
			return x

		dl_wrt_yhat = -(np.divide(self.y,yhat)-np.divide((1-self.y),(1-yhat)))
		dl_wrt_sig = yhat * (1-yhat)
		dl_wrt_z3 = dl_wrt_yhat * dl_wrt_sig

		dl_wrt_a2 = dl_wrt_z3.dot(self.params['w3'].T)
		dl_wrt_w3 = self.params['a2'].T.dot(dl_wrt_z3)
		dl_wrt_b3 = np.sum(dl_wrt_z3,axis=0)

		dl_wrt_z2 = dl_wrt_a2 * (self.params['a2']*(1-self.params['a2']))
		dl_wrt_a1 = dl_wrt_z2.dot(self.params['w2'].T)
		dl_wrt_w2 = self.params['a1'].T.dot(dl_wrt_z2)
		dl_wrt_b2 = np.sum(dl_wrt_z2,axis=0)

		dl_wrt_z1 = dl_wrt_a1 * dRelu(self.params['z1'])
		dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
		dl_wrt_b1 = np.sum(dl_wrt_z1,axis=0) 


		#update the weights and biases
		self.params['w1'] = self.params['w1'] - self.learning_rate * dl_wrt_w1
		self.params['w2'] = self.params['w2'] - self.learning_rate * dl_wrt_w2
		self.params['w3'] = self.params['w3'] - self.learning_rate * dl_wrt_w3
		self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
		self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
		self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3


	def fit(self,X,y):
		'''
		trains the neural network using the specified data and labels
		'''
		self.X = X
		self.y = y
		self.init_weights()

		for i in range(self.iterations):
			yhat, loss = self.forward_prop()
			self.back_prop(yhat)
			self.loss.append(loss)

	def predict(self,X):
		'''
		PREDICTS on a test data set
		'''
		z1 = X.dot(self.params['w1']) + self.params['b1']
		a1 = self.relu(z1)
		z2 = a1.dot(self.params['w2']) + self.params['b2']
		a2 = self.sigmoid(z2)
		z3 = a2.dot(self.params['w3']) + self.params['b3']
		pred = self.sigmoid(z3)
		return np.round(pred)

	def acc(self,y,yhat):
		'''
		calculates the accutacy between the predict valus and the truth labels
		'''
		acc = int(sum(y==yhat))/len(y) *100
		return acc

	def plot_loss(self):
		'''plots the loss curve'''
		plt.plot(self.loss)
		plt.xlabel('iterations')
		plt.ylabel('log loss')
		plt.title('loss curve for training')
		plt.show()


from demo import get_Xtrain_ytrain
from demo import get_Xtest_ytest
nn = NeuralNet()
Xtrain,ytrain = get_Xtrain_ytrain()
Xtest,ytest = get_Xtest_ytest()
nn.fit(Xtrain,ytrain)
predict = nn.predict(Xtest)
print(nn.acc(predict,ytest))
