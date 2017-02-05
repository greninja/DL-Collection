class rnn:
	def step(self,x):
		self.h = np.tanh(np.dot(self.W_hh,self.h) + np.dot(self.W_xh,self.x))
		y  = np.dot(self.W_hy,self.h)
		return y

		