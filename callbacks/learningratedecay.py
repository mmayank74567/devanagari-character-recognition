
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

class LearningRateDecay:
  
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]

		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")
  
	def print_lr(self, epochs):
		lrs = [self(i) for i in epochs]
		x = PrettyTable()
		column_names = ['Epoch#', 'Learning Rate']
		x.add_column(column_names[0], epochs)
		x.add_column(column_names[1], lrs)
		print(x)
  
class PolynomialDecay(LearningRateDecay):
	def __init__(self, baseLr, maxEpochs=75, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.power = power
		self.baseLr = baseLr

	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		alpha = (1 - (epoch / float(self.maxEpochs))) ** self.power
		

		# return the new learning rate
		return float(alpha)