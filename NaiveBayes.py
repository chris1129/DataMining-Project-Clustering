import numpy as np
from scipy.stats import norm
import kfold
#print(norm.cdf(x, mean, std))

class NaiveBayes:		
	def __init__(self, data_set):
		self.data_set=data_set
		self.subdata_for_true=[]
		self.subdata_for_false=[]
		for i in range(len(self.data_set)):
			if int(self.data_set[i][-1])==1:
				self.subdata_for_true.append(self.data_set[i])
			else:
				self.subdata_for_false.append(self.data_set[i])
		self.Prior_Probability_true=float(len(self.subdata_for_true))/len(self.data_set)
		self.Prior_Probability_false=float(len(self.subdata_for_false))/len(self.data_set)

	def Descriptor_Posterior_Probability(self,Hi,X):#P(X|Hi)
		subdata=self.subdata_for_true if Hi==1 else self.subdata_for_false
		subdata=np.array(subdata)
		pdf_prob=[]
		for i in range(len(X)):

			if isinstance((X[i]),float):
				mean=subdata[:,i].astype(np.float).mean()
				stdv=subdata[:,i].astype(np.float).std()			
				pdf_prob.append(norm.pdf(float(X[i]),mean,stdv))
				
			else:
				
				count=1
				for j in range(len(subdata)):
					if subdata[j][i]==X[i]:
						count+=1
				pdf_prob.append(1.0*count/len(subdata))
				
				
		ret=1
		#print("prob",pdf_prob)
		for i in range(len(pdf_prob)):
			ret*=pdf_prob[i]
		return ret

	def predict(self,X):
		Prob_true=self.Prior_Probability_true*self.Descriptor_Posterior_Probability(1,X)
		Prob_false=self.Prior_Probability_false*self.Descriptor_Posterior_Probability(0,X)
		return '1' if Prob_true>Prob_false else '0'
			









