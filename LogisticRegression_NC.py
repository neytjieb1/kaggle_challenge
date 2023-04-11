import numpy as np
import numpy.linalg
from tqdm import tqdm

class LogisticRegression:
    
    def __init__(self,alpha,lamb,delta,niter):
        # self.K = K
        # self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.delta = delta
        self.iter = niter
       
    
    def fit(self, K, y):        
        def sigmoid(u):                                                        
            return 1 / (1 + np.exp(-u))
    
        def log_loss(u):                                                                                                      
            return np.log(1 + np.exp(-u))
        
        def diff_log_loss(u):
            return -sigmoid(-u)
        
        def sec_log_loss(u):
            return sigmoid(u)*sigmoid(-u)
        
        def log_likelihood(k, y, alpha,lamb):  
            n = len(k)                                                                                                     
            return np.sum(log_loss(y*(k.dot(alpha))))/n + 0.5*lamb*alpha.T.dot(k).dot(alpha)
        
        def gradient(k, y, alpha,lamb):
            P = diff_log_loss(y*(k.dot(alpha)))
            return k.dot(np.diag(P)).dot(y) + lamb*k.dot(alpha)                        

        def hessian(k, y, alpha,lamb):                                                          
            W = sec_log_loss(y*(k.dot(alpha)))                             
            return k.dot(np.diag(W)).dot(k) + lamb*k
        
        self.K = K
        self.y = y
                                                                                                                         
        Delta = np.Infinity                                                                
        l = log_likelihood(self.K, self.y, self.alpha, self.lamb)                                                                 
                                                                                                                                                                                    
        i = 0   
        for i in tqdm(range(self.iter)):                                                                        
        # while abs(Delta) > self.delta and i < self.iter:      
            if abs(Delta)>self.delta:
                i += 1                                                                      
                g = gradient(self.K, self.y, self.alpha,self.lamb)                                                      
                hess = hessian(self.K, self.y, self.alpha,self.lamb)                                                 
                H_inv = numpy.linalg.inv(hess)                                                 
                
                diff_alpha = np.dot(H_inv, g.T) 

                self.alpha = self.alpha + diff_alpha                                                               
                                        
                l_new = log_likelihood(self.K, self.y, self.alpha,self.lamb)                                                      
                Delta = l - l_new                                                           
                l = l_new
            else:
                break
                    

    def predictproba(self,K_test):
        
        def sigmoid(u):                                                        
            return 1 / (1 + np.exp(-u))
        
        prob1 = sigmoid(K_test.dot(self.alpha))
        prob2 = sigmoid(-K_test.dot(self.alpha))
        return  np.array([prob1,prob2])
        

from sklearn.metrics.pairwise import euclidean_distances
def RBF(X,Y,sigma=1.5):
    XX = np.exp(-euclidean_distances(X,Y, squared=True)/(2*sigma**2))
    return XX
