from scipy.spatial.distance import mahalanobis
import sys
import numpy as np
sys.path.append('../')
import deb
import colorama
colorama.init()

x=np.array([1,2,3,4,5,6])
y=np.array([2,1,2,1,2,1])

cov_diag=np.array([5,6,7,8,9,1])

#cov_matrix = np.ones((6, 6))
cov_matrix = np.diag(cov_diag)
#cov_matrix[0,1] = 0
deb.prints(cov_matrix.shape)
deb.prints(cov_matrix)

def mahalanobis_distance2(x, y, covariance_matrix): 
    # covariance_matrix shape: (16, 16)
    
    x = np.expand_dims(x, axis=0) # shape: (1, 16)
    y = np.expand_dims(y, axis=0) # shape: (1, 16)
    
#        deb.prints(feature.shape)
#        deb.prints(covariance_matrix.shape)
#        deb.prints(np.transpose(feature).shape)
    out = np.dot(x, np.linalg.inv(covariance_matrix)) # shape: (1, 16)
#        out1 = np.matmul(feature, np.linalg.inv(covariance_matrix))
#        deb.prints(out.shape)
    out = np.squeeze(np.dot(out, np.transpose(y))) # shape: (1)
#        out1 = np.squeeze(np.matmul(out1, np.matrix.transpose(feature))) # shape: (1)
#        deb.prints(out.shape)
#        deb.prints(out)
#        deb.prints(out1)

    return np.sqrt(out)
def mahalanobis_distance3(x, y, covariance_matrix): 
    # covariance_matrix shape: (16, 16)
    
#    feature = np.expand_dims(feature, axis=0) # shape: (1, 16)
#        deb.prints(feature.shape)
#        deb.prints(covariance_matrix.shape)
#        deb.prints(np.transpose(feature).shape)
    out = np.dot(x, np.linalg.inv(covariance_matrix)) # shape: (1, 16)
#        out1 = np.matmul(feature, np.linalg.inv(covariance_matrix))
#        deb.prints(out.shape)
#    out = np.squeeze(np.dot(out, np.transpose(y))) # shape: (1)
    out = np.squeeze(np.dot(out, np.transpose(y)) # shape: (1)

#        out1 = np.squeeze(np.matmul(out1, np.matrix.transpose(feature))) # shape: (1)
#        deb.prints(out.shape)
#        deb.prints(out)
#        deb.prints(out1)

    return np.sqrt(out)

dist = mahalanobis(x, y, cov_matrix)

deb.prints(dist)

dist1 = mahalanobis_distance2(x, y, cov_matrix)

deb.prints(dist1)

dist2 = mahalanobis_distance3(x, y, cov_matrix)

deb.prints(dist2)