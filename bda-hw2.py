"""
__desc__ = Big Data Analytics homework 2
__author__ = "Trisha P Malhotra (tpm6421)"
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("places_rated_new.data",
                 names=['Climate', 'Housing', 'HlthCare', 'Crime', 'Transp', 'Educ', 'Arts', 'Recreat', 'Econ'])
# Original size :  330 * 9
oldX = df.as_matrix()
X = oldX[1:]
# New size : Matrix X : 329 * 9


from sklearn.preprocessing import StandardScaler

# Normalizing X's feature columns to zero mean and unit standard deviation
X = StandardScaler().fit_transform(X)
# print(X)
# print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
principalComponents = pca.fit_transform(X)
# 3 PC
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3',
                                    'principal component 4', 'principal component 5', 'principal component 6',
                                    'pc 7', 'pc 8', 'pc9'])
# print(principalDf[['PC 1', 'PC 2', 'PC 3']])
# print(principalDf)

# Explained_Variance_Ratio
# print(pca.components_)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_.sum()
# SUM of explained_Variance_ratio for 6 PCs: 0.8745,
# It implies 87% of the data variance is included by 6 principal components.


cumulative_Ex_var_ratio = pca.explained_variance_ratio_.cumsum()
cumulative_Ex_var_ratio_perc = cumulative_Ex_var_ratio[-1] * 100
print("Cumulative explained variance ratio :")
print(cumulative_Ex_var_ratio)
print("-------------------------------------------")
print("With " + str("default of 9") + " Principal components : Total Variance " + str(
    cumulative_Ex_var_ratio_perc) + " % ")
print("Thus, with minimum of 6 Principal Components :  Total Variance : 87.45 %, i.e more than 80 %")

print("---------------------------------------------")
print("Plotting Explained_variance_ratio V/S Number of Principal Components:")
# explained variance ratio as a function of the number of components:
pca2 = PCA().fit(X)
plt.figure(1)
plt.semilogy(pca2.explained_variance_ratio_.cumsum(), '--o')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal components')
plt.show()

# Loading vectors:
print("---------------------------------------------")
print("Loading Vectors:")
loading_vectors = pca.components_
print(loading_vectors)

print("----------------------------------------------")
print("Next, plotting Bi-plots: (PCA1-PCA2), (PCA1-PCA3), (PCA2-PCA3)")


# Biplots:

x_new = pca.fit_transform(X)

import numpy as np
def biplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')



# 1. PCA1 - PCA 2
plt.figure(2)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()


biplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()



# 2. PCA 1 - PCA 3
plt.figure(3)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(3))
plt.grid()


biplot(x_new[:,[0,2]],np.transpose(pca.components_[[0,2], :]))
plt.show()


# 3. PCA 2 - PCA 3
plt.figure(4)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(2))
plt.ylabel("PC{}".format(3))
plt.grid()


biplot(x_new[:,2:4],np.transpose(pca.components_[2:4, :]))
plt.show()



