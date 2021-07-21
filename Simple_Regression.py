# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:32:12 2020

@author: julia
"""

# robust standard errors

######### statsmodels
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# model
np.random.seed(0)
b0 = 10
b1 = 2
b2 = 5
x1longunsorted = np.random.uniform(0,100,50)
x1long = np.sort(x1longunsorted)
x1 = np.around(x1long,1)
x2long = np.random.uniform(0,100,50)
x2 = np.around(x2long,1)
elong  = np.random.normal(loc=0,scale=10,size=50)
e  = np.around(elong,1)
ylong  = b0 + np.dot(b1,x1long) + np.dot(b2,x2long) + elong
y  = np.around(ylong,1)

# dataframe
#np_data = np.array([y,x1,x2,e])
#np_data = np.transpose(np_data)
#df = pd.DataFrame({"y":y},{"x1":x1},{"x2":x2},{"e":e})
#df = pd.DataFrame(np_data,columns = ["y","x1","x2","e"])
np_data_dict = {"y":y,"x1":x1,"x2":x2,"e":e}
df = pd.DataFrame(np_data_dict)
df.index = df.index + 1
print(df)
X = df[["x1","x2"]].to_numpy()
X = sm.add_constant(X)
#print(X)

# regression
model = sm.OLS(y,X)                    # hasconst?
model = model.fit()
b0_head, b1_head, b2_head = np.around(model.params,2)
results = model.summary()
print(results, b0_head, b1_head, b2_head)

#diagram_3d_contour
def f(x1, x2):
   return b0_head + np.dot(x1,b1_head) + np.dot(x2,b2_head)
X1, X2 = np.meshgrid(x1, x2)
y_pred = f(X1,X2)

title_diag = ("y = " + str(b0) + ".00" + " + " + str(b1) + ".00" + 
              r"$\cdot x_1$" + " + " + str(b2) + ".00" + r"$\cdot x_2$" + 
              " + " + r"$\epsilon$" + "\n" +
               r"$\hat{y}$ = " + str(b0_head) + ' + ' + str(b1_head) +
               r"$\cdot x_1$" + " + " + str(b2_head) + r"$\cdot x_2$")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(x1, x2, y,"ro")                       # optional for data points
ax.contour3D(X1, X2, y_pred, 50, alpha = 0.1) # 50 lines # alpha for odacity
ax.invert_xaxis()
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel('y')
ax.set_title(title_diag)
plt.show()
