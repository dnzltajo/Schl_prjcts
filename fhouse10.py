import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats

plt.style.use('seaborn-poster')
x12=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
x123=np.arange(1,37)
x1=np.arange(1,13)
x2=np.arange(13,25)
x3=np.arange(25,37)
x4=np.arange(37,43)

y12=np.array([76,74,77,51,68,75,62,84,104,62,70,78, 52,82,72,60,82,66,68,131,107,130,55,76])
y1=np.array([76,74,77,51,68,75,62,84,104,62,70,78])
y2=np.array([52,82,72,60,82,66,68,131,107,130,55,76])
y3=np.array([80,82,87,68,88,69,85,24,63,75,71,82])
y4=np.array([87,83,119,95,70,39])
y123=np.array([76,74,77,51,68,75,62,84,104,62,70,78,52,82,72,60,82,66,68,131,107,130,55,76,80,82,87,68,88,69,85,24,63,75,71,82])

z = np.polyfit(x1,y1, 1) 
p = np.poly1d(z)

z2 = np.polyfit(x12,y12, 1) 
p2 = np.poly1d(z2)

z3 = np.polyfit(x123,y123, 1) 
p3 = np.poly1d(z3)

plt.subplot(3,2,1)
plt.plot(x1,y1,'o',label='Original Data')
plt.plot(x1,p(x1), 'k', label='Degree 3')
plt.title('Household 10',loc='center')
plt.subplot(3,2,2)
plt.plot(x2,y2,'o',label='Test Data')
plt.plot(x2,p(x2), '-ok', label='Predicted')
plt.title('Predicted Values',loc='center')
plt.legend(loc='upper right')
plt.subplot(3,2,3)
plt.plot(x12,y12,'o',label='Original Data')
plt.plot(x12,p2(x12), 'k', label='Degree 3')
plt.ylabel('kWh')
plt.subplot(3,2,4)
plt.plot(x3,y3,'o',label='Test Data')
plt.plot(x3,p2(x3), '-ok', label='Predicted')
plt.legend(loc='upper right')
plt.subplot(3,2,5)
plt.plot(x123,y123,'o',label='Original Data')
plt.plot(x123,p3(x123), 'k', label='Degree 3')
plt.xlabel('Month')
plt.subplot(3,2,6)
plt.plot(x4,y4,'o',label='Test Data')
plt.plot(x4,p3(x4), '-ok', label='Predicted')
plt.legend(loc='upper right')
plt.xlabel('Month')
plt.show()

mse =mean_squared_error(y2,p(x2))
print('MSE 1:%.16f'%mse)
mse2 =mean_squared_error(y3,p2(x3))
print('MSE 12:%.16f'%mse2)
mse3 =mean_squared_error(y4,p3(x4))
print('MSE 123:%.16f'%mse3)
avg=(mse+mse2+mse3)/3
print('Avg MSE:%.16f'%avg)

slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y1)
print("r-squared1: %f" % r_value**2)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x12, y12)
print("r-squared12: %f" % r_value2**2)
slope, intercept, r_value3, p_value, std_err = stats.linregress(x123, y123)
print("r-squared123: %f" % r_value3**2)