import matplotlib.pyplot as plt
import numpy as np

lines = [
  (50, 50, 40, 35),
  (52, 52, 42, 37),
  (38, 30, 25, 15),
]
x=[]
y=[]
for x1,y1,x2,y2 in lines:
	x +=[x1,x2]
	y +=[y1, y2]
z = np.polyfit(x, y, 1)
f = np.poly1d(z)

xnew = np.linspace(min(x), max(x), 10).astype(int)
ynew= f(xnew).astype(int)

xy = list(zip(xnew,ynew))

for x, y in xy:
	plt.plot(x,y,'ro')
# for x1,y1,x2,y2 in lines:
# 	plt.plot((x1,x2),(x1,y2), 'g')
plt.axis([0,60, 0,60])
plt.show()