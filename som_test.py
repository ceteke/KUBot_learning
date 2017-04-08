# load the digits dataset from scikit-learn
# 901 samples, about 180 samples per class
# the digits represented 0,1,2,3,4
from sklearn import datasets
digits = datasets.load_digits(n_class=4)
data = digits.data # matrix where each row is a vector that represent a digit.
num = digits.target # num[i] is the digit represented by data[i]

# training the som
from models import SOM
som = SOM(20,20,64,0.5,0.8)
print("Training")
i = 0
for d in data:
    i += 1
    som.update(d)
    if i > 1000:
        break
print("\n...ready!")

# plotting the results
from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
wmap = {}
figure(1)
im = 0
for x,t in zip(data,num): # scatterplot
	w = som.winner(x)
	wmap[w] = im
	text(w[0]+.5, w[1]+.5, str(t), color=cm.Dark2(t / 4.), fontdict={'weight': 'bold', 'size': 11})
	im = im + 1
axis([0,som.y,0,som.x])

figure(2,facecolor='white')
cnt = 0
for j in reversed(range(20)): # images mosaic
	for i in range(20):
		subplot(20,20,cnt+1,frameon=False, xticks=[], yticks=[])
		if (i,j) in wmap:
			imshow(digits.images[wmap[(i,j)]], cmap='Greys',  interpolation='nearest')
		else:
			imshow(zeros((8,8)), cmap='Greys')
		cnt = cnt + 1

show() # show the figure
