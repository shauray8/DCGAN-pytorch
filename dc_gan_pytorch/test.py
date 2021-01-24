import pickle
import matplotlib.pyplot as plt

open_file = open('images.pkl', "rb")
li = pickle.load(open_file)
plt.imshow(li[-1][0])
plt.show()