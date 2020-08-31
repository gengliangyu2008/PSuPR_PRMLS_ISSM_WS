import matplotlib.pyplot as plt

# plt.plot([1, 2, 3, 4, 5, 100])

# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed

plt.subplot(211)
plt.plot(range(12))
plt.show()
'''
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background
'''
