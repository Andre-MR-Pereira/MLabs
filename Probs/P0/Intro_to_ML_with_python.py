#     important libraries

#   Numpy – library for mathematics and algebra 
#   Scipy – scientific computing (advanced linear algebra routines, mathematical function
#   optimization, signal processing, special mathematical functions and statistical distributions)
#   Matplotlib (maplotlib.pyplot) – library for plotting data (different kinds of plot like histograms, pie, scatter) , showing images
#   Pandas – library to deal with tables (Dataframe Structure), similar to an Excel Spreadsheet
#   Tensorflow (machine learning library)
#   Keras  - high level
#   Scikit-learn  - library with machine learning models (also some datasets)



# Load in NumPy (remember to pip install numpy first)
import numpy as np

# Create vector
a = np.array([1,2,3], dtype='int32')
print(a)
# Create matrix
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)


# Get Shape
b.shape
# Get Type
a.dtype
# Get number of elements
a.size


# Accessing/Changing specific elements, rows, columns, etc
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)

# Get a specific element [r, c] - note that indexes start at 0
a[1, 5]
# Get a specific row 
a[0, :]
# Get a specific column
a[:, 2]
# subset of the elements  [startindex:endindex:stepsize]
a[0, 1:-1:2]
a[1,5] = 20
a[:,2] = [1,2]
print(a)


# Initializing Different Types of Arrays

# All 0s matrix
np.zeros((2,3))
# All 1s matrix
np.ones((4,2,2), dtype='int32')
# Any other number
np.full((2,2), 99)
# Any other number (full_like)
np.full_like(a, 4)
# Random decimal numbers 
np.random.rand(4,2)
# Random Integer values
np.random.randint(-4,8, size=(3,3))
# The identity matrix
np.identity(5)


# Mathematics

a = np.array([1,2,3,4])
print(a)
a + 2
a - 2
a * 2
a / 2
b = np.array([1,0,1,0])
a + b
a ** 2
np.cos(a)


# Linear Algebra

a = np.ones((2,3))
print(a)

b = np.full((3,2), 2)
print(b)

#matrix multiplication
np.matmul(a,b)

# Find the determinant
c = np.identity(3)
np.linalg.det(c)


# Statistics

a = np.array([[1,2,3],[4,5,6]])

np.min(a)
np.max(a, axis=1)


# Reorganizing Arrays

before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((2,4))
print(after)

# Save and Load Data from File

np.save('a.npy',a)
b=np.load('a.npy')
b

# Plotting
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5,6], dtype='int32')
b = np.ones((6), dtype='int32')
plt.plot(a,label='vector a')
plt.plot(b,label='vector b')
plt.title('my figure')

