
# coding: utf-8

# In[29]:


import tensorflow as tf
import pandas as pd


# In[30]:


iris_data = pd.read_csv('./Iris.csv')


# In[31]:


iris_data.head()


# In[32]:


# Turn Species into numbers
iris_data_one_encoded = pd.get_dummies(iris_data)
iris_data_one_encoded.head()


# In[33]:


# Get sample data, frac == percentage of sample
iris_test_data = iris_data_one_encoded.sample(frac=0.2, random_state=123)
print (iris_test_data.index)

# Get train data by dropping everything not in sample data using .drop()
iris_train_data = iris_data_one_encoded.drop(iris_test_data.index)
print (iris_train_data.index)


# In[34]:


# Get columns of training data
input_train = iris_train_data.filter([
    'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
])
input_train.head()


# In[35]:


# Get columns of test data
input_test = iris_test_data.filter([
    'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
])
input_test.head()


# In[36]:


# get the label of train
label_train = iris_train_data.filter([
    'Species_Iris-setosa',
    "Species_Iris-versicolor",
    "Species_Iris-virginica"
])

label_train.head()


# In[37]:


# get the label of test
label_test = iris_test_data.filter([
    'Species_Iris-setosa',
    "Species_Iris-versicolor",
    "Species_Iris-virginica"
])

label_test.head()


# In[43]:


# y = mx + b

# placeholder, given by user
# [0,0,0,0]
x = tf.placeholder(tf.float32, [None, 4])

# variable
# [0,0,0,0]
m = tf.Variable(tf.zeros([4, 3]))

# [
# [0]
# [0]
# [0]
# ]
b = tf.Variable(tf.zeros([3]))

# Turn line of best fit --> sigmoid function
y = tf.nn.softmax(tf.matmul(x, m) + b)

actual = tf.placeholder(tf.float32, [None, 3])


# In[49]:


# Cross Entropy, lf == loss_function
lf = tf.reduce_mean(-tf.reduce_sum(
    actual * tf.log(y), reduction_indices=[1]
))


# In[51]:


# Define train step (one attempt of guessing)
lr = 0.05 # Learning Rate
train_step = tf.train.GradientDescentOptimizer(lr).minimize(lf)


# In[52]:


sess = tf.InteractiveSession()


# In[55]:


tf.global_variables_initializer().run()
for step in range(100):
    sess.run(train_step, feed_dict={
        x: input_train, 
        actual: label_train
    })
    
print("Learned:")
print(sess.run(m))
print(sess.run(b))

