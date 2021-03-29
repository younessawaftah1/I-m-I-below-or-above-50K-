import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

np.random.seed(101)
tf.set_random_seed(101)

#Adding noise to the random linear data
x+=np.random.uniform(-4,4,50)
y+=np.random.uniform(-4,4,50)

#number of data points 
n=len(x)#50

#Plot training data 
plt.scatter(x,y)#function to create scatterplots, scatter plot is a type of plot that shows the data as a collection of points
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trainging data")
plt.show()

#To apply (y=wx+b)
X=tf.placeholder("float32")
Y=tf.placeholder("float32")
W=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

learning_rate=.01
epochs=1000

#y=wx+b
y_pred=tf.add(tf.multiply(W,X),b)

#cost function
cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

#Optimizer GDO
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    #initialize all vriables
    sess.run(tf.global_variables_initializer())
    #Iterating throw all epochs
    for epoch in range(epochs):
        #Feeding each data point into optimizer
        for(_x,_y) in zip(x,y):
            sess.run(optimizer,feed_dict={X:_x, Y:_y})
        #Displaying the result after every 50 epochs      
        if (epoch+1)%50 ==0:
            #Calculating the cost every epoch
            c=sess.run(cost,feed_dict={X:x, Y:y})
            print("Epoch: ",(epoch+1),"cost: ",c,"W=",sess.run(W),"b: ",sess.run(b))
     	# Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b)   

 # Calculating the predictions من الي خزنتهم فوق
predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

# Plotting the Results 
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show()

