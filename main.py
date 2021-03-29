import tensorflow as tf
import pandas as pd
#Read the data 
census = pd.read_csv("adult-all.csv",skiprows=1, header=None)

census.columns = ['age','workclass' , 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'gender', 'capital_gain',
        'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

census.head()
#Values of  income_bracket column
census['income_bracket'].unique()

#To convert to 0, 1
def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1
#Apply
census['income_bracket'] = census['income_bracket'].apply(label_fix)
#Show after Apply
census['income_bracket'].unique()

#Training and testing
from sklearn.model_selection import train_test_split
x_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,
                                    test_size=0.3,random_state=101)        

#Feature engineering
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#Collect the features 
feat_cols = [gender,occupation,marital_status,relationship,
education,workclass,native_country, 
age,education_num,capital_gain,
capital_loss,hours_per_week]

#Input the features 
input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,
                                        y=y_train,batch_size=1000,
                                        num_epochs=20,shuffle=True)

#Create the model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

model.train(input_fn=input_func,steps = 1000)

#Add the inputs into the model
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

#Calculate the accuracy 
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))

#Plot the Curve ROC AUC Score
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,final_preds)
roc_auc = auc(fpr, tpr)
print("ROC AUC Score: {}".format(roc_auc))
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
 