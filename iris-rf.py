import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


dagshub.init(repo_owner='avi350751', repo_name='mlflow-dagshub', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/avi350751/mlflow-dagshub.mlflow")

#load the dataset
iris = load_iris()
X = iris.data
y = iris.target

#Split the dataset into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Define the parameters of RF model
max_depth = 10
n_estimators = 100


# Apply mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run(run_name = 'avi-exp2'):

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.set_tag('author','rick')
    mlflow.set_tag('model','random_forest')
   
    #Create a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True,fmt ='d',cmap='Blues',xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #Save the plot as an artifact
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    #Log the model
    mlflow.sklearn.log_model(rf,"random_forest_model")
    print('accuracy: ', accuracy)