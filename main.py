import pickle
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('C:/Users/SARVJEET/MY Project/student-por.csv')
data=df[["studytime","freetime","G2","G1","absences","health","G3"]]

X=data.iloc[:,0:6].values
y=data.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model= LinearRegression()
model=model.fit(X_train, Y_train)
pickle.dump(model,open('model.pkl','wb'))
print("Model R2 Score on test data: ")
print(r2_score(Y_test,model.predict(X_test)))

print("Model Mean Square Error on test data: ")
print(mean_squared_error(Y_test,model.predict(X_test)))
model=pickle.load(open('model.pkl','rb'))
