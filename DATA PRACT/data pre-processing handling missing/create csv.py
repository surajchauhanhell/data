import pandas as pd

data = {
'Rollno':[1,2,3,4,5,6,7,8,9],
'Name':['Sudin','Shaima','Raina','Paul','Rahul','Gopal','Yatin','Jim','Nima'],
'Age':[44,46,27,38,46,None,59,36,45],
'Marks':[47,86,45,None,45,67,45,34,32],
'Class':['FY','SY','TY','SY','FY','TY','FY','FY','TY']
}

df = pd.DataFrame(data)

df.to_csv("student.csv",index=False)

print("CSV file created")
