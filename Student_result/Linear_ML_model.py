import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 🔹 Create Data
student_data = {
    "study_hours": [
        1,2,3,4,5,6,7,8,2.5,3.5,4.5,5.5,6.5,7.5,8.5,
        1.2,2.3,3.7,4.8,5.1,6.2,7.4,8.6,2.8,3.9,
        4.2,5.6,6.7,7.8,8.9,1.5,2.6,3.8,4.9,5.3,
        6.4,7.6,8.7,2.1,3.4,4.6,5.8,6.9,7.2,8.3,
        1.8,2.9,3.1,4.4,5.7
    ],
    "study_method": [
        "self","self","coaching","coaching","self",
        "coaching","self","coaching","self","coaching",
        "self","coaching","self","coaching","self",
        "self","coaching","self","coaching","self",
        "coaching","self","coaching","self","coaching",
        "self","coaching","self","coaching","self",
        "self","coaching","self","coaching","self",
        "coaching","self","coaching","self","coaching",
        "self","coaching","self","coaching","self",
        "self","coaching","self","coaching","self"
    ],
    "internet_access": [
        "yes","no","yes","yes","no",
        "yes","no","yes","yes","no",
        "yes","yes","no","yes","yes",
        "yes","no","yes","yes","no",
        "yes","no","yes","yes","no",
        "yes","yes","no","yes","yes",
        "no","yes","no","yes","yes",
        "yes","no","yes","yes","no",
        "yes","yes","no","yes","yes",
        "no","yes","yes","no","yes"
    ],
    "gender": [
        "male","female","male","female","male",
        "female","male","female","male","female",
        "male","female","male","female","male",
        "female","male","female","male","female",
        "male","female","male","female","male",
        "female","male","female","male","female",
        "male","female","male","female","male",
        "female","male","female","male","female",
        "male","female","male","female","male",
        "female","male","female","male","female"
    ],
    "exam_score": [
        35,40,45,50,55,60,65,70,42,48,52,58,63,68,72,
        36,43,49,54,56,62,67,73,44,50,
        51,59,64,69,74,38,46,48,53,57,
        61,66,71,41,47,52,60,65,68,72,
        39,45,47,50,58
    ]
}

df = pd.DataFrame(student_data)

# 🔹 Separate Label Encoders
le_internet = LabelEncoder()
le_gender = LabelEncoder()

df["internet_access"] = le_internet.fit_transform(df["internet_access"])
df["gender"] = le_gender.fit_transform(df["gender"])

# 🔹 One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[["study_method"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(["study_method"])
)

df = pd.concat([df, encoded_df], axis=1)
df.drop("study_method", axis=1, inplace=True)

# 🔹 Features & Target
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 🔹 Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 🔹 USER INPUT
user_study_hours = float(input("Enter study hours: "))
user_study_method = input("Enter study method (self/coaching): ").lower()
user_internet_access = input("Internet access (yes/no): ").lower()
user_gender = input("Gender (male/female): ").lower()

# 🔹 Create user dataframe
user_data = pd.DataFrame({
    "study_hours": [user_study_hours],
    "study_method": [user_study_method],
    "internet_access": [user_internet_access],
    "gender": [user_gender]
})

# 🔹 Encode user input
user_data["internet_access"] = le_internet.transform(user_data["internet_access"])
user_data["gender"] = le_gender.transform(user_data["gender"])

# 🔹 One-hot encode study_method
user_encoded = ohe.transform(user_data[["study_method"]])
user_encoded_df = pd.DataFrame(
    user_encoded,
    columns=ohe.get_feature_names_out(["study_method"])
)

# 🔹 Merge all
user_final = pd.concat(
    [user_data.drop("study_method", axis=1), user_encoded_df],
    axis=1
)

user_final = user_final[X.columns]

# 🔹 Prediction
predicted_score = model.predict(user_final)

print(f"\n🎯 Predicted Exam Score: {predicted_score[0]:.2f}")




# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")