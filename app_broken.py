import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Load and preprocess the data
df = pd.read_csv("DSP_13.csv", delimiter=";")
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the data into features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Define dictionaries for labeling categorical variables
symptoms_d = {0: "No symptoms", 1: "Some symptoms", 2: "Severe symptoms"}
disease_d = {0: "No disease", 1: "Mild disease", 2: "Severe disease"}
meds_d = {0: "No medication", 1: "Some medication", 2: "High medication"}

# Define the Streamlit app
def main():
    st.set_page_config(page_title="Health App")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://online.hbs.edu/Style%20Library/api/resize.aspx?imgpath=/online/PublishingImages/blog/health-care-economics.jpg&w=1200&h=630")

    with overview:
        st.title("Health App")

    with left:
        symptoms_silder = st.slider("Symptoms", min_value=0, max_value=5)
        age_slider = st.slider("Age", value=30, min_value=0, max_value=77)
        disease_slider = st.slider("Disease", min_value=0, max_value=5)

    with right:
        height_slider = st.slider("Height (cm)", value=170, min_value=0, max_value=200)
        meds_slider = st.slider("Medication", min_value=0, max_value=4)

    data = [[symptoms_silder, age_slider, disease_slider, height_slider, meds_slider]]
    health = model.predict(data)
    h_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Would this person be healthy?")
        st.subheader(("Yes" if health[0] == 1 else "No"))
        st.write("Prediction confidence: {0:.2f} %".format(h_confidence[0][1] * 100))

if __name__ == "__main__":
    main()
