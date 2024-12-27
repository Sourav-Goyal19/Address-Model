# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from typing import Tuple
# import joblib

# # Read data
# df = pd.read_csv("./Book2.csv")

# # Drop rows where key columns are missing
# df = df.dropna(subset=["Place", "District", "State", "Pincode"])

# # Encoding categorical features using LabelEncoder
# label_encoder = LabelEncoder()

# df["Place_encoded"] = label_encoder.fit_transform(df["Place"])
# df["District_encoded"] = label_encoder.fit_transform(df["District"])
# df["State_encoded"] = label_encoder.fit_transform(df["State"])

# # Use encoded columns as features and 'Place' as the target
# X = df[["District_encoded", "State_encoded", "Pincode"]]  # Features
# y = df["Place_encoded"]  # Target

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train a Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions and calculate accuracy
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# # Print accuracy
# print(f"Model accuracy: {accuracy}")

# # Save the trained model to a file
# joblib.dump(model, "address_model.pkl")
# joblib.dump(
#     label_encoder, "label_encoder.pkl"
# )  # Save LabelEncoder for reverse transformations

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load the dataset
data = pd.read_csv("your_dataset.csv")

# Fill missing values
data.fillna("", inplace=True)

# Combine relevant columns into a single address string
data["Address"] = (
    data["Place"]
    + " "
    + data["Sub-District"]
    + " "
    + data["District"]
    + " "
    + data["State"]
)

# Split the data into features and labels
X = data["Address"]
y = data["Pincode"]

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, padding="post")

# Convert labels to numerical format
y = y.astype(str)
tokenizer_y = Tokenizer()
tokenizer_y.fit_on_texts(y)
y_seq = tokenizer_y.texts_to_sequences(y)
y_pad = pad_sequences(y_seq, padding="post")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_pad, test_size=0.2, random_state=42
)

# Define the model
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=128, input_length=X_pad.shape[1]),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(len(tokenizer_y.word_index) + 1, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# Function to predict the correct address
def predict_address(input_address):
    input_seq = tokenizer.texts_to_sequences([input_address])
    input_pad = pad_sequences(input_seq, padding="post", maxlen=X_pad.shape[1])
    prediction = model.predict(input_pad)
    predicted_pincode = tokenizer_y.sequences_to_texts([np.argmax(prediction, axis=1)])
    return predicted_pincode


# Example usage
input_address = "D-23 block A Mohn Gardn 110084"
predicted_pincode = predict_address(input_address)
print(f"Predicted Pincode: {predicted_pincode}")
