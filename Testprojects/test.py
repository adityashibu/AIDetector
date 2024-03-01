from train import train_svm_model
from preprocess import preprocess_text

# Define the function to test a given input text
def test_input_text(input_text, trained_model, vectorizer):
    # Preprocess the input text
    preprocessed_input_text = preprocess_text(input_text)  # Implement this function to preprocess the input text

    # Vectorize the preprocessed text using the same CountVectorizer instance
    input_text_vectorized = vectorizer.transform([preprocessed_input_text])

    # Predict using the trained logistic regression model
    prediction = trained_model.predict(input_text_vectorized)[0]

    # Interpret the prediction
    if prediction == 0:
        print("The input text is classified as human-generated.")
    else:
        print("The input text is classified as AI-generated.")

if __name__ == "__main__":
    # Train the logistic regression model and get the trained model and vectorizer
    trained_model, vectorizer = train_svm_model("preprocessed_data.csv")

    # Test a sample input text
    sample_input_text = "AI has been in the forefront of technology, It has been helping people with multiple activities, mainly automating and making peoples lives easier."
    test_input_text(sample_input_text, trained_model, vectorizer)