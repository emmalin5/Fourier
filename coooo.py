import streamlit as st
import cohere

# Initialize the Cohere client
co = cohere.Client('X9Pnnul4KJFCncqCKcljES4qpglWShtrfvlRujAG') 

# Define the Streamlit app
def generate_text(prompt):
    try:
        # Use a valid model ID (e.g., 'command-xlarge')
        response = co.generate(
            model='command-xlarge',  # Replace with a known valid model if necessary
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        print(response)
        # Correct way to get the generated text
        return response.generations[0].text  # This accesses the first generation's text
    except coooo.errors.NotFoundError as e:
        return f"Error: Model not found. Please check the model name."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Cohere API with Streamlit")
st.write("Generate text using Cohere's API.")

# Input from the user
user_input = st.text_input("Enter your prompt:")

if user_input:
    # Generate response from Cohere
    response = generate_text(user_input)
    st.write(f"Generated Text: {response}")
