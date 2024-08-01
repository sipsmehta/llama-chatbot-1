import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model and tokenizer
model_name = "meta-llama/Meta-Llama-Guard-2-8B"  # Using a publicly available model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert model to bfloat16 precision
model = model.half()

# Move model to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Implement the chatbot
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def chatbot_response(user_input):
    logger.info(f"Received user input: {user_input}")  # Log the user input
    
    if not user_input.strip():
        logger.warning("Empty input received")
        return "Please provide a valid input."

    # Encode the input and create attention mask
    encoded_input = tokenizer.encode_plus(
        user_input,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=512
    )
    inputs = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    logger.info(f"Encoded input shape: {inputs.shape}")  

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {output_text}")  # Log the generated response
        return output_text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Create a Streamlit app
def main():
    st.title("Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your message?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        logger.info("Sending user input to chatbot_response function")  # Log before calling the function
        response = chatbot_response(prompt)
        logger.info("Received response from chatbot_response function")  # Log after receiving the response
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

# To run: streamlit run chatbot.py
