import streamlit as st
import transformers
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import os

# Streamlit page config
st.set_page_config(page_title="LLaMA Support Ticket Analyzer", page_icon="🎫", layout="wide")

@st.cache_resource
def initialize_pipeline():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Get the Hugging Face token from the environment variable
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")
        st.stop()

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Create a text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return pipeline

@st.cache_data
def load_dataset(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path, low_memory=False)
        st.success(f"Dataset loaded successfully with {len(dataset)} rows")
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Create an empty DataFrame if loading fails

def search_dataset(query, dataset):
    query_words = query.lower().split()
    results = dataset[
        dataset['subject'].str.lower().apply(lambda x: any(word in str(x) for word in query_words)) |
        dataset['description'].str.lower().apply(lambda x: any(word in str(x) for word in query_words))
    ]
    return results.to_dict(orient='records')

def analyze_common_issues(dataset, top_n=5):
    subject_counts = Counter(dataset['subject'])
    top_subjects = subject_counts.most_common(top_n)
    total = len(dataset)
    result = [(subject, count, (count/total)*100) for subject, count in top_subjects]
    return result

def generate_response(pipeline, prompt):
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = outputs[0]["generated_text"].strip()
    return response.split("Assistant:")[-1].strip()

def summarize_dataset(dataset):
    total_tickets = len(dataset)
    unique_subjects = dataset['subject'].nunique()
    top_issues = analyze_common_issues(dataset)
    
    summary = f"Dataset Summary:\n"
    summary += f"Total tickets: {total_tickets}\n"
    summary += f"Unique subjects: {unique_subjects}\n\n"
    summary += "Top 5 issues:\n"
    for i, (issue, count, percentage) in enumerate(top_issues, 1):
        summary += f"{i}. {issue}: {count} ({percentage:.1f}%)\n"
    
    return summary

def chatbot(user_input, pipeline, dataset):
    instructions = """
    You are an AI assistant for analyzing support ticket data. Your responses should be:
    Accurate: Only provide information from the dataset. Never invent data.
    Concise: Give brief, focused answers addressing the user's query.
    Data-driven: For common issues, summarize top ticket categories with counts and percentages.
    Honest: Admit when you lack relevant information.
    Consistent: Maintain coherence across responses.
    Direct: Avoid unnecessary preambles. Answer queries promptly and precisely.
    Base all responses on the support ticket dataset. If unsure, state your uncertainty clearly.
    """

    if "summarize" in user_input.lower() or "summarise" in user_input.lower():
        context = summarize_dataset(dataset)
    else:
        search_results = search_dataset(user_input, dataset)
        if search_results:
            context = "\n".join([f"Subject: {result['subject']}\nDescription: {result['description']}" for result in search_results[:3]])
        else:
            context = "No relevant information found in the dataset."

    prompt = f"""Instructions: {instructions}

User Query: {user_input}

Relevant Context from Support Tickets:
{context}"""

    response = generate_response(pipeline, prompt)
    return response

def main():
    st.title("🎫 LLaMA Support Ticket Analyzer")
    st.write("Ask questions about support tickets and get AI-generated responses based on the dataset.")

    # Initialize the pipeline
    with st.spinner("Initializing the AI model... This may take a few minutes."):
        pipeline = initialize_pipeline()

    # Load the dataset
    dataset_path = "support_tickets.csv"  # Update with your dataset path
    dataset = load_dataset(dataset_path)

    # Sidebar
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.write(f"Total tickets: {len(dataset)}")
    st.sidebar.write(f"Unique subjects: {dataset['subject'].nunique()}")

    st.sidebar.header("🔍 Example Queries")
    example_queries = [
        "Summarize the dataset",
        "What are the most common issues?",
        "Tell me about network connectivity problems",
        "How many tickets are related to software updates?",
    ]
    for query in example_queries:
        if st.sidebar.button(query):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                response = chatbot(query, pipeline, dataset)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about the support tickets?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chatbot(prompt, pipeline, dataset)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
