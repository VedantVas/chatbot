from langchain.prompts import (
    ChatPromptTemplate, # Lets you define a full conversation structure.
    HumanMessagePromptTemplate, # Represents a user’s message template.
    MessagesPlaceholder, # Placeholder for previous conversation messages (chat history).
    SystemMessagePromptTemplate, # Represents system-level instructions for the AI (like role and behavior).
)



from langchain_community.chat_message_histories import StreamlitChatMessageHistory #  Stores chat history so it can be used between turns in Streamlit.
from langchain_core.runnables.history import RunnableWithMessageHistory #Wraps an AI chain so it can keep track of conversation history across requests.
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.schema.output_parser import StrOutputParser

import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="AI Text Assistant", page_icon="")

# Title and initial description
st.title('AI Chatbot')
st.write("This chatbot is created by Vedant")
st.markdown("Hello! I'm your AI assistant. How can I assist you today?")
st.image("https://thumbs.dreamstime.com/b/robot-icon-chat-bot-sign-support-service-concept-chatbot-character-flat-style-robot-icon-chat-bot-sign-support-service-124978456.jpg", use_column_width=True)

# Function to get and store the API key securely in session state
def get_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key")
    return api_key

# Prompt user for API key input
api_key = get_api_key()

# Ensure the API key is available
if not api_key:
    st.warning("Please enter your API Key to continue.")
else:
    # Create a prompt template
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant. Please respond to user queries in English."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Initialize message history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    # Set up the Google AI model using the provided API key
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    # Combine prompt, model, and output parser
    chain = prompt | model | StrOutputParser()

    # Combine chain with history for maintaining session
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # Get user input
    user_input = st.text_input("Enter your question in English:", "")

    # Check if user input is provided
    if user_input:
        st.chat_message("human").write(user_input)

        # Assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Configuration dictionary
            config = {"configurable": {"session_id": "any"}}

            # Get response from AI model
            response = chain_with_history.stream({"question": user_input}, config)

            # Stream and display response in real-time
            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

    else:
        st.warning("Please enter your question.")
