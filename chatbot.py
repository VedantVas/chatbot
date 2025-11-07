from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from PIL import Image
import pytesseract
import docx
import fitz  # PyMuPDF for PDF reading
pytesseract.pytesseract.tesseract_cmd = "tesseract"


#Streamlit app configurations
st.set_page_config(page_title="AI Text Assistant", page_icon="")

# Title and description
st.title('AI Rajubot')
st.write("This chatbot is created by Vedant")
st.markdown("Hello! I'm your AI assistant. How can I assist you today?")
st.image("https://thumbs.dreamstime.com/b/robot-icon-chat-bot-sign-support-service-concept-chatbot-character-flat-style-robot-icon-chat-bot-sign-support-service-124978456.jpg", use_container_width=True)

#Function to get API key 
def get_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key")
    return api_key

api_key = get_api_key()

#File extraction Helpers
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(".pdf"):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)

#Ensure API key is given 
if not api_key:
    st.warning("Please enter your API Key to continue.")
else:
    # Prompt template
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant. Please respond to user queries in English."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    chain = prompt | model | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

   #File Upload and Image upload
    st.subheader("Upload a File or Image")

    uploaded_file = st.file_uploader("Choose a file (txt, pdf, docx, jpg, png)", 
                                     type=["txt", "pdf", "docx", "jpg", "jpeg", "png"])

    extracted_text = ""
    if uploaded_file is not None:
        if uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            extracted_text = extract_text_from_file(uploaded_file)

        st.text_area("Extracted Content", extracted_text, height=200)

   #User inputs Section ->>
    user_input = st.text_input("Enter your question in English:", "")

    if user_input:
        st.chat_message("human").write(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Combine extracted text with user question
            context = f"Context from uploaded file/image:\n{extracted_text}\n\n" if extracted_text else ""
            final_question = context + f"User Question: {user_input}"

            config = {"configurable": {"session_id": "any"}}

            response = chain_with_history.stream({"question": final_question}, config)

            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

    else:
        st.warning("Please enter your question.")
