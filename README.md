# 🍽️ AI Restaurant Booking Assistant 

This project is an AI-powered chatbot for **Restaurant** that helps users with FAQs and bookings through a conversational interface. Built using [Streamlit](https://streamlit.io/) and [LangGraph](https://github.com/langchain-ai/langgraph), it uses a PDF-based RAG (Retrieval-Augmented Generation) pipeline to respond accurately to customer queries.

---

## 🧠 Features

- 📄 Answers questions based on a PDF 
- 🔁 Maintains conversational context using LangGraph
- 🔎 Integrates a document retriever for factual answers
- 🤖 Powered by LLM (`llama-3.1-8b-instant`) via Groq
- 🧰 Uses LangChain tools for modular, callable retrieval logic
- 💬 Clean chat interface with [Streamlit's chat input and messages](https://docs.streamlit.io/library/api-reference/chat)
- 🔁 Reset chat option
- ❌ Supports user commands to stop the chat (e.g., "exit", "quit")

---

## 📁 Project Structure

├── app.py # Streamlit application entry point
├── components/
│ └── utility.py # Utility functions (load_file, get_retriever)
├── .env 
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🚀 How to Run

### 1. Clone the repository

```command prompt
git clone https://github.com/DipeshDhote/RAG_Based_Chatbot.git
cd RAG_Based_Chatbot
```

## Sample Interaction

User: What are your opening hours?

Bot: According to Document 1:

Restaurant is open daily from 12 PM to 11 PM.

User: Can I book a table for 4 people?

Bot: Sure! Please confirm the date and time for the booking.

