
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# --- Page setup ---
st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge – AI Business Intelligence Assistant")

# --- Step 1: Load CSV ---
try:
    df = pd.read_csv("sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.success("Loaded sales_data.csv successfully")
except Exception as e:
    st.error(f"CSV loading error: {e}")
    st.stop()

# --- Step 2: Data Preview ---
st.subheader("Data Preview")
st.write(df.head())

# --- Step 3: Visualizations ---
st.sidebar.subheader("Choose a visualization")
plot_type = st.sidebar.selectbox("Chart Type", ["Monthly Trend", "Sales by Product", "Sales by Region", "Sales by Gender"])

st.subheader(f" {plot_type}")
if plot_type == "Monthly Trend":
    monthly_sales = df.resample("ME", on="Date")["Sales"].sum()
    fig, ax = plt.subplots()
    monthly_sales.plot(ax=ax, marker="o")
    ax.set_title("Monthly Sales Trend")
    st.pyplot(fig)
elif plot_type == "Sales by Product":
    st.bar_chart(df.groupby("Product")["Sales"].sum())
elif plot_type == "Sales by Region":
    st.bar_chart(df.groupby("Region")["Sales"].sum())
elif plot_type == "Sales by Gender":
    st.bar_chart(df.groupby("Customer_Gender")["Sales"].sum())

# --- Step 4: Summarize data as documents ---
top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5)
monthly = df.resample("ME", on="Date")["Sales"].sum().reset_index()
gender = df.groupby("Customer_Gender")["Sales"].sum().to_string()
region = df.groupby("Region")["Sales"].sum().to_string()

product_summary = "\n".join([f"{p}: {s} sales" for p, s in top_products.items()])
trend_summary = "\n".join([f"{row['Date'].strftime('%B %Y')}: {row['Sales']} sales" for _, row in monthly.iterrows()])

docs = [
    Document(page_content=f"Top 5 Products:\n{product_summary}"),
    Document(page_content=f"Monthly Sales Trend:\n{trend_summary}"),
    Document(page_content=f"Sales by Gender:\n{gender}"),
    Document(page_content=f"Sales by Region:\n{region}")
]

# --- Step 5: OpenAI Embeddings + FAISS ---
embeddings = OpenAIEmbeddings()  #add OpenAPI key
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# --- Step 6: OpenAI QA chain with memory ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# --- Step 7: Q&A UI ---
st.sidebar.markdown("---")
st.sidebar.subheader(" Ask InsightForge")
user_query = st.sidebar.text_area("Your Question:", "Which region had the highest sales?")
if st.sidebar.button("Ask"):
    with st.spinner("Generating insight..."):
        result = qa_chain.invoke({"question": user_query})
        st.subheader(" Insight")
        st.success(result["answer"])
