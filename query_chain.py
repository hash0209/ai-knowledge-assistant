from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from prompts import customized_prompt
from dotenv import load_dotenv


load_dotenv()


def get_llm():
    return ChatGroq(temperature =0,model="llama3-8b-8192")


def create_qa_chain(llm , retriever,memory):
    return ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= retriever ,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": customized_prompt}



        
    )

def ask_a_question(qa_chain , query):
    result  = qa_chain.invoke({"question":query})
    return result["answer"]

