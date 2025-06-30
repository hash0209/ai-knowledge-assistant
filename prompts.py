from langchain_core.prompts import PromptTemplate


template = """ You are a chat assistant. With the help of context provided below , answer the question asked by the user 
Context : {context}

Question : {question}

"""

customized_prompt=PromptTemplate(input_variables=["context","question"],template=template)