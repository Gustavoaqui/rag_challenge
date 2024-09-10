

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompt_template import prompt_template
from vector_db_handler import VectorDBHandler

class RagAgent:
    llm = ChatOpenAI(model='gpt-4o')
    parser = StrOutputParser()
    prompt_template = prompt_template
    chain = prompt_template | llm | parser
    
    @classmethod
    def invoke(cls, user_question):
        docs = cls.rag_tool(user_question)
        bot_reply = cls.chain.invoke({"docs": docs, "user_question": user_question})
        return bot_reply
    
    @classmethod
    def rag_tool(cls, user_question):
        docs = VectorDBHandler.search_docs(user_question)
        return cls.process_docs(docs)
        

    @staticmethod
    def process_docs(docs):
        processed = docs[0].page_content
        return processed
    
    
print(RagAgent.invoke("What are recent advancements in natural language processing?"))