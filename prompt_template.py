from langchain_core.prompts.prompt import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["docs", "user_question"], 
    template="""
    You are a retriever assistant. Your role is to use the document provided to answer the user question.
    
    Docs:
    {docs}
    
    User question:
    {user_question}
    """
)