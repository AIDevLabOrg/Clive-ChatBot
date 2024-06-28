from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# pdf_loader = PyPDFDirectoryLoader("/content/Files/PDF_Files")

# csv_loader = CSVLoader('/content/Files/CSV_Files/Psych_data.csv')


# loader_all = MergedDataLoader(loaders=[pdf_loader, csv_loader])
# docs_all = loader_all.load()

# # Extract text content from documents
# all_texts = [doc.page_content for doc in docs_all]



# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents(all_texts)


model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key="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", openai_organization="org-VnBYTgqBSLTjZwvAe1aV0C0J")
# vectorstore = FAISS.from_documents(texts, model)
# vectorstore.save_local("vectors")

"""# Main Chatbot"""

loaded_vectors = FAISS.load_local("faiss_index", model, allow_dangerous_deserialization=True)


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="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",
    organization="org-VnBYTgqBSLTjZwvAe1aV0C0J")



prompt_template = """
You are an expert Chat Assistant who helps interns/trainees with their queries.

Given the context, answer the question.

{context}

Question: {question}

INSTRUCTIONS:
- IF the user greets, greet back.
- DO NOT greet with every response.
- IF the context is not similar to the question, respond with 'I don't know the answer'.
- Make the answers short conscise and precise.

FORMATTING INSTRUCTION:
- DO NOT add any asterisks in the response.
- Keep the response plain in simple strings.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    loaded_vectors.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

question = "How do you collaborate with teachers, parents, and other stakeholders to support a student's mental health?"
result = qa_chain.invoke({"question": question})
print(result['answer'])