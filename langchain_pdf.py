import gc
from langchain.document_loaders import PyPDFLoader
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


#pip install sentence-transformers faiss-cpu
#I want to know about UNA housing.
llm = LlamaCpp(
    model_path=r"tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf",# Please replace to your LLM file path
    n_ctx=10000,#Maximum number of tokens that the model can process at once
    temperature=0,#"Randomness" of model output
    max_tokens=200,#Number of text generated
    verbose=True,
    streaming=True,
)

def chain_main(message, pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Maximum number of characters in chunk
        chunk_overlap=20,  # Maximum number of characters for overlap
    )
    texts = text_splitter.split_text(text)

    index = FAISS.from_texts(
        texts=texts,
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
    )
    index.save_local("storage")

    qa = ConversationalRetrievalChain.from_llm(llm, chain_type="stuff", retriever=index.as_retriever(search_kwargs={"k": 1}))

    chat_history = []

    query = message
    result = qa({"question": query, "chat_history": chat_history})
    last_punctuation_index = max(result['answer'].rfind('.'), result['answer'].rfind('?'), result['answer'].rfind('!'))
    if last_punctuation_index != -1:
        output = result['answer'][:last_punctuation_index + 1]
    print("Answer:", output)

    # Free up resources that are no longer needed
    del reader, text, texts, index, qa, chat_history, query, result
    gc.collect()

    return output
