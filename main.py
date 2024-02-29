import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_openai import OpenAI

key = "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"
os.environ["OPENAI_API_KEY"] = key

llm = OpenAI(temperature=0.9, max_tokens=500)

st.title("Article QA")
st.caption("An LLM that can read news articles and answer your questions based on them")
st.sidebar.title("Links")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'link{i+1}')
    urls.append(url)

print(type(urls))

read_button = st.sidebar.button("Read")

file_path = "vector_database.pkl"

vdb = None
if read_button:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "."],
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vdb = FAISS.from_documents(chunks, embeddings)
    # print(vdb)
    # index = vdb.index
    #
    # num_vectors = index.ntotal
    #
    # # Get the dimensionality of the vectors in the index
    # dimensionality = index.d
    #
    # print("Dimensionality of vectors stored in FAISS index:", dimensionality)
    #
    # batch_size = 1000
    #
    # # Initialize an empty list to store all vectors
    # all_vectors = []
    #
    # # Iterate through vectors in batches
    # for i in range(0, num_vectors, batch_size):
    #     # Get the range of vectors to retrieve in this batch
    #     start_idx = i
    #     end_idx = min(i + batch_size, num_vectors)
    #
    #     # Retrieve vectors from the index
    #     batch_vectors = np.zeros((end_idx - start_idx, dimensionality), dtype=np.float32)
    #     index.reconstruct_n(start_idx, end_idx - start_idx, batch_vectors)
    #
    #     # Append vectors to the list
    #     all_vectors.append(batch_vectors)
    #
    # # Concatenate all vectors into a single numpy array
    # all_vectors = np.concatenate(all_vectors, axis=0)
    # with open(file_path, "wb") as f:
    #     pickle.dump(all_vectors, f)

query = st.text_input("Question :")
if query:
    if vdb is not None:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vdb.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])



