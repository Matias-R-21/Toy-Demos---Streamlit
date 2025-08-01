from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Bloque lateral izquierdo con explicación e input de API
st.sidebar.title("Configuración y ayuda")
st.sidebar.markdown("""
**Conversa con tu PDF**

Esta aplicación te permite subir un archivo PDF y hacerle preguntas usando inteligencia artificial. 
Solo necesitas tu clave de OpenAI y el PDF que deseas consultar. 
Tus preguntas serán respondidas en base al contenido del PDF.
""")

def get_openai_api_key():
    input_text = st.sidebar.text_input(
        label="Coloca tu API Key de OpenAI",
        placeholder="Ejemplo: sk-2twmA8tfCb8un4...",
        key="openai_api_key_input",
        type="password"
    )
    return input_text


openai_api_key = get_openai_api_key()

st.sidebar.link_button(label= "¿No tiene una API?" , url = "https://openai.com/es-ES/api/" )

st.title("Conversa con tu PDF 📄 - Utiliza RAG - LangChain 🦜🔗")

st.write("""⚠️ Tenga en cuenta que es una "Toy Demo". No utilice pdf 's pesados, tampoco suba información sensible o privada.""")

cargar_pdf = st.file_uploader("Sube tu archivo PDF", type=["pdf"])

if cargar_pdf is not None and openai_api_key:
    with open("temp.pdf", "wb") as f:
        f.write(cargar_pdf.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
else:
    st.info("Por favor, sube un PDF y coloca tu OpenAI API Key para continuar.")


def load_llm(openai_api_key):
    mi_chatbot = ChatOpenAI(model="gpt-3.5-turbo-0125",openai_api_key=openai_api_key)
    return mi_chatbot


#Prompt para cadena 1
system_prompt = ("Vas a actuar como si fuera un PDF capas de comunicarse con los usuarios."
                 "No respondas a las preguntas que no tengan relación con el contexto del PDF."
                 "Si parte de la pregunta o la consulta no tiene relación con el contexto, no la respondas."
             "Tu tarea es responder las preguntas de los usuarios en base al contexto proporcionado."
          "Utilice los siguientes fragmentos de contexto recuperados para responder a la pregunta."
          "Si no sabe la respuesta, diga que no la sabe."
          "Usa un minimo de tres oraciones y un máximo de 8 oraciones para generar la respuesta, y mantén la respuesta concisa."
          "\n\n"
          "{context}"
)


prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


spliters = CharacterTextSplitter(
    separator= "\n",
    chunk_size = 1000,
    chunk_overlap= 200
)

if cargar_pdf is not None and openai_api_key:


    pdf_splitter = spliters.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    vector_soter = FAISS.from_documents(pdf_splitter,embeddings)

    retriever = vector_soter.as_retriever(search_kwargs={"k": 3})


output_parser = StrOutputParser()


if cargar_pdf is not None and openai_api_key:
    st.markdown("Ahora podras conversar con tu PDF")

    question = st.text_input("Consultar: ")

    chatmodel = load_llm(openai_api_key=openai_api_key)

    rag_chain = {
            "context": itemgetter("question") | retriever,
            "input": itemgetter("question")
        } | prompt | chatmodel | output_parser    

    cadena = rag_chain.invoke({"question":question})

    st.write(cadena)
        


    

    

    



