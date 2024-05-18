from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
import os

from datetime import datetime

def hora_actual():
    return datetime.now().strftime("%H:%M:%S")

def cargar_archivos_locales(folder):
    """
    Esta función carga los datos en la base de datos vectorial
    Parameters
    ----------
    folder : str
        Carpeta donde están los archivos con los datos específicos

    Returns
    -------
    retriever : Chroma
        Base de datos vectoriales con los documentos splitteados

    """
    # Se cargan los pdf
    print(f"\nInicio de carga de documentos (RAG): {hora_actual()}.")

    # Obtener la lista de archivos PDF en el directorio
    pdf_files = [file for file in os.listdir(folder) if file.endswith('.pdf')]
    
    for filename in pdf_files:
        print(f'Cargando PDF: {filename}')
       
        # Cargar el PDF
        loader = PyPDFLoader(folder + filename, extract_images=True)
        docs = loader.load()
        
        # Splitting de texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)
        
        # Almacenamiento de resultados divididos
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=GPT4AllEmbeddings())
        
        # Retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        print(f'PDF {filename} procesado y almacenado en el vectorstore.')
    
    print(f'\nProceso de load de documentos finalizado: {hora_actual()}.')
    
    return retriever

# Sin datos de Juan Manuel Fernández
FOLDER_PATH = 'C:/Users/jumaf/Documents/GitHub/webinar-genIA-2024/data/vacio/'
# Con datos de Juan Manuel Fernández
FOLDER_PATH = 'C:/Users/jumaf/Documents/GitHub/webinar-genIA-2024/data/'

archivos_locales = cargar_archivos_locales(FOLDER_PATH)


from langchain_core.prompts import ChatPromptTemplate

template = """Responde la pregunta utilizando únicamente el idioma español.
No utilices el idioma inglés.

Basate para tu respuesta fundamentalmente en el siguiente contexto:

{context}

Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)



from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

llm_model = Ollama(model="llama2")

RAG_chain = (
    {"context": archivos_locales, "question": RunnablePassthrough()}
    | prompt
    | llm_model
)

response = RAG_chain.invoke("¿Quién es Juan Manuel Fernández?")
print(response)

