from langchain_community.llms import LlamaCpp
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import time

inicio = time.time()  # Marcar el tiempo de inicio

# Instancio la Base de Datos y el LLM (con Ollama)
db = SQLDatabase.from_uri("postgresql://postgres:888888@localhost:5432/exportaciones_basicas")

# Defino la ubicación del modelo y lo instancio
DIRECTORIO_LLM_MODELS = 'C:/Users/jumaf/OneDrive/Documentos/llm-models/'
MODEL_NAME = "llama-2-7b.Q5_K_S.gguf"
local_path = DIRECTORIO_LLM_MODELS + MODEL_NAME
llm = LlamaCpp(model_path=local_path, n_ctx=3850, temperature=0)

# Genero contexto para mi consulta
# 1. Genero 3 ejemplos de consulta con la respuesta esperada
examples = []

# 2. Me guardo el esquema de tablas en una variable
info_esquema_ddl = db.get_context()["table_info"]

# 3. Genero un prompt con los datos de contexto
example_prompt = PromptTemplate(input_variables=['input', 'query'], template='User input: {input}\nSQL query: {query}')
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Eres un experto en PostgreSQL. Dada una pregunta, debes crear un query sintácticamente correcto en PostgreSQL que luego pueda ejecutar. Es importante que respondas únicamente con el query, sin una palabra de más, salvo que se te indique lo contrario. \n\nA continuación te comparto la información relevante sobre el esquema de base de datos: {table_info}\n\nA su vez, a continuación se muestran una serie de ejemplos de preguntas y sus correspondientes respuestas con querys SQL.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "table_info"],
)

# Hago una pregunta que puedo responder con la DB
pregunta = "Dame el nombre y apellido de los docentes que conforman el equipo docente de la asignatura Bases de datos II (código 11078) para año 2023."

# Muestro el prompt en pantalla
print(prompt.format(input=pregunta, table_info=info_esquema_ddl))

# Creo la cadena y ejecuto la pregunta con el prompt definido
chain = create_sql_query_chain(llm, db, prompt)

respuesta = chain.invoke({"question": pregunta})
print(respuesta)

fin = time.time()  # Marcar el tiempo de finalización
tiempo_ejecucion = fin - inicio  # Calcular el tiempo de ejecución en segundos

print(f"\nTiempo de ejecución: {tiempo_ejecucion} segundos")

# Ejemplo de ejecución de consulta
# db.run("SELECT * FROM docentes LIMIT 10;")
