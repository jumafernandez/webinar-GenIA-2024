from langchain_community.llms import Ollama
import time

inicio = time.time()  # Marcar el tiempo de inicio

llm = Ollama(model="llama3")

pregunta = '¿Quién es Lionel Messi?'

respuesta = llm.invoke(pregunta)

fin = time.time()  # Marcar el tiempo de finalización
tiempo_ejecucion = fin - inicio  # Calcular el tiempo de ejecución en segundos

print(respuesta)

print(f"\nTiempo de ejecución: {tiempo_ejecucion} segundos")

