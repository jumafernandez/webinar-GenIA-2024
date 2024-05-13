from langchain_community.llms import LlamaCpp
import time

inicio = time.time()  # Marcar el tiempo de inicio

# Defino la ubicación del modelo
DIRECTORIO_LLM_MODELS = 'C:/Users/jumaf/OneDrive/Documentos/llm-models/'
MODEL_NAME = "llama-2-7b.Q5_K_S.gguf"
local_path = DIRECTORIO_LLM_MODELS + MODEL_NAME

llm = LlamaCpp(model_path=local_path)

pregunta = '¿Podrías compartirme los principales datos de la biografía de Lionel Messi?'

respuesta = llm.invoke(pregunta)

fin = time.time()  # Marcar el tiempo de finalización
tiempo_ejecucion = fin - inicio  # Calcular el tiempo de ejecución en segundos

print(respuesta)

print(f"\nTiempo de ejecución: {tiempo_ejecucion} segundos")

