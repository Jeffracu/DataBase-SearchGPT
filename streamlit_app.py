## ⚙️ Setup

import os
import openai
import warnings
import pandas as pd
import os
import streamlit as st

warnings.filterwarnings("ignore")

# Obtén la openai api key desde https://platform.openai.com/account/api-keys 🔑

from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.prompt import PromptTemplate

# Título de la página
st.set_page_config(page_title='🤖 Structural Database Search')
st.title('🤖 DataBase-SearchGPT: Asistente de Búsqueda en la Base de datos')

##Primero se deben subir los archivos csv que se van a utilizar como fuente de información 


# Lista de archivos csv
archivos_csv = ['https://github.com/Jeffracu/DataBase-SearchGPT/blob/master/informacionEstructuras.csv', "https://github.com/Jeffracu/DataBase-SearchGPT/blob/master/archivosMemorias.csv"]
# Crear una lista vacía para almacenar los dataframes
lista_dataframes = []

# Iterar sobre la lista de archivos csv
for archivo in archivos_csv:
    # Leer cada archivo csv y agregarlo a la lista de dataframes
    lista_dataframes.append(pd.read_csv(archivo))

# Usar el primer DataFrame como base
df_base = lista_dataframes[0]

# Iterar sobre los demás DataFrames y combinarlos con el DataFrame base
for df in lista_dataframes[1:]:
    df_base = pd.merge(df_base, how='inner', df, on='id_archivo')

# Añadir una opción para seleccionar el modelo de openai a utilizar
model_option = st.sidebar.selectbox(
    'Selecciona un modelo:',
    ('gpt-3.5-turbo-1106','gpt-4','gpt-4-1106-preview')
)

def generate_response(df_base, user_query):
 # Crear un objeto `ChatOpenAI` con la configuración deseada.
 llm = ChatOpenAI(temperature=0, model=model_option, openai_api_key=openai_api_key, streaming=True)
 df_base = df_base
 # Crear un objeto `PromptTemplate` con el formato de la respuesta deseada.
 _DEFAULT_TEMPLATE = """
  Dada una consulta del usuario {dialect}
Sólo utiliza la información de df_base
  1. Consulta los datos similares en df_base
  2. Devuelve una respuesta en español que incluya:
     - Referencias a proyectos con las condiciones exactas a la consulta de usuario en df_base con id_archivo
     - Referencias a proyectos con las condiciones similares a la consulta de usuario en df_base con id_archivo

  Respuesta {output}
  Pregunta {input}
  """
 PROMPT = PromptTemplate(input_variables=["input","dialect","output",],template=_DEFAULT_TEMPLATE)
  # Crear un agente `pandas_df_agent` que usa el modelo de lenguaje y el DataFrame base.
 pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df_base,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prompt_template=PROMPT,
        max_iterations=5,
        handle_parsing_errors=True,
  )
 
 response = pandas_df_agent({"input": user_query}, {"dialect": _DEFAULT_TEMPLATE}, include_run_info=True)
 result = generate_response(df_base, user_query)
 return st.success(result)


## Obtén las características de la estructura del usuario
caracteristicas_estructura = st.text_input('Ingresa las características para la búsqueda:', placeholder = 'Escribe tu consulta aquí ...')

# Agrega más información a la solicitud para una respuesta robusta
texto_ad1 = "Limítate a siempre actuar como buscador en la base de datos, además referencia proyectos usando id_archivo con las condiciones exactas en df_base a la siguiente consulta de usuario :\n"
texto_ad2 = "\n O referencia proyectos usando id_archivo con alguna condición similar a la consulta de usuario en df_base y describe las características de cada proyecto referenciado"
user_query = texto_ad1 + caracteristicas_estructura + texto_ad2

openai_api_key = st.sidebar.text_input('Inserte su OpenAI API Key', type='password', disabled=not (user_query))
#"""Análisis del agente"""
if not openai_api_key.startswith('sk-'):
 st.warning('Por favor ingrese su llave OpenAI API!', icon='⚠')
if openai_api_key.startswith('sk-'):
 st.header('Sugerencia')
 generate_response(df_base, user_query)
