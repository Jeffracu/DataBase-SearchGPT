#Copyright 2024 Jefferson Ramos.

## ⚙️ Setup

import os
import openai
import warnings
import pandas as pd
import os
import requests
import tabulate
import streamlit as st
warnings.filterwarnings("ignore")
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.prompt import PromptTemplate



# Título de la página
st.set_page_config(page_title='🤖 GPT Powered Database Search')
st.title('**GPT-Powered Database Search**')

@st.cache_data(ttl="2h")
def read_csv_from_github(archivo):
  """
  Lee un archivo CSV que se encuentra en la rama master del repositorio de GitHub.

  Parámetros:
    archiv: Nombre del archivo CSV.

  Devuelve:
    DataFrame con los datos del archivo CSV.
  """

  # Obtener el URL del archivo CSV.
  url1 = f'df_base.csv'

  # Verificar si el archivo existe.
  if not os.path.exists(url1):
    raise FileNotFoundError(f'El archivo {archivo} no existe.')

  # Leer el archivo CSV.
  df_db = pd.read_csv(url1, delimiter=',', parse_dates=True)

  return df_db

df_db = read_csv_from_github('df_base.csv')


#Modelo de openai a utilizar
st.sidebar.write('**Modelo OpenAI GPT:** gpt-4-1106-preview')

def generate_response(df_db, user_query):
 # Crear un objeto `ChatOpenAI` con la configuración deseada.
 llm = ChatOpenAI(temperature=0, model='gpt-4-1106-preview', openai_api_key=openai_api_key, streaming=True, max_tokens=None)
 # Crear un objeto `PromptTemplate` con el formato de la respuesta deseada.
 _DEFAULT_TEMPLATE = """
 Dada una consulta del usuario {input}
 Sólo utiliza la información de df_db 
 1. Consulta los datos exactos y similares en df_db
 2. Devuelve una respuesta en español que incluya:
   - Lista con referencias a todos los proyectos con las condiciones exactas a la consulta de usuario en df_db con id_archivo y su carpeta de ubicación
   - Lista con referencias a algunos proyectos con las condiciones similares a la consulta de usuario en df_db con id_archivo y su carpeta de ubicación

 Respuesta {output}
 Pregunta {input}
 """
 PROMPT = PromptTemplate(input_variables=["input","dialect"],template=_DEFAULT_TEMPLATE)
 # Crear un agente `pandas_df_agent` que usa el modelo de lenguaje y el DataFrame base.
 pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df_db,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prompt_template=PROMPT,
    max_iterations=5,
    handle_parsing_errors=True,
    )

 response = pandas_df_agent({"input": user_query}, {"dialect": _DEFAULT_TEMPLATE}, include_run_info=True)
  #response = pandas_df_agent({"input": user_query}, {"dialect": _DEFAULT_TEMPLATE}, include_run_info=True)
 result = response["output"]
 return st.success(result)

# Obtén la openai api key desde https://platform.openai.com/account/api-keys 🔑
openai_api_key = st.sidebar.text_input('Inserte su llave OpenAI API', type='password')
if not openai_api_key.startswith('sk-'):
 st.warning('Por favor, ingrese su llave OpenAI API en la barra desplegable de la izquierda', icon='⚠')
## Obtén las características de la estructura del usuario
caracteristicas_estructura = st.text_input('Ingresa las características para la búsqueda:', placeholder = 'Edificios en suelo tipo D ...', disabled=not (openai_api_key))

# Agrega más información a la solicitud para una respuesta robusta
texto_ad1 = "Actúa como buscador del dataframe df_db. Lista completamente (sin abreviar) todos los proyectos usando id_archivo, url (clicable, sin acortar) y nombre, si cumple con las condiciones exactas en el dataframe a la siguiente consulta de usuario :"
texto_ad2 = ". Si no hay coincidencias exactas referencia proyectos usando id_archivo, url (clicable, sin acortar) y nombre, con alguna condición similar a la consulta de usuario en el dataframe. Responde siempre en español y referencia siempre con una lista por id_archivo, url (clicable, sin acortar) y nombre"
user_query = texto_ad1 + caracteristicas_estructura + texto_ad2


if openai_api_key.startswith('sk-') and caracteristicas_estructura != '':
 with st.spinner('El asistente **🤖** está procesando su consulta...'):
  response = generate_response(df_db, user_query)

 if response:
  st.write('...respuesta generada')
  # ...

# Botón para acceder a la información del autor y los derechos de autor.
if st.sidebar.button('Información del autor y los derechos de autor'):
  st.sidebar.write('**Autor:**')
  st.sidebar.write('Jefferson Ramos')
  st.sidebar.write('**Licencia:**')
  st.sidebar.write('Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license.') 
  st.sidebar.write('Copyright 2024 Jefferson Ramos')
  


