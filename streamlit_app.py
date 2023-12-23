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
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.prompt import PromptTemplate

# Título de la página
st.set_page_config(page_title='🤖 Structural Database Search')
st.title('DataBase-SearchGPT: Asistente de Búsqueda en la Base de datos')

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
  url = f'df_base.csv'

  # Verificar si el archivo existe.
  if not os.path.exists(url):
    raise FileNotFoundError(f'El archivo {archivo} no existe.')

  # Leer el archivo CSV.
  df_db = pd.read_csv(url, delimiter=',', parse_dates=True)

  return df_db

df_db = read_csv_from_github('df_base.csv')


# Añadir una opción para seleccionar el modelo de openai a utilizar
model_option = st.sidebar.selectbox(
    'Confirma el modelo de OpenAI API:',
    ('gpt-4-1106-preview')
)

def generate_response(df_db, user_query):
 # Crear un objeto `ChatOpenAI` con la configuración deseada.
 llm = ChatOpenAI(temperature=0, model=model_option, openai_api_key=openai_api_key, streaming=True)
 # Crear un objeto `PromptTemplate` con el formato de la respuesta deseada.
 _DEFAULT_TEMPLATE = """
 Dada una consulta del usuario {dialect}
 Sólo utiliza la información de df_db que integra df_estructuras y df_memorias 
 1. Consulta los datos exactos y similares en df_estructuras
 2. Devuelve una respuesta en español que incluya:
   - Lista con referencias a todos los proyectos con las condiciones exactas a la consulta de usuario en df_estructuras con id_archivo y su url de ubicación
   - Lista con referencias a algunos proyectos con las condiciones similares a la consulta de usuario en df_estructuras con id_archivo y su url de ubicación

 Respuesta {output}
 Pregunta {input}
 """
 PROMPT = PromptTemplate(input_variables=["input","dialect","output"],template=_DEFAULT_TEMPLATE)
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
 result = response["output"]
 return st.success(result)

# Obtén la openai api key desde https://platform.openai.com/account/api-keys 🔑
openai_api_key = st.sidebar.text_input('Inserte su OpenAI API Key', type='password')
if not openai_api_key.startswith('sk-'):
 st.warning('Por favor ingrese su llave OpenAI API!', icon='⚠')
## Obtén las características de la estructura del usuario
caracteristicas_estructura = st.text_input('Ingresa las características para la búsqueda:', placeholder = 'Edificio en suelo tipo D ...', disabled=not (openai_api_key))

# Agrega más información a la solicitud para una respuesta robusta
texto_ad1 = "Limítate a siempre actuar como buscador en el dataframe. Lista todos los proyectos usando id_archivo y url si cumple con las condiciones exactas en el dataframe a la siguiente consulta de usuario :"
texto_ad2 = ". O referencia proyectos usando id_archivo y url con alguna condición similar a la consulta de usuario en el dataframe. Responde siempre en español y referencia siempre por id_archivo y url"
user_query = texto_ad1 + caracteristicas_estructura + texto_ad2


if openai_api_key.startswith('sk-') and caracteristicas_estructura != '':
 with st.spinner('El asistente 🤖 está procesando su consulta...'):
  response = generate_response(df_db, user_query)

 if response:
  st.write('...respuesta generada')
# ...

# Botón para acceder a la información del autor y los derechos de autor.
if st.sidebar.button('Información del autor y los derechos de autor'):
  st.sidebar.write('**Autor:** Jefferson Ramos')
  st.sidebar.write('**Derechos de autor:** 2023 Jefferson Ramos')
  st.sidebar.write('**Licencia:** MIT')


