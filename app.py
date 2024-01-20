from flask import *

import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

##############################################################################
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain, RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate


UPLOAD_FOLDER = './docs'
folder = './docs'

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def process_pdf(query, file_name):
	#storage.child("docs/" + file_name).download("/docs/", "docs/" + file_name)
  global agent_executor
  file_na = './docs/'+str(file_name)
  print('inside process: ',file_na)

  if (file_name.rsplit('.', 1)[1].lower() == 'txt'):
    loader = TextLoader(file_na)
  else:
    print('inside pdf loader')
    loader = PyPDFLoader(file_na, extract_images=False)
  #loader = TextLoader(file_name)

  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
  texts = text_splitter.split_documents(documents)
  embeddings = OpenAIEmbeddings()
  db = FAISS.from_documents(texts, embeddings)
  retriever = db.as_retriever()
  tool = create_retriever_tool(
                      retriever,
                      "search_state_of_union",
                      "Searches and returns excerpts from the 2022 State of the Union.",
                      )
  tools = [tool]
  #prompt = hub.pull("hwchase17/openai-tools-agent")

  prompt_template = """
    User: Use the following pieces of context to provide a concise answer
    to the question at the end. If you don't know the answer, just say
    that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Assistant:
    """




  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

  llm = ChatOpenAI(temperature=0)
  qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                        retriever=db.as_retriever(search_type="similarity", 
                                        search_kwargs={"k": 3, "lambda_mult": 0.2}), 
                                        return_source_documents=True,chain_type_kwargs={"prompt": prompt})
  result = qa_chain({"query": query})
  
  #agent = create_openai_tools_agent(llm, tools, prompt)
  #agent_executor = AgentExecutor(agent=agent, tools=tools)
  #result = agent_executor.invoke({"input": query})
  print('result', result['result'])
  #os.remove("docs/"+file_name)
  return result['result']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    print(request.form.get('btn'))
    if(request.form.get('btn') == 'index'):
      global file
      file = request.files['upload']
      if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(folder, filename))
        return redirect(url_for('qa'))
      else:
        return 'only .pdf and .txt files are supported'
    elif (request.form.get('btn') == 'qa'):
      question = request.form.get('question')
      answer = process_pdf(question, file.filename)
    return render_template('qa.html', answer = answer, question = question)
  return render_template('upload.html')
  
@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/qa/', methods=['GET', 'POST'])
def qa():
    return render_template('qa.html')
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
