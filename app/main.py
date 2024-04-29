from fastapi import FastAPI, Body, BackgroundTasks
from .celery.worker import search_companies, get_company_data
from fastapi.responses import JSONResponse
from string import ascii_lowercase
import time
import tqdm
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import google.ai.generativelanguage as glm

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from google.api_core import retry
from tqdm.auto import tqdm
from vertexai.generative_models import ChatSession

app = FastAPI()

# This will hold the final data
final_data = []
data_loading = False
data_loading_complete = False

analyze_loading = False
analyze_loading_complete = False

analysis_data = []
description = []
analysis_flag = False

def fetch_data(letters: str):
    global final_data, data_loading, data_loading_complete
    data_loading = True
    # Create tasks
    tasks = [search_companies.delay(letter) for letter in letters]
    print("Tasks created: ", len(tasks))
    start_time = time.time()
    count_hold = 0
    flag_list = [False] * len(tasks)

    company_data_tasks = []

    while True:
        for i, task in enumerate(tasks):
            if task.ready() and not flag_list[i]:
                flag_list[i] = True
                count_hold += 1
                print(f'{count_hold}/{len(tasks)} tasks are done', end='\r')
                for company in task.get():
                    company_data_tasks.append(get_company_data.delay(company))
                    
        if count_hold == len(tasks):
            break
        time.sleep(3)

    print(f"Found {len(company_data_tasks)} companies in {time.time() - start_time} seconds")

    temp = []
    # when all company data tasks are done, get the results and add them to final_data without using flag_list
    while True:
        if all([task.ready() for task in company_data_tasks]):
            for task in company_data_tasks:
                final_data.append(task.get())
            break
        time.sleep(3)

    data_loading = False
    data_loading_complete = True

def analysis(data):

    API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)

    
    # create dataframe from json
    # df_companies = pd.DataFrame(data["data"]["AllCompanies"])
    lst = []
    for d in data:
        lst.append(d["data"]['corporate'])
    # edit the description of the companies, include city and country in the description also say if the company is startup friendly or not
    
    df_companies = pd.DataFrame(lst)
    def edit_description(row):
        description = row["description"]
        city = row["hq_city"]
        country = row["hq_country"]
        startup_partners_count = row["startup_partners_count"]
        startup_themes = row["startup_themes"]
        new_description = f"{description}, which is based in {city} {country} with {startup_partners_count} startup partners. Among these startups,"
        # start_ups = startup_themes.split(",")
        for inds in startup_themes:
            new_description += f" {inds[1]} focuses on {inds[0]},"
        return new_description

    df_companies["description_extended"] = df_companies.apply(edit_description, axis=1)


    tqdm.pandas()

    def make_embed_text_fn(model):

        @retry.Retry(timeout=300.0)
        def embed_fn(text: str) -> list[float]:
            # Set the task_type to CLUSTERING.
            embedding = genai.embed_content(model=model,
                                            content=text,
                                            task_type="clustering")
            return embedding["embedding"]

        return embed_fn

    def create_embeddings(df):
        model = 'models/embedding-001'
        df['Embeddings'] = df['description_extended'].progress_apply(make_embed_text_fn(model))
        return df

    df_clustring = create_embeddings(df_companies)

    X = np.array(df_clustring['Embeddings'].tolist(), dtype=np.float32)

    tsne = TSNE(random_state=0, n_iter=1000)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['name'] = df_clustring['name']

    kmeans_model = KMeans(n_clusters=4, random_state=1, n_init='auto',max_iter=10000, algorithm='lloyd').fit(X)
    labels = kmeans_model.fit_predict(X)
    df_tsne['Cluster'] = labels

    # assign te clusters to df_companies
    df_companies['Cluster'] = df_tsne['Cluster']

    model = genai.GenerativeModel("models/gemini-1.0-pro")

    def get_chat_response(chat: ChatSession, prompt: str) -> str:
        text_response = []
        responses = chat.send_message(prompt, stream=True)
        for chunk in responses:
            text_response.append(chunk.text)
        return "".join(text_response)

    # Analysis prompts with chatbots

    description_prompt = "Generate a description representing this cluster by refering thier most common distinctive features(locations, industries , missions, visions, bussiness_models, their costumers). Companies in this cluster and their brief descriptions are:\n"
    descriptions = []
    title_prompt = "Generate a title for this cluster indicating its prominent feature, return only the title. No need to include the word 'cluster' in the title"
    titles = []
    # for each cluster, generate a chatbot 
    for cluster_id in range(4):
        cluster = df_companies[df_companies['Cluster'] == cluster_id]
        chat = model.start_chat()
        prompt = description_prompt + "\n".join([f"{row['name']}: {row['description_extended']}" for _, row in cluster.iterrows()])
        response = get_chat_response(chat, prompt)
        descriptions.append(response)
        prompt = title_prompt
        response = get_chat_response(chat, prompt)
        titles.append(response)
        # print(f"Cluster {cluster_id}:\n{response}")
    
    return titles, descriptions

def analyze_complete():
    global final_data, analyze_loading, analyze_loading_complete, analysis_data, description
    analyze_loading = True
    # analysis code here
    analysis_data, description = analysis(final_data)
    analyze_loading = False
    analyze_loading_complete = True

@app.get("/companies")
async def read_companies(background_tasks: BackgroundTasks):
    global final_data, data_loading, data_loading_complete
    if not data_loading and not data_loading_complete:
        letters = ascii_lowercase + " 0123456789öü"
        background_tasks.add_task(fetch_data, letters)
        return {"message": "Data is being fetched. Please wait."}
    elif data_loading:
        return {"message": "Data is still being fetched. Please wait."}
    else:
        return JSONResponse(content=final_data)
    
@app.get("/analyze")
async def analyze_data(background_tasks: BackgroundTasks):
    global final_data, data_loading, data_loading_complete, analyze_loading, analyze_loading_complete, analysis_flag
    # analysis_flag stands for when analysis data is posted one time, on the next time will analyze the data again
    
    if not data_loading and not data_loading_complete:
        letters = ascii_lowercase + " 0123456789öü"
        background_tasks.add_task(fetch_data, letters)
        return {"message": "Data is being fetched. Please wait."}
    elif data_loading:
        return {"message": "Data is still being fetched. Please wait."}
    else:
        if analysis_flag:
            background_tasks.add_task(analyze_complete)
            analysis_flag = False
            return {"message": "Analysis is being done. Please wait."}
        if not analyze_loading and not analyze_loading_complete:
            background_tasks.add_task(analyze_complete)
            return {"message": "Analysis is being done. Please wait."}
        elif analyze_loading:
            return {"message": "Analysis is being done. Please wait."}
        else:
            analysis_flag = True
            res = []
            for i in range(len(analysis_data)):
                res.append({"title": analysis_data[i], "description": description[i]})
            return JSONResponse(content=res)
    



    
