import time
import requests
from string import ascii_lowercase
# from celery import Celery
from .app import celery_app

@celery_app.task(name="search_companies")
def search_companies(tosearch):
    url = "https://ranking.glassdollar.com/graphql"
    payload = {
        "operationName": "GIMGetSearchResults",
        "variables": {
            "where": {
                "query": tosearch
            }
        },
        "query": "query GIMGetSearchResults($where: searchBarWhere) {\n  searchBar(where: $where)\n}\n"
    }
    response = requests.post(url, json=payload)
    data = response.json()

    if len(data['data']['searchBar']) == 0:
        return []
    elif len(data['data']['searchBar']['corporates']) >= 5:
        ids = []
        for next_letter in ascii_lowercase + " 0123456789öü.-":
            ids += search_companies(tosearch + next_letter)
        return ids
    else:
        ids = [company['id'] for company in data['data']['searchBar']['corporates']]
        return ids
    
@celery_app.task(name="get_company_data")
def get_company_data(id):
    payload = {
        "variables": {
            "id": id
        },
        "query": "query ($id: String!) {\n  corporate(id: $id) {\n    id\n    name\n    description\n    logo_url\n    hq_city\n    hq_country\n    website_url\n    linkedin_url\n    twitter_url\n    startup_partners_count\n    startup_partners {\n      master_startup_id\n      company_name\n      logo_url: logo\n      city\n      website\n      country\n      theme_gd\n      __typename\n    }\n    startup_themes\n    startup_friendly_badge\n    __typename\n  }\n}\n"
    }
    
    url = "https://ranking.glassdollar.com/graphql"
    response = requests.post(url, json=payload)
    return response.json()