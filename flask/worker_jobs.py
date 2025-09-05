import os, re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import joblib
from uuid import uuid4
from opensearchpy.exceptions import NotFoundError
from pydantic import BaseModel, Field
from typing import Annotated
from crewai import Agent, Task, Crew, Process, LLM



def extract_words_from_url(url: str) -> str:
    # Remove protocol (http://, https://) and query strings
    url = re.sub(r'^https?:\/\/', '', url) # remove http(s)://
    url = url.split('?')[0].split('#')[0] # remove query params / fragments

    # Split on common separators: / . - _ ~ etc.
    parts = re.split(r'[\./\-_\~]+', url)

    # Keep only alphabetic words (ignore numbers)
    words = [p for p in parts if p.isalpha()]

    # Join back into a single string
    return ' '.join(words)

def download_jobs():
    try:
        # load our job classification model
        clf_loaded = joblib.load("/models/job_classifier_model.pkl")
        vectorizer_loaded = joblib.load("/models/vectorizer.pkl")

        model = SentenceTransformer("all-MiniLM-L12-v2")

        client = OpenSearch(
            hosts=[{"host": "opensearch-node1", "port": "9200"}],
            use_ssl=False,
            verify_certs=False,
        )

        index_name = "job_evaluations_index"
        index_body = {
            "settings": {
                "analysis": {
                    "tokenizer": {"standard": {"type": "standard"}},
                    "analyzer": {
                        "default": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"],
                        }
                    },
                }
            },
            "mappings": {
                "properties": {
                    "persona_id": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "job_id": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "evaluation_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "evaluation_value": {"type": "integer"},
                    "keyword_match": {"type": "integer"},
                }
            },
        }

        # Create index if it doeesn't exist
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=index_body)

        query = {
            "size": 100,
            "query": {"term": {"content.keyword": {"value": "__EMPTY__"}}},
        }

        try:
            response = client.search(index="jobs_index", body=query)

            for i, doc in enumerate(response["hits"]["hits"]):
                try:
                    # Set up options for headless Chrome
                    # https://datawookie.dev/blog/2023/12/chrome-chromedriver-in-docker/
                    options = ChromeOptions()
                    options.headless = (
                        True  # Enable headless mode for invisible operation
                    )
                    options.add_argument(
                        "--window-size=1920,1200"
                    )  # Define the window size of the browser
                    options.add_argument("--headless")
                    options.add_argument("--disable-gpu")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    driver = Chrome(options=options)

                    driver.set_page_load_timeout(20)  # seconds

                    print(doc["_source"]["url"])
                    driver.get(doc["_source"]["url"])

                    # Parse the HTML with BeautifulSoup
                    soup = BeautifulSoup(driver.page_source, "html.parser")

                    # Extract the text, removing JavaScript and markup
                    scrape_results = soup.get_text(separator=" ", strip=True)

                    # encode and decode to remove bad bytes
                    scrape_results = scrape_results.encode(
                        "utf-8", errors="ignore"
                    ).decode("utf-8", errors="ignore")

                    MAX_CONTENT_LENGTH = 50000  # 50,000 characters is usually safe for OpenSearch as there is a document size limit
                    scrape_results = scrape_results[:MAX_CONTENT_LENGTH]

                    # Close the browser session cleanly to free up system resources
                    driver.quit()

                    X_test = vectorizer_loaded.transform([scrape_results])
                    pred = clf_loaded.predict(X_test)

                    classification_result = pred[0]

                    # print(f"Classifier Job Posting evalution: {str(classification_result)}")
                    status = doc["_source"]["status"]
                    if str(classification_result).lower() == "false":
                        status = "Not a Job"

                    # update the document with the entry
                    response = client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": scrape_results + " " + extract_words_from_url(doc["_source"]["url"]),
                                "date_downloaded": datetime.now().isoformat(),
                                "content_vector": model.encode(scrape_results).tolist(),
                                "status": status,
                            }
                        },
                    )
                except Exception as e:
                    print(f"Error updating URL {doc['_source']['url']}: {e}")
                    # update the document so we don't try and fail again
                    response = client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": "Error, unable to load content " + extract_words_from_url(doc["_source"]["url"]),
                                "date_downloaded": datetime.now().isoformat(),
                                "content_vector": model.encode(
                                    "Error, unable to load content"
                                ).tolist(),
                                "status": "Error Downloading",
                            }
                        },
                    )
                    continue
        except NotFoundError:
            print("No jobs index yet")

        print("Finished downloading jobs")
    except Exception as e:
        print(f"Error: {e}")


class Evaluation:
    def __init__(self, evaluation_name, evaluation_description, llm_task, persona_key):
        self.evaluation_name = evaluation_name
        self.evaluation_description = evaluation_description
        self.llm_task = llm_task
        self.persona_key = persona_key

    def runevaluation(self):
        try:
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            llm = LLM(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_API"])

            response = client.search(
                index="personas_index",
                # Query to match all documents
                body={"query": {"match_all": {}}},
                size=1000,
            )
            personas = []
            if response:
                personas = response["hits"]["hits"]

            response = client.search(
                index="jobs_index",
                # Query to match all documents
                body={"query": {"match_all": {}}},
                size=1000,
            )
            jobs = []
            if response:
                jobs = response["hits"]["hits"]

            for persona in personas:
                for job in jobs:
                    query = {
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "term": {
                                            "persona_id.keyword": {"value": persona["_id"]}
                                        }
                                    },
                                    {"term": {"job_id.keyword": {"value": job["_id"]}}},
                                    {"term": {"evaluation_name.keyword": {"value": self.evaluation_name}}}
                                ]
                            }
                        },
                    }
                    response = client.search(index="job_evaluations_index", body=query)
                    if response and response["hits"]["total"]["value"] == 0:

                        jobagent = Agent(
                            role="Evaluation",
                            goal="You are an expert at evaluating jobs.",
                            backstory="You are an expert at evaluating jobs.",
                            llm=llm,
                            verbose=True,
                            allow_delegation=False,
                            max_iter=10,
                        )

                        try:

                            class LocationEvaluation(BaseModel):
                                evaluationinteger: Annotated[
                                    int,
                                    Field(
                                        ge=1,
                                        le=100,
                                        description=self.evaluation_description,
                                    ),
                                ]

                            task1 = Task(
                                description=self.llm_task
                                + persona["_source"][self.persona_key]
                                + "\n### Job Posting:\n"
                                + job["_source"]["content"],
                                agent=jobagent,
                                expected_output=self.evaluation_description,
                                output_pydantic=LocationEvaluation,
                            )

                            crew = Crew(
                                agents=[jobagent],
                                tasks=[task1],
                                verbose=True,
                                process=Process.sequential,
                            )

                            result = crew.kickoff()

                            if result:
                                # print(f"Pydantic: {result.pydantic}")
                                # print(f"Tasks output: {result.tasks_output}")
                                # print(f"Token usage: {result.token_usage}")
                                # print(f"SCORE: {result["locationalignment"]}")

                                # count the number of times our item keyword(s) is found
                                keyword_item_match_number = 0
                                for check in persona["_source"][self.persona_key].split(
                                    ","
                                ):
                                    keyword_item_match_number += (
                                        job["_source"]["content"]
                                        .lower()
                                        .count(check.strip().lower())
                                    )

                                response = client.create(
                                    index="job_evaluations_index",
                                    id=str(uuid4()),
                                    body={
                                        "persona_id": persona["_id"],
                                        "job_id": job["_id"],
                                        "evaluation_name": self.evaluation_name,
                                        "evaluation_value": int(
                                            result["evaluationinteger"]
                                        ),
                                        "keyword_match": keyword_item_match_number,
                                    },
                                )
                        except Exception as e:
                            print(f"Error: {e}")
                            continue  # on in the for loop through jobs

        except Exception as e:
            print(f"Error: {e}")


def evaluate_location():
    eval = Evaluation(evaluation_name="Location",
                evaluation_description="An integer between 1 and 100 representing the match between the job and the desired location(s).",
                llm_task="""Read the following job description.  You're tasked with coming
                        up with a score value between 1-100 that quantifies the match of the job to a set of
                        desired locations for the position.  A low score denotes no match or a low match
                        to the desired location and a high score denotes a high match.
                        If the job doesn't explicitly mention a physical location, give it a score of 1.
                        The desired location(s) for the position are one or more of the following: """,
                persona_key="desired_location"
            )
    eval.runevaluation()

def evaluate_skills():
    eval = Evaluation(evaluation_name="Skills",
                evaluation_description="An integer between 1 and 100 representing the match between the job and the skillsets of the potential employee.",
                llm_task="""Read the following job description.  You're tasked with coming
                    up with a score value between 1-100 that quantifies the match of the job to a set of
                    employee skills.  A low score denotes no match or a low match
                    to the desired skills and a high score denotes a high match.
                    The potential employee skills for the position are the following: """,
                persona_key="skills",
            )
    eval.runevaluation()

def evaluate_include_keywords():
    eval = Evaluation(evaluation_name="Include Keywords",
                evaluation_description="An integer between 1 and 100 representing the match between the job and the keywords of the potential employee.",
                llm_task="""Read the following job description.  You're tasked with coming
                    up with a score value between 1-100 that quantifies the match of the job to a set of
                    employee keywords.  A low score denotes no match or a low match
                    to the desired keywords and a high score denotes a high match.
                    The potential keywords for the position are the following: """,
                persona_key="match_keywords",
            )
    eval.runevaluation()

def evaluate_exclude_keywords():
    eval = Evaluation(evaluation_name="Exclude Keywords",
                evaluation_description="An integer between 1 and 100 representing the match between the job and the excluded keywords of the potential employee.",
                llm_task="""Read the following job description.  You're tasked with coming
                    up with a score value between 1-100 that quantifies the match of the job to a set of
                    employee keywords that we don't want to show up.  A low score denotes a high match to these unwanted keywords and
                    and a high score denotes a low match where no or few keywords shows up.
                    The potential excluded keywords for the position are the following: """,
                persona_key="exclude_keywords",
            )
    eval.runevaluation()

def evaluate_resume_match():
    eval = Evaluation(evaluation_name="Resume Match",
                evaluation_description="An integer between 1 and 100 representing the match between the job and the resume of the potential employee.",
                llm_task="""Read the following job description.  You're tasked with coming
                    up with a score value between 1-100 that quantifies the match of the job to the resume text
                    of a potential employee.  A low score denotes a low match to the resume and
                    and a high score denotes a high match where there is strong alignment.
                    The resume of the potential employee is as follows: """,
                persona_key="resume",
            )
    eval.runevaluation()