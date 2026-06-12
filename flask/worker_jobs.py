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
from redis import Redis
from rq import Queue

EVALUATIONS = {
    "Location": {
        "description": "An integer between 1 and 100 representing the match between the job and the desired location(s).",
        "llm_task": """
    Read the following job description.

    You're tasked with coming up with a score value between 1-100 that
    quantifies the match of the job to a set of desired locations for the
    position.

    A low score denotes no match or a low match to the desired location.
    A high score denotes a high match.

    If the job doesn't explicitly mention a physical location,
    give it a score of 1.

    The desired location(s) for the position are:
    """,
        "persona_key": "desired_location",
    },
    "Skills": {
        "description": "An integer between 1 and 100 representing the match between the job and the skillsets of the potential employee.",
        "llm_task": """
    Read the following job description.

    You're tasked with coming up with a score value between 1-100 that
    quantifies the match of the job to a set of employee skills.

    The employee skills are:
    """,
        "persona_key": "skills",
    },
    "Include Keywords": {
        "description": "An integer between 1 and 100 representing the match between the job and the keywords of the potential employee.",
        "llm_task": """
    Read the following job description.

    You're tasked with coming up with a score value between 1-100 that
    quantifies the match of the job to a set of employee keywords.

    The keywords are:
    """,
        "persona_key": "match_keywords",
    },
    "Exclude Keywords": {
        "description": "An integer between 1 and 100 representing the inverse match between the job and excluded keywords.",
        "llm_task": """
    Read the following job description.

    You're tasked with coming up with a score value between 1-100.

    Low score = many excluded keywords found.
    High score = few or none found.

    The excluded keywords are:
    """,
        "persona_key": "exclude_keywords",
    },
    "Resume Match": {
        "description": "An integer between 1 and 100 representing the match between the job and the resume.",
        "llm_task": """
    Read the following job description.

    You're tasked with coming up with a score value between 1-100
    representing how well the resume matches the job.

    The resume is:
    """,
        "persona_key": "resume",
    },
}


def get_client():
    return OpenSearch(
        hosts=[{"host": "opensearch-node1", "port": "9200"}],
        use_ssl=False,
        verify_certs=False,
    )


def get_llm():
    return LLM(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_API"],
    )


def extract_words_from_url(url: str) -> str:
    # Remove protocol (http://, https://) and query strings
    url = re.sub(r"^https?:\/\/", "", url)  # remove http(s)://
    url = url.split("?")[0].split("#")[0]  # remove query params / fragments

    # Split on common separators: / . - _ ~ etc.
    parts = re.split(r"[\./\-_\~]+", url)

    # Keep only alphabetic words (ignore numbers)
    words = [p for p in parts if p.isalpha()]

    # Join back into a single string
    return " ".join(words)


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
                    options.add_argument(
                        "--window-size=1920,1200"
                    )  # Define the window size of the browser
                    options.add_argument("--headless=new")
                    options.add_argument("--disable-gpu")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--enable-javascript")
                    options.add_argument(
                        "--disable-blink-features=AutomationControlled"
                    )
                    options.add_argument(
                        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    options.add_argument("--lang=en-US,en")
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
                    client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": scrape_results
                                + " "
                                + extract_words_from_url(doc["_source"]["url"]),
                                "date_downloaded": datetime.now().isoformat(),
                                "content_vector": model.encode(scrape_results).tolist(),
                                "status": status,
                            }
                        },
                    )
                except Exception as e:
                    print(f"Error updating URL {doc['_source']['url']}: {e}")
                    # update the document so we don't try and fail again
                    client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": "Error, unable to load content "
                                + extract_words_from_url(doc["_source"]["url"]),
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


def schedule_evaluation(evaluation_name):
    client = get_client()

    q = Queue(
        connection=Redis(
            host="valkey",
            port=6379,
        )
    )

    personas = client.search(
        index="personas_index",
        body={"query": {"match_all": {}}},
        size=1000,
    )["hits"]["hits"]

    jobs = client.search(
        index="jobs_index",
        body={"query": {"match_all": {}}},
        size=1000,
    )["hits"]["hits"]

    for persona in personas:
        for job in jobs:

            existing = client.search(
                index="job_evaluations_index",
                body={
                    "size": 0,
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"persona_id.keyword": persona["_id"]}},
                                {"term": {"job_id.keyword": job["_id"]}},
                                {"term": {"evaluation_name.keyword": evaluation_name}},
                            ]
                        }
                    },
                },
            )

            if existing["hits"]["total"]["value"] > 0:
                continue

            rqjob = q.enqueue(
                run_single_evaluation,
                evaluation_name,
                persona["_id"],
                job["_id"],
                job_timeout="15m",
            )

            rqjob.meta.update(
                {
                    "persona_name": persona["_source"]["name"],
                    "job_url": job["_source"]["url"],
                }
            )
            rqjob.save_meta()


def run_single_evaluation(evaluation_name, persona_id, job_id):
    config = EVALUATIONS[evaluation_name]

    client = get_client()
    llm = get_llm()

    persona = client.get(
        index="personas_index",
        id=persona_id,
    )

    job = client.get(
        index="jobs_index",
        id=job_id,
    )

    existing = client.search(
        index="job_evaluations_index",
        body={
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"persona_id.keyword": persona_id}},
                        {"term": {"job_id.keyword": job_id}},
                        {"term": {"evaluation_name.keyword": evaluation_name}},
                    ]
                }
            },
        },
    )

    if existing["hits"]["total"]["value"] > 0:
        return

    jobagent = Agent(
        role="Evaluation",
        goal="You are an expert at evaluating jobs.",
        backstory="You are an expert at evaluating jobs.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )

    class EvaluationResult(BaseModel):
        evaluationinteger: Annotated[
            int,
            Field(
                ge=1,
                le=100,
                description=config["description"],
            ),
        ]

    task = Task(
        description=(
            config["llm_task"]
            + persona["_source"][config["persona_key"]]
            + "\n\n### Job Posting:\n"
            + job["_source"]["content"]
        ),
        agent=jobagent,
        expected_output=config["description"],
        output_pydantic=EvaluationResult,
    )

    crew = Crew(
        agents=[jobagent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
    )

    result = crew.kickoff()

    keyword_match_count = 0

    persona_value = persona["_source"].get(
        config["persona_key"],
        "",
    )

    if persona_value:
        for keyword in persona_value.split(","):
            keyword_match_count += (
                job["_source"]["content"].lower().count(keyword.strip().lower())
            )

    client.create(
        index="job_evaluations_index",
        id=str(uuid4()),
        body={
            "persona_id": persona_id,
            "job_id": job_id,
            "evaluation_name": evaluation_name,
            "evaluation_value": int(result["evaluationinteger"]),
            "keyword_match": keyword_match_count,
        },
    )
