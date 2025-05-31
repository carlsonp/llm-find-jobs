import threading
import os
import re
import hashlib
from datetime import datetime
from collections import defaultdict
from dateutil import parser

from crewai import Agent, Task, Crew, Process, LLM
import humanize
from flask_compress import Compress
from flask_socketio import SocketIO
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConflictError
from sentence_transformers import SentenceTransformer

from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.utilities import SearxSearchWrapper

from download_jobs import download_jobs

from uuid import uuid4

from redis import Redis
from rq import Queue

from pydantic import BaseModel, Field
from typing import List

from flask import Flask, redirect, render_template, request, url_for

q = Queue(connection=Redis(host='valkey', port=6379))


def create_app():
    app = Flask(__name__, instance_relative_config=True, static_folder="/static/")
    Compress(app)
    socketio = SocketIO(app)

    @app.route("/")
    def homepage():
        try:
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            response = None
            personas = []
            if client.indices.exists(index="personas_index"):
                response = client.search(
                    index="personas_index",
                    # Query to match all documents
                    body={"query": {"match_all": {}}},
                )
            if response:
                personas = response["hits"]["hits"]
            response = None

            query = {
                "size": 50,
                "_source": ["id", "url"],
                "query": {"match": {"status": "Favorite"}},
            }

            favorites = []
            if client.indices.exists(index="jobs_index"):
                response = client.search(index="jobs_index", body=query)
            if response:
                favorites = response["hits"]["hits"]
            response = None

            query = {
                "size": 0,
                "aggs": {
                    "status_counts": {"terms": {"field": "status.keyword", "size": 100}}
                },
            }

            buckets = []
            if client.indices.exists(index="jobs_index"):
                response = client.search(index="jobs_index", body=query)
            if response:
                buckets = response["aggregations"]["status_counts"]["buckets"]
            response = None

            return render_template(
                "index.html", personas=personas, favorites=favorites, buckets=buckets
            )
        except Exception as e:
            app.logger.error(e)
            return "Failure on homepage"

    @app.route("/health")
    def health():
        return "Ok"

    @app.route("/create_persona")
    def create_persona():
        return render_template("create_persona.html")

    @app.route("/post_create_persona", methods=["POST"])
    def post_create_persona():
        try:
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

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
                        "name": {"type": "text"},
                        "resume": {"type": "text"},
                        "skills": {"type": "text"},
                        "match_keywords": {"type": "text"},
                        "exclude_keywords": {"type": "text"},
                        "date_created": {"type": "date"},
                    }
                },
            }

            # Create index if it doesn't exist
            if not client.indices.exists(index="personas_index"):
                client.indices.create(index="personas_index", body=index_body)

            # add our document
            client.index(
                index="personas_index",
                id=str(uuid4()),
                body={
                    "name": request.form["name"],
                    "resume": request.form["resume"],
                    "skills": request.form["skills"],
                    "match_keywords": request.form["match_keywords"],
                    "exclude_keywords": request.form["exclude_keywords"],
                    "date_created": datetime.now().isoformat(),
                },
                refresh=True,
            )

            return redirect(url_for("homepage"))

        except Exception as e:
            app.logger.error(e)
            return "Failure on creation of persona"

    @app.route("/run_job_search")
    def run_job_search():
        try:
            return render_template("run_job_search.html")

        except Exception as e:
            app.logger.error(e)
            return "Failure in run job search"

    @app.route("/post_job_search", methods=["POST"])
    def post_job_search():
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            DuckDuckGoSearchRun()
            DuckDuckGoSearchResults()
            searx_tool = SearxSearchWrapper(searx_host=os.environ["SEARX_HOST"])

            # Define the index settings
            index_name = "jobs_index"
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
                    },
                    "index": {"knn": True},  # Enable KNN search for vectors
                },
                "mappings": {
                    "properties": {
                        "url": {"type": "text"},
                        "content": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "content_vector": {
                            "type": "knn_vector",
                            "dimension": 384,  # Dimensions of the embedding (for 'all-MiniLM-L12-v2')
                        },
                        "status": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "date_added": {"type": "date"},
                        "date_downloaded": {"type": "date"},
                    }
                },
            }

            # Create index if it doeesn't exist
            if not client.indices.exists(index=index_name):
                client.indices.create(index=index_name, body=index_body)

            url_dict = defaultdict(int)

            # search searx
            results_month = searx_tool.results(
                request.form["search-query"],
                num_results=20,
                time_range="month",  # day, month, year
            )
            for res in results_month:
                if "link" in res:
                    url_dict[res["link"]] += 1

            results_day = searx_tool.results(
                request.form["search-query"],
                num_results=20,
                time_range="day",  # day, month, year
            )
            for res in results_day:
                if "link" in res:
                    url_dict[res["link"]] += 1
            # search DuckDuckGo with retry and exponential backoff
            # rate limiting generally resets after about 1 minute?
            # max_retries = 5
            # retry_delay = 10  # start with this delay in seconds
            # attempt = 0

            # while attempt < max_retries:
            #     try:
            #         results = duckduckgosearchresults_tool.invoke(request.form['search-query'])
            #         for res in results:
            #             if "link" in res:
            #                 url_dict[res["link"]] += 1
            #         break  # success, exit retry loop
            #     except Exception as e:
            #         print(f"Error with DuckDuckGo search: {e}")
            #         attempt += 1
            #         if attempt == max_retries:
            #             app.logger.info(f"Failed DuckDuckGo search after {max_retries} attempts for query: {request.form['search-query']}")
            #             break
            #         sleep_time = retry_delay * (2 ** (attempt - 1))  # exponential backoff
            #         print(f"Retrying in {sleep_time} seconds...")
            #         time.sleep(sleep_time)

            for url, hits in url_dict.items():
                # do some basic cleanup on the URL
                url = url.replace('"', "")

                # don't process documents
                if (
                    url.endswith(".pdf")
                    or url.endswith(".xml")
                    or url.endswith(".docx")
                ):
                    continue

                # don't process URLs with just an IP address, we want full domain names
                ip_pattern = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")
                if ip_pattern.findall(url):
                    continue

                # don't process certain websites
                # arxiv.org - this is only for scholarly articles
                # archive.org - this is generally for old content
                # url= - usually redirect links
                bad_link = re.compile(r"arxiv.org|archive.org|url\=")
                if bad_link.findall(url):
                    continue

                # add the URL if we don't already have it in our database
                # store the md5 hash of the URL
                try:
                    client.create(
                        index=index_name,
                        id=hashlib.md5(url.encode()).hexdigest(),
                        body={
                            "url": url,
                            "content": "__EMPTY__",
                            "status": "Unknown",
                            "date_added": datetime.now().isoformat(),
                        },
                    )
                except ConflictError:
                    pass  # it's already in the database

            # perform a refresh on our index
            client.indices.refresh(index=index_name)

            # enqueue a job to redis queue to download the jobs we just added
            q.enqueue(download_jobs)

            return render_template(
                "post_job_search.html", query=request.form["search-query"]
            )

        except Exception as e:
            app.logger.error(e)
            return "Failure in post job search"

    @app.route("/generate_search_terms/<persona_id>", methods=["GET"])
    def generate_search_terms(persona_id):
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            llm = LLM(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_API"])

            searcher = Agent(
                role="Search",
                goal="""You are an expert at finding jobs.  You utilize all available tools to find jobs that match the search criteria.""",
                backstory="You are an expert at finding jobs and internships.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                max_iter=10,
            )

            class SearchList(BaseModel):
                search: List[str] = Field(
                    ...,
                    description="List of search string queries that match best with the skills and resume.",
                )

            response = client.search(
                index="personas_index",
                body={"query": {"term": {"_id": {"value": persona_id}}}},
            )
            search_terms = []
            if response:
                person = response["hits"]["hits"][0]

                app.logger.info(person)

                task1 = Task(
                    description="""Read the following set of skills and resume information.  You're looking for
                            open job positions that best match the skills and resume.  You're looking for a full-time position,
                            part-time position, or internship.  Leverage popular websites for careers and job postings
                            such as indeed.com and linkedin.com.
                            Find only jobs that are active and current.
                            You are not interested in general articles
                            about finding jobs in this area but want to develop specific search queries that focus soley
                            on results that will yield open job positions.  Use only the English language in the queries.  Come up with appropriate
                            search queries for search engines and job search boards that best fit the skills and resume.
                            Do not come up with search queries using the following terms: """
                    + person["_source"]["exclude_keywords"]
                    + """.  Provide
                            a list of 20 separated search queries. Do not use a numbered list.\n### Skills:\n"""
                    + person["_source"]["skills"]
                    + "\n### Resume:\n"
                    + person["_source"]["resume"],
                    agent=searcher,
                    expected_output="A list of 20 separated search queries for open job positions.",
                    output_pydantic=SearchList,
                )

                crew = Crew(
                    agents=[searcher],
                    tasks=[task1],
                    verbose=True,
                    process=Process.sequential,
                )

                result = crew.kickoff()

                app.logger.info(f"Pydantic: {result.pydantic}")
                app.logger.info(f"Tasks output: {result.tasks_output}")
                app.logger.info(f"Token usage: {result.token_usage}")

                search_terms = result["search"]

            return render_template(
                "generate_search_terms.html", search_terms=search_terms
            )
        except Exception as e:
            app.logger.error(e)
            return "Failure in generate search terms"
        
    @app.route("/tasks")
    def tasks():
        return render_template(
            "tasks.html", tasks=q.jobs
        )

    @app.route("/search", methods=["POST"])
    def search():
        try:
            lexical_results = []
            semantic_results = []
            response = None

            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            app.logger.info(request.form.getlist("statuses"))
            query = {
                "size": 10,
                "_source": ["id", "url", "status"],
                "query": {
                    "bool": {
                        "must": {"match": {"content": request.form["search"]}},
                        "filter": {
                            "terms": {
                                "status.keyword": request.form.getlist("statuses")
                            }
                        },
                    }
                },
            }
            if client.indices.exists(index="jobs_index"):
                response = client.search(index="jobs_index", body=query)
            else:
                lexical_results = []
            if response:
                lexical_results = response["hits"]["hits"]

            model = SentenceTransformer("all-MiniLM-L12-v2")

            # Generate the query embedding
            query_vector = model.encode(request.form["search"]).tolist()

            # Define the search query in OpenSearch to use the KNN (nearest neighbor) search
            search_query = {
                "_source": ["id", "url", "status"],
                "query": {
                    "bool": {
                        "must": {
                            "knn": {
                                "content_vector": {
                                    "vector": query_vector,
                                    "k": 10,  # Number of similar documents to retrieve
                                }
                            }
                        },
                        "filter": {
                            "terms": {
                                "status.keyword": request.form.getlist("statuses")
                            }
                        },
                    }
                },
            }

            # Execute the search
            if client.indices.exists(index="jobs_index"):
                response = client.search(index="jobs_index", body=search_query)
            else:
                semantic_results = []

            if response:
                semantic_results = response["hits"]["hits"]

            return render_template(
                "results.html",
                searx_host=os.environ["SEARX_HOST"],
                selected_statuses=request.form.getlist("statuses"),
                search_term=request.form["search"],
                lexical_results=lexical_results,
                semantic_results=semantic_results,
            )

        except Exception as e:
            app.logger.error(e)
            return "Failure searching"

    @app.route("/details/<document_id>", methods=["GET"])
    def details(document_id):
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            document = client.get(index="jobs_index", id=document_id)

            return render_template("details.html", document=document)
        except Exception as e:
            app.logger.error(e)
            return "Failure in showing document details"

    @app.route("/updatestatus/<document_id>/<status>/", methods=["GET"])
    def updatestatus(document_id, status):
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            if status == "NotJob":
                insert_status = "Not a Job"
            elif status == "NotRelevant":
                insert_status = "Not Relevant"
            elif status == "Favorite":
                insert_status = "Favorite"
            elif status == "Applied":
                insert_status = "Applied"
            elif status == "Rejected":
                insert_status = "Rejected"
            elif status == "NoLongerAvailable":
                insert_status = "No Longer Available"

            if insert_status:
                response = client.update(
                    index="jobs_index",
                    id=document_id,
                    body={"doc": {"status": insert_status}},
                )

            return redirect(f"/details/{document_id}")
        except Exception as e:
            app.logger.error(e)
            return "Failure in updating job status"

    @app.route("/list_recent_jobs")
    def list_recent_jobs():
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            # Read pagination params from query string
            search_after_date = request.args.get("search_after_date")
            search_after_id = request.args.get("search_after_id")

            size = 20  # items per page
            query = {
                "size": size,
                "_source": ["id", "url", "status", "date_downloaded"],
                "sort": [
                    {"date_downloaded": "desc"},
                    {"_id": "desc"},  # tiebreaker for stable sort
                ],
            }

            # Add search_after if values are passed
            if search_after_date and search_after_id:
                query["search_after"] = [search_after_date, search_after_id]

            if client.indices.exists(index="jobs_index"):
                response = client.search(index="jobs_index", body=query)
                jobs = response["hits"]["hits"]
            else:
                jobs = []

            next_search_after = None
            if jobs:
                last = jobs[-1]
                last_date = last["_source"]["date_downloaded"]
                last_id = last["_id"]
                next_search_after = {"date": last_date, "id": last_id}

            for job in jobs:
                downloaded_at = parser.isoparse(job["_source"]["date_downloaded"])
                job["_source"]["date_downloaded"] = humanize.naturaltime(
                    datetime.now(downloaded_at.tzinfo) - downloaded_at
                )

            return render_template(
                "list_recent_jobs.html", jobs=jobs, next_search_after=next_search_after
            )

        except Exception as e:
            app.logger.error(e)
            return "Failure in listing all jobs"

    @app.route("/delete_persona/<persona_id>/", methods=["GET"])
    def delete_persona(persona_id):
        try:
            # Connect to OpenSearch
            client = OpenSearch(
                hosts=[{"host": "opensearch-node1", "port": "9200"}],
                use_ssl=False,
                verify_certs=False,
            )

            response = client.delete(index="personas_index", id=persona_id)

            # perform a refresh on our index
            client.indices.refresh(index="personas_index")

            return redirect("/")
        except Exception as e:
            app.logger.error(e)
            return "Failure in deleting persona"

    return app
