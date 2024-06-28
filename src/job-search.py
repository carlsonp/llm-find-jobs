import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from crewai import Agent, Task
from crewai_tools import ScrapeWebsiteTool, SeleniumScrapingTool
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.utilities import SearxSearchWrapper
from langchain_openai import ChatOpenAI

# https://python.langchain.com/v0.2/docs/integrations/tools/
# https://docs.crewai.com/tools/ScrapeWebsiteTool/

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_API"])


scrapewebsite_tool = ScrapeWebsiteTool()
duckduckgosearchrun_tool = DuckDuckGoSearchRun()
duckduckgosearchresults_tool = DuckDuckGoSearchResults()
searx_tool = SearxSearchWrapper(searx_host=os.environ["SEARX_HOST"])
seleniumscraping_tool = SeleniumScrapingTool()

# https://github.com/langchain-ai/langchain/issues/855#issuecomment-1452900595
# make sure to add this setting to the searx settings config, also turn off bot protection
# search:
#   formats:
#     - html
#     - json


searcher = Agent(
    role="Search",
    goal="""You are an expert at finding jobs.  You utilize all available tools to find jobs that match the search criteria.""",
    backstory="You are an expert at finding jobs.",
    llm=llm,
    verbose=os.environ["CREW_VERBOSE_OUTPUT"],
    allow_delegation=False,
    max_iter=10,
)

task1 = Task(
    description="""Read the following set of skills and resume information.  You're looking for
             open job positions that best match those skills and resume.  You're looking for a full-time position,
             part-time position, or internship.  Leverage popular websites for careers and job postings
             such as indeed.com, linkedin.com, and glassdoor.com.  Remote jobs as opposed to in-person are preferred.
             You are not interested in general articles
             about finding jobs in this area but want to develop specific search queries that focus soley
             on results that will yield open job positions.  You want the highest possible salary
             for the open position.  Use only the English language in the queries.  Come up with appropriate
             search queries for search engines and job search boards that best fit the skills and resume. Provide
             a list of 20 semi-colon separated search queries. Do not use a numbered list.\n### Skills:\n"""
    + os.environ["SKILLS"]
    + "\n### Resume:\n"
    + os.environ["RESUME"],
    agent=searcher,
    expected_output="A list of 20 semi-colon separated search queries for open job positions.",
)
search_queries = task1.execute()

search_queries = search_queries.split(";")
augmented_queries = []
for query in search_queries:
    augmented_queries.append(f"site: linkedin.com {query}")
    augmented_queries.append(f"site: indeed.com {query}")
    augmented_queries.append(f"site: glassdoor.com {query}")
    augmented_queries.append(f"site: ziprecruiter.com {query}")
    augmented_queries.append(f"site: monster.com {query}")
    augmented_queries.append(f"site: careerbuilder.com {query}")
search_queries = search_queries + augmented_queries


if Path("/results/job-results.xlsx").is_file():
    df = pd.read_excel("job-results.xlsx")
else:
    df = pd.DataFrame(
        columns=[
            "search_votes",
            "url",
            "description",
            "is_job_posting",
            "relevance_score",
        ]
    )

# make sure the column data types are appropriate
df["is_job_posting"] = df["is_job_posting"].astype(str)
df["relevance_score"] = df["relevance_score"].astype(str)

print(df.head())

print(df.info())


url_dict = defaultdict(int)

with open("/results/search-queries.json", "w") as fout:
    json.dump(search_queries, fout, indent=4)

for query in search_queries:
    print(f"Query: {query}")
    results_month = searx_tool.results(
        query,
        num_results=20,
        time_range="month",  # day, month, year
    )
    for res in results_month:
        if "link" in res:
            url_dict[res["link"]] += 1

    results_day = searx_tool.results(
        query,
        num_results=20,
        time_range="day",  # day, month, year
    )
    for res in results_day:
        if "link" in res:
            url_dict[res["link"]] += 1

# sort from largest to smallest number of search hits
url_dict = dict(sorted(url_dict.items(), key=lambda item: item[1], reverse=True))

for url, votes in url_dict.items():
    # do some basic cleanup on the URL
    url = url.replace('"', "")
    # add the URL if we don't already have it in our list
    if url not in df["url"].values:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "search_votes": votes,
                            "url": url,
                            "description": "",
                            "is_job_posting": "",
                            "relevance_score": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )


# pip install openpyxl
df.to_excel("/results/job-results.xlsx", index=False)
