import os
import re

import joblib
import pandas as pd
from crewai import Agent, Task
from crewai_tools import SeleniumScrapingTool
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# https://python.langchain.com/v0.2/docs/integrations/tools/
# https://docs.crewai.com/tools/ScrapeWebsiteTool/

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_API"])

# load our job classification model
clf_loaded = joblib.load("/models/job_classifier_model.pkl")
vectorizer_loaded = joblib.load("/models/vectorizer.pkl")


scorer = Agent(
    role="Scorer",
    goal="""You score a job description as a potential good fit or not using a provided resume.
    Be realistic, don't be overly optimistic.
    Score the description using a 1-10 scale, where 1 is the lowest performance and 10 is the highest:
  Scale:
  1-3: Poor - The match between the job description and the resume is a terrible match.
  4-6: Average - The match between the job description and the resume has some good points but also has notable weaknesses.
  7-9: Good - The match between the job description and the resume is mostly effective with minor issues.
  10: Excellent - The match between the job description and the resume is exemplary with no apparent issues.
  Factors to Consider:
  Skills: How do the skills in the resume match up with the skills needed in the job description?
  Job Location: Does the expected job location (remote, on-site, hybrid, etc.) match what the candidate is looking for?
  Job Activites: Do the day-to-day activities of the job match the skills and expectations of the candidate and resume?
  Past Experience: Do the past experiences in the resume match up with the job description?
  """,
    backstory="You are an expert at matching up job seekers with jobs and scoring them on a scale of 1 to 10.",
    llm=llm,
    verbose=os.environ["CREW_VERBOSE_OUTPUT"],
    allow_delegation=False,
    max_iter=15,
)

df = pd.read_excel("/results/job-results.xlsx")

# make sure the column data types are appropriate
df["is_job_posting"] = df["is_job_posting"].astype(str)
df["relevance_score"] = df["relevance_score"].astype(str)

print(df.head())

print(df.info())

for index, row in tqdm(df.sample(n=3).iterrows()):
    print(f"URL: {row['url']}")
    print(f"Votes: {row['search_votes']}")

    # we have already processed this item
    if pd.isna(row["relevance_score"]):
        continue

    # don't process documents
    if row["url"].endswith(".pdf") or row["url"].endswith(".xml"):
        continue

    # don't process URLs with just an IP address, we want full domain names
    ip_pattern = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")
    if ip_pattern.findall(row["url"]):
        continue

    # don't process certain websites
    # arxiv.org - this is only for scholarly articles
    # archive.org - this is generally for old content
    # url= - usually redirect links
    bad_link = re.compile(r"arxiv.org|archive.org|url\=")
    if bad_link.findall(row["url"]):
        continue

    scrape_tool = SeleniumScrapingTool(website_url=row["url"])
    scraper = Agent(
        role="Scraper",
        goal="""You scrape information off websites.  Your goal is to pull relevant information.""",
        backstory="You are an expert at scraping and pulling information off websites.",
        llm=llm,
        verbose=os.environ["CREW_VERBOSE_OUTPUT"],
        allow_delegation=False,
        tools=[scrape_tool],
        max_iter=1,
    )
    task2 = Task(
        description="""Given a website URL, scrape the contents of the website.""",
        agent=scraper,
        expected_output="The contents of a website.",
    )
    scrape_results = task2.execute()

    # This job evaluation through the LLM doesn't work terribly well, so we use a separately trained
    # classification model.

    # jobevaluator = Agent(
    #     role='JobEvaluator',
    #     goal='''Given some text, evaluate if the information is an open job or internship posting or not.''',
    #     backstory='You are an expert job and internship posting evaluator.',
    #     llm=llm,
    #     verbose=os.environ['CREW_VERBOSE_OUTPUT'],
    #     allow_delegation=False,
    #     max_iter=5
    # )

    # task3 = Task(description=f'Given the following text, evaluate if this is an open job or internship posting or not. This information came from a web search. Provide a single value string of YES or NO.  Use YES if it is an open job position or internship and NO if it is not.\n### TEXT:\n {scrape_results}',
    #     agent=jobevaluator,
    #     expected_output='A single value of YES or NO.  YES if it is an open job position or internship and NO if it is not.')
    # evaluation_results = task3.execute().lower().replace("\'", "")

    X_test = vectorizer_loaded.transform([scrape_results])
    pred = clf_loaded.predict(X_test)

    evaluation_results = pred[0]

    print(f"Job Posting evalution: {str(evaluation_results)}")

    if evaluation_results:
        task4 = Task(
            description="Given the following skills and resume details, evaluate the provided job description and score it for relevance on a score from 1 to 10.  Be concise, only output a single number. \n### Skills:\n"
            + os.environ["SKILLS"]
            + "\n### Resume:\n"
            + os.environ["RESUME"]
            + "\n### Job Description:\n"
            + scrape_results,
            agent=scorer,
            expected_output="A single integer value score for relevance between 1 and 10.",
        )
        score = task4.execute().replace("'", "")
    else:
        score = "unknown"

    df.loc[df["url"] == row["url"], "is_job_posting"] = str(evaluation_results)
    df.loc[df["url"] == row["url"], "relevance_score"] = score


df.to_excel("/results/job-results.xlsx", index=False)
