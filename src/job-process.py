import os
import string

import joblib
import pandas as pd
import numpy as np
import spacy
from scipy.spatial.distance import cosine
from spacy.tokens import Doc
from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
from tqdm import tqdm

# No scientific notation
np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})


def remove_punctuation(text):
    # keep spaces
    punctuation = string.punctuation.replace(" ", "")
    return text.translate(str.maketrans(punctuation, " " * len(punctuation)))


# https://spacy.io/models/en#en_core_web_lg
nlp = spacy.load("en_core_web_lg")
# https://stackoverflow.com/questions/57231616/valueerror-e088-text-of-length-1027203-exceeds-maximum-of-1000000-spacy
# ValueError: [E088] Text of length 1218251 exceeds maximum of 1000000. The parser and NER models require roughly 1GB of temporary memory
# per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's
#  probably safe to increase the `nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.
nlp.max_length = 5000000

doc1 = nlp(remove_punctuation(os.environ["SKILLS"] + "\n" + os.environ["RESUME"]))
# Remove stop words
filtered_tokens = [token for token in doc1 if not token.is_stop]
filtered_doc1 = Doc(doc1.vocab, words=[token.text for token in filtered_tokens])
vector1 = filtered_doc1.vector


# https://python.langchain.com/v0.2/docs/integrations/tools/
# https://docs.crewai.com/tools/ScrapeWebsiteTool/

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_API"])

# load our job classification model
clf_loaded = joblib.load("/models/job_classifier_model.pkl")
vectorizer_loaded = joblib.load("/models/vectorizer.pkl")


scorer = Agent(
    role="Scorer",
    goal="""You score and compare a set of skills a person has compared to a job description.
    Be realistic, don't be overly optimistic.
  """,
    backstory="You are an expert at evaluating skills in employment.",
    llm=llm,
    verbose=os.environ["CREW_VERBOSE_OUTPUT"],
    allow_delegation=False,
    max_iter=5,
)

summarizer = Agent(
    role="Summarizer",
    goal="""Your goal is summarize the provided website contents into one sentence.""",
    backstory="You are an expert at summarizing information from websites.",
    llm=llm,
    verbose=os.environ["CREW_VERBOSE_OUTPUT"],
    allow_delegation=False,
    max_iter=5,
)

df = pd.read_excel("/results/job-results.xlsx")

# make sure the column data types are appropriate
df["description"] = df["description"].astype(str)
df["is_job_posting"] = df["is_job_posting"].astype(str)
df["llm_is_job_posting"] = df["llm_is_job_posting"].astype(str)
df["relevance_score"] = df["relevance_score"].astype(str)

print(df.head())

print(df.info())

for index, row in tqdm(df.iterrows(), total=len(df)):

    # we have already processed this item, continue to the next one
    if (
        not pd.isna(row["relevance_score"])
        and row["relevance_score"] != "nan"
        and not pd.isnull(row["relevance_score"])
    ):
        continue

    # print(f"Relevance score: {row['relevance_score']}")

    print(f"URL: {row['url']}")
    print(f"Votes: {row['search_votes']}")

    # Set up options for headless Chrome
    # https://datawookie.dev/blog/2023/12/chrome-chromedriver-in-docker/
    options = ChromeOptions()
    options.headless = True  # Enable headless mode for invisible operation
    options.add_argument(
        "--window-size=1920,1200"
    )  # Define the window size of the browser
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = Chrome(options=options)

    driver.set_page_load_timeout(20)  # seconds

    try:
        driver.get(row["url"])
    except TimeoutException:
        print(f"Timeout fetching: {row['url']}")
        df.loc[df["url"] == row["url"], "relevance_score"] = "unknown"
        continue

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extract the text, removing JavaScript and markup
    scrape_results = soup.get_text().lower()

    # Close the browser session cleanly to free up system resources
    driver.quit()

    # remove punctuation
    scrape_results = remove_punctuation(scrape_results)

    # print(f"Scrape results: {scrape_results}")

    task2 = Task(
        description="""Summarize the following website contents into one sentence and one sentence only.  Be concise.\n### Website:\n"""
        + scrape_results,
        agent=summarizer,
        expected_output="A one sentence summary of the website.",
    )
    df.loc[df["url"] == row["url"], "description"] = task2.execute()

    jobevaluator = Agent(
        role="JobEvaluator",
        goal="""Given some text, evaluate if the information is an open job or internship posting or not.""",
        backstory="You are an expert job and internship posting evaluator.",
        llm=llm,
        verbose=os.environ["CREW_VERBOSE_OUTPUT"],
        allow_delegation=False,
        max_iter=5,
    )

    task3 = Task(
        description=f"Given the following text, evaluate if this is an open job or internship posting or not. This information came from a web search. Provide a single value string of YES or NO.  Use YES if it is an open job position or internship and NO if it is not.\n### TEXT:\n {scrape_results}",
        agent=jobevaluator,
        expected_output="A single value of YES or NO.  YES if it is an open job position or internship and NO if it is not.",
    )
    llm_evaluation_results = task3.execute().lower().replace("'", "")

    # print(f"LLM Job Posting evaluation: {str(llm_evaluation_results)}")

    X_test = vectorizer_loaded.transform([scrape_results])
    pred = clf_loaded.predict(X_test)

    new_text = scrape_results.split()  # Split text into words
    vocabulary = vectorizer_loaded.vocabulary_  # Get the vectorizer's vocabulary

    # Find words not in the vocabulary
    # unseen_words = [word for word in new_text if word not in vocabulary]
    # print(f"Unseen Words: {unseen_words}")

    # and words in the vocabulary
    # seen_words = [word for word in new_text if word in vocabulary]
    # print(f"Seen Words: {seen_words}")

    pred_proba = clf_loaded.predict_proba(X_test)
    # print(f"Predicted Probabilities: {pred_proba}")

    evaluation_results = pred[0]

    # print(f"Classifier Job Posting evalution: {str(evaluation_results)}")

    task4 = Task(
        description="Given the following set of Skills a person has, compare them to the provided Job Description.  Evaluate the amount of overlap of skills and relevance to the job description.  Provide a single value of low relevance, medium relevance, or high relevance based on your evaluation. \n### Skills:\n"
        + os.environ["SKILLS"]
        + "\n### Job Description:\n"
        + scrape_results,
        agent=scorer,
        expected_output="A value of low relevance, medium relevance, or high relevance.",
    )
    score = task4.execute().replace("'", "")

    # don't always run the entire thing through as this can run out of memory, so we slice to what we can run on in-memory
    doc2 = nlp(scrape_results[: nlp.max_length - 1])
    # Get the vectors of the documents
    vector2 = doc2.vector
    # Compute cosine similarity, value between 0 and 1, higher is better
    cosine_sim = 1 - cosine(vector1, vector2)

    df.loc[df["url"] == row["url"], "is_job_posting"] = str(evaluation_results)
    df.loc[df["url"] == row["url"], "llm_is_job_posting"] = str(llm_evaluation_results)
    df.loc[df["url"] == row["url"], "relevance_score"] = score
    df.loc[df["url"] == row["url"], "cosine_similarity"] = cosine_sim

    df.to_excel("/results/job-results.xlsx", index=False)
