import string
from datetime import datetime
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import joblib
from opensearchpy.exceptions import NotFoundError


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

        query = {
            "size": 50,
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

                    # app.logger.info(f"Classifier Job Posting evalution: {str(classification_result)}")
                    status = doc["_source"]["status"]
                    if str(classification_result).lower() == "false":
                        status = "Not a Job"

                    # update the document with the entry
                    response = client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": scrape_results,
                                "date_downloaded": datetime.now().isoformat(),
                                "content_vector": model.encode(scrape_results).tolist(),
                                "status": status,
                            }
                        },
                    )
                except Exception as e:
                    print(
                        f"Error updating URL {doc['_source']['url']}: {e}"
                    )
                    # update the document so we don't try and fail again
                    response = client.update(
                        index="jobs_index",
                        id=doc["_id"],
                        body={
                            "doc": {
                                "content": "Error, unable to load content",
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
