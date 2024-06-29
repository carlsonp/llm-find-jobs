# llm-find-jobs

A job posting classification model and LLM to help in finding open jobs.  Also
leverages CrewAI and LLMs for searching and finding relevant positions.

## Requirements

* A large-language model (LLM) to use, e.g. `phi3` deployed via open-webui
* Searx
* Docker (w/buildkit enabled)
* docker compose

## Setup

Setup Searx:

Make sure to [add this setting](https://github.com/langchain-ai/langchain/issues/855#issuecomment-1452900595)
to the searx settings config

```file
 search:
   formats:
     - html
     - json
```

Also turn off [bot protection](https://docs.searxng.org/admin/searx.limiter.html).  Make sure
Searx is NOT accessible to the wider internet.

```file
server:
  ...
  limiter: false  # rate limit the number of request on the instance, block some bots
```

Adjust Searx as needed for whatever search engines you prefer.

* Download the following datasets and put the extracted files in `./data/`.

Datasets:

* [https://www.kaggle.com/datasets/madhab/jobposts](https://www.kaggle.com/datasets/madhab/jobposts)
* [https://www.kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
* [https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences)

Next, edit the file `.env` and adjust as needed.

Build the docker containers:

```shell
docker compose build
```

Train the job classification model.  This can take a few minutes depending
on your hardware.

```shell
docker compose run trainclassifier
```

Search and download the job information.  This can be run multiple
times if needed, it will append additional URLs that it finds
to the list to process later.  This will also output
a file `./results/search-queries.json` which shows what search
queries were generated by the LLM.

```shell
docker compose run llm python3 /src/job-search.py
```

Process the job search data, it uses the LLM and the job classification
model to process the extracted text data from the website URLs.

```shell
docker compose run llm python3 /src/job-process.py
```

That's it, now look at `./results/job-results.xlsx` for the final
output.

## Usage and Notes

* Even if the classification model or LLM deems the entry is not related to an open job or internship
position, it may be still worthwhile to look at the website and see.  Many sites leverage
bot detection and this may hamper the ability to scrape the website contents and make an accurate
determination.

## References

* [https://github.com/joaomdmoura/crewAI-examples](https://github.com/joaomdmoura/crewAI-examples)
* [https://docs.crewai.com](https://docs.crewai.com)
* [https://python.langchain.com/v0.2/docs/integrations/tools/](https://python.langchain.com/v0.2/docs/integrations/tools/)
* [https://docs.searxng.org/index.html](https://docs.searxng.org/index.html)
* [https://github.com/searxng/searxng-docker](https://github.com/searxng/searxng-docker)
* [https://github.com/open-webui/open-webui](https://github.com/open-webui/open-webui)
