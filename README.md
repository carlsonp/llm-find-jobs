# llm-find-jobs

A job posting classification model and LLM to help in finding open jobs.  Also
leverages CrewAI and LLMs for searching and finding relevant positions.

## Requirements

* A large-language model (LLM) to use, e.g. `llama3.2` deployed via open-webui
* Searx
* Docker (w/buildkit enabled)
* docker compose

## Setup

Setup Searx:

Edit `./searxng/settings.yml` and enter your own value for the `secret_key`.

Adjust Searx as needed for whatever search engines you prefer.
Make sure Searx is NOT accessible to the wider internet.

* Download the following datasets and put the extracted files in `./data/`.

Datasets:

* [https://www.kaggle.com/datasets/madhab/jobposts](https://www.kaggle.com/datasets/madhab/jobposts)
* [https://www.kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
* [https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences)

Copy `.env-copy` to `.env`.

Next, edit the file `.env` and adjust as needed.

Build the docker containers:

```shell
docker compose build --pull
```

Train the job classification model.  This can take a few minutes depending
on your hardware.  Make sure to run on a system with at least 64 GB of RAM.

```shell
docker compose run trainclassifier
```

Bring up the entire stack including OpenSearch and the Flask application.

```shell
docker compose up -d
```

Access [http://127.0.0.1:5000](http://127.0.0.1:5000) for the web-ui.

## Usage and Notes

* Even if the classification model or LLM deems the entry is not related to an open job or internship
position, it may be still worthwhile to look at the website and see.  Many sites leverage
bot detection and this may hamper the ability to scrape the website contents and make an accurate
determination.
* Personas can be used to create tailored resumes or tweaked versions of skills for specific
types of jobs.

## Development

The OpenSearch Dashboard can be accessed on [http://localhost:5601](http://localhost:5601)
which is helpful in debugging the database.

Make sure to [add this setting](https://github.com/langchain-ai/langchain/issues/855#issuecomment-1452900595)
to the searx settings config

```file
 search:
   formats:
     - html
     - json
```

Also turn off [bot protection](https://docs.searxng.org/admin/searx.limiter.html).

```file
server:
  ...
  limiter: false  # rate limit the number of request on the instance, block some bots
```

## References

* [https://github.com/joaomdmoura/crewAI-examples](https://github.com/joaomdmoura/crewAI-examples)
* [https://docs.crewai.com](https://docs.crewai.com)
* [https://python.langchain.com/v0.2/docs/integrations/tools/](https://python.langchain.com/v0.2/docs/integrations/tools/)
* [https://docs.searxng.org/index.html](https://docs.searxng.org/index.html)
* [https://github.com/searxng/searxng-docker](https://github.com/searxng/searxng-docker)
* [https://github.com/open-webui/open-webui](https://github.com/open-webui/open-webui)
* [https://ai.plainenglish.io/opensearch-overview-lexical-vs-semantic-search-queries-7e4bea9566d7](https://ai.plainenglish.io/opensearch-overview-lexical-vs-semantic-search-queries-7e4bea9566d7)
