###### foo

![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)
<p align="left">
  <a href="#-features">Features</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#%EF%B8%8F-architecture">Architecture</a> ·
  <a href="#%EF%B8%8F-installation">Install</a> ·
  <a href="#%EF%B8%8F-running-the-streamlit-app">Run</a> ·
  <a href="#-loaders">Loaders</a> ·
  <a href="#%EF%B8%8F-scraping">Scraping</a> ·
  <a href="#%EF%B8%8F-retrieval-sources">Retrievers</a> ·
  <a href="#-domain-fetchers">Fetchers</a> ·
  <a href="#-generation-providers">AI</a> ·
  <a href="#-requirements">Requirements</a> ·
  <a href="#-example-usage">Examples</a> ·
</p>

___

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/foo/)

Foo is a data loading, scraping, retrieval, geospatial, environmental,
astronomical, demographic, generative-AI, and data-processing workspace. It is designed
to give users explicit, hands-on control over how content is loaded, extracted, queried, fetched,
cleaned, analyzed, visualized, and routed into downstream machine-learning or agentic workflows.

Foo is modular by design. Loaders, scrapers, fetchers, generators, databases, and external APIs can
operate independently while remaining composable inside one cohesive interface. The application
supports local files, web pages, public archives, Google services, government data sources,
geospatial APIs, environmental APIs, astronomical APIs, demographic APIs, and multiple LLM
providers.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo-demo.gif)
___

## ☁️ Cloud

<table>
<tr>
<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://chonky.nicehill-0e7bfe90.centralus.azurecontainerapps.io">
<img src="https://img.shields.io/badge/Docker-App-2496ED?logo=docker&logoColor=white" alt="Docker App">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://fooo-py.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://drive.google.com/file/d/1q3ZhnGBGaJjgfiiRirOrONHv2wF8_CtH/view?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://leeroy.usw-16.palantirfoundry.com/shares/links/eitihztkzx76q">
<img src="https://img.shields.io/badge/Databricks%20Repo-Foo--Py-FF3621?logo=databricks&logoColor=white" alt="Databricks Notebook">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://leeroy.usw-16.palantirfoundry.com/shares/links/r7ukk3ybt65bk">
<img src="https://img.shields.io/badge/Palantir%20Foundry-Repo-101113?logo=palantir&logoColor=white" alt="Palantir Repo">
</a>
</td>
</tr>
</table>


## ✨ Features

| Capability                  | Description                                                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modular document loading    | Load text, CSV, XML, PDF, Markdown, HTML, JSON, PowerPoint, Excel, arXiv, Wikipedia, GitHub, web pages, crawled websites, notebooks, and cloud files.                                  |
| Web scraping                | Extract titles, plain text, raw HTML, headings, paragraphs, lists, tables, articles, sections, divisions, blockquotes, hyperlinks, and image references.                               |
| Public retrieval            | Query arXiv, Google Drive, Wikipedia, Google Custom Search, NASA Open Science, GovInfo, Congress.gov, Internet Archive, Grokipedia, Jupyter notebooks, cloud files, and cloud buckets. |
| Geospatial workflows        | Query geocoding, Google Maps, Google Weather, OpenWeather, historical weather, USGS earthquakes, NASA Earth Observatory, USGS National Map, USGS ScienceBase, and OpenSky.             |
| Environmental workflows     | Query AirNow, NOAA Climate Data, NASA EONET, EPA EnviroFacts, NOAA Tides and Currents, EPA UV Index, PurpleAir, OpenAQ, NASA FIRMS, and USGS Water Data.                               |
| Astronomical workflows      | Query U.S. Naval Observatory, Satellite Center, Astro Catalog, AstroQuery, StarMap, SIMBAD, Space Weather, Star Chart, and near-Earth object data.                                     |
| Demographic and health data | Query U.S. Census, CDC Socrata, U.S. HealthData, WHO Global, United Nations, World Population, CDC WONDER, PubMed, and Open City Data.                                                 |
| Generative AI               | Use ChatGPT, Grok, Claude, Gemini, and Mistral through a shared prompt and parameter interface.                                                                                        |
| SQLite management           | Import Excel workbooks, browse tables, perform CRUD operations, filter, aggregate, visualize, alter schema, and run read-only SQL.                                                     |
| Text analytics              | Compute token counts, vocabulary, type-token ratio, hapax ratio, stopword ratio, lexical density, top tokens, and optional readability metrics.                                        |

## 🕹️ Application Modes

| Mode                | Purpose                                                                                      | Major Components                                                                                                                                                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Loading**         | Load local, web, corpus, repository, and cloud documents into shared document/session state. | Text, NLTK corpora, CSV, XML, PDF, Markdown, HTML, JSON, PowerPoint, Excel, arXiv, Wikipedia, GitHub, Web Loader, Web Crawler.                                                                                                                 |
| **Scraping**        | Scrape a target URL or recursively crawl pages and extract structured web content.           | Page title, basic text, raw HTML, headings, paragraphs, lists, tables, articles, sections, divisions, blockquotes, hyperlinks, images.                                                                                                         |
| **Retrieval**       | Query public collections, archives, search services, cloud files, and cloud buckets.         | arXiv, Google Drive, Wikipedia, Google Search, NASA Open Science, GovInfo, U.S. Congress, Internet Archive, Grokipedia, Jupyter Notebook, Google Cloud File, AWS S3 File, OneDrive, Google Speech-to-Text, AWS S3 Bucket, Google Cloud Bucket. |
| **Geospatial**      | Retrieve location, weather, map, flight, and earth-science data.                             | Geocoding, Google Maps, Google Weather, OpenWeather, Historical Weather, USGS Earthquakes, NASA Earth Observatory, USGS National Map, USGS ScienceBase, OpenSky.                                                                               |
| **Environmental**   | Retrieve environmental, climate, water, fire, air-quality, UV, and sensor data.              | AirNow, NOAA Climate Data, NASA EONET, EPA EnviroFacts, NOAA Tides and Currents, EPA UV Index, PurpleAir, OpenAQ, NASA FIRMS, USGS Water Data.                                                                                                 |
| **Astronomical**    | Retrieve astronomical, satellite, star, space-weather, and near-Earth object data.           | U.S. Naval Observatory, Satellite Center, Astro Catalog, AstroQuery, StarMap, SIMBAD, Space Weather, Star Chart, Near-Earth Objects.                                                                                                           |
| **Demographic**     | Retrieve demographic, health, population, city, and public-health records.                   | U.S. Census, CDC Socrata, U.S. Health, WHO Global, United Nations, World Population, CDC WONDER, PubMed Search, Open City Data.                                                                                                                |
| **Generation**      | Generate or analyze text using multiple AI providers.                                        | ChatGPT, Grok, Claude, Gemini, Mistral.                                                                                                                                                                                                        |

## 🏛️ Architecture

```text
📥 Loader → 🧹 Text Processing → 🧠 Session State → 🔍 Retrieval / Analysis / Generation
      │                  │                    │
      ├── 🕸️ Scraper      ├── 📊 Metrics        ├── 🗄️ SQLite
      ├── 🌐 Fetcher      ├── 🧩 Documents      ├── 📈 Visualization
      └── ☁️ Cloud        └── 📝 Raw Text       └── 🤖 LLM Providers
```

Foo uses a Streamlit UI over modular Python classes. The application imports loader classes from
`loaders.py`, provider classes from `generators.py`, and API/data-source wrappers from `fetchers.py`.
Shared working state is coordinated through `st.session_state`, allowing loaded documents, raw text,
processed text, tokens, metrics, and database results to flow between controls.

![](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/images/foo-architecture.png)

___

## 🗂️ Directory Structure

```text
foo/
├── app.py                 # Streamlit user interface
├── config.py              # App title, mode map, defaults, labels, API references, and constants
├── core.py                # Optional package core abstractions
├── data.py                # Data helpers and shared data abstractions
├── fetchers.py            # External API, archive, geospatial, environmental, and science fetchers
├── generators.py          # ChatGPT, Claude, Grok, Mistral, and Gemini wrappers
├── loaders.py             # File, web, cloud, repository, and corpus loaders
├── scrapers.py            # HTML extraction helpers
├── requirements.txt       # Python dependencies
├── stores/
│   └── sqlite/            # SQLite database storage
└── resources/
    └── images/            # README and UI image assets
```
## 🔑 API Set-up

- [Science APIs](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/API-Setup.md) 
- [OpenAI](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/environments.md) 
- [Gemini AI](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/gemini.md) 
- [Grok AI](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/xai.md) 
- [Mistral AI](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/mistral.md) 
- [Claude AI](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/setup/claude.md) 

## 🛡️ Installation

```bash
git clone https://github.com/is-leeroy-jenkins/Foo.git
cd Foo
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Linux or macOS:

```bash
git clone https://github.com/is-leeroy-jenkins/Foo.git
cd Foo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ▶️ Running the Streamlit App

```bash
streamlit run app.py
```

The application initializes Streamlit in wide layout, loads the configured mode map from `config.py`,
and displays the active mode selector in the sidebar under **🕹️ Mode**.

## 🚀 Quick Start

### Run the Application

```bash
streamlit run app.py
```

### Load a Document

1. Open **Loading** mode.
2. Expand a loader such as **PDF Loader**, **Excel Loader**, **Web Loader**, or **GitHub Loader**.
3. Select or enter the source.
4. Click **Load**.
5. Review the document preview panel and corpus metrics.

### Scrape a Web Page

1. Open **Scraping** mode.
2. Enter a target URL.
3. Select core output and structured extraction options.
4. Optionally enable recursive crawl controls.
5. Click **Run Scraper**.

### Query a Public Source

1. Open **Retrieval** mode.
2. Expand a source such as **ArXiv**, **Google Search**, **Gov Info**, or **US Congress**.
3. Enter the query and parameters.
4. Click **Submit**.
5. Review rendered summaries, rows, and raw results.

![](https://github.com/is-leeroy-jenkins/foo/blob/main/resources/images/foo-workflows.png)

___

## 📤 Loaders

| Loader                | Input                                                | Purpose                                                                                        |
| --------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Text Loader**       | `.txt` files                                         | Loads plain text into document/session state.                                                  |
| **Corpora Loader**    | NLTK corpora or local text directory                 | Loads Brown, Gutenberg, Reuters, WebText, Inaugural, State of the Union, or local text files.  |
| **CSV Loader**        | `.csv` files                                         | Loads delimited tabular text as documents.                                                     |
| **XML Loader**        | `.xml` files                                         | Supports semantic XML loading, document splitting, structured tree loading, and XPath queries. |
| **PDF Loader**        | `.pdf` files                                         | Loads PDF content in single or element mode, with plain or OCR extraction options.             |
| **Markdown Loader**   | `.md`, `.markdown` files                             | Loads Markdown content into document state.                                                    |
| **HTML Loader**       | `.html`, `.htm` files                                | Loads local HTML files.                                                                        |
| **JSON Loader**       | `.json` files                                        | Loads JSON or JSON Lines.                                                                      |
| **PowerPoint Loader** | `.pptx` files                                        | Loads PowerPoint slide content.                                                                |
| **Excel Loader**      | `.xlsx`, `.xls` files                                | Loads Excel sheets and stores sheet data in SQLite tables.                                     |
| **ArXiv Loader**      | Query text                                           | Retrieves arXiv documents.                                                                     |
| **Wikipedia Loader**  | Query text                                           | Retrieves Wikipedia documents.                                                                 |
| **GitHub Loader**     | GitHub API URL, repository, branch, file-type filter | Loads repository files matching the selected filter.                                           |
| **Web Loader**        | One or more URLs                                     | Loads web documents.                                                                           |
| **Web Crawler**       | Start URL                                            | Recursively crawls web pages with depth/domain controls.                                       |

## 🕸️ Scraping

| Output Category           | Supported Extraction                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| Core output               | Page title, basic text, raw HTML.                                                              |
| Structured text           | Headings, paragraphs, lists, tables, articles, sections, divisions, blockquotes.               |
| Link and media extraction | Hyperlinks and images.                                                                         |
| Crawl controls            | Recursive crawl, max depth, max pages, same-domain-only filtering.                             |
| Results                   | Per-page metadata, plain text, raw HTML, discovered links, extracted records, and error lists. |

## 🏛️ Retrieval Sources

| Source                    | Purpose                                                                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ArXiv**                 | Retrieve research documents by query or identifier.                                                                                                         |
| **Google Drive**          | Retrieve documents or snippets from Google Drive.                                                                                                           |
| **Wikipedia**             | Retrieve Wikipedia article content and metadata.                                                                                                            |
| **Google Search**         | Use Google Custom Search with exact terms, exclusions, file type, date restriction, site search, image search, country, language, and safe-search controls. |
| **Open Science**          | Query NASA Open Science / OSDR dataset, metadata, assays, and data endpoints.                                                                               |
| **Gov Info**              | Search GovInfo, retrieve package summaries, or browse collections.                                                                                          |
| **US Congress**           | Query Congress.gov congresses, bills, bill details, laws, law details, reports, and report details.                                                         |
| **Internet Archive**      | Search archived media and text collections.                                                                                                                 |
| **Grokipedia**            | Retrieve Grokipedia pages or search results.                                                                                                                |
| **Jupyter Notebook**      | Load notebook content.                                                                                                                                      |
| **Google Cloud File**     | Load a single Google Cloud file.                                                                                                                            |
| **AWS S3 File**           | Load a single AWS S3 file.                                                                                                                                  |
| **OneDrive**              | Load OneDrive-hosted documents.                                                                                                                             |
| **Google Speech-to-Text** | Transcribe audio using Google Speech-to-Text.                                                                                                               |
| **AWS S3 Bucket**         | Load records from an S3 bucket.                                                                                                                             |
| **Google Cloud Bucket**   | Load records from a Google Cloud bucket.                                                                                                                    |

## 🌎 Domain Fetchers

### Geospatial

| Fetcher                    | Purpose                                                                          |
| -------------------------- | -------------------------------------------------------------------------------- |
| **Geocoding**              | Resolve address/location text into coordinates and normalized location metadata. |
| **Google Maps**            | Query Google Maps functionality such as place/location operations.               |
| **Google Weather**         | Retrieve Google Weather data.                                                    |
| **Open Weather**           | Retrieve OpenWeather/Open-Meteo style weather data.                              |
| **Historical Weather**     | Retrieve historical weather data.                                                |
| **USGS Earthquakes**       | Retrieve earthquake events and feature records.                                  |
| **NASA Earth Observatory** | Retrieve NASA Earth Observatory content.                                         |
| **The National Map**       | Retrieve USGS National Map results.                                              |
| **USGS ScienceBase**       | Retrieve ScienceBase records.                                                    |
| **OpenSky**                | Retrieve aviation/open-sky records.                                              |

### Environmental

| Fetcher                     | Purpose                                                   |
| --------------------------- | --------------------------------------------------------- |
| **AirNow**                  | Retrieve air-quality observations and forecasts.          |
| **NOAA Climate Data**       | Retrieve climate data.                                    |
| **NASA EONET**              | Retrieve natural event records.                           |
| **EPA EnviroFacts**         | Retrieve EPA environmental facility or data records.      |
| **NOAA Tides and Currents** | Retrieve tides, currents, stations, and water-level data. |
| **EPA UV Index**            | Retrieve UV index information.                            |
| **PurpleAir**               | Retrieve PurpleAir sensor data.                           |
| **OpenAQ**                  | Retrieve open air-quality data.                           |
| **NASA FIRMS**              | Retrieve fire/hotspot data.                               |
| **USGS Water Data**         | Retrieve USGS water data.                                 |

### Astronomical

| Fetcher                  | Purpose                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| **US Naval Observatory** | Retrieve celestial navigation/time data for observer coordinates. |
| **Satellite Center**     | Retrieve satellite or ground station data.                        |
| **Astro Catalog**        | Retrieve astronomical catalog data.                               |
| **AstroQuery**           | Query astronomical services.                                      |
| **Star Map**             | Generate or retrieve star map data.                               |
| **SIMBAD**               | Query SIMBAD astronomical objects.                                |
| **Space Weather**        | Retrieve space weather data.                                      |
| **Star Chart**           | Generate or retrieve star chart information.                      |
| **Near Earth Objects**   | Retrieve near-Earth object or related object data.                |

### Demographic and Health

| Fetcher                | Purpose                                                   |
| ---------------------- | --------------------------------------------------------- |
| **U.S. Census Bureau** | Retrieve Census records.                                  |
| **CDC Socrata**        | Retrieve CDC Socrata datasets.                            |
| **U.S. Health**        | Retrieve HealthData.gov or similar public health records. |
| **WHO Global**         | Retrieve WHO Global Health Observatory data.              |
| **United Nations**     | Retrieve United Nations data.                             |
| **World Population**   | Retrieve world population datasets.                       |
| **CDC WONDER**         | Retrieve CDC WONDER data.                                 |
| **PubMed Search**      | Search PubMed records.                                    |
| **Open City Data**     | Retrieve city/open-data records.                          |

## 🤖 Generation Providers

| Provider  | Mode Expander | Purpose                                                        |
| --------- | ------------- | -------------------------------------------------------------- |
| OpenAI    | **ChatGPT**   | General text generation and analysis through the Chat wrapper. |
| xAI       | **Grok**      | Text generation and analysis through the Grok wrapper.         |
| Anthropic | **Claude**    | Text generation and analysis through the Claude wrapper.       |
| Google    | **Gemini**    | Text generation and analysis through the Gemini wrapper.       |
| Mistral   | **Mistral**   | Text generation and analysis through the Mistral wrapper.      |

## 📦 Requirements

The table below reflects the requirements implied by the active imports, loaders, fetchers, and UI
surface in `app.py`. Some provider-specific loaders/fetchers may require additional credentials or
cloud SDKs depending on deployment.

| Requirement              | Import / Package Name                 | Purpose                                                               | Required By                                             |
| ------------------------ | ------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| Python                   | `python>=3.10`                        | Runtime for modern typing syntax and Streamlit application execution. | Entire application.                                     |
| Streamlit                | `streamlit`                           | Web application framework.                                            | UI, sidebar, modes, expanders, controls, session state. |
| Altair                   | `altair`                              | Declarative charting support.                                         | Visualization and chart-compatible workflows.           |
| Pandas                   | `pandas`                              | Dataframes, Excel ingestion, SQL result rendering, tabular previews.  | Loaders, Data Management, result tables.                |
| NumPy                    | `numpy`                               | Numeric arrays and vector calculations.                               | Text/vector utilities and analysis helpers.             |
| Plotly                   | `plotly`                              | Interactive charts and visualizations.                                | Data Management visualization engine.                   |
| BeautifulSoup            | `beautifulsoup4`                      | HTML parsing and link/text extraction.                                | Scraping mode and HTML preview helpers.                 |
| Requests                 | `requests`                            | HTTP request support.                                                 | Web fetchers and API wrappers.                          |
| Crawl4AI                 | `crawl4ai`                            | JavaScript-capable or enhanced crawling support.                      | Web crawling workflows.                                 |
| LangChain Core           | `langchain-core`                      | `Document` object model for loaded/retrieved records.                 | Loaders and retrieval result handling.                  |
| LXML                     | `lxml`                                | XML parsing and XPath operations.                                     | XML Loader.                                             |
| NLTK                     | `nltk`                                | Tokenization, stopwords, WordNet, corpora, text metrics.              | Loading metrics and Corpora Loader.                     |
| TextStat                 | `textstat`                            | Optional readability metrics.                                         | Readability panel.                                      |
| Astroquery               | `astroquery`                          | Astronomical service access, including SIMBAD.                        | Astronomical mode.                                      |
| SQLite                   | `sqlite3`                             | Local database storage and SQL execution.                             | Data Management and local stores.                       |
| OpenPyXL                 | `openpyxl`                            | Excel `.xlsx` read/write engine.                                      | Excel Loader and Data Management import.                |
| Python PPTX              | `python-pptx`                         | PowerPoint text extraction support.                                   | PowerPoint Loader.                                      |
| PyMuPDF                  | `PyMuPDF`                             | PDF extraction support where used by PDF loader internals.            | PDF Loader.                                             |
| Unstructured             | `unstructured`                        | Optional document extraction for complex files.                       | PDF/XML/document loader implementations.                |
| Python DOCX / Docx2Txt   | `python-docx` / `docx2txt`            | Word document extraction support.                                     | WordLoader.                                             |
| Boto3                    | `boto3`                               | AWS S3 file and bucket access.                                        | AWS S3 File and AWS S3 Bucket loaders.                  |
| Google API Client        | `google-api-python-client`            | Google Drive and Google API access.                                   | Google Drive and cloud workflows.                       |
| Google Auth              | `google-auth`, `google-auth-oauthlib` | Google credentials and OAuth flows.                                   | Google Drive, Google Cloud, Google Speech-to-Text.      |
| Google Cloud Storage     | `google-cloud-storage`                | Google Cloud bucket/file access.                                      | Google Cloud File and Google Cloud Bucket loaders.      |
| Google Cloud Speech      | `google-cloud-speech`                 | Speech-to-text transcription.                                         | Google Speech-to-Text loader.                           |
| ArXiv                    | `arxiv`                               | arXiv search and document retrieval.                                  | ArXiv Loader and Retrieval mode.                        |
| Streamlit Runtime Extras | `watchdog`                            | Optional local development file watching.                             | Local Streamlit development.                            |
| Environment Variables    | `python-dotenv`                       | Optional `.env` loading for API keys.                                 | Local configuration.                                    |
| Typing Extensions        | `typing-extensions`                   | Backported typing support where needed.                               | Compatibility support.                                  |


## 🔑 Configuration

| Key / Setting             | Purpose                                                                          | Used By                                    |
| ------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------ |
| `APP_TITLE`               | Streamlit page title.                                                            | App setup.                                 |
| `FAVICON`                 | Browser/page icon.                                                               | App setup.                                 |
| `LOGO`                    | Sidebar/application logo.                                                        | App setup.                                 |
| `MODE_MAP`                | Mode names displayed in the sidebar.                                             | Sidebar mode selector.                     |
| `SESSION_STATE_DEFAULTS`  | Shared session-state defaults.                                                   | Startup initialization.                    |
| `REQUIRED_CORPORA`        | NLTK resources to verify/download.                                               | Loading mode and text analytics.           |
| `DB_PATH`                 | SQLite database path.                                                            | Data Management and persistent app tables. |
| `GOOGLE_API_KEY`          | Google service key.                                                              | Google Search and related Google fetchers. |
| `GOOGLE_CSE_ID`           | Google Custom Search Engine ID.                                                  | Google Search.                             |
| `GOOGLE_ACCOUNT_FILE`     | Google service account credential file.                                          | Google Drive and Google Cloud workflows.   |
| `GOOGLE_DRIVE_FOLDER_ID`  | Default Drive folder identifier.                                                 | Google Drive retrieval.                    |
| `GOOGLE_DRIVE_TOKEN_PATH` | Optional token persistence path.                                                 | Google Drive retrieval.                    |
| `LANGSMITH_API_KEY`       | Optional LangSmith tracing key.                                                  | LangChain tracing where configured.        |
| Provider model lists      | `GPT_MODELS`, `GROK_MODELS`, `CLAUDE_MODELS`, `GEMINI_MODELS`, `MISTRAL_MODELS`. | Generation mode model selectors.           |

## 🔍 Example Usage

### Scrape Web Page Paragraphs

```python
from foo.scrapers import WebExtractor

extractor = WebExtractor()
paragraphs = extractor.scrape_paragraphs("https://example.com")
print(paragraphs)
```

### Load and Chunk a PDF

```python
from foo.loaders import PdfLoader

loader = PdfLoader()
documents = loader.load("docs/report.pdf")
chunks = loader.split(documents, chunk=1000, overlap=100)
print(chunks)
```

### Query a Fetcher

```python
from foo.fetchers import Wikipedia

fetcher = Wikipedia(language="en", max_documents=5)
documents = fetcher.fetch("Natural language processing")
for document in documents:
    print(document.metadata)
    print(document.page_content[:500])
```

### Run a Read-Only SQLite Query

```python
import sqlite3
import pandas as pd

with sqlite3.connect("stores/sqlite/data.db") as connection:
    df_results = pd.read_sql_query("SELECT * FROM Prompts LIMIT 10;", connection)

print(df_results)
```

## ⚙️ Technical Notes

| Topic                 | Note                                                                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Session state         | The application uses `st.session_state` for mode state, model parameters, loaded documents, raw text, processed text, tokens, vocabulary, token counts, API configuration, and database working state. |
| Safety                | SQL execution is intentionally constrained to read-only query forms through `is_safe_query`.                                                                                                           |
| Loader contract       | Loaded content is promoted into shared document state with raw text and active loader metadata.                                                                                                        |
| SQLite                | Data Management uses local SQLite tables and supports schema operations through guarded helper functions.                                                                                              |
| Metrics               | Text metrics are computed from either processed text or raw text depending on what is available.                                                                                                       |
| Optional dependencies | Some loaders and fetchers are only needed when their corresponding mode/expander is used.                                                                                                              |
| Credentials           | API keys are entered through sidebar configuration or loaded from configuration/environment variables.                                                                                                 |

## 🔑 AI API Key

| Provider | Setup Link                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------ |
| OpenAI   | [OpenAI API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/openai.md) |
| Grok     | [Grok API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/xai.md)      |
| Gemini   | [Gemini API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/gemini.md) |

#### Data Services

| Service        | Link                                                                                           | Service      | Link                                                                              |
| -------------- | ---------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------- |
| OpenAI         | [Platform](https://platform.openai.com/home)                                                   | Grok         | [Account](https://accounts.x.ai/account)                                          |
| Gemini         | [AI Studio](https://aistudio.google.com/api-keys)                                              | Claude       | [API Keys](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve)       |
| Mistral        | [Console](https://chat.mistral.ai/1)                                                           | NASA         | [NASA API](https://api.nasa.gov/)                                                 |
| Geolocation    | [Google Geolocation](https://developers.google.com/maps/documentation/geolocation/get-api-key) | Google Maps  | [Google Maps](https://developers.google.com/maps/documentation/embed/get-api-key) |
| Gov Data       | [GovInfo API](https://api.govinfo.gov/docs/)                                                   | The News API | [Register](https://www.thenewsapi.com/register)                                   |
| Google Weather | [Weather API](https://developers.google.com/maps/documentation/weather/get-api-key)            | Grokipedia   | [PyPI](https://pypi.org/project/grokipedia-api/)                                  |
| CDC            | [CDC Data](https://data.cdc.gov/login)                                                         | Purple Air   | [Developer Portal](https://develop.purpleair.com/)                                |
| FIRMS          | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/usfs/api/map_key/)                           | CENSUS       | [API Key](https://api.census.gov/data/key_signup.html)                            |
| Wikipedia      | [Wikimedia APIs](https://www.mediawiki.org/wiki/Wikimedia_APIs/Get_started)                    |              |                                                                                   |

## 📝 License

![License: Public Domain](https://img.shields.io/badge/license-public%20domain-brightgreen.svg)

* MIT License [here](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt)
* Copyright © 2022–2025 Terry D. Eppler
