# Fetching and Scraping

Foo separates lightweight scraping helpers from broader fetcher classes.

- `scrapers.py` contains HTML extraction utilities.
- `fetchers.py` contains web, archive, geospatial, environmental, astronomical, public-data, and retriever wrappers.

# Fetching and Scraping

Foo separates external data retrieval from HTML extraction. This distinction matters because
fetching, scraping, and loading solve related but different problems.

The `fetchers.py` module retrieves content from web pages, APIs, public data services, scientific
data sources, geospatial systems, environmental data services, weather services, astronomy services,
and government datasets. The `scrapers.py` module focuses on extracting usable text or structured
elements from HTML pages.

Together, these modules support workflows that begin with a URL, API endpoint, search query, data
service, or public dataset and end with normalized content that can be displayed, stored, analyzed,
exported, or passed into downstream AI workflows.

## 🌐 Retrieval vs. Scraping

Foo uses separate modules for retrieval and extraction.

| Concern     | Module        | Responsibility                                                                             |
| ----------- | ------------- | ------------------------------------------------------------------------------------------ |
| Fetching    | `fetchers.py` | Retrieves external content from web pages, APIs, search services, and public-data sources. |
| Scraping    | `scrapers.py` | Extracts readable text and structured HTML elements from web pages.                        |
| Loading     | `loaders.py`  | Loads documents and files into document-oriented objects.                                  |
| Writing     | `writers.py`  | Exports processed results to output files such as Markdown.                                |
| Persistence | `data.py`     | Stores structured or vector-oriented results.                                              |

This separation keeps the application maintainable. A fetcher should retrieve data. A scraper should
extract usable content. A loader should normalize source documents. A writer should serialize
output.

## 🧭 Web Workflow Overview

A typical web workflow follows this sequence:

```text
User enters URL or query
        |
        v
Fetcher retrieves content
        |
        v
HTML or response payload is captured
        |
        v
Scraper or structured extraction method parses useful content
        |
        v
Results are displayed in Streamlit
        |
        v
Results can be stored, exported, or reused
```

For a basic web page, the workflow is:

```text
URL
 |
 v
WebFetcher.fetch(...)
 |
 v
HTTP response + raw HTML
 |
 v
html_to_text(...) or scrape_* methods
 |
 v
Plain text, links, headings, paragraphs, tables, images, or other extracted values
```

For an API or public-data service, the workflow is:

```text
Query parameters
 |
 v
Source-specific fetcher
 |
 v
JSON, CSV-like records, or normalized dictionaries
 |
 v
Display, persistence, export, or downstream processing
```

## 🧱 Fetcher Layer

The fetcher layer lives in:

```text
fetchers.py
```

The base fetcher class is:

```text
Fetcher
```

Provider-specific and source-specific fetchers inherit from or follow the same pattern as the base
fetcher. These classes centralize retrieval behavior so the Streamlit UI does not need to know the
details of each external service.

The fetcher layer includes classes for sources such as:

| Category                    | Example Classes                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Web retrieval               | `WebFetcher`, `WebCrawler`                                                                                                                                          |
| Search and reference        | `GoogleSearch`, `Wikipedia`, `TheNews`, `Grokipedia`                                                                                                                |
| Maps and location           | `GoogleMaps`, `GoogleGeocoding`                                                                                                                                     |
| Weather and climate         | `OpenWeather`, `GoogleWeather`, `HistoricalWeather`, `ClimateData`                                                                                                  |
| Astronomy and space         | `NavalObservatory`, `SatelliteCenter`, `EarthObservatory`, `NearbyObjects`, `SpaceWeather`, `AstroCatalog`, `AstroQuery`, `StarMap`, `StarChart`                    |
| Government and public data  | `GovData`, `Congress`, `InternetArchive`, `CensusData`, `Socrata`                                                                                                   |
| Environmental data          | `USGSEarthquakes`, `USGSWaterData`, `USGSTheNationalMap`, `USGSScienceBase`, `AirNow`, `EnviroFacts`, `TidesAndCurrents`, `UvIndex`, `PurpleAir`, `OpenAQ`, `Firms` |
| Health and population data  | `HealthData`, `GlobalHealthData`, `UnitedNations`, `WorldPopulation`, `Wonder`                                                                                      |
| Aviation and sky data       | `OpenSky`                                                                                                                                                           |
| Science discovery           | `OpenScience`, `ArXiv`                                                                                                                                              |
| Cloud/content source access | `GoogleDrive`                                                                                                                                                       |

This list reflects the breadth of Foo’s retrieval layer. Not every fetcher serves the same kind of
data, but each class should expose a predictable interface for validating request input, calling the
source, and returning a structured result.

## 🕸️ WebFetcher

`WebFetcher` is the central web-page retrieval class.

It handles common web operations such as:

* Performing an HTTP GET request.
* Capturing response metadata.
* Storing raw HTML.
* Parsing HTML with BeautifulSoup.
* Converting HTML into plain text.
* Normalizing URLs.
* Checking same-domain relationships.
* Extracting links.
* Extracting structured HTML elements.
* Returning a canonical result object or structured extraction dictionary.

Conceptually:

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")

if result:
    text = fetcher.html_to_text(result.text)
```

The exact returned object depends on the method used. Refer to the API reference for method
signatures and return annotations.

## 🧹 Structured HTML Extraction

`WebFetcher` supports structured extraction from HTML after a page is retrieved.

Common extraction targets include:

| Extraction Target | Typical HTML Elements              |
| ----------------- | ---------------------------------- |
| Headings          | `h1`, `h2`, `h3`, `h4`, `h5`, `h6` |
| Paragraphs        | `p`                                |
| Lists             | `ul`, `ol`, `li`                   |
| Tables            | `table`, `tr`, `td`, `th`          |
| Articles          | `article`                          |
| Sections          | `section`                          |
| Divisions         | `div`                              |
| Blockquotes       | `blockquote`                       |
| Hyperlinks        | `a[href]`                          |
| Images            | `img[src]`                         |

This is useful when a user needs more than a flat plain-text page body. For example, a documentation
workflow may only need headings and paragraphs, while a data-discovery workflow may need tables and
hyperlinks.

## 🧪 Example: Fetch and Convert HTML to Text

Use this pattern when the goal is to retrieve a page and produce readable text.

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")

if result is not None:
    html = getattr(result, "text", "") or ""
    text = fetcher.html_to_text(html)
    print(text)
```

This pattern is appropriate for:

* Quick page review.
* Converting an HTML article into plain text.
* Preparing scraped text for summarization.
* Producing a source text block for downstream processing.

## 🔗 Example: Extract Links from a Page

Use link extraction when the workflow needs to discover URLs on a page.

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")

if result is not None:
    html = getattr(result, "text", "") or ""
    links = fetcher.extract_links("https://example.com", html)

    for link in links:
        print(link)
```

This pattern is appropriate for:

* Crawling same-domain links.
* Collecting source references.
* Identifying download links.
* Building a source inventory.
* Supporting follow-up retrieval workflows.

## 🧾 Example: Extract Structured Data

Use structured extraction when the user wants specific page components.

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")

if result is not None:
    html = getattr(result, "text", "") or ""

    extracted = fetcher.extract_structured_data(
        url="https://example.com",
        html=html,
        selected_methods=[
            "scrape_headings",
            "scrape_paragraphs",
            "scrape_tables",
            "scrape_hyperlinks",
        ],
    )

    print(extracted)
```

This pattern is appropriate for workflows where page structure matters. It keeps headings,
paragraphs, tables, and links separate instead of flattening all page content into one text block.

## 🧰 Scraper Layer

The scraper layer lives in:

```text
scrapers.py
```

The main classes are:

| Class          | Purpose                                                                                |
| -------------- | -------------------------------------------------------------------------------------- |
| `Extractor`    | Base class for HTML-to-text extraction state.                                          |
| `WebExtractor` | Concrete web extraction class that retrieves HTML and extracts specific page elements. |

`WebExtractor` provides targeted extraction methods for common HTML structures. These methods
retrieve a page, parse it, and return cleaned values from specific tags.

The scraper layer is useful when the workflow is explicitly about extracting page elements rather
than calling a broader API or public-data fetcher.

## 🧱 WebExtractor

`WebExtractor` uses HTTP retrieval and BeautifulSoup parsing to extract targeted page components.

Common methods include:

| Method                    | Extracted Content                                                 |
| ------------------------- | ----------------------------------------------------------------- |
| `scrape(...)`             | Retrieves a page and returns a canonical result.                  |
| `scrape_headings(...)`    | Extracts heading text from `h1` through `h6`.                     |
| `scrape_paragraphs(...)`  | Extracts paragraph text from `p` elements.                        |
| `scrape_lists(...)`       | Extracts list item text from `li` elements.                       |
| `scrape_tables(...)`      | Extracts flattened table cell values from `td` and `th` elements. |
| `scrape_articles(...)`    | Extracts text from `article` elements.                            |
| `scrape_sections(...)`    | Extracts text from `section` elements.                            |
| `scrape_divisions(...)`   | Extracts text from `div` elements.                                |
| `scrape_blockquotes(...)` | Extracts text from `blockquote` elements.                         |
| `scrape_hyperlinks(...)`  | Extracts hyperlink targets from `a[href]`.                        |
| `scrape_images(...)`      | Extracts image references from `img[src]`.                        |

Use these methods when the desired output is a list of specific HTML elements.

## 🧪 Example: Scrape Headings

```python
from scrapers import WebExtractor

extractor = WebExtractor()
headings = extractor.scrape_headings("https://example.com")

for heading in headings or []:
    print(heading)
```

This pattern is useful for quickly understanding page structure.

## 🧪 Example: Scrape Tables

```python
from scrapers import WebExtractor

extractor = WebExtractor()
cells = extractor.scrape_tables("https://example.com")

for cell in cells or []:
    print(cell)
```

This pattern is useful when a page contains HTML tables that need to be reviewed, stored, or
transformed.

## 🧪 Example: Scrape Links and Images

```python
from scrapers import WebExtractor

extractor = WebExtractor()

links = extractor.scrape_hyperlinks("https://example.com")
images = extractor.scrape_images("https://example.com")

print("Links")
for link in links or []:
    print(link)

print("Images")
for image in images or []:
    print(image)
```

This pattern is useful for content inventory, page auditing, documentation migration, and source
discovery.

## 🧭 Streamlit Web-Scraping Flow

The Streamlit application includes web-scraping workflow logic that coordinates user options, page
retrieval, extraction, and display.

The application flow includes helper behavior for:

* Coercing extraction results into displayable string lists.
* Extracting page titles from raw HTML.
* Truncating long text for display.
* Normalizing URLs.
* Checking whether links belong to the same domain.
* Extracting links from HTML.
* Scraping a single page.
* Capturing status code, encoding, title, plain text, raw HTML, discovered links, extracted data,
  and errors.

The UI-facing result shape can be understood as:

```text
{
    "url": "...",
    "status_code": ...,
    "encoding": "...",
    "title": "...",
    "plain_text": "...",
    "raw_html": "...",
    "links_discovered": [...],
    "data": {...},
    "errors": [...]
}
```

This gives the Streamlit UI enough information to show both the source page metadata and the
extracted content.

## ⚙️ Selecting Extraction Methods

When a page is scraped through the UI, selected methods determine what content is extracted.

Typical selected methods include:

```text
scrape_headings
scrape_paragraphs
scrape_lists
scrape_tables
scrape_articles
scrape_sections
scrape_divisions
scrape_blockquotes
scrape_hyperlinks
scrape_images
```

A focused extraction is usually better than extracting everything. For example:

| Goal                          | Recommended Methods                                    |
| ----------------------------- | ------------------------------------------------------ |
| Understand document structure | `scrape_headings`, `scrape_sections`                   |
| Extract article text          | `scrape_articles`, `scrape_paragraphs`                 |
| Build a link inventory        | `scrape_hyperlinks`                                    |
| Extract tabular data          | `scrape_tables`                                        |
| Audit media references        | `scrape_images`                                        |
| Capture quoted source text    | `scrape_blockquotes`                                   |
| Review broad page content     | `scrape_headings`, `scrape_paragraphs`, `scrape_lists` |

## 🧱 Public Data Fetchers

Many fetchers retrieve structured data rather than HTML.

Examples include fetchers for:

* Weather forecasts.
* Historical weather.
* Air quality.
* Earthquakes.
* Water data.
* National map products.
* ScienceBase records.
* Census data.
* Socrata datasets.
* Health datasets.
* Global health datasets.
* United Nations data.
* World population data.
* Space weather.
* Astronomy catalogs.
* Public archives.
* Congressional data.

These fetchers usually return dictionaries, lists, rows, or JSON-like structures rather than raw
HTML.

A conceptual pattern is:

```python
from fetchers import AirNow

fetcher = AirNow()

# Use the fetcher-specific API documented in the API reference.
# Returned data can be displayed, stored, exported, or passed downstream.
```

Because each external service has its own required parameters, use the API reference for each
fetcher’s method signatures.

## 🧰 Tool Schema Support

Several fetcher classes include `create_schema(...)` methods that construct tool-style schema
definitions.

This supports workflows where an external source can be exposed as a callable tool definition. A
schema generally identifies:

* Function name.
* Tool or service name.
* Description.
* JSON-schema parameter definitions.
* Required parameter names.

This pattern is useful when a fetcher may be used in an AI-tooling workflow or function-calling
context.

## 🔐 Error Handling

Fetcher and scraper classes should use the Foo structured error pattern where handled exception
paths exist.

The standard pattern is:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "fetchers"
    exception.cause = "WebFetcher"
    exception.method = "fetch( self, url: str, time: int = 10 ) -> Result | None"
    Logger().write(exception)
    raise exception
```

For scraper classes:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "scrapers"
    exception.cause = "WebExtractor"
    exception.method = "scrape_headings( self, uri: str ) -> List[ str ]"
    Logger().write(exception)
    raise exception
```

Use stable method signatures in `exception.method`. Do not include raw URLs, API keys, tokens, page
contents, scraped text, or file contents in error metadata.

## 🛡️ Safe Fetching Practices

When adding or modifying fetchers, follow safe retrieval practices:

* Validate required parameters before making a network request.
* Use explicit timeouts.
* Normalize URLs before crawling or link expansion.
* Avoid uncontrolled recursive crawling.
* Restrict same-domain crawling when appropriate.
* Handle empty or malformed responses.
* Preserve status code and source metadata when useful.
* Do not log raw response bodies.
* Do not log secrets or API keys.
* Keep provider-specific configuration in `config.py`.
* Keep UI rendering out of fetcher classes.

## 🧪 Testing Fetchers

When testing a fetcher, verify:

* Required arguments are validated.
* Timeouts are applied.
* HTTP errors are handled.
* Empty responses do not crash the workflow.
* Return values match the documented type.
* Error metadata identifies the correct module, class, and method.
* The fetcher does not print or log sensitive response content.
* The Streamlit UI can display the result without type errors.

For web fetchers, test with:

* A valid URL.
* An unreachable URL.
* A URL returning non-HTML content.
* A page with relative links.
* A page with tables.
* A page with no matching elements for the selected scraper.

## 🧪 Testing Scrapers

When testing a scraper, verify:

* The method returns an empty list or documented fallback when no matching elements exist.
* The method returns cleaned strings rather than raw tag objects.
* Tables are flattened predictably.
* Links and image references are returned as strings.
* Exceptions are wrapped and logged where the class uses handled exception paths.
* Long page content does not break the Streamlit display.

## 📦 Relationship to Loaders

Fetching and scraping overlap with loading, but they are not the same thing.

Use a loader when the workflow starts with a known document or file-like source:

```text
PDF
CSV
Excel
Word
Markdown
JSON
PowerPoint
Notebook
Cloud document
```

Use a fetcher when the workflow starts with a service, endpoint, API, search, public data source, or
URL.

Use a scraper when the workflow starts with HTML and the goal is to extract specific elements.

```text
Loader  -> normalize documents
Fetcher -> retrieve external data
Scraper -> extract HTML content
```

## 🗄️ Relationship to Persistence

Fetched and scraped results can be persisted after extraction.

Good SQLite candidates include:

* URL inventories.
* Page titles.
* Status codes.
* Extracted headings.
* Extracted links.
* Extracted table rows.
* API response metadata.
* Search results.
* Public-data records.

Good Chroma candidates include:

* Cleaned page text.
* Article bodies.
* Paragraph collections.
* Documentation pages.
* Source chunks for semantic search.
* Retrieval-augmented generation context.

A common pattern is:

```text
Fetch or scrape
      |
      v
Normalize extracted values
      |
      v
Store metadata in SQLite
      |
      v
Store text chunks in Chroma when semantic retrieval is needed
```

## 🧾 Relationship to Writers

Fetched and scraped content can also be exported through the writer layer.

Useful export targets include:

* Markdown research notes.
* Page inventories.
* Link inventories.
* Extracted tables.
* Scraped documentation pages.
* API response summaries.
* Web research packets.

The writer layer should handle serialization. Fetchers and scrapers should not grow their own
unrelated export logic.

## 📖 API Reference

Use the API reference for exact class definitions, method signatures, and return annotations:

* [Fetchers API](api/fetchers.md)
* [Scrapers API](api/scrapers.md)
* [Core API](api/core.md)
* [Data API](api/data.md)
* [Writers API](api/writers.md)

The API pages are generated from the source modules with MkDocs and mkdocstrings.

## ✅ Maintenance Checklist

When updating `fetchers.py` or `scrapers.py`, verify the following:

* The module compiles.
* Public classes and methods have Google-style docstrings.
* Required arguments are validated.
* Network calls use timeouts.
* Return values match annotations.
* Existing exception handlers preserve behavior.
* Wrapped exceptions are logged before re-raising where the logging pattern is used.
* Error metadata is stable and does not include live request values.
* Source-specific logic remains inside the correct class.
* UI rendering remains in `app.py`.
* New user-facing workflows are documented.
* API reference pages render without Griffe warnings.

## 🧭 Summary

Foo’s fetching and scraping layer is the application’s external information acquisition system.
Fetchers retrieve content from web pages, APIs, and public-data services. Scrapers extract usable
text and structured elements from HTML. Keeping those responsibilities separate makes Foo easier to
extend, easier to document, and safer to maintain.

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")
text = fetcher.html_to_text(result.text)
```
