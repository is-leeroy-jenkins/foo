# Fetchers API

The `fetchers.py` module is Foo’s external data retrieval layer. It contains classes that retrieve
information from web pages, search providers, public APIs, government datasets, science services,
environmental systems, weather services, astronomy services, mapping services, aviation services,
and other network-accessible sources.

This module should remain focused on retrieval and response shaping. It should not contain Streamlit
UI rendering, database persistence logic, Markdown serialization, or unrelated document-loading
behavior.

## 🧭 Purpose

The Fetchers API provides source-specific classes for retrieving external information.

The module supports:

* Basic web-page retrieval.
* Web crawling.
* HTML-to-text conversion.
* Link extraction.
* Structured HTML extraction.
* Search and reference retrieval.
* Public-data API retrieval.
* Weather and climate data retrieval.
* Geospatial and mapping retrieval.
* Environmental monitoring retrieval.
* Health and population data retrieval.
* Astronomy and space data retrieval.
* Government and civic data retrieval.
* Tool-schema creation for source-specific function workflows.

The fetcher layer gives Foo a consistent place to isolate external-service behavior.

## 🧱 Module Role

`fetchers.py` sits between the Streamlit interface and external data sources.

```text
Streamlit UI
     |
     v
fetchers.py
     |
     +--> Web pages
     +--> Search APIs
     +--> Public data APIs
     +--> Weather and climate services
     +--> Environmental data services
     +--> Astronomy and space services
     +--> Government datasets
     +--> Mapping and geocoding services
```

The rest of the application should call fetcher classes instead of embedding external-service
request logic directly in `app.py`.

## 🧩 Core Fetcher Pattern

The base class is:

```text
Fetcher
```

Most source-specific classes inherit from `Fetcher` or from a specialized fetcher subclass such as
`WebFetcher`.

A typical fetcher should:

1. Validate required arguments.
2. Prepare request parameters.
3. Call the external source.
4. Normalize or package the response.
5. Return a documented result.
6. Log wrapped exceptions where handled exception paths exist.

Conceptual example:

```python
from fetchers import Wikipedia

fetcher = Wikipedia()
result = fetcher.fetch("Python programming language")

print(result)
```

The exact arguments and return types vary by class. Use the generated API reference at the bottom of
this page for method signatures.

## 🌐 Web Fetching Classes

The web retrieval classes handle ordinary page fetching and crawling.

| Class        | Purpose                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| `WebFetcher` | Retrieves web pages and provides helper methods for converting, normalizing, and extracting page content. |
| `WebCrawler` | Extends web retrieval into crawling workflows.                                                            |

`WebFetcher` includes helper methods such as:

* `fetch(...)`
* `html_to_text(...)`
* `coerce_items(...)`
* `extract_title(...)`
* `truncate_text(...)`
* `normalize_url(...)`
* `same_domain(...)`
* `extract_links(...)`
* `extract_structured_data(...)`
* HTML element extraction helpers for paragraphs, lists, tables, articles, sections, divisions,
  blockquotes, hyperlinks, and images.

Use `WebFetcher` when the workflow begins with a URL and the goal is to retrieve page content,
inspect response metadata, extract links, or convert HTML to usable text.

## 🧪 WebFetcher Example

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")

if result is not None:
    text = fetcher.html_to_text(result.text)
    links = fetcher.extract_links(result.url, result.text)

    print(text)
    print(links)
```

Use this pattern for simple page retrieval and content extraction.

## 🕸️ WebCrawler Example

```python
from fetchers import WebCrawler

crawler = WebCrawler()

# Use the crawler methods documented in the API reference.
# Crawler workflows should use controlled limits and same-domain checks.
```

Crawler workflows should be constrained. Avoid uncontrolled recursion, unbounded link expansion, or
long-running crawls without clear limits.

## 🔎 Search and Reference Fetchers

Foo includes fetchers for search and reference sources.

| Class             | Purpose                                                               |
| ----------------- | --------------------------------------------------------------------- |
| `Wikipedia`       | Retrieves information from Wikipedia-oriented workflows.              |
| `TheNews`         | Retrieves news-oriented content.                                      |
| `GoogleSearch`    | Retrieves search results.                                             |
| `Grokipedia`      | Retrieves Grokipedia search or page content.                          |
| `InternetArchive` | Retrieves Internet Archive search results or related archive records. |

Use these classes when the workflow begins with a search term, topic, or reference query rather than
a local file.

Example pattern:

```python
from fetchers import GoogleSearch

search = GoogleSearch()
results = search.fetch("Foo Python documentation")

print(results)
```

## 🗺️ Maps and Geocoding Fetchers

Foo includes mapping and geocoding fetchers.

| Class             | Purpose                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| `GoogleMaps`      | Handles location validation, geocoding, coordinate lookup, and directions requests. |
| `GoogleGeocoding` | Handles forward geocoding, reverse geocoding, and place lookup workflows.           |

These classes are useful when a workflow needs to convert between addresses, coordinates, place
identifiers, or route information.

Example pattern:

```python
from fetchers import GoogleGeocoding

geocoder = GoogleGeocoding()
result = geocoder.fetch_forward("Washington, DC")

print(result)
```

## 🌦️ Weather and Climate Fetchers

Foo includes weather and climate-oriented fetchers.

| Class               | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `GoogleWeather`     | Retrieves current, forecast, historical, and alert-oriented weather data. |
| `OpenWeather`       | Retrieves current, hourly, or daily weather data.                         |
| `HistoricalWeather` | Retrieves historical weather data.                                        |
| `ClimateData`       | Retrieves climate datasets and climate records.                           |
| `UvIndex`           | Retrieves UV index data by ZIP code or city/state.                        |
| `TidesAndCurrents`  | Retrieves NOAA tides and currents style data.                             |

Weather and climate fetchers usually require location, coordinate, date, unit, or provider-specific
parameters.

Example pattern:

```python
from fetchers import OpenWeather

weather = OpenWeather()
result = weather.fetch_current(
    latitude=38.9072,
    longitude=-77.0369,
)

print(result)
```

Use the API reference for exact method signatures because provider requirements vary.

## 🌍 Environmental and Earth Data Fetchers

Foo includes environmental and Earth science fetchers.

| Class                | Purpose                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `USGSEarthquakes`    | Retrieves earthquake feeds and query results.                                                |
| `USGSWaterData`      | Retrieves monitoring locations, time-series metadata, latest values, and water data records. |
| `USGSTheNationalMap` | Retrieves National Map datasets and products.                                                |
| `USGSScienceBase`    | Retrieves ScienceBase items and related records.                                             |
| `AirNow`             | Retrieves current or forecast air-quality data.                                              |
| `EnviroFacts`        | Retrieves EPA EnviroFacts table data.                                                        |
| `PurpleAir`          | Retrieves PurpleAir sensor data.                                                             |
| `OpenAQ`             | Retrieves OpenAQ air-quality locations, measurements, and latest readings.                   |
| `Firms`              | Retrieves fire and thermal anomaly data.                                                     |
| `EoNet`              | Retrieves NASA EONET event and category data.                                                |
| `EarthObservatory`   | Retrieves Earth Observatory event, category, source, and layer information.                  |
| `GlobalImagery`      | Retrieves imagery service metadata or WMS map products.                                      |

These classes are appropriate for public environmental data acquisition and geospatial analysis
workflows.

Example pattern:

```python
from fetchers import AirNow

air = AirNow()
result = air.fetch_current_zip("20001")

print(result)
```

## 🛰️ Astronomy, Space, and Sky Fetchers

Foo includes astronomy and space-oriented fetchers.

| Class              | Purpose                                                                               |
| ------------------ | ------------------------------------------------------------------------------------- |
| `NavalObservatory` | Retrieves celestial navigation or astronomical data from Naval Observatory workflows. |
| `SatelliteCenter`  | Retrieves observatory, ground-station, or location information.                       |
| `NearbyObjects`    | Retrieves near-Earth object, fireball, and related NASA data.                         |
| `SpaceWeather`     | Retrieves NOAA space-weather endpoint data.                                           |
| `AstroCatalog`     | Retrieves astronomy catalog objects or cone-search results.                           |
| `AstroQuery`       | Retrieves object search, object IDs, and region-search results.                       |
| `StarMap`          | Retrieves links or snapshots for objects and coordinates.                             |
| `StarChart`        | Retrieves object charts, coordinate charts, and static chart products.                |

These fetchers support workflows that need astronomical object lookup, coordinate-based sky
products, satellite or observatory information, and space-weather records.

Example pattern:

```python
from fetchers import AstroQuery

astro = AstroQuery()
records = astro.object_search("Vega")

print(records)
```

## 🏛️ Government and Public Data Fetchers

Foo includes government, civic, and public dataset fetchers.

| Class              | Purpose                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `GovData`          | Searches and retrieves public dataset metadata.                          |
| `Congress`         | Retrieves congressional records, bills, laws, reports, and related data. |
| `CensusData`       | Retrieves Census variables and data rows.                                |
| `Socrata`          | Retrieves Socrata metadata and dataset rows.                             |
| `HealthData`       | Retrieves health-data metadata and rows from compatible portals.         |
| `GlobalHealthData` | Retrieves global health indicator and query results.                     |
| `UnitedNations`    | Retrieves UN datasets and SDMX query data.                               |
| `WorldPopulation`  | Retrieves population catalog and raster metadata.                        |
| `Wonder`           | Builds and submits CDC WONDER-style query workflows.                     |

These fetchers support data-discovery, public policy, public health, demographic, and
government-data workflows.

Example pattern:

```python
from fetchers import CensusData

census = CensusData()
variables = census.fetch_variables()

print(variables)
```

Provider requirements differ significantly. Use the API reference for exact arguments.

## ✈️ Aviation and Movement Data Fetchers

Foo includes aviation and movement-oriented data retrieval.

| Class     | Purpose                                                                      |
| --------- | ---------------------------------------------------------------------------- |
| `OpenSky` | Retrieves states, tracks, arrivals, departures, or related aviation records. |

OpenSky workflows may require ICAO identifiers, airport codes, time windows, bounding boxes,
credentials, or other provider-specific parameters.

Example pattern:

```python
from fetchers import OpenSky

opensky = OpenSky()

# Use fetcher-specific methods documented in the API reference.
```

## 📚 Science and Research Fetchers

Foo includes fetchers for scientific and research-oriented sources.

| Class         | Purpose                                                             |
| ------------- | ------------------------------------------------------------------- |
| `ArXiv`       | Retrieves ArXiv research records.                                   |
| `OpenScience` | Retrieves open-science datasets, metadata, assays, or data records. |

Use these fetchers when the workflow begins with a research query, dataset identifier, or scientific
source.

Example pattern:

```python
from fetchers import ArXiv

arxiv = ArXiv()
records = arxiv.fetch("machine learning")

print(records)
```

## ☁️ Cloud and Content Source Fetchers

Foo includes content-source fetchers.

| Class         | Purpose                                                                                |
| ------------- | -------------------------------------------------------------------------------------- |
| `GoogleDrive` | Retrieves or works with Google Drive source content and related mode/template options. |

Cloud content fetchers should isolate provider-specific authentication, source metadata, and
retrieval behavior from the UI layer.

Example pattern:

```python
from fetchers import GoogleDrive

drive = GoogleDrive()

print(drive.mode_options)
print(drive.mime_options)
```

## 🧰 Tool Schema Methods

Many fetcher classes include a `create_schema(...)` method.

These methods construct tool-style schemas that describe a source-specific operation. They are
useful when a fetcher may be exposed to an AI tool-calling or function-calling workflow.

A schema generally defines:

* Tool name.
* Description.
* Parameter object.
* Required parameter names.
* Source-specific argument metadata.

Classes with schema-oriented methods include several public-data, weather, science, astronomy,
geospatial, and mapping fetchers.

Conceptual pattern:

```python
from fetchers import GovData

fetcher = GovData()
schema = fetcher.create_schema()

print(schema)
```

Use schemas as metadata for tool definitions, not as replacements for the fetcher methods
themselves.

## 🔐 Error Handling

Fetcher classes use Foo’s structured error logging pattern where handled exception paths exist.

The preferred pattern is:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "fetchers"
    exception.cause = "WebFetcher"
    exception.method = "fetch( self, url: str, time: int = 10 ) -> Result | None"
    Logger().write(exception)
    raise exception
```

Error metadata should be stable and reviewer-safe.

Do not include:

* API keys.
* Tokens.
* Raw credentials.
* Full response bodies.
* Full scraped pages.
* Raw user documents.
* Long query payloads.
* Sensitive request values.
* File contents.

The goal is to preserve diagnostic context without logging sensitive content.

## 🛡️ Safe Retrieval Guidance

External retrieval code should be conservative.

When adding or modifying a fetcher:

* Validate required arguments before making the request.
* Use explicit request timeouts.
* Keep provider-specific constants in `config.py`.
* Avoid uncontrolled recursive crawling.
* Normalize URLs before comparison or crawling.
* Restrict same-domain crawling where appropriate.
* Preserve response metadata when useful.
* Avoid logging raw response bodies.
* Avoid swallowing exceptions silently.
* Return documented fallback values.
* Keep UI display logic out of fetcher classes.

## 🧪 Testing Fetchers

A fetcher test should verify both success and failure behavior.

Test cases should include:

* Valid request inputs.
* Missing required inputs.
* Invalid parameter values.
* Empty provider responses.
* HTTP error responses.
* Provider timeout behavior.
* Non-JSON responses when JSON is expected.
* Non-HTML responses when HTML is expected.
* Authentication failure where credentials are required.
* Rate-limit or unavailable-service behavior when feasible.
* Return type consistency.
* Error logging in handled exception paths.

For web fetchers, also test:

* Relative links.
* Absolute links.
* Same-domain checks.
* Pages with no matching elements.
* Pages with tables.
* Pages with images.
* Pages with invalid or missing titles.

## 🧾 Return Shape Guidance

Fetcher methods should return predictable values.

Good return shapes include:

* `Result`
* `dict`
* `list[dict]`
* `list[str]`
* `str`
* `None` as an explicit fallback when documented

Avoid returning unrelated shapes from the same method depending on the branch. If a method may
return `None`, document it and annotate it.

For public-data fetchers, consider returning a packaged response containing:

```text
source
query
records
summary
metadata
errors
```

This structure makes Streamlit display, persistence, and export workflows easier to implement
consistently.

## 🧩 Relationship to Streamlit UI

`app.py` should use fetchers as service classes.

The UI should be responsible for:

* Collecting user input.
* Selecting the fetcher.
* Displaying results.
* Handling user-facing messages.
* Passing results to persistence or writer workflows.

The fetcher should be responsible for:

* Validating fetcher-specific input.
* Calling the external service.
* Returning the result.
* Logging handled exceptions.

This separation keeps UI code maintainable.

## 🗄️ Relationship to Persistence

Fetcher results can be stored by `data.py`.

Good SQLite candidates:

* Source name.
* URL.
* Query parameters.
* Status code.
* Record count.
* Fetch timestamp.
* Source metadata.
* Normalized table rows.

Good Chroma candidates:

* Cleaned page text.
* Article bodies.
* Extracted paragraphs.
* Documentation chunks.
* Search result snippets.
* Generated retrieval context.

A common pattern is:

```text
fetchers.py
    |
    v
normalized records or text
    |
    +--> data.py / SQLite
    |
    +--> data.py / Chroma
```

## 🧾 Relationship to Writers

Fetcher results can be exported by `writers.py`.

Good writer outputs include:

* Markdown research notes.
* Source inventories.
* API response summaries.
* Web page summaries.
* Link inventories.
* Extracted tables.
* Public-data snapshots.

Fetchers should not grow unrelated writer logic. They should return data; writers should serialize
it.

## 📖 API Documentation

The generated API reference for this module is rendered below.
