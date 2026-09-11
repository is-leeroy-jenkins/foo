# User Guide

Foo is a Streamlit application for collecting, loading, scraping, retrieving, generating, storing,
inspecting, and exporting information for analysis and machine-learning workflows.

The application is launched from `app.py` and organized around workflow modes. Each mode exposes a
set of controls in the Streamlit interface and routes the user’s request to the appropriate Foo
module.

```text
app.py
  |
  +-- loaders.py       Document and source ingestion
  +-- fetchers.py      Web, API, public-data, science, geospatial, and provider retrieval
  +-- scrapers.py      HTML extraction
  +-- generators.py    AI-provider generation workflows
  +-- data.py          SQLite and Chroma persistence
  +-- writers.py       Markdown and output writing
  +-- models.py        Pydantic schema objects
  +-- core.py          Shared validation and result primitives
```

Use this guide when operating Foo from the Streamlit interface.

## 🚀 Starting Foo

Run Foo from the repository root.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies if they are not already installed:

```powershell
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```powershell
streamlit run app.py
```

Streamlit usually opens the application automatically. If it does not, open the local URL printed in
the terminal. It is usually:

```text
http://localhost:8501
```

Run Foo from the repository root so relative paths, local assets, configuration references, and
documentation paths resolve consistently.

## 🧭 Application Navigation

Foo uses mode-based navigation. The active mode determines which controls and workflows appear in
the interface.

The primary modes are:

| Mode            | Primary Purpose                                                                 | Main Source Module             |
| --------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| Loading         | Load files, documents, notebooks, cloud objects, and document-like sources.     | `loaders.py`                   |
| Scraping        | Extract headings, paragraphs, tables, links, images, and other HTML elements.   | `scrapers.py`                  |
| Retrieval       | Retrieve records from web, search, archive, public-data, and reference sources. | `fetchers.py`                  |
| Geospatial      | Work with weather, maps, geocoding, coordinates, tides, and location data.      | `fetchers.py`                  |
| Demographic     | Work with Census, health, population, United Nations, and public-health data.   | `fetchers.py`                  |
| Environmental   | Work with air, water, climate, fire, earthquake, and environmental data.        | `fetchers.py`                  |
| Astronomical    | Work with astronomy, sky, satellite, observatory, and space-weather data.       | `fetchers.py`                  |
| Generation      | Run AI-provider workflows.                                                      | `generators.py`                |
| Output          | Export processed results to durable artifacts.                                  | `writers.py`                   |

Some modes may expose multiple source-specific panels because Foo supports many provider and
data-source classes.

## 🖥️ Interface Layout

Foo uses Streamlit’s rerun model. Every interaction can rerun `app.py`, so the application uses
`st.session_state` to preserve selected options, loaded documents, user inputs, results, and
intermediate workflow values.

The interface generally has two areas:

| Area              | Purpose                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| Sidebar           | Select modes, providers, source classes, options, inputs, and actions.                                       |
| Main content area | Display explanations, previews, tables, charts, JSON payloads, generated text, errors, and result summaries. |

Session state may preserve:

* Active mode.
* Selected provider or loader.
* Model settings.
* Loaded documents.
* Raw and processed text.
* Scrape results.
* Fetch results.
* Database selections.
* Dataframe filters.
* Generated output.
* Token, vocabulary, and text-processing state.

Use reset or clear controls when switching workflows if stale state appears to affect the current
run.

## 📥 Loading Mode

Use **Loading** mode when the workflow begins with a file, document, notebook, cloud object,
repository source, or document-like source.

Loading mode is backed by `loaders.py`.

Supported loader categories include:

| Category                        | Examples                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| Plain and structured files      | Text, CSV, XML, JSON.                                                                            |
| Documents                       | PDF, Word, Markdown, HTML, PowerPoint.                                                           |
| Spreadsheets                    | Excel, CSV.                                                                                      |
| Web and research sources        | Web pages, ArXiv, Wikipedia, GitHub, PubMed, open city data.                                     |
| Cloud and office sources        | OneDrive, Google-backed content, Google Cloud files, AWS files, storage buckets, Outlook, email. |
| Notebook and audio-derived text | Jupyter notebooks, Google Speech-to-Text workflows.                                              |

A normal loading workflow is:

```text
Select Loading mode
        |
        v
Choose a loader
        |
        v
Provide source input
        |
        v
Run the load action
        |
        v
Review loaded documents, text, or metadata
        |
        v
Reuse, store, summarize, or export the result
```

Use Loading mode when the output should be document content or document-like records.

## 📄 Loading Example

To load a PDF:

1. Select **Loading** mode.
2. Select `PdfLoader`.
3. Provide the PDF path or upload the file if the UI exposes an upload control.
4. Run the loader.
5. Review the returned document text and metadata.
6. Decide whether to chunk, summarize, store, or export the content.

After loading, the content can move into:

* Streamlit preview.
* Text normalization.
* Chunking.
* Tokenization.
* AI summarization.
* SQLite metadata storage.
* Chroma semantic indexing.
* Markdown export.

## 🧹 Scraping Mode

Use **Scraping** mode when the workflow begins with a web page and the goal is to extract specific
HTML elements.

Scraping mode is backed by `scrapers.py` and related helper logic in `app.py`.

Common extraction targets include:

* Headings.
* Paragraphs.
* Lists.
* Tables.
* Articles.
* Sections.
* Divisions.
* Blockquotes.
* Hyperlinks.
* Images.

A normal scraping workflow is:

```text
Select Scraping mode
        |
        v
Enter a URL
        |
        v
Select extraction methods
        |
        v
Run the scrape
        |
        v
Review extracted values
```

Use scraping when page structure matters. For example, scraping is appropriate for building a link
inventory, collecting all page headings, extracting a table, or locating image references.

## 🔗 Scraping Example

To create a link and heading inventory:

1. Select **Scraping** mode.
2. Enter the target page URL.
3. Select `scrape_headings`.
4. Select `scrape_hyperlinks`.
5. Run the scrape.
6. Review the extracted headings and links.
7. Export the inventory if needed.

This is useful for documentation audits, migration planning, and source discovery.

## 🌐 Retrieval Mode

Use **Retrieval** mode when the workflow begins with a query, provider, archive, search tool, public
source, or API.

Retrieval mode is backed by `fetchers.py`.

Retrieval sources may include:

* Web fetchers.
* Web crawlers.
* Wikipedia.
* News sources.
* Google Search.
* Google Drive.
* Grokipedia.
* Internet Archive.
* GovData.
* Congress.
* Open science sources.

A normal retrieval workflow is:

```text
Select Retrieval mode
        |
        v
Choose a source
        |
        v
Enter query terms or provider-specific options
        |
        v
Run retrieval
        |
        v
Review returned records, links, summaries, tables, or payloads
```

Use Retrieval mode when the source has service-specific behavior and should not be treated as a
simple local document.

## 🧪 Retrieval Example

To retrieve reference content:

1. Select **Retrieval** mode.
2. Choose a source such as `Wikipedia`, `GoogleSearch`, or `InternetArchive`.
3. Enter the search query or source identifier.
4. Run the retrieval.
5. Review returned records.
6. Store metadata, export a summary, or use the result as generation context.

Use focused queries first. Large or vague queries can return results that are difficult to inspect.

## 🗺️ Geospatial Mode

Use **Geospatial** mode for location-oriented workflows.

This mode may coordinate fetchers such as:

* `GoogleMaps`
* `GoogleGeocoding`
* `GoogleWeather`
* `OpenWeather`
* `HistoricalWeather`
* `TidesAndCurrents`
* `UvIndex`

Common geospatial tasks include:

* Forward geocoding.
* Reverse geocoding.
* Directions lookup.
* Weather lookup.
* Forecast retrieval.
* Historical weather retrieval.
* Tide and current retrieval.
* UV index lookup.

Typical inputs include:

* Address.
* City and state.
* ZIP code.
* Latitude and longitude.
* Date or date range.
* Units.
* Radius.
* Provider-specific options.

A normal geospatial workflow is:

```text
Select Geospatial mode
        |
        v
Choose geospatial source or operation
        |
        v
Enter location or coordinate inputs
        |
        v
Run retrieval
        |
        v
Review maps, weather data, coordinates, route results, or tables
```

## 🌱 Environmental Mode

Use **Environmental** mode for environmental, climate, air, water, fire, earthquake, and
Earth-science data.

This mode may coordinate fetchers such as:

* `USGSEarthquakes`
* `USGSWaterData`
* `USGSTheNationalMap`
* `USGSScienceBase`
* `AirNow`
* `ClimateData`
* `EoNet`
* `EnviroFacts`
* `TidesAndCurrents`
* `UvIndex`
* `PurpleAir`
* `OpenAQ`
* `Firms`
* `EarthObservatory`

Common environmental tasks include:

* Retrieving earthquake feeds.
* Querying water monitoring locations.
* Retrieving National Map records.
* Retrieving ScienceBase items.
* Looking up air-quality observations.
* Querying climate datasets.
* Looking up environmental facilities or tables.
* Retrieving fire and thermal anomaly data.
* Reviewing NASA event records.

A normal environmental workflow is:

```text
Select Environmental mode
        |
        v
Choose environmental source
        |
        v
Enter required source-specific options
        |
        v
Run retrieval
        |
        v
Review returned data
```

Environmental sources often require specific identifiers, dates, bounding boxes, state codes,
coordinates, or result limits. Start with a small query.

## 👥 Demographic Mode

Use **Demographic** mode for Census, health, population, United Nations, and public-health-oriented
datasets.

This mode may coordinate fetchers such as:

* `CensusData`
* `Socrata`
* `HealthData`
* `GlobalHealthData`
* `UnitedNations`
* `WorldPopulation`
* `Wonder`

Common demographic tasks include:

* Looking up Census variables.
* Retrieving Census records.
* Querying Socrata datasets.
* Searching health-data portals.
* Retrieving global health indicators.
* Querying United Nations datasets.
* Reviewing population data.
* Building CDC WONDER-style query workflows.

A normal demographic workflow is:

```text
Select Demographic mode
        |
        v
Choose demographic or health source
        |
        v
Enter dataset, geography, year, variable, indicator, or filter options
        |
        v
Run retrieval
        |
        v
Review records, tables, or summaries
```

Because demographic datasets can be large, begin with narrow filters and small result limits.

## ✨ Astronomical Mode

Use **Astronomical** mode for astronomy, sky, satellite, observatory, space-weather, and related
data.

This mode may coordinate fetchers such as:

* `NavalObservatory`
* `SatelliteCenter`
* `NearbyObjects`
* `OpenScience`
* `SpaceWeather`
* `AstroCatalog`
* `AstroQuery`
* `StarMap`
* `StarChart`
* `OpenSky`

Common astronomical tasks include:

* Object search.
* Cone search.
* Coordinate lookup.
* Star chart retrieval.
* Star map retrieval.
* Near-Earth object retrieval.
* Fireball data retrieval.
* Space-weather lookup.
* Observatory or satellite-related data lookup.
* Aviation state or track lookup where applicable.

A normal astronomical workflow is:

```text
Select Astronomical mode
        |
        v
Choose astronomy or sky source
        |
        v
Enter object name, coordinates, radius, date, or source-specific options
        |
        v
Run retrieval
        |
        v
Review returned records, coordinates, links, charts, or observations
```

Start with a simple object lookup before using broader coordinate or radius-based searches.

## 🤖 Generation Mode

Use **Generation** mode for AI-assisted workflows through provider wrappers in `generators.py`.

Supported provider wrapper classes include:

* `Chat`
* `Grok`
* `Gemini`
* `Claude`
* `Mistral`

Generation mode may expose controls for:

* Provider selection.
* Model selection.
* Prompt text.
* System instructions.
* Temperature.
* Maximum tokens.
* Top-percent or top-p style options.
* Frequency penalty.
* Presence penalty.
* Response format.
* Tool choice.
* Web search.
* Allowed domains.
* File search.
* Reasoning effort.
* Thinking configuration.
* Streaming.
* Image analysis.
* Image generation.
* Translation.
* Transcription.

A normal generation workflow is:

```text
Select Generation mode
        |
        v
Choose provider and model
        |
        v
Enter prompt and settings
        |
        v
Run generation
        |
        v
Review generated output
```

Generated output should be reviewed before use in official documentation, analysis, or deliverables.

## 🧠 Generation Guidance

Use a simple prompt first, then add constraints.

Good example:

```text
Summarize the loaded document in five bullets. Focus on purpose, inputs, outputs, dependencies, and risks.
```

For search-assisted generation, use domain limits when source control matters.

Good example:

```text
Search only official documentation sources and summarize the installation steps.
```

When using reasoning or thinking controls, confirm that the selected provider and model support
those settings. Provider support is not uniform.

## 🛡️ Data Safety

Separate read-only actions from mutation actions.

Read-oriented actions include:

* List tables.
* View schema.
* Read table.
* Preview rows.
* Filter rows.
* Aggregate rows.
* Visualize rows.
* View indexes.
* Create profile summaries.

Mutation-oriented actions include:

* Create table.
* Insert records.
* Drop table.
* Add column.
* Rename column.
* Drop column.
* Rename table.
* Create index.

Before running mutation actions, confirm the target table, operation, and expected result.

## 🧾 Output and Export

Use output workflows when you want to preserve a result outside the current Streamlit session.

The primary documented writer is `MarkdownWriter` in `writers.py`.

Good export candidates include:

* Loaded document summaries.
* Scraped page outlines.
* Link inventories.
* Extracted table summaries.
* Fetched API records.
* Public-data snapshots.
* Generated AI responses.
* SQLite table exports.
* Data profiles.
* Tool configuration summaries.

A strong Markdown output artifact should include:

```text
Title
Source
Method
Result
Metadata
Notes
```

Do not export secrets, credentials, raw private documents, or sensitive user content unless the
workflow explicitly requires it.

## 🔄 Combining Workflows

Foo is most useful when modes are combined.

### Load, Summarize, Export

```text
Loading mode
    |
    v
Load PDF or Word document
    |
    v
Generation mode
    |
    v
Summarize document
    |
    v
Output workflow
    |
    v
Write Markdown summary
```

### Scrape, Store, Search

```text
Scraping mode
    |
    v
Extract page paragraphs and links
    |
    v
the persistence API
    |
    v
Store metadata in SQLite
    |
    v
Chroma workflow
    |
    v
Store text chunks for semantic retrieval
```

### Retrieve, Analyze, Report

```text
Retrieval mode
    |
    v
Fetch public-data records
    |
    v
the persistence API
    |
    v
Filter and aggregate results
    |
    v
Output workflow
    |
    v
Write Markdown report
```

### Fetch, Generate, Document

```text
Retrieval mode
    |
    v
Fetch source material
    |
    v
Generation mode
    |
    v
Generate explanation or documentation draft
    |
    v
Output workflow
    |
    v
Write reviewable Markdown
```

## 🔍 Reviewing Results

Review results before storing, exporting, or using them in generation workflows.

Check:

* Source identifier.
* Query or input options.
* Returned status.
* Record count.
* Content preview.
* Metadata.
* Error messages.
* Missing values.
* Unexpected result shape.
* Sensitive content.
* Whether generated text needs revision.

A successful run does not guarantee that the result is complete, correct, or appropriate for
downstream use.

## 🔐 Credentials and Configuration

Some workflows require API keys, service credentials, provider settings, local files, or environment
variables.

Configuration should be managed through `config.py` or environment-backed values referenced by
`config.py`.

Do not hard-code credentials in:

* `app.py`
* Documentation pages.
* Example snippets.
* Output artifacts.
* Git commits.
* Markdown exports.

When a provider workflow fails, check for missing or invalid credentials, unsupported models,
missing optional dependencies, or service-specific configuration issues.

## ⚠️ Common Issues

| Issue                         | Likely Cause                                                                  | Suggested Check                                               |
| ----------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| App does not start            | Missing dependency or syntax error                                            | Run `python -m py_compile app.py` and reinstall requirements. |
| Mode renders but action fails | Missing source input or provider configuration                                | Check required fields and config values.                      |
| Loader returns no documents   | Bad path, unsupported file, malformed content, or missing optional dependency | Test with a small known-good file.                            |
| Scraper returns empty lists   | Page lacks selected elements or blocks automated access                       | Try headings or paragraphs first.                             |
| Fetcher returns error         | Bad query, provider issue, missing key, timeout, or unsupported parameter     | Simplify the request and check credentials.                   |
| Generator fails               | Missing API key, unsupported model, invalid option, or provider error         | Try a simple prompt with default options.                     |
| Data operation fails          | Invalid table, unsafe SQL, missing database, or schema mismatch               | Start with table listing and preview.                         |
| Markdown export looks wrong   | Unescaped content, broken table, or unclosed code fence                       | Preview Markdown before publishing.                           |

## 🧰 Recommended Work Habits

Use controlled, repeatable workflows.

1. Start with a small input.
2. Confirm the selected mode renders.
3. Run a simple operation.
4. Review the result.
5. Save or export only after review.
6. Increase query size or complexity gradually.
7. Keep output artifacts organized.
8. Avoid mixing temporary outputs with official documentation.
9. Keep credentials out of output and documentation.
10. Run documentation builds before publishing documentation changes.

## 📚 Related Documentation

Use these pages for deeper guidance:

* [Application](app.md)
* [Architecture](architecture.md)
* [Loading Data](loading.md)
* [Fetching and Scraping](fetching-scraping.md)
* [Generation](generation.md)
* [Output](outputs.md)
* [Development](development.md)
* [API Reference](api/index.md)

## ✅ User Checklist

Before ending a Foo session, confirm:

* Important results were reviewed.
* Needed outputs were exported.
* Local database changes were intentional.
* Sensitive content was not written unintentionally.
* Generated text was reviewed before use.
* Errors were captured or noted.
* Temporary files are separated from documentation files.
* Documentation changes still build with MkDocs if docs were edited.

## 🧭 Summary

Foo is a Streamlit application for moving information from source material into reviewable, reusable
outputs. Use Loading mode for documents, Scraping mode for HTML extraction, Retrieval and
specialized data modes for external sources, Generation mode for AI-assisted workflows, Data
Management mode for local persistence, and Output workflows for durable Markdown artifacts.
