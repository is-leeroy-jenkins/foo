# User Interface

`app.py` is the Foo application entry point. It defines the Streamlit user interface, initializes
session state, renders workflow controls, coordinates user-selected modes, calls the supporting
service modules, and displays results.

This page documents `app.py` as an application orchestration layer. It is intentionally separate
from the generated API reference because `app.py` contains Streamlit runtime code and may execute UI
logic at import time. That makes it better suited for manual application documentation than direct
`mkdocstrings` rendering unless the file is later refactored into import-safe functions.

## 🧭 Purpose

The purpose of `app.py` is to provide the user-facing Foo application.

It coordinates:

* Page setup.
* Session-state initialization.
* Sidebar controls.
* Mode selection.
* Document loading workflows.
* Web scraping workflows.
* Retrieval and public-data workflows.
* Geospatial workflows.
* Environmental workflows.
* Astronomical workflows.
* Demographic and population workflows.
* AI generation workflows.
* Persistence workflows.
* Result rendering.
* Tables, charts, metrics, previews, and status messages.

`app.py` is not just a launcher. It is the top-level orchestration file that connects the Streamlit
UI to Foo’s service modules.

## 🧱 Application Role

Foo is organized around a Streamlit application layer and a set of supporting modules.

```text
app.py
  |
  +--> config.py
  +--> core.py
  +--> loaders.py
  +--> fetchers.py
  +--> scrapers.py
  +--> generators.py
  +--> data.py
  +--> models.py
  +--> writers.py
```

The application layer should remain responsible for user interaction and workflow coordination.

The supporting modules should remain responsible for the actual domain work:

| Module          | Role                                                                        |
| --------------- | --------------------------------------------------------------------------- |
| `config.py`     | Application constants, paths, UI labels, options, and configuration values. |
| `core.py`       | Shared validation and result primitives.                                    |
| `loaders.py`    | Document and source ingestion.                                              |
| `fetchers.py`   | Web, API, public-data, and provider retrieval.                              |
| `scrapers.py`   | HTML extraction.                                                            |
| `generators.py` | AI-provider workflows.                                                      |
| `data.py`       | SQLite and Chroma persistence.                                              |
| `models.py`     | Structured Pydantic models.                                                 |
| `writers.py`    | Output serialization.                                                       |

## ⚙️ Application Modes

Foo uses mode-based navigation to separate major workflow areas.

The configured application modes are:

| Mode              | Purpose                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `Loading`         | Load source content from files, documents, notebooks, cloud sources, and related loaders. |
| `Scraping`        | Extract content from web pages and HTML structures.                                       |
| `Retrieval`       | Retrieve data from public collections, archives, APIs, search, and reference sources.     |
| `Geospatial`      | Retrieve weather, maps, geocoding, and location-oriented information.                     |
| `Demographic`     | Retrieve health, population, Census, and related public-data records.                     |
| `Environmental`   | Retrieve environmental, air, water, climate, earthquake, and related public datasets.     |
| `Astronomical`    | Retrieve astronomy, sky, satellite, observatory, and space-related data.                  |
| `Generation`      | Run AI generation workflows through supported provider wrappers.                          |
| `Data Management` | Inspect, create, query, update, and manage local data resources.                          |

These modes allow the UI to expose many capabilities without mixing every control into one page.

## 🖥️ Streamlit Layout

The Streamlit interface is responsible for presenting the application in a structured way.

`app.py` uses Streamlit elements such as:

* Sidebar controls.
* Select boxes.
* Radio controls.
* Buttons.
* File uploaders.
* Expanders.
* Tabs.
* Dataframes.
* Tables.
* Metrics.
* Charts.
* Status and error messages.
* Markdown output.
* JSON previews.
* Code blocks.

The UI should collect user intent, validate enough input to prevent obvious misuse, call the
appropriate service class, and display the result.

## 🧠 Session State

`app.py` uses `st.session_state` extensively. Session state is the runtime memory of the Streamlit
application.

Session state tracks values such as:

* Active mode.
* Selected model.
* Generation settings.
* Loader results.
* Loaded documents.
* Raw text.
* Processed text.
* Tokens.
* Vocabulary.
* Token counts.
* Loader paths.
* Uploaded files.
* Scraper results.
* Fetcher results.
* Provider-specific inputs.
* Data-management selections.
* SQLite table state.
* Visualization state.
* Clear/reset flags.

Because Streamlit reruns the script after user interaction, session state is necessary to preserve
workflow context between reruns.

## 📥 Loading Mode

Loading mode connects the UI to the loader classes in `loaders.py`.

This mode supports workflows that begin with files, source paths, cloud documents, notebooks,
web-accessible documents, or source-specific loader inputs.

Typical loading workflow:

```text
User selects loader
        |
        v
User provides path, file, URI, or source reference
        |
        v
app.py validates and calls the loader
        |
        v
loader returns documents or source content
        |
        v
app.py displays, chunks, tokenizes, or stores results
```

Loading mode may support downstream operations such as:

* Viewing raw document text.
* Promoting loaded documents into session state.
* Clearing active documents.
* Chunking text.
* Tokenizing text.
* Creating vocabulary summaries.
* Displaying token counts.
* Passing content into generation or storage workflows.

The loader classes should perform ingestion. `app.py` should coordinate the UI and user-facing
result display.

## 🧹 Scraping Mode

Scraping mode connects the UI to HTML extraction workflows.

This mode supports workflows that begin with a URL and selected extraction methods.

Typical scraping workflow:

```text
User enters URL
        |
        v
User selects extraction methods
        |
        v
app.py calls scraper or web-fetch helper logic
        |
        v
HTML is retrieved and parsed
        |
        v
headings, paragraphs, links, images, tables, or other elements are extracted
        |
        v
app.py displays extracted results
```

Scraping mode is useful for:

* Page outlines.
* Paragraph extraction.
* Link inventories.
* Image inventories.
* Table extraction.
* Documentation migration.
* Source discovery.
* Web content review.

Scraping should remain separate from general public-data retrieval. If the workflow is about an API
or external data service, use a fetcher. If the workflow is about extracting HTML elements, use
scraping.

## 🌐 Retrieval Mode

Retrieval mode connects the UI to source-specific fetcher classes in `fetchers.py`.

This mode supports public collections, archives, search providers, reference sources, and general
retrieval workflows.

Examples include:

* Wikipedia.
* News retrieval.
* Google Search.
* Google Drive.
* Grokipedia.
* Internet Archive.
* Government data sources.
* Congressional data.
* Open science sources.

Typical retrieval workflow:

```text
User selects retrieval source
        |
        v
User provides source-specific query or options
        |
        v
app.py calls the selected fetcher
        |
        v
fetcher retrieves and normalizes data
        |
        v
app.py displays result payloads, summaries, or tables
```

The fetcher should know how to call the source. The UI should know how to present the options and
results.

## 🗺️ Geospatial Mode

Geospatial mode supports location, weather, mapping, and coordinate-oriented workflows.

This mode may coordinate fetchers such as:

* `GoogleMaps`
* `GoogleGeocoding`
* `GoogleWeather`
* `OpenWeather`
* `HistoricalWeather`
* `TidesAndCurrents`
* `UvIndex`

Common tasks include:

* Address lookup.
* Reverse geocoding.
* Directions.
* Weather lookup.
* Forecast retrieval.
* Historical weather queries.
* Tide and current records.
* UV index lookup.

Typical geospatial workflow:

```text
User selects geospatial source
        |
        v
User enters address, coordinates, ZIP code, dates, or mode-specific options
        |
        v
app.py calls the appropriate fetcher
        |
        v
fetcher returns structured location or weather data
        |
        v
app.py displays maps, tables, metrics, or summaries
```

## 🌱 Environmental Mode

Environmental mode supports environmental, climate, air, water, fire, and Earth-science data
retrieval.

This mode may coordinate fetchers such as:

* `USGSEarthquakes`
* `USGSWaterData`
* `USGSTheNationalMap`
* `USGSScienceBase`
* `AirNow`
* `ClimateData`
* `EoNet`
* `EnviroFacts`
* `PurpleAir`
* `OpenAQ`
* `Firms`
* `EarthObservatory`

Common tasks include:

* Earthquake feed retrieval.
* Water monitoring data retrieval.
* National Map data retrieval.
* ScienceBase item lookup.
* Air-quality data retrieval.
* Climate dataset lookup.
* Event and hazard discovery.
* Environmental facility or table lookup.
* Fire and thermal anomaly retrieval.

The UI should present source-specific controls clearly because environmental fetchers often require
different combinations of dates, coordinates, identifiers, state codes, bounding boxes, or result
limits.

## ✨ Astronomical Mode

Astronomical mode supports sky, astronomy, satellite, observatory, and space-data workflows.

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

Common tasks include:

* Astronomical object lookup.
* Sky-coordinate lookup.
* Cone search.
* Star chart retrieval.
* Near-Earth object lookup.
* Fireball data lookup.
* Space-weather data retrieval.
* Satellite or observatory-related data lookup.
* Aviation state or track lookup where applicable.

Typical astronomical workflow:

```text
User selects astronomy or sky source
        |
        v
User provides object name, coordinates, radius, date, or source-specific options
        |
        v
app.py calls the relevant fetcher
        |
        v
fetcher returns records, links, coordinates, charts, or observations
        |
        v
app.py displays the result
```

## 👥 Demographic Mode

Demographic mode supports population, health, Census, and public-health-oriented datasets.

This mode may coordinate fetchers such as:

* `CensusData`
* `Socrata`
* `HealthData`
* `GlobalHealthData`
* `UnitedNations`
* `WorldPopulation`
* `Wonder`

Common tasks include:

* Census variable lookup.
* Census data retrieval.
* Socrata metadata and row retrieval.
* Health dataset lookup.
* Global health indicator retrieval.
* United Nations dataset queries.
* Population catalog retrieval.
* CDC WONDER-style query workflows.

Demographic workflows often require careful parameter selection. The UI should make source, dataset,
geography, year, predicates, fields, limits, and filters visible to the user where applicable.

## 🤖 Generation Mode

Generation mode connects the UI to provider wrappers in `generators.py`.

This mode may coordinate classes such as:

* `Chat`
* `Grok`
* `Gemini`
* `Claude`
* `Mistral`

The UI may expose controls for:

* Prompt input.
* System instruction input.
* Model selection.
* Temperature.
* Maximum tokens.
* Top-percent or top-p style settings.
* Frequency penalty.
* Presence penalty.
* Response format.
* Tool choice.
* Web search.
* Allowed domains.
* Reasoning or thinking configuration.
* Streaming.
* Storage options.

Typical generation workflow:

```text
User selects provider/model
        |
        v
User enters prompt and options
        |
        v
app.py calls provider wrapper
        |
        v
provider wrapper builds request and extracts response
        |
        v
app.py displays generated output
```

The provider wrapper should own provider-specific request construction. `app.py` should own the
controls and display.

## 🧰 Utility Functions

`app.py` includes utility functions that support display, normalization, text processing, database
operations, and result rendering.

Examples include:

| Utility Area         | Representative Functions                                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Provider invocation  | `_filter_kwargs_for_callable`, `_invoke_provider`                                                                                        |
| Result rendering     | `_render_output`, `_render_result_metadata`, `_render_summary_kv`, `_render_rows_table`                                                  |
| Preview rendering    | `_render_xml_preview`, `_render_html_preview`, `_render_fallback_raw`                                                                    |
| Model selection      | `_model_selector`                                                                                                                        |
| Search display       | `render_google_results`                                                                                                                  |
| Styling              | `style_subheaders`, `set_blue_divider`                                                                                                   |
| Image handling       | `encode_image`                                                                                                                           |
| Text processing      | `normalize_text`, `chunk_text`, `sanitize_markdown`, `normalize`                                                                         |
| Similarity           | `cosine_similarity`                                                                                                                      |
| UI metrics           | `metric_with_tooltip`                                                                                                                    |
| Database operations  | `initialize_database`, `create_connection`, `list_tables`, `read_table`, `render_table`                                                  |
| Dataframe operations | `apply_filters`, `create_aggregation`, `create_visualization`, `convert_dataframe`                                                       |
| SQLite maintenance   | `create_custom_table`, `is_safe_query`, `create_identifier`, `get_indexes`, `add_column`, `rename_column`, `drop_column`, `rename_table` |
| Loader state         | `_promote_loader_documents`, `_clear_loader_documents`                                                                                   |

These functions support the Streamlit page. If a helper grows into reusable business logic, consider
moving it into an appropriate module.

## 🔄 Result Rendering

`app.py` includes several rendering helpers that make heterogeneous results displayable.

Different providers and source modules may return:

* Strings.
* Dictionaries.
* Lists.
* Tuples.
* DataFrames.
* XML-like content.
* HTML-like content.
* JSON-like payloads.
* Response objects.
* Custom result containers.
* `None`.

The rendering helpers exist so the UI can display results predictably without forcing every provider
or loader to return exactly the same object type.

Good result rendering should:

* Preserve useful metadata.
* Avoid crashing on unexpected shapes.
* Provide tables where data is tabular.
* Provide readable previews for long text.
* Avoid dumping excessively large payloads without truncation or expanders.
* Present errors clearly.

## 🧾 Text Processing

`app.py` includes text helper functions for normalization, chunking, tokenization-related display,
and Markdown sanitization.

These utilities support workflows where loaded, scraped, fetched, or generated text needs to be
processed before display, storage, or downstream generation.

Common text-processing tasks include:

* Normalizing whitespace.
* Splitting text into chunks.
* Counting tokens.
* Creating vocabulary summaries.
* Cleaning Markdown.
* Preparing text for display.
* Comparing text similarity.

If text-processing behavior becomes central and reusable, consider moving it to a dedicated module.

## 📊 Visualization

The application includes dataframe and visualization support for persistence workflows.

Visualization helpers may use tabular data to produce:

* Tables.
* Aggregations.
* Charts.
* Metrics.
* Filtered views.
* Converted downloadable data.

`app.py` is an appropriate place for user-facing visualization assembly because visualization is a
UI responsibility. Data retrieval and database operations should still remain in the appropriate
service modules.

## 🔐 Error Handling

`app.py` should display user-facing errors clearly, but reusable service-layer error handling should
remain in the service modules.

Application-level error handling should:

* Show concise user-facing messages.
* Avoid exposing secrets.
* Avoid dumping full tracebacks into the UI by default.
* Avoid displaying raw credentials, tokens, API keys, or private file contents.
* Preserve enough context for the user to understand what failed.
* Delegate structured exception logging to service modules where that pattern already exists.

When helper functions in `app.py` catch exceptions directly, they should use stable diagnostic
language and avoid logging or displaying sensitive content.

## 🛠️ Development Guidance

When modifying `app.py`, follow these rules:

* Keep Streamlit UI logic in `app.py`.
* Keep source-specific loading behavior in `loaders.py`.
* Keep public-data and API retrieval behavior in `fetchers.py`.
* Keep HTML extraction behavior in `scrapers.py`.
* Keep AI-provider request construction in `generators.py`.
* Keep persistence behavior in `data.py`.
* Keep schema declarations in `models.py`.
* Keep output serialization in `writers.py`.
* Avoid duplicating provider logic in UI code.
* Avoid hard-coding user-specific paths.
* Avoid hard-coding credentials.
* Use stable session-state keys.
* Use unique Streamlit widget keys.
* Keep destructive actions clearly labeled.
* Prefer expanders or tabs for large control groups.

## 🧪 Testing the Application

Because `app.py` is a Streamlit application file, testing should include both syntax checks and
manual UI checks.

Run a compile check:

```powershell
python -m py_compile app.py
```

Run the application:

```powershell
streamlit run app.py
```

Then manually check:

* The page loads without import errors.
* Sidebar mode selection works.
* Each mode renders without crashing.
* Required inputs are validated.
* Buttons have unique keys.
* Uploaders work for supported file types.
* Results display correctly.
* Clear/reset buttons reset the intended state.
* Data-management actions do not accidentally mutate the wrong table.
* Provider workflows fail gracefully when credentials are missing.
* Long results are displayed in tables, expanders, or previews.

## 🧪 Mode Review Checklist

Use this checklist when reviewing changes to `app.py`.

| Area            | Check                                                                    |
| --------------- | ------------------------------------------------------------------------ |
| Startup         | App starts with `streamlit run app.py`.                                  |
| Session state   | Required keys are initialized before use.                                |
| Navigation      | Mode selector changes the visible workflow.                              |
| Loading         | Loader controls call the correct loader classes.                         |
| Scraping        | Scraper controls extract expected HTML elements.                         |
| Retrieval       | Fetcher controls pass valid arguments.                                   |
| Geospatial      | Location and coordinate inputs are validated.                            |
| Environmental   | Source-specific environmental controls are clear.                        |
| Astronomical    | Object, coordinate, and radius inputs are handled safely.                |
| Demographic     | Dataset, year, field, and geography inputs are handled safely.           |
| Generation      | Provider, model, prompt, tools, and reasoning settings work as expected. |
| Output          | Results are displayed in readable form.                                  |
| Errors          | Failures show useful but safe user-facing messages.                      |

## 📦 Dependency Awareness

`app.py` imports many runtime dependencies because it coordinates the application.

Dependency categories include:

* Streamlit UI.
* Dataframes and visualization.
* HTML and XML parsing.
* Natural language processing.
* LangChain document objects.
* Provider wrappers.
* Loader classes.
* Fetcher classes.
* Scraper utilities.
* SQLite.
* Configuration.

When adding imports to `app.py`, consider whether the dependency belongs in a service module
instead. A dependency used only by one loader, fetcher, generator, scraper, writer, or data class
should usually live in that module, not in `app.py`.

## 📚 Documentation Relationship

`app.py` should be documented manually through this page rather than automatically rendered with
`mkdocstrings` unless it is refactored to be import-safe.

Recommended documentation structure:

```text
docs/app.md                  Manual Streamlit application documentation.
docs/api/core.md             Generated API reference.
docs/api/data.md             Generated API reference.
docs/api/fetchers.md         Generated API reference.
docs/api/generators.md       Generated API reference.
docs/api/loaders.md          Generated API reference.
docs/api/models.md           Generated API reference.
docs/api/scrapers.md         Generated API reference.
docs/api/writers.md          Generated API reference.
```

If `app.py` is later refactored into an import-safe package module with pure functions, a generated
API page can be reconsidered.

## 🚧 Refactoring Guidance

`app.py` is large and carries many responsibilities because Streamlit applications often grow around
UI state and mode-specific controls.

If the file becomes difficult to maintain, consider refactoring by extracting mode renderers into
separate modules:

```text
ui/
  loading_page.py
  scraping_page.py
  retrieval_page.py
  geospatial_page.py
  environmental_page.py
  astronomical_page.py
  demographic_page.py
  generation_page.py
  data_management_page.py
```

A refactor should preserve behavior and move code gradually.

Good candidates for extraction include:

* Mode-specific render functions.
* Repeated widget groups.
* Data-management UI helpers.
* Visualization builders.
* Result preview components.
* Provider-specific UI panels.
* Loader-specific UI panels.
* Fetcher-specific UI panels.

Refactoring should not change runtime behavior unless the change is intentional and tested.

## ✅ Maintenance Checklist

Before accepting changes to `app.py`, confirm:

* The file compiles.
* Streamlit starts successfully.
* Sidebar mode selection works.
* Each mode renders.
* Session-state keys are initialized before use.
* Widget keys are unique.
* Provider options are synchronized with `config.py`.
* Loader options are synchronized with `loaders.py`.
* Fetcher options are synchronized with `fetchers.py`.
* Generator options are synchronized with `generators.py`.
* Data-management actions are safe and clear.
* Long result payloads use expanders, tables, or previews.
* Errors are user-safe.
* No credentials are hard-coded.
* No user-specific absolute paths are introduced.
* Documentation pages remain consistent with the UI modes.

## 🧭 Summary

`app.py` is the user-facing Foo application. It owns the Streamlit interface, session state, mode
selection, UI controls, workflow orchestration, and result display. The supporting modules own the
reusable logic. Keeping that separation clear makes Foo easier to maintain, document, test, and
eventually refactor.
