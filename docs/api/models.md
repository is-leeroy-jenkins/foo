# Models API

The `models.py` module defines Foo’s structured data models. These models provide typed containers
for prompts, files, errors, reasoning options, messages, locations, coordinates, forecasts,
directions, tools, function schemas, and provider-oriented search or computer-use objects.

The module is intentionally declarative. It should describe structured application objects, not
perform network calls, render Streamlit components, write files, manage databases, or call AI
providers directly.

## 🧭 Purpose

The Models API provides a common schema layer for Foo.

The module supports:

* Prompt representation.
* File metadata representation.
* Error object representation.
* Reasoning configuration.
* Document metadata.
* Chat or message payloads.
* Location and coordinate data.
* Weather forecast data.
* Directions data.
* Sky-coordinate data.
* Tool and function schema objects.
* File-search configuration.
* Web-search configuration.
* Computer-use configuration.

Models help keep data structures explicit and predictable across the rest of the application.

## 🧱 Module Role

`models.py` sits below the service modules as a shared schema layer.

```text
Streamlit UI
    |
    +--> loaders.py
    +--> fetchers.py
    +--> scrapers.py
    +--> generators.py
    +--> data.py
    +--> writers.py
              |
              v
          models.py
```

The models should be safe to import anywhere in Foo because they should not perform runtime side
effects.

## 🧩 Model Overview

The module defines the following primary model classes:

| Model            | Purpose                                                                     |
| ---------------- | --------------------------------------------------------------------------- |
| `Prompt`         | Represents prompt content and related prompt metadata.                      |
| `File`           | Represents file metadata used by source, provider, or workflow operations.  |
| `Error`          | Represents structured error information.                                    |
| `Reasoning`      | Represents reasoning or effort configuration for providers that support it. |
| `Document`       | Represents document-style content or document metadata.                     |
| `Message`        | Represents a chat or conversation message.                                  |
| `Location`       | Represents a named or address-like location.                                |
| `GeoCoordinates` | Represents latitude and longitude coordinate data.                          |
| `Forecast`       | Represents weather forecast fields.                                         |
| `Directions`     | Represents route or directions data.                                        |
| `SkyCoordinates` | Represents astronomical coordinate data.                                    |
| `Tool`           | Represents a provider tool definition.                                      |
| `Function`       | Represents a callable function schema.                                      |
| `FileSearch`     | Represents file-search tool configuration.                                  |
| `WebSearch`      | Represents web-search tool configuration.                                   |
| `ComputerUse`    | Represents computer-use tool configuration.                                 |

These classes provide typed structure for data that may otherwise be passed around as loosely shaped
dictionaries.

## 📦 Pydantic Model Pattern

Foo models are Pydantic-style schema classes.

A model should:

* Define clear field names.
* Use explicit type annotations.
* Provide defaults only when a field is truly optional or has a stable default.
* Avoid hidden network, filesystem, or provider side effects.
* Keep validation declarative when possible.
* Keep operational behavior in the modules that use the models.

Conceptual pattern:

```python
from models import Message

message = Message(
    role="user",
    content="Summarize this document.",
)

print(message)
```

Use the generated API reference at the bottom of this page for exact field definitions.

## 📝 Prompt

`Prompt` represents prompt-oriented data.

Use it when a workflow needs to pass prompt text or prompt metadata through the application in a
structured way.

Typical uses include:

* Text-generation prompts.
* Search-assisted generation prompts.
* Document summarization instructions.
* Translation prompts.
* Analysis instructions.
* Workflow-specific generation inputs.

Conceptual example:

```python
from models import Prompt

prompt = Prompt(
    text="Summarize the loaded document.",
)
```

The exact available fields are documented in the rendered API reference.

## 📁 File

`File` represents file metadata.

Use it when the workflow needs to describe a file without necessarily loading the entire file body
into memory.

Typical uses include:

* Uploaded files.
* Source files.
* Provider file references.
* File-search inputs.
* Document inventory records.
* File metadata persistence.
* Writer output references.

Common metadata concepts include:

* File name.
* File path.
* MIME type.
* Size.
* Provider identifier.
* Source reference.

Conceptual example:

```python
from models import File

source_file = File(
    name="example.pdf",
)
```

## ⚠️ Error

`Error` represents structured error metadata.

Use it when a workflow needs to pass error details through a typed object rather than as an
unstructured string.

Typical uses include:

* UI-safe error display.
* Logging metadata.
* Diagnostics.
* Workflow status records.
* Error-report serialization.

This model is separate from the `boogr.Error` exception wrapper used in the logging pattern. Do not
confuse the two:

| Object         | Purpose                                               |
| -------------- | ----------------------------------------------------- |
| `models.Error` | Pydantic-style structured error data.                 |
| `boogr.Error`  | Exception wrapper used for application error logging. |

## 🧠 Reasoning

`Reasoning` represents reasoning or effort configuration.

Use it for providers or workflows that expose reasoning-related options, such as:

* Effort level.
* Reasoning mode.
* Thinking budget.
* Reasoning configuration object.
* Provider-specific reasoning settings.

The generator wrappers should decide whether a selected provider or model supports reasoning. The
model class should only represent the configuration shape.

## 📄 Document

`Document` represents document-style data.

Use it when Foo needs a structured document object separate from third-party document types.

Typical uses include:

* Source document metadata.
* Loaded document content.
* Document references.
* Chunk metadata.
* Search result documents.
* Generation context documents.

This model should not replace LangChain `Document` objects where the workflow explicitly expects
LangChain types. It is useful when Foo needs its own application-level document schema.

## 💬 Message

`Message` represents a chat or conversation message.

Typical uses include:

* User messages.
* Assistant messages.
* System or developer instructions.
* Provider request payloads.
* Conversation history.
* Message display records.

Conceptual example:

```python
from models import Message

message = Message(
    role="user",
    content="Explain Foo's loader layer.",
)
```

Generator classes can use message-shaped data when constructing provider requests.

## 📍 Location

`Location` represents a named, address-like, or place-oriented location.

Typical uses include:

* Weather queries.
* Geocoding.
* Mapping workflows.
* Directions requests.
* Environmental data queries.
* Public-data lookups by place.

A location model should describe the place, not retrieve data for it. Retrieval belongs in
`fetchers.py`.

## 🌐 GeoCoordinates

`GeoCoordinates` represents latitude and longitude values.

Typical uses include:

* Weather requests.
* Map lookups.
* Geocoding results.
* Environmental data queries.
* Earth science data queries.
* Nearby object queries.
* Directions workflows.

Conceptual example:

```python
from models import GeoCoordinates

coordinates = GeoCoordinates(
    latitude=38.9072,
    longitude=-77.0369,
)
```

Coordinate validation should match the field definitions in the source model.

## 🌦️ Forecast

`Forecast` represents weather forecast data.

Typical uses include:

* Current weather display.
* Forecast display.
* Weather-fetcher results.
* Weather summaries.
* Local storage of weather records.
* Writer export of weather snapshots.

The fetcher layer should retrieve forecast data. The model should represent the result shape.

## 🧭 Directions

`Directions` represents route or directions data.

Typical uses include:

* Mapping results.
* Route summaries.
* Distance and duration records.
* Turn-by-turn output where supported.
* Geospatial workflow results.

Directions retrieval belongs in `fetchers.py`. The model should remain a schema object.

## ✨ SkyCoordinates

`SkyCoordinates` represents astronomical coordinate data.

Typical uses include:

* Star maps.
* Astronomy catalog lookups.
* Coordinate-based sky charts.
* Observatory workflows.
* Near-object or region-search workflows.

This model supports the astronomy and space-oriented fetchers without embedding fetcher behavior in
the schema itself.

## 🧰 Tool

`Tool` represents a provider tool definition.

Typical uses include:

* Web-search tool configuration.
* File-search tool configuration.
* Function-calling tool definitions.
* Provider request payloads.
* Tool-selection UI state.
* Tool metadata persistence.

Tool execution should remain in the generator or fetcher layer. The model should only represent the
tool configuration.

## 🔧 Function

`Function` represents a callable function schema.

Typical uses include:

* Function-calling metadata.
* Tool schema construction.
* Provider function definitions.
* Input parameter schemas.
* Required parameter declarations.

Conceptual shape:

```text
Function
    name
    description
    parameters
```

The exact fields are documented in the API reference.

## 🔎 FileSearch

`FileSearch` represents file-search tool configuration.

Typical uses include:

* Provider-backed file-search workflows.
* Search options.
* Vector-store references.
* File context configuration.
* Search-result controls.

File search may involve external provider behavior, vector stores, or local retrieval systems. The
model should represent configuration, while the generator or data layer performs the actual
operation.

## 🌍 WebSearch

`WebSearch` represents web-search tool configuration.

Typical uses include:

* Allowed domain configuration.
* Search context-size options.
* Provider-backed web-search tools.
* Search-assisted generation.
* Current-information workflows.

The model should not call the web directly. Fetching and provider calls belong in `fetchers.py` or
`generators.py`.

## 🖥️ ComputerUse

`ComputerUse` represents computer-use tool configuration.

Typical uses include:

* Tool metadata.
* Environment constraints.
* Provider tool payloads.
* Computer-use workflow configuration.

Computer-use capabilities should be handled carefully. The model should only describe the
configuration shape and should not execute actions.

## 🔗 Relationship to Generators

Generator wrappers may use model classes to shape provider requests.

Common relationships include:

| Model         | Generator Use                              |
| ------------- | ------------------------------------------ |
| `Prompt`      | Prompt input and metadata.                 |
| `Message`     | Conversation messages.                     |
| `Reasoning`   | Reasoning or thinking configuration.       |
| `Tool`        | Provider tool definitions.                 |
| `Function`    | Function-calling schemas.                  |
| `FileSearch`  | File-search tool options.                  |
| `WebSearch`   | Web-search tool options.                   |
| `ComputerUse` | Computer-use tool options.                 |
| `File`        | Provider file references or file metadata. |

Provider wrappers should convert models into provider-specific request payloads when needed.

## 🔗 Relationship to Fetchers

Fetcher classes may use model classes to represent structured results or request parameters.

Common relationships include:

| Model            | Fetcher Use                                               |
| ---------------- | --------------------------------------------------------- |
| `Location`       | Location-based weather, mapping, and public-data queries. |
| `GeoCoordinates` | Coordinate-based requests.                                |
| `Forecast`       | Weather result representation.                            |
| `Directions`     | Mapping or route result representation.                   |
| `SkyCoordinates` | Astronomy and sky-coordinate workflows.                   |
| `Error`          | Structured diagnostic payloads.                           |

Fetchers should perform retrieval. Models should represent request or response structure.

## 🔗 Relationship to Loaders

Loader classes may use model classes to represent source files, document records, or metadata.

Common relationships include:

| Model      | Loader Use                                 |
| ---------- | ------------------------------------------ |
| `File`     | Source file metadata.                      |
| `Document` | Application-level document representation. |
| `Error`    | Structured load error payloads.            |

Where a loader returns LangChain `Document` objects, the Foo `Document` model should not be
substituted unless the workflow expects the Foo model.

## 🔗 Relationship to Persistence

Persistence workflows may serialize model instances into SQLite rows or vector-store metadata.

Good candidates for serialization include:

* File metadata.
* Prompt metadata.
* Message records.
* Location records.
* Forecast records.
* Tool configuration.
* Search configuration.
* Error summaries.

Before persisting model data, decide whether the value belongs in:

```text
SQLite
- Structured fields
- Metadata
- Audit-friendly rows

Chroma
- Text content
- Embedded document chunks
- Semantic-retrieval records
```

## 🔗 Relationship to Writers

Writer classes can serialize model instances into Markdown, JSON-like summaries, reports, or
documentation artifacts.

Examples include:

* Prompt inventories.
* Tool configuration summaries.
* File inventories.
* Message transcripts.
* Forecast summaries.
* Location records.
* Error reports.

The writer layer should handle formatting and output. The model layer should remain declarative.

## 🛡️ Model Design Guidance

When adding or modifying models:

* Keep models declarative.
* Use explicit type annotations.
* Use defaults only when they are meaningful.
* Avoid provider calls.
* Avoid file I/O.
* Avoid database access.
* Avoid Streamlit rendering.
* Avoid hidden network requests.
* Keep class docstrings Google-style.
* Use field names that match application vocabulary.
* Avoid duplicating third-party objects unless Foo needs its own schema.
* Keep sensitive fields clearly named so callers know what not to log.

## 🔐 Sensitive Data Guidance

Models may carry sensitive or user-provided values.

Do not blindly log full model instances when they may contain:

* API keys.
* Tokens.
* Provider credentials.
* Prompt text containing private content.
* File contents.
* Uploaded document text.
* User messages.
* Location-sensitive details.
* Raw provider responses.
* Error traces containing sensitive paths.

When logging model-related failures, log stable metadata such as module, class, method, and safe
status information rather than full payloads.

## 🧪 Testing Models

Model tests should verify:

* Required fields are enforced.
* Optional fields use correct defaults.
* Type annotations match expected input.
* Serialization produces expected keys.
* Invalid values fail predictably where validation exists.
* Models do not perform side effects during construction.
* Models can be imported without initializing providers or external services.
* API documentation renders model fields correctly.

If a model is intended to support provider request payloads, test conversion at the
generator-wrapper level rather than adding provider behavior to the model itself.

## 📦 Dependency Guidance

The models layer should remain dependency-light.

Pydantic is appropriate because the module is a schema layer. Avoid adding heavy runtime
dependencies to `models.py` unless the model itself requires them for validation or field
representation.

Do not add dependencies for:

* HTTP requests.
* HTML parsing.
* Streamlit UI rendering.
* Database access.
* AI provider clients.
* File parsing.
* Cloud storage access.

Those dependencies belong in the modules that perform those operations.

## 📖 API Documentation

The generated API reference for this module is rendered below.
