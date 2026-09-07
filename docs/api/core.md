# Core API

The `core.py` module contains Foo’s smallest shared runtime primitives. It provides a required-value
validation helper and a lightweight `Result` container used to normalize HTTP response data for
downstream fetcher, scraper, writer, UI, and documentation workflows.

The core module should remain small, stable, and dependency-light. It is not the place for
provider-specific logic, Streamlit rendering, database operations, or source-specific loading
behavior.

## 🧭 Purpose

The core API exists to provide reusable primitives that multiple Foo modules can rely on without
creating circular or provider-specific dependencies.

The module currently provides:

| Object     | Type     | Purpose                                                                      |
| ---------- | -------- | ---------------------------------------------------------------------------- |
| `throw_if` | Function | Validates required argument values before a workflow continues.              |
| `Result`   | Class    | Wraps a `requests.Response` object in a small, inspectable result container. |

These primitives support the larger Foo architecture by keeping common validation and response
normalization behavior in one predictable location.

## 🧱 Module Responsibilities

`core.py` is responsible for:

* Required-value validation.
* Consistent validation error messages.
* Lightweight HTTP response result normalization.
* Basic result serialization through `to_dict()`.
* Simple response-content availability checks through `has_html`.

`core.py` is not responsible for:

* Streamlit UI rendering.
* Provider API calls.
* File loading.
* Web crawling.
* HTML parsing.
* Database persistence.
* Markdown writing.
* AI generation.
* Application configuration.

Keeping this module narrow makes it easier to reuse across loaders, fetchers, scrapers, writers, and
UI helper code.

## ✅ Required Argument Validation

The `throw_if(name, value)` function validates mandatory arguments.

It raises a `ValueError` when:

* The value is `None`.
* The value is a string that becomes empty after whitespace trimming.

This makes it useful for methods that need early, readable validation before performing file access,
network requests, database operations, or provider calls.

Example:

```python
from core import throw_if

def fetch_page(url: str) -> None:
	throw_if("url", url)

	# Continue only after url has passed basic validation.
```

The function is intentionally simple. It does not perform type validation, URL validation, path
validation, schema validation, or provider-specific checks. Those validations should remain in the
module or class that understands the workflow-specific requirements.

## 📦 Result Container

The `Result` class wraps a `requests.Response` object and exposes the most commonly used response
fields as direct attributes.

The constructor copies these fields from the source response:

| Attribute     | Description                                    |
| ------------- | ---------------------------------------------- |
| `response`    | Original `requests.Response` object.           |
| `url`         | Final response URL.                            |
| `status_code` | HTTP status code returned by the server.       |
| `text`        | Response body text.                            |
| `encoding`    | Response text encoding reported by `requests`. |
| `headers`     | Response headers from the source response.     |

This gives callers a stable object for passing response information between modules without
repeatedly reaching into the original response object.

Conceptual usage:

```python
import requests
from core import Result

response = requests.get("https://example.com", timeout=10)
result = Result(response)

print(result.url)
print(result.status_code)
print(result.has_html)
```

## 🔎 Serialization

`Result.to_dict()` converts a result into a plain dictionary.

The returned dictionary includes:

```text
url
status_code
text
encoding
headers
```

This is useful for:

* Streamlit display.
* Debugging.
* Testing.
* JSON-like serialization.
* Passing normalized response metadata to downstream workflows.
* Writing response summaries to output files.

Example:

```python
result_dict = result.to_dict()

print(result_dict["status_code"])
print(result_dict["headers"])
```

The `headers` value is converted to a standard dictionary so callers receive a detached mapping
rather than the mutable header object from the original response.

## 🧪 HTML Availability Check

The `Result.has_html` property returns `True` when the response text is represented as a string.

This is a lightweight content-availability check. It does not prove that the response body is valid
HTML. It only indicates that text content is available for possible parsing, scraping, display, or
export.

Example:

```python
if result.has_html:
	print(result.text)
```

For true HTML validation or parsing, use the scraper or fetcher utilities that understand HTML
structure.

## 🧰 Interactive Inspection

`Result.__dir__()` returns a stable list of visible member names.

The exposed names are:

```text
url
status_code
text
encoding
headers
has_html
to_dict
from_response
```

This helps with interactive inspection, development tooling, and UI surfaces that display available
members.

The current `__dir__()` output includes `from_response`, although the current `Result` class does
not define a `from_response` method. If that method is not planned, remove the name from `__dir__()`
in the source. If the method is planned, add it explicitly and document it.

## 🔗 Relationship to Fetchers

The `Result` class is most directly related to `fetchers.py`.

A fetcher that performs an HTTP request can wrap the returned `requests.Response` in `Result` before
passing it to the UI, scraper, writer, or persistence layer.

Typical relationship:

```text
WebFetcher
   |
   v
requests.Response
   |
   v
core.Result
   |
   +--> Streamlit display
   +--> HTML extraction
   +--> Markdown writing
   +--> Metadata persistence
```

This keeps response handling consistent across multiple retrieval workflows.

## 🔗 Relationship to Scrapers

Scrapers may use response text from a `Result` object as the raw material for parsing.

The relationship is:

```text
Result.text
   |
   v
BeautifulSoup / HTML parser
   |
   v
headings, paragraphs, tables, links, images, or cleaned text
```

The `Result` object should not perform the scraping itself. It should only carry response
information.

## 🔗 Relationship to Writers

Writers may use `Result.to_dict()` or selected `Result` attributes when exporting fetched content or
response summaries.

Examples of writer-friendly fields include:

* Source URL.
* Status code.
* Encoding.
* Headers.
* Response text.
* Extracted content produced from the response text.

The writer layer should remain responsible for formatting and output serialization.

## 🔗 Relationship to Persistence

The data-management layer can persist selected fields from a `Result` object.

Good SQLite candidates include:

* URL.
* Status code.
* Encoding.
* Header metadata.
* Fetch timestamp when supplied by the calling workflow.
* Processing status.
* Source identifier.

The full response body may be better stored as an output artifact or chunked into a vector store
when it is large or intended for retrieval workflows.

## 🧪 Usage Pattern: Validate, Fetch, Wrap

A common pattern is:

```python
import requests
from core import Result, throw_if

def get_result(url: str) -> Result:
	throw_if("url", url)

	response = requests.get(url, timeout=10)
	return Result(response)
```

This pattern keeps required-value validation explicit and keeps response normalization consistent.

## 🧪 Usage Pattern: Convert Result for Display

A result can be converted into a dictionary for display or inspection.

```python
payload = result.to_dict()

for key, value in payload.items():
	print(key, value)
```

This is useful when a UI needs a generic mapping rather than direct object attributes.

## 🧪 Usage Pattern: Guard Before Processing Text

Use `has_html` before passing response text into text-processing or HTML-processing logic.

```python
if result.has_html:
	html = result.text
else:
	html = ""
```

For workflows that require valid HTML, this should be followed by parser-level checks in the scraper
or fetcher module.

## ⚠️ Maintenance Notes

The core module should be changed carefully because it is used by other modules.

When modifying `core.py`:

* Preserve existing public names unless a breaking change is intentional.
* Keep `throw_if` simple and predictable.
* Avoid adding provider-specific validation to `throw_if`.
* Keep `Result` focused on response normalization.
* Avoid adding Streamlit, database, provider, or writer dependencies.
* Keep return annotations accurate.
* Keep docstrings compatible with MkDocs, mkdocstrings, and Griffe.
* Verify that any names returned by `__dir__()` actually exist on the class.

## 🧾 API Documentation

The generated API reference for this module is rendered below.
