# Loaders API

The `loaders.py` module is Foo’s document ingestion layer. It contains classes that load content
from local files, structured document formats, web resources, notebooks, cloud-backed files, storage
buckets, email sources, and source-specific content providers.

This module should remain focused on loading and normalizing source content. It should not contain
Streamlit layout code, general-purpose web API retrieval logic, database persistence behavior, AI
generation logic, or output-file serialization.

## 🧭 Purpose

The Loaders API provides source-specific classes for converting input material into
document-oriented content that Foo can display, process, chunk, store, search, summarize, or export.

The module supports loading from:

* Plain text files.
* CSV files.
* XML files.
* Web resources.
* PDF files.
* Excel workbooks.
* Word documents.
* Markdown files.
* HTML files.
* JSON files.
* ArXiv sources.
* Wikipedia sources.
* GitHub sources.
* PowerPoint files.
* Outlook and email sources.
* SharePoint Framework sources.
* OneDrive documents.
* Google-backed sources.
* PubMed search results.
* Open city data sources.
* Jupyter notebooks.
* Google Cloud files.
* AWS files.
* Google Speech-to-Text sources.
* Google Cloud Storage buckets.
* AWS S3 buckets.

The loader layer gives Foo a consistent place to isolate source-ingestion behavior.

## 🧱 Module Role

`loaders.py` sits between raw source material and the rest of the Foo application.

```text
Source material
      |
      v
loaders.py
      |
      +--> LangChain Document objects
      +--> raw text
      +--> structured records
      +--> metadata
      +--> downstream processing
```

The loader layer is usually the first step in a workflow that begins with documents or files rather
than a live API query.

## 🧩 Class Overview

The module defines a base loader class and multiple source-specific loader classes.

| Class                      | Purpose                                                                         |
| -------------------------- | ------------------------------------------------------------------------------- |
| `Loader`                   | Base loader class that defines common loader state and default loader behavior. |
| `TextLoader`               | Loads plain text content.                                                       |
| `CsvLoader`                | Loads CSV content.                                                              |
| `XmlLoader`                | Loads XML content.                                                              |
| `WebLoader`                | Loads web content into document-oriented objects.                               |
| `PdfLoader`                | Loads PDF content.                                                              |
| `ExcelLoader`              | Loads Excel workbook content.                                                   |
| `WordLoader`               | Loads Word document content.                                                    |
| `MarkdownLoader`           | Loads Markdown content.                                                         |
| `HtmlLoader`               | Loads HTML content.                                                             |
| `JsonLoader`               | Loads JSON content.                                                             |
| `ArXivLoader`              | Loads ArXiv source content.                                                     |
| `WikiLoader`               | Loads Wikipedia source content.                                                 |
| `GithubLoader`             | Loads GitHub source content.                                                    |
| `PowerPointLoader`         | Loads PowerPoint content.                                                       |
| `OutlookLoader`            | Loads Outlook-related content.                                                  |
| `WebCrawler`               | Loads crawled web content.                                                      |
| `SpfxLoader`               | Loads SharePoint Framework-oriented content.                                    |
| `OneDriveDocLoader`        | Loads OneDrive document content.                                                |
| `GoogleLoader`             | Loads Google-backed source content.                                             |
| `EmailLoader`              | Loads email source content.                                                     |
| `PubMedSearchLoader`       | Loads PubMed search results.                                                    |
| `OpenCityLoader`           | Loads open city data content.                                                   |
| `JupyterNotebookLoader`    | Loads Jupyter notebook content.                                                 |
| `GoogleCloudFileLoader`    | Loads files from Google Cloud.                                                  |
| `AwsFileLoader`            | Loads files from AWS-backed storage.                                            |
| `GoogleSpeechToTextLoader` | Loads speech-to-text content from Google services.                              |
| `GoogleBucketLoader`       | Loads content from Google Cloud Storage buckets.                                |
| `AwsBucketLoader`          | Loads content from AWS S3 buckets.                                              |

Use the generated API reference at the bottom of this page for exact constructor signatures, method
signatures, return annotations, and docstring-rendered details.

## ⚙️ Base Loader

The `Loader` class provides the common loader shape.

A loader typically tracks state such as:

* Source path.
* Source URI.
* Loaded documents.
* Raw text.
* Metadata.
* Chunking configuration.
* Loader-specific options.
* Runtime result state.

The base class gives source-specific loaders a consistent inheritance point and shared conceptual
contract.

Conceptual usage:

```python
from loaders import Loader

loader = Loader()

print(loader)
```

In normal workflows, use a concrete loader such as `TextLoader`, `PdfLoader`, `CsvLoader`, or
`GithubLoader` rather than using `Loader` directly.

## 📄 Local File Loaders

Several loader classes focus on local or file-like sources.

| Class                   | Source Type             |
| ----------------------- | ----------------------- |
| `TextLoader`            | Plain text files.       |
| `CsvLoader`             | CSV files.              |
| `XmlLoader`             | XML files.              |
| `PdfLoader`             | PDF files.              |
| `ExcelLoader`           | Excel workbooks.        |
| `WordLoader`            | Word documents.         |
| `MarkdownLoader`        | Markdown files.         |
| `HtmlLoader`            | HTML files.             |
| `JsonLoader`            | JSON files.             |
| `PowerPointLoader`      | PowerPoint files.       |
| `JupyterNotebookLoader` | Jupyter notebook files. |

Use these classes when the workflow begins with a file path, uploaded file, or known local document
source.

## 🧪 Example: Load a Text File

```python
from loaders import TextLoader

loader = TextLoader()
documents = loader.load("data/example.txt")

for document in documents or []:
    print(document.page_content)
```

Use `TextLoader` for simple text ingestion where no document-specific parsing is required beyond
reading text content and packaging it for processing.

## 🧪 Example: Load a CSV File

```python
from loaders import CsvLoader

loader = CsvLoader()
documents = loader.load("data/example.csv")

for document in documents or []:
    print(document.metadata)
    print(document.page_content)
```

CSV loading is useful when tabular rows need to be transformed into document-like records or
processed for search, summarization, analysis, or storage.

## 🧪 Example: Load a PDF

```python
from loaders import PdfLoader

loader = PdfLoader()
documents = loader.load("data/example.pdf")

for document in documents or []:
    print(document.page_content[:500])
```

PDF loading is appropriate for reports, manuals, public documents, research papers, forms, and other
document-style files.

## 🧪 Example: Load Markdown

```python
from loaders import MarkdownLoader

loader = MarkdownLoader()
documents = loader.load("docs/index.md")

for document in documents or []:
    print(document.page_content)
```

Markdown loading is useful for documentation analysis, site migration, static-site review, or
preparing documentation pages for downstream summarization.

## 🧪 Example: Load JSON

```python
from loaders import JsonLoader

loader = JsonLoader()
documents = loader.load("data/example.json")

for document in documents or []:
    print(document.page_content)
```

JSON loading is useful when structured API exports, local configuration snapshots, or data records
need to be converted into a processing-friendly representation.

## 🌐 Web and Online Content Loaders

Some loader classes target online or network-accessible content.

| Class                | Source Type                          |
| -------------------- | ------------------------------------ |
| `WebLoader`          | Web pages or web-accessible content. |
| `WebCrawler`         | Crawled web pages.                   |
| `ArXivLoader`        | ArXiv papers or search results.      |
| `WikiLoader`         | Wikipedia content.                   |
| `GithubLoader`       | GitHub repository or file content.   |
| `PubMedSearchLoader` | PubMed search results.               |
| `OpenCityLoader`     | Open city data sources.              |

Use these loaders when the source is best treated as a document-oriented source rather than a
source-specific public-data API query.

## 🧪 Example: Load Web Content

```python
from loaders import WebLoader

loader = WebLoader()
documents = loader.load("https://example.com")

for document in documents or []:
    print(document.page_content[:500])
```

Use `WebLoader` when the goal is to turn web-accessible content into document objects. Use
`fetchers.WebFetcher` when the goal is HTTP response inspection, link discovery, structured
extraction, or custom retrieval behavior.

## 🧪 Example: Load Wikipedia Content

```python
from loaders import WikiLoader

loader = WikiLoader()
documents = loader.load("Python programming language")

for document in documents or []:
    print(document.page_content[:500])
```

Use `WikiLoader` when the workflow needs encyclopedia-style content loaded as documents.

## 🧪 Example: Load GitHub Content

```python
from loaders import GithubLoader

loader = GithubLoader()

# Use the exact method signature shown in the generated API reference.
documents = loader.load("https://github.com/example/project")

for document in documents or []:
    print(document.metadata)
```

Use `GithubLoader` for repository or source-file ingestion workflows.

## ☁️ Cloud and Storage Loaders

Foo includes loaders for cloud-backed files and storage buckets.

| Class                   | Source Type                   |
| ----------------------- | ----------------------------- |
| `OneDriveDocLoader`     | OneDrive documents.           |
| `GoogleLoader`          | Google-backed source content. |
| `GoogleCloudFileLoader` | Google Cloud files.           |
| `AwsFileLoader`         | AWS-backed files.             |
| `GoogleBucketLoader`    | Google Cloud Storage buckets. |
| `AwsBucketLoader`       | AWS S3 buckets.               |

Use these classes when the workflow begins with cloud-hosted content that should be loaded into
Foo’s document pipeline.

## 🧪 Example: Load a Cloud File

```python
from loaders import GoogleCloudFileLoader

loader = GoogleCloudFileLoader()

# Use the exact method signature shown in the generated API reference.
documents = loader.load("bucket-or-file-reference")

for document in documents or []:
    print(document.metadata)
```

Cloud loaders may require provider credentials, bucket names, file identifiers, project identifiers,
or environment configuration. Keep credential configuration outside the loader method body and do
not hard-code secrets.

## ✉️ Email and Office Loaders

Foo includes loaders for email and office-oriented sources.

| Class               | Source Type              |
| ------------------- | ------------------------ |
| `OutlookLoader`     | Outlook-related content. |
| `EmailLoader`       | Email source content.    |
| `OneDriveDocLoader` | OneDrive documents.      |
| `WordLoader`        | Word documents.          |
| `PowerPointLoader`  | PowerPoint decks.        |
| `ExcelLoader`       | Excel workbooks.         |

These loaders support workflows where operational content is stored in office formats or
communication systems.

## 🗣️ Speech-to-Text Loader

`GoogleSpeechToTextLoader` supports speech-to-text-oriented ingestion through Google services.

Use this class when audio-derived text should enter the same downstream pipeline as documents,
fetched content, or scraped content.

Conceptual pattern:

```python
from loaders import GoogleSpeechToTextLoader

loader = GoogleSpeechToTextLoader()

# Use the exact method signature shown in the generated API reference.
documents = loader.load("audio-source-reference")
```

Speech-to-text workflows may require credentials, supported audio formats, language configuration,
and service-specific options.

## 🧾 Document Output Shape

Many loaders return document-like objects, commonly LangChain `Document` instances.

A document object normally contains:

| Field          | Meaning                                                                          |
| -------------- | -------------------------------------------------------------------------------- |
| `page_content` | Main text content loaded from the source.                                        |
| `metadata`     | Source metadata such as path, URI, page number, row number, or provider details. |

Conceptual result:

```python
Document(
    page_content="Loaded text...",
    metadata={
        "source": "data/example.pdf",
        "page": 1,
    },
)
```

Not every loader returns the same metadata keys. Use the API reference and source-specific behavior
to determine what metadata is available.

## 🔄 Loader-to-Processing Workflow

A common Foo workflow is:

```text
Source file or document reference
          |
          v
Concrete loader class
          |
          v
Document objects or normalized text
          |
          +--> Streamlit display
          +--> chunking
          +--> SQLite metadata storage
          +--> Chroma semantic storage
          +--> AI generation
          +--> Markdown output
```

The loader’s job is to get source content into a usable internal form. Later modules decide how to
process, store, search, summarize, or export it.

## 🧩 Relationship to Fetchers

Loaders and fetchers overlap in some web-facing workflows, but their responsibilities are different.

Use a loader when the source should become document content.

Use a fetcher when the workflow needs retrieval-specific behavior such as:

* HTTP status-code inspection.
* Link extraction.
* HTML element extraction.
* Provider-specific API records.
* Search results.
* Public-data query responses.
* Weather, government, environmental, or astronomy data.

```text
Loader  -> document ingestion
Fetcher -> external data retrieval
```

For example:

* Use `WebLoader` to load a page as document content.
* Use `WebFetcher` to inspect response status, extract links, and parse page structure.
* Use a source-specific fetcher for APIs and public datasets.

## 🧩 Relationship to Scrapers

Loaders normalize source content into documents.

Scrapers extract specific HTML elements such as headings, paragraphs, tables, links, images,
sections, and blockquotes.

```text
HtmlLoader / WebLoader
        |
        v
document-oriented content

WebExtractor
        |
        v
specific HTML elements
```

Use a scraper when the workflow needs targeted HTML extraction rather than broad document ingestion.

## 🗄️ Relationship to Persistence

Loader results can be persisted through `data.py`.

Good SQLite candidates include:

* Source path.
* Source URI.
* File type.
* Row number.
* Page number.
* Document count.
* Chunk count.
* Load timestamp.
* Loader class name.
* Metadata extracted during loading.

Good Chroma candidates include:

* Loaded document text.
* Document chunks.
* Embeddings.
* Chunk metadata.
* Semantic search collections.

A common pattern is to store source metadata in SQLite and searchable document chunks in Chroma.

## 🤖 Relationship to Generators

Loaded documents can be passed to generator workflows.

Common use cases include:

* Summarizing documents.
* Extracting key points.
* Comparing source documents.
* Drafting documentation from source files.
* Creating question-answering context.
* Preparing retrieval-augmented generation context.
* Translating loaded text.

The loader should not call the AI provider directly. It should return content that the generator
layer can use.

## 🧾 Relationship to Writers

Loaded content can be exported through `writers.py`.

Examples:

* Export loaded document text to Markdown.
* Export a document inventory.
* Export extracted metadata.
* Export transformed source content.
* Export summaries produced after loading and generation.

The writer layer should handle output serialization. Loader classes should not grow unrelated
Markdown or report-generation behavior.

## 🔐 Error Handling

Loader classes use Foo’s structured error logging pattern where handled exception paths exist.

The preferred pattern is:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "loaders"
    exception.cause = "PdfLoader"
    exception.method = "load( self, path: str ) -> List[ Document ] | None"
    Logger().write(exception)
    raise exception
```

Error metadata should be stable and reviewer-safe.

Do not include:

* API keys.
* Tokens.
* Raw credentials.
* Full file paths unless explicitly intended.
* Full file contents.
* Raw document text.
* OCR text.
* Email bodies.
* User-provided private content.
* Large binary content.

The purpose of logging is to identify the failing module, class, and method without capturing
sensitive source material.

## 🛡️ Safe Loading Guidance

When adding or modifying a loader:

* Validate required arguments before accessing the source.
* Keep provider credentials in configuration or environment variables.
* Do not hard-code user-specific file paths.
* Do not log source file contents.
* Do not log raw document bodies.
* Return documented fallback values.
* Preserve existing exception behavior.
* Keep source-specific logic inside the concrete loader class.
* Keep UI rendering in `app.py`.
* Keep storage in `data.py`.
* Keep output serialization in `writers.py`.
* Keep provider-specific retrieval logic in `fetchers.py` when the workflow is retrieval-oriented
  rather than document-ingestion-oriented.

## 🧪 Testing Loaders

A loader test should verify both success and failure behavior.

Test cases should include:

* Valid source path or source reference.
* Missing source path.
* Empty source path.
* Unsupported file extension where applicable.
* Empty file where applicable.
* Malformed source content where applicable.
* Missing provider credentials where applicable.
* Missing optional dependency where applicable.
* Return type consistency.
* Metadata consistency.
* Error logging in handled exception paths.

For file-based loaders, test with:

* A small valid file.
* A missing file.
* A malformed file.
* A file with non-ASCII text when relevant.
* A file with multiple pages, rows, sheets, or slides when relevant.

For cloud loaders, test with:

* A valid object reference.
* A missing object reference.
* Invalid credentials.
* Missing permissions.
* Empty remote content.
* Provider timeout or unavailable-service behavior.

## 🧪 Return Shape Guidance

Loader methods should return predictable values.

Good return shapes include:

* `list[Document]`
* `Document`
* `str`
* `dict`
* `list[dict]`
* `None` as a documented fallback

Avoid returning unrelated types from different branches of the same method. If a loader can return
`None`, annotate and document that possibility.

Document metadata should be preserved when possible because downstream workflows often need source
traceability.

## 📦 Dependency Guidance

Many loader classes rely on optional third-party packages.

Examples may include libraries for:

* PDF parsing.
* Excel reading.
* Word document parsing.
* PowerPoint parsing.
* HTML parsing.
* Cloud storage access.
* Email parsing.
* Notebook parsing.
* LangChain document loaders.

When adding or modifying a dependency:

* Add runtime dependencies to `requirements.txt`.
* Keep documentation-only dependencies in `requirements-docs.txt`.
* Note platform-specific dependencies in user documentation when needed.
* Avoid making optional loaders break unrelated workflows.
* Handle missing optional dependencies clearly.

## 🧰 Adding a New Loader

When adding a new loader class:

1. Add the class to `loaders.py`.
2. Inherit from `Loader` when common loader behavior applies.
3. Initialize default state in `__init__`.
4. Validate required arguments.
5. Use a clear `load(...)` or source-specific method name.
6. Return a predictable and annotated result.
7. Preserve metadata.
8. Log wrapped exceptions where the loader has handled exception paths.
9. Add or update Streamlit controls only if the loader should be user-facing.
10. Update user-facing documentation.
11. Confirm the generated API page renders without Griffe warnings.

## 📖 API Documentation

The generated API reference for this module is rendered below.
