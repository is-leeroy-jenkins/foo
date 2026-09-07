# Loading Data

Foo’s loading workflow turns files, documents, notebooks, cloud objects, and document-like sources
into content that the rest of the application can display, process, summarize, store, search, or
export.

The loading layer is implemented in `loaders.py` and coordinated through the Streamlit interface in
`app.py`.

## 🧭 Purpose

Use Loading mode when the workflow begins with source material that should be treated as document
content.

Loading is appropriate for:

* Reading local files.
* Loading structured files.
* Loading office documents.
* Loading notebooks.
* Loading Markdown and HTML files.
* Loading cloud-backed files.
* Loading repository or web-accessible document content.
* Loading research or reference content into document objects.
* Preparing text for chunking, tokenization, summarization, semantic search, or export.

The loader layer is not intended to be the primary place for general public API retrieval,
structured web extraction, database persistence, or AI-provider calls. Those responsibilities belong
to `fetchers.py`, `scrapers.py`, `data.py`, and `generators.py`.

## 🧱 Loading Layer

The loading layer sits near the beginning of the Foo processing pipeline.

```text
Source file, document, notebook, or cloud object
                |
                v
            loaders.py
                |
                v
Document objects, raw text, metadata, or structured content
                |
                +--> Streamlit preview
                +--> text processing
                +--> tokenization
                +--> chunking
                +--> generation
                +--> SQLite metadata storage
                +--> Chroma semantic storage
                +--> Markdown output
```

A loader’s job is to normalize source material into a form that Foo can use. Later workflow stages
decide whether that content should be displayed, stored, summarized, searched, or exported.

## 📦 Supported Loader Categories

Foo includes loader classes for a broad set of source types.

| Category                    | Loader Classes                                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Plain and structured files  | `TextLoader`, `CsvLoader`, `XmlLoader`, `JsonLoader`                                                           |
| Document formats            | `PdfLoader`, `WordLoader`, `PowerPointLoader`, `MarkdownLoader`, `HtmlLoader`                                  |
| Spreadsheet formats         | `ExcelLoader`, `CsvLoader`                                                                                     |
| Web and research sources    | `WebLoader`, `WebCrawler`, `ArXivLoader`, `WikiLoader`, `GithubLoader`, `PubMedSearchLoader`, `OpenCityLoader` |
| Office and email sources    | `OutlookLoader`, `EmailLoader`, `OneDriveDocLoader`, `SpfxLoader`                                              |
| Cloud and storage sources   | `GoogleLoader`, `GoogleCloudFileLoader`, `AwsFileLoader`, `GoogleBucketLoader`, `AwsBucketLoader`              |
| Notebook and speech sources | `JupyterNotebookLoader`, `GoogleSpeechToTextLoader`                                                            |

The exact constructor arguments, method signatures, return annotations, and source-specific behavior
are documented in the generated [Loaders API](api/loaders.md).

## 🖥️ Loading Mode in the Application

In the Streamlit application, Loading mode is the user-facing entry point for loader workflows.

A typical Loading mode workflow is:

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
Review loaded content
        |
        v
Reuse, store, summarize, or export the result
```

Depending on the selected loader, source input may be:

* A local file path.
* An uploaded file.
* A URL.
* A GitHub reference.
* A Wikipedia topic.
* An ArXiv query or identifier.
* A cloud object reference.
* A bucket name.
* A notebook path.
* A source-specific identifier.

The UI should only expose controls that are relevant to the selected loader.

## 📄 Local File Loading

Use local file loaders when the source is available on disk or uploaded through the UI.

Common local-file workflows include:

| Source           | Recommended Loader      |
| ---------------- | ----------------------- |
| `.txt`           | `TextLoader`            |
| `.csv`           | `CsvLoader`             |
| `.xml`           | `XmlLoader`             |
| `.json`          | `JsonLoader`            |
| `.pdf`           | `PdfLoader`             |
| `.docx`          | `WordLoader`            |
| `.xlsx`          | `ExcelLoader`           |
| `.pptx`          | `PowerPointLoader`      |
| `.md`            | `MarkdownLoader`        |
| `.html` / `.htm` | `HtmlLoader`            |
| `.ipynb`         | `JupyterNotebookLoader` |

Use a local-file loader when the workflow goal is document ingestion rather than API retrieval.

## 🧪 Example: Load a Text File

```python
from loaders import TextLoader

loader = TextLoader()
documents = loader.load("data/example.txt")

for document in documents or []:
    print(document.page_content)
```

This workflow is useful for simple source text, logs, notes, documentation files, or plain-text
exports.

## 🧪 Example: Load a CSV File

```python
from loaders import CsvLoader

loader = CsvLoader()
documents = loader.load("data/example.csv")

for document in documents or []:
    print(document.metadata)
    print(document.page_content)
```

CSV loading is useful when rows need to become document-like records for review, processing,
summarization, storage, or semantic search.

## 🧪 Example: Load a PDF File

```python
from loaders import PdfLoader

loader = PdfLoader()
documents = loader.load("data/example.pdf")

for document in documents or []:
    print(document.page_content[:500])
```

PDF loading is useful for reports, technical documents, public documents, research papers, manuals,
forms, and other page-based documents.

## 🧪 Example: Load a Word Document

```python
from loaders import WordLoader

loader = WordLoader()
documents = loader.load("data/example.docx")

for document in documents or []:
    print(document.page_content[:500])
```

Word loading is useful for policy documents, drafts, memos, reports, instructions, and other
office-document workflows.

## 🧪 Example: Load an Excel Workbook

```python
from loaders import ExcelLoader

loader = ExcelLoader()
documents = loader.load("data/example.xlsx")

for document in documents or []:
    print(document.metadata)
    print(document.page_content)
```

Excel loading is useful for workbook review, tabular-data extraction, sheet inspection, and
workflows that transform spreadsheet rows into document-like content.

## 🧪 Example: Load Markdown

```python
from loaders import MarkdownLoader

loader = MarkdownLoader()
documents = loader.load("docs/index.md")

for document in documents or []:
    print(document.page_content)
```

Markdown loading is useful for documentation review, MkDocs site inspection, README analysis, and
migration workflows.

## 🧪 Example: Load JSON

```python
from loaders import JsonLoader

loader = JsonLoader()
documents = loader.load("data/example.json")

for document in documents or []:
    print(document.page_content)
```

JSON loading is useful for structured exports, API response captures, configuration snapshots, and
local data records that need to be normalized for processing.

## 🌐 Web and Research Loading

Foo includes loader classes for document-like web and research sources.

Use these loaders when the source should be treated as document content:

| Source Type                          | Recommended Loader   |
| ------------------------------------ | -------------------- |
| Web page document content            | `WebLoader`          |
| Crawled web content                  | `WebCrawler`         |
| ArXiv paper or search result content | `ArXivLoader`        |
| Wikipedia topic content              | `WikiLoader`         |
| GitHub repository or file content    | `GithubLoader`       |
| PubMed search result content         | `PubMedSearchLoader` |
| Open city data content               | `OpenCityLoader`     |

These loaders are appropriate when the output should become document objects or source text.

Use `fetchers.py` instead when the workflow requires source-specific API calls, response metadata,
public-data records, search result payloads, weather data, government data, environmental data, or
astronomy data.

## 🧪 Example: Load Web Content

```python
from loaders import WebLoader

loader = WebLoader()
documents = loader.load("https://example.com")

for document in documents or []:
    print(document.page_content[:500])
```

Use this pattern when a web page should be converted into document-like content.

## 🧪 Example: Load Wikipedia Content

```python
from loaders import WikiLoader

loader = WikiLoader()
documents = loader.load("Python programming language")

for document in documents or []:
    print(document.page_content[:500])
```

Use this pattern for encyclopedia-style reference content.

## 🧪 Example: Load GitHub Content

```python
from loaders import GithubLoader

loader = GithubLoader()

documents = loader.load("https://github.com/example/project")

for document in documents or []:
    print(document.metadata)
```

Use the exact method signature from the [Loaders API](api/loaders.md), because GitHub loading may
require source-specific arguments.

## ☁️ Cloud Loading

Foo includes loaders for cloud-backed files and object storage.

Cloud-oriented loaders include:

* `OneDriveDocLoader`
* `GoogleLoader`
* `GoogleCloudFileLoader`
* `AwsFileLoader`
* `GoogleBucketLoader`
* `AwsBucketLoader`

Use these loaders when content is stored in a provider-backed location and should be brought into
Foo’s document pipeline.

Cloud workflows may require:

* Provider credentials.
* Bucket names.
* Object keys.
* File identifiers.
* Project identifiers.
* Source paths.
* Permissions.
* Optional provider SDK dependencies.

Keep credentials in environment-backed configuration or `config.py` references. Do not hard-code
credentials in source code, examples, Markdown files, or output artifacts.

## 🧪 Example: Load a Google Cloud File

```python
from loaders import GoogleCloudFileLoader

loader = GoogleCloudFileLoader()

documents = loader.load("bucket-or-file-reference")

for document in documents or []:
    print(document.metadata)
```

Use the API reference for the exact expected source argument and provider-specific behavior.

## 🧪 Example: Load an AWS File

```python
from loaders import AwsFileLoader

loader = AwsFileLoader()

documents = loader.load("s3-object-reference")

for document in documents or []:
    print(document.metadata)
```

AWS workflows may require configured credentials, bucket names, regions, and object keys.

## ✉️ Office and Email Loading

Foo includes loaders for office and email-oriented source material.

Relevant loaders include:

* `OutlookLoader`
* `EmailLoader`
* `OneDriveDocLoader`
* `WordLoader`
* `PowerPointLoader`
* `ExcelLoader`

Use these loaders when operational content is stored in office formats or communication sources.

Common use cases include:

* Loading email content for review.
* Loading Outlook-related content.
* Loading OneDrive documents.
* Loading office documents into text.
* Preparing office content for summarization.
* Exporting office content to Markdown.

Email and office content can contain sensitive information. Review loaded content before storing,
exporting, or passing it to a provider workflow.

## 🗣️ Speech-to-Text Loading

`GoogleSpeechToTextLoader` supports speech-to-text-oriented ingestion.

Use it when audio-derived text should enter Foo’s document processing pipeline.

Typical workflow:

```text
Audio source
     |
     v
GoogleSpeechToTextLoader
     |
     v
Text transcript or document-like result
     |
     v
Review, summarize, store, or export
```

Speech-to-text workflows may require provider credentials, supported audio formats, language
options, and service-specific configuration.

## 🧾 Document Result Shape

Many loaders return LangChain `Document` objects or document-like values.

A typical document object includes:

| Field          | Meaning                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------- |
| `page_content` | Main loaded text.                                                                                   |
| `metadata`     | Source metadata such as path, page number, row number, sheet name, source URI, or provider details. |

Conceptual shape:

```python
Document(
    page_content="Loaded source text...",
    metadata={
        "source": "data/example.pdf",
        "page": 1,
    },
)
```

Metadata is important because downstream workflows need source traceability.

## 🔍 Reviewing Loaded Content

After loading content, review the result before storing or generating from it.

Check:

* Did the loader return documents?
* How many documents were returned?
* Is `page_content` populated?
* Is metadata present?
* Does the source path or URI look correct?
* Are pages, rows, sheets, slides, or records represented as expected?
* Are there encoding issues?
* Are there missing pages or empty rows?
* Does the content contain sensitive information?
* Is the result small enough for downstream processing?

For large documents, preview a small portion first.

## 🧠 Text Processing After Loading

Once content is loaded, Foo can apply text-processing workflows.

Common post-load operations include:

* Text preview.
* Text normalization.
* Chunking.
* Tokenization.
* Vocabulary inspection.
* Token count review.
* Similarity comparison.
* Summarization.
* Semantic indexing.
* Markdown export.

A common flow is:

```text
Loaded documents
        |
        v
Extract page_content
        |
        v
Normalize text
        |
        v
Chunk text
        |
        v
Store chunks or pass to generation
```

Chunking is useful when content is too large for a single downstream operation.

## 🧩 Relationship to Fetching

Loading and fetching are related but different.

Use loading when the source is a document-like object.

Use fetching when the source is an external service, query, public API, or retrieval endpoint.

| Question                                                           | Use     |
| ------------------------------------------------------------------ | ------- |
| “I have a PDF, CSV, Markdown file, or Word document.”              | Loader  |
| “I need to call a public API or source-specific service.”          | Fetcher |
| “I need status code, links, or raw HTML response metadata.”        | Fetcher |
| “I need to turn a file or web page into document objects.”         | Loader  |
| “I need weather, Census, USGS, air-quality, or astronomy records.” | Fetcher |

If the source is a web page and the goal is document ingestion, use `WebLoader`. If the goal is
response inspection or structured retrieval, use `WebFetcher`.

## 🧩 Relationship to Scraping

Loading and scraping are also different.

Use loading when the goal is document ingestion.

Use scraping when the goal is extracting specific HTML elements.

| Goal                                | Use                                   |
| ----------------------------------- | ------------------------------------- |
| Load an HTML page as document text  | `HtmlLoader` or `WebLoader`           |
| Extract headings from a page        | `WebExtractor.scrape_headings(...)`   |
| Extract links from a page           | `WebExtractor.scrape_hyperlinks(...)` |
| Extract table cells from a page     | `WebExtractor.scrape_tables(...)`     |
| Load Markdown, PDF, or Word content | Loader                                |

Scrapers return specific extracted page elements. Loaders return document content.

## 🗄️ Relationship to Persistence

Loaded results can be stored through the data-management layer.

Good SQLite candidates include:

* Source path.
* Source URI.
* Loader class name.
* File type.
* Page count.
* Row count.
* Sheet count.
* Document count.
* Chunk count.
* Load timestamp.
* Metadata summary.

Good Chroma candidates include:

* Loaded text.
* Document chunks.
* Embeddings.
* Chunk metadata.
* Semantic-search collections.

A common pattern is:

```text
Loader output
     |
     +--> SQLite for metadata
     |
     +--> Chroma for searchable text chunks
```

## 🤖 Relationship to Generation

Loaded content can be passed to generator workflows.

Common generation tasks include:

* Summarize a loaded document.
* Extract key points.
* Produce documentation from source files.
* Compare two loaded documents.
* Translate loaded text.
* Generate questions and answers.
* Generate an executive summary.
* Create retrieval context for a later prompt.

Review loaded content before passing it to a provider workflow, especially when it contains private
or sensitive material.

## 🧾 Relationship to Output

Loaded content can be exported through the output layer.

Good output artifacts include:

* Document summaries.
* Source inventories.
* Metadata reports.
* Extracted text files.
* Markdown conversions.
* Chunk inventories.
* Review packets.
* Load result reports.

Use `MarkdownWriter` when the result should become a reviewable Markdown artifact.

## 🔐 Sensitive Content Guidance

Loaded files can contain sensitive content.

Before storing, exporting, or sending loaded content to an AI provider, review whether it includes:

* API keys.
* Tokens.
* Passwords.
* Private emails.
* Personal information.
* Sensitive file paths.
* Internal documents.
* Proprietary material.
* Raw legal, financial, or personnel records.
* Unredacted source text.

Do not write or transmit sensitive loaded content unless the workflow explicitly requires it and the
user understands the destination.

## 🛡️ Safe Loading Practices

Use these practices when loading source material:

* Start with a small known-good file.
* Validate the file path or source reference.
* Confirm the loader matches the file type.
* Avoid loading very large files until the workflow is tested.
* Preview loaded content before storing or exporting.
* Keep credentials outside source code.
* Do not log full file contents.
* Do not log raw document text.
* Use explicit source metadata.
* Keep document ingestion separate from retrieval, scraping, generation, and writing.

## ⚠️ Common Loading Issues

| Issue                                 | Likely Cause                                                             | Suggested Check                                                  |
| ------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Loader returns no documents           | Bad path, unsupported format, empty file, or missing optional dependency | Test with a small known-good file.                               |
| File not found                        | Wrong relative path or working directory                                 | Run from repository root or use an absolute path during testing. |
| Encoding error                        | Source text uses unexpected encoding                                     | Try a smaller sample or convert to UTF-8.                        |
| PDF text is incomplete                | PDF may contain scanned pages or images                                  | Use OCR workflow if available.                                   |
| CSV rows look wrong                   | Delimiter, quoting, or encoding issue                                    | Open the CSV and inspect delimiter/headers.                      |
| Excel loader misses sheets            | Workbook structure or loader setting issue                               | Test with a workbook containing one simple sheet.                |
| Web loading fails                     | URL inaccessible, blocked, timed out, or not document-like               | Try `WebFetcher` for response inspection.                        |
| Cloud loading fails                   | Credentials, permissions, bucket name, object key, or dependency issue   | Verify provider configuration.                                   |
| Loader works locally but fails in app | Streamlit working directory or uploaded-file handling issue              | Confirm path handling and session-state values.                  |

## 🧪 Testing Loading Workflows

When testing loaders, use focused test inputs.

Recommended test set:

* Small `.txt` file.
* Small `.csv` file with headers.
* Simple `.json` file.
* One-page `.pdf`.
* Small `.docx`.
* Small `.xlsx`.
* Simple `.md`.
* Simple `.html`.
* Small `.ipynb`.
* Known reachable web page.
* Known missing file.
* Empty file where applicable.
* Malformed file where applicable.

For each test, confirm:

* The loader returns the documented type.
* Empty or missing inputs are handled safely.
* Metadata is preserved.
* Content is readable.
* Errors are understandable.
* Sensitive content is not logged.
* Downstream preview does not crash.

## 🧰 Adding a New Loader Workflow

When adding a new user-facing loader workflow:

1. Add or verify the loader class in `loaders.py`.
2. Confirm the loader compiles.
3. Confirm the loader has Google-style docstrings.
4. Confirm method return annotations are accurate.
5. Add Streamlit controls in `app.py`.
6. Use unique widget keys.
7. Validate required inputs before calling the loader.
8. Display a preview of results.
9. Store results in session state only when useful.
10. Add export or persistence options only after preview works.
11. Update this page if the workflow is user-facing.
12. Update [Loaders API](api/loaders.md) if the public API changes.

## 📖 API Reference

Use the generated API page for source-level details:

* [Loaders API](api/loaders.md)

The API reference includes class definitions, method signatures, return annotations, and docstrings
generated from `loaders.py`.

## ✅ Loading Checklist

Before treating a load as successful, confirm:

* The selected loader matches the source type.
* Required input values were provided.
* The loader returned data.
* The result type is expected.
* `page_content` or equivalent content is populated.
* Metadata is present and useful.
* The result preview is readable.
* Sensitive content has been reviewed.
* The content is suitable for downstream use.
* The output or storage destination is appropriate.

## 🧭 Summary

Foo’s loading workflow is the entry point for document-oriented source material. Loaders convert
files, documents, notebooks, cloud objects, and source-specific content into usable internal
representations. Once loaded, content can be reviewed, chunked, stored, summarized, searched, or
exported by the rest of the application.
