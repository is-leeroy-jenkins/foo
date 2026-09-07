# Output

Foo produces output from loaded documents, fetched data, scraped web content, AI-generated
responses, database records, and structured model objects. The output layer turns those results into
reviewable artifacts that can be inspected, stored, shared, committed to a repository, or used in
downstream analysis workflows.

The primary source module for output writing is `writers.py`. The main documented writer class is
`MarkdownWriter`.

## 🧭 Purpose

The purpose of Foo’s output workflow is to make processed information durable and reviewable.

Foo workflows often begin with temporary runtime state:

* A loaded document.
* A scraped web page.
* A fetched public-data response.
* A generated AI response.
* A database query result.
* A set of extracted links.
* A set of extracted tables.
* A processed text chunk collection.
* A structured model object.

Without an output step, those results may only exist in the Streamlit session. The output workflow
provides a way to turn them into files or other artifacts that can be reused.

## 🧱 Output Layer

The output layer sits at the end of the Foo processing pipeline.

```text
Source content
      |
      v
Load / fetch / scrape / generate / query
      |
      v
Process, normalize, summarize, or inspect
      |
      v
writers.py
      |
      v
Output artifacts
```

The writer layer should not retrieve, load, scrape, generate, or query data on its own. It should
receive already-produced content and serialize it into a useful format.

## 📝 Markdown Output

Markdown is the primary documented output format for Foo.

Markdown is useful because it is:

* Human-readable.
* GitHub-compatible.
* MkDocs-compatible.
* Easy to diff in version control.
* Easy to edit manually.
* Suitable for reports, notes, summaries, inventories, and documentation drafts.
* Portable across local documentation, repositories, and static sites.

Foo’s `MarkdownWriter` should be used when a workflow needs to export processed content as a `.md`
file.

## 📦 Output Sources

Foo can produce output from several modules.

| Source Module   | Typical Output                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `loaders.py`    | Loaded document text, source metadata, document inventories, chunk summaries.                   |
| `fetchers.py`   | API results, public-data snapshots, source metadata, response summaries, link inventories.      |
| `scrapers.py`   | Headings, paragraphs, tables, links, images, blockquotes, extracted page sections.              |
| `generators.py` | AI-generated responses, summaries, translations, transcriptions, documentation drafts.          |
| `data.py`       | SQLite table exports, query results, profile summaries, vector-search results.                  |
| `models.py`     | Structured prompt, file, message, location, forecast, tool, and search configuration summaries. |
| `app.py`        | User-selected workflow results assembled for display or export.                                 |

The output layer should preserve enough context for the user to understand where the artifact came
from and how it was produced.

## 🧾 Recommended Output Structure

A strong Markdown output artifact should include:

1. A clear title.
2. Source information.
3. Processing context.
4. The main result.
5. Supporting metadata.
6. Warnings or limitations where applicable.
7. Timestamp or run information when useful.

Recommended structure:

```markdown
# Output Title

## Source

Source details, file path, URL, provider, table name, or workflow input.

## Method

Short explanation of how the result was produced.

## Result

Main output content.

## Metadata

Structured metadata, status information, counts, or options.

## Notes

Limitations, warnings, next steps, or interpretation notes.
```

This format works well for both user-facing reports and developer-facing workflow artifacts.

## 📄 Document Output

Loaded documents can be exported when a user needs to inspect or preserve source content.

Good candidates include:

* Plain text extracted from files.
* PDF page text.
* Word document text.
* Markdown file content.
* HTML file content.
* CSV row summaries.
* JSON summaries.
* Notebook content.
* PowerPoint slide text.
* Cloud document content.

Example document-output structure:

```markdown
# Loaded Document Export

## Source

`data/example.pdf`

## Loader

`PdfLoader`

## Document Count

3

## Extracted Text

Document text appears here.

## Metadata

| Field | Value |
| --- | --- |
| Source | data/example.pdf |
| Loader | PdfLoader |
| Pages | 3 |
```

This kind of output is useful for review, documentation, and downstream processing.

## 🌐 Fetch Output

Fetched data can be exported when a user needs a durable record of a retrieval workflow.

Good candidates include:

* Web page fetch summaries.
* Search result inventories.
* Public-data query responses.
* Weather query summaries.
* Geospatial lookup results.
* Environmental data snapshots.
* Astronomy query results.
* Government-data records.
* Health or demographic query results.

Example fetch-output structure:

```markdown
# Fetch Result

## Source

`https://example.com`

## Fetcher

`WebFetcher`

## Response Metadata

| Field | Value |
| --- | --- |
| Status Code | 200 |
| Encoding | utf-8 |
| Result Type | HTML |

## Summary

The page was retrieved successfully.

## Extracted Links

- https://example.com/about
- https://example.com/docs
```

For public-data responses, include query parameters and record counts when possible.

## 🧹 Scrape Output

Scraped data can be exported when a user needs an inventory or extracted page content.

Good candidates include:

* Page outlines.
* Headings.
* Paragraphs.
* Tables.
* Link inventories.
* Image inventories.
* Blockquotes.
* Article sections.
* Documentation migration notes.

Example scrape-output structure:

```markdown
# Scrape Summary

## Source

`https://example.com/docs`

## Extraction Methods

- `scrape_headings`
- `scrape_paragraphs`
- `scrape_hyperlinks`

## Headings

- Overview
- Installation
- Configuration
- API Reference

## Links

| Text | URL |
| --- | --- |
| Documentation | https://example.com/docs |
| Repository | https://github.com/example/project |
```

Scrape output should distinguish between extracted content and generated commentary.

## 🤖 Generation Output

Generated content can be exported when a user needs to preserve AI output outside the active
session.

Good candidates include:

* Summaries.
* Explanations.
* Translations.
* Transcriptions.
* Search-assisted answers.
* Draft documentation.
* Prompt results.
* Comparison notes.
* Generated reports.

Example generation-output structure:

```markdown
# Generated Summary

## Provider

`Chat`

## Model

`<model-name>`

## Prompt Summary

Summarize the Foo architecture.

## Output

Generated response appears here.

## Notes

Review generated text before using it as official documentation.
```

Avoid storing sensitive prompt content unless the workflow explicitly requires it.

## 🗄️ Data Output

Database records and query results can be exported when a user needs a report, table snapshot, or
audit-friendly record.

Good candidates include:

* SQLite table exports.
* Query results.
* Table profiles.
* Aggregation summaries.
* Processing history.
* Source inventories.
* Stored fetch metadata.
* Stored generation metadata.
* Vector-search result summaries.

Example data-output structure:

```markdown
# Table Export

## Database

`Foo.db`

## Table

`sources`

## Row Count

25

## Records

| id | name | uri |
| --- | --- | --- |
| 1 | Example | https://example.com |
```

For large tables, export summaries or filtered views rather than dumping everything into one
Markdown file.

## 📊 Visualization Output

Foo may generate charts, tables, or summarized visual views from dataframe-oriented workflows.

When documenting visualization output, include:

* Data source.
* Filter settings.
* Aggregation settings.
* Chart type.
* Columns used.
* Row count.
* Interpretation notes.

Example:

```markdown
# Visualization Summary

## Source Table

`fetch_results`

## Chart Type

Bar chart

## Grouping Column

`source`

## Value Column

`record_count`

## Notes

The chart summarizes record counts by source.
```

Charts themselves may be saved as image files when the workflow supports it. The Markdown output can
then reference the image with a relative path.

## 🧩 Model Output

Pydantic model objects can be exported as structured summaries.

Good candidates include:

* Prompt records.
* Message objects.
* File metadata.
* Location records.
* Forecast records.
* Tool definitions.
* Function schemas.
* File-search settings.
* Web-search settings.

Example model-output structure:

```markdown
# Tool Definition

## Tool Name

`web_search`

## Description

Searches the web for source-backed information.

## Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| query | string | yes | Search query text. |
| domains | list[string] | no | Allowed domains. |
```

Model output is useful when documenting provider tools, workflow settings, or structured application
state.

## 🔄 Output Workflow

A typical output workflow follows this sequence:

```text
User runs workflow
      |
      v
Foo produces result
      |
      v
User reviews result in Streamlit
      |
      v
User chooses export/write action
      |
      v
Foo builds Markdown content
      |
      v
MarkdownWriter writes file
      |
      v
User reviews artifact
```

The user should normally be able to inspect the result before writing it to a durable file.

## 🧪 MarkdownWriter Usage

The primary writer pattern is:

```python
from writers import MarkdownWriter

writer = MarkdownWriter()

writer.write(
    path="outputs/example.md",
    content="# Example\n\nGenerated content appears here.\n",
)
```

Use the exact source signature from the API reference when implementing against the current
`writers.py`.

The caller is responsible for assembling the Markdown content. The writer is responsible for writing
it.

## 📁 Recommended Output Directories

Use predictable output directories.

Recommended structure:

```text
outputs/
├─ documents/
├─ fetches/
├─ scrapes/
├─ generations/
├─ data/
├─ reports/
└─ exports/
```

For documentation assets that should be included in the MkDocs site, use:

```text
docs/images/
```

For generated Markdown pages that should become part of the documentation site, place them under:

```text
docs/
```

Do not mix temporary workflow outputs with published documentation unless the artifact is
intentionally part of the docs.

## 🧾 File Naming Guidance

Output filenames should be stable, descriptive, and safe.

Recommended naming patterns:

```text
loaded-document-summary.md
web-fetch-summary.md
scrape-link-inventory.md
generation-summary.md
sqlite-table-profile.md
source-inventory.md
public-data-snapshot.md
```

Avoid filenames that include:

* API keys.
* Tokens.
* Full user queries.
* Sensitive personal information.
* Long URLs.
* Characters that are invalid on Windows.
* Ambiguous names such as `output.md`, `test.md`, or `final.md`.

When multiple outputs are created, use timestamps or stable identifiers.

Example:

```text
scrape-summary-2026-06-12.md
fetch-results-airnow-20001.md
generation-summary-foo-architecture.md
```

## 🔐 Sensitive Content Guidance

Output artifacts may contain user-provided or retrieved content. Treat them carefully.

Before writing output, consider whether the content includes:

* API keys.
* Tokens.
* Credentials.
* Raw prompts.
* Private user messages.
* Full uploaded documents.
* Email content.
* File paths with user-identifying directories.
* Sensitive URLs with query parameters.
* Provider responses containing private context.
* Unreviewed generated text.

Do not write sensitive content unless the workflow explicitly requires it and the user understands
what is being saved.

## 🛡️ Safe Writing Practices

When writing files:

* Validate the output path.
* Validate that content exists.
* Use UTF-8 encoding for text output.
* Create parent directories only when intended.
* Avoid overwriting important files accidentally.
* Use deterministic formatting where possible.
* Keep relative links valid.
* Keep Markdown tables well-formed.
* Close code fences.
* Avoid logging full output content.
* Avoid writing credentials or secrets.
* Make destructive write behavior explicit.

Output should be useful and safe by default.

## 🧰 Relationship to Writers API

The user-facing output workflow is documented here. The implementation details are documented in the
Writers API page.

Use:

* [Writers API](api/writers.md)

The API page documents source-level classes, methods, return annotations, and docstrings generated
from `writers.py`.

## 🧩 Relationship to Application Page

The Streamlit UI controls the user-facing output experience.

`app.py` should decide:

* Which result is being exported.
* Which output button or action is available.
* What preview is shown before writing.
* Whether the output path is user-selected or generated.
* How success or failure is displayed.

`writers.py` should perform the file write.

## 🗄️ Relationship to Persistence

Persistence workflows may produce output from SQLite or Chroma.

Examples include:

* Export a table profile.
* Export filtered rows.
* Export an aggregation.
* Export a source inventory.
* Export semantic-search results.
* Export processing history.

The data layer should retrieve the records. The output layer should format and write them.

## 🤖 Relationship to Generation

Generation workflows often produce text that should be reviewed before export.

Recommended pattern:

```text
Generate response
      |
      v
Display response
      |
      v
User reviews content
      |
      v
Write Markdown artifact
```

Do not treat generated output as final just because it was written to a file. Generated content
should be reviewed when used in official documentation, technical instructions, or project
deliverables.

## ✅ Output Review Checklist

Before accepting an output workflow, confirm:

* The output has a clear title.
* The source is identified.
* The method or workflow is identified.
* The main result is readable.
* Metadata is included when useful.
* Tables render correctly.
* Code fences are closed.
* Relative links are valid.
* The output path is safe.
* The file extension matches the content.
* Sensitive content is not written unintentionally.
* Generated text is marked or reviewed when appropriate.
* The writer method returns the documented value.
* Errors are handled safely.

## 🧭 Summary

Foo’s output workflow turns temporary processing results into durable artifacts. Loaded documents,
fetched data, scraped content, generated responses, database records, and model objects can all be
exported when they are ready for review or reuse. The writer layer should remain focused on
serialization, while the rest of the application remains responsible for producing the content.
