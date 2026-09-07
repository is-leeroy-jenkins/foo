# Generation

Foo’s generation workflow provides the user-facing AI layer of the application. It connects the
Streamlit interface in `app.py` to provider wrapper classes in `generators.py`.

Use Generation mode when the workflow requires model-generated text, search-assisted answers,
document summarization, translation, transcription, image-related provider workflows, or
provider-specific reasoning and tool configuration.

## 🧭 Purpose

The generation layer gives Foo a consistent place to run AI-provider workflows without embedding
provider-specific request construction directly in the Streamlit UI.

Generation mode supports workflows such as:

* Generating text from a prompt.
* Summarizing loaded documents.
* Producing explanations from retrieved or scraped content.
* Creating documentation drafts.
* Translating text.
* Transcribing audio where supported.
* Searching the web through provider-supported tools.
* Searching files through provider-supported tools.
* Analyzing images where supported.
* Generating images where supported.
* Using reasoning or thinking settings where supported by the selected provider and model.
* Extracting usable output text from provider-specific response objects.

The generation layer should receive user intent and prepared inputs, call the provider wrapper, and
return generated content for display, storage, or export.

## 🧱 Generation Layer

The generation layer sits between the Streamlit application and external AI providers.

```text
User prompt and options
        |
        v
app.py
        |
        v
generators.py
        |
        +--> Chat
        +--> Grok
        +--> Gemini
        +--> Claude
        +--> Mistral
        |
        v
Provider response
        |
        v
Extracted text or structured output
        |
        v
Display, store, or export
```

`app.py` should handle controls and result display. `generators.py` should handle provider-specific
request building, tool configuration, response extraction, and error handling.

## 🤖 Provider Wrappers

The main provider wrapper classes are:

| Class       | Provider / Role          | Typical Use                                                                                                |
| ----------- | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `Generator` | Base generator class     | Shared generator state and common conceptual contract.                                                     |
| `Chat`      | OpenAI-oriented wrapper  | Text generation, image workflows, document workflows, web search, file search, translation, transcription. |
| `Grok`      | xAI Grok wrapper         | Text generation and search-assisted generation.                                                            |
| `Gemini`    | Google Gemini wrapper    | Text generation, thinking configuration, and search-assisted generation.                                   |
| `Claude`    | Anthropic Claude wrapper | Text generation and web-search-style generation.                                                           |
| `Mistral`   | Mistral wrapper          | Text generation and schema-oriented workflows.                                                             |

Each provider exposes different request parameters, tools, response formats, and model capabilities.
For that reason, provider-specific logic belongs inside the matching wrapper class.

## ⚙️ Generation Mode in the Application

Generation mode is coordinated by `app.py`.

The UI may expose controls for:

* Provider selection.
* Model selection.
* Prompt input.
* System instructions.
* Message history.
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
* Thinking level or thinking budget.
* Streaming.
* Store response setting.
* Image input or image prompt where supported.
* Translation input.
* Transcription input.

A typical Generation mode workflow is:

```text
Select Generation mode
        |
        v
Choose provider and model
        |
        v
Enter prompt and options
        |
        v
Run generation
        |
        v
Review response
        |
        v
Store, export, or reuse the result
```

Provider options should only be shown or used when the selected provider and model support them.

## 💬 Text Generation

Text generation is the most common generation workflow.

Use it when the user wants a written response from a prompt.

Typical examples:

* Summarize a source document.
* Explain extracted data.
* Draft documentation.
* Rewrite loaded content.
* Create a report outline.
* Generate a comparison.
* Extract likely requirements from source text.
* Convert notes into a structured summary.

Basic pattern:

```python
from generators import Chat

chat = Chat()

result = chat.generate_text(
    prompt="Summarize Foo's architecture in five bullets."
)

print(result)
```

Use the exact signatures from the [Generators API](api/generators.md) when implementing against the
current source.

## 🧾 Document Summarization

Document summarization should usually begin in Loading mode.

Recommended workflow:

```text
Loading mode
    |
    v
Load document
    |
    v
Review loaded text
    |
    v
Generation mode
    |
    v
Summarize document
    |
    v
Review generated summary
    |
    v
Export if needed
```

Good summarization prompts are specific about the desired output.

Example prompt:

```text
Summarize the loaded document for a technical maintainer. Focus on purpose, inputs, outputs, dependencies, configuration requirements, and risks.
```

Avoid sending entire large documents into a generation workflow without reviewing chunking, token
limits, and sensitivity first.

## 🌐 Search-Assisted Generation

Search-assisted generation uses provider-supported tools to include external search behavior in the
generation workflow.

Use search-assisted generation when the answer depends on:

* Current information.
* External documentation.
* Source discovery.
* Web references.
* Time-sensitive details.
* Official guidance.
* Publicly available supporting sources.

Typical workflow:

```text
Prompt
  |
  v
Allowed domains or search settings
  |
  v
Provider web-search tool
  |
  v
Generated response
```

Example:

```python
from generators import Chat

chat = Chat()

result = chat.search_web(
    prompt="Find official guidance for MkDocs navigation and summarize the key points."
)

print(result)
```

When source control matters, use allowed domains or narrow search settings.

Good prompt pattern:

```text
Search only official documentation sources. Summarize the setup steps and note any version-specific caveats.
```

## 🗂️ File Search

File-search workflows allow a provider or retrieval system to use uploaded or indexed files as
context.

Use file search when:

* Source documents are already available.
* The answer should be grounded in a known file set.
* You need question-answering over project documentation.
* You need to search internal source documents.
* You want generated output constrained to supplied materials.

Typical workflow:

```text
Load or register files
        |
        v
Configure file-search tool
        |
        v
Ask question or provide instruction
        |
        v
Provider uses file context
        |
        v
Review response
```

File-search behavior varies by provider. Use the API reference and current UI controls for exact
supported options.

## 🧠 Reasoning and Thinking Settings

Some providers support reasoning or thinking settings. The feature is not consistent across
providers.

Examples of provider-specific checks include:

| Provider Wrapper | Capability Checks                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `Chat`           | `supports_reasoning(...)`                                                                     |
| `Grok`           | `supports_reasoning_effort(...)`, `supports_reasoning_object(...)`, `is_reasoning_model(...)` |
| `Gemini`         | `supports_thinking_level(...)`, `supports_thinking_budget(...)`                               |
| `Claude`         | `_supports_thinking(...)`                                                                     |

Use reasoning or thinking controls only when the selected model supports them.

Do not assume that a setting available for one provider is valid for another provider. The wrapper
class should decide how to apply provider-specific options.

## 🧰 Tool Configuration

Provider wrappers may build tool definitions for web search, file search, function-style operations,
or provider-native tools.

Tool configuration may include:

* Tool type.
* Tool name.
* Tool description.
* Parameters.
* Required fields.
* Allowed domains.
* Search context size.
* Parallel tool behavior.
* Tool choice.
* Provider-specific options.

Tool construction should stay in `generators.py`.

The Streamlit UI should collect user settings. The provider wrapper should translate those settings
into the provider-specific request payload.

## 🧾 Response Formats

Generation workflows may support different response formats.

Common response-format concepts include:

* Plain text.
* JSON-like structured output.
* Provider-specific text format objects.
* Tool-call output.
* Search-grounded responses.
* Image or media references where supported.

Use plain text unless the workflow explicitly requires structured output.

Structured output is useful when the result should be parsed, stored, tested, or transformed. For
example:

```text
Return a JSON object with keys: title, summary, risks, next_steps.
```

When using structured output, inspect the provider response before storing or exporting it.

## 🧪 Translation

Translation workflows convert source text into another language.

Recommended workflow:

```text
Provide source text
        |
        v
Choose provider/model
        |
        v
Specify target language
        |
        v
Run translation
        |
        v
Review output
```

Example prompt:

```text
Translate the following text into Spanish. Preserve technical terms and Markdown formatting.
```

Review translations before publication, especially for legal, medical, technical, policy, or
public-facing content.

## 🎙️ Transcription

Transcription workflows convert audio input into text where supported by the provider wrapper.

Recommended workflow:

```text
Provide audio input
        |
        v
Choose transcription workflow
        |
        v
Run transcription
        |
        v
Review text output
        |
        v
Store or export if needed
```

Audio workflows may require supported file formats, provider credentials, and provider-specific
transcription settings.

Do not treat transcription output as perfect. Review it before using it as an official record or
documentation source.

## 🖼️ Image Workflows

Some provider wrappers may support image generation or image analysis.

Use image generation when the workflow requires a new visual asset.

Use image analysis when the workflow requires interpreting an existing image.

Possible use cases:

* Architecture diagrams.
* Documentation illustrations.
* Screenshot interpretation.
* Visual content summaries.
* UI analysis.
* Generated image assets.

Image workflows may produce files, URLs, or provider-specific objects. Review the returned object
shape before trying to store or export it.

## 🔄 Combining Generation with Other Modes

Generation mode is most useful when paired with other Foo workflows.

### Load and Summarize

```text
Loading mode
    |
    v
Load document
    |
    v
Generation mode
    |
    v
Summarize loaded text
    |
    v
Output workflow
    |
    v
Write Markdown summary
```

### Scrape and Explain

```text
Scraping mode
    |
    v
Extract headings and paragraphs
    |
    v
Generation mode
    |
    v
Explain page structure
    |
    v
Output workflow
    |
    v
Write review notes
```

### Retrieve and Draft

```text
Retrieval mode
    |
    v
Fetch public-data records
    |
    v
Generation mode
    |
    v
Draft explanation or report
    |
    v
Review and export
```

### Query and Report

```text
the persistence API
    |
    v
Query local table
    |
    v
Generation mode
    |
    v
Summarize results
    |
    v
Output workflow
    |
    v
Write report
```

## 🧪 Prompting Guidance

Prompts should be specific, scoped, and testable.

Weak prompt:

```text
Tell me about this.
```

Better prompt:

```text
Summarize the loaded document in five bullets. Include purpose, primary inputs, primary outputs, dependencies, and implementation risks.
```

Weak prompt:

```text
Make docs.
```

Better prompt:

```text
Create a MkDocs page for Foo's data-management layer. Include purpose, SQLite usage, Chroma usage, examples, maintenance guidance, and links to the API reference.
```

Good prompts specify:

* Task.
* Source context.
* Audience.
* Output format.
* Constraints.
* Required sections.
* What to exclude.

## 🔍 Reviewing Generated Output

Review generated output before using it.

Check:

* Does it answer the actual prompt?
* Is it grounded in the supplied source?
* Did it invent unsupported features?
* Does it preserve technical terms?
* Does it include unsupported claims?
* Does it expose sensitive information?
* Does the format match the requested output?
* Are code examples syntactically plausible?
* Are Markdown tables and fences valid?
* Should the output be edited before export?

Generated content should be treated as draft material unless it has been reviewed.

## 🔐 Sensitive Content Guidance

Generation workflows may send user content to external providers. Review source material before
generation.

Do not send or store sensitive content unless the workflow explicitly requires it.

Sensitive content may include:

* API keys.
* Tokens.
* Passwords.
* Credentials.
* Private messages.
* Email content.
* Internal documents.
* Legal records.
* Medical records.
* Financial records.
* Personnel records.
* Raw uploaded files.
* User-identifying paths.
* Proprietary data.

When logging generation errors, do not log full prompts, full documents, raw provider responses,
files, audio, image data, or credentials.

## 🛡️ Safe Generation Practices

Use these practices:

* Start with a small prompt.
* Use default model settings first.
* Review provider and model selection.
* Use allowed domains for search-assisted generation when source control matters.
* Do not enable tools unnecessarily.
* Do not send large documents without chunking or review.
* Avoid storing sensitive prompts or generated content unintentionally.
* Review output before publishing.
* Keep provider credentials in configuration.
* Keep provider-specific logic inside `generators.py`.
* Keep UI rendering inside `app.py`.

## ⚠️ Common Generation Issues

| Issue                     | Likely Cause                                                                   | Suggested Check                                            |
| ------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| Provider call fails       | Missing API key, invalid credential, unavailable service, or unsupported model | Check `config.py` and environment variables.               |
| Prompt is rejected        | Missing required prompt text or invalid provider payload                       | Try a shorter prompt with default options.                 |
| Reasoning option fails    | Selected model does not support reasoning/thinking setting                     | Disable reasoning or choose a supported model.             |
| Tool call fails           | Tool configuration is invalid for selected provider                            | Disable tools and retest basic generation.                 |
| Search gives poor results | Query too broad or domains too loose                                           | Narrow the prompt and allowed domains.                     |
| Output is too long        | Max-token setting too high or prompt too broad                                 | Add length and format constraints.                         |
| Output is generic         | Prompt lacks source context or concrete requirements                           | Include source, audience, and required sections.           |
| Output invents features   | Prompt asks for documentation without grounding source                         | Provide source context and require source-grounded claims. |
| Exported Markdown breaks  | Code fences, tables, or links are malformed                                    | Preview and edit before publishing.                        |

## 🧰 Adding a New Generation Workflow

When adding a new generation workflow:

1. Confirm the provider wrapper supports the underlying operation.
2. Add provider-specific logic to `generators.py`, not `app.py`.
3. Keep credentials in `config.py` or environment-backed configuration.
4. Add UI controls in `app.py`.
5. Validate required inputs before provider calls.
6. Keep tool construction in the provider wrapper.
7. Keep response extraction in the provider wrapper.
8. Avoid logging prompts, documents, credentials, or raw provider responses.
9. Return a predictable type.
10. Add or update documentation.
11. Confirm the generated API page renders without Griffe warnings.
12. Test the workflow with simple input before complex input.

## 📖 API Reference

Use the generated API page for implementation details:

* [Generators API](api/generators.md)

The API reference includes provider wrapper classes, method signatures, return annotations, and
docstrings generated from `generators.py`.

## ✅ Generation Checklist

Before treating a generation result as usable, confirm:

* The correct provider was selected.
* The correct model was selected.
* Required credentials are configured.
* Prompt text is present.
* Optional tools are intentional.
* Reasoning or thinking settings are supported by the selected model.
* Search domain restrictions are appropriate.
* The output answers the task.
* The output does not include unsupported claims.
* The output does not expose sensitive content.
* The output is reviewed before export or publication.

## 🧭 Summary

Foo’s generation workflow connects the Streamlit application to provider wrapper classes in
`generators.py`. It supports text generation, search-assisted generation, document summarization,
translation, transcription, image-related workflows, and provider-specific tool or reasoning
settings. Keep generation behavior provider-specific, review outputs carefully, and use the output
layer when generated content needs to become a durable artifact.
