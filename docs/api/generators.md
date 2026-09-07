# Generators API

The `generators.py` module is Foo’s AI generation layer. It contains provider wrapper classes for
generating text, searching with model-backed tools, handling provider-specific configuration,
extracting response text, and exposing selected AI workflows to the Streamlit application.

This module should remain focused on AI-provider interaction and response handling. It should not
contain Streamlit layout code, database persistence logic, document loading logic, HTML scraping
logic, or output-file serialization.

## 🧭 Purpose

The Generators API provides a consistent wrapper layer around supported AI providers.

The module supports:

* Provider-specific client initialization.
* Prompt and instruction handling.
* Text generation.
* Web-search-assisted generation.
* Tool configuration.
* Response-format configuration.
* Reasoning or thinking configuration where supported.
* Image generation or image analysis for providers that support those workflows.
* Document summarization.
* File search.
* Translation.
* Transcription.
* Provider response extraction.
* Structured diagnostic logging in handled exception paths.

The module allows the Streamlit UI to call provider wrapper classes without embedding
provider-specific request construction directly in `app.py`.

## 🧱 Module Role

`generators.py` sits between Foo’s application layer and external AI providers.

```text
Streamlit UI
     |
     v
generators.py
     |
     +--> OpenAI / Chat
     +--> xAI Grok
     +--> Google Gemini
     +--> Anthropic Claude
     +--> Mistral
```

The UI should collect user options and pass them to generator classes. The generator classes should
validate the inputs, build provider-specific requests, call the provider, extract results, and
return documented values.

## 🧩 Class Overview

The module defines these primary classes:

| Class       | Purpose                                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `Generator` | Base generator class with shared generator state and a placeholder `fetch(...)` method.                               |
| `Grok`      | xAI Grok wrapper for response generation and web-search workflows.                                                    |
| `Gemini`    | Google Gemini wrapper for generation, thinking configuration, and search workflows.                                   |
| `Claude`    | Anthropic Claude wrapper for text generation and web-search-style workflows.                                          |
| `Mistral`   | Mistral wrapper for generation and schema creation.                                                                   |
| `Chat`      | OpenAI-oriented wrapper for text, image, document, file-search, web-search, translation, and transcription workflows. |

Each provider wrapper has its own configuration and helper methods because each provider exposes
different models, request parameters, tools, reasoning options, and response structures.

## ⚙️ Base Generator

The `Generator` class provides shared generator state and a common shape for provider wrappers.

It tracks default generation settings such as:

* `model`
* `instructions`
* `messages`
* `text`
* `prompt`
* `tools`
* `temperature`
* `top_percent`
* `max_tokens`
* `response_format`
* `stream`
* `store`

The base class is intentionally minimal. Provider-specific classes are responsible for implementing
the real request behavior.

Conceptual usage:

```python
from generators import Generator

generator = Generator()

print(generator.model)
print(generator.temperature)
```

The base `fetch(...)` method is a placeholder-style method. Use a provider-specific wrapper such as
`Chat`, `Grok`, `Gemini`, `Claude`, or `Mistral` for actual generation workflows.

## 💬 Chat

`Chat` is the OpenAI-oriented generation wrapper.

It supports a broad set of AI workflows, including:

* Text generation.
* Image generation.
* Image analysis.
* Document summarization.
* Web search.
* File search.
* Translation.
* Transcription.
* Response-format selection.
* Reasoning-effort options.
* Tool construction.
* Provider response extraction.
* Application data export through `get_data()` and `dump()`.

Primary methods include:

| Method                     | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `normalize_domains(...)`   | Normalizes allowed domains for web-search workflows.                   |
| `supports_reasoning(...)`  | Indicates whether the selected model supports reasoning configuration. |
| `build_instructions(...)`  | Builds provider instructions from prompt and context fields.           |
| `build_text_format(...)`   | Builds response-format configuration.                                  |
| `build_tools(...)`         | Builds provider tool definitions.                                      |
| `extract_output_text(...)` | Extracts text from provider response objects.                          |
| `fetch(...)`               | Executes the primary generation workflow.                              |
| `generate_text(...)`       | Generates text from a prompt.                                          |
| `generate_image(...)`      | Generates an image from a prompt.                                      |
| `analyze_image(...)`       | Analyzes image input.                                                  |
| `summarize_document(...)`  | Summarizes document content.                                           |
| `search_web(...)`          | Performs a web-search-assisted generation workflow.                    |
| `search_files(...)`        | Performs a file-search workflow.                                       |
| `translate(...)`           | Translates supplied text.                                              |
| `transcribe(...)`          | Transcribes supplied audio.                                            |
| `get_format_options(...)`  | Returns supported response-format options.                             |
| `get_model_options(...)`   | Returns supported model options.                                       |
| `get_effort_options(...)`  | Returns supported reasoning-effort options.                            |
| `get_data(...)`            | Returns current wrapper state as a dictionary.                         |
| `dump(...)`                | Serializes wrapper state.                                              |

Use `Chat` when the workflow targets OpenAI-compatible generation features.

## 🧪 Chat Example: Generate Text

```python
from generators import Chat

chat = Chat()
result = chat.generate_text(
    prompt="Summarize the purpose of Foo in one paragraph."
)

print(result)
```

Use this pattern for simple text generation where no image, file-search, transcription, or
web-search workflow is required.

## 🧪 Chat Example: Search the Web

```python
from generators import Chat

chat = Chat()
result = chat.search_web(
    prompt="Find current documentation guidance for MkDocs and mkdocstrings."
)

print(result)
```

Use web-search workflows when the prompt depends on current information or external source
discovery.

## 🧪 Chat Example: Summarize a Document

```python
from generators import Chat

chat = Chat()
summary = chat.summarize_document(
    prompt="Summarize this document for a technical maintainer."
)

print(summary)
```

Use document summarization when loaded document text or file-search context is already available to
the workflow.

## 🐦 Grok

`Grok` is the xAI Grok provider wrapper.

It supports provider-specific behavior for:

* Domain normalization.
* Reasoning-effort detection.
* Reasoning-object detection.
* Reasoning-model detection.
* Instruction construction.
* Tool construction.
* Response-format construction.
* Response creation.
* Text extraction.
* Text generation.
* Web search.

Primary methods include:

| Method                           | Purpose                                                           |
| -------------------------------- | ----------------------------------------------------------------- |
| `normalize_domains(...)`         | Normalizes web-search domain restrictions.                        |
| `supports_reasoning_effort(...)` | Indicates whether the selected model supports reasoning effort.   |
| `supports_reasoning_object(...)` | Indicates whether the selected model supports a reasoning object. |
| `is_reasoning_model(...)`        | Indicates whether the selected model is a reasoning model.        |
| `build_instructions(...)`        | Builds instructions for the Grok request.                         |
| `build_tools(...)`               | Builds Grok-compatible tool definitions.                          |
| `build_response_format(...)`     | Builds response-format configuration.                             |
| `extract_output_text(...)`       | Extracts output text from Grok response structures.               |
| `create_response(...)`           | Creates a provider response using the configured request payload. |
| `fetch(...)`                     | Executes the primary generation workflow.                         |
| `generate_text(...)`             | Generates text.                                                   |
| `search_web(...)`                | Executes web-search-assisted generation.                          |

Use `Grok` when the workflow targets xAI models or Grok-backed search/generation features.

## 🧪 Grok Example

```python
from generators import Grok

grok = Grok()
result = grok.generate_text(
    prompt="Explain the role of Foo's fetcher layer."
)

print(result)
```

Use the API reference below for exact argument signatures and supported configuration fields.

## 💎 Gemini

`Gemini` is the Google Gemini provider wrapper.

It supports provider-specific behavior for:

* Domain normalization.
* Stop-sequence normalization.
* Thinking-level support checks.
* Thinking-budget support checks.
* System-instruction construction.
* Thinking-configuration construction.
* Tool construction.
* Request-configuration construction.
* Text extraction.
* Text generation.
* Web-search-assisted generation.

Primary methods include:

| Method                          | Purpose                                                             |
| ------------------------------- | ------------------------------------------------------------------- |
| `normalize_domains(...)`        | Normalizes domain restrictions.                                     |
| `normalize_stop_sequences(...)` | Normalizes stop-sequence input.                                     |
| `supports_thinking_level(...)`  | Indicates whether the model supports thinking-level configuration.  |
| `supports_thinking_budget(...)` | Indicates whether the model supports thinking-budget configuration. |
| `build_system_instruction(...)` | Builds Gemini system instruction content.                           |
| `build_thinking_config(...)`    | Builds thinking configuration where supported.                      |
| `build_tools(...)`              | Builds Gemini-compatible tool configuration.                        |
| `build_config(...)`             | Builds the provider request configuration.                          |
| `extract_text(...)`             | Extracts response text from Gemini response objects.                |
| `fetch(...)`                    | Executes the primary generation workflow.                           |
| `generate_text(...)`            | Generates text.                                                     |
| `search_web(...)`               | Executes web-search-assisted generation.                            |

Use `Gemini` when the workflow targets Google Gemini generation models or Gemini-backed
search/generation behavior.

## 🧪 Gemini Example

```python
from generators import Gemini

gemini = Gemini()
result = gemini.generate_text(
    prompt="Describe the difference between loaders and fetchers in Foo."
)

print(result)
```

Use model-specific configuration methods when the selected Gemini model supports thinking or tool
behavior.

## 🧠 Claude

`Claude` is the Anthropic Claude provider wrapper.

It supports provider-specific behavior for:

* Domain normalization.
* Thinking support checks.
* Provider response text extraction.
* Text generation.
* Web-search-style generation.

Primary methods include:

| Method                    | Purpose                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| `_normalize_domains(...)` | Normalizes allowed domains for search-style workflows.                |
| `_supports_thinking(...)` | Indicates whether the selected model supports thinking configuration. |
| `_extract_text(...)`      | Extracts text from Claude response structures.                        |
| `fetch(...)`              | Executes the primary generation workflow.                             |
| `generate_text(...)`      | Generates text.                                                       |
| `search_web(...)`         | Executes web-search-assisted or search-style generation.              |

Methods beginning with an underscore are internal helpers. They are documented because they exist in
the source and may appear in the generated API reference, but application workflows should normally
call `fetch(...)`, `generate_text(...)`, or `search_web(...)`.

## 🧪 Claude Example

```python
from generators import Claude

claude = Claude()
result = claude.generate_text(
    prompt="Write a short explanation of Foo's data-management layer."
)

print(result)
```

Use the API reference for exact supported fields and provider-specific behavior.

## 🌊 Mistral

`Mistral` is the Mistral provider wrapper.

It supports:

* Provider response text extraction.
* Primary fetch/generation behavior.
* Schema creation.

Primary methods include:

| Method               | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| `_extract_text(...)` | Extracts text from Mistral response objects.    |
| `fetch(...)`         | Executes the primary Mistral workflow.          |
| `create_schema(...)` | Creates a schema definition for tool-style use. |

Use `Mistral` when the workflow targets a Mistral-backed text-generation or schema-oriented
operation.

## 🧪 Mistral Example

```python
from generators import Mistral

mistral = Mistral()
result = mistral.fetch(
    prompt="Summarize the Foo architecture."
)

print(result)
```

Use the exact method signatures from the generated API reference.

## 🧰 Tool Construction

Several generator wrappers build tool definitions for provider requests.

Tool construction can include:

* Web-search tools.
* File-search tools.
* Response-format tools.
* Provider-native function or tool objects.
* Domain restrictions.
* Search context size.
* Parallel tool settings.
* Tool-choice settings.

Tool behavior differs by provider. For that reason, each provider wrapper owns its own
`build_tools(...)` or equivalent helper.

General pattern:

```text
User options
    |
    v
build_tools(...)
    |
    v
Provider-specific request payload
    |
    v
Provider response
```

Keep tool construction inside provider wrappers. Do not duplicate provider-specific tool payloads in
`app.py`.

## 🧾 Response Extraction

Each provider returns a different response shape.

The generator layer includes extraction helpers such as:

* `extract_output_text(...)`
* `extract_text(...)`
* `_extract_text(...)`

These methods isolate provider-specific parsing logic so the rest of the application can work with
returned strings instead of provider response objects.

Conceptual pattern:

```text
Provider response object
        |
        v
extract text helper
        |
        v
plain string result
```

If a provider changes its response format, update the extraction helper instead of changing the
Streamlit UI.

## ⚙️ Reasoning and Thinking Configuration

Several providers expose reasoning or thinking configuration, but the feature is not uniform across
providers.

Foo keeps provider-specific checks in the provider classes:

| Provider Class | Capability Checks                                                                             |
| -------------- | --------------------------------------------------------------------------------------------- |
| `Chat`         | `supports_reasoning(...)`                                                                     |
| `Grok`         | `supports_reasoning_effort(...)`, `supports_reasoning_object(...)`, `is_reasoning_model(...)` |
| `Gemini`       | `supports_thinking_level(...)`, `supports_thinking_budget(...)`                               |
| `Claude`       | `_supports_thinking(...)`                                                                     |

This avoids applying a provider option to a model that does not support it.

## 🔎 Search-Assisted Generation

Several generator classes include `search_web(...)`.

Search-assisted generation should be used when the response depends on external information, source
discovery, or current web content.

Typical search-assisted workflow:

```text
Prompt
  |
  v
Normalize domains / search options
  |
  v
Build provider tools
  |
  v
Call provider
  |
  v
Extract text
  |
  v
Return response string
```

Use allowed-domain options when a workflow should be constrained to specific sources.

## 🧪 Output and State Inspection

`Chat` includes methods for inspecting or serializing current wrapper state:

| Method                    | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `get_format_options(...)` | Returns response-format options.              |
| `get_model_options(...)`  | Returns model options.                        |
| `get_effort_options(...)` | Returns reasoning-effort options.             |
| `get_data(...)`           | Returns current wrapper data as a dictionary. |
| `dump(...)`               | Serializes wrapper data.                      |

These methods are useful for Streamlit controls, diagnostics, and configuration review.

## 🔐 Error Handling

Generator classes use Foo’s structured error logging pattern where handled exception paths exist.

The preferred pattern is:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "generators"
    exception.cause = "Chat"
    exception.method = "generate_text( self, prompt: str ) -> str"
    Logger().write(exception)
    raise exception
```

Error metadata should be stable and reviewer-safe.

Do not include:

* API keys.
* Tokens.
* Full prompts when they may contain user data.
* Full documents.
* Full provider responses.
* Raw image data.
* Audio content.
* File contents.
* Credentials.
* Sensitive request payloads.

The logger should help identify the failing module, class, and method without capturing sensitive
content.

## 🛡️ Provider Configuration Guidance

When adding or modifying provider wrappers:

* Keep API keys in `config.py` or environment-backed configuration.
* Do not hard-code credentials.
* Validate required prompt or input values before calling the provider.
* Keep provider-specific payload construction inside the provider class.
* Keep response parsing inside extraction helpers.
* Preserve existing return behavior.
* Preserve existing fallback behavior.
* Use stable error metadata.
* Avoid logging prompts, documents, files, tokens, or raw provider responses.
* Document meaningful return values.
* Keep UI rendering out of `generators.py`.

## 🧪 Testing Generators

When testing generator classes, verify:

* Missing prompts are rejected or handled as documented.
* Provider clients initialize only when required configuration exists.
* Tool payloads are built correctly for the selected provider.
* Reasoning or thinking options are only applied to supported models.
* Response extraction handles normal provider responses.
* Response extraction handles empty or unexpected provider responses.
* Search workflows honor domain restrictions when supported.
* Return values match the documented type.
* Existing exception handlers log before re-raising.
* Error metadata does not include sensitive content.
* The Streamlit UI can display returned values without type errors.

Provider tests may require credentials. Keep tests that require live providers separate from tests
that validate local helper behavior.

## 🧩 Relationship to Streamlit UI

`app.py` should call generator classes as service wrappers.

The UI should handle:

* Mode selection.
* Prompt input.
* Model selection.
* Temperature and token settings.
* Tool-selection controls.
* Response display.
* Error display.
* Passing output to writers or persistence workflows.

The generator classes should handle:

* Provider configuration.
* Provider request building.
* Provider calls.
* Provider response extraction.
* Provider-specific validation.
* Provider-specific tool construction.

This separation prevents provider logic from spreading across the UI code.

## 🗄️ Relationship to Persistence

Generator outputs may be persisted through `data.py`.

Good SQLite candidates include:

* Prompt metadata.
* Model name.
* Provider name.
* Workflow mode.
* Generation timestamp.
* Response summary.
* Tool configuration metadata.
* Source references.
* Usage metadata when available.

Good Chroma candidates include:

* Generated summaries.
* Generated knowledge-base entries.
* Generated explanations.
* Text output intended for semantic retrieval.

Do not store secrets, credentials, or sensitive user content unless the workflow explicitly requires
secure persistence.

## 🧾 Relationship to Writers

Generator outputs can be exported through `writers.py`.

Good export candidates include:

* Markdown summaries.
* Research notes.
* Generated documentation drafts.
* Extracted explanations.
* Search-assisted responses.
* Translation output.
* Transcription output.
* Document summaries.

Generator classes should return content. Writer classes should serialize it.

## 📖 API Documentation

The generated API reference for this module is rendered below.
