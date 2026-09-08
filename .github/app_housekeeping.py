from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Dict, List, Tuple

APP_PATH = Path('app.py')


def load_counts(tree: ast.AST) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            counts[node.id] = counts.get(node.id, 0) + 1
    return counts


def repo_python_name_count(name: str, exclude: Path) -> int:
    count = 0
    for path in Path('.').rglob('*.py'):
        if path == exclude or '.git' in path.parts or 'site' in path.parts:
            continue
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.NAME and token.string == name:
                count += 1
    return count


def string_reference_count(tree: ast.AST, name: str) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and name in node.value:
            count += 1
    return count


def remove_unused_functions(source: str) -> Tuple[str, List[str]]:
    tree = ast.parse(source)
    loads = load_counts(tree)
    candidates: List[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('__'):
            continue
        if loads.get(node.name, 0) != 0:
            continue
        if repo_python_name_count(node.name, APP_PATH) != 0:
            continue
        if string_reference_count(tree, node.name) != 0:
            continue
        candidates.append(node)

    lines = source.splitlines(keepends=True)
    removed: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    for node in candidates:
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        end = node.end_lineno or node.lineno
        spans.append((start, end, node.name))

    for start, end, name in sorted(spans, reverse=True):
        del lines[start - 1:end]
        removed.append(name)

    removed.reverse()
    return ''.join(lines), removed


def rename_private_functions(source: str) -> Tuple[str, Dict[str, str]]:
    tree = ast.parse(source)
    existing = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith('_')
    }
    rename_map: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith('_') or node.name.startswith('__'):
            continue
        target = node.name.lstrip('_')
        if target in existing:
            raise RuntimeError(f'Cannot rename {node.name}: {target} already exists.')
        rename_map[node.name] = target

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in rename_map:
            token = tokenize.TokenInfo(token.type, rename_map[token.string], token.start, token.end, token.line)
        tokens.append(token)
    return tokenize.untokenize(tokens), rename_map


def annotation_text(annotation: ast.AST | None) -> str:
    return ast.unparse(annotation) if annotation is not None else 'Any'


def function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[Tuple[str, str]]:
    arguments: List[Tuple[str, str]] = []
    for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arguments.append((argument.arg, annotation_text(argument.annotation)))
    if node.args.vararg is not None:
        arguments.append(('*' + node.args.vararg.arg, annotation_text(node.args.vararg.annotation)))
    if node.args.kwarg is not None:
        arguments.append(('**' + node.args.kwarg.arg, annotation_text(node.args.kwarg.annotation)))
    return arguments


def humanize(name: str) -> str:
    return name.replace('_', ' ').strip().capitalize() + '.'


def normalize_docstring_headings(doc: str) -> str:
    if not doc:
        return doc
    doc = re.sub(r'(?m)^(\s*)Parameters\s*:', r'\1Args:', doc)
    doc = re.sub(r'(?m)^(\s*)Parametes\s*:', r'\1Args:', doc)
    return doc


def doc_has_purpose(doc: str) -> bool:
    return bool(re.search(r'(?m)^\s*Purpose\s*:', doc))


def doc_has_args(doc: str) -> bool:
    return bool(re.search(r'(?m)^\s*Args\s*:', doc))


def doc_has_returns(doc: str) -> bool:
    return bool(re.search(r'(?m)^\s*Returns\s*:', doc))


def build_missing_sections(node: ast.FunctionDef | ast.AsyncFunctionDef, doc: str) -> str:
    doc = normalize_docstring_headings(doc)
    sections: List[str] = []
    if not doc_has_purpose(doc):
        sections.extend(['Purpose:', f'    {humanize(node.name)}'])

    if not doc_has_args(doc):
        sections.append('Args:')
        arguments = function_arguments(node)
        if arguments:
            for name, type_name in arguments:
                sections.append(f'    {name} ({type_name}): Runtime value consumed by this helper.')
        else:
            sections.append('    None.')

    if not doc_has_returns(doc):
        return_type = annotation_text(node.returns)
        sections.append('Returns:')
        if return_type in {'None', 'NoneType'}:
            sections.append('    None: This function does not return a value.')
        else:
            sections.append(f'    {return_type}: Result produced by this helper.')

    if not doc.strip():
        return '\n'.join(sections)
    if not sections:
        return doc
    return doc.rstrip() + '\n\n' + '\n'.join(sections)


def render_docstring(doc: str, indent: str) -> List[str]:
    rendered = [indent + '"""\n']
    for line in doc.splitlines():
        rendered.append(indent + line.rstrip() + '\n')
    rendered.append(indent + '"""\n')
    return rendered


def ensure_docstrings(source: str) -> Tuple[str, int]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes: List[Tuple[int, int, List[str]]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=True) or ''
        updated = build_missing_sections(node, doc)
        if doc and updated == doc:
            continue

        first_statement = node.body[0]
        statement_line = lines[first_statement.lineno - 1]
        indent = statement_line[:len(statement_line) - len(statement_line.lstrip())]

        if doc:
            doc_node = first_statement
            changes.append((doc_node.lineno, doc_node.end_lineno or doc_node.lineno,
                            render_docstring(updated, indent)))
        else:
            changes.append((first_statement.lineno, first_statement.lineno - 1,
                            render_docstring(updated, indent)))

    for start, end, replacement in sorted(changes, reverse=True):
        if end < start:
            lines[start - 1:start - 1] = replacement
        else:
            lines[start - 1:end] = replacement

    return ''.join(lines), len(changes)


def append_section(path: str, marker: str, content: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding='utf-8')
    if marker in text:
        return
    text = text.rstrip() + '\n\n' + content.strip() + '\n'
    file_path.write_text(text, encoding='utf-8')


def update_docs() -> None:
    append_section('README.md', '## 🧩 Document Processing', '''
## 🧩 Document Processing

Loading mode supports an explicit document-processing workflow:

```text
Documents
  -> RecursiveCharacterTextSplitter
  -> Chunks
  -> Embeddings
  -> Vector Store
```

The workflow is available for Text, CSV, PDF, Excel, Word, Markdown, HTML, JSON, and PowerPoint loaders. Chunk size and overlap are configurable per loader.

Embedding providers:

- OpenAI
- Google Generative AI
- Mistral AI
- Hugging Face
- Local GGUF through `llama-cpp-python`

Vector stores:

- Chroma for local persistent storage
- Pinecone for managed vector storage using an existing Pinecone index

`embedders.py` implements the LangChain-compatible embedding providers. `stores/vector.py` implements Chroma and Pinecone persistence. Loading results are presented through Document, Chunks, and Embeddings tabs.
''')

    append_section('docs/loading.md', '## 🧩 Chunking, Embedding, and Vector Storage', '''
## 🧩 Chunking, Embedding, and Vector Storage

The local document loaders for Text, CSV, PDF, Excel, Word, Markdown, HTML, JSON, and PowerPoint expose the same processing controls.

```text
Loaded LangChain Documents
        |
        v
RecursiveCharacterTextSplitter
        |
        v
Chunk Documents
        |
        v
Embedding Provider
  +-- OpenAI
  +-- Google Generative AI
  +-- Mistral AI
  +-- Hugging Face
  +-- Local GGUF
        |
        v
Vector Store
  +-- Chroma
  +-- Pinecone
```

The Chunks tab shows chunk metadata and text. The Embeddings tab shows provider, model, vector dimensions, source text, and generated vectors. The Store action persists the current chunks through the selected vector-store provider. Pinecone requires `PINECONE_API_KEY` and an existing index whose dimension matches the selected embedding model.
''')

    append_section('docs/app.md', '## 🧠 Document Processing Controls', '''
## 🧠 Document Processing Controls

`app.py` owns the Streamlit controls, session-state transitions, and orchestration for chunking, embedding, and vector storage. Reusable embedding implementations are provided by `embedders.py`; vector-store wrappers are provided by `stores/vector.py`.

The application maintains distinct derived state for chunk documents, embedding vectors, and the selected vector store. Source signatures prevent stale chunks or embeddings from being reused after the loaded document set changes.
''')

    append_section('docs/user-guide.md', '## 🧩 Document Processing Workflow', '''
## 🧩 Document Processing Workflow

1. Load a supported local document.
2. Set chunk size and overlap.
3. Select **Chunk**.
4. Select an embedding provider and model. For Local GGUF, provide the local `.gguf` model path.
5. Select **Embed**.
6. Select Chroma or Pinecone as the vector store.
7. For Pinecone, provide the existing index name and optional namespace.
8. Select **Store**.
9. Review Document, Chunks, and Embeddings tabs on the right side of Loading mode.
''')

    append_section('docs/index.md', '## 🧩 Document Processing', '''
## 🧩 Document Processing

Foo supports configurable LangChain document chunking, hosted or local embeddings, and Chroma/Pinecone vector persistence directly from Loading mode. See [Loading Data](loading.md) for the supported loaders, providers, and storage workflow.
''')

    append_section('docs/architecture.md', '## 🧩 Document Processing Components', '''
## 🧩 Document Processing Components

```text
app.py
  |
  +-- loaders.py                 source documents
  |
  +-- RecursiveCharacterTextSplitter
  |
  +-- embedders.py               OpenAI / Google / Mistral / Hugging Face / Local GGUF
  |
  +-- stores/vector.py           Chroma / Pinecone
```

`app.py` remains the Streamlit controller for this workflow. `embedders.py` contains reusable LangChain `Embeddings` implementations and provider construction. `stores/vector.py` contains vector-store persistence wrappers. No general-purpose pipeline module is used for this feature.
''')

    append_section('docs/development.md', '## Function Naming and Docstrings', '''
## Function Naming and Docstrings

Functions defined in `app.py` use descriptive public-style names without leading underscores. Every retained function uses a Google-style docstring with `Purpose:`, `Args:`, and `Returns:` sections. Functions that have no executable reference in the repository should be removed rather than retained as dormant helpers.
''')

    Path('docs/api/embedders.md').write_text('''# 🧠 Embedders

## Purpose

`embedders.py` provides LangChain-compatible embedding implementations used by Loading mode.

## Providers

| Provider | Implementation |
| --- | --- |
| OpenAI | `OpenAIEmbeddings` |
| Google Generative AI | `GoogleGenerativeAIEmbeddings` |
| Mistral AI | `MistralAIEmbeddings` |
| Hugging Face | `HuggingFaceEmbeddings` |
| Local GGUF | `LocalGGUFEmbeddings` using `llama-cpp-python` |

`EmbeddingFactory` resolves the selected provider behind the common LangChain `Embeddings` contract. Local GGUF models are loaded lazily from the configured filesystem path.
''', encoding='utf-8')

    Path('docs/api/vector-stores.md').write_text('''# 🗄️ Vector Stores

## Purpose

`stores/vector.py` provides the vector-storage implementations used by the Loading document-processing workflow.

## Stores

| Store | Behavior |
| --- | --- |
| Chroma | Local persistent storage under the configured persistence directory. |
| Pinecone | Managed storage in an existing Pinecone index and optional namespace. |

Both stores consume LangChain `Document` objects and a LangChain `Embeddings` implementation. Pinecone index creation remains external because the index dimension must match the selected embedding model.
''', encoding='utf-8')

    mkdocs = Path('mkdocs.yml')
    text = mkdocs.read_text(encoding='utf-8')
    if '      - Embedders: api/embedders.md' not in text:
        anchor = '      - Data: api/data.md\n'
        if anchor not in text:
            raise RuntimeError('MkDocs API navigation anchor was not found.')
        text = text.replace(anchor, anchor + '      - Embedders: api/embedders.md\n', 1)
    if '      - Vector Stores: api/vector-stores.md' not in text:
        anchor = '      - Writers: api/writers.md\n'
        if anchor not in text:
            raise RuntimeError('MkDocs vector-store navigation anchor was not found.')
        text = text.replace(anchor, anchor + '      - Vector Stores: api/vector-stores.md\n', 1)
    mkdocs.write_text(text, encoding='utf-8')


def validate_app(source: str) -> None:
    tree = ast.parse(source)
    loads = load_counts(tree)
    failures: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('_'):
            failures.append(f'Leading underscore remains: {node.name}@{node.lineno}')
        doc = ast.get_docstring(node, clean=True) or ''
        if not (doc_has_purpose(doc) and doc_has_args(doc) and doc_has_returns(doc)):
            failures.append(f'Noncompliant docstring: {node.name}@{node.lineno}')
        if loads.get(node.name, 0) == 0 and repo_python_name_count(node.name, APP_PATH) == 0 \
                and string_reference_count(tree, node.name) == 0:
            failures.append(f'Unused function remains: {node.name}@{node.lineno}')
    if failures:
        raise RuntimeError('\n'.join(failures))


def main() -> None:
    source = APP_PATH.read_text(encoding='utf-8')
    source, removed = remove_unused_functions(source)
    source, renamed = rename_private_functions(source)
    source, doc_changes = ensure_docstrings(source)
    validate_app(source)
    APP_PATH.write_text(source, encoding='utf-8')
    update_docs()
    print('Removed unused functions:', removed)
    print('Renamed functions:', renamed)
    print('Docstring changes:', doc_changes)


if __name__ == '__main__':
    main()
