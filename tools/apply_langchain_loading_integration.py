'''Apply the approved LangChain Loading-mode integration to app.py.

Purpose:
    Performs anchored edits against complete loader sections so the existing Loading-mode
    structure, loader-specific controls, clear/load logic, and trailing NLP metrics remain intact.
'''
from __future__ import annotations
from pathlib import Path
import re


APP_PATH = Path( 'app.py' )

TARGET_LOADERS = [
    ( 'Text Loader', 'TextLoader', 'txt' ),
    ( 'CSV Loader', 'CsvLoader', 'csv' ),
    ( 'PDF Loader', 'PdfLoader', 'pdf' ),
    ( 'Excel Loader', 'ExcelLoader', 'excel' ),
    ( 'Word Loader', 'WordLoader', 'word' ),
    ( 'Markdown Loader', 'MarkdownLoader', 'md' ),
    ( 'HTML Loader', 'HtmlLoader', 'html' ),
    ( 'JSON Loader', 'JsonLoader', 'json' ),
    ( 'PowerPoint Loader', 'PowerPointLoader', 'pptx' ),
]


def throw_if( name: str, value: object ) -> None:
    """Input guard.

    Purpose:
        Validates required patch inputs before any source transformation occurs.

    Args:
        name (str): Name value used by the operation.
        value (object): Value value used by the operation.

    Returns:
        None: This function performs validation and does not return a value.
    """
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )

    if isinstance( value, str ) and not value:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


def insert_imports( source: str ) -> str:
    """Insert LangChain pipeline UI imports.

    Purpose:
        Adds the shared Loading-mode pipeline helpers immediately after the existing processor
        import without changing the surrounding import order.

    Args:
        source (str): Complete app.py source text.

    Returns:
        str: Source text containing the pipeline helper imports.
    """
    throw_if( 'source', source )
    if 'from langchain_pipeline import (' in source:
        return source

    anchor = 'from processors import PdfParser\n'
    if anchor not in source:
        raise RuntimeError( 'Could not locate the processors import anchor.' )

    addition = (
        "from langchain_pipeline import (render_langchain_inputs, render_langchain_actions,\n"
        "\treset_langchain_controls, render_loading_tabs)\n"
    )
    return source.replace( anchor, anchor + addition, 1 )


def locate_loader_section( source: str, label: str ) -> tuple[ int, int ]:
    """Locate one complete loader expander section.

    Purpose:
        Anchors changes to the selected loader expander and stops at the next sibling loader
        marker so no adjacent UI or execution block is reconstructed.

    Args:
        source (str): Complete app.py source text.
        label (str): Existing loader expander label.

    Returns:
        tuple[int, int]: Start and end offsets for the complete loader section.
    """
    throw_if( 'source', source )
    throw_if( 'label', label )
    start_anchor = f"\t\t\twith st.expander( label='{label}'"
    start = source.find( start_anchor )
    if start < 0:
        raise RuntimeError( f'Could not locate loader expander: {label}' )

    sibling_marker = '\n\t\t\t# ----------------------------\n\t\t\t# ------ Expander'
    end = source.find( sibling_marker, start + len( start_anchor ) )
    if end < 0:
        raise RuntimeError( f'Could not locate the end of loader expander: {label}' )
    return start, end


def add_loader_controls( section: str, loader_name: str, key_prefix: str ) -> str:
    """Add LangChain controls and actions to one loader section.

    Purpose:
        Inserts shared chunking and embedding inputs before the existing Load/Clear/Save row,
        extends the existing Clear execution path with pipeline reset state, and places Chunk,
        Embed, and Store actions at the bottom of the loader expander.

    Args:
        section (str): Complete source text for one loader expander.
        loader_name (str): Loader class name bound to the expander.
        key_prefix (str): Unique Streamlit widget-key prefix.

    Returns:
        str: Updated complete loader section.
    """
    throw_if( 'section', section )
    throw_if( 'loader_name', loader_name )
    throw_if( 'key_prefix', key_prefix )
    input_call = f"\t\t\t\trender_langchain_inputs( '{loader_name}', '{key_prefix}' )\n\t\t\t\t\n"
    action_call = (
        "\n\t\t\t\t# --------------------------------------------------\n"
        "\t\t\t\t# LangChain Actions\n"
        "\t\t\t\t# --------------------------------------------------\n"
        f"\t\t\t\trender_langchain_actions( '{loader_name}', '{key_prefix}' )\n"
    )

    if input_call.strip( ) not in section:
        button_match = re.search(
            r'(?m)^\t{4}# (?:Buttons: )?Load / Clear / Save(?: controls)?\s*$',
            section,
        )
        if button_match is None:
            raise RuntimeError( f'Could not locate Load/Clear/Save controls for {loader_name}.' )
        separator = section.rfind( '\t\t\t\t# --------------------------------------------------\n', 0,
            button_match.start( ) )
        insert_at = separator if separator >= 0 else button_match.start( )
        section = section[ :insert_at ] + input_call + section[ insert_at: ]

    clear_pattern = re.compile( r'(?m)^(\t{4}if clear_[A-Za-z0-9_]+[^\n]*:\n)' )
    clear_match = clear_pattern.search( section )
    if clear_match is None:
        raise RuntimeError( f'Could not locate Clear execution block for {loader_name}.' )

    reset_line = f"\t\t\t\t\treset_langchain_controls( '{key_prefix}' )\n"
    clear_block_start = clear_match.end( )
    if reset_line not in section[ clear_match.start( ): ]:
        section = section[ :clear_block_start ] + reset_line + section[ clear_block_start: ]

    load_marker = '\t\t\t\t# --------------------------------------------------\n\t\t\t\t# Load'
    clear_match = clear_pattern.search( section )
    load_index = section.find( load_marker, clear_match.end( ) )
    if load_index < 0:
        raise RuntimeError( f'Could not locate Load execution block for {loader_name}.' )

    clear_block = section[ clear_match.start( ):load_index ]
    if 'st.rerun( )' not in clear_block:
        section = section[ :load_index ] + '\t\t\t\t\tst.rerun( )\n\t\t\t\t\n' + section[ load_index: ]

    if action_call.strip( ) not in section:
        section = section.rstrip( ) + action_call
    return section


def patch_loader_sections( source: str ) -> str:
    """Patch all approved chunkable loader sections.

    Purpose:
        Applies the shared LangChain controls only to the loader classes already identified as
        chunkable in Foo's existing configuration contract.

    Args:
        source (str): Complete app.py source text.

    Returns:
        str: Source text containing all approved loader integrations.
    """
    throw_if( 'source', source )
    for label, loader_name, key_prefix in TARGET_LOADERS:
        start, end = locate_loader_section( source, label )
        section = source[ start:end ]
        updated = add_loader_controls( section, loader_name, key_prefix )
        source = source[ :start ] + updated + source[ end: ]
    return source


def patch_right_column( source: str ) -> str:
    """Replace only the Loading-mode right-column document renderer.

    Purpose:
        Moves the existing document preview into the shared Document tab and adds Chunks and
        Embeddings tabs while preserving the complete NLP metrics block that follows it.

    Args:
        source (str): Complete app.py source text.

    Returns:
        str: Source text using the three-tab Loading-mode result renderer.
    """
    throw_if( 'source', source )
    start_marker = (
        '\t# ------------------------------------------------------------------\n'
        '\t# RIGHT COLUMN — DOCUMENT RENDERING\n'
        '\t# ------------------------------------------------------------------\n'
    )
    end_marker = (
        '\t# ------------------------------------------------------------------\n'
        '\t# NLP METRIC CALCULATIONS\n'
        '\t# ------------------------------------------------------------------\n'
    )
    start = source.find( start_marker )
    end = source.find( end_marker, start + len( start_marker ) )
    if start < 0 or end < 0:
        raise RuntimeError( 'Could not locate the complete Loading-mode right-column section.' )

    replacement = (
        start_marker
        + '\twith right:\n'
        + '\t\trender_loading_tabs( )\n'
        + '\t\n'
    )
    return source[ :start ] + replacement + source[ end: ]


def main( ) -> None:
    """Apply the LangChain Loading-mode integration.

    Purpose:
        Reads app.py, applies all anchored transformations, validates that expected integration
        markers are present, and writes the complete updated source file.

    Returns:
        None: This function updates app.py on disk.
    """
    if not APP_PATH.exists( ):
        raise FileNotFoundError( str( APP_PATH ) )

    source = APP_PATH.read_text( encoding='utf-8' )
    source = insert_imports( source )
    source = patch_loader_sections( source )
    source = patch_right_column( source )

    expected = [
        'render_loading_tabs( )',
        "render_langchain_inputs( 'TextLoader', 'txt' )",
        "render_langchain_inputs( 'PowerPointLoader', 'pptx' )",
        "render_langchain_actions( 'TextLoader', 'txt' )",
        "render_langchain_actions( 'PowerPointLoader', 'pptx' )",
    ]
    for marker in expected:
        if marker not in source:
            raise RuntimeError( f'Missing required integration marker: {marker}' )

    APP_PATH.write_text( source, encoding='utf-8' )


if __name__ == '__main__':
    main( )
