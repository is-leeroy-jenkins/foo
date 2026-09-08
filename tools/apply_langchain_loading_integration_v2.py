'''Execute the anchored Loading-mode integration with resilient loader anchors.

Purpose:
    Reuses the approved integration transformer while locating loader labels regardless of
    quote style and preserving the existing loader-specific Load, Clear, and Save structures.
'''
from __future__ import annotations
import importlib.util
from pathlib import Path
import re
from types import ModuleType


BASE_SCRIPT = Path( 'tools' ) / 'apply_langchain_loading_integration.py'

TARGET_LOADERS = [
    ( 'Text Loader', 'TextLoader', 'txt' ),
    ( 'CSV Loader', 'CsvLoader', 'csv' ),
    ( 'PDF Loader', 'PdfLoader', 'pdf' ),
    ( 'Excel Loader', 'ExcelLoader', 'excel' ),
    ( 'Word Document Loader', 'WordLoader', 'word' ),
    ( 'Markdown Loader', 'MarkdownLoader', 'md' ),
    ( 'HTML Loader', 'HtmlLoader', 'html' ),
    ( 'JSON Loader', 'JsonLoader', 'json' ),
    ( 'Power Point Loader', 'PowerPointLoader', 'pptx' ),
]


def throw_if( name: str, value: object ) -> None:
    """Input guard.

    Purpose:
        Validates required script inputs before the source transformer is loaded.

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


def load_transformer( ) -> ModuleType:
    """Load the base integration transformer.

    Purpose:
        Imports the existing anchored transformer without executing its command-line entry point.

    Returns:
        ModuleType: Loaded integration-transformer module.
    """
    if not BASE_SCRIPT.exists( ):
        raise FileNotFoundError( str( BASE_SCRIPT ) )

    spec = importlib.util.spec_from_file_location( 'foo_loading_integration', BASE_SCRIPT )
    if spec is None or spec.loader is None:
        raise RuntimeError( 'Could not load the base Loading-mode integration transformer.' )

    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def locate_loader_section( source: str, label: str ) -> tuple[ int, int ]:
    """Locate one loader section using either Python quote style.

    Purpose:
        Anchors the complete loader expander without requiring its pre-existing Streamlit label
        to use a specific quote character.

    Args:
        source (str): Complete app.py source text.
        label (str): Existing loader expander label.

    Returns:
        tuple[int, int]: Start and end offsets for the complete loader section.
    """
    throw_if( 'source', source )
    throw_if( 'label', label )
    start_pattern = re.compile(
        rf'(?m)^\t{{3}}with st\.expander\( label=[\'\"]{re.escape( label )}[\'\"]',
    )
    start_match = start_pattern.search( source )
    if start_match is None:
        raise RuntimeError( f'Could not locate loader expander: {label}' )

    sibling_pattern = re.compile( r'(?m)^\t{3}# -+\n\t{3}# -+ Expander' )
    sibling_match = sibling_pattern.search( source, start_match.end( ) )
    if sibling_match is None:
        raise RuntimeError( f'Could not locate the end of loader expander: {label}' )
    return start_match.start( ), sibling_match.start( )


def add_loader_controls( section: str, loader_name: str, key_prefix: str ) -> str:
    """Add LangChain controls and actions to one loader section.

    Purpose:
        Inserts chunking and embedding controls before the existing Load/Clear/Save row, extends
        the existing Clear path with pipeline reset state, and appends Chunk, Embed, and Store
        actions without reconstructing source-specific loader logic.

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
            r'(?m)^\t{4}# (?:Buttons: )?Load / Clear / Save(?: controls)?(?: .*)?\s*$',
            section,
        )
        if button_match is None:
            raise RuntimeError( f'Could not locate Load/Clear/Save controls for {loader_name}.' )

        separator_match = None
        for candidate in re.finditer( r'(?m)^\t{4}# -+\n', section[ :button_match.start( ) ] ):
            separator_match = candidate

        insert_at = separator_match.start( ) if separator_match is not None else button_match.start( )
        section = section[ :insert_at ] + input_call + section[ insert_at: ]

    clear_pattern = re.compile( r'(?m)^(\t{4}if clear_[A-Za-z0-9_]+[^\n]*:\n)' )
    clear_match = clear_pattern.search( section )
    if clear_match is None:
        raise RuntimeError( f'Could not locate Clear execution block for {loader_name}.' )

    reset_line = f"\t\t\t\t\treset_langchain_controls( '{key_prefix}' )\n"
    if reset_line not in section[ clear_match.start( ): ]:
        section = section[ :clear_match.end( ) ] + reset_line + section[ clear_match.end( ): ]

    clear_match = clear_pattern.search( section )
    load_if_match = re.search(
        r'(?m)^\t{4}if load_[A-Za-z0-9_]+[^\n]*:\n',
        section[ clear_match.end( ): ],
    )
    if load_if_match is None:
        raise RuntimeError( f'Could not locate Load execution block for {loader_name}.' )

    load_if_index = clear_match.end( ) + load_if_match.start( )
    clear_block = section[ clear_match.start( ):load_if_index ]
    if 'st.rerun( )' not in clear_block:
        section = section[ :load_if_index ] + '\t\t\t\t\tst.rerun( )\n\t\t\t\t\n' + section[ load_if_index: ]

    if action_call.strip( ) not in section:
        section = section.rstrip( ) + action_call
    return section


def main( ) -> None:
    """Run the resilient Loading-mode integration.

    Purpose:
        Overrides only loader-location and loader-control patch behavior and delegates import,
        right-column, validation, and file-write operations to the existing anchored transformer.

    Returns:
        None: This function updates app.py through the base transformer.
    """
    transformer = load_transformer( )
    transformer.TARGET_LOADERS = TARGET_LOADERS
    transformer.locate_loader_section = locate_loader_section
    transformer.add_loader_controls = add_loader_controls
    transformer.main( )


if __name__ == '__main__':
    main( )
