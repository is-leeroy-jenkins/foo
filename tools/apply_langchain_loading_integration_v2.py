'''Execute the anchored Loading-mode integration with quote-agnostic loader anchors.

Purpose:
    Reuses the approved integration transformer while locating loader labels regardless of
    whether the existing Streamlit expander uses single or double quotes.
'''
from __future__ import annotations
import importlib.util
from pathlib import Path
import re
from types import ModuleType


BASE_SCRIPT = Path( 'tools' ) / 'apply_langchain_loading_integration.py'


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


def main( ) -> None:
    """Run the quote-agnostic Loading-mode integration.

    Purpose:
        Overrides only the loader-section locator and delegates all source edits and validation
        to the existing anchored transformer.

    Returns:
        None: This function updates app.py through the base transformer.
    """
    transformer = load_transformer( )
    transformer.locate_loader_section = locate_loader_section
    transformer.main( )


if __name__ == '__main__':
    main( )
