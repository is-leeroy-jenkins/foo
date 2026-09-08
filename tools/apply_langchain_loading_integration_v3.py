'''Apply the final Loading-mode integration boundary correction.

Purpose:
    Runs the guarded Loading-mode transformer and corrects the one known container transition
    where the JSON loader is immediately followed by the Web Documents expander.
'''
from __future__ import annotations
import importlib.util
from pathlib import Path
import re
from types import ModuleType


V2_SCRIPT = Path( 'tools' ) / 'apply_langchain_loading_integration_v2.py'
APP_PATH = Path( 'app.py' )


def load_transformer( ) -> ModuleType:
    """Load the version-two integration transformer.

    Purpose:
        Imports the current anchored transformer without executing its command-line entry point.

    Returns:
        ModuleType: Loaded integration-transformer module.
    """
    if not V2_SCRIPT.exists( ):
        raise FileNotFoundError( str( V2_SCRIPT ) )

    spec = importlib.util.spec_from_file_location( 'foo_loading_integration_v2', V2_SCRIPT )
    if spec is None or spec.loader is None:
        raise RuntimeError( 'Could not load the Loading-mode integration transformer.' )

    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def correct_json_boundary( source: str ) -> str:
    """Move JSON pipeline actions before the Web Documents sibling container.

    Purpose:
        Corrects the deterministic Local Documents boundary produced after the base transformer
        reattaches the Web Documents expander so JSON actions remain inside the JSON loader.

    Args:
        source (str): Complete transformed app.py source text.

    Returns:
        str: Source text with the JSON action block preceding the Web Documents expander.
    """
    pattern = re.compile(
        r"(?m)"
        r"(^\t{3}with st\.expander\( label='Web Documents', expanded=False \):\n)"
        r"(\t{4}# -+\n"
        r"\t{4}# LangChain Actions\n"
        r"\t{4}# -+\n"
        r"\t{4}render_langchain_actions\( 'JsonLoader', 'json' \)\n)"
    )
    match = pattern.search( source )
    if match is None:
        raise RuntimeError( 'Could not locate the generated JSON/Web Documents boundary.' )

    return source[ :match.start( ) ] + match.group( 2 ) + match.group( 1 ) + source[ match.end( ): ]


def main( ) -> None:
    """Apply and finalize the Loading-mode integration.

    Purpose:
        Executes the anchored integration, applies the JSON container-boundary correction, and
        writes the final complete app.py source for compilation and structural validation.

    Returns:
        None: This function updates app.py on disk.
    """
    transformer = load_transformer( )
    transformer.main( )
    source = APP_PATH.read_text( encoding='utf-8' )
    source = correct_json_boundary( source )
    APP_PATH.write_text( source, encoding='utf-8' )


if __name__ == '__main__':
    main( )
