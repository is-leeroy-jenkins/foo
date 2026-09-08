'''Run the document-processing transformer with conditional Clear-anchor support.'''
from __future__ import annotations
from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path( '.github' ) / 'apply_document_processing_fix.py'


def main( ) -> None:
    source = SCRIPT_PATH.read_text( encoding='utf-8' )
    old = "clear_pattern = re.compile( r'(?m)^(?P<indent>[ \\t]*)if clear_' + re.escape( key_prefix ) + r':' )"
    new = "clear_pattern = re.compile( r'(?m)^(?P<indent>[ \\t]*)if clear_' + re.escape( key_prefix ) + r'\\b[^:\\n]*:' )"
    if old not in source:
        raise RuntimeError( 'Clear-anchor transformer expression was not found.' )
    source = source.replace( old, new, 1 )
    SCRIPT_PATH.write_text( source, encoding='utf-8' )
    subprocess.run( [ sys.executable, str( SCRIPT_PATH ) ], check=True )


if __name__ == '__main__':
    main( )
