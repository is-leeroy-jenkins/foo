'''Run the document-processing transformer with the JSON/Web Documents boundary corrected.'''
from __future__ import annotations
from pathlib import Path
import subprocess
import sys


APP_PATH = Path( 'app.py' )
SCRIPT_PATH = Path( '.github' ) / 'apply_document_processing_fix_v2.py'


def main( ) -> None:
    subprocess.run( [ sys.executable, str( SCRIPT_PATH ) ], check=True )
    source = APP_PATH.read_text( encoding='utf-8' )
    action = "\t\t\t\trender_document_processing_actions( 'JsonLoader', 'json' )"
    web_anchor = "\t\twith st.expander( label='Web Documents', expanded=False ):"
    action_position = source.find( action )
    web_position = source.find( web_anchor )
    if action_position < 0 or web_position < 0:
        raise RuntimeError( 'JSON/Web Documents integration anchors were not found.' )

    if action_position > web_position:
        source = source[ :action_position ] + source[ action_position + len( action ): ]
        source = source.replace( '\n\n\n', '\n\n', 1 )
        web_position = source.find( web_anchor )
        source = source[ :web_position ] + action + '\n\n' + source[ web_position: ]

    APP_PATH.write_text( source, encoding='utf-8' )


if __name__ == '__main__':
    main( )
