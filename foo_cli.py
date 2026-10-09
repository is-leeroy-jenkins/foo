"""Start Foo's installed Streamlit application.

Purpose:
    Provide a console entry point without importing the Streamlit UI as a module.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main( ) -> None:
    """Launch Foo using the installed Python interpreter.

    Returns:
        None: Exits with Streamlit's process return code.
    """
    from foo_assets import __file__ as assets_file

    spec = importlib.util.find_spec( 'app' )
    if spec is None or spec.origin is None:
        raise RuntimeError( 'The Foo application module was not installed.' )

    assets = Path( assets_file ).resolve( ).parent
    user_path = Path( os.getenv( 'LOCALAPPDATA', str( Path.home( ) / '.foo' ) ) ) / 'Foo'
    user_path.mkdir( parents=True, exist_ok=True )
    images = assets / 'resources' / 'images'
    if images.is_dir( ):
        shutil.copytree( images, user_path / 'resources' / 'images', dirs_exist_ok=True )
    settings = user_path / '.streamlit'
    settings.mkdir( parents=True, exist_ok=True )
    config = settings / 'config.toml'
    if not config.exists( ):
        shutil.copy2( assets / 'streamlit_config.toml', config )
    ( user_path / 'stores' / 'sqlite' / 'datamodels' ).mkdir( parents=True, exist_ok=True )
    ( user_path / 'logging' ).mkdir( parents=True, exist_ok=True )

    environment = os.environ.copy( )
    environment[ 'FOO_DESKTOP' ] = '1'
    environment[ 'LOCALAPPDATA' ] = str( user_path.parent )
    environment[ 'LOG_DIR' ] = str( user_path / 'logging' )
    command = [sys.executable, '-m', 'streamlit', 'run', spec.origin]
    status = subprocess.call( command, cwd=str( user_path ), env=environment )
    raise SystemExit( status )


if __name__ == '__main__':
    main( )
