"""Foo desktop application launcher.

Purpose:
    Host the Streamlit application in an isolated local server process and
    display its interface in a native Microsoft Edge WebView2 window.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def get_application_directory( ) -> Path:
    """Return the immutable application payload directory.

    Returns:
        Path: Directory containing the bundled Streamlit application.
    """
    return Path( getattr( sys, '_MEIPASS', Path( __file__ ).resolve( ).parent ) )


def find_available_port( ) -> int:
    """Choose a local TCP port.

    Returns:
        int: Available local TCP port selected by the operating system.
    """
    with socket.socket( socket.AF_INET, socket.SOCK_STREAM ) as server:
        server.bind( ('127.0.0.1', 0) )
        return int( server.getsockname( )[ 1 ] )


def run_server( ) -> None:
    """Execute Streamlit in the child process.

    Returns:
        None: The call blocks until the Streamlit server exits.
    """
    from streamlit.web import bootstrap

    base = get_application_directory( )
    app = base / 'app.py'
    os.chdir( base )
    bootstrap.run( str( app ), False, [ ], {
        'server.address': '127.0.0.1',
        'server.port': int( os.environ[ 'FOO_DESKTOP_PORT' ] ),
        'server.headless': True,
        'browser.gatherUsageStats': False,
        'server.fileWatcherType': 'none',
        'server.enableCORS': True,
        'server.enableXsrfProtection': True,
    } )


def wait_for_server( process: subprocess.Popen, port: int ) -> None:
    """Wait until Streamlit responds to its local health endpoint.

    Args:
        process (subprocess.Popen): Streamlit child process.
        port (int): Bound Streamlit port.

    Returns:
        None: Streamlit is responding to health probes.

    Raises:
        RuntimeError: Server exits or does not become ready in time.
    """
    endpoint = f'http://127.0.0.1:{port}/_stcore/health'
    deadline = time.monotonic( ) + 120
    while time.monotonic( ) < deadline:
        if process.poll( ) is not None:
            raise RuntimeError( 'Foo Streamlit process terminated during startup.' )
        try:
            with urlopen( endpoint, timeout=1 ) as response:
                if response.status == 200:
                    return
        except (URLError, OSError, TimeoutError):
            time.sleep( 0.35 )
    raise RuntimeError( 'Foo Streamlit process did not become ready within 120 seconds.' )


def main( ) -> None:
    """Run the Windows desktop UI and stop its server when the window closes.

    Returns:
        None: Cleans up the Streamlit child before exiting.
    """
    os.environ[ 'FOO_DESKTOP' ] = '1'
    if len( sys.argv ) > 1 and sys.argv[ 1 ] == '--serve':
        run_server( )
        return

    import webview

    port = find_available_port( )
    environment = os.environ.copy( )
    environment[ 'FOO_DESKTOP_PORT' ] = str( port )
    environment[ 'FOO_DESKTOP' ] = '1'
    local_data = Path( environment[ 'LOCALAPPDATA' ] ) / 'Foo'
    local_data.mkdir( parents=True, exist_ok=True )
    environment[ 'LOG_DIR' ] = str( local_data / 'logging' )
    server = subprocess.Popen(
        [sys.executable, '--serve'],
        cwd=str( get_application_directory( ) ),
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )
    try:
        wait_for_server( server, port )
        webview.create_window( 'Foo', f'http://127.0.0.1:{port}',
            width=1440, height=900, min_size=(1024, 700) )
        webview.start( gui='edgechromium' )
    finally:
        if server.poll( ) is None:
            server.terminate( )
            try:
                server.wait( timeout=10 )
            except subprocess.TimeoutExpired:
                server.kill( )
                server.wait( )


if __name__ == '__main__':
    main( )
