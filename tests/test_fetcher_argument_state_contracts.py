"""Regression tests for required fetcher arguments and meaningful instance state.

Purpose:
    Verify validated argument-to-member flow, provider call contracts, and
    the absence of overwritten GlobalImagery mode assignments.
"""
from __future__ import annotations

import ast
import copy
import re
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Reject missing required test inputs.

    Purpose:
        Supply the same presence contract for isolated class regression tests.

    Args:
        name (str): The argument being validated.
        value (object): Candidate argument value.

    Returns:
        None: Raises ValueError when the candidate is missing.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'Argument {name} cannot be empty.' )


def load_classes( ) -> dict[str, type]:
    """Load the two actual class definitions without external provider imports.

    Purpose:
        Isolate source-defined method behavior while replacing filesystem and HTTP
        dependencies with standard-library mocks.

    Returns:
        dict[str, type]: Compiled PurpleAir and GlobalImagery implementations.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ), filename=str( SOURCE ) )
    selected = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'PurpleAir', 'GlobalImagery' ) ]

    for node in selected:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]

    module = ast.Module( body=[
        ast.ImportFrom( module='__future__', names=[ ast.alias( name='annotations' ) ],
            level=0 ),
        *selected
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal inherited constructor for isolated service tests."""

        def __init__( self ) -> None:
            """Initialize the isolated base instance.

            Purpose:
                Preserve the constructor call without importing provider libraries.

            Returns:
                None: No base state is necessary for these tests.
            """
            pass

    class Error( Exception ):
        """Expected wrapped fetcher error type."""

    class Logger:
        """No-op logger to preserve exception-handler execution."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped exception.

            Purpose:
                Avoid I/O during isolated error-path tests.

            Args:
                exception (Exception): Reported exception.

            Returns:
                None: Reporting is intentionally isolated.
            """
            pass

    space = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'cfg': SimpleNamespace( NASA_API_KEY='test-key',
            PURPLEAIR_API_KEY='test-key', AGENTS='test-agent' ),
        'throw_if': throw_if,
        'requests': SimpleNamespace( get=Mock( ) ),
        'urllib': urllib,
        're': re,
        'Path': Path,
    }
    exec( compile( module, str( SOURCE ), 'exec' ), space )
    return { name: space[ name ] for name in ( 'PurpleAir', 'GlobalImagery' ) }


class FetcherStateContractTests( unittest.TestCase ):
    """Test actual source methods with isolated HTTP and file operations."""

    @classmethod
    def setUpClass( cls ) -> None:
        """Compile the affected classes for all test methods.

        Purpose:
            Reuse actual source definitions without importing optional dependencies.

        Returns:
            None: Stores the implementations on the test class.
        """
        cls.classes = load_classes( )

    def test_purpleair_dispatch_guards_before_delegation( self ) -> None:
        """Reject missing mandatory input before calling the delegated fetcher.

        Purpose:
            Verify checked instance member forwarding in both dispatch modes.

        Returns:
            None: Assertions verify call arguments and early rejection.
        """
        air = self.classes[ 'PurpleAir' ]( )
        with patch.object( air, 'fetch_sensors', return_value={ 'ok': True } ) as fetch:
            result = air.fetch( mode='sensors', nwlng=-78, nwlat=39,
                selng=-77, selat=38 )
            self.assertTrue( result[ 'ok' ] )
            self.assertEqual( fetch.call_args.kwargs[ 'nwlng' ], air.nwlng )
            self.assertEqual( fetch.call_args.kwargs[ 'selat' ], air.selat )
            fetch.reset_mock( )
            with self.assertRaises( Exception ):
                air.fetch( mode='sensors', nwlng=None, nwlat=39,
                    selng=-77, selat=38 )
            fetch.assert_not_called( )

        with patch.object( air, 'fetch_sensor', return_value={ 'ok': True } ) as fetch:
            air.fetch( mode='sensor', sensor_index=123 )
            self.assertEqual( fetch.call_args.kwargs[ 'sensor_index' ], air.sensor_index )
            fetch.reset_mock( )
            with self.assertRaises( Exception ):
                air.fetch( mode='sensor', sensor_index=None )
            fetch.assert_not_called( )

    def test_wms_url_and_request_use_checked_instance_members( self ) -> None:
        """Ensure WMS mandatory arguments become actively consumed instance state.

        Purpose:
            Verify formatted query parameters, response metadata, and saved image bytes.

        Returns:
            None: Assertions verify the mocked request and output file.
        """
        imagery = self.classes[ 'GlobalImagery' ]( )
        response = Mock( )
        response.headers = { 'Content-Type': 'image/png' }
        response.status_code = 200
        response.content = b'png-bytes'
        with tempfile.TemporaryDirectory( ) as directory:
            # Mock the request without importing the network client.
            request = imagery.__class__.fetch_wms_map.__globals__[ 'requests' ].get
            request.reset_mock( )
            request.return_value = response
            result = imagery.fetch_wms_map( layer='MODIS_Layer',
                image_date='2026-11-10', bbox=( -80, -40, 80, 40 ),
                output_dir=directory, output_name='map.png' )
            self.assertEqual( imagery.layer, 'MODIS_Layer' )
            self.assertEqual( imagery.image_date, '2026-11-10' )
            self.assertEqual( imagery.bbox, ( -80.0, -40.0, 80.0, 40.0 ) )
            self.assertEqual( result[ 'layer' ], imagery.layer )
            self.assertEqual( result[ 'bbox' ][ 'west' ], imagery.bbox[ 0 ] )
            self.assertEqual( result[ 'mode' ], 'wms_map' )
            self.assertIn( 'LAYERS=MODIS_Layer', request.call_args.kwargs[ 'url' ] )
            self.assertEqual( Path( result[ 'image_path' ] ).read_bytes( ), b'png-bytes' )

    def test_wms_missing_inputs_never_reach_http( self ) -> None:
        """Reject required missing WMS arguments without issuing a provider request.

        Purpose:
            Confirm guard checks in the public map method and URL builder.

        Returns:
            None: Assertions verify both input paths reject invalid values.
        """
        imagery = self.classes[ 'GlobalImagery' ]( )
        request = imagery.__class__.fetch_wms_map.__globals__[ 'requests' ].get
        request.reset_mock( )
        for layer, image_date, bbox in [
            ( None, '2026-11-10', ( -80, -40, 80, 40 ) ),
            ( 'MODIS_Layer', None, ( -80, -40, 80, 40 ) ),
            ( 'MODIS_Layer', '2026-11-10', None ),
        ]:
            with self.subTest( layer=layer, image_date=image_date, bbox=bbox ):
                with self.assertRaises( Exception ):
                    imagery.fetch_wms_map( layer=layer, image_date=image_date, bbox=bbox )
                with self.assertRaises( Exception ):
                    imagery.build_wms_url( layer=layer, image_date=image_date, bbox=bbox )
        request.assert_not_called( )

    def test_wms_delegators_do_not_write_unused_mode( self ) -> None:
        """Ensure convenience wrappers do not assign a mode their delegate overwrites.

        Purpose:
            Enforce the source-state rule on both imagery convenience methods.

        Returns:
            None: Assertions verify unchanged mode state before a mocked delegation.
        """
        imagery = self.classes[ 'GlobalImagery' ]( )
        imagery.mode = 'sentinel'
        with patch.object( imagery, 'fetch_wms_map', return_value={ 'ok': True } ) as fetch:
            imagery.fetch_map_services( )
            self.assertEqual( imagery.mode, 'sentinel' )
            imagery.fetch_mercator_map( )
            self.assertEqual( imagery.mode, 'sentinel' )
            self.assertEqual( fetch.call_count, 2 )


if __name__ == '__main__':
    unittest.main( )
