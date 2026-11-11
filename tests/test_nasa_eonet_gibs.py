"""Isolated functional regressions for NASA EONET and NASA GIBS WMS fetchers.

Purpose:
    Exercise the actual source class methods with mocked HTTP responses and filesystem
    operations, including invalid provider payloads and WMS coordinate conventions.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import math
import re
import tempfile
import unittest
import urllib
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Guard required values in isolated provider tests.

    Purpose:
        Match Foo's required-argument presence contract.

    Args:
        name (str): Required argument name.
        value (object): Required argument value.

    Returns:
        None: Invalid values raise ValueError.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Compile the actual provider classes with isolated dependencies.

    Purpose:
        Execute source methods without importing optional application packages or making
        network requests.

    Returns:
        tuple: Provider classes and shared mock HTTP get callable.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    selected = [ copy.deepcopy( node ) for node in module.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'EarthObservatory', 'GlobalImagery' ) ]

    for node in selected:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]

    selected_module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        *selected
    ], type_ignores=[ ] )
    ast.fix_missing_locations( selected_module )

    class Fetcher:
        """Isolated base class for the provider clients."""

    class Error( Exception ):
        """Expected wrapped provider failure."""

    class Logger:
        """Disable external error reporting in tests."""

        def write( self, exception: Exception ) -> None:
            """Accept a source exception without emitting output.

            Purpose:
                Preserve the production exception path.

            Args:
                exception (Exception): Raised provider failure.

            Returns:
                None: No external effects.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'cfg': SimpleNamespace( AGENTS='test-agent', NASA_API_KEY='test-key' ),
        'requests': SimpleNamespace( get=http ),
        'urllib': urllib,
        'dt': dt,
        're': re,
        'math': math,
        'Path': Path
    }
    exec( compile( selected_module, str( SOURCE ), 'exec' ), symbols )
    classes = { name: symbols[ name ] for name in ( 'EarthObservatory', 'GlobalImagery' ) }
    return classes, http


class NASAProviderRegressionTests( unittest.TestCase ):
    """Validate EONET response semantics and WMS provider contracts."""

    def setUp( self ) -> None:
        """Initialize isolated test providers.

        Purpose:
            Prepare fresh provider instances and a mocked HTTP client.

        Returns:
            None: Stores test dependencies.
        """
        classes, self.http = load_fetchers( )
        self.eonet = classes[ 'EarthObservatory' ]( )
        self.gibs = classes[ 'GlobalImagery' ]( )

    def respond( self, payload: object, content_type: str='application/json' ) -> None:
        """Configure the fake provider response.

        Purpose:
            Simulate a deterministic HTTP success with JSON or image content.

        Args:
            payload (object): JSON response payload.
            content_type (str): Response MIME type.

        Returns:
            None: Changes the mock response.
        """
        response = Mock( )
        response.json.return_value = payload
        response.headers = { 'Content-Type': content_type }
        response.status_code = 200
        response.content = b'image-bytes'
        self.http.return_value = response

    def test_eonet_categories_and_nested_layers( self ) -> None:
        """Flatten layer groups and preserve separate category listings.

        Purpose:
            Verify provider shape handling matches EONET nested categories.

        Returns:
            None: Assertions validate records, modes, and requested endpoints.
        """
        self.respond( { 'categories': [
            { 'id': 'wildfires', 'layers': [ { 'id': 'layer-a' } ] },
            { 'id': 'storms', 'layers': [ { 'id': 'layer-b' } ] }
        ] } )
        layers = self.eonet.fetch_layers( )
        self.assertEqual( [ item[ 'id' ] for item in layers[ 'layers' ] ],
            [ 'layer-a', 'layer-b' ] )
        self.assertEqual( layers[ 'mode' ], 'layers' )
        self.assertTrue( self.http.call_args.kwargs[ 'url' ].endswith( '/layers' ) )

        self.respond( { 'categories': [ { 'id': 'wildfires' } ] } )
        categories = self.eonet.fetch_categories( )
        self.assertEqual( categories[ 'categories' ][ 0 ][ 'id' ], 'wildfires' )

        self.respond( { 'layers': [ { 'id': 'layer-c' } ] } )
        one = self.eonet.fetch_layers( category='wildfires' )
        self.assertEqual( one[ 'layers' ][ 0 ][ 'id' ], 'layer-c' )
        self.assertTrue( self.http.call_args.kwargs[ 'url' ].endswith(
            '/layers/wildfires' ) )

    def test_eonet_events_filters_and_dates( self ) -> None:
        """Preserve supported events filters and date parameters.

        Purpose:
            Ensure explicit dates replace the relative-days filter.

        Returns:
            None: Assertions verify request parameter construction.
        """
        self.respond( { 'events': [ { 'id': 'test-event' } ] } )
        result = self.eonet.fetch_events( status='closed', limit=5,
            start_date='2026-01-01', end_date='2026-01-02' )
        self.assertEqual( result[ 'events' ][ 0 ][ 'id' ], 'test-event' )
        self.assertEqual( result[ 'params' ][ 'start' ], '2026-01-01' )
        self.assertNotIn( 'days', result[ 'params' ] )
        self.assertEqual( result[ 'params' ][ 'limit' ], 5 )

    def test_eonet_invalid_data_rejected_before_result( self ) -> None:
        """Reject provider errors, invalid record types, and filter values.

        Purpose:
            Prevent invalid responses and options from appearing as valid results.

        Returns:
            None: Assertions verify errors and early input validation.
        """
        invalid_filters = [
            lambda: self.eonet.fetch_events( status='unknown' ),
            lambda: self.eonet.fetch_events( limit=0 ),
            lambda: self.eonet.fetch_events( start_date='2026-01-01' ),
            lambda: self.eonet.fetch_events( start_date='2026-02-01',
                end_date='2026-01-01' ),
            lambda: self.eonet.fetch_layers( category='invalid/path' )
        ]
        for operation in invalid_filters:
            self.http.reset_mock( )
            with self.subTest( operation=operation ):
                with self.assertRaises( Exception ):
                    operation( )
                self.http.assert_not_called( )

        for payload in ( [ ], { 'error': 'API failed' },
                         { 'categories': [ { 'layers': 'invalid' } ] } ):
            self.respond( payload )
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.eonet.fetch_layers( )

    def test_wms_version_axis_order_and_crs( self ) -> None:
        """Select WMS 1.1.1 SRS versus WMS 1.3.0 CRS and axis order.

        Purpose:
            Protect EPSG:4326 latitude-first axis order for WMS 1.3.0.

        Returns:
            None: Assertions verify encoded WMS URL parameters.
        """
        old = self.gibs.build_wms_url( layer='MODIS_Test', image_date='2026-01-01',
            bbox=( -80, -30, 80, 30 ), version='1.1.1' )
        params = urllib.parse.parse_qs( urllib.parse.urlparse( old ).query )
        self.assertEqual( params[ 'SRS' ], [ 'EPSG:4326' ] )
        self.assertEqual( params[ 'BBOX' ], [ '-80.0,-30.0,80.0,30.0' ] )

        new = self.gibs.build_wms_url( layer='MODIS_Test', image_date='2026-01-01',
            bbox=( -80, -30, 80, 30 ), version='1.3.0' )
        params = urllib.parse.parse_qs( urllib.parse.urlparse( new ).query )
        self.assertEqual( params[ 'CRS' ], [ 'EPSG:4326' ] )
        self.assertEqual( params[ 'BBOX' ], [ '-30.0,-80.0,30.0,80.0' ] )
        self.assertNotIn( 'SRS', params )

        polar = self.gibs.build_wms_url( layer='MODIS_Test',
            image_date='2026-01-01', bbox=( -100, -100, 100, 100 ),
            projection='epsg3413' )
        self.assertIn( 'EPSG%3A3413', polar )

    def test_wms_invalid_parameters_rejected( self ) -> None:
        """Reject malformed geometry, WMS options, and image dimensions.

        Purpose:
            Ensure provider requests cannot proceed with invalid WMS contracts.

        Returns:
            None: Assertions confirm early rejection.
        """
        basic = { 'layer': 'MODIS_Test', 'image_date': '2026-01-01',
            'bbox': ( -80, -30, 80, 30 ) }
        cases = [
            { 'layer': '' },
            { 'bbox': ( -80, -30, 80 ) },
            { 'bbox': ( 80, -30, -80, 30 ) },
            { 'bbox': ( float( 'nan' ), -30, 80, 30 ) },
            { 'projection': 'unsupported' },
            { 'version': '1.0.0' },
            { 'quality': '../bad' },
            { 'width': 0 },
            { 'image_format': 'application/json' },
            { 'transparent': 'true' }
        ]
        for changed in cases:
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.gibs.build_wms_url( **{ **basic, **changed } )

    def test_wms_fetch_uses_stored_request_state( self ) -> None:
        """Save an image and expose the validated WMS metadata.

        Purpose:
            Ensure checked arguments flow through instance members into the request.

        Returns:
            None: Assertions verify path, HTTP call, and response metadata.
        """
        self.respond( { }, content_type='image/png' )
        with tempfile.TemporaryDirectory( ) as directory:
            result = self.gibs.fetch_wms_map( layer='MODIS_Test',
                image_date='2026-01-01', bbox=( -80, -30, 80, 30 ),
                output_dir=directory, output_name='map.png' )
            self.assertEqual( result[ 'layer' ], self.gibs.layer )
            self.assertEqual( result[ 'bbox' ][ 'west' ], self.gibs.bbox[ 0 ] )
            self.assertEqual( self.http.call_args.kwargs[ 'url' ], self.gibs.url )
            self.assertEqual( Path( result[ 'image_path' ] ).read_bytes( ),
                b'image-bytes' )

    def test_wms_wrappers_do_not_write_overwritten_mode( self ) -> None:
        """Prevent reintroduction of redundant wrapper-state assignments.

        Purpose:
            Verify convenience functions delegate without setting an unused mode.

        Returns:
            None: Assertions validate state preservation.
        """
        self.gibs.mode = 'sentinel'
        with patch.object( self.gibs, 'fetch_wms_map',
            return_value={ 'ok': True } ) as request:
            self.gibs.fetch_map_services( )
            self.assertEqual( self.gibs.mode, 'sentinel' )
            self.gibs.fetch_mercator_map( )
            self.assertEqual( self.gibs.mode, 'sentinel' )
            self.assertEqual( request.call_count, 2 )


if __name__ == '__main__':
    unittest.main( )
