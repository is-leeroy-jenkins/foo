"""Isolated AirNow and EPA Envirofacts provider regressions.

Purpose:
    Exercise production fetcher contracts using mocked HTTP without external dependencies.
"""
from __future__ import annotations

import ast
import copy
import math
import re
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate required input.

    Purpose:
        Supply the production-style argument guard for isolated tests.

    Args:
        name (str): Argument name.
        value (object): Argument value.

    Returns:
        None: Raises on missing input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Load two production classes and a mocked HTTP transport.

    Purpose:
        Execute the source implementations without unrelated integrations.

    Returns:
        tuple[dict[str, type], Mock]: Provider classes and HTTP mock.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef ) and node.name in ( 'AirNow', 'EnviroFacts' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal base provider."""

    class Error( Exception ):
        """Application error wrapper."""

    class Logger:
        """No-op application logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped exception.

            Purpose:
                Preserve the production error pathway.

            Args:
                exception (Exception): Error to log.

            Returns:
                None: No result.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger, 'throw_if': throw_if,
        'math': math, 're': re, 'urllib': urllib,
        'cfg': SimpleNamespace( AGENTS='Foo regression tests',
            AIRNOW_API_KEY='secret-test-key' ),
        'requests': SimpleNamespace( get=http ),
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in ( 'AirNow', 'EnviroFacts' ) }, http


class AirNowContractsTests( unittest.TestCase ):
    """Test AirNow search modes and request handling."""

    def setUp( self ) -> None:
        """Initialize independent AirNow state.

        Purpose:
            Set up mocked provider responses.

        Returns:
            None: Stores client and HTTP mock.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'AirNow' ]( )
        self.http.return_value.json.return_value = [ ]

    def test_four_modes_and_private_api_key( self ) -> None:
        """Preserve four request modes without credential disclosure.

        Purpose:
            Verify request authentication and public response metadata.

        Returns:
            None: Assertions validate provider contracts.
        """
        calls = (
            ( 'current-zip', { 'zip_code': '22201' } ),
            ( 'forecast-zip', { 'zip_code': '22201', 'date': '2026-11-10' } ),
            ( 'current-latlon', { 'latitude': 38.9, 'longitude': -77.1 } ),
            ( 'forecast-latlon', { 'latitude': 38.9, 'longitude': -77.1,
                'date': '2026-11-10' } )
        )
        for mode, args in calls:
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode, **args )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'rows' ], [ ] )
                self.assertNotIn( 'API_KEY', result[ 'params' ] )
                self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'API_KEY' ],
                    'secret-test-key' )

    def test_coordinates_and_zip_validation( self ) -> None:
        """Reject invalid geographic inputs before HTTP.

        Purpose:
            Prevent nonfinite, out-of-range, and malformed search parameters.

        Returns:
            None: Assertions verify failures.
        """
        cases = (
            ( 'current-zip', { 'zip_code': 'invalid' } ),
            ( 'forecast-zip', { 'zip_code': '1234', 'date': '2026-11-10' } ),
            ( 'current-latlon', { 'latitude': float( 'nan' ), 'longitude': -77 } ),
            ( 'current-latlon', { 'latitude': 91, 'longitude': -77 } ),
            ( 'forecast-latlon', { 'latitude': 38, 'longitude': 181,
                'date': '2026-11-10' } ),
            ( 'current-zip', { 'zip_code': '22201', 'distance': 0 } )
        )
        for mode, args in cases:
            with self.subTest( mode=mode, args=args ):
                self.http.reset_mock( )
                with self.assertRaises( Exception ):
                    self.client.fetch( mode=mode, **args )
                self.http.assert_not_called( )

    def test_reject_wrong_params_timeout_and_payload( self ) -> None:
        """Reject invalid transport arguments and provider responses.

        Purpose:
            Fail before packaging malformed data.

        Returns:
            None: Assertions verify exceptions.
        """
        for params, time in ( ( [ 'bad' ], 20 ), ( { }, 0 ) ):
            with self.assertRaises( Exception ):
                self.client.request( 'observation/zipCode/current/', params, time )
        for payload in ( { 'error': 'provider failed' }, 'bad', [ None ] ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_current_zip( '22201' )


class EnviroFactsContractsTests( unittest.TestCase ):
    """Test Envirofacts path and response contracts."""

    def setUp( self ) -> None:
        """Initialize mocked Envirofacts client.

        Purpose:
            Avoid external service calls.

        Returns:
            None: Stores provider and HTTP mock.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'EnviroFacts' ]( )
        self.http.return_value.json.return_value = [ ]

    def test_zero_based_pagination_and_state_filter( self ) -> None:
        """Build documented EPA table paths.

        Purpose:
            Preserve the exact row cap and TRI state column.

        Returns:
            None: Assertions verify URL segments.
        """
        result = self.client.fetch_table( table_name='TRI_FACILITY',
            state_code='VA', limit=25 )
        self.assertEqual( result[ 'summary' ][ 'count' ], 0 )
        self.assertIn( '/STATE_ABBR/VA/ROWS/0%3A24/JSON',
            self.http.call_args.kwargs[ 'url' ] )

    def test_facility_filter_and_row_mapping( self ) -> None:
        """Preserve filtered rows and friendly field names.

        Purpose:
            Confirm existing table shaping and result metadata.

        Returns:
            None: Assertions verify output.
        """
        self.http.return_value.json.return_value = [
            { 'FACILITY_NAME': 'Example Plant', 'STATE_ABBR': 'VA' }
        ]
        result = self.client.fetch_table( table_name='TRI_FACILITY',
            facility_name='Example Plant', limit=1 )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Facility Name' ], 'Example Plant' )
        self.assertIn( '/ROWS/0%3A0/JSON',
            self.http.call_args.kwargs[ 'url' ] )

    def test_reject_invalid_table_and_provider_payload( self ) -> None:
        """Reject unsupported tables and invalid provider results.

        Purpose:
            Avoid empty successes for malformed JSON.

        Returns:
            None: Assertions verify exceptions.
        """
        with self.assertRaises( Exception ):
            self.client.fetch_table( table_name='NOT_A_TABLE' )
        for payload in ( { 'error': 'broken' }, 'broken', [ None ] ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_table( table_name='TRI_FACILITY' )


if __name__ == '__main__':
    unittest.main( )
