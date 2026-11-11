"""Isolated NOAA Tides & Currents and EPA UV Index regression tests.

Purpose:
    Exercise both production provider wrappers with mocked HTTP responses.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import math
import re
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Reject missing required inputs.

    Purpose:
        Preserve Foo's required-input contract in isolated tests.

    Args:
        name (str): Argument name.
        value (object): Supplied value.

    Returns:
        None: Raises when input is absent.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Compile the production provider classes in isolation.

    Purpose:
        Run deterministic tests without loading unrelated integrations.

    Returns:
        tuple[dict[str, type], Mock]: Wrapper classes and HTTP mock.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef ) and node.name in (
            'TidesAndCurrents', 'UvIndex' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal provider base."""

    class Error( Exception ):
        """Wrapped provider exception."""

    class Logger:
        """No-op logger."""

        def write( self, exception: Exception ) -> None:
            """Accept wrapped errors without disk writes.

            Purpose:
                Keep production exception paths testable.

            Args:
                exception (Exception): Logged exception.

            Returns:
                None: No result.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger,
        'throw_if': throw_if, 'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo test client' ),
        'dt': dt, 'math': math, 're': re, 'urllib': urllib
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in (
        'TidesAndCurrents', 'UvIndex' ) }, http


class TidesAndCurrentsContractsTests( unittest.TestCase ):
    """Protect NOAA station and tidal data workflows."""

    def setUp( self ) -> None:
        """Initialize a mocked tide station.

        Purpose:
            Provide an independent provider instance for each test.

        Returns:
            None: Sets test fixtures.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'TidesAndCurrents' ]( )
        self.http.return_value.json.return_value = { 'stations': [
            { 'id': '9414290', 'name': 'San Francisco' }
        ] }

    def test_station_metadata_mode( self ) -> None:
        """Preserve station metadata retrieval.

        Purpose:
            Check station rows and request URLs.

        Returns:
            None: Assertions validate results.
        """
        result = self.client.fetch_station( '9414290' )
        self.assertEqual( result[ 'mode' ], 'station' )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Name' ], 'San Francisco' )

    def test_water_level_and_prediction_modes( self ) -> None:
        """Keep both NOAA data products functional.

        Purpose:
            Protect request product parameters and normalized rows.

        Returns:
            None: Assertions validate modes and query arguments.
        """
        for mode, field in ( ( 'water-level', 'data' ),
            ( 'tide-predictions', 'predictions' ) ):
            with self.subTest( mode=mode ):
                self.http.return_value.json.return_value = {
                    field: [ { 't': '2026-11-11 12:00', 'v': '1.4' } ]
                }
                result = self.client.fetch( mode=mode, station_id='9414290',
                    begin_date='20261110', end_date='20261111' )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'summary' ][ 'count' ], 1 )
                self.assertIn( 'product', self.http.call_args.kwargs[ 'params' ] )

    def test_calendar_validation_before_http( self ) -> None:
        """Reject impossible dates and reversed windows.

        Purpose:
            Fail before provider access.

        Returns:
            None: Assertions verify no requests.
        """
        for begin, end in ( ( '20260230', '20260301' ),
            ( '20261112', '20261110' ), ( '20261301', '20261302' ) ):
            with self.subTest( begin=begin, end=end ):
                self.http.reset_mock( )
                with self.assertRaises( Exception ):
                    self.client.fetch_water_level( '9414290', begin, end )
                self.http.assert_not_called( )

    def test_noaa_json_errors_and_params( self ) -> None:
        """Reject NOAA error payloads and wrong request types.

        Purpose:
            Prevent false successful responses.

        Returns:
            None: Assertions verify failures.
        """
        with self.assertRaises( Exception ):
            self.client.request( self.client.data_url, [ 'bad' ] )
        for payload in ( { 'error': { 'message': 'invalid station' } },
            { 'data': { 'v': 'invalid' } }, [ ] ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_water_level( '9414290', '20261110', '20261111' )


class UvIndexContractsTests( unittest.TestCase ):
    """Protect all EPA daily and hourly UV endpoint modes."""

    def setUp( self ) -> None:
        """Initialize a mocked UV Index provider.

        Purpose:
            Isolate HTTP transport and instance state.

        Returns:
            None: Sets test fixtures.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'UvIndex' ]( )
        self.http.return_value.json.return_value = [ ]

    def test_all_four_modes_use_current_epa_base( self ) -> None:
        """Preserve four modes using published EPA service paths.

        Purpose:
            Protect ZIP and city/state URLs.

        Returns:
            None: Assertions verify modes and request URLs.
        """
        cases = (
            ( 'daily-zip', { 'zip_code': '22201' } ),
            ( 'hourly-zip', { 'zip_code': '22201' } ),
            ( 'daily-city-state', { 'city': 'Arlington', 'state': 'VA' } ),
            ( 'hourly-city-state', { 'city': 'Arlington', 'state': 'VA' } )
        )
        for mode, options in cases:
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode, **options )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'rows' ], [ ] )
                self.assertTrue( self.http.call_args.kwargs[ 'url' ].startswith(
                    'https://data.epa.gov/dmapservice/' ) )

    def test_reject_bad_json_and_request_params( self ) -> None:
        """Reject invalid UV payload and parameter objects.

        Purpose:
            Avoid reporting malformed provider responses as empty forecasts.

        Returns:
            None: Assertions verify exceptions.
        """
        with self.assertRaises( Exception ):
            self.client.request( 'https://data.epa.gov/dmapservice/test', [ 'bad' ] )
        for payload in ( { 'error': 'failure' }, 'failure', [ None ] ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_daily_zip( '22201' )

    def test_uv_zero_and_nonfinite_values( self ) -> None:
        """Preserve true zero UV readings and ignore NaN.

        Purpose:
            Prevent incorrect UV summaries when alternative keys are present.

        Returns:
            None: Assertions verify summary values.
        """
        self.http.return_value.json.return_value = [
            { 'UV_VALUE': 0, 'INDEX': 9 },
            { 'UV_VALUE': 'nan' }
        ]
        result = self.client.fetch_hourly_zip( '22201' )
        self.assertEqual( result[ 'summary' ][ 'max_uv' ], 0.0 )
        self.assertEqual( result[ 'summary' ][ 'count' ], 2 )


if __name__ == '__main__':
    unittest.main( )
