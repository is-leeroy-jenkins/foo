"""Regression tests for Open-Meteo forecast and historical weather fetchers.

Purpose:
    Compile actual Foo fetcher classes with mocked HTTP to check optional
    timezone handling, geocoding errors, coordinates and archive responses.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Guard mandatory parameter values.

    Purpose:
        Supply Foo's required-value validation in the isolated runtime.

    Args:
        name (str): Argument name.
        value (object): Argument value.

    Returns:
        None: Raises ValueError when an argument is absent.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_weather( ) -> tuple[dict[str, type], Mock]:
    """Compile the production weather classes without optional dependencies.

    Purpose:
        Execute the source methods with a deterministic fake HTTP client.

    Returns:
        tuple[dict[str, type], Mock]: Weather classes and mocked HTTP GET.
    """
    source = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in source.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'OpenWeather', 'HistoricalWeather' ) ]
    for node in nodes:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Mock base class."""

    class Error( Exception ):
        """Source-compatible error wrapper."""

    class Logger:
        """Suppress application logging during regression tests."""

        def write( self, exception: Exception ) -> None:
            """Accept an error.

            Purpose:
                Preserve source exception pathways without external effects.

            Args:
                exception (Exception): Wrapped exception.

            Returns:
                None: No external effect.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo Open-Meteo tests' ),
        'dt': dt,
        'math': math
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in
        ( 'OpenWeather', 'HistoricalWeather' ) }, http


class OpenMeteoWeatherTests( unittest.TestCase ):
    """Check regressions in both weather classes."""

    def setUp( self ) -> None:
        """Initialize isolated clients and HTTP responses.

        Purpose:
            Reset mock provider state for every test.

        Returns:
            None: Stores production weather objects.
        """
        classes, self.http = load_weather( )
        self.forecast = classes[ 'OpenWeather' ]( )
        self.archive = classes[ 'HistoricalWeather' ]( )
        self.respond( { 'latitude': 38.9, 'longitude': -77.0,
            'timezone': 'America/New_York' } )

    def respond( self, payload: object ) -> None:
        """Set mocked JSON response data.

        Purpose:
            Make each test independent of the live Open-Meteo service.

        Args:
            payload (object): Provider JSON payload.

        Returns:
            None: Updates the mocked HTTP response.
        """
        self.http.return_value.json.return_value = payload

    def test_all_forecast_modes_allow_blank_timezone( self ) -> None:
        """Permit auto timezone selection in three forecast modes.

        Purpose:
            Ensure optional blank timezone values use the existing auto default.

        Returns:
            None: Assertions inspect query parameters and returned modes.
        """
        for name in ( 'fetch_current', 'fetch_hourly', 'fetch_daily' ):
            with self.subTest( method=name ):
                result = (getattr( self.forecast, name ))(
                    lat=38.9, long=-77.0, zone='' )
                self.assertEqual( result[ 'params' ][ 'timezone' ], 'auto' )
                self.assertEqual( result[ 'data' ][ 'latitude' ], 38.9 )

    def test_forecast_dispatch_allows_blank_timezone( self ) -> None:
        """Retain dispatcher functionality with auto timezone.

        Purpose:
            Confirm the previously rejected zone input flows to forecasts.

        Returns:
            None: Assertions inspect the selected API request.
        """
        self.respond( { 'results': [
            { 'name': 'Arlington', 'latitude': 38.9, 'longitude': -77.0 }
        ] } )
        self.http.return_value.json.side_effect = [
            { 'results': [ { 'latitude': 38.9, 'longitude': -77.0 } ] },
            { 'latitude': 38.9, 'longitude': -77.0 }
        ]
        result = self.forecast.fetch( location='Arlington', zone='' )
        self.assertEqual( result[ 'params' ][ 'timezone' ], 'auto' )
        self.assertEqual( result[ 'mode' ], 'current' )

    def test_archive_accepts_blank_timezone_and_date( self ) -> None:
        """Preserve historical date range and auto timezone.

        Purpose:
            Verify the archive URL and equal start/end dates.

        Returns:
            None: Assertions inspect archive request parameters.
        """
        result = self.archive.fetch_historical( lat=38.9, long=-77.0,
            date=dt.date( 2024, 1, 5 ), zone='' )
        self.assertEqual( result[ 'date' ], '2024-01-05' )
        self.assertEqual( result[ 'params' ][ 'start_date' ], '2024-01-05' )
        self.assertEqual( result[ 'params' ][ 'end_date' ], '2024-01-05' )
        self.assertEqual( result[ 'params' ][ 'timezone' ], 'auto' )

    def test_archive_rejects_invalid_date_before_network( self ) -> None:
        """Reject a non-date object.

        Purpose:
            Prevent attribute errors when constructing archive query parameters.

        Returns:
            None: Assertions verify early input rejection.
        """
        with self.assertRaises( Exception ):
            self.archive.fetch_historical( lat=38.9, long=-77.0,
                date='2024-01-05' )
        self.http.assert_not_called( )

    def test_invalid_coordinates_rejected_by_both_classes( self ) -> None:
        """Reject impossible or non-finite coordinates.

        Purpose:
            Prevent invalid latitude/longitude values reaching Open-Meteo.

        Returns:
            None: Assertions verify early validation.
        """
        cases = [
            lambda: self.forecast.fetch_daily( lat=91.0, long=0.0 ),
            lambda: self.forecast.fetch_hourly( lat=0.0, long=float( 'nan' ) ),
            lambda: self.forecast.fetch_current( lat=0.0, long=float( 'inf' ) ),
            lambda: self.archive.fetch_historical( lat=-91.0, long=0.0,
                date=dt.date( 2024, 1, 5 ) )
        ]
        for operation in cases:
            self.http.reset_mock( )
            with self.subTest( operation=operation ):
                with self.assertRaises( Exception ):
                    operation( )
                self.http.assert_not_called( )

    def test_api_error_payloads_rejected_by_both_classes( self ) -> None:
        """Reject Open-Meteo error documents.

        Purpose:
            Treat provider errors as failures rather than empty weather data.

        Returns:
            None: Assertions validate error handling.
        """
        self.respond( { 'error': True, 'reason': 'Unsupported location' } )
        with self.assertRaises( Exception ):
            self.forecast.fetch_current( 38.9, -77.0 )
        with self.assertRaises( Exception ):
            self.archive.fetch_historical( 38.9, -77.0,
                dt.date( 2024, 1, 5 ) )

    def test_invalid_json_and_geocoding_shapes_rejected( self ) -> None:
        """Reject malformed provider response structures.

        Purpose:
            Avoid calling dictionary operations on lists or strings.

        Returns:
            None: Assertions validate response shape enforcement.
        """
        self.respond( [ 1, 2 ] )
        with self.assertRaises( Exception ):
            self.forecast.fetch_current( 38.9, -77.0 )
        with self.assertRaises( Exception ):
            self.archive.fetch_historical( 38.9, -77.0,
                dt.date( 2024, 1, 5 ) )
        with self.assertRaises( Exception ):
            self.forecast.geocode_location( 'Arlington' )
        self.respond( { 'results': 'unexpected' } )
        with self.assertRaises( Exception ):
            self.archive.geocode_location( 'Arlington' )

    def test_invalid_geocoding_count_rejected_before_http( self ) -> None:
        """Validate the shared geocoding result-limit input.

        Purpose:
            Reject nonpositive result counts without provider calls.

        Returns:
            None: Assertions verify early rejection for both classes.
        """
        for client in ( self.forecast, self.archive ):
            self.http.reset_mock( )
            with self.subTest( client=type( client ).__name__ ):
                with self.assertRaises( Exception ):
                    client.geocode_location( 'Arlington', count=0 )
                self.http.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
