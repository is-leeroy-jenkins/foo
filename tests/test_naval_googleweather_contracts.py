"""Source-based regressions for NavalObservatory and GoogleWeather fetchers.

Purpose:
    Test two live provider classes with mocked HTTP and geocoding, without
    importing unrelated optional Foo packages or making external requests.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import math
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Guard required input values.

    Purpose:
        Mimic Foo's required argument contract in isolated source tests.

    Args:
        name (str): Required parameter.
        value (object): Supplied argument.

    Returns:
        None: Raises ValueError for absent values.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_providers( ) -> tuple[dict[str, type], Mock]:
    """Compile original provider classes in an isolated runtime.

    Purpose:
        Retain actual class methods while replacing dependencies with mocks.

    Returns:
        tuple[dict[str, type], Mock]: Provider classes and fake HTTP transport.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'NavalObservatory', 'GoogleWeather' ) ]
    for node in nodes:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Mock Fetcher base type."""

    class Error( Exception ):
        """Wrapped Foo error."""

    class Logger:
        """No-op application logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped provider error.

            Purpose:
                Preserve the exception pathway without side effects.

            Args:
                exception (Exception): Wrapped failure.

            Returns:
                None: No external effects.
            """
            pass

    class GoogleMaps:
        """Mock geocoder dependency."""

        def geocode_location( self, address: str ) -> tuple[float, float]:
            """Supply coordinates for a valid address.

            Purpose:
                Avoid Google Maps requests in weather regression tests.

            Args:
                address (str): Location query.

            Returns:
                tuple[float, float]: Representative latitude and longitude.
            """
            return 38.9072, -77.0369

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'GoogleMaps': GoogleMaps,
        'throw_if': throw_if,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo provider tests',
            GOOGLE_WEATHER_API_KEY='super-secret-weather-key' ),
        'dt': dt,
        'math': math,
        're': re
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { key: symbols[ key ] for key in
        ( 'NavalObservatory', 'GoogleWeather' ) }, http


class NavalObservatoryContractsTests( unittest.TestCase ):
    """Verify USNO celestial navigation parameters and error handling."""

    def setUp( self ) -> None:
        """Initialize mock USNO provider.

        Purpose:
            Reset dependencies for every test.

        Returns:
            None: Stores client and HTTP mock.
        """
        classes, self.http = load_providers( )
        self.client = classes[ 'NavalObservatory' ]( )
        self.http.return_value.json.return_value = { 'properties': { 'success': True } }

    def test_celnav_valid_result( self ) -> None:
        """Preserve valid USNO request and response shape.

        Purpose:
            Verify checked values pass into stored request parameters.

        Returns:
            None: Assertions validate result.
        """
        result = self.client.fetch_celnav( date_value='2026-01-01',
            time_value='12:00', latitude=38.9, longitude=-77.0, time=12 )
        self.assertEqual( result[ 'mode' ], 'celnav' )
        self.assertEqual( result[ 'params' ][ 'coords' ], '38.9,-77.0' )
        self.assertEqual( self.http.call_args.kwargs[ 'timeout' ], 12 )
        self.assertEqual( result[ 'data' ][ 'properties' ][ 'success' ], True )

    def test_celnav_rejects_invalid_dates_coordinates_and_timeouts( self ) -> None:
        """Reject invalid celestial navigation requests locally.

        Purpose:
            Ensure invalid date bounds, non-finite coordinates and timeout
            are rejected before an HTTP request.

        Returns:
            None: Assertions verify no transport calls.
        """
        valid = { 'date_value': '2026-01-01', 'time_value': '12:00',
            'latitude': 38.9, 'longitude': -77.0 }
        cases = [
            { 'date_value': '1700-01-01' },
            { 'date_value': '2051-01-01' },
            { 'time_value': '99:99' },
            { 'latitude': float( 'nan' ) },
            { 'longitude': float( 'inf' ) },
            { 'latitude': 91.0 },
            { 'time': 0 },
        ]
        for changed in cases:
            self.http.reset_mock( )
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.client.fetch_celnav( **{ **valid, **changed } )
                self.http.assert_not_called( )

    def test_celnav_provider_errors_and_invalid_json( self ) -> None:
        """Reject unsuccessful or malformed USNO payloads.

        Purpose:
            Prevent API errors from being passed to callers as valid data.

        Returns:
            None: Assertions validate wrapped failures.
        """
        for payload in ( [], { 'error': 'invalid date' } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_celnav( '2026-01-01', '12:00', 0.0, 0.0 )


class GoogleWeatherContractsTests( unittest.TestCase ):
    """Verify response parsing and API-key confidentiality."""

    def setUp( self ) -> None:
        """Initialize the weather provider.

        Purpose:
            Provide fake Google Maps and HTTP services.

        Returns:
            None: Stores client and transport mock.
        """
        classes, self.http = load_providers( )
        self.client = classes[ 'GoogleWeather' ]( )
        self.http.return_value.json.return_value = { 'temperature': 20 }

    def test_weather_request_redacts_api_key( self ) -> None:
        """Ensure the credential is used only for HTTP authentication.

        Purpose:
            Exclude Google Weather's API key from return metadata.

        Returns:
            None: Assertions validate request and result boundaries.
        """
        result = self.client.fetch_current( 'Arlington, Virginia' )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'key' ],
            'super-secret-weather-key' )
        self.assertNotIn( 'key', result[ 'params' ] )
        self.assertEqual( result[ 'mode' ], 'current' )
        self.assertEqual( result[ 'data' ][ 'temperature' ], 20 )

    def test_all_five_weather_methods_redact_credentials( self ) -> None:
        """Protect credential secrecy across each public weather mode.

        Purpose:
            Verify all existing operations use the shared safe result path.

        Returns:
            None: Assertions validate result metadata.
        """
        for method in (
            self.client.fetch_current,
            self.client.fetch_hourly_forecast,
            self.client.fetch_daily_forecast,
            self.client.fetch_hourly_history,
            self.client.fetch_alerts,
        ):
            with self.subTest( method=method.__name__ ):
                result = method( 'Arlington, Virginia' )
                self.assertNotIn( 'key', result[ 'params' ] )
                self.assertEqual( result[ 'data' ][ 'temperature' ], 20 )

    def test_weather_rejects_non_object_json( self ) -> None:
        """Reject malformed Google Weather response structures.

        Purpose:
            Enforce JSON object payloads expected by downstream callers.

        Returns:
            None: Assertions verify provider failure.
        """
        self.http.return_value.json.return_value = [ 1, 2 ]
        with self.assertRaises( Exception ):
            self.client.fetch_current( 'Arlington, Virginia' )

    def test_weather_rejects_bad_request_params_and_timeout( self ) -> None:
        """Validate the shared request API.

        Purpose:
            Fail before HTTP calls on invalid parameter types and timeouts.

        Returns:
            None: Assertions verify no HTTP request.
        """
        for params, timeout in ( ( [ 'invalid' ], 10 ), ( { 'a': 'b' }, 0 ) ):
            self.http.reset_mock( )
            with self.subTest( params=params, timeout=timeout ):
                with self.assertRaises( Exception ):
                    self.client.request( path='currentConditions:lookup',
                        params=params, time=timeout )
                self.http.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
