"""Isolated NASA DONKI SpaceWeather provider-contract tests.

Purpose:
    Exercise actual Foo SpaceWeather source methods using mocked HTTP transport,
    including optional dates, endpoint validation and credential confidentiality.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate mandatory input.

    Purpose:
        Supply the production required-value guard in the isolated test context.

    Args:
        name (str): Parameter name.
        value (object): Supplied value.

    Returns:
        None: Raises ValueError on missing input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_space_weather( ) -> tuple[object, Mock]:
    """Load the production class with mocked dependencies.

    Purpose:
        Avoid importing unrelated application packages or accessing the network.

    Returns:
        tuple[object, Mock]: Provider and mocked requests.get method.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    node = next( copy.deepcopy( item ) for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == 'SpaceWeather' )
    node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    isolated = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        node
    ], type_ignores=[ ] )
    ast.fix_missing_locations( isolated )

    class Fetcher:
        """Minimal base class."""

    class Error( Exception ):
        """Provider error wrapper."""

    class Logger:
        """No-op logger."""

        def write( self, exception: Exception ) -> None:
            """Accept wrapped exceptions.

            Purpose:
                Preserve application error pathways.

            Args:
                exception (Exception): Wrapped provider error.

            Returns:
                None: No side effects.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( NASA_API_KEY='private-test-key',
            AGENTS='Foo SpaceWeather contract tests' ),
        'dt': dt
    }
    exec( compile( isolated, str( SOURCE ), 'exec' ), symbols )
    return symbols[ 'SpaceWeather' ]( ), http


class SpaceWeatherContractsTests( unittest.TestCase ):
    """Verify all DONKI fetch modes and critical failure paths."""

    def setUp( self ) -> None:
        """Initialize mocked DONKI state.

        Purpose:
            Prevent shared HTTP side effects between tests.

        Returns:
            None: Stores fresh provider and transport.
        """
        self.client, self.http = load_space_weather( )
        self.respond( [ { 'activityID': 'test-cme' } ] )

    def respond( self, payload: object ) -> None:
        """Set the JSON response.

        Purpose:
            Simulate DONKI HTTP success without external requests.

        Args:
            payload (object): Response JSON document.

        Returns:
            None: Reconfigures the response mock.
        """
        response = Mock( )
        response.json.return_value = payload
        self.http.return_value = response

    def test_no_dates_use_nasa_defaults( self ) -> None:
        """Allow DONKI's documented default date range.

        Purpose:
            Prevent empty date strings from being sent to the provider.

        Returns:
            None: Assertions verify filtered request parameters.
        """
        result = self.client.fetch( mode='cme' )
        self.assertEqual( result[ 'mode' ], 'cme' )
        self.assertNotIn( 'startDate', self.http.call_args.kwargs[ 'params' ] )
        self.assertNotIn( 'endDate', self.http.call_args.kwargs[ 'params' ] )
        self.assertTrue( result[ 'url' ].endswith( '/CME' ) )

    def test_explicit_dates_and_route( self ) -> None:
        """Preserve explicitly supplied valid date windows.

        Purpose:
            Verify date range forwarding and endpoint routing.

        Returns:
            None: Assertions verify recorded request state.
        """
        result = self.client.fetch( mode='gst',
            start_date='2026-01-01', end_date='2026-01-02' )
        self.assertEqual( result[ 'params' ][ 'startDate' ], '2026-01-01' )
        self.assertEqual( result[ 'params' ][ 'endDate' ], '2026-01-02' )
        self.assertEqual( result[ 'endpoint' ], 'GST' )

    def test_all_supported_modes( self ) -> None:
        """Preserve 11 DONKI modes.

        Purpose:
            Ensure all established dispatcher mappings remain reachable.

        Returns:
            None: Assertions validate endpoint routing.
        """
        expected = {
            'cme': 'CME', 'cme_analysis': 'CMEAnalysis',
            'gst': 'GST', 'ips': 'IPS', 'flr': 'FLR',
            'sep': 'SEP', 'mpc': 'MPC', 'rbe': 'RBE',
            'hss': 'HSS', 'wsa_enlil': 'WSAEnlilSimulations',
            'notifications': 'notifications'
        }
        for mode, endpoint in expected.items( ):
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode )
                self.assertEqual( result[ 'endpoint' ], endpoint )

    def test_reject_invalid_date_and_endpoint_before_network( self ) -> None:
        """Validate endpoints and date pairs.

        Purpose:
            Prevent unsupported routes, malformed dates and reversed ranges.

        Returns:
            None: Assertions verify no network request is issued.
        """
        cases = [
            { 'endpoint': '../../bad', 'start_date': '', 'end_date': '' },
            { 'endpoint': 'CME', 'start_date': '2026-01-01', 'end_date': '' },
            { 'endpoint': 'CME', 'start_date': 'bad', 'end_date': '2026-01-02' },
            { 'endpoint': 'CME', 'start_date': '2026-01-03', 'end_date': '2026-01-02' }
        ]
        for values in cases:
            self.http.reset_mock( )
            with self.subTest( values=values ):
                with self.assertRaises( Exception ):
                    self.client.fetch_endpoint( **values )
                self.http.assert_not_called( )

    def test_cme_analysis_validates_numeric_limits( self ) -> None:
        """Reject invalid CMEAnalysis filters.

        Purpose:
            Enforce valid speed and half-angle bounds.

        Returns:
            None: Assertions verify early rejection.
        """
        for values in ( { 'speed': -1 }, { 'half_angle': 181 } ):
            self.http.reset_mock( )
            with self.subTest( values=values ):
                with self.assertRaises( Exception ):
                    self.client.fetch_endpoint( endpoint='CMEAnalysis',
                        start_date='', end_date='', **values )
                self.http.assert_not_called( )

    def test_api_key_not_returned_to_callers( self ) -> None:
        """Keep the NASA credential out of result metadata.

        Purpose:
            Retain authentication in HTTP transport but exclude it from results.

        Returns:
            None: Assertions verify secret isolation.
        """
        result = self.client.fetch( mode='notifications' )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'api_key' ],
            'private-test-key' )
        self.assertNotIn( 'api_key', result[ 'params' ] )
        self.assertEqual( result[ 'mode' ], 'notifications' )

    def test_reject_invalid_timeout_and_bad_payloads( self ) -> None:
        """Reject invalid timeouts and malformed API responses.

        Purpose:
            Prevent malformed responses from appearing successful.

        Returns:
            None: Assertions confirm errors.
        """
        with self.assertRaises( Exception ):
            self.client.fetch( mode='cme', time=0 )
        self.http.assert_not_called( )
        for payload in ( 42, 'invalid', { 'error': 'rejected' } ):
            self.respond( payload )
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch( mode='cme' )

    def test_demo_key_fallback( self ) -> None:
        """Support NASA's limited-use public demo credential.

        Purpose:
            Allow requests without a configured private API key.

        Returns:
            None: Assertions verify transport authentication.
        """
        self.client.api_key = ''
        result = self.client.fetch( mode='cme' )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'api_key' ],
            'DEMO_KEY' )
        self.assertNotIn( 'api_key', result[ 'params' ] )


if __name__ == '__main__':
    unittest.main( )
