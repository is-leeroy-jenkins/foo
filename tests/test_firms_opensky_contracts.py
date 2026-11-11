"""Isolated FIRMS and OpenSky API contract regressions.

Purpose:
    Test production classes with mocked HTTP without live provider requests.
"""
from __future__ import annotations

import ast
import copy
import csv
import datetime as dt
import io
import math
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate mandatory input.

    Purpose:
        Provide the same required-argument guard as Foo.

    Args:
        name (str): Argument name.
        value (object): Candidate argument.

    Returns:
        None: Raises for missing input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock, Mock]:
    """Compile actual provider classes with isolated HTTP calls.

    Purpose:
        Avoid unrelated dependencies and network access.

    Returns:
        tuple[dict[str, type], Mock, Mock]: Providers and GET/POST mocks.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef ) and node.name in ( 'Firms', 'OpenSky' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal provider base."""

    class Error( Exception ):
        """Foo error wrapper."""

    class Logger:
        """No-op log writer."""

        def write( self, exception: Exception ) -> None:
            """Accept an application error.

            Purpose:
                Preserve the production exception pathway.

            Args:
                exception (Exception): Source error.

            Returns:
                None: No result.
            """
            pass

    http_get = Mock( )
    http_post = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger, 'throw_if': throw_if,
        'requests': SimpleNamespace( get=http_get, post=http_post ),
        'cfg': SimpleNamespace( AGENTS='Foo contract tests',
            FIRMS_MAP_KEY='secret-map-key',
            OPENSKY_API_CLIENT_ID='', OPENSKY_API_CLIENT_SECRET='',
            OPENSKY_API_CREDENTIALS='' ),
        'math': math, 'dt': dt, 're': re, 'csv': csv, 'io': io,
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in ( 'Firms', 'OpenSky' ) }, http_get, http_post


class FirmsContractsTests( unittest.TestCase ):
    """Verify FIRMS area and data-availability modes."""

    def setUp( self ) -> None:
        """Create FIRMS fixtures.

        Purpose:
            Isolate API responses and client state.

        Returns:
            None: Stores test dependencies.
        """
        classes, self.http, self.post = load_fetchers( )
        self.client = classes[ 'Firms' ]( )
        self.http.return_value.text = 'latitude,longitude,acq_date\n38.9,-77.1,2026-11-11\n'

    def test_area_and_availability_without_leaking_key( self ) -> None:
        """Preserve both FIRMS retrieval modes.

        Purpose:
            Verify normalized rows and credential-free returned URLs.

        Returns:
            None: Assertions validate outputs.
        """
        result = self.client.fetch_area( )
        self.assertEqual( result[ 'mode' ], 'area' )
        self.assertEqual( result[ 'summary' ][ 'count' ], 1 )
        self.assertNotIn( 'secret-map-key', result[ 'url' ] )
        self.assertIn( 'secret-map-key', self.http.call_args.kwargs[ 'url' ] )
        self.http.return_value.text = 'source,start_date,end_date\nVIIRS,2026-11-10,2026-11-11\n'
        result = self.client.fetch_data_availability( )
        self.assertEqual( result[ 'mode' ], 'data-availability' )
        self.assertEqual( result[ 'summary' ][ 'count' ], 1 )
        self.assertNotIn( 'secret-map-key', result[ 'url' ] )

    def test_reject_nonfinite_coordinates_before_http( self ) -> None:
        """Reject NaN and infinite geographic bounds.

        Purpose:
            Fail before sending invalid spatial queries.

        Returns:
            None: Assertions verify validation.
        """
        for area in ( 'nan,0,20,40', '0,-inf,20,40',
            '0,0,inf,40' ):
            self.http.reset_mock( )
            with self.subTest( area=area ):
                with self.assertRaises( Exception ):
                    self.client.fetch_area( area_coordinates=area )
                self.http.assert_not_called( )

    def test_invalid_csv_and_missing_key( self ) -> None:
        """Reject malformed CSV and missing credentials.

        Purpose:
            Avoid reporting API errors as observations.

        Returns:
            None: Assertions verify exceptions.
        """
        for response in ( 'error: invalid key', '<html>Error</html>',
            'latitude,longitude\n1,2,3\n' ):
            self.http.return_value.text = response
            with self.subTest( response=response ):
                with self.assertRaises( Exception ):
                    self.client.fetch_area( )
        self.client.map_key = ''
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_area( )
        self.http.assert_not_called( )


class OpenSkyContractsTests( unittest.TestCase ):
    """Verify OpenSky state, flight and track contracts."""

    def setUp( self ) -> None:
        """Initialize mocked OpenSky.

        Purpose:
            Avoid network access and credentials in baseline tests.

        Returns:
            None: Stores client and HTTP transport.
        """
        classes, self.http, self.post = load_fetchers( )
        self.client = classes[ 'OpenSky' ]( )
        self.http.return_value.json.return_value = { 'time': 1, 'states': [ ] }
        self.post.return_value.json.return_value = { 'access_token': 'token-one' }

    def test_states_and_credential_transition( self ) -> None:
        """Do not reuse previous OAuth2 headers for anonymous calls.

        Purpose:
            Prevent stale bearer tokens crossing credential modes.

        Returns:
            None: Assertions verify response and headers.
        """
        result = self.client.fetch_states( )
        self.assertEqual( result[ 'count' ], 0 )
        self.client.client_id = 'client'
        self.client.client_secret = 'secret'
        self.client.fetch_states( client_id='client', client_secret='secret' )
        self.assertEqual( self.http.call_args.kwargs[ 'headers' ][ 'Authorization' ],
            'Bearer token-one' )
        self.client.client_id = ''
        self.client.client_secret = ''
        self.client.fetch_states( )
        self.assertNotIn( 'Authorization', self.http.call_args.kwargs[ 'headers' ] )

    def test_reject_nonfinite_bbox_and_invalid_epoch( self ) -> None:
        """Reject spatial and time argument errors.

        Purpose:
            Block invalid provider requests.

        Returns:
            None: Assertions verify pre-request failures.
        """
        for bbox in ( ( float( 'nan' ), -77, 40, -76 ),
            ( 38, float( 'inf' ), 40, -76 ) ):
            self.http.reset_mock( )
            with self.subTest( bbox=bbox ):
                with self.assertRaises( Exception ):
                    self.client.fetch_states( lamin=bbox[ 0 ], lomin=bbox[ 1 ],
                        lamax=bbox[ 2 ], lomax=bbox[ 3 ] )
                self.http.assert_not_called( )
        with self.assertRaises( Exception ):
            self.client.validate_epoch( 'begin', 10.5 )

    def test_flight_windows_and_response_shapes( self ) -> None:
        """Preserve flight results while rejecting unsupported ranges.

        Purpose:
            Validate provider response contracts and temporal bounds.

        Returns:
            None: Assertions verify behavior.
        """
        self.http.return_value.json.return_value = [ {
            'icao24': 'abc123', 'firstSeen': 1, 'lastSeen': 2
        } ]
        result = self.client.fetch_flights_aircraft( 'abc123', 100, 200 )
        self.assertEqual( result[ 'count' ], 1 )
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_flights_aircraft( 'abc123', 100, 200000 )
        self.http.assert_not_called( )
        self.http.return_value.json.return_value = { 'error': 'bad data' }
        with self.assertRaises( Exception ):
            self.client.fetch_flights_aircraft( 'abc123', 100, 200 )

    def test_track_and_oauth_error_validation( self ) -> None:
        """Reject malformed tracks and absent OAuth2 access tokens.

        Purpose:
            Avoid success on invalid provider responses.

        Returns:
            None: Assertions verify exceptions.
        """
        self.http.return_value.json.return_value = {
            'icao24': 'abc123', 'path': [ [ 100, 38.9, -77.1, 100, 10, False ] ]
        }
        result = self.client.fetch_track_aircraft( 'abc123' )
        self.assertEqual( result[ 'count' ], 1 )
        self.http.return_value.json.return_value = { 'path': { 'wrong': True } }
        with self.assertRaises( Exception ):
            self.client.fetch_track_aircraft( 'abc123' )
        self.post.return_value.json.return_value = { }
        with self.assertRaises( Exception ):
            self.client.fetch_states( client_id='client', client_secret='secret' )


if __name__ == '__main__':
    unittest.main( )
