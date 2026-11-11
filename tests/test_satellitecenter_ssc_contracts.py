"""Isolated regression tests for NASA SSC SatelliteCenter fetcher.

Purpose:
    Exercise the production class with mocked SSCWS responses and HTTP requests
    to verify location argument and provider response contracts.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate a mandatory argument.

    Purpose:
        Reproduce Foo's required input guard in the isolated source tests.

    Args:
        name (str): Input name.
        value (object): User-provided value.

    Returns:
        None: Raises ValueError when the required value is absent.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_satellite_center( ) -> tuple[object, Mock, Mock]:
    """Load the production SatelliteCenter class with mocked dependencies.

    Purpose:
        Avoid unrelated imports and network traffic while executing source code.

    Returns:
        tuple[object, Mock, Mock]: Provider instance, HTTP mock and SSCWS mock.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    node = next( copy.deepcopy( item ) for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == 'SatelliteCenter' )
    node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    selected = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        node
    ], type_ignores=[ ] )
    ast.fix_missing_locations( selected )

    class Fetcher:
        """Minimal isolated base class."""

    class Error( Exception ):
        """Source-compatible provider error."""

    class Logger:
        """Disable external error logging."""

        def write( self, exception: Exception ) -> None:
            """Accept a source error.

            Purpose:
                Preserve production logging behavior without side effects.

            Args:
                exception (Exception): Wrapped provider failure.

            Returns:
                None: No output.
            """
            pass

    http = Mock( )
    ssc = Mock( )
    ssc.return_value.get_observatories.return_value = {
        'HttpStatus': 200, 'Observatory': [ { 'Id': 'ace' } ] }
    ssc.return_value.get_ground_stations.return_value = {
        'HttpStatus': 200, 'GroundStation': [ { 'Id': 'station1' } ] }
    response = Mock( )
    response.json.return_value = { 'HttpStatus': 200, 'Data': [ 1 ] }
    http.return_value = response

    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'SscWs': ssc,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo SSC test' ),
        'dt': dt,
        're': re
    }
    exec( compile( selected, str( SOURCE ), 'exec' ), symbols )
    return symbols[ 'SatelliteCenter' ]( ), http, ssc


class SatelliteCenterContractsTests( unittest.TestCase ):
    """Verify SSC metadata and locations request contracts."""

    def setUp( self ) -> None:
        """Initialize isolated provider mocks.

        Purpose:
            Ensure tests do not share transport state.

        Returns:
            None: Stores fresh fixtures.
        """
        self.client, self.http, self.ssc = load_satellite_center( )

    def test_observatories_response_and_state( self ) -> None:
        """Accept and retain valid observatory metadata.

        Purpose:
            Verify SSCWS SDK results remain dictionaries.

        Returns:
            None: Assertions validate the response and state.
        """
        result = self.client.fetch_observatories( )
        self.assertEqual( result[ 'Observatory' ][ 0 ][ 'Id' ], 'ace' )
        self.assertEqual( self.client.observatories, result[ 'Observatory' ] )
        self.ssc.assert_called_once( )

    def test_ground_stations_response_and_state( self ) -> None:
        """Accept and retain valid ground-station metadata.

        Purpose:
            Verify SSCWS SDK result fields remain available.

        Returns:
            None: Assertions validate response shape.
        """
        result = self.client.fetch_ground_stations( )
        self.assertEqual( result[ 'GroundStation' ][ 0 ][ 'Id' ], 'station1' )
        self.assertEqual( self.client.ground_stations, result[ 'GroundStation' ] )

    def test_valid_locations_query( self ) -> None:
        """Preserve SSC URL, coordinate-system, and resolution parameters.

        Purpose:
            Verify the source method uses the validated request state.

        Returns:
            None: Assertions validate generated provider request and result.
        """
        result = self.client.fetch_locations(
            observatories='ace,wind',
            start_time='20260101T000000Z',
            end_time='20260102T000000Z',
            coordinate_systems='gse', resolution_factor=2, time=30 )
        self.assertEqual( result[ 'Data' ], [ 1 ] )
        self.assertIn( '/locations/ace,wind/', self.http.call_args.kwargs[ 'url' ] )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ],
            { 'resolutionFactor': 2 } )
        self.assertEqual( self.http.call_args.kwargs[ 'timeout' ], 30 )

    def test_reject_invalid_location_inputs_before_http( self ) -> None:
        """Reject invalid location request values locally.

        Purpose:
            Prevent malformed routes, timestamps and spatial coordinates.

        Returns:
            None: Assertions ensure HTTP is not called.
        """
        valid = {
            'observatories': 'ace',
            'start_time': '20260101T000000Z',
            'end_time': '20260102T000000Z'
        }
        cases = [
            { 'observatories': '../ace' },
            { 'coordinate_systems': 'invalid' },
            { 'start_time': '2026-01-01T00:00:00Z' },
            { 'start_time': '20260103T000000Z' },
            { 'resolution_factor': 0 },
            { 'time': 0 },
        ]
        for changed in cases:
            self.http.reset_mock( )
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.client.fetch_locations( **{ **valid, **changed } )
                self.http.assert_not_called( )

    def test_reject_invalid_json_response( self ) -> None:
        """Reject an invalid SSC location payload type.

        Purpose:
            Prevent successful HTTP status from masking an invalid JSON body.

        Returns:
            None: Assertions verify a wrapped error.
        """
        self.http.return_value.json.return_value = [ 1, 2, 3 ]
        with self.assertRaises( Exception ):
            self.client.fetch_locations( 'ace', '20260101T000000Z',
                '20260102T000000Z' )

    def test_reject_ssc_error_responses( self ) -> None:
        """Reject SSC-level non-success status results.

        Purpose:
            Validate errors returned by both SDK and direct location requests.

        Returns:
            None: Assertions verify provider errors are raised.
        """
        self.ssc.return_value.get_observatories.return_value = {
            'HttpStatus': 500, 'HttpText': 'request failed' }
        with self.assertRaises( Exception ):
            self.client.fetch_observatories( )

        self.ssc.return_value.get_ground_stations.return_value = {
            'HttpStatus': 404, 'ErrorMessage': 'missing stations' }
        with self.assertRaises( Exception ):
            self.client.fetch_ground_stations( )

        self.http.return_value.json.return_value = {
            'HttpStatus': 400, 'ErrorMessage': 'bad observation' }
        with self.assertRaises( Exception ):
            self.client.fetch_locations( 'ace', '20260101T000000Z',
                '20260102T000000Z' )

    def test_dispatch_preserves_all_modes( self ) -> None:
        """Protect the public dispatcher.

        Purpose:
            Ensure all three provider operations remain reachable.

        Returns:
            None: Assertions validate mode routing.
        """
        self.assertIn( 'Observatory', self.client.fetch( 'observatories' ) )
        self.assertIn( 'GroundStation', self.client.fetch( 'ground_stations' ) )
        result = self.client.fetch( mode='locations', query='ace',
            start_time='20260101T000000Z', end_time='20260102T000000Z' )
        self.assertEqual( result[ 'Data' ], [ 1 ] )


if __name__ == '__main__':
    unittest.main( )
