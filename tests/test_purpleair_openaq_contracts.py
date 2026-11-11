"""Isolated PurpleAir and OpenAQ provider contract regressions.

Purpose:
    Exercise source implementations with mocked HTTP and no external API calls.
"""
from __future__ import annotations

import ast
import copy
import math
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Reject missing mandatory arguments.

    Purpose:
        Supply the Foo-style guard for isolated class execution.

    Args:
        name (str): Argument name.
        value (object): Candidate value.

    Returns:
        None: Raises ValueError for absent inputs.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Compile the actual fetchers using mocked dependencies.

    Purpose:
        Avoid importing unrelated Foo integrations during focused tests.

    Returns:
        tuple[dict[str, type], Mock]: Two production classes and HTTP mock.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body if
        isinstance( node, ast.ClassDef ) and node.name in ( 'PurpleAir', 'OpenAQ' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal fetcher superclass."""

    class Error( Exception ):
        """Wrapped Foo error."""

    class Logger:
        """No-op logger."""

        def write( self, exception: Exception ) -> None:
            """Accept application errors without disk output.

            Purpose:
                Exercise the production exception path.

            Args:
                exception (Exception): Wrapped error.

            Returns:
                None: No result.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger, 'throw_if': throw_if,
        'math': math, 're': re, 'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo API contract tests',
            PURPLEAIR_API_KEY='purple-test-key', OPENAQ_API_KEY='openaq-test-key' )
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in ( 'PurpleAir', 'OpenAQ' ) }, http


class PurpleAirContractsTests( unittest.TestCase ):
    """Verify PurpleAir sensor listing and detail mode."""

    def setUp( self ) -> None:
        """Initialize mocked PurpleAir.

        Purpose:
            Isolate each test from HTTP and previous instance state.

        Returns:
            None: Stores client and mock transport.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'PurpleAir' ]( )
        self.http.return_value.json.return_value = {
            'fields': [ 'sensor_index', 'name', 'pm2.5' ],
            'data': [ [ 101, 'Alpha', 6.2 ] ]
        }

    def test_sensor_listing_and_detail( self ) -> None:
        """Preserve both existing PurpleAir modes.

        Purpose:
            Validate row shaping and authenticated requests.

        Returns:
            None: Assertions verify response contracts.
        """
        rows = self.client.fetch_sensors( -78.0, 39.0, -76.0, 38.0 )
        self.assertEqual( rows[ 'mode' ], 'sensors' )
        self.assertEqual( rows[ 'rows' ][ 0 ][ 'Name' ], 'Alpha' )
        self.assertEqual( self.http.call_args.kwargs[ 'headers' ][ 'X-API-Key' ],
            'purple-test-key' )
        self.http.return_value.json.return_value = {
            'sensor': { 'sensor_index': 101, 'name': 'Alpha' }
        }
        detail = self.client.fetch_sensor( 101 )
        self.assertEqual( detail[ 'mode' ], 'sensor' )
        self.assertEqual( detail[ 'rows' ][ 0 ][ 'Sensor Index' ], 101 )

    def test_reject_nonfinite_bbox_without_http( self ) -> None:
        """Reject NaN and infinity coordinates.

        Purpose:
            Ensure malformed bounding boxes fail before network access.

        Returns:
            None: Assertions verify no provider calls.
        """
        for bbox in ( ( float( 'nan' ), 39, -76, 38 ),
            ( -78, float( 'inf' ), -76, 38 ),
            ( -78, 39, float( '-inf' ), 38 ) ):
            self.http.reset_mock( )
            with self.subTest( bbox=bbox ):
                with self.assertRaises( Exception ):
                    self.client.fetch_sensors( *bbox )
                self.http.assert_not_called( )

    def test_reject_malformed_json_params_and_credential( self ) -> None:
        """Reject invalid provider results and request inputs.

        Purpose:
            Fail explicitly instead of returning fabricated sensor rows.

        Returns:
            None: Assertions verify errors.
        """
        with self.assertRaises( Exception ):
            self.client.request( 'sensors', [ 'not a mapping' ] )
        for payload in ( { 'error': 'denied' }, { 'fields': [ 'name' ] },
            { 'fields': [ 'name' ], 'data': [ None ] } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_sensors( -78, 39, -76, 38 )
        self.client.api_key = ''
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_sensor( 101 )
        self.http.assert_not_called( )

    def test_nonfinite_pm25_ignored_in_summary( self ) -> None:
        """Exclude NaN and infinite PM2.5 values.

        Purpose:
            Retain finite measurements in result summaries.

        Returns:
            None: Assertions verify the maximum.
        """
        self.http.return_value.json.return_value = {
            'fields': [ 'sensor_index', 'name', 'pm2.5' ],
            'data': [ [ 101, 'Alpha', float( 'nan' ) ],
                [ 102, 'Beta', 8.0 ] ]
        }
        result = self.client.fetch_sensors( -78, 39, -76, 38 )
        self.assertEqual( result[ 'summary' ][ 'max_pm25' ], 8.0 )


class OpenAQContractsTests( unittest.TestCase ):
    """Verify OpenAQ v3 resource and measurement contracts."""

    def setUp( self ) -> None:
        """Initialize mock OpenAQ.

        Purpose:
            Give every test a fresh client and response.

        Returns:
            None: Stores client and HTTP mock.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'OpenAQ' ]( )
        self.http.return_value.json.return_value = { 'results': [ ] }

    def test_all_six_modes( self ) -> None:
        """Preserve each existing OpenAQ mode.

        Purpose:
            Ensure endpoint dispatch and stable normalized output.

        Returns:
            None: Assertions validate result envelopes.
        """
        cases = (
            ( 'countries', { } ), ( 'providers', { } ), ( 'parameters', { } ),
            ( 'parameter_latest', { 'parameter_id': 2 } ),
            ( 'locations', { } ), ( 'latest', { 'location_id': 2178 } )
        )
        for mode, options in cases:
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode, **options )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'rows' ], [ ] )
                self.assertIn( '/v3/', self.http.call_args.kwargs[ 'url' ] )

    def test_open_aq_latest_v3_identifiers( self ) -> None:
        """Retain sensor and location IDs in the newest measurements.

        Purpose:
            Shape documented v3 result keys and parameter units.

        Returns:
            None: Assertions inspect output fields.
        """
        self.http.return_value.json.return_value = { 'results': [ {
            'locationsId': 2178, 'sensorsId': 991,
            'value': 5.2, 'parameter': { 'id': 2,
                'name': 'pm25', 'units': 'µg/m³' }
        } ] }
        result = self.client.fetch_latest( 2178 )
        row = result[ 'rows' ][ 0 ]
        self.assertEqual( row[ 'Location Id' ], 2178 )
        self.assertEqual( row[ 'Sensor Id' ], 991 )
        self.assertEqual( row[ 'Unit' ], 'µg/m³' )

    def test_invalid_coordinates_and_payloads( self ) -> None:
        """Reject malformed geographic filters and response collections.

        Purpose:
            Prevent incorrect results being interpreted as empty datasets.

        Returns:
            None: Assertions verify early failures and provider errors.
        """
        for coordinate in ( 'nan,-77', '38,inf', '91,-77' ):
            self.http.reset_mock( )
            with self.subTest( coordinate=coordinate ):
                with self.assertRaises( Exception ):
                    self.client.fetch_locations( coordinates=coordinate )
                self.http.assert_not_called( )
        with self.assertRaises( Exception ):
            self.client.request( 'locations', [ 'invalid' ] )
        for payload in ( [ ], { 'error': 'denied' },
            { 'results': { 'id': 2178 } }, { 'results': [ None ] } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_locations( )

    def test_missing_api_key_rejected_before_http( self ) -> None:
        """Require credentials before issuing provider requests.

        Purpose:
            Avoid sending unauthenticated requests.

        Returns:
            None: Assertions verify no network traffic.
        """
        self.client.api_key = ''
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_providers( )
        self.http.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
