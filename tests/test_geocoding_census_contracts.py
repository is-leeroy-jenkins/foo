"""Isolated regressions for GoogleGeocoding and CensusData provider contracts.

Purpose:
    Execute production class methods with mocked HTTP, verifying geocoding
    constraints, Census request construction and API-key confidentiality.
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
    """Guard required method arguments.

    Purpose:
        Reproduce Foo's required argument check in the isolated runtime.

    Args:
        name (str): Argument identifier.
        value (object): Supplied argument.

    Returns:
        None: Invalid required arguments raise ValueError.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Load production source classes without optional dependencies.

    Purpose:
        Compile source AST classes using deterministic HTTP and configuration mocks.

    Returns:
        tuple[dict[str, type], Mock]: Provider classes and HTTP transport mock.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in module.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'GoogleGeocoding', 'CensusData' ) ]
    for node in nodes:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    selected = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( selected )

    class Fetcher:
        """Isolated parent class."""

    class Error( Exception ):
        """Wrapped provider failure."""

    class Logger:
        """Disable application logging in test execution."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped exception.

            Purpose:
                Preserve the production exception pathway.

            Args:
                exception (Exception): Provider failure.

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
        'cfg': SimpleNamespace( AGENTS='Foo provider regression tests',
            GOOGLE_API_KEY='google-private-key',
            CENSUS_API_KEY='census-private-key' ),
        'math': math,
        're': re,
    }
    exec( compile( selected, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ]
        for name in ( 'GoogleGeocoding', 'CensusData' ) }, http


class GeocodingContractsTests( unittest.TestCase ):
    """Validate Google Geocoding's public operating modes and credentials."""

    def setUp( self ) -> None:
        """Initialize an isolated client and response.

        Purpose:
            Ensure transport calls cannot leak between tests.

        Returns:
            None: Saves mock provider state.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'GoogleGeocoding' ]( )
        self.http.return_value.json.return_value = {
            'status': 'OK', 'results': [ { 'place_id': 'test-place' } ] }

    def test_forward_reverse_and_place_modes_redact_key( self ) -> None:
        """Protect the credential across every geocoding mode.

        Purpose:
            Retain outbound authentication without exposing keys in result metadata.

        Returns:
            None: Assertions inspect provider results.
        """
        operations = [
            lambda: self.client.fetch_forward( 'Arlington, VA' ),
            lambda: self.client.fetch_reverse( 38.9, -77.0 ),
            lambda: self.client.fetch_place( 'test-place' )
        ]
        for operation in operations:
            with self.subTest( operation=operation ):
                result = operation( )
                self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'key' ],
                    'google-private-key' )
                self.assertNotIn( 'key', result[ 'params' ] )
                self.assertEqual( result[ 'status' ], 'OK' )

    def test_geocoding_rejects_invalid_coordinates_and_timeout( self ) -> None:
        """Reject invalid reverse-geocoding inputs.

        Purpose:
            Prevent invalid spatial coordinates and timeouts from reaching Google.

        Returns:
            None: Assertions check that no network calls occur.
        """
        cases = [
            lambda: self.client.fetch_reverse( 91.0, 0.0 ),
            lambda: self.client.fetch_reverse( 0.0, -181.0 ),
            lambda: self.client.fetch_reverse( float( 'nan' ), 0.0 ),
            lambda: self.client.fetch_reverse( 0.0, float( 'inf' ) ),
            lambda: self.client.fetch_forward( 'Arlington', time=0 ),
        ]
        for operation in cases:
            self.http.reset_mock( )
            with self.subTest( operation=operation ):
                with self.assertRaises( Exception ):
                    operation( )
                self.http.assert_not_called( )

    def test_geocoding_preserves_provider_errors( self ) -> None:
        """Raise provider errors rather than returning misleading results.

        Purpose:
            Preserve pre-existing Google status validation.

        Returns:
            None: Assertions verify wrapped failures.
        """
        self.http.return_value.json.return_value = {
            'status': 'REQUEST_DENIED', 'error_message': 'request invalid' }
        with self.assertRaises( Exception ):
            self.client.fetch_forward( 'Arlington' )

    def test_geocoding_rejects_bad_parameter_types( self ) -> None:
        """Reject a non-dictionary request mapping.

        Purpose:
            Preserve the required request parameter contract.

        Returns:
            None: Assertions verify early rejection.
        """
        with self.assertRaises( Exception ):
            self.client.request( params=[ 'not-a-dictionary' ] )
        self.http.assert_not_called( )


class CensusContractsTests( unittest.TestCase ):
    """Validate Census metadata/data fetcher operations."""

    def setUp( self ) -> None:
        """Initialize an isolated Census client and transport.

        Purpose:
            Supply a fresh response for each regression case.

        Returns:
            None: Client and HTTP mock are initialized.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'CensusData' ]( )
        self.http.return_value.url = 'https://api.census.gov/data/2022/acs/acs5?key=SHOULD_NOT_LEAK'
        self.http.return_value.json.return_value = {
            'variables': { 'NAME': { 'label': 'Area name' } } }

    def test_variables_metadata_no_key_needed( self ) -> None:
        """Use Census variable metadata without a credential.

        Purpose:
            Exclude a configured API key from metadata request and result.

        Returns:
            None: Assertions check payload and request metadata.
        """
        result = self.client.fetch_variables( year='2022', dataset='acs/acs5' )
        self.assertTrue( result[ 'url' ].endswith( '/variables.json' ) )
        self.assertNotIn( 'key', self.http.call_args.kwargs[ 'params' ] )
        self.assertNotIn( 'key', result[ 'params' ] )
        self.assertNotIn( 'SHOULD_NOT_LEAK', result[ 'url' ] )
        self.assertIn( 'variables', result[ 'data' ] )

    def test_census_data_preserves_query_and_shapes_table( self ) -> None:
        """Retain Census geography and row shaping contracts.

        Purpose:
            Validate outbound credential and returned headers/data shape.

        Returns:
            None: Assertions verify filters and field values.
        """
        self.http.return_value.json.return_value = [
            [ 'NAME', 'B01001_001E', 'state' ],
            [ 'Virginia', '8631393', '51' ]
        ]
        result = self.client.fetch_data( year='2022', dataset='acs/acs5',
            fields='NAME,B01001_001E', geography_for='state:*' )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'key' ],
            'census-private-key' )
        self.assertNotIn( 'key', result[ 'params' ] )
        self.assertNotIn( 'SHOULD_NOT_LEAK', result[ 'url' ] )
        self.assertEqual( result[ 'data' ][ 'count' ], 1 )
        self.assertEqual( result[ 'data' ][ 'rows' ][ 0 ][ 'state' ], '51' )

    def test_invalid_census_urls_and_timeouts_rejected( self ) -> None:
        """Reject malformed Census dataset identifiers and timeouts.

        Purpose:
            Avoid incorrect URL routes and invalid HTTP timeouts.

        Returns:
            None: Assertions verify failures occur before HTTP calls.
        """
        cases = [
            lambda: self.client.fetch_variables( 'bad', 'acs/acs5' ),
            lambda: self.client.fetch_variables( '2022', '../acs/acs5' ),
            lambda: self.client.fetch_variables( '2022', 'acs//acs5' ),
            lambda: self.client.fetch_variables( '2022', 'acs/acs5', time=0 ),
            lambda: self.client.fetch_data( '2022', 'acs/acs5',
                fields='NAME', geography_for='state:*', time=0 )
        ]
        for operation in cases:
            self.http.reset_mock( )
            with self.subTest( operation=operation ):
                with self.assertRaises( Exception ):
                    operation( )
                self.http.assert_not_called( )

    def test_invalid_census_payloads_rejected( self ) -> None:
        """Reject unsupported response types.

        Purpose:
            Prevent malformed provider JSON from masquerading as data or metadata.

        Returns:
            None: Assertions verify failure paths.
        """
        self.http.return_value.json.return_value = [ { 'error': 'bad' } ]
        with self.assertRaises( Exception ):
            self.client.fetch_variables( '2022', 'acs/acs5' )
        self.http.return_value.json.return_value = { 'error': 'bad query' }
        with self.assertRaises( Exception ):
            self.client.fetch_data( '2022', 'acs/acs5', 'NAME', 'state:*' )


if __name__ == '__main__':
    unittest.main( )
