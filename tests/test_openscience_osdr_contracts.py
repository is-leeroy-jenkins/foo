"""Isolated functional regression tests for the NASA OSDR OpenScience fetcher.

Purpose:
    Exercise the existing provider class against mocked HTTP responses, ensuring
    query fields and filters follow NASA's Biological Data API contract.
"""
from __future__ import annotations

import ast
import copy
import urllib.parse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Guard required inputs.

    Purpose:
        Implement Foo's mandatory argument presence contract in isolation.

    Args:
        name (str): Input name.
        value (object): Candidate input value.

    Returns:
        None: Raises ValueError when an input is missing.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_open_science( ) -> tuple[object, Mock]:
    """Compile the source OpenScience class with mocked dependencies.

    Purpose:
        Test actual source behavior without optional runtime packages or network.

    Returns:
        tuple[object, Mock]: Provider instance and mocked HTTP GET function.
    """
    source = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    selected = next( copy.deepcopy( node ) for node in source.body
        if isinstance( node, ast.ClassDef ) and node.name == 'OpenScience' )
    selected.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        selected
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Isolated fetcher parent."""

    class Error( Exception ):
        """Source error wrapper."""

    class Logger:
        """Null application logger."""

        def write( self, exception: Exception ) -> None:
            """Accept wrapped failures.

            Purpose:
                Preserve the production exception-handling path.

            Args:
                exception (Exception): Wrapped source failure.

            Returns:
                None: No logging side effects are required.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'cfg': SimpleNamespace( AGENTS='Foo OSDR regression tests' ),
        'requests': SimpleNamespace( get=http ),
        'urllib': SimpleNamespace( parse=urllib.parse ),
        're': __import__( 're' ),
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return symbols[ 'OpenScience' ]( ), http


class OpenScienceContractTests( unittest.TestCase ):
    """Validate OSDR dataset and field-query execution."""

    def setUp( self ) -> None:
        """Create isolated provider state.

        Purpose:
            Reset HTTP responses and API configuration between tests.

        Returns:
            None: Instance members hold mock dependencies.
        """
        self.client, self.http = load_open_science( )
        self.respond( { 'results': [ { 'id': 'OSD-48' } ] } )

    def respond( self, payload: object, content_type: str='application/json' ) -> None:
        """Configure the API response.

        Purpose:
            Simulate provider HTTP success with JSON or text responses.

        Args:
            payload (object): Payload returned by response.json().
            content_type (str): Content-Type header.

        Returns:
            None: Configures the mocked transport.
        """
        response = Mock( )
        response.headers = { 'Content-Type': content_type }
        response.json.return_value = payload
        response.text = 'a,b\\n1,2'
        self.http.return_value = response

    def test_dataset_endpoint_and_response_shape( self ) -> None:
        """Retain dataset URL and output contract.

        Purpose:
            Verify accession-based REST requests use the validated URL.

        Returns:
            None: Assertions inspect URL and result.
        """
        result = self.client.fetch_dataset( accession='OSD-48' )
        self.assertEqual( result[ 'mode' ], 'dataset' )
        self.assertTrue( result[ 'url' ].endswith( '/v2/dataset/OSD-48/' ) )
        self.assertEqual( result[ 'data' ][ 'results' ][ 0 ][ 'id' ], 'OSD-48' )

    def test_metadata_query_filters_are_individual_params( self ) -> None:
        """Preserve NASA filter keys instead of nesting a query string.

        Purpose:
            Ensure caller-supplied metadata selectors become HTTP fields.

        Returns:
            None: Assertions inspect the request and result state.
        """
        result = self.client.fetch_metadata(
            query='id.accession=OSD-48&study.characteristics.organism=Mus%20musculus',
            format_value='json' )
        self.assertEqual( result[ 'mode' ], 'metadata' )
        params = self.http.call_args.kwargs[ 'params' ]
        self.assertEqual( params[ 'id.accession' ], 'OSD-48' )
        self.assertEqual( params[ 'study.characteristics.organism' ], 'Mus musculus' )
        self.assertEqual( params[ 'format' ], 'json' )
        self.assertNotIn( 'query', params )

    def test_assays_and_data_queries_preserve_fields( self ) -> None:
        """Preserve OSDR assay and data selectors.

        Purpose:
            Verify two additional query modes use field-specific URL parameters.

        Returns:
            None: Assertions inspect provider dispatch.
        """
        for mode, method in (
            ( 'assays', self.client.fetch_assays ),
            ( 'data', self.client.fetch_data )
        ):
            with self.subTest( mode=mode ):
                result = method( query='id.accession=OSD-48&file.data%20type=counts' )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'params' ][ 'id.accession' ], 'OSD-48' )
                self.assertEqual( result[ 'params' ][ 'file.data type' ], 'counts' )
                self.assertNotIn( 'query', result[ 'params' ] )

    def test_query_dispatch_retains_all_four_modes( self ) -> None:
        """Verify the established public dispatcher.

        Purpose:
            Protect dataset, metadata, assay and data routing.

        Returns:
            None: Assertions verify all supported mode names.
        """
        for mode in ( 'dataset', 'metadata', 'assays', 'data' ):
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode, accession='OSD-48',
                    query='id.accession=OSD-48' )
                self.assertEqual( result[ 'mode' ], mode )

    def test_invalid_accession_rejected_before_network( self ) -> None:
        """Reject malformed REST route identifiers.

        Purpose:
            Prevent unsupported accession text from entering dataset URLs.

        Returns:
            None: Assertions verify early validation.
        """
        with self.assertRaises( Exception ):
            self.client.fetch_dataset( accession='../OSD-48' )
        self.http.assert_not_called( )

    def test_bad_timeout_and_provider_errors_rejected( self ) -> None:
        """Reject invalid HTTP request conditions.

        Purpose:
            Enforce a positive timeout and detect NASA error payloads.

        Returns:
            None: Assertions verify both failure paths.
        """
        with self.assertRaises( Exception ):
            self.client.fetch_dataset( accession='OSD-48', time=0 )
        self.http.assert_not_called( )
        self.respond( { 'error': 'invalid query' } )
        with self.assertRaises( Exception ):
            self.client.fetch_metadata( query='id.accession=OSD-48' )

    def test_csv_response_stays_text( self ) -> None:
        """Preserve non-JSON API output.

        Purpose:
            Verify tabular CSV responses can be returned without JSON parsing.

        Returns:
            None: Assertions validate result data type.
        """
        self.respond( None, content_type='text/csv' )
        self.http.return_value.json.side_effect = ValueError( 'not JSON' )
        result = self.client.fetch_assays(
            query='id.accession=OSD-48', format_value='csv' )
        self.assertEqual( result[ 'data' ], 'a,b\\n1,2' )


if __name__ == '__main__':
    unittest.main( )
