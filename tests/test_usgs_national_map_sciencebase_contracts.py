"""Regression tests for USGS National Map and ScienceBase fetcher contracts.

Purpose:
    Verify production class behavior using isolated, mocked HTTP responses.
"""
from __future__ import annotations

import ast
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate required input.

    Purpose:
        Match the production guard for isolated class execution.

    Args:
        name (str): Input name.
        value (object): Input value.

    Returns:
        None: Raises ValueError for empty required input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Load actual provider implementations with mocked dependencies.

    Purpose:
        Compile only the two production fetchers without unrelated integrations.

    Returns:
        tuple[dict[str, type], Mock]: Provider classes and HTTP mock.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef ) and node.name in (
            'USGSTheNationalMap', 'USGSScienceBase' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal provider parent."""

    class Error( Exception ):
        """Application error wrapper."""

    class Logger:
        """No-op test logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a logged error.

            Purpose:
                Preserve the production exception pathway.

            Args:
                exception (Exception): Source exception.

            Returns:
                None: No output.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger,
        'throw_if': throw_if, 'math': math,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo contract tests' )
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in (
        'USGSTheNationalMap', 'USGSScienceBase' ) }, http


class NationalMapContractsTests( unittest.TestCase ):
    """Validate National Map dataset and product retrieval."""

    def setUp( self ) -> None:
        """Initialize a mocked provider.

        Purpose:
            Create independent response and request state.

        Returns:
            None: Stores test dependencies.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'USGSTheNationalMap' ]( )
        self.http.return_value.json.return_value = { 'items': [ ] }

    def test_dataset_and_product_modes( self ) -> None:
        """Retain both operating modes.

        Purpose:
            Protect the existing Foo response envelope.

        Returns:
            None: Assertions verify response fields.
        """
        for mode, result in (
            ( 'datasets', self.client.fetch_datasets( ) ),
            ( 'products', self.client.fetch_products( ) )
        ):
            with self.subTest( mode=mode ):
                self.assertEqual( result[ 'mode' ], mode )
                self.assertEqual( result[ 'rows' ], [ ] )
                self.assertEqual( result[ 'summary' ][ 'count' ], 0 )

    def test_reject_invalid_bbox_without_http( self ) -> None:
        """Reject nonfinite and out-of-range coordinates.

        Purpose:
            Block invalid spatial requests before network execution.

        Returns:
            None: Assertions verify exceptions and no requests.
        """
        for bbox in ( 'nan,0,10,10', '-181,0,10,10',
            '0,-91,10,10', '0,0,inf,10', '10,0,0,10' ):
            with self.subTest( bbox=bbox ):
                self.http.reset_mock( )
                with self.assertRaises( Exception ):
                    self.client.fetch_products( bbox=bbox )
                self.http.assert_not_called( )

    def test_invalid_payload_and_parameters( self ) -> None:
        """Reject provider errors and wrong argument types.

        Purpose:
            Prevent silent success on malformed provider results.

        Returns:
            None: Assertions verify expected failures.
        """
        with self.assertRaises( Exception ):
            self.client.request( 'datasets', [ 'wrong' ] )
        for payload in ( 'invalid', { 'error': 'failure' } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_datasets( )


class ScienceBaseContractsTests( unittest.TestCase ):
    """Validate ScienceBase search and item retrieval."""

    def setUp( self ) -> None:
        """Initialize mocked ScienceBase.

        Purpose:
            Isolate HTTP requests and response records.

        Returns:
            None: Stores the test instance.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'USGSScienceBase' ]( )
        self.http.return_value.json.return_value = { 'items': [ ] }

    def test_search_parameters_and_json_format( self ) -> None:
        """Retain valid search behavior.

        Purpose:
            Confirm documented pagination and JSON response selection.

        Returns:
            None: Assertions verify request parameters.
        """
        result = self.client.fetch_items( q='water', max_items=25, offset=5 )
        self.assertEqual( result[ 'mode' ], 'items' )
        params = self.http.call_args.kwargs[ 'params' ]
        self.assertEqual( params[ 'format' ], 'json' )
        self.assertEqual( params[ 'max' ], 25 )
        self.assertEqual( params[ 'offset' ], 5 )

    def test_reject_unaligned_pagination( self ) -> None:
        """Enforce five-record increments.

        Purpose:
            Prevent unsupported ScienceBase pagination requests.

        Returns:
            None: Assertions verify validation before HTTP.
        """
        for max_items, offset in ( ( 6, 0 ), ( 25, 3 ), ( 1, 0 ) ):
            with self.subTest( max_items=max_items, offset=offset ):
                self.http.reset_mock( )
                with self.assertRaises( Exception ):
                    self.client.fetch_items( max_items=max_items, offset=offset )
                self.http.assert_not_called( )

    def test_item_detail_mode( self ) -> None:
        """Preserve valid item-detail paths.

        Purpose:
            Protect item lookup and normalized rows.

        Returns:
            None: Assertions verify endpoint and response.
        """
        self.http.return_value.json.return_value = {
            'id': '58f7fadae4b0b7ea5451fc5c', 'title': 'Sample'
        }
        result = self.client.fetch_item( '58f7fadae4b0b7ea5451fc5c' )
        self.assertEqual( result[ 'mode' ], 'item' )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Title' ], 'Sample' )
        self.assertIn( '/item/58f7fadae4b0b7ea5451fc5c',
            self.http.call_args.kwargs[ 'url' ] )

    def test_reject_invalid_endpoint_params_payload( self ) -> None:
        """Validate endpoint paths and provider errors.

        Purpose:
            Reject unsafe paths, wrong arguments, and malformed JSON.

        Returns:
            None: Assertions verify failures.
        """
        for endpoint in ( 'item/', 'item/a/b', 'item/a?format=json' ):
            with self.subTest( endpoint=endpoint ):
                with self.assertRaises( Exception ):
                    self.client.validate_endpoint( endpoint )
        with self.assertRaises( Exception ):
            self.client.request( 'items', [ 'wrong' ] )
        for payload in ( [ ], { 'error': 'failure' }, { 'status': 400 } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_items( )


if __name__ == '__main__':
    unittest.main( )
