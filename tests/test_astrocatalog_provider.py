"""Offline integration tests for the AstroCatalog SIMBAD and VizieR replacement.

Purpose:
    Verify real AstroCatalog method bodies with fake astronomy provider clients,
    avoiding network access and optional science-package installation.
"""
from __future__ import annotations

import ast
import copy
import io
import re
import unittest
import urllib
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


class FakeTable:
    """Minimal Astropy table substitute for deterministic provider tests."""

    def __init__( self ) -> None:
        """Initialize the table with one row.

        Purpose:
            Supply predictable provider content and column labels.

        Returns:
            None: Stores the test row.
        """
        self.colnames = [ 'MAIN_ID', 'TYPE' ]
        self.rows = [ { 'MAIN_ID': 'M31', 'TYPE': 'Galaxy' } ]

    def __iter__( self ):
        """Yield table records.

        Purpose:
            Match the iteration behavior used by the table serializer.

        Returns:
            Iterator: Table row dictionaries.
        """
        return iter( self.rows )

    def write( self, output: io.StringIO, format: str, overwrite: bool ) -> None:
        """Write a deterministic serialized table.

        Purpose:
            Mock output conversion for CSV and TSV requests.

        Args:
            output (io.StringIO): Writer receiving data.
            format (str): Astroquery output format.
            overwrite (bool): Allow replacing existing output.

        Returns:
            None: Writes the test string.
        """
        output.write( 'MAIN_ID,TYPE\\nM31,Galaxy\\n' )


def throw_if( name: str, value: object ) -> None:
    """Enforce required-input presence in the isolated implementation.

    Purpose:
        Match the source presence guard without loading Foo dependencies.

    Args:
        name (str): Required argument name.
        value (object): Candidate value.

    Returns:
        None: Invalid values raise ValueError.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} is required' )


def load_catalog( ):
    """Compile the real AstroCatalog class with mocked provider dependencies.

    Purpose:
        Exercise source methods rather than a separate reimplementation.

    Returns:
        type: The compiled provider class.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    catalog = next( node for node in module.body
        if isinstance( node, ast.ClassDef ) and node.name == 'AstroCatalog' )
    catalog = copy.deepcopy( catalog )
    catalog.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    code = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        catalog
    ], type_ignores=[ ] )
    ast.fix_missing_locations( code )

    class Fetcher:
        """Isolated base class."""

    class Error( Exception ):
        """Expected provider exception wrapper."""

    class Logger:
        """No-I/O exception logger."""

        def write( self, exception: Exception ) -> None:
            """Accept an exception.

            Purpose:
                Match the production logger's interface.

            Args:
                exception (Exception): Exception being recorded.

            Returns:
                None: No external logging occurs.
            """
            pass

    table = FakeTable( )
    simbad = Mock( )
    simbad.query_object.return_value = table
    simbad.query_region.return_value = table
    vizier = Mock( )
    vizier.query_object.return_value = { 'J/TEST': table }
    vizier.query_region.return_value = { 'J/TEST': table }
    numpy = SimpleNamespace( ma=SimpleNamespace( is_masked=lambda value: False ),
        generic=type( 'NumpyGeneric', ( ), { } ) )
    names = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'cfg': SimpleNamespace( AGENTS='test-agent' ),
        'throw_if': throw_if,
        'Simbad': Mock( return_value=simbad ),
        'Vizier': Mock( return_value=vizier ),
        'SkyCoord': lambda *values, **options: values,
        'u': SimpleNamespace( deg=1, arcmin=1 ),
        'np': numpy,
        'urllib': urllib,
        're': re,
        'io': io
    }
    exec( compile( code, str( SOURCE ), 'exec' ), names )
    return names[ 'AstroCatalog' ], simbad, vizier


class AstroCatalogProviderTests( unittest.TestCase ):
    """Exercise the public astronomy catalog workflows without HTTP calls."""

    def setUp( self ) -> None:
        """Prepare isolated provider clients.

        Purpose:
            Create clean provider state for every test.

        Returns:
            None: Initializes the class under test.
        """
        provider, self.simbad, self.vizier = load_catalog( )
        self.catalog = provider( )

    def test_simbad_object_and_instance_state( self ) -> None:
        """Resolve a named object with the SIMBAD client.

        Purpose:
            Verify results are serializable and request state is retained.

        Returns:
            None: Assertions verify SIMBAD delegation and returned rows.
        """
        result = self.catalog.fetch( mode='object_query', query='M31' )
        self.simbad.query_object.assert_called_once_with( 'M31' )
        self.assertEqual( result[ 'provider' ], 'simbad' )
        self.assertEqual( result[ 'tables' ][ 0 ][ 'rows' ][ 0 ][ 'MAIN_ID' ], 'M31' )
        self.assertEqual( self.catalog.name, 'M31' )
        self.assertEqual( result[ 'count' ], 1 )

    def test_vizier_cone_search_and_csv( self ) -> None:
        """Query a VizieR catalog with stored coordinates.

        Purpose:
            Preserve provider selection and structured CSV-compatible results.

        Returns:
            None: Assertions verify VizieR query parameters and returned output.
        """
        result = self.catalog.fetch( mode='cone_search', ra='10', dec='20',
            radius=2, arguments='provider=vizier,catalog=J/TEST',
            data_format='csv' )
        self.assertEqual( result[ 'provider' ], 'vizier' )
        self.assertEqual( result[ 'catalog' ], 'J/TEST' )
        self.assertEqual( result[ 'count' ], 1 )
        self.assertIn( 'formatted_tables', result )
        self.vizier.query_region.assert_called_once( )
        self.assertEqual( self.catalog.right_ascension, '10' )
        self.assertEqual( self.catalog.declination, '20' )

    def test_invalid_inputs_are_rejected_without_provider_calls( self ) -> None:
        """Reject missing queries, invalid formats, and unsupported legacy paths.

        Purpose:
            Prevent provider traffic after argument validation failures.

        Returns:
            None: Assertions ensure rejected requests invoke no astronomy clients.
        """
        invalid = [
            lambda: self.catalog.fetch( query='' ),
            lambda: self.catalog.fetch( mode='cone_search', ra='', dec='20' ),
            lambda: self.catalog.fetch_object( name='M31', quantity='legacy' ),
            lambda: self.catalog.fetch_object( name='M31', data_format='xml' ),
            lambda: self.catalog.request( route='', params={ } ),
            lambda: self.catalog.request( route='M31',
                params={ 'provider': 'vizier' } ),
            lambda: self.catalog.request( route='M31', time=0 ),
        ]
        for operation in invalid:
            with self.subTest( operation=operation ):
                with self.assertRaises( Exception ):
                    operation( )
        self.simbad.query_object.assert_not_called( )
        self.vizier.query_object.assert_not_called( )

    def test_required_route_guard_and_stored_delegation( self ) -> None:
        """Enforce mandatory route validation and instance-member forwarding.

        Purpose:
            Check the source structure of the corrected public request interface.

        Returns:
            None: Assertions verify the established source contract.
        """
        module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
        cls = next( node for node in module.body
            if isinstance( node, ast.ClassDef ) and node.name == 'AstroCatalog' )
        methods = { node.name: node for node in cls.body
            if isinstance( node, ast.FunctionDef ) }
        request = methods[ 'request' ]
        calls = [ node for node in ast.walk( request )
            if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name )
            and node.func.id == 'throw_if' ]
        self.assertTrue( any( isinstance( node.args[ 0 ], ast.Constant )
            and node.args[ 0 ].value == 'route' for node in calls ) )
        for name in ( 'fetch_object', 'cone_search' ):
            method = methods[ name ]
            request_calls = [ node for node in ast.walk( method )
                if isinstance( node, ast.Call ) and isinstance( node.func, ast.Attribute )
                and node.func.attr == 'request' ]
            self.assertEqual( len( request_calls ), 1 )
            arguments = request_calls[ 0 ].keywords
            self.assertTrue( all( isinstance( kw.value, ast.Attribute )
                and isinstance( kw.value.value, ast.Name )
                and kw.value.value.id == 'self' for kw in arguments ) )


if __name__ == '__main__':
    unittest.main( )
