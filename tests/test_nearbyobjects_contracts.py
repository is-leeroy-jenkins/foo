"""Isolated regression tests for the JPL NearbyObjects fetcher.

Purpose:
    Verify provider payload handling and NASA JPL request-option contracts
    without network access or optional application dependencies.
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
    """Validate mandatory input arguments.

    Purpose:
        Reproduce the required argument guard for isolated tests.

    Args:
        name (str): Argument name.
        value (object): Supplied argument.

    Returns:
        None: Invalid inputs raise ValueError.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_nearby_objects( ) -> tuple[object, Mock]:
    """Load the production class into a mocked provider environment.

    Purpose:
        Compile the source implementation without requiring unrelated packages.

    Returns:
        tuple[object, Mock]: Provider instance and fake HTTP get callable.
    """
    module = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    node = next( copy.deepcopy( item ) for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == 'NearbyObjects' )
    node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    selected = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        node
    ], type_ignores=[ ] )
    ast.fix_missing_locations( selected )

    class Fetcher:
        """Mock provider base class."""

    class Error( Exception ):
        """Wrapped provider error."""

    class Logger:
        """Mock application logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a provider exception.

            Purpose:
                Preserve the source logging path without external effects.

            Args:
                exception (Exception): Wrapped failure.

            Returns:
                None: No external effects.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo test client' ),
        'dt': dt,
        're': re
    }
    exec( compile( selected, str( SOURCE ), 'exec' ), symbols )
    return symbols[ 'NearbyObjects' ]( ), http


class NearbyObjectsContractsTests( unittest.TestCase ):
    """Verify the supported NASA JPL near-Earth-object modes."""

    def setUp( self ) -> None:
        """Create a new mocked provider client.

        Purpose:
            Isolate HTTP state between test methods.

        Returns:
            None: Initializes the provider and fake transport.
        """
        self.client, self.http = load_nearby_objects( )
        self.respond( { 'count': 1, 'fields': [ 'date' ], 'data': [ [ '2026-01-01' ] ] } )

    def respond( self, payload: object ) -> None:
        """Configure the mocked HTTP response.

        Purpose:
            Supply a JSON response payload to production request code.

        Args:
            payload (object): JSON document to return.

        Returns:
            None: Updates the mocked transport.
        """
        response = Mock( )
        response.json.return_value = payload
        self.http.return_value = response

    def test_cad_valid_response_and_state( self ) -> None:
        """Preserve CAD response fields and checked request state.

        Purpose:
            Verify URL, parameters, and timeout use the stored attributes.

        Returns:
            None: Assertions verify request details and returned payload.
        """
        result = self.client.fetch_close_approaches(
            start_date='2026-01-01', end_date='2026-01-02', limit=4, time=15 )
        self.assertEqual( result[ 'count' ], 1 )
        self.assertEqual( result[ 'params' ][ 'limit' ], 4 )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ], self.client.params )
        self.assertEqual( self.http.call_args.kwargs[ 'timeout' ], self.client.timeout )
        self.assertTrue( self.client.url.endswith( '/cad.api' ) )

    def test_cad_invalid_options_rejected_before_http( self ) -> None:
        """Reject invalid CAD filters locally.

        Purpose:
            Protect date ordering, distance units, sorting, and positive limits.

        Returns:
            None: Assertions verify all invalid input paths.
        """
        valid = { 'start_date': '2026-01-01', 'end_date': '2026-01-02' }
        cases = [
            { 'start_date': '2026-01-03' },
            { 'start_date': 'not-a-date' },
            { 'limit': 0 },
            { 'sort': 'arbitrary' },
            { 'dist_max': '10parsecs' }
        ]
        for changed in cases:
            self.http.reset_mock( )
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.client.fetch_close_approaches( **{ **valid, **changed } )
                self.http.assert_not_called( )

    def test_nhats_rejects_unsupported_constraints( self ) -> None:
        """Reject invalid mission filters without an HTTP request.

        Purpose:
            Cover delta-V, duration, stay, launch, magnitude, and opportunities.

        Returns:
            None: Assertions verify invalid filter handling.
        """
        cases = [
            { 'dv': 3.0 },
            { 'dur': 61 },
            { 'stay': 9 },
            { 'launch': '2045-2050' },
            { 'h': 31 },
            { 'occ': 9 }
        ]
        for changed in cases:
            self.http.reset_mock( )
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.client.fetch_nhats_summary( **changed )
                self.http.assert_not_called( )

    def test_nhats_object_and_fireball_modes( self ) -> None:
        """Preserve valid object and fireball responses.

        Purpose:
            Check request dispatch and returned fields for two NASA endpoints.

        Returns:
            None: Assertions validate mode-specific output.
        """
        object_result = self.client.fetch_nhats_object( designation='99942' )
        self.assertEqual( object_result[ 'params' ][ 'des' ], '99942' )
        self.assertTrue( self.client.url.endswith( '/nhats.api' ) )
        fireball_result = self.client.fetch_fireballs(
            date_min='2026-01-01', limit=2 )
        self.assertEqual( fireball_result[ 'params' ][ 'date-min' ], '2026-01-01' )
        self.assertEqual( fireball_result[ 'count' ], 1 )
        self.assertTrue( self.client.url.endswith( '/fireball.api' ) )

    def test_invalid_fireball_inputs_rejected( self ) -> None:
        """Reject invalid fireball query values.

        Purpose:
            Prevent invalid dates and zero limits from reaching NASA.

        Returns:
            None: Assertions verify early rejection.
        """
        for changed in ( { 'limit': 0 }, { 'date_min': 'not-a-date' } ):
            self.http.reset_mock( )
            with self.subTest( changed=changed ):
                with self.assertRaises( Exception ):
                    self.client.fetch_fireballs( **changed )
                self.http.assert_not_called( )

    def test_api_errors_and_invalid_json_rejected( self ) -> None:
        """Reject NASA-level errors even with successful HTTP status.

        Purpose:
            Prevent malformed JSON or provider error documents from masquerading
            as successful astronomical results.

        Returns:
            None: Assertions validate response rejection.
        """
        for payload in ( [ ], { 'error': 'NASA rejected query' } ):
            self.respond( payload )
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_fireballs( )

    def test_nonpositive_timeout_rejected_before_http( self ) -> None:
        """Reject invalid provider request timeouts.

        Purpose:
            Validate the shared request path before a network call.

        Returns:
            None: Assertions verify timeout behavior.
        """
        with self.assertRaises( Exception ):
            self.client.fetch_fireballs( time=0 )
        self.http.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
