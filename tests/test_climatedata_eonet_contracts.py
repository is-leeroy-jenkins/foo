"""Regression tests for NOAA ClimateData and NASA EONET wrappers.

Purpose:
    Exercise production provider contracts with deterministic mocked HTTP.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'


def throw_if( name: str, value: object ) -> None:
    """Validate required input.

    Purpose:
        Preserve Foo's argument-guard behavior in isolated tests.

    Args:
        name (str): Argument name.
        value (object): Argument value.

    Returns:
        None: Raises ValueError for missing input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Compile selected production classes with mock dependencies.

    Purpose:
        Avoid importing unrelated provider dependencies or making network requests.

    Returns:
        tuple[dict[str, type], Mock]: Provider classes and HTTP mock.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef ) and node.name in ( 'ClimateData', 'EoNet' ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal parent class."""

    class Error( Exception ):
        """Application error wrapper."""

    class Logger:
        """Test logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped error.

            Purpose:
                Preserve source error handling without writing logs.

            Args:
                exception (Exception): Wrapped exception.

            Returns:
                None: No result.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher, 'Error': Error, 'Logger': Logger,
        'throw_if': throw_if, 'requests': SimpleNamespace( get=http ),
        'cfg': SimpleNamespace( AGENTS='Foo regression tests' ),
        'dt': dt, 'math': math,
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { name: symbols[ name ] for name in ( 'ClimateData', 'EoNet' ) }, http


class ClimateDataContractsTests( unittest.TestCase ):
    """Verify NOAA dataset and observation requests."""

    def setUp( self ) -> None:
        """Initialize a mocked NOAA client.

        Purpose:
            Supply JSON response metadata for both API modes.

        Returns:
            None: Stores provider and HTTP mock.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'ClimateData' ]( )
        self.http.return_value.headers = { 'Content-Type': 'application/json' }
        self.http.return_value.json.return_value = { 'results': [ ] }

    def test_datasets_and_data_modes( self ) -> None:
        """Retain both NOAA operating modes.

        Purpose:
            Verify stable response envelopes and query parameters.

        Returns:
            None: Assertions validate results.
        """
        datasets = self.client.fetch_datasets( keyword='daily' )
        self.assertEqual( datasets[ 'mode' ], 'datasets' )
        self.assertEqual( datasets[ 'summary' ][ 'count' ], 0 )
        self.http.return_value.json.return_value = [ ]
        data = self.client.fetch_data( dataset='daily-summaries',
            start_date='2026-01-01', end_date='2026-01-02' )
        self.assertEqual( data[ 'mode' ], 'data' )
        self.assertEqual( data[ 'rows' ], [ ] )
        self.assertEqual( self.http.call_args.kwargs[ 'params' ][ 'format' ], 'json' )

    def test_reject_invalid_dates_before_http( self ) -> None:
        """Reject malformed and reversed climate dates.

        Purpose:
            Validate dates in both optional and mandatory input modes.

        Returns:
            None: Assertions verify no network call.
        """
        for dates in ( ( '2026-02-30', '2026-03-01' ),
            ( '2026-03-02', '2026-03-01' ),
            ( '2026-01-01T08:00:00Z', '2026-01-02' ) ):
            self.http.reset_mock( )
            with self.subTest( dates=dates ):
                with self.assertRaises( Exception ):
                    self.client.fetch_data( dataset='daily-summaries',
                        start_date=dates[ 0 ], end_date=dates[ 1 ] )
                self.http.assert_not_called( )
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_datasets( start_date='not-a-date' )
        self.http.assert_not_called( )

    def test_reject_invalid_noaa_payloads( self ) -> None:
        """Reject errors, non-JSON bodies and non-dict params.

        Purpose:
            Avoid silent success on malformed provider results.

        Returns:
            None: Assertions verify failures.
        """
        with self.assertRaises( Exception ):
            self.client.request( self.client.search_url, [ 'wrong' ] )
        for payload in ( { 'errorCode': 400 }, { 'errors': [ 'invalid' ] } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_datasets( )
        self.http.return_value.headers = { 'Content-Type': 'text/html' }
        self.http.return_value.text = '<html>Unavailable</html>'
        with self.assertRaises( Exception ):
            self.client.fetch_datasets( )


class EoNetContractsTests( unittest.TestCase ):
    """Verify EONET event and category workflows."""

    def setUp( self ) -> None:
        """Initialize EONET mock HTTP.

        Purpose:
            Separate each test's request and response state.

        Returns:
            None: Stores test client.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'EoNet' ]( )
        self.http.return_value.json.return_value = { 'events': [ ] }

    def test_events_and_categories_modes( self ) -> None:
        """Preserve both EONET API modes.

        Purpose:
            Protect result structures and active mode metadata.

        Returns:
            None: Assertions verify normalized outputs.
        """
        events = self.client.fetch_events( )
        self.assertEqual( events[ 'mode' ], 'events' )
        self.assertEqual( events[ 'summary' ][ 'count' ], 0 )
        self.http.return_value.json.return_value = { 'categories': [ ] }
        categories = self.client.fetch_categories( )
        self.assertEqual( categories[ 'mode' ], 'categories' )
        self.assertEqual( categories[ 'rows' ], [ ] )

    def test_reject_invalid_bbox_and_dates( self ) -> None:
        """Validate finite geographic and real calendar inputs.

        Purpose:
            Prevent malformed provider queries from reaching HTTP.

        Returns:
            None: Assertions verify early failure.
        """
        cases = (
            { 'bbox': 'nan,45,-70,20' },
            { 'bbox': '-100,inf,-70,20' },
            { 'bbox': '-100,45,-70,-91' },
            { 'start_date': '2026-02-30', 'end_date': '2026-03-01' },
            { 'start_date': '2026-03-02', 'end_date': '2026-03-01' },
        )
        for kwargs in cases:
            with self.subTest( kwargs=kwargs ):
                self.http.reset_mock( )
                with self.assertRaises( Exception ):
                    self.client.fetch_events( **kwargs )
                self.http.assert_not_called( )

    def test_reject_invalid_eonet_payloads( self ) -> None:
        """Reject malformed event and category collections.

        Purpose:
            Prevent empty or mis-shaped responses from appearing successful.

        Returns:
            None: Assertions verify provider error handling.
        """
        with self.assertRaises( Exception ):
            self.client.request( 'events', [ 'wrong' ] )
        for payload in ( [ ], { 'error': 'failure' },
            { 'events': { 'id': 'bad' } }, { 'events': [ None ] } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_events( )

    def test_geometry_with_missing_fields( self ) -> None:
        """Handle partial provider geometry records.

        Purpose:
            Preserve events while tolerating missing geometry details.

        Returns:
            None: Assertions verify normalized rows.
        """
        self.http.return_value.json.return_value = { 'events': [
            { 'id': 'EONET-test', 'title': 'Test Event',
                'geometry': [ None ] }
        ] }
        result = self.client.fetch_events( )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Geometry Count' ], 1 )
        self.assertIsNone( result[ 'rows' ][ 0 ][ 'Latitude' ] )


if __name__ == '__main__':
    unittest.main( )
