"""Isolated USGS earthquake and water-provider contract regressions.

Purpose:
    Execute the actual Foo fetcher classes with mocked HTTP transport to
    validate GeoJSON, OGC API response and geographic request contracts.
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
    """Validate a mandatory argument.

    Purpose:
        Provide Foo's required-input guard in isolated tests.

    Args:
        name (str): Argument name.
        value (object): Input value.

    Returns:
        None: Raises ValueError for missing input.
    """
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'{name} cannot be empty.' )


def load_fetchers( ) -> tuple[dict[str, type], Mock]:
    """Compile production classes without unrelated dependencies.

    Purpose:
        Test production behavior using deterministic HTTP responses.

    Returns:
        tuple[dict[str, type], Mock]: Classes and mocked requests.get.
    """
    tree = ast.parse( SOURCE.read_text( encoding='utf-8' ) )
    nodes = [ copy.deepcopy( node ) for node in tree.body
        if isinstance( node, ast.ClassDef )
        and node.name in ( 'USGSEarthquakes', 'USGSWaterData' ) ]
    for node in nodes:
        node.bases = [ ast.Name( id='Fetcher', ctx=ast.Load( ) ) ]
    module = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        *nodes
    ], type_ignores=[ ] )
    ast.fix_missing_locations( module )

    class Fetcher:
        """Minimal class parent."""

    class Error( Exception ):
        """Wrapped source failure."""

    class Logger:
        """Null error logger."""

        def write( self, exception: Exception ) -> None:
            """Accept a wrapped exception.

            Purpose:
                Preserve the source error pathway without logging side effects.

            Args:
                exception (Exception): Wrapped error.

            Returns:
                None: No result.
            """
            pass

    http = Mock( )
    symbols = {
        'Fetcher': Fetcher,
        'Error': Error,
        'Logger': Logger,
        'throw_if': throw_if,
        'cfg': SimpleNamespace( AGENTS='Foo USGS contract tests',
            USGS_WATERDATA_API_KEY='test-usgs-credential' ),
        'requests': SimpleNamespace( get=http ),
        'dt': dt,
        'math': math,
    }
    exec( compile( module, str( SOURCE ), 'exec' ), symbols )
    return { n: symbols[ n ] for n in
        ( 'USGSEarthquakes', 'USGSWaterData' ) }, http


class EarthquakeContractsTests( unittest.TestCase ):
    """Verify USGS seismic searches and feed parsing."""

    def setUp( self ) -> None:
        """Initialize a fake seismic service.

        Purpose:
            Isolate every test from HTTP calls and state leakage.

        Returns:
            None: Stores provider instance and mock.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'USGSEarthquakes' ]( )
        self.http.return_value.json.return_value = {
            'type': 'FeatureCollection',
            'metadata': { 'title': 'USGS test', 'count': 0 },
            'features': [ ]
        }

    def test_feed_result_keeps_standard_structure( self ) -> None:
        """Retain the feed result envelope.

        Purpose:
            Protect summary and raw result fields.

        Returns:
            None: Assertions inspect the returned dictionary.
        """
        result = self.client.fetch_feed( )
        self.assertEqual( result[ 'mode' ], 'feed' )
        self.assertEqual( result[ 'summary' ][ 'count' ], 0 )
        self.assertEqual( result[ 'rows' ], [ ] )

    def test_date_range_and_iso_dates( self ) -> None:
        """Validate chronological date windows.

        Purpose:
            Reject reversed and malformed dates before sending HTTP requests.

        Returns:
            None: Assertions verify request boundaries.
        """
        for start, end in (
            ( '2026-01-03', '2026-01-01' ),
            ( '2026-invalid', '2026-01-03' ),
            ( '2026-01-01T12:00:00Z', '2026-01-02T12:00:00' )
        ):
            self.http.reset_mock( )
            with self.subTest( start=start, end=end ):
                with self.assertRaises( Exception ):
                    self.client.fetch_search( start, end )
                self.http.assert_not_called( )
        result = self.client.fetch_search( '2026-01-01', '2026-01-03' )
        self.assertEqual( result[ 'params' ][ 'starttime' ], '2026-01-01' )

    def test_radius_requires_current_request_coordinates( self ) -> None:
        """Prevent stale spatial filters from authorizing a new radius.

        Purpose:
            Ensure all radial inputs belong to the same query.

        Returns:
            None: Assertions verify failure before network I/O.
        """
        self.client.fetch_search( '2026-01-01', '2026-01-02',
            latitude=38.9, longitude=-77.0 )
        self.http.reset_mock( )
        with self.assertRaises( Exception ):
            self.client.fetch_search( '2026-01-01', '2026-01-02',
                max_radius_km=100.0 )
        self.http.assert_not_called( )

    def test_reject_nonfinite_spatial_and_magnitude_values( self ) -> None:
        """Reject NaN and infinite query parameters.

        Purpose:
            Prevent unsupported numerical values reaching the USGS service.

        Returns:
            None: Assertions verify input validation.
        """
        cases = [
            { 'latitude': float( 'nan' ), 'longitude': -77.0 },
            { 'latitude': 38.9, 'longitude': float( 'inf' ) },
            { 'min_magnitude': float( 'nan' ) },
            { 'max_radius_km': float( 'inf' ), 'latitude': 38.9,
                'longitude': -77.0 }
        ]
        for extra in cases:
            self.http.reset_mock( )
            with self.subTest( extra=extra ):
                with self.assertRaises( Exception ):
                    self.client.fetch_search( '2026-01-01', '2026-01-02', **extra )
                self.http.assert_not_called( )

    def test_reject_invalid_seismic_geojson( self ) -> None:
        """Reject malformed or unsuccessful GeoJSON.

        Purpose:
            Avoid returning empty records for provider errors.

        Returns:
            None: Assertions verify error handling.
        """
        for payload in ( [ ], { 'error': 'bad request' },
            { 'metadata': { } } ):
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.fetch_feed( )


class WaterDataContractsTests( unittest.TestCase ):
    """Verify USGS Water Data OGC provider contracts."""

    def setUp( self ) -> None:
        """Initialize mock water service.

        Purpose:
            Isolate all requests from the live OGC API.

        Returns:
            None: Stores fake transport and provider.
        """
        classes, self.http = load_fetchers( )
        self.client = classes[ 'USGSWaterData' ]( )
        self.http.return_value.url = 'https://api.waterdata.usgs.gov/mock?key=secret'
        self.http.return_value.json.return_value = {
            'type': 'FeatureCollection', 'features': [ ] }

    def test_all_water_modes_keep_response_envelope( self ) -> None:
        """Preserve four Water Data dispatch modes.

        Purpose:
            Ensure normalized rows and summary remain available.

        Returns:
            None: Assertions inspect all supported results.
        """
        for mode in ( 'monitoring-locations', 'time-series-metadata',
            'latest-continuous', 'latest-daily' ):
            with self.subTest( mode=mode ):
                result = self.client.fetch( mode=mode,
                    monitoring_location_id='USGS-01646500' if mode != 'monitoring-locations' else '' )
                self.assertEqual( result[ 'mode' ], mode )
                self.assertIn( 'summary', result )
                self.assertEqual( result[ 'rows' ], [ ] )

    def test_water_request_retains_header_authentication( self ) -> None:
        """Authenticate without leaking the key in returned URLs.

        Purpose:
            Keep credentials in transport headers only.

        Returns:
            None: Assertions inspect transport and result URL.
        """
        result = self.client.request( 'monitoring-locations', { 'limit': 3 } )
        self.assertEqual( self.http.call_args.kwargs[ 'headers' ][ 'X-API-Key' ],
            'test-usgs-credential' )
        self.assertNotIn( 'key=secret', result[ 'url' ] )
        self.assertEqual( result[ 'params' ][ 'limit' ], 3 )

    def test_water_request_rejects_invalid_params_timeout( self ) -> None:
        """Reject malformed request inputs before network calls.

        Purpose:
            Prevent invalid HTTP parameters and timeout arguments.

        Returns:
            None: Assertions verify early rejection.
        """
        for values in ( { 'params': [ 1 ] }, { 'params': { }, 'time': 0 } ):
            self.http.reset_mock( )
            with self.subTest( values=values ):
                with self.assertRaises( Exception ):
                    self.client.request( 'monitoring-locations',
                        **{ 'params': { }, **values } )
                self.http.assert_not_called( )

    def test_water_provider_errors_rejected( self ) -> None:
        """Reject OGC error objects and non-object JSON.

        Purpose:
            Prevent unsuccessful provider calls from becoming empty water records.

        Returns:
            None: Assertions verify error pathways.
        """
        payloads = (
            [ 1, 2 ],
            { 'error': 'bad request' },
            { 'status': 400, 'detail': 'invalid filter' },
        )
        for payload in payloads:
            self.http.return_value.json.return_value = payload
            with self.subTest( payload=payload ):
                with self.assertRaises( Exception ):
                    self.client.request( 'monitoring-locations', { 'limit': 1 } )


if __name__ == '__main__':
    unittest.main( )
