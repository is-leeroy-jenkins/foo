"""Mocked regression tests for Foo PurpleAir request and validation workflows.

Purpose:
    Exercise successful sensor retrieval and invalid caller inputs without making
    network requests or requiring a live PurpleAir API credential.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from boogr import Error
from fetchers import PurpleAir


class PurpleAirValidationTests( unittest.TestCase ):
    """Verify the PurpleAir public fetch paths and inline input constraints."""

    def setUp( self ) -> None:
        """Initialize isolated provider state.

        Purpose:
            Provide a controlled credential and header for mocked request tests.

        Returns:
            None: Prepares the fetcher for each test.
        """
        self.air = PurpleAir( )
        self.air.api_key = 'test-key'
        self.air.headers[ 'X-API-Key' ] = self.air.api_key

    def test_fetch_sensors( self ) -> None:
        """Preserve geographic query parameters and normalized sensor rows.

        Purpose:
            Verify the list-sensor response and constructor credential reuse.

        Returns:
            None: Assertions validate the mocked HTTP request and response.
        """
        response = Mock( )
        response.url = 'https://api.purpleair.com/v1/sensors'
        response.json.return_value = {
            'fields': [ 'sensor_index', 'name', 'pm2.5' ],
            'data': [ [ 123, 'Sensor One', 7.5 ] ]
        }

        with patch( 'fetchers.requests.get', return_value=response ) as request:
            result = self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38 )

        self.assertEqual( result[ 'mode' ], 'sensors' )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Sensor Index' ], 123 )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'PM2.5' ], 7.5 )
        self.assertEqual( request.call_args.kwargs[ 'headers' ][ 'X-API-Key' ],
            'test-key' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'nwlng' ], -78.0 )

    def test_fetch_single_sensor( self ) -> None:
        """Preserve single-sensor mode dispatch and response shaping.

        Purpose:
            Verify the public fetch dispatcher and positive sensor identifier.

        Returns:
            None: Assertions validate mocked provider output.
        """
        response = Mock( )
        response.url = 'https://api.purpleair.com/v1/sensors/123'
        response.json.return_value = {
            'sensor': { 'sensor_index': 123, 'name': 'Sensor One', 'model': 'PA-II' }
        }

        with patch( 'fetchers.requests.get', return_value=response ) as request:
            result = self.air.fetch( mode='sensor', sensor_index=123 )

        self.assertEqual( result[ 'rows' ][ 0 ][ 'Sensor Index' ], 123 )
        self.assertEqual( result[ 'rows' ][ 0 ][ 'Model' ], 'PA-II' )
        self.assertTrue( request.call_args.kwargs[ 'url' ].endswith( '/sensors/123' ) )

    def test_invalid_arguments_do_not_send_requests( self ) -> None:
        """Reject malformed provider arguments before an HTTP call.

        Purpose:
            Protect required fields, bounding-box limits, optional ranges,
            endpoint paths, sensor identifiers, and request timeouts.

        Returns:
            None: Assertions verify wrapped validation failures and zero HTTP calls.
        """
        operations = [
            lambda: self.air.fetch_sensor( sensor_index=None ),
            lambda: self.air.fetch_sensor( sensor_index=0 ),
            lambda: self.air.fetch_sensors( nwlng=None, nwlat=39, selng=-77, selat=38 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, location_type=2 ),
            lambda: self.air.fetch_sensors( nwlng=-77, nwlat=39,
                selng=-78, selat=38 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=38,
                selng=-77, selat=39 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, max_age=-1 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, modified_since=-1 ),
            lambda: self.air.request( endpoint='invalid' ),
            lambda: self.air.request( endpoint='sensors/0' ),
            lambda: self.air.request( endpoint='sensors', time=0 ),
            lambda: self.air.fetch( mode='unsupported' )
        ]

        with patch( 'fetchers.requests.get' ) as request:
            for operation in operations:
                with self.subTest( operation=operation ):
                    with self.assertRaises( Error ):
                        operation( )
            request.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
