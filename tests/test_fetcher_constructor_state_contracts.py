"""Regression guards for constructor-established fetcher credentials.

Purpose:
    Prevent request paths from revalidating credential members assigned in constructors,
    while preserving required caller-input guards and provider-specific domain checks.
"""
from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'
CREDENTIALS = {
    'PurpleAir': ( 'api_key', ),
    'OpenAQ': ( 'api_key', ),
    'Firms': ( 'map_key', ),
    'GoogleSearch': ( 'api_key', 'cse_id' ),
    'GoogleMaps': ( 'api_key', ),
    'GoogleWeather': ( 'api_key', ),
    'GovData': ( 'api_key', ),
    'Congress': ( 'api_key', ),
    'GoogleGeocoding': ( 'api_key', ),
    'AirNow': ( 'api_key', ),
}
REMOVED_WRAPPERS = {
    'PurpleAir': 'validate_api_key',
    'OpenAQ': 'validate_api_key',
    'Firms': 'validate_map_key',
}
DOMAIN_VALIDATORS = {
    'PurpleAir': ( 'validate_mode', 'validate_endpoint', 'validate_bbox',
                   'validate_sensor_index', 'validate_location_type' ),
    'OpenAQ': ( 'validate_mode', 'validate_endpoint', 'validate_coordinates',
                'validate_radius', 'validate_positive_integer' ),
    'Firms': ( 'validate_mode', 'validate_source', 'validate_sensor',
               'validate_day_range', 'validate_date', 'validate_area_coordinates' ),
}


def get_classes( ) -> dict[str, ast.ClassDef]:
    """Return the source classes used by the regression contract.

    Purpose:
        Parse the current fetcher implementation and index its top-level classes.

    Returns:
        dict[str, ast.ClassDef]: Named fetcher class definitions.
    """
    source = SOURCE.read_text( encoding='utf-8' )
    module = ast.parse( source, filename=str( SOURCE ) )
    return { node.name: node for node in module.body if isinstance( node, ast.ClassDef ) }


def test_constructor_credentials_are_not_revalidated( ) -> None:
    """Reject request-time guards against constructor-established credentials.

    Purpose:
        Verify that required input guards remain distinct from stored credential state.

    Returns:
        None: Assertions report violations of the fetcher validation contract.
    """
    classes = get_classes( )

    for class_name, credentials in CREDENTIALS.items( ):
        cls = classes[ class_name ]
        constructor = next( node for node in cls.body
            if isinstance( node, ast.FunctionDef ) and node.name == '__init__' )
        assigned = { target.attr for node in ast.walk( constructor )
            if isinstance( node, (ast.Assign, ast.AnnAssign) )
            for target in (node.targets if isinstance( node, ast.Assign ) else [ node.target ])
            if isinstance( target, ast.Attribute ) and isinstance( target.value, ast.Name )
            and target.value.id == 'self' }
        assert set( credentials ).issubset( assigned ), class_name

        for method in cls.body:
            if not isinstance( method, ast.FunctionDef ) or method.name == '__init__':
                continue

            for node in ast.walk( method ):
                if not isinstance( node, ast.Call ):
                    continue
                if not isinstance( node.func, ast.Name ) or node.func.id != 'throw_if':
                    continue
                if len( node.args ) < 2:
                    continue
                value = node.args[ 1 ]
                if isinstance( value, ast.Attribute ) and isinstance( value.value, ast.Name ):
                    assert not ( value.value.id == 'self' and value.attr in credentials ), (
                        class_name, method.name, value.attr )


def test_redundant_credential_wrappers_are_removed( ) -> None:
    """Ensure the obsolete credential validation helpers have no callers.

    Purpose:
        Reject unused credential-specific wrappers and stale introspection entries.

    Returns:
        None: Assertions verify wrapper removal and updated public introspection.
    """
    classes = get_classes( )

    for class_name, wrapper in REMOVED_WRAPPERS.items( ):
        cls = classes[ class_name ]
        assert all( not isinstance( node, ast.FunctionDef ) or node.name != wrapper
            for node in cls.body ), class_name
        assert all( not (isinstance( node, ast.Call )
            and isinstance( node.func, ast.Attribute ) and node.func.attr == wrapper)
            for node in ast.walk( cls ) ), class_name
        assert all( not (isinstance( node, ast.Constant ) and node.value == wrapper)
            for node in ast.walk( cls ) ), class_name


def test_domain_validation_remains_available( ) -> None:
    """Ensure provider-domain validation is not lost during credential cleanup.

    Purpose:
        Protect API modes, coordinates, provider identifiers, and range restrictions.

    Returns:
        None: Assertions verify that required validator methods remain.
    """
    classes = get_classes( )

    for class_name, validators in DOMAIN_VALIDATORS.items( ):
        methods = { node.name for node in classes[ class_name ].body
            if isinstance( node, ast.FunctionDef ) }
        assert set( validators ).issubset( methods ), class_name
