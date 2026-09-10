"""Regression contracts for Foo source processing and provider alignment."""
from __future__ import annotations
import ast
from pathlib import Path


ROOT = Path( __file__ ).resolve( ).parents[ 1 ]


def read( name: str ) -> str:
	"""Return repository text for a contract assertion."""
	return ( ROOT / name ).read_text( encoding='utf-8' )


def test_python_sources_parse( ) -> None:
	"""Every corrected Python source must parse."""
	for name in ( 'app.py', 'config.py', 'embedders.py', 'fetchers.py', 'stores/vector.py' ):
		ast.parse( read( name ), filename=name )


def test_opensky_ui_passes_end_keyword( ) -> None:
	"""OpenSky UI must use the fetcher's public end argument."""
	source = read( 'app.py' )
	assert 'end=int( end ) if int( end or 0 ) > 0 else None' in source
	assert '\n\t\t\t\t\t\tnd=' not in source


def test_chunking_precedes_word_tokenization( ) -> None:
	"""Recursive splitting must occur before per-chunk word tokenization."""
	source = read( 'app.py' )
	split_position = source.index( 'chunks = splitter.split_documents( source_documents )' )
	token_position = source.index( "tokens = word_tokenize( chunk.page_content or '' )" )
	assert split_position < token_position
	assert "'Token Count': metadata.get( 'token_count', 0 )" in source
	assert "'Tokens': metadata.get( 'tokens', [ ] )" in source


def test_source_controls_and_tabs_cover_data_modes( ) -> None:
	"""All requested modes must expose source processing and right-side tabs."""
	source = read( 'app.py' )
	assert source.count( 'render_source_processing_controls(' ) >= 74
	for prefix in ( 'web_scraper', 'retrieval', 'geospatial', 'environmental', 'astronomical',
			'demographic' ):
		assert f"render_document_processing_tabs( key_prefix='{prefix}' )" in source


def test_runtime_credentials_update_configuration( ) -> None:
	"""Sidebar credentials must update both environment and live configuration."""
	source = read( 'app.py' )
	assert 'os.environ[ attr ] = val' in source
	assert 'setattr( cfg, attr, val )' in source
	config = read( 'config.py' )
	assert "os.getenv( 'USGS_WATERDATA_API_KEY' ) or os.getenv( 'USGS_API_KEY' )" in config


def test_google_and_opensky_provider_contracts( ) -> None:
	"""Provider adapters must validate Google status and use the named OpenSky secret."""
	source = read( 'fetchers.py' )
	assert "status and status != 'OK'" in source
	assert "Google Geocoding failed: {status}{detail}" in source
	assert 'cfg.OPENSKY_API_CLIENT_SECRET or cfg.OPENSKY_API_CREDENTIALS' in source


def test_vector_storage_is_non_destructive_and_reuses_embeddings( ) -> None:
	"""Ordinary writes must preserve collections and reuse displayed vectors."""
	app = read( 'app.py' )
	vector = read( 'stores/vector.py' )
	assert 'store.add_embeddings( documents=list( chunked_documents ), embeddings=embeddings )' in app
	assert vector.count( 'reset_collection( )' ) == 1
	clear_position = vector.index( 'def clear( self )' )
	reset_position = vector.index( 'reset_collection( )' )
	assert clear_position < reset_position
	for method in ( 'connect', 'add_documents', 'add_embeddings', 'similarity_search',
			'similarity_search_with_score', 'delete', 'clear', 'count', 'health', 'as_retriever' ):
		assert f'def {method}(' in vector


def test_optional_vector_and_embedding_providers_are_lazy_loaded( ) -> None:
	"""Unused optional providers must not be application-startup imports."""
	embedders = read( 'embedders.py' )
	vector = read( 'stores/vector.py' )
	for import_line in ( 'from langchain_openai import OpenAIEmbeddings',
			'from langchain_google_genai import GoogleGenerativeAIEmbeddings',
			'from langchain_chroma import Chroma', 'from pinecone import Pinecone' ):
		locations = [ line for line in ( embedders + vector ).splitlines( ) if import_line in line ]
		assert locations
		assert all( line.startswith( ( '            ', '\t\t\t' ) ) for line in locations )


if __name__ == '__main__':
	tests = [ value for name, value in globals( ).items( )
		if name.startswith( 'test_' ) and callable( value ) ]
	for test in tests:
		test( )
		print( f'PASS {test.__name__}' )
	print( f'{len( tests )} passed' )
