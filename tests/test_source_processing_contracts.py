"""Regression contracts for Foo source processing and provider alignment."""
from __future__ import annotations
import ast
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


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


class SessionState( dict ):
	"""Minimal attribute-access session state for isolated source lifecycle tests."""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__


def source_namespace( ) -> dict:
	"""Load the actual source-processing functions without starting Streamlit or providers."""
	tree = ast.parse( read( 'app.py' ), filename='app.py' )
	names = { 'normalize', 'build_source_documents', 'promote_source_result',
		'_promote_loader_documents', 'activate_processing_source', 'ensure_document_processing_state',
		'clear_document_processing_outputs', 'clear_if_active', 'document_signature',
		'rebuild_raw_text_from_documents', 'reset_document_processing_controls', }
	nodes = [ node for node in tree.body if
		( isinstance( node, ast.FunctionDef ) and node.name in names ) or
		( isinstance( node, ast.AnnAssign ) and isinstance( node.target, ast.Name ) and
			node.target.id == 'DOCUMENT_PROCESSING_STATE' ) ]
	assert { node.name for node in nodes if isinstance( node, ast.FunctionDef ) } == names

	def validate( name: str, value: Any ) -> None:
		"""Reject missing values in the isolated lifecycle test fixture."""
		if value is None or value == '':
			raise ValueError( name )

	namespace = { 'Any': Any, 'Dict': Dict, 'List': List, 'Document': SimpleNamespace,
		'json': json, 'sha256': sha256, 'throw_if': validate, 'Error': RuntimeError,
		'Logger': lambda: SimpleNamespace( write=lambda error: None ),
		'st': SimpleNamespace( session_state=SessionState( ) ), }
	exec( compile( ast.Module( body=nodes, type_ignores=[ ] ), 'app.py', 'exec' ), namespace )
	return namespace


def test_empty_source_records_and_metadata_only_results( ) -> None:
	"""Do not promote empty record collections; preserve standalone provider metadata."""
	ns = source_namespace( )
	build = ns[ 'build_source_documents' ]
	for result in ( None, { }, [ ], { 'items': [ ], 'count': 0 }, { 'rows': [ ] } ):
		assert build( 'Retrieval', 'ArXiv', result ) == [ ]
	assert len( build( 'Astronomical', 'Star Map', { 'interactive_url': 'https://example.org' } ) ) == 1
	assert len( build( 'Retrieval', 'ArXiv', { 'items': [ ], 'rows': [ { 'value': 1 } ] } ) ) == 1


def test_loader_result_text_and_metadata_are_preserved( ) -> None:
	"""Cached loader records retain original content and source metadata."""
	ns = source_namespace( )
	for mode, source, field in (
		( 'Retrieval', 'Jupyter Notebook', 'Content' ),
		( 'Retrieval', 'Google Cloud File', 'Content' ),
		( 'Retrieval', 'AWS S3 File', 'Content' ),
		( 'Retrieval', 'OneDrive', 'Content' ),
		( 'Retrieval', 'Google Speech-to-Text', 'Content' ),
		( 'Retrieval', 'AWS S3 Bucket', 'Content' ),
		( 'Retrieval', 'Google Cloud Bucket', 'Content' ),
		( 'Demographic', 'pub_med_search', 'Summary' ),
		( 'Demographic', 'open_city_data', 'Row' ),
	):
		result = { 'items': [ { field: 'Original document text',
			'Metadata': { 'source': 'original-source', 'page': 3 } } ] }
		docs = ns[ 'build_source_documents' ]( mode, source, result )
		assert len( docs ) == 1
		assert docs[ 0 ].page_content == 'Original document text'
		assert docs[ 0 ].metadata[ 'source' ] == 'original-source'
		assert docs[ 0 ].metadata[ 'page' ] == 3
		assert docs[ 0 ].metadata[ 'mode' ] == mode


def test_promotion_retains_unchanged_and_invalidates_changed_documents( ) -> None:
	"""Only changed source documents invalidate their derived processing results."""
	ns = source_namespace( )
	state = ns[ 'st' ].session_state
	promote = ns[ 'promote_source_result' ]
	result = { 'items': [ { 'Content': 'first', 'Metadata': { } } ] }
	promote( 'Retrieval', 'Jupyter Notebook', result )
	state.chunk_source_signature = ns[ 'document_signature' ]( state.documents )
	chunks = [ 'first' ]
	state.chunks = chunks
	state.tokens = [ 'first' ]
	state.embeddings = [ [ 1.0 ] ]
	state.processed_text = 'processed'
	promote( 'Retrieval', 'Jupyter Notebook', result )
	assert state.chunks is chunks
	assert state.processed_text == 'processed'
	assert state.embeddings == [ [ 1.0 ] ]
	promote( 'Retrieval', 'Jupyter Notebook', { 'items': [ { 'Content': 'changed',
		'Metadata': { } } ] } )
	assert state.raw_text == 'changed'
	assert state.chunks is None
	assert state.tokens is None
	assert state.embeddings is None


def test_source_switch_and_inactive_clear_preserve_owned_state( ) -> None:
	"""Restore matching saved processing state and isolate an inactive clear request."""
	ns = source_namespace( )
	state = ns[ 'st' ].session_state
	promote = ns[ 'promote_source_result' ]
	a = { 'items': [ { 'Content': 'alpha', 'Metadata': { } } ] }
	b = { 'items': [ { 'Content': 'beta', 'Metadata': { } } ] }
	promote( 'Retrieval', 'Jupyter Notebook', a )
	state.chunk_source_signature = ns[ 'document_signature' ]( state.documents )
	state.chunks = [ 'alpha' ]
	state.tokens = [ 'alpha' ]
	state.embeddings = [ [ 1.0 ] ]
	promote( 'Retrieval', 'OneDrive', b )
	assert state.chunks is None
	promote( 'Retrieval', 'Jupyter Notebook', a )
	assert state.chunks == [ 'alpha' ]
	assert state.tokens == [ 'alpha' ]
	assert state.embeddings == [ [ 1.0 ] ]
	ns[ 'reset_document_processing_controls' ]( 'excel' )
	ns[ 'clear_if_active' ]( 'ExcelLoader' )
	promote( 'Retrieval', 'OneDrive', { } )
	assert state.active_loader == 'Jupyter Notebook'
	assert state.raw_text == 'alpha'
	assert state.chunks == [ 'alpha' ]
	assert state.embeddings == [ [ 1.0 ] ]
	assert state.excel_processing_reset_request is True
	promote( 'Retrieval', 'Jupyter Notebook', { } )
	assert state.active_loader is None
	assert not state.documents
	assert state.raw_documents == [ ]
	assert state.raw_text == ''
	assert state.embeddings is None


def test_generation_provider_submit_contracts( ) -> None:
	"""All five provider submit handlers must invoke and promote their results."""
	tree = ast.parse( read( 'app.py' ), filename='app.py' )
	generation = next( node for node in ast.walk( tree ) if isinstance( node, ast.If )
		and isinstance( node.test, ast.Compare ) and isinstance( node.test.left, ast.Name )
		and node.test.left.id == 'mode' and any( isinstance( item, ast.Constant )
			and item.value == 'Generation' for item in node.test.comparators ) )
	for flag, provider, method in ( ( 'chat_submit', 'ChatGPT', '_invoke_provider' ),
		( 'groq_submit', 'Grok', '_invoke_provider' ), ( 'claude_submit', 'Claude', 'generate_text' ),
		( 'gemini_submit', 'Gemini', '_invoke_provider' ), ( 'mistral_submit', 'Mistral', 'fetch' ) ):
		handler = next( node for node in ast.walk( generation ) if isinstance( node, ast.If )
			and isinstance( node.test, ast.Name ) and node.test.id == flag )
		calls = [ node for node in ast.walk( handler ) if isinstance( node, ast.Call ) ]
		assert any( ( isinstance( call.func, ast.Name ) and call.func.id == method ) or
			( isinstance( call.func, ast.Attribute ) and call.func.attr == method ) for call in calls )
		assert any( isinstance( call.func, ast.Name ) and call.func.id == '_promote_generation_result'
			and any( item.arg == 'provider' and isinstance( item.value, ast.Constant )
				and item.value.value == provider for item in call.keywords ) for call in calls )


def test_generation_restores_owned_results_and_preserves_other_sources( ) -> None:
	"""Generation promotion and clearing must respect the shared source lifecycle."""
	ns = source_namespace( )
	state = ns[ 'st' ].session_state
	state.generation_documents = [ ]
	state.generation_raw_result = None
	state.generation_active_source = ''
	warnings = [ ]
	ns[ 'st' ].warning = warnings.append
	tree = ast.parse( read( 'app.py' ), filename='app.py' )
	names = { '_promote_generation_result', '_clear_generation_result' }
	nodes = [ node for node in ast.walk( tree ) if isinstance( node, ast.FunctionDef )
		and node.name in names ]
	assert len( nodes ) == 2
	exec( compile( ast.Module( body=nodes, type_ignores=[ ] ), 'app.py', 'exec' ), ns )
	promote = ns[ '_promote_generation_result' ]
	clear = ns[ '_clear_generation_result' ]
	promote( 'Claude', 'Generated text' )
	assert state.active_loader == 'Claude'
	assert state.processing_active_source == 'Claude'
	assert state.raw_text == 'Generated text'
	assert state.generation_raw_result == 'Generated text'
	state.chunk_source_signature = ns[ 'document_signature' ]( state.documents )
	state.chunks = [ 'Generated text' ]
	promote( 'Claude', 'Generated text' )
	assert state.chunks == [ 'Generated text' ]
	promote( 'Claude', '   ' )
	assert warnings and state.raw_text == 'Generated text'
	ns[ 'promote_source_result' ]( 'Retrieval', 'ArXiv', [ 'Research text' ] )
	clear( 'Claude' )
	assert state.active_loader == 'ArXiv'
	assert state.raw_text == 'Research text'
	assert state.generation_active_source == ''
	assert state.generation_documents == [ ]


def test_generation_right_column_uses_shared_processing( ) -> None:
	"""Generation must render its result and shared document-processing tabs."""
	source = read( 'app.py' ).split( "elif mode == 'Generation':", 1 )[ 1 ]
	assert '\n\twith right:' in source
	assert "_render_output( st, st.session_state[ 'generation_raw_result' ] )" in source
	assert "render_source_processing_controls( generation_source, 'generation_processing' )" in source
	assert "render_document_processing_tabs( key_prefix='generation' )" in source


def test_xml_load_actions_guard_missing_upload_and_promote_owned_documents( ) -> None:
	"""Missing XML uploads cannot promote documents from a previous loader."""
	tree = ast.parse( read( 'app.py' ), filename='app.py' )
	for label in ( 'Load XML (Semantic)', 'Load XML Tree' ):
		action = next( node for node in ast.walk( tree ) if isinstance( node, ast.If )
			and isinstance( node.test, ast.Call )
			and isinstance( node.test.func, ast.Attribute ) and node.test.func.attr == 'button'
			and node.test.args and isinstance( node.test.args[ 0 ], ast.Constant )
			and node.test.args[ 0 ].value == label )
		assert len( action.body ) == 1
		guard = action.body[ 0 ]
		assert isinstance( guard, ast.If )
		assert ast.unparse( guard.test ) == 'xml_file is None'
		warnings = [ ]
		ns = source_namespace( )
		ns[ 'st' ].warning = warnings.append
		ns[ 'xml_file' ] = None
		state = ns[ 'st' ].session_state
		ns[ 'promote_source_result' ]( 'Loading', 'PdfLoader', [ 'Existing PDF' ] )
		exec( compile( ast.Module( body=action.body, type_ignores=[ ] ), 'app.py', 'exec' ), ns )
		assert warnings
		assert state.active_loader == 'PdfLoader'
		assert state.raw_text == 'Existing PDF'
		assert any( isinstance( call, ast.Call ) and isinstance( call.func, ast.Name )
			and call.func.id == '_promote_loader_documents'
			and isinstance( call.args[ 1 ], ast.Constant ) and call.args[ 1 ].value == 'XmlLoader'
			for statement in guard.orelse for call in ast.walk( statement ) )


def test_structured_xml_document_promotion_synchronizes_state( ) -> None:
	"""Structured XML replaces the canonical document collection and raw text."""
	ns = source_namespace( )
	state = ns[ 'st' ].session_state
	ns[ 'promote_source_result' ]( 'Loading', 'PdfLoader', [ 'Existing PDF' ] )
	xml = ns[ 'Document' ]( page_content='<root><item>value</item></root>',
		metadata={ 'loader': 'XmlLoader', 'source': 'example.xml', 'load_mode': 'structured_xml' } )
	ns[ '_promote_loader_documents' ]( [ xml ], 'XmlLoader' )
	assert state.active_loader == 'XmlLoader'
	assert state.processing_active_source == 'XmlLoader'
	assert state.documents == [ xml ]
	assert state.raw_documents == [ xml ]
	assert state.raw_text == xml.page_content


def test_xml_actions_and_chunking_handle_wrapped_errors( ) -> None:
	"""All XML action boundaries and shared chunking display wrapped errors locally."""
	tree = ast.parse( read( 'app.py' ), filename='app.py' )
	for label in ( 'Load XML (Semantic)', 'Split Semantic Documents', 'Load XML Tree',
		'Run XPath Query' ):
		action = next( node for node in ast.walk( tree ) if isinstance( node, ast.If )
			and isinstance( node.test, ast.Call )
			and isinstance( node.test.func, ast.Attribute ) and node.test.func.attr == 'button'
			and node.test.args and isinstance( node.test.args[ 0 ], ast.Constant )
			and node.test.args[ 0 ].value == label )
		assert any( isinstance( node, ast.ExceptHandler ) and isinstance( node.type, ast.Name )
			and node.type.id == 'Error' and any( isinstance( call, ast.Call )
				and isinstance( call.func, ast.Attribute ) and call.func.attr == 'error'
				for call in ast.walk( node ) ) for node in ast.walk( action ) )
	chunk = next( node for node in ast.walk( tree ) if isinstance( node, ast.If )
		and isinstance( node.test, ast.Name ) and node.test.id == 'chunk_clicked' )
	assert any( isinstance( node, ast.ExceptHandler ) and isinstance( node.type, ast.Name )
		and node.type.id == 'Error' and not any( isinstance( item, ast.Raise )
			for item in ast.walk( node ) ) for node in ast.walk( chunk ) )


def test_processing_widget_overlap_bounds( ) -> None:
	"""The overlap slider must retain a valid range at the minimum chunk size."""
	source = read( 'app.py' )
	tree = ast.parse( source, filename='app.py' )
	function = next( node for node in tree.body if isinstance( node, ast.FunctionDef )
		and node.name == 'render_document_processing_inputs' )
	slider = next( node for node in ast.walk( function ) if isinstance( node, ast.Call )
		and isinstance( node.func, ast.Attribute ) and node.func.attr == 'slider'
		and node.args and isinstance( node.args[ 0 ], ast.Constant )
		and node.args[ 0 ].value == 'Chunk Overlap' )
	keywords = { item.arg: item.value for item in slider.keywords }
	for size in ( 1, 2, 1000, 5000 ):
		namespace = { 'chunk_size': size, 'max_overlap': size - 1 }
		lower = eval( compile( ast.Expression( body=keywords[ 'min_value' ] ),
			'app.py', 'eval' ), namespace )
		upper = eval( compile( ast.Expression( body=keywords[ 'max_value' ] ),
			'app.py', 'eval' ), namespace )
		disabled = eval( compile( ast.Expression( body=keywords[ 'disabled' ] ),
			'app.py', 'eval' ), namespace )
		assert lower < upper
		assert disabled == ( size == 1 )
		assert upper == max( 1, size - 1 )
	assert 'st.session_state[ overlap_key ] = max_overlap' in ast.unparse( function ).replace(
		'st.session_state[overlap_key]', 'st.session_state[ overlap_key ]' )


def test_incompatible_numeric_widget_keys_are_isolated( ) -> None:
	"""Loading and Demographic numeric limits must not share incompatible keys."""
	source = read( 'app.py' )
	loading = source.split( "if mode == 'Loading':", 1 )[ 1 ].split(
		"elif mode == 'Scraping':", 1 )[ 0 ]
	demographic = source.split( "elif mode == 'Demographic':", 1 )[ 1 ].split(
		"elif mode == 'Generation':", 1 )[ 0 ]
	for key in ( 'pubmed_max_docs', 'open_city_limit' ):
		assert f"key='loading_{key}'" in loading
		assert f"key='{key}'" not in loading
		assert f"key='{key}'" in demographic


def test_claude_budget_is_adjusted_before_widget_creation( ) -> None:
	"""A reduced maximum token count cannot leave an out-of-range widget value."""
	source = read( 'app.py' )
	tree = ast.parse( source, filename='app.py' )
	widget = next( node for node in ast.walk( tree ) if isinstance( node, ast.Call )
		and isinstance( node.func, ast.Attribute ) and node.func.attr == 'number_input'
		and node.args and isinstance( node.args[ 0 ], ast.Constant )
		and node.args[ 0 ].value == 'Thinking Budget' )
	keywords = { item.arg: item.value for item in widget.keywords }
	assert 'value' not in keywords
	assert ast.unparse( keywords[ 'max_value' ] ) == 'max_budget'
	assert ast.unparse( keywords[ 'disabled' ] ) == 'claude_max_tokens <= 1024'
	assert source.index( "st.session_state[ 'claude_thinking_budget_chat' ] = max_budget" ) < source.index(
		"claude_thinking_budget = st.number_input( 'Thinking Budget'" )


if __name__ == '__main__':
	tests = [ value for name, value in globals( ).items( )
		if name.startswith( 'test_' ) and callable( value ) ]
	for test in tests:
		test( )
		print( f'PASS {test.__name__}' )
	print( f'{len( tests )} passed' )
