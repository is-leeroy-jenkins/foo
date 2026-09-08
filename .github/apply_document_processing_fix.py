'''Apply the approved document chunking, embedding, and vector-storage integration to app.py.

Purpose:
    Adds Streamlit orchestration for the nine existing chunkable loaders, removes the obsolete
    Loading-mode NLP metrics path, and preserves the existing document-loading contracts.
'''
from __future__ import annotations
from pathlib import Path
import re


APP_PATH = Path( 'app.py' )
TARGET_LOADERS = [
    ( 'Text Loader', 'TextLoader', 'txt' ),
    ( 'CSV Loader', 'CsvLoader', 'csv' ),
    ( 'PDF Loader', 'PdfLoader', 'pdf' ),
    ( 'Excel Loader', 'ExcelLoader', 'excel' ),
    ( 'Word Document Loader', 'WordLoader', 'word' ),
    ( 'Markdown Loader', 'MarkdownLoader', 'md' ),
    ( 'HTML Loader', 'HtmlLoader', 'html' ),
    ( 'JSON Loader', 'JsonLoader', 'json' ),
    ( 'Power Point Loader', 'PowerPointLoader', 'pptx' ),
]

HELPERS = r'''
# =====================================================================
# DOCUMENT CHUNKING / EMBEDDING / VECTOR STORAGE
# =====================================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_PROVIDER = 'Hugging Face'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHROMA_DIRECTORY = Path( 'stores' ) / 'chroma'
EMBEDDING_MODELS: Dict[ str, List[ str ] ] = {
	'OpenAI': [ 'text-embedding-3-small', 'text-embedding-3-large' ],
	'Google Generative AI': [ 'gemini-embedding-2-preview' ],
	'Mistral AI': [ 'mistral-embed' ],
	'Hugging Face': [
		'sentence-transformers/all-MiniLM-L6-v2',
		'sentence-transformers/all-mpnet-base-v2',
	],
}
VECTOR_STORES = [ 'Chroma', 'Pinecone' ]

DOCUMENT_PROCESSING_STATE: Dict[ str, Any ] = {
	'chunked_documents': None,
	'chunk_source_signature': '',
	'chunk_size_used': 0,
	'chunk_overlap_used': 0,
	'df_chunking': None,
	'embedder': None,
	'embeddings': None,
	'embedding_provider': '',
	'embedding_model': '',
	'embedding_model_path': '',
	'embedding_documents': None,
	'df_embedding': None,
	'vector_store': None,
	'vector_store_provider': '',
	'vector_store_name': '',
	'vector_store_namespace': '',
}


def ensure_document_processing_state( ) -> None:
	"""Initialize document-processing session state.

	Purpose:
		Writes every chunking, embedding, and vector-storage state key before UI or execution
		paths read it.

	Returns:
		None: This function updates Streamlit session state.
	"""
	for key, value in DOCUMENT_PROCESSING_STATE.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = value


def clear_document_processing_outputs( ) -> None:
	"""Clear derived document-processing outputs.

	Purpose:
		Clears chunk, embedding, and vector-store state without modifying loaded source documents.

	Returns:
		None: This function updates Streamlit session state.
	"""
	ensure_document_processing_state( )
	for key, value in DOCUMENT_PROCESSING_STATE.items( ):
		st.session_state[ key ] = value
	st.session_state[ 'chunks' ] = None
	st.session_state[ 'df_chunks' ] = None


def reset_document_processing_controls( key_prefix: str ) -> None:
	"""Request a reset for one loader's processing controls.

	Args:
		key_prefix (str): Unique Streamlit key prefix assigned to the loader.

	Returns:
		None: This function updates Streamlit session state.
	"""
	throw_if( 'key_prefix', key_prefix )
	clear_document_processing_outputs( )
	st.session_state[ f'{key_prefix}_processing_reset_request' ] = True


def document_signature( documents: List[ Document ] ) -> str:
	"""Create a deterministic signature for loaded documents.

	Args:
		documents (List[Document]): Loaded source documents.

	Returns:
		str: SHA-256 signature representing content and metadata.
	"""
	throw_if( 'documents', documents )
	digest = sha256( )
	for document in documents:
		digest.update( str( document.page_content or '' ).encode( 'utf-8', errors='ignore' ) )
		metadata = json.dumps( document.metadata or { }, sort_keys=True, default=str )
		digest.update( metadata.encode( 'utf-8', errors='ignore' ) )
	return digest.hexdigest( )


def create_chunk_dataframe( chunks: List[ Document ] ) -> DataFrame:
	"""Create the read-only chunk display dataframe.

	Args:
		chunks (List[Document]): Chunk documents to project.

	Returns:
		DataFrame: Chunk metadata and text.
	"""
	throw_if( 'chunks', chunks )
	rows: List[ Dict[ str, Any ] ] = [ ]
	for chunk in chunks:
		metadata = chunk.metadata or { }
		rows.append( {
			'Chunk ID': metadata.get( 'chunk_id', '' ),
			'Document ID': metadata.get( 'document_id', '' ),
			'Source': metadata.get( 'source', '' ),
			'Characters': len( chunk.page_content or '' ),
			'Chunk Text': chunk.page_content or '',
		} )
	return pd.DataFrame( rows )


def create_embedding_dataframe( chunks: List[ Document ], vectors: List[ List[ float ] ],
	provider: str, model: str ) -> DataFrame:
	"""Create the read-only embedding display dataframe.

	Args:
		chunks (List[Document]): Chunk documents represented by the vectors.
		vectors (List[List[float]]): Generated embedding vectors.
		provider (str): Embedding provider used to generate the vectors.
		model (str): Model name or local GGUF path label.

	Returns:
		DataFrame: Embedding metadata, source text, and vectors.
	"""
	throw_if( 'chunks', chunks )
	throw_if( 'vectors', vectors )
	throw_if( 'provider', provider )
	throw_if( 'model', model )
	rows: List[ Dict[ str, Any ] ] = [ ]
	for index, chunk in enumerate( chunks ):
		vector = vectors[ index ]
		metadata = chunk.metadata or { }
		rows.append( {
			'Chunk ID': metadata.get( 'chunk_id', '' ),
			'Provider': provider,
			'Model': model,
			'Dimensions': len( vector ),
			'Text': chunk.page_content or '',
			'Embedding': vector,
		} )
	return pd.DataFrame( rows )


def render_document_processing_inputs( loader_name: str, key_prefix: str ) -> None:
	"""Render chunking, embedding, and vector-storage controls for one loader.

	Args:
		loader_name (str): Loader class name associated with the expander.
		key_prefix (str): Unique Streamlit key prefix assigned to the loader.

	Returns:
		None: This function renders Streamlit controls.
	"""
	throw_if( 'loader_name', loader_name )
	throw_if( 'key_prefix', key_prefix )
	ensure_document_processing_state( )

	size_key = f'{key_prefix}_chunk_size'
	overlap_key = f'{key_prefix}_chunk_overlap'
	provider_key = f'{key_prefix}_embedding_provider'
	model_key = f'{key_prefix}_embedding_model'
	path_key = f'{key_prefix}_embedding_model_path'
	store_key = f'{key_prefix}_vector_store_provider'
	index_key = f'{key_prefix}_pinecone_index'
	namespace_key = f'{key_prefix}_pinecone_namespace'
	reset_key = f'{key_prefix}_processing_reset_request'

	if st.session_state.get( reset_key, False ):
		st.session_state[ size_key ] = DEFAULT_CHUNK_SIZE
		st.session_state[ overlap_key ] = DEFAULT_CHUNK_OVERLAP
		st.session_state[ provider_key ] = DEFAULT_EMBEDDING_PROVIDER
		st.session_state[ model_key ] = DEFAULT_EMBEDDING_MODEL
		st.session_state[ path_key ] = ''
		st.session_state[ store_key ] = VECTOR_STORES[ 0 ]
		st.session_state[ index_key ] = ''
		st.session_state[ namespace_key ] = ''
		st.session_state[ reset_key ] = False

	defaults = {
		size_key: DEFAULT_CHUNK_SIZE,
		overlap_key: DEFAULT_CHUNK_OVERLAP,
		provider_key: DEFAULT_EMBEDDING_PROVIDER,
		model_key: DEFAULT_EMBEDDING_MODEL,
		path_key: '',
		store_key: VECTOR_STORES[ 0 ],
		index_key: '',
		namespace_key: '',
	}
	for key, value in defaults.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = value

	if int( st.session_state[ overlap_key ] ) >= int( st.session_state[ size_key ] ):
		st.session_state[ overlap_key ] = max( 0, int( st.session_state[ size_key ] ) // 5 )

	chunk_col, overlap_col = st.columns( 2 )
	with chunk_col:
		st.number_input( 'Chunk Size', min_value=1, step=1, key=size_key )
	with overlap_col:
		st.number_input(
			'Chunk Overlap',
			min_value=0,
			max_value=max( 0, int( st.session_state[ size_key ] ) - 1 ),
			step=1,
			key=overlap_key,
		)

	provider_col, model_col = st.columns( 2 )
	with provider_col:
		provider = st.selectbox(
			'Embedding Provider',
			options=list( EMBEDDING_MODELS.keys( ) ) + [ 'Local GGUF' ],
			key=provider_key,
		)

	with model_col:
		if provider == 'Local GGUF':
			st.text_input(
				'Local GGUF Model',
				key=path_key,
				placeholder='Path to a local embedding GGUF model',
			)
		else:
			model_options = EMBEDDING_MODELS[ provider ]
			if st.session_state.get( model_key, '' ) not in model_options:
				st.session_state[ model_key ] = model_options[ 0 ]
			st.selectbox( 'Embedding Model', options=model_options, key=model_key )

	store_col, target_col = st.columns( 2 )
	with store_col:
		store_provider = st.selectbox( 'Vector Store', options=VECTOR_STORES, key=store_key )
	with target_col:
		if store_provider == 'Pinecone':
			st.text_input( 'Pinecone Index', key=index_key, placeholder='Existing Pinecone index' )
		else:
			st.text_input(
				'Chroma Directory',
				value=str( CHROMA_DIRECTORY ),
				disabled=True,
				key=f'{key_prefix}_chroma_directory',
			)

	if store_provider == 'Pinecone':
		st.text_input(
			'Pinecone Namespace',
			key=namespace_key,
			placeholder='Optional namespace',
		)


def render_document_processing_actions( loader_name: str, key_prefix: str ) -> None:
	"""Render and execute Chunk, Embed, and Store actions for one loader.

	Args:
		loader_name (str): Loader class name associated with the expander.
		key_prefix (str): Unique Streamlit key prefix assigned to the loader.

	Returns:
		None: This function renders controls and updates Streamlit session state.
	"""
	throw_if( 'loader_name', loader_name )
	throw_if( 'key_prefix', key_prefix )
	ensure_document_processing_state( )

	documents = st.session_state.get( 'documents' ) or [ ]
	active_documents = st.session_state.get( 'active_loader' ) == loader_name and bool( documents )
	current_signature = document_signature( documents ) if active_documents else ''
	chunk_signature = st.session_state.get( 'chunk_source_signature', '' )
	if active_documents and chunk_signature and chunk_signature != current_signature:
		clear_document_processing_outputs( )
		chunk_signature = ''

	chunked_documents = st.session_state.get( 'chunked_documents' ) or [ ]
	embeddings = st.session_state.get( 'embeddings' ) or [ ]
	embedder = st.session_state.get( 'embedder' )
	current_size = int( st.session_state[ f'{key_prefix}_chunk_size' ] )
	current_overlap = int( st.session_state[ f'{key_prefix}_chunk_overlap' ] )
	chunk_config_current = (
		st.session_state.get( 'chunk_size_used', 0 ) == current_size
		and st.session_state.get( 'chunk_overlap_used', 0 ) == current_overlap
	)
	can_embed = active_documents and bool( chunked_documents ) \
		and chunk_signature == current_signature and chunk_config_current

	provider = str( st.session_state[ f'{key_prefix}_embedding_provider' ] )
	model = str( st.session_state.get( f'{key_prefix}_embedding_model', '' ) )
	model_path = str( st.session_state.get( f'{key_prefix}_embedding_model_path', '' ) )
	embedding_config_current = (
		st.session_state.get( 'embedding_provider', '' ) == provider
		and st.session_state.get( 'embedding_model', '' ) == model
		and st.session_state.get( 'embedding_model_path', '' ) == model_path
	)
	can_store = can_embed and bool( embeddings ) and embedder is not None and embedding_config_current

	chunk_col, embed_col, store_col = st.columns( 3 )
	chunk_clicked = chunk_col.button(
		'Chunk', key=f'{key_prefix}_chunk_documents', icon='✂️',
		disabled=not active_documents, width='stretch',
	)
	embed_clicked = embed_col.button(
		'Embed', key=f'{key_prefix}_embed_documents', icon='🧬',
		disabled=not can_embed, width='stretch',
	)
	store_clicked = store_col.button(
		'Store', key=f'{key_prefix}_store_vectors', icon='🗄️',
		disabled=not can_store, width='stretch',
	)

	if chunk_clicked:
		source_documents: List[ Document ] = [ ]
		for index, document in enumerate( documents, start=1 ):
			metadata = dict( document.metadata or { } )
			metadata.setdefault( 'document_id', index )
			source_documents.append( Document(
				page_content=document.page_content,
				metadata=metadata,
			) )

		splitter = RecursiveCharacterTextSplitter(
			chunk_size=current_size,
			chunk_overlap=current_overlap,
		)
		chunks = splitter.split_documents( source_documents )
		for index, chunk in enumerate( chunks, start=1 ):
			metadata = dict( chunk.metadata or { } )
			metadata[ 'chunk_id' ] = f'chunk-{index:06d}'
			metadata[ 'chunk_size' ] = current_size
			metadata[ 'chunk_overlap' ] = current_overlap
			chunk.metadata = metadata

		st.session_state[ 'chunked_documents' ] = chunks
		st.session_state[ 'chunks' ] = [ chunk.page_content for chunk in chunks ]
		st.session_state[ 'df_chunking' ] = create_chunk_dataframe( chunks )
		st.session_state[ 'df_chunks' ] = st.session_state[ 'df_chunking' ]
		st.session_state[ 'chunk_source_signature' ] = current_signature
		st.session_state[ 'chunk_size_used' ] = current_size
		st.session_state[ 'chunk_overlap_used' ] = current_overlap
		st.session_state[ 'embedder' ] = None
		st.session_state[ 'embeddings' ] = None
		st.session_state[ 'embedding_provider' ] = ''
		st.session_state[ 'embedding_model' ] = ''
		st.session_state[ 'embedding_model_path' ] = ''
		st.session_state[ 'embedding_documents' ] = None
		st.session_state[ 'df_embedding' ] = None
		st.session_state[ 'vector_store' ] = None
		st.session_state[ 'vector_store_provider' ] = ''
		st.session_state[ 'vector_store_name' ] = ''
		st.session_state[ 'vector_store_namespace' ] = ''
		st.success( f'Created {len( chunks )} chunk(s).' )

	if embed_clicked:
		factory = EmbeddingFactory( )
		embedder = factory.create(
			provider=provider,
			model=model,
			model_path=model_path,
		)
		texts = [ chunk.page_content for chunk in chunked_documents ]
		vectors = embedder.embed_documents( texts )
		if len( vectors ) != len( chunked_documents ):
			raise RuntimeError( 'Embedding count does not match the chunk count.' )
		dimensions = { len( vector ) for vector in vectors }
		if len( dimensions ) != 1:
			raise RuntimeError( 'Embedding vectors do not have a consistent dimension.' )
		if not all( math.isfinite( float( value ) ) for vector in vectors for value in vector ):
			raise RuntimeError( 'Embedding vectors contain non-finite values.' )

		display_model = model_path if provider == 'Local GGUF' else model
		st.session_state[ 'embedder' ] = embedder
		st.session_state[ 'embeddings' ] = vectors
		st.session_state[ 'embedding_provider' ] = provider
		st.session_state[ 'embedding_model' ] = model
		st.session_state[ 'embedding_model_path' ] = model_path
		st.session_state[ 'embedding_documents' ] = list( chunked_documents )
		st.session_state[ 'df_embedding' ] = create_embedding_dataframe(
			chunks=list( chunked_documents ),
			vectors=vectors,
			provider=provider,
			model=display_model,
		)
		st.session_state[ 'vector_store' ] = None
		st.session_state[ 'vector_store_provider' ] = ''
		st.session_state[ 'vector_store_name' ] = ''
		st.session_state[ 'vector_store_namespace' ] = ''
		st.success(
			f'Generated {len( vectors )} embedding(s) with {next( iter( dimensions ) )} dimensions.'
		)

	if store_clicked:
		store_provider = str( st.session_state[ f'{key_prefix}_vector_store_provider' ] )
		if store_provider == 'Chroma':
			store_name = f'foo_{loader_name.lower( ).replace( "loader", "" )}_documents'
			store = ChromaStore( )
			vector_store = store.create(
				documents=list( chunked_documents ),
				embedder=embedder,
				collection_name=store_name,
				persist_directory=str( CHROMA_DIRECTORY ),
			)
			namespace = ''
		else:
			store_name = str( st.session_state[ f'{key_prefix}_pinecone_index' ] )
			namespace = str( st.session_state[ f'{key_prefix}_pinecone_namespace' ] )
			store = PineconeStore( )
			vector_store = store.create(
				documents=list( chunked_documents ),
				embedder=embedder,
				index_name=store_name,
				namespace=namespace,
				api_key=cfg.PINECONE_API_KEY,
			)

		st.session_state[ 'vector_store' ] = vector_store
		st.session_state[ 'vector_store_provider' ] = store_provider
		st.session_state[ 'vector_store_name' ] = store_name
		st.session_state[ 'vector_store_namespace' ] = namespace
		st.success( f'Stored {len( chunked_documents )} chunk(s) in {store_provider}: {store_name}.' )


def render_loading_tabs( ) -> None:
	"""Render Loading-mode document, chunk, and embedding tabs.

	Purpose:
		Preserves the existing document preview while exposing current chunk and embedding results.

	Returns:
		None: This function renders Streamlit content.
	"""
	ensure_document_processing_state( )
	documents = st.session_state.get( 'documents' ) or [ ]
	current_signature = document_signature( documents ) if documents else ''
	derived_current = bool( documents ) \
		and current_signature == st.session_state.get( 'chunk_source_signature', '' )
	document_tab, chunk_tab, embedding_tab = st.tabs( [ 'Document', 'Chunks', 'Embeddings' ] )

	with document_tab:
		if not documents:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f"Active Loader: {st.session_state.get( 'active_loader', '' )}" )
			st.write( f'Documents: {len( documents )}' )
			for index, document in enumerate( documents[ :5 ] ):
				with st.expander( f'Document {index + 1}', expanded=True ):
					st.json( document.metadata )
					st.text_area(
						'Content', document.page_content[ : ], height=450,
						key=f'preview_doc_{index}',
					)

	with chunk_tab:
		df_chunking = st.session_state.get( 'df_chunking' )
		if not derived_current or not isinstance( df_chunking, DataFrame ) or df_chunking.empty:
			st.info( 'No chunks created for the active documents.' )
		else:
			st.caption( f'Chunks: {len( df_chunking )}' )
			st.data_editor(
				df_chunking, disabled=True, hide_index=True,
				use_container_width=True, height=520, key='loading_df_chunking',
			)

	with embedding_tab:
		df_embedding = st.session_state.get( 'df_embedding' )
		if not derived_current or not isinstance( df_embedding, DataFrame ) or df_embedding.empty:
			st.info( 'No embeddings generated for the active documents.' )
		else:
			provider = st.session_state.get( 'embedding_provider', '' )
			model = st.session_state.get( 'embedding_model_path', '' ) \
				if provider == 'Local GGUF' else st.session_state.get( 'embedding_model', '' )
			st.caption( f'Provider: {provider} | Model: {model}' )
			store_provider = st.session_state.get( 'vector_store_provider', '' )
			store_name = st.session_state.get( 'vector_store_name', '' )
			if store_provider and store_name:
				st.caption( f'Vector Store: {store_provider} | Target: {store_name}' )
			st.data_editor(
				df_embedding, disabled=True, hide_index=True,
				use_container_width=True, height=520, key='loading_df_embedding',
			)
'''


def throw_if( name: str, value: object ) -> None:
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )
    if isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'Argument "{name}" cannot be empty!' )


def leading_width( line: str ) -> int:
    prefix = line[ :len( line ) - len( line.lstrip( ' \t' ) ) ]
    return len( prefix.replace( '\t', '    ' ) )


def add_imports( source: str ) -> str:
    imports = [
        'from hashlib import sha256',
        'import math',
        'from langchain_text_splitters import RecursiveCharacterTextSplitter',
        'from embeddings import EmbeddingFactory',
        'from stores.vector import ChromaStore, PineconeStore',
    ]
    anchor = 'from __future__ import annotations\n'
    throw_if( 'future import anchor', anchor if anchor in source else '' )
    missing = [ value for value in imports if value not in source ]
    if missing:
        source = source.replace( anchor, anchor + '\n'.join( missing ) + '\n', 1 )
    return source


def add_helpers( source: str ) -> str:
    if 'def render_document_processing_inputs(' in source:
        return source
    anchor = '_streamlit_data_editor = st.data_editor'
    position = source.find( anchor )
    throw_if( 'helper anchor', anchor if position >= 0 else '' )
    return source[ :position ] + HELPERS + '\n\n' + source[ position: ]


def loader_bounds( source: str, label: str ) -> tuple[ int, int, str ]:
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)with st\.expander\( label=['\"]" + re.escape( label ) + r"['\"]"
    )
    match = pattern.search( source )
    if match is None:
        raise RuntimeError( f'Loader expander not found: {label}' )
    indent = match.group( 'indent' )
    next_pattern = re.compile( r'(?m)^' + re.escape( indent ) + r'with st\.expander\(' )
    next_match = next_pattern.search( source, match.end( ) )
    end = next_match.start( ) if next_match else source.find( '\t# RIGHT COLUMN — DOCUMENT RENDERING' )
    if end < 0:
        raise RuntimeError( f'Loader boundary not found: {label}' )
    return match.start( ), end, indent


def integrate_loader( source: str, label: str, loader_name: str, key_prefix: str ) -> str:
    start, end, indent = loader_bounds( source, label )
    block = source[ start:end ]
    inner = indent + '\t'
    input_call = f"{inner}render_document_processing_inputs( '{loader_name}', '{key_prefix}' )"
    action_call = f"{inner}render_document_processing_actions( '{loader_name}', '{key_prefix}' )"

    if input_call not in block:
        button_index = block.find( 'Buttons: Load / Clear / Save' )
        if button_index < 0:
            raise RuntimeError( f'Button anchor not found: {label}' )
        line_start = block.rfind( '\n', 0, button_index ) + 1
        separator_start = block.rfind( '\n', 0, max( 0, line_start - 1 ) ) + 1
        block = block[ :separator_start ] + input_call + '\n\n' + block[ separator_start: ]

    clear_pattern = re.compile( r'(?m)^(?P<indent>[ \t]*)if clear_' + re.escape( key_prefix ) + r':' )
    clear_match = clear_pattern.search( block )
    if clear_match is None:
        raise RuntimeError( f'Clear anchor not found: {label}' )
    reset_call = clear_match.group( 'indent' ) + "\treset_document_processing_controls( '" + key_prefix + "' )"
    if reset_call not in block:
        insertion = clear_match.end( )
        block = block[ :insertion ] + '\n' + reset_call + block[ insertion: ]

    if action_call not in block:
        block = block.rstrip( ) + '\n\n' + action_call + '\n\n'

    return source[ :start ] + block + source[ end: ]


def replace_loading_results( source: str ) -> str:
    right_marker = '\t# RIGHT COLUMN — DOCUMENT RENDERING'
    nlp_marker = '\t# NLP METRIC CALCULATIONS'
    right_index = source.find( right_marker )
    nlp_index = source.find( nlp_marker, right_index )
    if right_index < 0 or nlp_index < 0:
        raise RuntimeError( 'Loading results anchors were not found.' )

    with_index = source.find( '\twith right:', right_index, nlp_index )
    if with_index < 0:
        raise RuntimeError( 'Loading right-column block was not found.' )
    nlp_separator = source.rfind( '\t# ------------------------------------------------------------------', with_index, nlp_index )
    if nlp_separator < 0:
        nlp_separator = nlp_index
    source = source[ :with_index ] + '\twith right:\n\t\trender_loading_tabs( )\n\n' + source[ nlp_separator: ]

    nlp_index = source.find( nlp_marker, with_index )
    nlp_separator = source.rfind( '\t# ------------------------------------------------------------------', with_index, nlp_index )
    lines = source[ nlp_separator: ].splitlines( keepends=True )
    marker_line = next( index for index, line in enumerate( lines ) if 'NLP METRIC CALCULATIONS' in line )
    marker_indent = leading_width( lines[ marker_line ] )
    end_line = len( lines )
    for index in range( marker_line + 1, len( lines ) ):
        line = lines[ index ]
        if line.strip( ) and leading_width( line ) < marker_indent:
            end_line = index
            break
    removal_end = nlp_separator + sum( len( line ) for line in lines[ :end_line ] )
    return source[ :nlp_separator ] + source[ removal_end: ]


def normalize_generated_whitespace( source: str ) -> str:
    return '\n'.join( line.rstrip( ) for line in source.splitlines( ) ) + '\n'


def main( ) -> None:
    source = APP_PATH.read_text( encoding='utf-8' )
    source = add_imports( source )
    source = add_helpers( source )
    for label, loader_name, key_prefix in TARGET_LOADERS:
        source = integrate_loader( source, label, loader_name, key_prefix )
    source = replace_loading_results( source )
    source = normalize_generated_whitespace( source )
    APP_PATH.write_text( source, encoding='utf-8' )


if __name__ == '__main__':
    main( )
