'''
    ******************************************************************************************
      Assembly:                Foo
      Filename:                langchain_pipeline.py
      Author:                  Terry D. Eppler
      Created:                 09-08-2026
      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-08-2026
    ******************************************************************************************
    <copyright file="langchain_pipeline.py" company="Terry D. Eppler">

         Foo is a python framework for web scraping information into ML pipelines.

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

    </copyright>
    <summary>
        LangChain document chunking, embedding, vector-storage, and Loading-mode UI helpers.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pandas import DataFrame
import streamlit as st

from stores.vector import ChromaStore

# =====================================================================
# LANGCHAIN PIPELINE CONFIGURATION
# =====================================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_PROVIDER = 'HuggingFace'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHROMA_DIRECTORY = Path( 'stores' ) / 'chroma'

EMBEDDING_MODELS: Dict[ str, List[ str ] ] = {
    'OpenAI': [
        'text-embedding-3-small',
        'text-embedding-3-large',
    ],
    'GoogleGenerativeAI': [
        'gemini-embedding-2-preview',
    ],
    'MistralAI': [
        'mistral-embed',
    ],
    'HuggingFace': [
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2',
    ],
}

PIPELINE_STATE_DEFAULTS: Dict[ str, Any ] = {
    'chunk_size': DEFAULT_CHUNK_SIZE,
    'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
    'chunk_source_signature': '',
    'df_chunking': None,
    'df_embedding': None,
    'embedder': None,
    'embeddings': None,
    'embedding_provider': None,
    'embedding_model': None,
    'embedding_source': None,
    'embedding_documents': None,
    'df_embedding_input': None,
    'df_embedding_output': None,
    'vector_store': None,
    'vector_store_collection': '',
}


def throw_if( name: str, value: object ) -> None:
    """Input guard.

    Purpose:
        Validates that a required argument contains a usable value before the surrounding
        LangChain workflow continues.

    Args:
        name (str): Name value used by the operation.
        value (object): Value value used by the operation.

    Returns:
        None: This function performs validation and does not return a value.
    """
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )

    if isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'Argument "{name}" cannot be empty!' )

    if isinstance( value, ( list, tuple, dict, set ) ) and len( value ) == 0:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


class DocumentChunker( ):
    """LangChain document chunking component.

    Purpose:
        Splits loaded LangChain documents with ``RecursiveCharacterTextSplitter`` while
        preserving source metadata and adding deterministic document and chunk identifiers.
    """

    documents: List[ Document ]
    chunk_size: int
    chunk_overlap: int
    splitter: RecursiveCharacterTextSplitter | None
    chunks: List[ Document ]

    def __init__( self ) -> None:
        """Initialize the document chunker.

        Purpose:
            Initializes reusable state used by later LangChain document-splitting operations.

        Returns:
            None: This method initializes instance state.
        """
        self.documents = [ ]
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        self.splitter = None
        self.chunks = [ ]

    def split( self, documents: List[ Document ], chunk_size: int,
        chunk_overlap: int ) -> List[ Document ]:
        """Split LangChain documents.

        Purpose:
            Uses ``RecursiveCharacterTextSplitter`` to split source documents while retaining
            their existing metadata for downstream embedding and vector-storage operations.

        Args:
            documents (List[Document]): Loaded LangChain documents to split.
            chunk_size (int): Maximum character-oriented chunk size supplied to LangChain.
            chunk_overlap (int): Character overlap supplied to LangChain.

        Returns:
            List[Document]: Chunked LangChain documents with provenance metadata.
        """
        throw_if( 'documents', documents )
        throw_if( 'chunk_size', chunk_size )
        throw_if( 'chunk_overlap', chunk_overlap )
        self.documents = list( documents )
        self.chunk_size = int( chunk_size )
        self.chunk_overlap = int( chunk_overlap )

        source_documents: List[ Document ] = [ ]
        for index, document in enumerate( self.documents, start=1 ):
            metadata = dict( document.metadata or { } )
            metadata.setdefault( 'document_id', index )
            source_documents.append( Document(
                page_content=document.page_content,
                metadata=metadata,
            ) )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.chunks = self.splitter.split_documents( source_documents )

        for index, chunk in enumerate( self.chunks, start=1 ):
            metadata = dict( chunk.metadata or { } )
            metadata[ 'chunk_id' ] = f'chunk-{index:06d}'
            metadata[ 'chunk_size' ] = self.chunk_size
            metadata[ 'chunk_overlap' ] = self.chunk_overlap
            chunk.metadata = metadata

        return self.chunks


class EmbeddingFactory( ):
    """LangChain embedding-provider factory.

    Purpose:
        Creates one provider-specific LangChain ``Embeddings`` implementation behind a common
        interface used by both embedding generation and vector storage.
    """

    provider: str
    model: str

    def __init__( self ) -> None:
        """Initialize the embedding factory.

        Purpose:
            Initializes provider and model state before a LangChain embedding implementation is
            requested.

        Returns:
            None: This method initializes instance state.
        """
        self.provider = ''
        self.model = ''

    def create( self, provider: str, model: str ) -> Embeddings:
        """Create a LangChain embedding implementation.

        Purpose:
            Selects OpenAI, Google Generative AI, Mistral AI, or Hugging Face embeddings while
            keeping provider-specific construction outside the Streamlit UI.

        Args:
            provider (str): Embedding provider selected by the user.
            model (str): Provider model selected by the user.

        Returns:
            Embeddings: Configured LangChain embedding implementation.
        """
        throw_if( 'provider', provider )
        throw_if( 'model', model )
        self.provider = provider
        self.model = model

        if self.provider == 'OpenAI':
            return OpenAIEmbeddings( model=self.model )

        if self.provider == 'GoogleGenerativeAI':
            return GoogleGenerativeAIEmbeddings( model=self.model )

        if self.provider == 'MistralAI':
            return MistralAIEmbeddings( model=self.model )

        if self.provider == 'HuggingFace':
            return HuggingFaceEmbeddings( model_name=self.model )

        raise ValueError( f'Unsupported embedding provider: {self.provider}' )


# =====================================================================
# SESSION-STATE UTILITIES
# =====================================================================

def ensure_langchain_state( ) -> None:
    """Initialize LangChain pipeline session state.

    Purpose:
        Writes every shared chunking, embedding, and vector-storage state key before UI or
        pipeline routines read it.

    Returns:
        None: This function updates Streamlit session state.
    """
    for key, value in PIPELINE_STATE_DEFAULTS.items( ):
        if key not in st.session_state:
            st.session_state[ key ] = value


def clear_langchain_outputs( ) -> None:
    """Clear derived LangChain pipeline outputs.

    Purpose:
        Clears chunk, embedding, dataframe, and vector-store state without changing the loaded
        source-document contract.

    Returns:
        None: This function updates Streamlit session state.
    """
    ensure_langchain_state( )
    st.session_state[ 'chunks' ] = None
    st.session_state[ 'chunked_documents' ] = None
    st.session_state[ 'df_chunks' ] = None
    st.session_state[ 'df_chunking' ] = None
    st.session_state[ 'chunk_source_signature' ] = ''
    st.session_state[ 'embedder' ] = None
    st.session_state[ 'embeddings' ] = None
    st.session_state[ 'embedding_provider' ] = None
    st.session_state[ 'embedding_model' ] = None
    st.session_state[ 'embedding_source' ] = None
    st.session_state[ 'embedding_documents' ] = None
    st.session_state[ 'df_embedding_input' ] = None
    st.session_state[ 'df_embedding_output' ] = None
    st.session_state[ 'df_embedding' ] = None
    st.session_state[ 'vector_store' ] = None
    st.session_state[ 'vector_store_collection' ] = ''


def reset_langchain_controls( key_prefix: str ) -> None:
    """Request a LangChain control reset.

    Purpose:
        Marks one loader's LangChain controls for reset on the next Streamlit rerun and clears
        all derived chunking, embedding, and vector-store results.

    Args:
        key_prefix (str): Unique Streamlit key prefix assigned to the loader expander.

    Returns:
        None: This function updates Streamlit session state.
    """
    throw_if( 'key_prefix', key_prefix )
    clear_langchain_outputs( )
    st.session_state[ f'{key_prefix}_langchain_reset_request' ] = True


def document_signature( documents: List[ Document ] ) -> str:
    """Create a deterministic source-document signature.

    Purpose:
        Detects whether chunk or embedding outputs belong to the currently loaded documents so
        stale derived state cannot be reused after another loader operation.

    Args:
        documents (List[Document]): Loaded source documents to fingerprint.

    Returns:
        str: SHA-256 signature representing document content and metadata.
    """
    throw_if( 'documents', documents )
    digest = sha256( )

    for document in documents:
        digest.update( str( document.page_content or '' ).encode( 'utf-8', errors='ignore' ) )
        metadata = json.dumps( document.metadata or { }, sort_keys=True, default=str )
        digest.update( metadata.encode( 'utf-8', errors='ignore' ) )

    return digest.hexdigest( )


def create_chunk_dataframe( chunks: List[ Document ] ) -> DataFrame:
    """Create the chunking dataframe.

    Purpose:
        Projects canonical LangChain chunk documents into the read-only table displayed by the
        Loading-mode Chunks tab.

    Args:
        chunks (List[Document]): Chunked LangChain documents to project.

    Returns:
        DataFrame: Chunk metadata and text arranged for Streamlit display.
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


def create_embedding_dataframe( chunks: List[ Document ], embeddings: List[ List[ float ] ],
    provider: str, model: str ) -> DataFrame:
    """Create the embedding dataframe.

    Purpose:
        Projects chunk provenance, embedding configuration, vector dimensions, source text, and
        generated vectors into the read-only Loading-mode Embeddings table.

    Args:
        chunks (List[Document]): Chunk documents represented by the vectors.
        embeddings (List[List[float]]): Numeric vectors returned by LangChain.
        provider (str): Embedding provider used to generate the vectors.
        model (str): Embedding model used to generate the vectors.

    Returns:
        DataFrame: Embedding metadata and vectors arranged for Streamlit display.
    """
    throw_if( 'chunks', chunks )
    throw_if( 'embeddings', embeddings )
    throw_if( 'provider', provider )
    throw_if( 'model', model )
    rows: List[ Dict[ str, Any ] ] = [ ]

    for index, chunk in enumerate( chunks ):
        vector = embeddings[ index ]
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


# =====================================================================
# STREAMLIT CONTROLS
# =====================================================================

def render_langchain_inputs( loader_name: str, key_prefix: str ) -> None:
    """Render LangChain chunking and embedding controls.

    Purpose:
        Adds chunk-size, overlap, embedding-provider, and embedding-model controls to one loader
        expander without changing that loader's existing source-specific controls.

    Args:
        loader_name (str): Loader class name associated with the expander.
        key_prefix (str): Unique Streamlit key prefix assigned to the expander.

    Returns:
        None: This function renders Streamlit controls and updates session state.
    """
    throw_if( 'loader_name', loader_name )
    throw_if( 'key_prefix', key_prefix )
    ensure_langchain_state( )

    size_key = f'{key_prefix}_chunk_size'
    overlap_key = f'{key_prefix}_chunk_overlap'
    provider_key = f'{key_prefix}_embedding_provider'
    model_key = f'{key_prefix}_embedding_model'
    reset_key = f'{key_prefix}_langchain_reset_request'

    if st.session_state.get( reset_key, False ):
        st.session_state[ size_key ] = DEFAULT_CHUNK_SIZE
        st.session_state[ overlap_key ] = DEFAULT_CHUNK_OVERLAP
        st.session_state[ provider_key ] = DEFAULT_EMBEDDING_PROVIDER
        st.session_state[ model_key ] = DEFAULT_EMBEDDING_MODEL
        st.session_state[ reset_key ] = False

    if size_key not in st.session_state:
        st.session_state[ size_key ] = DEFAULT_CHUNK_SIZE

    if overlap_key not in st.session_state:
        st.session_state[ overlap_key ] = DEFAULT_CHUNK_OVERLAP

    if provider_key not in st.session_state:
        st.session_state[ provider_key ] = DEFAULT_EMBEDDING_PROVIDER

    provider = st.session_state.get( provider_key, DEFAULT_EMBEDDING_PROVIDER )
    if provider not in EMBEDDING_MODELS:
        st.session_state[ provider_key ] = DEFAULT_EMBEDDING_PROVIDER
        provider = DEFAULT_EMBEDDING_PROVIDER

    model_options = EMBEDDING_MODELS[ provider ]
    if st.session_state.get( model_key, '' ) not in model_options:
        st.session_state[ model_key ] = model_options[ 0 ]

    if int( st.session_state[ overlap_key ] ) >= int( st.session_state[ size_key ] ):
        st.session_state[ overlap_key ] = max( 0, int( st.session_state[ size_key ] ) // 5 )

    chunk_col, overlap_col = st.columns( 2 )
    with chunk_col:
        st.number_input(
            'Chunk Size',
            min_value=1,
            value=int( st.session_state[ size_key ] ),
            step=1,
            key=size_key,
        )

    with overlap_col:
        st.number_input(
            'Chunk Overlap',
            min_value=0,
            max_value=max( 0, int( st.session_state[ size_key ] ) - 1 ),
            value=int( st.session_state[ overlap_key ] ),
            step=1,
            key=overlap_key,
        )

    provider_col, model_col = st.columns( 2 )
    with provider_col:
        provider = st.selectbox(
            'Embedding Provider',
            options=list( EMBEDDING_MODELS.keys( ) ),
            key=provider_key,
        )

    model_options = EMBEDDING_MODELS[ provider ]
    if st.session_state.get( model_key, '' ) not in model_options:
        st.session_state[ model_key ] = model_options[ 0 ]

    with model_col:
        st.selectbox(
            'Embedding Model',
            options=model_options,
            key=model_key,
        )


def render_langchain_actions( loader_name: str, key_prefix: str ) -> None:
    """Render and execute LangChain pipeline actions.

    Purpose:
        Adds Chunk, Embed, and Store actions to a loader expander. Each action consumes the
        authoritative LangChain state produced by the preceding step and writes distinct
        downstream state without overloading the source-document contract.

    Args:
        loader_name (str): Loader class name associated with the expander.
        key_prefix (str): Unique Streamlit key prefix assigned to the expander.

    Returns:
        None: This function renders buttons and updates Streamlit session state.
    """
    throw_if( 'loader_name', loader_name )
    throw_if( 'key_prefix', key_prefix )
    ensure_langchain_state( )

    documents = st.session_state.get( 'documents' ) or [ ]
    active_loader = st.session_state.get( 'active_loader' )
    active_documents = active_loader == loader_name and bool( documents )
    current_signature = document_signature( documents ) if active_documents else ''
    chunk_signature = st.session_state.get( 'chunk_source_signature', '' )
    chunked_documents = st.session_state.get( 'chunked_documents' ) or [ ]
    embeddings = st.session_state.get( 'embeddings' ) or [ ]
    embedder = st.session_state.get( 'embedder' )

    can_embed = active_documents and bool( chunked_documents ) and \
        chunk_signature == current_signature
    can_store = can_embed and bool( embeddings ) and embedder is not None

    chunk_col, embed_col, store_col = st.columns( 3 )
    chunk_clicked = chunk_col.button(
        'Chunk',
        key=f'{key_prefix}_chunk_documents',
        icon='✂️',
        disabled=not active_documents,
        width='stretch',
    )
    embed_clicked = embed_col.button(
        'Embed',
        key=f'{key_prefix}_embed_documents',
        icon='🧬',
        disabled=not can_embed,
        width='stretch',
    )
    store_clicked = store_col.button(
        'Store',
        key=f'{key_prefix}_store_vectors',
        icon='🗄️',
        disabled=not can_store,
        width='stretch',
    )

    if chunk_clicked:
        chunker = DocumentChunker( )
        chunks = chunker.split(
            documents=list( documents ),
            chunk_size=int( st.session_state[ f'{key_prefix}_chunk_size' ] ),
            chunk_overlap=int( st.session_state[ f'{key_prefix}_chunk_overlap' ] ),
        )
        df_chunking = create_chunk_dataframe( chunks )
        st.session_state[ 'chunked_documents' ] = chunks
        st.session_state[ 'chunks' ] = [ chunk.page_content for chunk in chunks ]
        st.session_state[ 'df_chunking' ] = df_chunking
        st.session_state[ 'df_chunks' ] = df_chunking
        st.session_state[ 'chunk_source_signature' ] = current_signature
        st.session_state[ 'embedder' ] = None
        st.session_state[ 'embeddings' ] = None
        st.session_state[ 'embedding_documents' ] = None
        st.session_state[ 'df_embedding_input' ] = None
        st.session_state[ 'df_embedding_output' ] = None
        st.session_state[ 'df_embedding' ] = None
        st.session_state[ 'vector_store' ] = None
        st.session_state[ 'vector_store_collection' ] = ''
        st.success( f'Created {len( chunks )} LangChain chunk(s).' )

    if embed_clicked:
        provider = str( st.session_state[ f'{key_prefix}_embedding_provider' ] )
        model = str( st.session_state[ f'{key_prefix}_embedding_model' ] )
        factory = EmbeddingFactory( )
        embedder = factory.create( provider=provider, model=model )
        texts = [ chunk.page_content for chunk in chunked_documents ]
        vectors = embedder.embed_documents( texts )

        if len( vectors ) != len( chunked_documents ):
            raise RuntimeError( 'Embedding count does not match the chunk count.' )

        dimensions = { len( vector ) for vector in vectors }
        if len( dimensions ) != 1:
            raise RuntimeError( 'Embedding vectors do not have a consistent dimension.' )

        if not all( math.isfinite( float( value ) ) for vector in vectors for value in vector ):
            raise RuntimeError( 'Embedding vectors contain non-finite values.' )

        df_embedding = create_embedding_dataframe(
            chunks=list( chunked_documents ),
            embeddings=vectors,
            provider=provider,
            model=model,
        )
        df_embedding_input = pd.DataFrame( {
            'Chunk ID': [
                ( chunk.metadata or { } ).get( 'chunk_id', '' )
                for chunk in chunked_documents
            ],
            'Text': texts,
        } )
        st.session_state[ 'embedder' ] = embedder
        st.session_state[ 'embeddings' ] = vectors
        st.session_state[ 'embedding_provider' ] = provider
        st.session_state[ 'embedding_model' ] = model
        st.session_state[ 'embedding_source' ] = loader_name
        st.session_state[ 'embedding_documents' ] = list( chunked_documents )
        st.session_state[ 'df_embedding_input' ] = df_embedding_input
        st.session_state[ 'df_embedding_output' ] = df_embedding
        st.session_state[ 'df_embedding' ] = df_embedding
        st.session_state[ 'vector_store' ] = None
        st.session_state[ 'vector_store_collection' ] = ''
        st.success(
            f'Generated {len( vectors )} embedding(s) with {next( iter( dimensions ) )} dimensions.'
        )

    if store_clicked:
        collection_name = f'foo_{loader_name.lower( ).replace( "loader", "" )}_documents'
        store = ChromaStore( )
        vector_store = store.create(
            documents=list( chunked_documents ),
            embedder=embedder,
            collection_name=collection_name,
            persist_directory=str( CHROMA_DIRECTORY ),
        )
        st.session_state[ 'vector_store' ] = vector_store
        st.session_state[ 'vector_store_collection' ] = collection_name
        st.success( f'Stored {len( chunked_documents )} chunk(s) in Chroma: {collection_name}.' )


# =====================================================================
# LOADING-MODE RESULTS
# =====================================================================

def render_loading_tabs( ) -> None:
    """Render Loading-mode document, chunk, and embedding tabs.

    Purpose:
        Preserves the existing loaded-document preview while adding distinct read-only tables for
        chunk and embedding state in the right-hand Loading-mode area.

    Returns:
        None: This function renders Streamlit content.
    """
    ensure_langchain_state( )
    document_tab, chunk_tab, embedding_tab = st.tabs( [ 'Document', 'Chunks', 'Embeddings' ] )

    with document_tab:
        documents = st.session_state.get( 'documents' ) or [ ]
        if not documents:
            st.info( 'No documents loaded.' )
        else:
            st.caption( f"Active Loader: {st.session_state.get( 'active_loader', '' )}" )
            st.write( f'Documents: {len( documents )}' )
            for index, document in enumerate( documents[ :5 ] ):
                with st.expander( f'Document {index + 1}', expanded=True ):
                    st.json( document.metadata )
                    st.text_area(
                        'Content',
                        document.page_content[ : ],
                        height=450,
                        key=f'preview_doc_{index}',
                    )

    with chunk_tab:
        df_chunking = st.session_state.get( 'df_chunking' )
        if not isinstance( df_chunking, DataFrame ) or df_chunking.empty:
            st.info( 'No chunks created.' )
        else:
            st.caption( f'Chunks: {len( df_chunking )}' )
            st.data_editor(
                df_chunking,
                disabled=True,
                hide_index=True,
                use_container_width=True,
                height=520,
                key='loading_df_chunking',
            )

    with embedding_tab:
        df_embedding = st.session_state.get( 'df_embedding' )
        if not isinstance( df_embedding, DataFrame ) or df_embedding.empty:
            st.info( 'No embeddings generated.' )
        else:
            provider = st.session_state.get( 'embedding_provider', '' )
            model = st.session_state.get( 'embedding_model', '' )
            collection = st.session_state.get( 'vector_store_collection', '' )
            st.caption( f'Provider: {provider} | Model: {model}' )
            if collection:
                st.caption( f'Chroma Collection: {collection}' )
            st.data_editor(
                df_embedding,
                disabled=True,
                hide_index=True,
                use_container_width=True,
                height=520,
                key='loading_df_embedding',
            )
