'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                vector.py
	  Author:                  Terry D. Eppler
	  Created:                 09-08-2026
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        09-10-2026
	******************************************************************************************
	<summary>
		Non-destructive Chroma and Pinecone vector-store lifecycle implementations.
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

from boogr import Error, Logger
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def throw_if( name: str, value: object ) -> None:
	"""Validate a required runtime value."""
	if value is None or isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, ( list, tuple, dict, set ) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


def raise_logged_error( error: Exception, cause: str, method: str ) -> None:
	"""Log an ordinary exception once and raise Foo's structured error."""
	if isinstance( error, Error ):
		raise error
	exception = Error( error )
	exception.module = 'stores.vector'
	exception.cause = cause
	exception.method = method
	Logger( ).write( exception )
	raise exception


def scalar_metadata( metadata: Dict[ str, Any ] ) -> Dict[ str, Any ]:
	"""Convert metadata values to vector-provider-compatible scalar values."""
	result: Dict[ str, Any ] = { }
	for key, value in dict( metadata or { } ).items( ):
		if isinstance( value, ( str, int, float, bool ) ):
			result[ str( key ) ] = value
		elif value is not None:
			result[ str( key ) ] = json.dumps( value, ensure_ascii=False, default=str )
	return result


class ChromaStore( ):
	"""Persistent, non-destructive Chroma vector-store wrapper."""

	def __init__( self ) -> None:
		self.documents: List[ Document ] = [ ]
		self.embedder: Embeddings | None = None
		self.collection_name = ''
		self.persist_directory = ''
		self.vector_store: Chroma | None = None

	def connect( self, collection_name: str, persist_directory: str,
			embedder: Embeddings ) -> Chroma:
		"""Connect to an existing or new persistent Chroma collection."""
		try:
			throw_if( 'collection_name', collection_name )
			throw_if( 'persist_directory', persist_directory )
			throw_if( 'embedder', embedder )
			self.collection_name = str( collection_name ).strip( )
			self.persist_directory = str( persist_directory ).strip( )
			self.embedder = embedder
			Path( self.persist_directory ).mkdir( parents=True, exist_ok=True )
			from langchain_chroma import Chroma
			self.vector_store = Chroma( collection_name=self.collection_name,
				embedding_function=self.embedder, persist_directory=self.persist_directory )
			return self.vector_store
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'connect( self, **kwargs ) -> Chroma' )

	def create( self, documents: List[ Document ], embedder: Embeddings,
			collection_name: str, persist_directory: str ) -> Chroma:
		"""Connect and add documents without clearing existing collection data."""
		try:
			self.connect( collection_name=collection_name, persist_directory=persist_directory,
				embedder=embedder )
			self.add_documents( documents=documents )
			return self.vector_store
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'create( self, **kwargs ) -> Chroma' )

	def add_documents( self, documents: List[ Document ] ) -> List[ str ]:
		"""Embed and add documents to the connected collection."""
		try:
			throw_if( 'documents', documents )
			throw_if( 'vector_store', self.vector_store )
			self.documents = list( documents )
			ids = self.get_ids( self.documents )
			return self.vector_store.add_documents( documents=self.documents, ids=ids )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'add_documents( self, documents )' )

	def add_embeddings( self, documents: List[ Document ],
			embeddings: List[ List[ float ] ] ) -> List[ str ]:
		"""Add precomputed embeddings without embedding the documents again."""
		try:
			throw_if( 'documents', documents )
			throw_if( 'embeddings', embeddings )
			throw_if( 'vector_store', self.vector_store )
			if len( documents ) != len( embeddings ):
				raise ValueError( 'Embedding count must match document count.' )
			ids = self.get_ids( documents )
			self.vector_store._collection.upsert( ids=ids, embeddings=embeddings,
				metadatas=[ scalar_metadata( document.metadata or { } ) for document in documents ],
				documents=[ document.page_content or '' for document in documents ] )
			return ids
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'add_embeddings( self, **kwargs )' )

	def get_ids( self, documents: List[ Document ] ) -> List[ str ]:
		"""Return and validate stable document identifiers."""
		ids = [ str( ( document.metadata or { } ).get( 'chunk_id' ) )
			for document in documents ]
		if any( not value or value == 'None' for value in ids ):
			raise ValueError( 'Every document must contain a stable chunk_id.' )
		return ids

	def similarity_search( self, query: str, count: int=4,
			filters: Dict[ str, Any ]=None ) -> List[ Document ]:
		"""Return the most similar documents for a query."""
		try:
			throw_if( 'query', query )
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.similarity_search( query=query, k=int( count ), filter=filters )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'similarity_search( self, **kwargs )' )

	def similarity_search_with_score( self, query: str, count: int=4,
			filters: Dict[ str, Any ]=None ) -> List[ tuple[ Document, float ] ]:
		"""Return similar documents and their scores."""
		try:
			throw_if( 'query', query )
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.similarity_search_with_score( query=query, k=int( count ),
				filter=filters )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'similarity_search_with_score( self, **kwargs )' )

	def get( self, ids: List[ str ]=None, filters: Dict[ str, Any ]=None ) -> Dict[ str, Any ]:
		"""Return stored records selected by IDs or metadata."""
		try:
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.get( ids=ids, where=filters )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'get( self, **kwargs )' )

	def delete( self, ids: List[ str ] ) -> None:
		"""Delete selected records by stable identifier."""
		try:
			throw_if( 'ids', ids )
			throw_if( 'vector_store', self.vector_store )
			self.vector_store.delete( ids=ids )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'delete( self, ids ) -> None' )

	def clear( self ) -> None:
		"""Explicitly clear the connected collection."""
		try:
			throw_if( 'vector_store', self.vector_store )
			self.vector_store.reset_collection( )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'clear( self ) -> None' )

	def count( self ) -> int:
		"""Return the connected collection record count."""
		try:
			throw_if( 'vector_store', self.vector_store )
			return int( self.vector_store._collection.count( ) )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'count( self ) -> int' )

	def health( self ) -> Dict[ str, Any ]:
		"""Return connection and collection health information."""
		return { 'connected': self.vector_store is not None, 'collection': self.collection_name,
			'count': self.count( ) if self.vector_store is not None else 0 }

	def as_retriever( self, search_kwargs: Dict[ str, Any ]=None ) -> Any:
		"""Return a LangChain retriever for the connected collection."""
		try:
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.as_retriever( search_kwargs=search_kwargs or { } )
		except Exception as e:
			raise_logged_error( e, 'ChromaStore', 'as_retriever( self, **kwargs )' )


class PineconeStore( ):
	"""Managed Pinecone index lifecycle and retrieval wrapper."""

	def __init__( self ) -> None:
		self.documents: List[ Document ] = [ ]
		self.embedder: Embeddings | None = None
		self.index_name = ''
		self.namespace = ''
		self.api_key = ''
		self.client: Pinecone | None = None
		self.index: Any = None
		self.vector_store: PineconeVectorStore | None = None

	def create_index( self, index_name: str, dimension: int, api_key: str,
			metric: str='cosine', cloud: str='aws', region: str='us-east-1' ) -> None:
		"""Create a serverless index when it does not exist."""
		try:
			throw_if( 'index_name', index_name )
			throw_if( 'dimension', dimension )
			throw_if( 'api_key', api_key )
			self.api_key = str( api_key ).strip( )
			from pinecone import Pinecone, ServerlessSpec
			self.client = Pinecone( api_key=self.api_key )
			if not self.client.indexes.exists( str( index_name ).strip( ) ):
				self.client.create_index( name=str( index_name ).strip( ), dimension=int( dimension ),
					metric=str( metric ).strip( ), spec=ServerlessSpec( cloud=cloud, region=region ) )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'create_index( self, **kwargs )' )

	def connect( self, index_name: str, namespace: str, api_key: str,
			embedder: Embeddings ) -> PineconeVectorStore:
		"""Connect to an existing Pinecone index."""
		try:
			throw_if( 'index_name', index_name )
			throw_if( 'api_key', api_key )
			throw_if( 'embedder', embedder )
			self.index_name = str( index_name ).strip( )
			self.namespace = str( namespace or '' ).strip( )
			self.api_key = str( api_key ).strip( )
			self.embedder = embedder
			from langchain_pinecone import PineconeVectorStore
			from pinecone import Pinecone
			self.client = Pinecone( api_key=self.api_key )
			if not self.client.indexes.exists( self.index_name ):
				raise ValueError( f'Pinecone index does not exist: {self.index_name}' )
			self.index = self.client.index( name=self.index_name )
			self.vector_store = PineconeVectorStore( index=self.index, embedding=self.embedder,
				namespace=self.namespace or None )
			return self.vector_store
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'connect( self, **kwargs )' )

	def create( self, documents: List[ Document ], embedder: Embeddings, index_name: str,
			namespace: str, api_key: str ) -> PineconeVectorStore:
		"""Connect and add documents to an existing index."""
		try:
			self.connect( index_name=index_name, namespace=namespace, api_key=api_key,
				embedder=embedder )
			self.add_documents( documents=documents )
			return self.vector_store
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'create( self, **kwargs )' )

	def add_documents( self, documents: List[ Document ] ) -> List[ str ]:
		"""Embed and add documents to the connected index."""
		try:
			throw_if( 'documents', documents )
			throw_if( 'vector_store', self.vector_store )
			ids = [ str( ( document.metadata or { } ).get( 'chunk_id' ) )
				for document in documents ]
			return self.vector_store.add_documents( documents=documents, ids=ids )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'add_documents( self, documents )' )

	def add_embeddings( self, documents: List[ Document ],
			embeddings: List[ List[ float ] ] ) -> List[ str ]:
		"""Upsert precomputed embeddings without embedding documents again."""
		try:
			throw_if( 'documents', documents )
			throw_if( 'embeddings', embeddings )
			throw_if( 'index', self.index )
			if len( documents ) != len( embeddings ):
				raise ValueError( 'Embedding count must match document count.' )
			vectors: List[ Dict[ str, Any ] ] = [ ]
			ids: List[ str ] = [ ]
			for document, embedding in zip( documents, embeddings ):
				identifier = str( ( document.metadata or { } ).get( 'chunk_id' ) )
				metadata = scalar_metadata( document.metadata or { } )
				metadata[ 'text' ] = document.page_content or ''
				vectors.append( { 'id': identifier, 'values': embedding, 'metadata': metadata } )
				ids.append( identifier )
			self.index.upsert( vectors=vectors, namespace=self.namespace or None )
			return ids
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'add_embeddings( self, **kwargs )' )

	def similarity_search( self, query: str, count: int=4,
			filters: Dict[ str, Any ]=None ) -> List[ Document ]:
		"""Return the most similar documents for a query."""
		try:
			throw_if( 'query', query )
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.similarity_search( query=query, k=int( count ), filter=filters )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'similarity_search( self, **kwargs )' )

	def similarity_search_with_score( self, query: str, count: int=4,
			filters: Dict[ str, Any ]=None ) -> List[ tuple[ Document, float ] ]:
		"""Return similar documents and their scores."""
		try:
			throw_if( 'query', query )
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.similarity_search_with_score( query=query, k=int( count ),
				filter=filters )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'similarity_search_with_score( self, **kwargs )' )

	def get( self, ids: List[ str ] ) -> Any:
		"""Return selected Pinecone records by stable identifier."""
		try:
			throw_if( 'ids', ids )
			throw_if( 'index', self.index )
			return self.index.fetch( ids=ids, namespace=self.namespace or None )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'get( self, ids )' )

	def delete( self, ids: List[ str ] ) -> None:
		"""Delete selected vector identifiers."""
		try:
			throw_if( 'ids', ids )
			throw_if( 'index', self.index )
			self.index.delete( ids=ids, namespace=self.namespace or None )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'delete( self, ids )' )

	def clear( self ) -> None:
		"""Explicitly remove every vector in the selected namespace."""
		try:
			throw_if( 'index', self.index )
			self.index.delete( delete_all=True, namespace=self.namespace or None )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'clear( self )' )

	def count( self ) -> int:
		"""Return the selected namespace vector count."""
		try:
			throw_if( 'index', self.index )
			stats = self.index.describe_index_stats( )
			namespaces = getattr( stats, 'namespaces', { } ) or { }
			entry = namespaces.get( self.namespace or '', { } )
			return int( getattr( entry, 'vector_count', 0 ) if not isinstance( entry, dict )
				else entry.get( 'vector_count', 0 ) )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'count( self ) -> int' )

	def health( self ) -> Dict[ str, Any ]:
		"""Return connection and namespace health information."""
		return { 'connected': self.vector_store is not None, 'index': self.index_name,
			'namespace': self.namespace, 'count': self.count( ) if self.index is not None else 0 }

	def as_retriever( self, search_kwargs: Dict[ str, Any ]=None ) -> Any:
		"""Return a LangChain retriever for the connected index."""
		try:
			throw_if( 'vector_store', self.vector_store )
			return self.vector_store.as_retriever( search_kwargs=search_kwargs or { } )
		except Exception as e:
			raise_logged_error( e, 'PineconeStore', 'as_retriever( self, **kwargs )' )
