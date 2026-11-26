'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                data.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="data.py" company="Terry D. Eppler">

	     data.py
	     Copyright ©  2025  Terry Eppler

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

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    data.py

    Purpose:
        Provides database and vector-store support for the Foo application. The module
        contains shared validation, relational database helpers for SQLite-style data
        operations, provider path utilities, and a ChromaDB wrapper for persistent vector
        storage and semantic retrieval workflows.
  </summary>
  ******************************************************************************************
'''
import json
import numpy as np
import pandas as pd
import os
import sqlite3
from sqlite3 import Connection, Cursor
from typing import Optional, Any, List, Tuple
from boogr import Error, Logger
from boogr.enums import Source, Provider
import chromadb
from chromadb.config import Settings
import config as cfg


def throw_if( name: str, value: object ) -> None:
	"""Throw if.
    
        Purpose:
            Provides a input guard used by the Gipity Streamlit application. The function
            supports UI state management, provider coordination, data normalization, or display
            behavior required by the surrounding workflow.
    
        Args:
            name (str): Value supplied to the helper.
            value (object): Value supplied to the helper.
    
        Raises:
            Error: Re-raised after the exception is wrapped and written to the application logger.
    """
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, str ) and (not value.strip( )):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class DB( ):
	"""Provide database provider configuration helpers.
	
	Purpose:
		Stores common database provider metadata and resolves provider-specific driver,
		data-path, and connection-string values used by Foo database workflows. The class
		acts as a lightweight base for concrete database implementations.
	
	Attributes:
		provider (Optional[Provider]): Selected database provider enum value.
		source (Optional[Source]): Data source enum value associated with the database.
		table_name (Optional[str]): Active table name used by downstream operations.
		column_names (Optional[List[str]]): Active column names associated with a table.
		path (Optional[str]): Resolved provider data path.
		driver (Optional[str]): Provider driver string.
	"""
	provider: Optional[ Provider ]
	source: Optional[ Source ]
	table_name: Optional[ str ]
	column_names: Optional[ List[ str ] ]
	path: Optional[ str ]
	driver: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize database configuration state.
		
		Purpose:
			Initializes provider, source, table, path, and driver members to empty runtime
			state. Concrete database subclasses and property accessors populate these fields
			later when provider-specific database work is requested.
		"""
		self.provider = None
		self.source = None
		self.table_name = None
		self.path = None
		self.driver = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible database configuration members.
		
		Purpose:
			Provides a stable ordering of attributes and helper properties for interactive
			inspection, documentation surfaces, and UI components that expose database
			configuration details.
		
		Returns:
			Ordered member names exposed by the database base object.
		"""
		return [ 'source',
		         'provider',
		         'table_name',
		         'get_driver_info',
		         'path',
		         'adriver',
		         'access_path',
		         'get_data_path',
		         'get_connection_string' ]
	
	@property
	def driver_info( self ) -> str:
		"""Get provider driver information.
		
		Purpose:
			Returns the configured driver string for Access or SQL Server providers and falls
			back to the configured base directory for other provider values. The property also
			stores the selected driver on the instance for later connection-string assembly.
		
		Returns:
			Resolved provider driver string or configured base directory fallback.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.provider.name == 'Access':
				self.driver = cfg.ACCESS_DRIVER
				return self.driver
			elif self.provider.name == 'SqlServer':
				self.driver = cfg.SQLSERVER_DRIVER
				return self.driver
			else:
				return cfg.BASEDIR
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'data'
			_exc.cause = 'DB'
			_exc.method = 'getdriver_info( self )'
			Logger( ).write( _exc )
			raise _exc
	
	@property
	def data_path( self ) -> str:
		"""Get provider data path.
		
		Purpose:
			Builds the provider-specific storage path used by Foo database operations. The
			property maps SQLite, Access, and SQL Server provider values to their configured
			store directories and preserves a SQLite fallback for unsupported providers.
		
		Returns:
			Resolved provider data path.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.provider.name == 'SQLite':
				self.path = cfg.BASEDIR + r'\stores\sqlite'
				return self.path
			elif self.provider.name == 'Access':
				self.path = cfg.BASEDIR + r'\stores\access'
				return self.path
			elif self.provider.name == 'SqlServer':
				self.path = cfg.BASEDIR + r'\stores\sqlserver'
				return self.path
			else:
				self.path = cfg.BASEDIR + r'\stores\sqlite'
				return self.path
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'data'
			_exc.cause = 'DB'
			_exc.method = 'get_data_path( self )'
			Logger( ).write( _exc )
			raise _exc
	
	@property
	def connection_string( self ) -> str:
		"""Get provider connection string.
		
		Purpose:
			Constructs the connection string associated with the selected database provider.
			Access providers combine the configured Access driver with the resolved path,
			SQL Server providers build an attach-database connection string, and other
			providers return the resolved path fallback.
		
		Returns:
			Provider connection string or path fallback.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			_path = self.data_path
			if self.provider.name == Provider.Access.name:
				return self.driver_info + _path
			elif self.provider.name == Provider.SqlServer.name:
				return r'DRIVER={ ODBC Driver 17 for SQL Server };Server=.\SQLExpress;' \
					+ f'AttachDBFileName={_path}' \
					+ f'DATABASE={_path}Trusted_Connection=yes;'
			else:
				return f'{_path} '
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'data'
			_exc.cause = 'DB'
			_exc.method = 'get_connection_string( self )'
			Logger( ).write( _exc )
			raise _exc

class SQLite( DB ):
	"""Provide SQLite database operations.
	
	Purpose:
		Wraps a local SQLite connection and cursor with common table creation, insert,
		batch insert, query, update, delete, Excel import, and close operations. The class
		keeps the existing Foo database behavior while exposing a consistent object surface
		for data-management workflows.
	
	Attributes:
		db_path (Optional[str]): Path to the SQLite database file.
		connection (Optional[Connection]): Active SQLite connection.
		cursor (Optional[Cursor]): Active SQLite cursor.
		file_path (Optional[str]): Current file path used by import workflows.
		where (Optional[str]): Current SQL WHERE clause fragment.
		file_name (Optional[str]): Current source file name.
		table_name (Optional[str]): Current table name.
		placeholders (Optional[str]): Placeholder state used by insert operations.
		column_names (Optional[List[str]]): Active column names.
		params (Optional[Tuple]): SQL parameter tuple.
		tables (Optional[List]): Cached table list.
	"""
	db_path: Optional[ str ]
	connection: Optional[ Connection ]
	cursor: Optional[ Cursor ]
	file_path: Optional[ str ]
	where: Optional[ str ]
	file_name: Optional[ str ]
	table_name: Optional[ str ]
	placeholders: Optional[ str ]
	column_names: Optional[ List[ str ] ]
	params: Optional[ Tuple ]
	tables: Optional[ List ]
	
	def __init__( self ) -> None:
		"""Initialize SQLite connection state.
		
		Purpose:
			Initializes the SQLite wrapper with the default Foo data database path, opens the
			connection and cursor, and prepares runtime members used by subsequent CRUD,
			batch-insert, and import operations.
		"""
		self.db_path = r'stores\sqlite\datamodels\Data.db'
		self.connection = sqlite3.connect( self.db_path )
		self.cursor = self.connection.cursor( )
		self.file_path = None
		self.where = None
		self.pairs = None
		self.sql = None
		self.file_name = None
		self.table_name = None
		self.placeholders = ''
		self.params = ( )
		self.column_names = [ ]
		self.tables = [ ]
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible SQLite members.
		
		Purpose:
			Provides a stable ordering of SQLite wrapper attributes and operations for
			interactive inspection, documentation, and UI display surfaces.
		
		Returns:
			Ordered member names exposed by the SQLite wrapper.
		"""
		return [ 'db_path', 'connection', 'cursor', 'path', 'where', 'pairs', 'sql', 'file_name',
			'table_name', 'placeholders', 'columns', 'params', 'column_names', 'tables', 'close',
			'import_excel', 'delete', 'update', 'insert', 'create_table', 'fetch_one',
			'fetch_all' ]
	
	def create( self ) -> None:
		"""Create the embeddings table.
		
		Purpose:
			Creates the default ``embeddings`` table when it does not already exist. The
			table stores source-file names, chunk indexes, chunk text, serialized embedding
			vectors, and creation timestamps for downstream retrieval workflows.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.cursor.execute( """
                                 CREATE TABLE IF NOT EXISTS embeddings
                                 (
                                     id
                                     INTEGER
                                     PRIMARY
                                     KEY
                                     AUTOINCREMENT,
                                     source_file
                                     TEXT
                                     NOT
                                     NULL,
                                     chunk_index
                                     INTEGER
                                     NOT
                                     NULL,
                                     chunk_text
                                     TEXT
                                     NOT
                                     NULL,
                                     embedding
                                     TEXT
                                     NOT
                                     NULL,
                                     created_at
                                     TEXT
                                     DEFAULT
                                     CURRENT_TIMESTAMP
                                 )""" )
			
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'create( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def create_table( self, sql: str ) -> None:
		"""Create a table from SQL.
		
		Purpose:
			Executes a caller-provided SQL table-creation statement against the active SQLite
			connection and commits the result. The method retains the SQL statement on the
			instance for inspection and preserves the existing direct-SQL behavior.
		
		Args:
			sql (str): SQL statement to execute.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'sql', sql )
			self.sql = sql
			self.cursor.execute( self.sql )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'create_table( self, sql: str ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def insert( self, table: str, columns: List[ str ], values: Tuple[ Any, ... ] ) -> None:
		"""Insert one record.
		
		Purpose:
			Builds and executes an INSERT statement for the specified table, column list, and
			value tuple. The method uses positional SQLite placeholders for values while
			preserving the caller-provided table and column naming behavior.
		
		Args:
			table (str): Target table name.
			columns (List[str]): Column names receiving values.
			values (Tuple[Any, ...]): Values to insert into the target columns.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'table', table )
			throw_if( 'columns', columns )
			throw_if( 'values', values )
			self.placeholders = ', '.join( '?' for _ in values )
			col_names = ', '.join( columns )
			self.sql = f'INSERT INTO {table} ({col_names}) VALUES ({self.placeholders})'
			self.cursor.execute( self.sql, values )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = ('insert( self, **kwargs ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def insert_many( self, source_file: str, chunks: List[ str ], vectors: np.ndarray ) -> None:
		"""Insert multiple embedding records.
		
		Purpose:
			Serializes a batch of embedding vectors and inserts them with their source file,
			chunk index, and chunk text into the active table. This supports bulk persistence
			of text chunks and embeddings generated by ingestion workflows.
		
		Args:
			source_file (str): Name or path of the source document.
			chunks (List[str]): Cleaned text chunks corresponding to the embeddings.
			vectors (np.ndarray): Matrix of embedding vectors to serialize and store.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'source_file', source_file )
			throw_if( 'chunks', chunks )
			throw_if( 'vectors', vectors )
			records = [ (source_file, i, chunks[ i ], json.dumps( vectors[ i ].tolist( ) ))
			            for i in range( len( chunks ) ) ]
			
			self.sql = f''' INSERT INTO {self.table_name} ({self.file_name}, chunk_index,
					chunk_text, embedding) VALUES (?, ?, ?, ?) '''
			self.cursor.executemany( self.sql, records )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'insert_many'
			Logger( ).write( exception )
			raise exception
	
	def fetch_all( self, table: str ) -> List[ Tuple ] | None:
		"""Fetch all rows from a table.
		
		Purpose:
			Executes a SELECT statement against the specified table and returns all rows from
			the active cursor. The method keeps the generated SQL on the instance for later
			inspection or debugging.
		
		Args:
			table (str): Table name to query.
		
		Returns:
			Rows returned by the SQLite cursor, or ``None`` if the wrapped error path is raised.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'table', table )
			self.sql = f'SELECT * FROM {table}'
			self.cursor.execute( self.sql )
			return self.cursor.fetchall( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'fetch_all( self, df: str ) -> List[ Tuple ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_one( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> Tuple | None:
		"""Fetch one matching row.
		
		Purpose:
			Builds a SELECT statement with a caller-provided WHERE clause, executes it with
			the stored parameter tuple, and returns the first matching row from the active
			cursor.
		
		Args:
			table (str): Table name to query.
			where (str): WHERE clause fragment without the ``WHERE`` keyword.
			params (Tuple[Any, ...]): Parameters intended for the WHERE clause.
		
		Returns:
			First matching row from the cursor, or ``None`` when no row is found.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f'SELECT * FROM {self.table_name} WHERE {self.where} LIMIT 1'
			self.cursor.execute( self.sql, self.params )
			return self.cursor.fetchone( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'fetch_one( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def update( self, table: str, pairs: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""Update matching rows.
		
		Purpose:
			Executes an UPDATE statement using the provided table name, SET clause fragment,
			WHERE clause fragment, and SQLite parameter tuple. The method commits the active
			connection after the update is executed.
		
		Args:
			table (str): Table name to update.
			pairs (str): SET clause fragment containing target assignments.
			where (str): WHERE clause fragment without the ``WHERE`` keyword.
			params (Tuple[Any, ...]): Parameters for the SET and WHERE clauses.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'pairs', pairs )
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f'UPDATE {self.table_name} SET {pairs} WHERE {self.where}'
			self.cursor.execute( self.sql, self.params )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'update( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""Delete matching rows.
		
		Purpose:
			Executes a DELETE statement against the specified table using the provided WHERE
			clause and parameter tuple. The method commits the active SQLite connection after
			the delete operation completes.
		
		Args:
			table (str): Table name to delete from.
			where (str): WHERE clause fragment without the ``WHERE`` keyword.
			params (Tuple[Any, ...]): Parameters for the WHERE clause.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'where', where )
			throw_if( 'table', table )
			throw_if( 'params', params )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f"DELETE FROM {self.table_name} WHERE {self.where}"
			self.cursor.execute( self.sql, self.params )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'delete( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def import_excel( self, path: str ) -> None:
		"""Import worksheets from an Excel workbook.
		
		Purpose:
			Reads every worksheet from an Excel workbook into pandas DataFrames and writes
			each worksheet into the active SQLite database using the sheet name as the table
			name. Existing tables with the same sheet names are replaced.
		
		Args:
			path (str): Path to the Excel workbook.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_name = os.path.basename( self.file_path )
			_excel = pd.ExcelFile( self.file_path )
			for _sheet in _excel.sheet_names:
				_df = _excel.parse( _sheet )
				_df.to_sql( _sheet, self.connection, if_exists='replace', index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'import_excel( self, path: str ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def close( self ) -> None:
		"""Close the active database connection.
		
		Purpose:
			Closes the SQLite connection when it exists. The method preserves the current
			defensive behavior by doing nothing when the connection member is already empty.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.connection is not None:
				self.connection.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'close( self ) -> None'
			Logger( ).write( exception )
			raise exception

class Chroma:
	"""Provide persistent ChromaDB vector storage.
	
	Purpose:
		Wraps a ChromaDB client and collection for persistent embedding storage, similarity
		querying, deletion, record counting, collection clearing, and persistence. The class
		provides Foo vector workflows with a compact interface over the underlying ChromaDB
		collection operations.
	
	Attributes:
		client (Optional[chromadb.Client]): Instantiated ChromaDB client.
		collection (Optional[chromadb.Collection]): Vector collection used for add, query,
			delete, count, clear, and persistence operations.
	"""
	client: Optional[ chromadb.Client ]
	collection: Optional[ chromadb.Collection ]
	
	def __init__( self, path: str = './chroma', collection: str = 'embeddings' ) -> None:
		"""Initialize Chroma vector storage.
		
		Purpose:
			Initializes a ChromaDB client using the supplied persistence path and retrieves or
			creates the named collection. This prepares the vector-store wrapper for document,
			embedding, metadata, and semantic query operations.
		
		Args:
			path (str): Directory used by ChromaDB for persistent storage.
			collection (str): Name of the ChromaDB collection to retrieve or create.
		"""
		self.client = chromadb.Client(
			Settings( persist_directory=path, anonymized_telemetry=False ) )
		self.collection = self.client.get_or_create_collection( name=collection )
	
	def add( self, ids: List[ str ], texts: List[ str ], embeddings: List[ List[ float ] ],
			metadatas: Optional[ List[ dict ] ] = None ) -> None:
		"""Add records to the vector collection.
		
		Purpose:
			Adds document text, vector embeddings, unique identifiers, and optional metadata to
			the configured ChromaDB collection. This method is the primary persistence path for
			new vector-search records.
		
		Args:
			ids (List[str]): Unique identifiers for each vector-store record.
			texts (List[str]): Document strings associated with the embeddings.
			embeddings (List[List[float]]): Vector representations for the supplied documents.
			metadatas (Optional[List[dict]]): Optional metadata dictionaries for filtering or tagging.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.collection.add( documents=texts, embeddings=embeddings, ids=ids,
				metadatas=metadatas )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'add( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def query( self, text: List[ str ], num: int = 5, where: Optional[ dict ] = None ) -> List[
		                                                                                      str ] | None:
		"""Query similar vector-store records.
		
		Purpose:
			Performs a ChromaDB similarity search using one or more query texts, a requested
			result count, and an optional metadata filter. The method returns the first group
			of matching documents from the ChromaDB query result.
		
		Args:
			text (List[str]): Query text values submitted to ChromaDB.
			num (int): Number of top matches to request.
			where (Optional[dict]): Optional metadata filter applied to the query.
		
		Returns:
			Most relevant document strings returned by the vector search.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			result = self.collection.query( query_texts=text, n_results=num, where=where or { } )
			return result.get( 'documents', [ ] )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'query( )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, ids: List[ str ] ) -> None:
		"""Delete vector-store records.
		
		Purpose:
			Deletes one or more records from the configured ChromaDB collection by document ID.
			This supports targeted cleanup of obsolete or incorrectly indexed vector records.
		
		Args:
			ids (List[str]): Unique document IDs to delete from the collection.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.collection.delete( ids=ids )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'delete( )'
			Logger( ).write( exception )
			raise exception
	
	def count( self ) -> int | None:
		"""Count records in the vector collection.
		
		Purpose:
			Returns the total number of records stored in the configured ChromaDB collection.
			The value can be used by diagnostics, UI summaries, and validation checks.
		
		Returns:
			Number of records in the ChromaDB collection.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			return self.collection.count( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'count( )'
			Logger( ).write( exception )
			raise exception
	
	def clear( self ) -> None:
		"""Clear the vector collection.
		
		Purpose:
			Deletes all documents from the configured ChromaDB collection using the existing
			collection-wide delete behavior. This supports resetting local vector-store state
			for rebuild or testing workflows.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.collection.delete( where={ } )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'clear( )'
			Logger( ).write( exception )
			raise exception
	
	def persist( self ) -> None:
		"""Persist vector-store state.
		
		Purpose:
			Calls the ChromaDB client persistence operation to save the current collection
			state to disk. This preserves vector-store changes for later application runs.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client.persist( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'persist'
			Logger( ).write( exception )
			raise exception
			
