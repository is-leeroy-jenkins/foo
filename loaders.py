'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                laoders.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loaders.py" company="Terry D. Eppler">

	     loaders.py
	     Copyright ©  2022  Terry Eppler

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
    loaders.py — LangChain document loaders, source adapters, and chunking utilities.

    Purpose:
        Provides Foo's document-ingestion layer for local files, web pages, scholarly sources,
        collaboration platforms, cloud object stores, email, notebooks, and speech content.
        The module resolves source paths, converts content into LangChain Document objects,
        preserves source metadata, and divides loaded content into overlapping chunks for
        retrieval, embedding, and retrieval-augmented generation workflows.
  </summary>
  ******************************************************************************************
'''
import arxiv
import docx2txt

from boogr import Error, Logger
import config as cfg
import glob
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (CSVLoader, Docx2txtLoader, PyPDFLoader,
                                                  JSONLoader, GithubFileLoader,
                                                  UnstructuredExcelLoader, RecursiveUrlLoader,
                                                  WebBaseLoader, YoutubeLoader, ArxivLoader,
                                                  WikipediaLoader, UnstructuredEmailLoader,
                                                  SharePointLoader, GoogleDriveLoader,
                                                  UnstructuredPowerPointLoader,
                                                  OutlookMessageLoader, OneDriveLoader,
                                                  UnstructuredXMLLoader, PubMedLoader,
                                                  OpenCityDataLoader, NotebookLoader,
                                                  S3FileLoader, )

from langchain_google_community import (GCSFileLoader, SpeechToTextLoader)
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_google_community import GCSDirectoryLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import os
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
import wikipedia
from lxml import etree

def throw_if( name: str, value: object ) -> None:
	"""Throw if.

	Purpose:
	    Validates that a required argument contains a usable value so failures occur before provider, filesystem, or parsing work begins.

	Args:
	    name (str): Argument name included in validation error messages.
	    value (object): Candidate value to validate or normalize.

	Returns:
	    None: This method updates instance state or validates input and does not return a value.

	Raises:
	    ValueError: Raised when the method cannot satisfy its documented value requirement.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and (not value.strip( )):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Loader( ):
	"""Loader component.

	Purpose:
	    Defines shared file resolution, document loading, and chunking behavior for LangChain document loaders.

	Attributes:
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    pattern (Optional[str]): Path pattern, delimiter, or matching expression used by the loader.
	    expanded (Optional[List[str]]): Expanded path expressions evaluated during source resolution.
	    candidates (Optional[List[str]]): Candidate source paths collected before existence checks.
	    resolved (Optional[List[str]]): Existing source paths resolved from candidate paths or glob patterns.
	    loader (Optional[BaseLoader]): Concrete LangChain loader configured for the selected source.
	    splitter (Optional[RecursiveCharacterTextSplitter | CharacterTextSplitter]): Text splitter used to divide documents into retrieval-sized chunks.
	    chunk_size (Optional[int]): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Optional[int]): Number of tokens or characters repeated between adjacent chunks.
	"""
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ BaseLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def verify_exists( self, path: str ) -> str | None:
		"""Verify exists.

		Purpose:
		    Resolves a required file path and raises an explicit error when the file does not exist.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			if not os.path.isfile( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			else:
				self.file_path = path
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
		"""Resolve paths.

		Purpose:
		    Expands a path or glob expression into a sorted collection of unique existing files.

		Args:
		    pattern (str): Pattern supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'pattern', pattern )
			self.candidates.append( pattern )
			for p in self.candidates:
				if os.path.isfile( p ):
					self.resolved.append( p )
				else:
					for m in glob.glob( p ):
						if os.path.isfile( m ):
							self.resolved.append( m )
			
			if not self.resolved:
				raise FileNotFoundError( f'No files matched or existed for input: {pattern}' )
			return sorted( set( self.resolved ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = 'resolve_paths( self, pattern: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def load_documents( self, path: str, encoding: Optional[ str ],
		csv_args: Optional[ Dict[ str, Any ] ], source_column: Optional[ str ] ) -> List[ Document ]:
		"""Load documents.

		Purpose:
		    Loads a source file through the configured LangChain loader and returns its document objects.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    encoding (Optional[str]): Text encoding used when reading the source file.
		    csv_args (Optional[Dict[str, Any]]): CSV parser options forwarded to the LangChain loader.
		    source_column (Optional[str]): Column whose value is recorded as document source metadata.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = BaseLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			Logger( ).write( exception )
			raise exception
	
	def split_documents( self, docs: List[ Document ], chunk: int=1000,
		overlap: int=200 ) -> List[ Document ]:
		"""Split documents.

		Purpose:
		    Divides LangChain documents into overlapping token-aware chunks suitable for retrieval and embedding.

		Args:
		    docs (List[Document]): LangChain documents to split into smaller retrieval units.
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
				model_name='gpt-4o', chunk_size=self.chunk_size, overlap=self.overlap_amount )
			return self.splitter.split_documents( documents=self.documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, **kwargs ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception

class TextLoader( Loader ):
	"""TextLoader component.

	Purpose:
	    Loads UTF-8 text files into LangChain documents and supports token-based or character-based chunking.

	Attributes:
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    splitter (Optional[RecursiveCharacterTextSplitter | CharacterTextSplitter]): Text splitter used to divide documents into retrieval-sized chunks.
	    raw_text (Optional[str]): Unmodified text read from the current source file.
	    separator (Optional[str]): Preferred boundary used by character-based splitting.
	    length_function (Optional[object]): Callable used by the splitter to measure candidate chunks.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	raw_text: Optional[ str ]
	separator: Optional[ str ]
	length_function: Optional[ object ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.raw_text = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.separator = "\n\n"
		self.length_function = len
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'documents', 'splitter', 'pattern', 'file_path', 'expanded', 'candidates',
			'resolved', 'chunk_size', 'overlap_amount', 'raw_text', 'separator', 'length_function',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'split_tokens',
			'split_chars', ]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    filepath (str): Filesystem path of the source document.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			
			with open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ) as handle:
				self.raw_text = handle.read( )
			
			self.documents = [ Document( page_content=self.raw_text if isinstance(
				self.raw_text, str ) else '', metadata={ 'source': os.path.basename(
				self.file_path ), 'loader':  'TextLoader', 'path': self.file_path, } ) ]
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'load( self, filepath: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_tokens( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None:
		"""Split tokens.

		Purpose:
		    Split tokens using the class state and returns data required by the surrounding workflow.

		Args:
		    size (int): Maximum chunk size used by the text splitter.
		    amount (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
				encoding_name='cl100k_base', chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount )
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'tokens'
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = ('split_tokens( self, size: int=1000, amount: int=200 ) -> List[ '
			                    'Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split_chars( self, size: int=1000, amount: int=200, seps: str="\n\n" ) -> List[ Document ]:
		"""Split chars.

		Purpose:
		    Split chars using the class state and returns data required by the surrounding workflow.

		Args:
		    size (int): Maximum chunk size used by the text splitter.
		    amount (int): Number of characters or tokens repeated between adjacent chunks.
		    seps (str): Separator string used to identify preferred character boundaries.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.separator = seps
			self.splitter = CharacterTextSplitter( separator=self.separator,
				chunk_size=self.chunk_size, chunk_overlap=self.overlap_amount,
				length_function=self.length_function )
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'chars'
			
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'split_chars( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class CsvLoader( Loader ):
	"""CsvLoader component.

	Purpose:
	    Loads delimited records into LangChain documents with configurable columns, delimiters, and quote characters.

	Attributes:
	    loader (Optional[CSVLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    splitter (Optional[RecursiveCharacterTextSplitter]): Text splitter used to divide documents into retrieval-sized chunks.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    quote_char (Optional[str]): Current quote char retained by the CsvLoader workflow between related operations.
	    csv_args (Optional[Dict[str, Any]]): Current csv args retained by the CsvLoader workflow between related operations.
	    columns (Optional[List[str]]): Current columns retained by the CsvLoader workflow between related operations.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	file_path: Optional[ str ]
	quote_char: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	columns: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.columns = None
		self.csv_args = None
		self.documents = None
		self.quote_char = '"'
		self.pattern = ','
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'delimiter', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', 'csv_args', 'columns', ]
	
	def load( self, filepath: str, columns: Optional[ List[ str ] ] = None, delimiter: str=',',
		quotechar: str='"' ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    filepath (str): Filesystem path of the source document.
		    columns (Optional[List[str]]): CSV columns included as document content.
		    delimiter (str): Character separating fields in the delimited file.
		    quotechar (str): Character enclosing quoted fields in the delimited file.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.columns = columns
			self.pattern = delimiter if isinstance( delimiter, str ) and delimiter else ','
			self.quote_char = quotechar if isinstance( quotechar, str ) and quotechar else '"'
			self.csv_args = { 'delimiter': self.pattern, 'quotechar': self.quote_char, }
			
			if isinstance( self.columns, list ) and self.columns:
				self.csv_args[ 'fieldnames' ] = self.columns
			
			self.loader = CSVLoader( file_path=self.file_path, csv_args=self.csv_args,
				content_columns=self.columns, )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = ('load( self, filepath: str, columns: Optional[ List[ str ] '
			                    ']=None, '
			                    'delimiter: str=",", quotechar: str=\'"\' ) -> List[ Document ] | '
			                    'None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    size (int): Maximum chunk size used by the text splitter.
		    amount (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = ('split( self, **args ) ->  List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class XmlLoader( Loader ):
	"""XmlLoader component.

	Purpose:
	    Loads XML as LangChain document elements and exposes XPath-based tree inspection.

	Attributes:
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    loader (Optional[UnstructuredXMLLoader]): Concrete LangChain loader configured for the selected source.
	    splitter (Optional[RecursiveCharacterTextSplitter]): Text splitter used to divide documents into retrieval-sized chunks.
	    chunk_size (Optional[int]): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Optional[int]): Number of tokens or characters repeated between adjacent chunks.
	    xml_tree (Optional[etree._ElementTree]): Current xml tree retained by the XmlLoader workflow between related operations.
	    xml_root (Optional[etree._Element]): Current xml root retained by the XmlLoader workflow between related operations.
	    xml_namespaces (Optional[Dict[str, str]]): Current xml namespaces retained by the XmlLoader workflow between related operations.
	"""
	
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	loader: Optional[ UnstructuredXMLLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	xml_tree: Optional[ etree._ElementTree ]
	xml_root: Optional[ etree._Element ]
	xml_namespaces: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.loader = None
		self.splitter = None
		self.chunk_size = None
		self.overlap_amount = None
		self.xml_tree = None
		self.xml_root = None
		self.xml_namespaces = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'file_path', 'expanded', 'candidates',
			'resolved', 'chunk_size', 'overlap_amount', 'xml_tree', 'xml_root', 'xml_namespaces',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'split', 'load_tree',
			'get_elements', ]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    filepath (str): Filesystem path of the source document.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			self.file_path = self.verify_exists( filepath )
			self.loader = UnstructuredXMLLoader( file_path=self.file_path, mode='elements' )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'XmlLoader'
			exception.method = 'load(self, filepath: str)'
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    size (int): Maximum chunk size used by the text splitter.
		    amount (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded via load().' )
			self.chunk_size = size
			self.overlap_amount = amount
			split_docs = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return split_docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'XmlLoader'
			exception.method = 'split(self, size: int=1000, amount: int=200)'
			Logger( ).write( exception )
			raise exception
	
	def load_tree( self, filepath: str ) -> etree._ElementTree | None:
		"""Load tree.

		Purpose:
		    Loads tree into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    filepath (str): Filesystem path of the source document.

		Returns:
		    etree._ElementTree | None: Provider, loader, or normalized application value produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			self.file_path = self.verify_exists( filepath )
			parser = etree.XMLParser( recover=True, remove_comments=True, remove_blank_text=True )
			self.xml_tree = etree.parse( self.file_path, parser )
			self.xml_root = self.xml_tree.getroot( )
			self.xml_namespaces = { prefix if prefix is not None else 'default': uri for prefix,
			uri
				in (self.xml_root.nsmap or { }).items( ) }
			
			return self.xml_tree
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'XmlLoader'
			exception.method = 'load_tree(self, filepath: str)'
			Logger( ).write( exception )
			raise exception
	
	def get_elements( self, xpath: str ) -> List[ etree._Element ] | None:
		"""Get elements.

		Purpose:
		    Get elements using the class state and returns data required by the surrounding workflow.

		Args:
		    xpath (str): Xpath supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[etree._Element] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.xml_root is None:
				raise ValueError( 'XML tree not loaded. Call load_tree() first.' )
			elements = self.xml_root.xpath( xpath, namespaces=self.xml_namespaces )
			return list( elements )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'XmlLoader'
			exception.method = 'get_elements(self, xpath: str)'
			Logger( ).write( exception )
			raise exception

class WebLoader( Loader ):
	"""WebLoader component.

	Purpose:
	    Loads individual web pages or recursively follows links and converts retrieved content into LangChain documents.

	Attributes:
	    loader (Optional[RecursiveUrlLoader | WebBaseLoader]): Concrete LangChain loader configured for the selected source.
	    url (Optional[str]): Most recent endpoint or resource URL used by the instance.
	    web_paths (Optional[str | List[str]]): Current web paths retained by the WebLoader workflow between related operations.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    max_depth (Optional[int]): Upper bound applied to depth.
	    timeout (Optional[int]): Maximum request duration, in seconds, applied to provider calls.
	    ignore (Optional[bool]): Current ignore retained by the WebLoader workflow between related operations.
	    with_progress (Optional[bool]): Current with progress retained by the WebLoader workflow between related operations.
	    recursive (Optional[bool]): Current recursive retained by the WebLoader workflow between related operations.
	    prevent_outside (Optional[bool]): Current prevent outside retained by the WebLoader workflow between related operations.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, recursive: bool=False, max_depth: int=2, prevent_outside: bool=True,
		timeout: int=10, ignore: bool=True, progress: bool=True ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Args:
		    recursive (bool): Whether loading follows links or descends into child resources.
		    max_depth (int): Maximum number of link levels traversed from the starting page.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.url = None
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'url', 'web_paths',
			'max_depth', 'timeout', 'ignore', 'with_progress', 'recursive', 'prevent_outside',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'load_pages', 'split', ]
	
	def load( self, urls: str | List[ str ], depth: int=2, timeout: int=10, ignore: bool=True,
		progress: bool=True, prevent_outside: bool=True ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    depth (int): Maximum number of link levels traversed from the starting page.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.recursive:
				return self.load_recursive( urls=urls, depth=depth, timeout=timeout, ignore=ignore,
					prevent_outside=prevent_outside )
			
			return self.load_pages( urls=urls, timeout=timeout, ignore=ignore, progress=progress )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'load( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int=10, ignore: bool=True,
		progress: bool=True ) -> List[ Document ] | None:
		"""Load pages.

		Purpose:
		    Loads pages into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'urls', urls )
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			self.loader = WebBaseLoader( web_paths=self.web_paths,
				show_progress=self.with_progress, continue_on_failure=self.ignore,
				requests_kwargs={ 'timeout': self.timeout } )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'load_pages( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int=2, timeout: int=10,
		ignore: bool=True, prevent_outside: bool=True ) -> List[ Document ] | None:
		"""Load recursive.

		Purpose:
		    Loads recursive into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    depth (int): Maximum number of link levels traversed from the starting page.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'urls', urls )
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			self.loader = RecursiveUrlLoader( url=self.url, max_depth=self.max_depth,
				timeout=self.timeout, continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'load_recursive( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'split( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class PdfLoader( Loader ):
	"""PdfLoader component.

	Purpose:
	    Loads PDF text with configurable page aggregation, extraction layout, image extraction, and document chunking.

	Attributes:
	    loader (Optional[PyPDFLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    mode (Optional[str]): Current mode retained by the PdfLoader workflow between related operations.
	    extraction (Optional[str]): Current extraction retained by the PdfLoader workflow between related operations.
	    include_images (Optional[bool]): Flag controlling whether include images behavior is enabled.
	    image_format (Optional[str]): Current image format retained by the PdfLoader workflow between related operations.
	    custom_delimiter (Optional[str]): Current custom delimiter retained by the PdfLoader workflow between related operations.
	    image_parser (Optional[RapidOCRBlobParser]): Current image parser retained by the PdfLoader workflow between related operations.
	    enable_tables (Any): Flag controlling whether enable tables behavior is enabled.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	extraction: Optional[ str ]
	include_images: Optional[ bool ]
	image_format: Optional[ str ]
	custom_delimiter: Optional[ str ]
	image_parser: Optional[ RapidOCRBlobParser ]
	
	def __init__( self, size: int=1000, overlap: int=150, has_tables: bool=True,
		include: bool=True ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Args:
		    size (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.
		    has_tables (bool): Whether has tables behavior is enabled for the operation.
		    include (bool): Whether optional embedded content, such as images or outputs, is included.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.enable_tables = has_tables
		self.include_images = include
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = size
		self.overlap_amount = overlap
		self.loader = None
		self.mode = None
		self.extraction = None
		self.image_format = None
		self.custom_delimiter = None
		self.image_parser = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'mode', 'extraction',
			'include_images', 'image_format', 'custom_delimiter', 'image_parser', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', 'mode_options',
			'extraction_options', 'image_options', ]
	
	@property
	def mode_options( self ) -> List[ str ]:
		"""Mode options.

		Purpose:
		    Returns supported mode choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'page', 'single' ]
	
	@property
	def extraction_options( self ) -> List[ str ]:
		"""Extraction options.

		Purpose:
		    Returns supported extraction choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'plain', 'layout' ]
	
	@property
	def image_options( self ) -> List[ str ]:
		"""Image options.

		Purpose:
		    Returns supported image choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'html-img', 'markdown-img', 'text-img' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize mode.

		Purpose:
		    Normalizes mode into the canonical representation expected by the surrounding workflow.

		Args:
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'elements':
			return 'page'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def _normalize_extraction( self, extract: str ) -> str:
		"""Normalize extraction.

		Purpose:
		    Normalizes extraction into the canonical representation expected by the surrounding workflow.

		Args:
		    extract (str): PDF text-extraction strategy selected by the caller.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = extract.strip( ).lower( ) if isinstance( extract, str ) else 'plain'
		
		if value == 'ocr':
			return 'layout'
		
		if value not in self.extraction_options:
			return 'plain'
		
		return value
	
	def _normalize_image_format( self, format: str ) -> str:
		"""Normalize image format.

		Purpose:
		    Normalizes image format into the canonical representation expected by the surrounding workflow.

		Args:
		    format (str): Output or embedded-content format selected by the caller.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = format.strip( ).lower( ) if isinstance( format, str ) else 'markdown-img'
		if value == 'text':
			return 'markdown-img'
		
		if value not in self.image_options:
			return 'markdown-img'
		
		return value
	
	def load( self, filepath: str, mode: str='single', extract: str='plain',
		include: bool=False, format: str='markdown-img' ) -> List[ Document ]:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    filepath (str): Filesystem path of the source document.
		    mode (str): Provider or loader operating mode selected for the request.
		    extract (str): PDF text-extraction strategy selected by the caller.
		    include (bool): Whether optional embedded content, such as images or outputs, is included.
		    format (str): Output or embedded-content format selected by the caller.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', filepath )
			self.file_path = self.verify_exists( filepath )
			self.mode = self._normalize_mode( mode )
			self.extraction = self._normalize_extraction( extract )
			self.include_images = include
			self.image_format = self._normalize_image_format( format )
			if self.include_images:
				self.image_parser = RapidOCRBlobParser( )
				self.loader = PyPDFLoader( file_path=self.file_path, mode=self.mode,
					extraction_mode=self.extraction, extract_images=self.include_images,
					images_inner_format=self.image_format, images_parser=self.image_parser )
			else:
				self.loader = PyPDFLoader( file_path=self.file_path, mode=self.mode,
					extraction_mode=self.extraction )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = 'load( self, **kwars ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = ('split( self, **kwargs ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class ExcelLoader( Loader ):
	"""ExcelLoader component.

	Purpose:
	    Loads spreadsheet content into LangChain documents using the selected Unstructured partitioning mode.

	Attributes:
	    loader (Optional[UnstructuredExcelLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    mode (Optional[str]): Current mode retained by the ExcelLoader workflow between related operations.
	    has_headers (Optional[bool]): Flag controlling whether has headers behavior is enabled.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
		self.has_headers = True
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'mode', 'has_headers',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'split', 'mode_options', ]
	
	@property
	def mode_options( self ) -> List[ str ]:
		"""Mode options.

		Purpose:
		    Returns supported mode choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize mode.

		Purpose:
		    Normalizes mode into the canonical representation expected by the surrounding workflow.

		Args:
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str='single', has_headers: bool=True ) -> List[ Document ]:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    mode (str): Provider or loader operating mode selected for the request.
		    has_headers (bool): Whether has headers behavior is enabled for the operation.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.has_headers = has_headers
			self.loader = UnstructuredExcelLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = 'load( self, , **kwars  ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class WordLoader( Loader ):
	"""WordLoader component.

	Purpose:
	    Loads Microsoft Word documents into LangChain documents and supports subsequent chunking.

	Attributes:
	    loader (Optional[Docx2txtLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_splits = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'split( self, **kwars  ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class MarkdownLoader( Loader ):
	"""MarkdownLoader component.

	Purpose:
	    Loads Markdown content using configurable document partitioning and supports subsequent chunking.

	Attributes:
	    loader (Optional[UnstructuredMarkdownLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (str | None): Resolved filesystem path of the current source or output file.
	    documents (List[Document] | None): LangChain documents loaded or produced by the most recent operation.
	    mode (Optional[str]): Current mode retained by the MarkdownLoader workflow between related operations.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'mode', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', 'mode_options', ]
	
	@property
	def mode_options( self ) -> List[ str ]:
		"""Mode options.

		Purpose:
		    Returns supported mode choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize mode.

		Purpose:
		    Normalizes mode into the canonical representation expected by the surrounding workflow.

		Args:
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredMarkdownLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = ('load( self, path: str, mode: str="single" ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class HtmlLoader( Loader ):
	"""HtmlLoader component.

	Purpose:
	    Loads HTML files into LangChain documents and supports content chunking for retrieval workflows.

	Attributes:
	    loader (Optional[UnstructuredHTMLLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (str | None): Resolved filesystem path of the current source or output file.
	    documents (List[Document] | None): LangChain documents loaded or produced by the most recent operation.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HtmlLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class JsonLoader( Loader ):
	"""JsonLoader component.

	Purpose:
	    Loads JSON records into LangChain documents using a configurable jq schema and text-content behavior.

	Attributes:
	    loader (Optional[JSONLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    jq_schema (Optional[str]): Current jq schema retained by the JsonLoader workflow between related operations.
	    content_key (Optional[str]): Current content key retained by the JsonLoader workflow between related operations.
	    text_content (Optional[bool]): Current text content retained by the JsonLoader workflow between related operations.
	    json_lines (Optional[bool]): Current json lines retained by the JsonLoader workflow between related operations.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ JSONLoader ]
	file_path: Optional[ str ]
	jq_schema: Optional[ str ]
	content_key: Optional[ str ]
	text_content: Optional[ bool ]
	json_lines: Optional[ bool ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.jq_schema = '.'
		self.content_key = None
		self.text_content = True
		self.json_lines = False
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'jq_schema', 'content_key',
			'text_content', 'json_lines', 'verify_exists', 'resolve_paths', 'split_documents',
			'load', 'split', ]
	
	def load( self, filepath: str, jq_schema: str='.', content_key: Optional[ str ] = None,
		is_text: bool=True, is_lines: bool=False ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    filepath (str): Filesystem path of the source document.
		    jq_schema (str): Jq schema supplied by the caller and interpreted according to the method contract.
		    content_key (Optional[str]): Content key supplied by the caller and interpreted according to the method contract.
		    is_text (bool): Whether is text behavior is enabled for the operation.
		    is_lines (bool): Whether is lines behavior is enabled for the operation.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.jq_schema = jq_schema if isinstance( jq_schema,
				str ) and jq_schema.strip( ) else '.'
			self.content_key = (content_key.strip( ) if isinstance( content_key,
				str ) and content_key.strip( ) else None)
			self.text_content = bool( is_text )
			self.json_lines = bool( is_lines )
			kwargs = { 'file_path': self.file_path, 'jq_schema': self.jq_schema,
				'text_content': self.text_content, 'json_lines': self.json_lines, }
			
			if self.content_key:
				kwargs[ 'content_key' ] = self.content_key
			
			self.loader = JSONLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = 'load( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class ArXivLoader( Loader ):
	"""ArXivLoader component.

	Purpose:
	    Retrieves arXiv papers as LangChain documents and supports chunking their content.

	Attributes:
	    loader (Optional[ArxivLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    max_documents (Optional[int]): Upper bound applied to documents.
	    max_characters (Optional[int]): Upper bound applied to characters.
	    include_metadata (Optional[bool]): Flag controlling whether include metadata behavior is enabled.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ ArxivLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_metadata = False
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'max_documents',
			'max_characters', 'include_metadata', 'verify_exists', 'resolve_paths',
			'split_documents', 'load', 'split', ]
	
	def load( self, query: str, max_chars: int=1000 ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    max_chars (int): Max chars supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_characters = max_chars
			self.loader = ArxivLoader( query=self.query,
				doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class WikiLoader( Loader ):
	"""WikiLoader component.

	Purpose:
	    Retrieves Wikipedia pages as LangChain documents and supports chunking their article text.

	Attributes:
	    loader (Optional[WikipediaLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    max_documents (Optional[int]): Upper bound applied to documents.
	    max_characters (Optional[int]): Upper bound applied to characters.
	    include_all (Optional[bool]): Flag controlling whether include all behavior is enabled.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ WikipediaLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all = False
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'max_documents',
			'max_characters', 'include_all', 'verify_exists', 'resolve_paths', 'split_documents',
			'load', 'split', ]
	
	def load( self, query: str, max_docs: int=25, max_chars: int=4000,
		include_all: bool=False ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    max_docs (int): Max docs supplied by the caller and interpreted according to the method contract.
		    max_chars (int): Max chars supplied by the caller and interpreted according to the method contract.
		    include_all (bool): Whether include all behavior is enabled for the operation.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			
			self.query = query
			self.max_documents = max_docs
			self.max_characters = max_chars
			self.include_all = include_all
			
			self.loader = WikipediaLoader( query=self.query, load_max_docs=self.max_documents,
				load_all_available_meta=self.include_all,
				doc_content_chars_max=self.max_characters )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = ('load( self, query: str, max_docs: int=25, max_chars: int=4000, '
			                    'include_all: bool=False ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class GithubLoader( Loader ):
	"""GithubLoader component.

	Purpose:
	    Loads files from a GitHub repository using repository, branch, path, and file-filter settings.

	Attributes:
	    loader (Optional[GithubFileLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    repo (Optional[str]): Current repo retained by the GithubLoader workflow between related operations.
	    branch (Optional[str]): Current branch retained by the GithubLoader workflow between related operations.
	    access_token (Optional[str]): Current access token retained by the GithubLoader workflow between related operations.
	    github_url (Optional[str]): URL associated with the current github resource or endpoint.
	    file_filter (Optional[object]): Current file filter retained by the GithubLoader workflow between related operations.
	    pattern (Optional[str]): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ GithubFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	repo: Optional[ str ]
	branch: Optional[ str ]
	access_token: Optional[ str ]
	github_url: Optional[ str ]
	file_filter: Optional[ object ]
	pattern: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.github_url = None
		self.repo = None
		self.branch = None
		self.access_token = None
		self.file_filter = None
		self.pattern = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'repo', 'branch',
			'access_token', 'github_url', 'file_filter', 'verify_exists', 'resolve_paths',
			'split_documents', 'load', 'split', ]
	
	def load( self, url: str, repo: str, branch: str, filetype: str='.md',
		access_token: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    url (str): Absolute endpoint or resource URL.
		    repo (str): Repo supplied by the caller and interpreted according to the method contract.
		    branch (str): Branch supplied by the caller and interpreted according to the method contract.
		    filetype (str): Filetype supplied by the caller and interpreted according to the method contract.
		    access_token (Optional[str]): Access token supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'repo', repo )
			throw_if( 'branch', branch )
			
			self.github_url = url
			self.repo = repo
			self.branch = branch
			self.access_token = access_token.strip( ) if isinstance( access_token,
				str ) and access_token.strip( ) else None
			self.pattern = filetype.strip( ) if isinstance( filetype,
				str ) and filetype.strip( ) else '.md'
			self.file_filter = lambda file_path: file_path.endswith( self.pattern )
			
			kwargs = { 'repo': self.repo, 'branch': self.branch, 'github_api_url': self.github_url,
				'file_filter': self.file_filter, }
			
			if self.access_token:
				kwargs[ 'access_token' ] = self.access_token
			
			self.loader = GithubFileLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = ('load( self, url: str, repo: str, branch: str, '
			                    'filetype: str=".md", access_token: Optional[ str ]=None ) '
			                    '-> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class PowerPointLoader( Loader ):
	"""PowerPointLoader component.

	Purpose:
	    Loads one or more PowerPoint presentations into LangChain documents using configurable partitioning.

	Attributes:
	    loader (Optional[UnstructuredPowerPointLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    mode (Optional[str]): Current mode retained by the PowerPointLoader workflow between related operations.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'query', 'mode',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'load_multiple',
			'split', ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize mode.

		Purpose:
		    Normalizes mode into the canonical representation expected by the surrounding workflow.

		Args:
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    str: Normalized text produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'multiple':
			return 'elements'
		
		if value not in [ 'single', 'elements' ]:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    mode (str): Provider or loader operating mode selected for the request.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredPowerPointLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = ('load( self, path: str, mode: str="single" ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		"""Load multiple.

		Purpose:
		    Loads multiple into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.load( path, mode='elements' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load_multiple( self, path: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class OutlookLoader( Loader ):
	"""OutlookLoader component.

	Purpose:
	    Loads Outlook message files into LangChain documents and supports message-content chunking.

	Attributes:
	    loader (Optional[OutlookMessageLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    max_documents (Optional[int]): Upper bound applied to documents.
	    max_characters (Optional[int]): Upper bound applied to characters.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ OutlookMessageLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'max_charactes',
			'max_documents', 'verify_exists', 'resolve_paths', 'split_documents', 'load',
			'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class WebCrawler( Loader ):
	"""WebCrawler component.

	Purpose:
	    Loads web content from one or more pages or recursively traverses a site into LangChain documents.

	Attributes:
	    loader (Optional[RecursiveUrlLoader | WebBaseLoader]): Concrete LangChain loader configured for the selected source.
	    url (Optional[str]): Most recent endpoint or resource URL used by the instance.
	    web_paths (Optional[str | List[str]]): Current web paths retained by the WebCrawler workflow between related operations.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    max_depth (Optional[int]): Upper bound applied to depth.
	    timeout (Optional[int]): Maximum request duration, in seconds, applied to provider calls.
	    ignore (Optional[bool]): Current ignore retained by the WebCrawler workflow between related operations.
	    with_progress (Optional[bool]): Current with progress retained by the WebCrawler workflow between related operations.
	    recursive (Optional[bool]): Current recursive retained by the WebCrawler workflow between related operations.
	    prevent_outside (Optional[bool]): Current prevent outside retained by the WebCrawler workflow between related operations.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, url: str, recursive: bool=False, max_depth: int=2,
		prevent_outside: bool=True, timeout: int=10, ignore: bool=True,
		progress: bool=True ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Args:
		    url (str): Absolute endpoint or resource URL.
		    recursive (bool): Whether loading follows links or descends into child resources.
		    max_depth (int): Maximum number of link levels traversed from the starting page.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.url = url
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
		self.loader = RecursiveUrlLoader( url=self.url, max_depth=self.max_depth,
			timeout=self.timeout, continue_on_failure=self.ignore,
			prevent_outside=self.prevent_outside )
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'url', 'web_paths',
			'max_depth', 'timeout', 'ignore', 'with_progress', 'recursive', 'prevent_outside',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'load_pages', 'split', ]
	
	def load( self, urls: str | List[ str ], depth: int=2, timeout: int=10, ignore: bool=
	True,
		progress: bool=True, prevent_outside: bool=True ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    depth (int): Maximum number of link levels traversed from the starting page.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.recursive:
				return self.load_recursive( urls=urls, depth=depth, timeout=timeout, ignore=ignore,
					prevent_outside=prevent_outside )
			
			return self.load_pages( urls=urls, timeout=timeout, ignore=ignore, progress=progress )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = 'load( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int=10, ignore: bool=True,
		progress: bool=True ) -> List[ Document ] | None:
		"""Load pages.

		Purpose:
		    Loads pages into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    progress (bool): Whether the loader reports progress while retrieving multiple resources.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			
			self.loader = WebBaseLoader( web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore, requests_kwargs={ 'timeout': self.timeout } )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = ('load_pages( self, urls: str | List[ str ], timeout: int=10, '
			                    'ignore: bool=True, progress: bool=True ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int=2, timeout: int=10,
		ignore: bool=True, prevent_outside: bool=True ) -> List[ Document ] | None:
		"""Load recursive.

		Purpose:
		    Loads recursive into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    urls (str | List[str]): Single URL or collection of URLs to load.
		    depth (int): Maximum number of link levels traversed from the starting page.
		    timeout (int): Maximum request duration in seconds.
		    ignore (bool): Whether individual retrieval failures are skipped instead of aborting the load.
		    prevent_outside (bool): Whether recursive loading is restricted to the starting domain.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			
			self.loader = RecursiveUrlLoader( url=self.url, max_depth=self.max_depth,
				timeout=self.timeout, continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = 'load_recursive( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> '
			                    'List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class SpfxLoader( Loader ):
	"""SpfxLoader component.

	Purpose:
	    Loads SharePoint files or folders into LangChain documents using configured Microsoft Graph credentials.

	Attributes:
	    loader (Optional[SharePointLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    library_id (Optional[str]): Identifier of the current library resource.
	    subsite_id (Optional[str]): Identifier of the current subsite resource.
	    folder_id (Optional[str]): Identifier of the current folder resource.
	    object_ids (Optional[List[str]]): Current object ids retained by the SpfxLoader workflow between related operations.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    with_token (Optional[bool]): Current with token retained by the SpfxLoader workflow between related operations.
	    is_recursive (Optional[bool]): Current is recursive retained by the SpfxLoader workflow between related operations.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ SharePointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	library_id: Optional[ str ]
	subsite_id: Optional[ str ]
	folder_id: Optional[ str ]
	object_ids: Optional[ List[ str ] ]
	query: Optional[ str ]
	with_token: Optional[ bool ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.folder_id = None
		self.library_id = None
		self.subsite_id = None
		self.object_ids = [ ]
		self.with_token = None
		self.is_recursive = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'folder_id', 'library_id',
			'subsite_id', 'object_id', 'with_token', 'is_recursive', 'verify_exists',
			'resolve_paths', 'split_documents', 'load', 'split', ]
	
	def load( self, library_id: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    library_id (str): Provider identifier of the target library resource.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'library_id', library_id )
			self.library_id = library_id
			self.is_recursive = True
			self.with_token = True
			self.loader = SharePointLoader( document_library_id=self.library_id,
				recursive=self.is_recursive, auth_with_token=self.with_token )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		"""Load folder.

		Purpose:
		    Loads folder into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    library_id (str): Provider identifier of the target library resource.
		    folder_id (str): Provider identifier of the target folder resource.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'library_id', library_id )
			throw_if( 'folder_id', folder_id )
			self.library_id = library_id
			self.folder_id = folder_id
			self.loader = SharePointLoader( document_library_id=self.library_id,
				folder_id=self.folder_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class OneDriveDocLoader( Loader ):
	"""OneDriveDocLoader component.

	Purpose:
	    Loads Microsoft OneDrive files or folders into LangChain documents.

	Attributes:
	    loader (Optional[OneDriveLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    client_id (Optional[str]): Identifier of the current client resource.
	    drive_id (Optional[str]): Identifier of the current drive resource.
	    client_secret (Optional[str]): Current client secret retained by the OneDriveDocLoader workflow between related operations.
	    query (Any): Most recent search text or model prompt submitted by the instance.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.drive_id = None
		self.client_id = None
		self.client_secret = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'query', 'drive_id',
			'client_id', 'client_secret', 'verify_exists', 'resolve_paths', 'split_documents',
			'load', 'load_folder', 'split', ]
	
	@property
	def file_options( self ) -> List[ str ]:
		"""File options.

		Purpose:
		    Returns supported file choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'pdf', 'doc', 'docx', 'txt' ]
	
	def load( self, id: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    id (str): Id supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		"""Load folder.

		Purpose:
		    Loads folder into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    id (str): Id supplied by the caller and interpreted according to the method contract.
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class GoogleLoader( Loader ):
	"""GoogleLoader component.

	Purpose:
	    Loads Google Drive files or folders into LangChain documents using configured service credentials.

	Attributes:
	    loader (Optional[GoogleDriveLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    file_id (Optional[str]): Identifier of the current file resource.
	    folder_id (Optional[str]): Identifier of the current folder resource.
	    is_recursive (Optional[bool]): Current is recursive retained by the GoogleLoader workflow between related operations.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ GoogleDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	file_id: Optional[ str ]
	folder_id: Optional[ str ]
	query: Optional[ str ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.file_id = None
		self.folder_id = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_recursive = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'query', 'folder_id',
			'file_id', 'is_recursive', 'verify_exists', 'resolve_paths', 'split_documents', 'load',
			'load_folder', 'split', ]
	
	@property
	def file_options( self ) -> List[ str ]:
		"""File options.

		Purpose:
		    Returns supported file choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'document', 'sheet', 'pdf' ]
	
	def load_file( self, file_id: str, recursive: bool=False ) -> List[ Document ] | None:
		"""Load file.

		Purpose:
		    Loads file into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    file_id (str): Provider identifier of the target file resource.
		    recursive (bool): Whether loading follows links or descends into child resources.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'recursive', recursive )
			self.file_id = file_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( file_ids=[ self.file_id ],
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_File( self, file_id: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, folder_id: str, recursive: bool=False ) -> List[ Document ] | None:
		"""Load folder.

		Purpose:
		    Loads folder into LangChain documents while preserving source metadata required by downstream retrieval.

		Args:
		    folder_id (str): Provider identifier of the target folder resource.
		    recursive (bool): Whether loading follows links or descends into child resources.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'folder_id', folder_id )
			self.folder_id = folder_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( folder_id=self.folder_id,
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class EmailLoader( Loader ):
	"""EmailLoader component.

	Purpose:
	    Loads email files into LangChain documents and supports chunking their extracted content.

	Attributes:
	    loader (Optional[UnstructuredEmailLoader]): Concrete LangChain loader configured for the selected source.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    has_attachments (Optional[bool]): Flag controlling whether has attachments behavior is enabled.
	    mode (Optional[str]): Current mode retained by the EmailLoader workflow between related operations.
	    pattern (Any): Path pattern, delimiter, or matching expression used by the loader.
	    chunk_size (Any): Maximum number of tokens or characters placed in each document chunk.
	    overlap_amount (Any): Number of tokens or characters repeated between adjacent chunks.
	"""
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'splitter', 'pattern', 'file_path', 'expanded',
			'candidates', 'resolved', 'chunk_size', 'overlap_amount', 'has_attachments', 'mode',
			'verify_exists', 'resolve_paths', 'split_documents', 'load', 'split', ]
	
	def load( self, path: str, mode: str='single', attachments: bool=True ) -> List[
		Document ]:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    mode (str): Provider or loader operating mode selected for the request.
		    attachments (bool): Whether attachments behavior is enabled for the operation.

		Returns:
		    List[Document]: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.has_attachments = attachments
			self.loader = UnstructuredEmailLoader( file_path=self.file_path, mode=self.mode,
				process_attachments=self.has_attachments )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class PubMedSearchLoader( Loader ):
	"""PubMedSearchLoader component.

	Purpose:
	    Retrieves PubMed search results as LangChain documents and supports chunking article content.

	Attributes:
	    loader (Optional[PubMedLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    max_docs (Optional[int]): Upper bound applied to docs.
	"""
	loader: Optional[ PubMedLoader ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_docs: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.query = None
		self.max_docs = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'query', 'max_docs', 'chunk_size', 'overlap_amount',
			'load',
			'split', 'split_documents', ]
	
	def load( self, query: str, max_docs: int=5 ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    max_docs (int): Max docs supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_docs = max_docs
			self.loader = PubMedLoader( query=self.query, load_max_docs=self.max_docs )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
				'load( self, query: str, max_docs: int=5 ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
				'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class OpenCityLoader( Loader ):
	"""OpenCityLoader component.

	Purpose:
	    Loads records from an Open City Data endpoint into LangChain documents.

	Attributes:
	    loader (Optional[OpenCityDataLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    city_id (Optional[str]): Identifier of the current city resource.
	    dataset_id (Optional[str]): Identifier of the current dataset resource.
	    limit (Optional[int]): Current limit retained by the OpenCityLoader workflow between related operations.
	"""
	loader: Optional[ OpenCityDataLoader ]
	documents: Optional[ List[ Document ] ]
	city_id: Optional[ str ]
	dataset_id: Optional[ str ]
	limit: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.city_id = None
		self.dataset_id = None
		self.limit = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'city_id', 'dataset_id', 'limit', 'chunk_size',
			'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, city_id: str, dataset_id: str, limit: int=100 ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    city_id (str): Provider identifier of the target city resource.
		    dataset_id (str): Provider identifier of the target dataset resource.
		    limit (int): Maximum number of records or characters permitted by the operation.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'city_id', city_id )
			throw_if( 'dataset_id', dataset_id )
			self.city_id = city_id
			self.dataset_id = dataset_id
			self.limit = limit
			self.loader = OpenCityDataLoader( city_id=self.city_id, dataset_id=self.dataset_id,
				limit=self.limit )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = ('load( self, city_id: str, dataset_id: str, limit: int=100 ) '
			                    '-> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = (
				'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class JupyterNotebookLoader( Loader ):
	"""JupyterNotebookLoader component.

	Purpose:
	    Loads Jupyter notebooks into LangChain documents with configurable cell and output inclusion.

	Attributes:
	    loader (Optional[NotebookLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    include_outputs (Optional[bool]): Flag controlling whether include outputs behavior is enabled.
	    max_output_length (Optional[int]): Upper bound applied to output length.
	    remove_newline (Optional[bool]): Current remove newline retained by the JupyterNotebookLoader workflow between related operations.
	    traceback (Optional[bool]): Current traceback retained by the JupyterNotebookLoader workflow between related operations.
	"""
	loader: Optional[ NotebookLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	include_outputs: Optional[ bool ]
	max_output_length: Optional[ int ]
	remove_newline: Optional[ bool ]
	traceback: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.file_path = None
		self.include_outputs = None
		self.max_output_length = None
		self.remove_newline = None
		self.traceback = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'file_path', 'include_outputs', 'max_output_length',
			'remove_newline', 'traceback', 'chunk_size', 'overlap_amount', 'load', 'split',
			'split_documents', ]
	
	def load( self, path: str, include_outputs: bool=False, max_output_length: int=10,
		remove_newline: bool=False, traceback: bool=False ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    path (str): Filesystem or resource path identifying the input or output.
		    include_outputs (bool): Whether include outputs behavior is enabled for the operation.
		    max_output_length (int): Max output length supplied by the caller and interpreted according to the method contract.
		    remove_newline (bool): Whether remove newline behavior is enabled for the operation.
		    traceback (bool): Whether traceback behavior is enabled for the operation.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.include_outputs = include_outputs
			self.max_output_length = max_output_length
			self.remove_newline = remove_newline
			self.traceback = traceback
			
			self.loader = NotebookLoader( self.file_path, include_outputs=self.include_outputs,
				max_output_length=self.max_output_length, remove_newline=self.remove_newline,
				traceback=self.traceback )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = 'load( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = (
				'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class GoogleCloudFileLoader( Loader ):
	"""GoogleCloudFileLoader component.

	Purpose:
	    Loads a single Google Cloud Storage object into LangChain documents.

	Attributes:
	    loader (Optional[GCSFileLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    project_name (Optional[str]): Current project name retained by the GoogleCloudFileLoader workflow between related operations.
	    bucket (Optional[str]): Current bucket retained by the GoogleCloudFileLoader workflow between related operations.
	    blob (Optional[str]): Current blob retained by the GoogleCloudFileLoader workflow between related operations.
	"""
	loader: Optional[ GCSFileLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	blob: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.blob = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'project_name', 'bucket', 'blob', 'chunk_size',
			'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, project_name: str, bucket: str, blob: str ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    project_name (str): Project name supplied by the caller and interpreted according to the method contract.
		    bucket (str): Bucket supplied by the caller and interpreted according to the method contract.
		    blob (str): Blob supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			throw_if( 'blob', blob )
			self.project_name = project_name
			self.bucket = bucket
			self.blob = blob
			self.loader = GCSFileLoader( project_name=self.project_name, bucket=self.bucket,
				blob=self.blob )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = ('load( self, project_name: str, bucket: str, blob: str ) '
			                    '-> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
				'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class AwsFileLoader( Loader ):
	"""AwsFileLoader component.

	Purpose:
	    Loads a single Amazon S3 object into LangChain documents.

	Attributes:
	    loader (Optional[S3FileLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    bucket (Optional[str]): Current bucket retained by the AwsFileLoader workflow between related operations.
	    key (Optional[str]): Current key retained by the AwsFileLoader workflow between related operations.
	    aws_access_key_id (Optional[str]): Identifier of the current aws access key resource.
	    aws_secret_access_key (Optional[str]): Current aws secret access key retained by the AwsFileLoader workflow between related operations.
	    aws_session_token (Optional[str]): Current aws session token retained by the AwsFileLoader workflow between related operations.
	    region_name (Optional[str]): Current region name retained by the AwsFileLoader workflow between related operations.
	"""
	loader: Optional[ S3FileLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	key: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.key = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'bucket', 'key', 'aws_access_key_id',
			'aws_secret_access_key', 'aws_session_token', 'region_name', 'chunk_size',
			'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, bucket: str, key: str, aws_access_key_id: Optional[ str ] = None,
		aws_secret_access_key: Optional[ str ] = None, aws_session_token: Optional[ str ] = None,
		region_name: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    bucket (str): Bucket supplied by the caller and interpreted according to the method contract.
		    key (str): Key supplied by the caller and interpreted according to the method contract.
		    aws_access_key_id (Optional[str]): Provider identifier of the target aws access key resource.
		    aws_secret_access_key (Optional[str]): Aws secret access key supplied by the caller and interpreted according to the method contract.
		    aws_session_token (Optional[str]): Aws session token supplied by the caller and interpreted according to the method contract.
		    region_name (Optional[str]): Region name supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'key', key )
			
			self.bucket = bucket
			self.key = key
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			
			kwargs: Dict[ str, Any ] = { }
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			
			self.loader = S3FileLoader( self.bucket, self.key, **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = 'load( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) '
			                    '-> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class GoogleSpeechToTextLoader( Loader ):
	"""GoogleSpeechToTextLoader component.

	Purpose:
	    Transcribes supported audio through Google Speech-to-Text and returns LangChain documents.

	Attributes:
	    loader (Optional[SpeechToTextLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    project_id (Optional[str]): Identifier of the current project resource.
	    file_path (Optional[str]): Resolved filesystem path of the current source or output file.
	    config (Optional[Dict[str, Any]]): Provider-specific generation configuration for the active request.
	"""
	loader: Optional[ SpeechToTextLoader ]
	documents: Optional[ List[ Document ] ]
	project_id: Optional[ str ]
	file_path: Optional[ str ]
	config: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_id = None
		self.file_path = None
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'project_id', 'file_path', 'config', 'chunk_size',
			'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, project_id: str, file_path: str,
		config: Optional[ Dict[ str, Any ] ] = None ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    project_id (str): Provider identifier of the target project resource.
		    file_path (str): Filesystem path of the source document.
		    config (Optional[Dict[str, Any]]): Config supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'project_id', project_id )
			throw_if( 'file_path', file_path )
			
			self.project_id = project_id
			self.file_path = file_path
			self.config = config
			
			if self.config:
				self.loader = SpeechToTextLoader( project_id=self.project_id,
					file_path=self.file_path, config=self.config )
			else:
				self.loader = SpeechToTextLoader( project_id=self.project_id,
					file_path=self.file_path )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = ('load( self, project_id: str, file_path: str, '
			                    'config: Optional[ Dict[ str, Any ] ]=None ) -> List[ Document ] | '
			                    'None')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) '
			                    '-> List[ Document ] | None')
			Logger( ).write( exception )
			raise exception

class GoogleBucketLoader( Loader ):
	"""GoogleBucketLoader component.

	Purpose:
	    Loads all matching objects from a Google Cloud Storage bucket into LangChain documents.

	Attributes:
	    loader (Optional[GCSDirectoryLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    project_name (Optional[str]): Current project name retained by the GoogleBucketLoader workflow between related operations.
	    bucket (Optional[str]): Current bucket retained by the GoogleBucketLoader workflow between related operations.
	    prefix (Optional[str]): Current prefix retained by the GoogleBucketLoader workflow between related operations.
	    continue_on_failure (Optional[bool]): Current continue on failure retained by the GoogleBucketLoader workflow between related operations.
	"""
	loader: Optional[ GCSDirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	continue_on_failure: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.prefix = None
		self.continue_on_failure = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'project_name', 'bucket', 'prefix', 'continue_on_failure',
			'chunk_size', 'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, project_name: str, bucket: str, prefix: Optional[ str ] = None,
		continue_on_failure: bool=False ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    project_name (str): Project name supplied by the caller and interpreted according to the method contract.
		    bucket (str): Bucket supplied by the caller and interpreted according to the method contract.
		    prefix (Optional[str]): Prefix supplied by the caller and interpreted according to the method contract.
		    continue_on_failure (bool): Whether continue on failure behavior is enabled for the operation.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			self.project_name = project_name
			self.bucket = bucket
			self.prefix = prefix
			self.continue_on_failure = continue_on_failure
			kwargs: Dict[ str, Any ] = { 'project_name': self.project_name, 'bucket': self.bucket,
				'continue_on_failure': self.continue_on_failure, }
			
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			
			self.loader = GCSDirectoryLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = 'load( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = 'split( self, **kwargs0 ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class AwsBucketLoader( Loader ):
	"""AwsBucketLoader component.

	Purpose:
	    Loads all matching objects from an Amazon S3 bucket into LangChain documents.

	Attributes:
	    loader (Optional[S3DirectoryLoader]): Concrete LangChain loader configured for the selected source.
	    documents (Optional[List[Document]]): LangChain documents loaded or produced by the most recent operation.
	    bucket (Optional[str]): Current bucket retained by the AwsBucketLoader workflow between related operations.
	    prefix (Optional[str]): Current prefix retained by the AwsBucketLoader workflow between related operations.
	    aws_access_key_id (Optional[str]): Identifier of the current aws access key resource.
	    aws_secret_access_key (Optional[str]): Current aws secret access key retained by the AwsBucketLoader workflow between related operations.
	    aws_session_token (Optional[str]): Current aws session token retained by the AwsBucketLoader workflow between related operations.
	    region_name (Optional[str]): Current region name retained by the AwsBucketLoader workflow between related operations.
	    endpoint_url (Optional[str]): URL associated with the current endpoint resource or endpoint.
	"""
	loader: Optional[ S3DirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	endpoint_url: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.prefix = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
		self.endpoint_url = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'loader', 'documents', 'bucket', 'prefix', 'aws_access_key_id',
			'aws_secret_access_key', 'aws_session_token', 'region_name', 'endpoint_url',
			'chunk_size', 'overlap_amount', 'load', 'split', 'split_documents', ]
	
	def load( self, bucket: str, prefix: Optional[ str ] = None,
		aws_access_key_id: Optional[ str ] = None, aws_secret_access_key: Optional[ str ] = None,
		aws_session_token: Optional[ str ] = None, region_name: Optional[ str ] = None,
		endpoint_url: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.

		Purpose:
		    Loads the selected source into LangChain documents using the instance configuration.

		Args:
		    bucket (str): Bucket supplied by the caller and interpreted according to the method contract.
		    prefix (Optional[str]): Prefix supplied by the caller and interpreted according to the method contract.
		    aws_access_key_id (Optional[str]): Provider identifier of the target aws access key resource.
		    aws_secret_access_key (Optional[str]): Aws secret access key supplied by the caller and interpreted according to the method contract.
		    aws_session_token (Optional[str]): Aws session token supplied by the caller and interpreted according to the method contract.
		    region_name (Optional[str]): Region name supplied by the caller and interpreted according to the method contract.
		    endpoint_url (Optional[str]): Endpoint url supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket = bucket
			self.prefix = prefix
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			self.endpoint_url = endpoint_url
			
			kwargs: Dict[ str, Any ] = { }
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			if self.endpoint_url:
				kwargs[ 'endpoint_url' ] = self.endpoint_url
			
			self.loader = S3DirectoryLoader( self.bucket, **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = 'load( self, **args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		"""Split.

		Purpose:
		    Split using the class state and returns data required by the surrounding workflow.

		Args:
		    chunk (int): Maximum chunk size used by the text splitter.
		    overlap (int): Number of characters or tokens repeated between adjacent chunks.

		Returns:
		    List[Document] | None: LangChain documents produced or transformed by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = 'split( self, **kwargs ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
