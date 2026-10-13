'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                writers.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022

	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="writers.py" company="Terry D. Eppler">

	     Foo is a python framework for web scraping information into ML pipelines.
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
	writers.py

	Purpose:
		Provides Markdown writer utilities used by Foo workflows to persist extracted
		text and fetch results as documentation-friendly Markdown files. The module
		contains a base writer for plain text output and a specialized writer that
		serializes response metadata as YAML front matter followed by the response
		body.
	</summary>
	******************************************************************************************
'''
from pathlib import Path
from typing import Optional
from core import Result
from boogr import Error, Logger


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

class Writer( ):
	"""Persist text content as UTF-8 Markdown files.

	Purpose:
		Provides the base filesystem wrapper used by Foo output workflows. Required method
		arguments are validated and stored as instance state before directory creation and file
		writing, making the active payload and destination available to callers and diagnostic tools.

	Attributes:
		text (Optional[str]): Text payload supplied to the current write operation.
		filename (Optional[str]): Base filename used for the generated Markdown document.
		directory (Optional[str]): Destination directory supplied to the current write operation.
		output_path (Optional[Path]): Directory containing the generated Markdown file.
		file_path (Optional[Path]): Path of the most recently written Markdown file.
		result (Optional[Result]): Result object retained by specialized writers.
		body (Optional[str]): Serialized body retained by specialized writers.
	"""
	text: Optional[ str ]
	filename: Optional[ str ]
	directory: Optional[ str ]
	output_path: Optional[ Path ]
	file_path: Optional[ Path ]
	result: Optional[ Result ]
	body: Optional[ str ]

	def __init__( self ) -> None:
		"""Initialize writer state.

		Purpose:
			Creates an empty write context for payload, destination, generated path, and result state.
			The constructor does not create directories or write files.

		Returns:
			None: Initializes instance state without returning a value.
		"""
		self.text = None
		self.filename = None
		self.directory = None
		self.output_path = None
		self.file_path = None
		self.result = None
		self.body = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable ordering of writer state and operations for introspection, generated
			documentation, and interactive application tooling.

		Returns:
			list[str]: Ordered public attribute and method names exposed by the writer.
		"""
		return [ 'text', 'filename', 'directory', 'output_path', 'file_path', 'result', 'body',
			'write' ]

	def write( self, text: str, filename: str, directory: str='output' ) -> Path | None:
		"""Write text to a Markdown file.

		Purpose:
			Validates and stores the content, filename, and output directory before invoking pathlib.
			The method creates missing parent directories, writes UTF-8 content, retains the final
			file path, and returns it to the caller.

		Args:
			text (str): Text content to persist in the Markdown file.
			filename (str): Base filename without the ``.md`` extension.
			directory (str): Directory in which the Markdown file is created.

		Returns:
			Path | None: Path of the successfully written Markdown file.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'filename', filename )
			throw_if( 'directory', directory )
			self.text = str( text )
			self.filename = str( filename ).strip( )
			self.directory = str( directory ).strip( )
			self.output_path = Path( self.directory )
			self.output_path.mkdir( parents=True, exist_ok=True )
			self.file_path = self.output_path / f'{self.filename}.md'
			self.file_path.write_text( self.text, encoding='utf-8' )
			return self.file_path
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'writers'
			exception.cause = 'Writer'
			exception.method = 'write( self, **kwargs ) -> Path | None'
			Logger( ).write( exception )
			raise exception


class MarkdownWriter( Writer ):
	"""Serialize Foo results as Markdown with YAML front matter.

	Purpose:
		Extends the base writer with result-aware serialization. The destination path and result
		object are validated and stored before pathlib is invoked, then source metadata is written
		as YAML front matter followed by the response body.

	Attributes:
		path (Optional[str]): Destination path supplied to the current result write operation.
		front_matter (Optional[str]): YAML metadata block generated from the current result.
		text (Optional[str]): Text payload inherited from the base writer.
		filename (Optional[str]): Base filename inherited from the base writer.
		directory (Optional[str]): Output directory inherited from the base writer.
		output_path (Optional[Path]): Directory containing the generated Markdown file.
		file_path (Optional[Path]): Resolved path of the generated Markdown file.
		result (Optional[Result]): Result object currently being serialized.
		body (Optional[str]): Response body written after the YAML front matter.
	"""
	path: Optional[ str ]
	front_matter: Optional[ str ]

	def __init__( self ) -> None:
		"""Initialize Markdown result writer state.

		Purpose:
			Initializes inherited filesystem state together with the destination-path and front-matter
			members used when serializing Foo results.

		Returns:
			None: Initializes instance state without returning a value.
		"""
		super( ).__init__( )
		self.path = None
		self.front_matter = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable ordering of inherited writer state and result-specific serialization
			members for introspection and generated documentation.

		Returns:
			list[str]: Ordered public attribute and method names exposed by the writer.
		"""
		return [ 'path', 'front_matter', 'text', 'filename', 'directory', 'output_path',
			'file_path', 'result', 'body', 'write' ]

	def write( self, result: Result, path: str ) -> Path | None:
		"""Write a Foo result as a Markdown document.

		Purpose:
			Validates and stores the result and destination path before invoking pathlib. The method
			creates missing parent directories, emits source URL and status code metadata as YAML
			front matter, appends a newline-terminated response body, and returns the resolved path.

		Args:
			result (Result): Foo result containing response URL, status code, and text content.
			path (str): Destination file path for the generated Markdown document.

		Returns:
			Path | None: Resolved path of the successfully written Markdown file.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'result', result )
			throw_if( 'path', path )
			self.result = result
			self.path = str( path ).strip( )
			self.file_path = Path( self.path ).resolve( )
			self.output_path = self.file_path.parent
			self.output_path.mkdir( parents=True, exist_ok=True )
			self.front_matter = (
					'---\n' + f'source_url: {self.result.url}\n' + f'status_code: '
					                                               f'{self.result.status_code}\n'
					+ '---\n\n')
			self.body = (
				self.result.text if self.result.text.endswith( '\n' ) else self.result.text + '\n')
			self.text = self.front_matter + self.body
			self.file_path.write_text( self.text, encoding='utf-8' )
			return self.file_path
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'writers'
			exception.cause = 'MarkdownWriter'
			exception.method = 'write( self, result: Result, path: str ) -> Path | None'
			Logger( ).write( exception )
			raise exception
