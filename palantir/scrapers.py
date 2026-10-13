'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                scrapers.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="scrapers.py" company="Terry D. Eppler">

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
    scrapers.py

    Purpose:
        Provides synchronous HTML scraping utilities for the Foo application. The module
        defines a small extractor base class and a requests-backed web extractor capable of
        retrieving pages, converting HTML to compact text, extracting common HTML element
        groups, and constructing tool schema dictionaries for model-facing workflows.
  </summary>
  ******************************************************************************************
'''
from typing import Any, Optional, List, Pattern, Dict
from bs4 import BeautifulSoup
from requests import Response, HTTPError
from boogr import Error, Logger
import config as cfg
import re
import requests
from core import Result


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

class Extractor( ):
	"""Provide shared state for HTML extraction workflows.

	Purpose:
		Defines the raw-document, parsed-document, and extracted-text state used by concrete
		extractors. Subclasses populate these members while retrieving and transforming source
		content for display, indexing, or model-facing workflows.

	Attributes:
		raw_html (Optional[str]): Unmodified HTML retained before parsing or text conversion.
		extracted_text (Optional[str]): Plain text produced from the current HTML document.
		soup (Optional[BeautifulSoup]): Parsed document tree for the current HTML source.
	"""
	raw_html: Optional[ str ]
	extracted_text: Optional[ str ]
	soup: Optional[ BeautifulSoup ]

	def __init__( self ) -> None:
		"""Initialize shared extraction state.

		Purpose:
			Creates an empty extraction context that concrete subclasses can populate without
			performing network, parsing, or transformation work during construction.

		Returns:
			None: Initializes instance state without returning a value.
		"""
		self.raw_html = None
		self.extracted_text = None
		self.soup = None

	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable public-member ordering for introspection, generated documentation,
			and interactive tooling that inspects extractor capabilities.

		Returns:
			List[str]: Ordered public member names exposed by the base extractor.
		"""
		return [ 'raw_html', 'extracted_text', 'soup' ]


class WebExtractor( Extractor ):
	"""Fetch and extract structured content from HTML pages.

	Purpose:
		Wraps synchronous HTTP retrieval with reusable HTML parsing operations. The class stores
		validated request arguments as instance state before invoking ``requests``, then exposes
		plain-text conversion, semantic element extraction, link and image discovery, and dynamic
		tool-schema construction for downstream ML and model orchestration workflows.

	Attributes:
		agents (Optional[str]): User-agent value sent with HTTP requests.
		url (Optional[str]): Normalized URL used by the current request.
		html (Optional[str]): HTML currently being transformed or parsed.
		timeout (Optional[int]): Maximum request duration in seconds.
		headers (Optional[Dict[str, str]]): HTTP headers supplied to the wrapped requests client.
		response (Optional[Response]): Most recent HTTP response returned by ``requests``.
		result (Optional[Result]): Canonical Foo result created from the most recent response.
		re_tag (Optional[Pattern]): Pattern used to remove remaining HTML tags.
		re_ws (Optional[Pattern]): Pattern used to collapse repeated whitespace.
		function (Optional[str]): Function name assigned to the current tool schema.
		tool (Optional[str]): Service name assigned to the current tool schema.
		description (Optional[str]): Human-readable behavior description for the current schema.
		parameters (Optional[Dict[str, Any]]): JSON Schema property definitions for tool arguments.
		required (Optional[List[str]]): Required argument names included in the current schema.
	"""
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, str ] ]
	response: Optional[ Response ]
	result: Optional[ Result ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	function: Optional[ str ]
	tool: Optional[ str ]
	description: Optional[ str ]
	parameters: Optional[ Dict[ str, Any ] ]
	required: Optional[ List[ str ] ]

	def __init__( self ) -> None:
		"""Initialize the web extraction client.

		Purpose:
			Prepares request headers, timeout defaults, regular-expression helpers, response state,
			and schema state used by subsequent retrieval and extraction operations. Construction
			does not perform network activity.

		Returns:
			None: Initializes instance state without returning a value.
		"""
		super( ).__init__( )
		self.timeout = 10
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = None
		self.html = None
		self.response = None
		self.result = None
		self.headers = { }
		self.agents = cfg.AGENTS
		self.function = None
		self.tool = None
		self.description = None
		self.parameters = None
		self.required = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable ordering of request state, schema state, and extraction operations
			for
			introspection, generated documentation, and interactive application tooling.

		Returns:
			List[str]: Ordered public attribute and method names exposed by the extractor.
		"""
		return [ 'agents', 'url', 'html', 'timeout', 'headers', 'response', 'result', 'raw_html',
			'extracted_text', 'soup', 'function', 'tool', 'description', 'parameters', 'required',
			'scrape', 'html_to_text', 'scrape_paragraphs', 'scrape_lists', 'scrape_tables',
			'scrape_articles', 'scrape_headings', 'scrape_divisions', 'scrape_sections',
			'scrape_blockquotes', 'scrape_hyperlinks', 'scrape_images', 'create_schema' ]

	def scrape( self, url: str, time: int=10 ) -> Result | None:
		"""Fetch an HTML resource and return its canonical result.

		Purpose:
			Validates and stores the request URL and timeout before invoking ``requests.get``. The
			method retains the response, verifies its status, captures the response HTML, prepares a
			BeautifulSoup document tree, and returns Foo's normalized ``Result`` representation.

		Args:
			url (str): Absolute HTTP or HTTPS URL of the resource to retrieve.
			time (int): Maximum number of seconds permitted for the HTTP request.

		Returns:
			Result | None: Canonical Foo result containing the successful HTTP response.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'time', time )
			self.url = str( url ).strip( )
			self.timeout = int( time )
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			self.result = Result( self.response )
			return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape( self, url: str, time: int=10 ) -> Result | None'
			Logger( ).write( exception )
			raise exception

	def html_to_text( self, html: str ) -> str:
		"""Convert HTML markup into compact plain text.

		Purpose:
			Stores the source document before removing script and style blocks, introducing readable
			boundaries around block elements, stripping residual tags, and collapsing repeated
			whitespace. The resulting text is suitable for indexing, display, or model input.

		Args:
			html (str): Raw HTML document to transform.

		Returns:
			str: Normalized plain text extracted from the HTML document.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'html', html )
			self.html = str( html )
			self.raw_html = self.html
			self.html = re.sub( r'<script[\s\S]*?</script>', ' ', self.html,
				flags=re.IGNORECASE )
			self.html = re.sub( r'<style[\s\S]*?</style>', ' ', self.html,
				flags=re.IGNORECASE )
			self.html = re.sub( r'</?(p|div|br|li|h[1-6])[^>]*>', '\n', self.html,
				flags=re.IGNORECASE )
			self.extracted_text = re.sub( self.re_tag, ' ', self.html )
			self.extracted_text = re.sub( self.re_ws, ' ', self.extracted_text ).strip( )
			return self.extracted_text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'html_to_text( self, html: str ) -> str'
			Logger( ).write( exception )
			raise exception

	def scrape_paragraphs( self, uri: str ) -> List[ str ] | None:
		"""Extract paragraph text from an HTML page.

		Purpose:
			Retrieves the supplied page through the configured request state and returns non-empty
			text from each paragraph element in document order.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty paragraph strings in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			blocks = [ p.get_text( ' ', strip=True ) for p in self.soup.find_all( 'p' ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_paragraphs( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_lists( self, uri: str ) -> List[ str ] | None:
		"""Extract list-item text from an HTML page.

		Purpose:
			Retrieves the supplied page and returns the readable content of non-empty ``li`` elements,
			providing a simple sequence for indexing or downstream content analysis.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty list-item strings in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			items = [ item.get_text( ' ', strip=True ) for item in self.soup.find_all( 'li' ) ]
			return [ item for item in items if item ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_lists( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_tables( self, uri: str ) -> List[ str ] | None:
		"""Extract flattened table-cell text from an HTML page.

		Purpose:
			Retrieves the target page and flattens non-empty header and data cells from every table
			into a document-order sequence suitable for quick inspection or text-based indexing.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty ``th`` and ``td`` values in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			results: List[ str ] = [ ]
			for table in self.soup.find_all( 'table' ):
				for row in table.find_all( 'tr' ):
					for cell in row.find_all( [ 'td', 'th' ] ):
						text = cell.get_text( ' ', strip=True )
						if text:
							results.append( text )
			return results
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_tables( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_articles( self, uri: str ) -> List[ str ] | None:
		"""Extract article-level text from an HTML page.

		Purpose:
			Retrieves the target page and consolidates readable text from each semantic ``article``
			element so callers can treat article blocks as document-like units.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty article text blocks in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			blocks = [ article.get_text( ' ', strip=True )
				for article in self.soup.find_all( 'article' ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_articles( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_headings( self, uri: str ) -> List[ str ] | None:
		"""Extract heading text from an HTML page.

		Purpose:
			Retrieves the target page and returns visible text from ``h1`` through ``h6`` elements,
			providing a structural outline for summarization, navigation, or content inspection.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty heading strings in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			heading_tags = [ 'h1', 'h2', 'h3', 'h4', 'h5', 'h6' ]
			blocks = [ heading.get_text( ' ', strip=True )
				for heading in self.soup.find_all( heading_tags ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_headings( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_divisions( self, uri: str ) -> List[ str ] | None:
		"""Extract division text from an HTML page.

		Purpose:
			Retrieves the target page and returns cleaned text from generic ``div`` containers,
			providing broad coverage for pages that do not use semantic content elements.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty division text blocks in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			blocks = [ division.get_text( ' ', strip=True )
				for division in self.soup.find_all( 'div' ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_divisions( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_sections( self, uri: str ) -> List[ str ] | None:
		"""Extract semantic section text from an HTML page.

		Purpose:
			Retrieves the target page and returns readable content from semantic ``section`` elements,
			preserving page-level content groupings for indexing or analysis.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty section text blocks in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			blocks = [ section.get_text( ' ', strip=True )
				for section in self.soup.find_all( 'section' ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_sections( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_blockquotes( self, uri: str ) -> List[ str ] | None:
		"""Extract blockquote text from an HTML page.

		Purpose:
			Retrieves the target page and returns readable text from semantic ``blockquote`` elements,
			capturing quoted or cited material separately from surrounding prose.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty blockquote strings in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			blocks = [ quote.get_text( ' ', strip=True )
				for quote in self.soup.find_all( 'blockquote' ) ]
			return [ block for block in blocks if block ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_blockquotes( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_hyperlinks( self, uri: str ) -> List[ str ] | None:
		"""Extract hyperlink references from an HTML page.

		Purpose:
			Retrieves the target page and returns non-empty ``href`` values from anchor elements.
			References remain in their source form so callers can decide whether and how to resolve
			relative paths.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty hyperlink references in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			links = [ anchor.get( 'href' ) for anchor in self.soup.find_all( 'a' )
				if anchor.get( 'href' ) ]
			return links
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_hyperlinks( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def scrape_images( self, uri: str ) -> List[ str ] | None:
		"""Extract image references from an HTML page.

		Purpose:
			Retrieves the target page and returns non-empty ``src`` values from image elements.
			References remain in their source form so callers can resolve or filter them according to
			the surrounding workflow.

		Args:
			uri (str): Fully qualified URL of the HTML page to inspect.

		Returns:
			List[str] | None: Non-empty image source references in document order.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = str( uri ).strip( )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.raw_html = self.html
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			images = [ image.get( 'src' ) for image in self.soup.find_all( 'img' )
				if image.get( 'src' ) ]
			return images
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_images( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception

	def create_schema( self, function: str, tool: str, description: str,
			parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Create a JSON-compatible tool schema.

		Purpose:
			Validates and stores schema inputs before constructing a function definition that names
			the wrapped service, describes its behavior, and declares JSON Schema properties and
			required arguments for model tool-calling workflows.

		Args:
			function (str): Function name exposed to the model or orchestration layer.
			tool (str): Service or system used by the generated function.
			description (str): Human-readable explanation of the function behavior.
			parameters (dict): Mapping of parameter names to JSON Schema fragments.
			required (list[str]): Required parameter names. When ``None``, all properties are required.

		Returns:
			Dict[str, str] | None: JSON-compatible function schema.

		Raises:
			Error: Re-raised after the source exception is wrapped with structured diagnostic metadata.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			self.function = str( function ).strip( )
			self.tool = str( tool ).strip( )
			self.description = str( description ).strip( )
			self.parameters = parameters
			self.required = required
			if not isinstance( self.parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

			if self.required is None:
				self.required = list( self.parameters.keys( ) )

			return {
				'name': self.function,
				'description': (
					f'{self.description} This function uses the {self.tool} service.'
				),
				'parameters': {
					'type': 'object',
					'properties': self.parameters,
					'required': self.required
				}
			}
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'create_schema( self, **args ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception
