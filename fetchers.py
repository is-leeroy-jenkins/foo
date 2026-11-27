"""
  ******************************************************************************************
      Assembly:                Foo
      Filename:                fetchers.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file='fetchers.py' company='Terry D. Eppler'>

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
    fetchers.py — provider fetchers, API clients, web scraping helpers, and data retrieval wrappers.

    Purpose:
        Provides the data-fetching layer for Foo. The module defines a common Fetcher base
        class and concrete wrappers for web pages, crawlers, public datasets, scientific
        APIs, geospatial services, weather services, astronomy services, archive providers,
        and other retrieval workflows used by the Streamlit application and ML pipelines.
</summary>
  ******************************************************************************************
  
"""
from __future__ import annotations

import base64
import datetime as dt
import math
import io
import re
import os
import urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Pattern, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from PIL.Image import Image
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from bs4 import BeautifulSoup
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import crawl4ai
import csv
from google import genai
from grokipedia_api import GrokipediaClient
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_googledrive.retrievers import GoogleDriveRetriever
from playwright.sync_api import sync_playwright
from owslib.wms import WebMapService
from requests import Response
from sscws.sscws import SscWs
import time
import config as cfg
from boogr import Error, Logger
from core import Result
import xml.etree.ElementTree as ET

def throw_if( name: str, value: object ) -> None:
	"""Throw if.

	Purpose:
	    Validates that a required argument contains a usable value so failures occur before
	    provider, filesystem, or parsing work begins.

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
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( path: str ) -> str:
	"""Encode image.

	Purpose:
	    Reads an image from disk and returns its bytes as a Base64-encoded UTF-8 string for APIs
	    that accept inline image data.

	Args:
	    path (str): Filesystem or resource path identifying the input or output.

	Returns:
	    str: Normalized text produced by the operation.

	Raises:
	    ValueError: Raised when the method cannot satisfy its documented value requirement.
	"""

	throw_if( 'path', path )

	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

class Fetcher:
	"""Shared retrieval contract and request state.

	Purpose:
	    Defines the common instance state for Foo retrieval adapters. Subclasses implement the fetch operation and determine the concrete result format.

	Attributes:
	    timeout (Optional[int]): Request timeout in seconds.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest resource URL.
	    result (Optional[Result]): Base retrieval result; concrete adapters may use other formats.
	    query (Optional[str]): Latest search or provider query.
	"""
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	url: Optional[ str ]
	result: Optional[ Result ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize shared retrieval state.

		Purpose:
		    Set timeout, headers, response, URL, result, and query to None without issuing a request.

		Returns:
		    None: Initializes instance state.
		"""
		self.timeout = None
		self.headers = None
		self.response = None
		self.url = None
		self.result = None
		self.query = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return exposed base member names.

		Purpose:
		    Provide the explicit member list used by dir() for the base retrieval adapter.

		Returns:
		    list[str]: Timeout, headers, response, URL, result, query, and fetch member names.
		"""
		return [ 'timeout', 'headers', 'response', 'url', 'result', 'query', 'fetch' ]
	
	def fetch( self, query: str, url: str, time: int=10 ) -> Result | None:
		"""Define the subclass retrieval contract.

		Purpose:
		    Provide the base method signature for retrieval adapters. This implementation always raises NotImplementedError and issues no request.

		Args:
		    query (str): Search text or provider query.
		    url (str): Target resource URL.
		    time (int): Requested timeout in seconds. Defaults to 10.

		Returns:
		    Result | None: Declared subclass result contract; the base method never returns.

		Raises:
		    NotImplementedError: Always raised because subclasses must implement retrieval.
		"""
		raise NotImplementedError( 'Must be implemented by a subclass.' )

class WebFetcher( Fetcher ):
	"""HTTP and HTML retrieval adapter.

	Purpose:
	    Fetches web pages over HTTP and exposes text conversion, URL resolution, link discovery, structured extraction, and function-schema construction.

	Attributes:
	    soup (Optional[BeautifulSoup]): Parsed HTML document tree.
	    agents (Optional[str]): Configured HTTP user-agent.
	    url (Optional[str]): Latest requested page URL.
	    html (Optional[str]): HTML returned by the latest request.
	    text (Optional[str]): Latest extracted or supplied text.
	    source_url (Optional[str]): Base URL used during extraction or link resolution.
	    source_html (Optional[str]): HTML supplied to an extraction method.
	    selected_methods (Optional[List[str]]): Selected structured extraction method names.
	    re_tag (Optional[Pattern]): Compiled markup-removal expression.
	    re_ws (Optional[Pattern]): Compiled whitespace expression.
	    response (Optional[Response]): Latest HTTP response.
	    result (Optional[Result]): Latest HTTP response wrapper.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    timeout (Optional[int]): Request timeout in seconds.
	    limit (int): Most recent text truncation character count.
	    href (str): Current hyperlink reference.
	    absolute_url (str): Resolved hyperlink before normalization.
	    parsed_url (urllib.parse.ParseResult): Parsed resolved hyperlink.
	    normalized_path (str): Resolved URL path or root path.
	    normalized_url (urllib.parse.ParseResult): Parsed URL with normalized path and no fragment.
	    left_url (str): First URL in the latest host comparison.
	    right_url (str): Second URL in the latest host comparison.
	    left_host (str): First parsed network location.
	    right_host (str): Second parsed network location.
	    results (List[str] | Dict[str, List[str]]): Latest link list or structured extraction groups.
	    seen (set[str]): Values encountered during deduplication.
	    candidate (str): Current normalized hyperlink candidate.
	    registry (Dict[str, Tuple[str, Any]]): Extraction method names mapped to display labels and callable extractors.
	    method_name (str): Current structured extraction method name.
	    label (str): Display label for the active extractor.
	    extractor (Any): Callable for the active extraction group.
	    values (List[str]): Current group values before deduplication.
	    deduped (List[str]): Current group values after deduplication.
	    value (str): Current value considered during deduplication.
	"""
	soup: Optional[ BeautifulSoup ]
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	text: Optional[ str ]
	source_url: Optional[ str ]
	source_html: Optional[ str ]
	selected_methods: Optional[ List[ str ] ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	result: Optional[ Result ]
	headers: Optional[ Dict[ str, Any ] ]
	timeout: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize HTTP and HTML extraction state.

		Purpose:
		    Configure the ten-second timeout, compiled tag and whitespace patterns, user-agent and Accept headers, and empty response and extraction state. No network request is issued.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.timeout = 10
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = None
		self.html = None
		self.text = None
		self.source_url = None
		self.source_html = None
		self.selected_methods = None
		self.response = None
		self.result = None
		self.soup = None
		self.headers = { }
		self.agents = cfg.AGENTS
		
		self.headers[ 'User-Agent' ] = self.agents
		self.headers[ 'Accept' ] = ('text/html,application/xhtml+xml,application/xml;q=0.9,'
		                            '*/*;q=0.8')
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed retrieval and extraction members.

		Purpose:
		    Provide the explicit member list used by dir(), including HTTP retrieval and HTML extraction methods.

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'agents', 'url', 'html', 'text', 'source_url', 'source_html', 'selected_methods',
				'timeout', 'headers', 'response', 'result', 'soup', 're_tag', 're_ws', 'fetch',
				'html_to_text', 'coerce_items', 'extract_title', 'truncate_text', 'normalize_url',
				'same_domain', 'extract_links', 'extract_structured_data', 'scrape_headings',
				'scrape_paragraphs', 'scrape_lists', 'scrape_tables', 'scrape_articles',
				'scrape_sections', 'scrape_divisions', 'scrape_blockquotes', 'scrape_hyperlinks',
				'scrape_images', 'create_schema' ]
	
	def fetch( self, url: str, time: int=10 ) -> Result | None:
		"""Retrieve a web page over HTTP.

		Purpose:
		    Send a GET request with configured headers and timeout, check its status, store HTML and a BeautifulSoup tree, and wrap the response in Result.

		Args:
		    url (str): Nonblank target URL.
		    time (int): Request timeout in seconds. Defaults to 10.

		Returns:
		    Result: Wrapper around the successful HTTP response.

		Raises:
		    Error: Raised when input validation, HTTP retrieval, parsing, or result construction fails.
		"""
		try:

			throw_if( 'url', url )

			self.url = url

			throw_if( 'time', time )
			self.timeout = time
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.html = self.response.text or ''
			self.soup = BeautifulSoup( self.html, 'html.parser' )
			self.result = Result( self.response )
			
			return self.result
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'fetch( self, url: str, time: int=10 ) -> Result | None'
			Logger( ).write( exception )
			raise exception
	
	def html_to_text( self, html: str ) -> str:
		"""Convert HTML into plain text.

		Purpose:
		    Remove script and style blocks, strip markup, and collapse whitespace, storing the resulting text.

		Args:
		    html (str): Nonblank HTML content.

		Returns:
		    str: Extracted text with whitespace collapsed.

		Raises:
		    Error: Raised when validation or text conversion fails.
		"""
		try:
			throw_if( 'html', html )
			self.source_html = html
			self.text = re.sub( r'<script[\s\S]*?</script>', ' ', self.source_html,
				flags=re.IGNORECASE )
			self.text = re.sub( r'<style[\s\S]*?</style>', ' ', self.text, flags=re.IGNORECASE )
			self.text = re.sub( r'</?(p|div|br|li|h[1-6]|section|article|blockquote)[^>]*>', '\n',
				self.text, flags=re.IGNORECASE )
			self.text = re.sub( self.re_tag, ' ', self.text )
			self.text = re.sub( self.re_ws, ' ', self.text ).strip( )
			
			return self.text
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'html_to_text( self, html: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def coerce_items( self, value: Any ) -> List[ str ]:
		"""Convert extraction values into strings.

		Purpose:
		    Convert a scalar into a single string, convert non-None list entries to strings, or return an empty list for None.

		Args:
		    value (Any): Scalar, list, or None to convert.

		Returns:
		    List[str]: Converted strings in input order.
		"""
		if value is None:
			return [ ]
		
		if isinstance( value, list ):
			return [ str( item ) for item in value if item is not None ]
		
		return [ str( value ) ]
	
	def extract_title( self, html: str ) -> str:
		"""Extract the HTML page title.

		Purpose:
		    Read the title through BeautifulSoup, falling back to a regular-expression search, and collapse whitespace.

		Args:
		    html (str): Nonblank HTML content.

		Returns:
		    str: Normalized title, or an empty string when absent.

		Raises:
		    Error: Raised when validation or title extraction fails.
		"""
		try:
			throw_if( 'html', html )
			self.source_html = html
			self.soup = BeautifulSoup( self.source_html, 'html.parser' )
			if self.soup.title and self.soup.title.string:
				return re.sub( r'\s+', ' ', self.soup.title.string ).strip( )
			
			match = re.search( r'<title[^>]*>(.*?)</title>', self.source_html, 
				flags=re.IGNORECASE | re.DOTALL )
			
			if not match:
				return ''
			
			return re.sub( r'\s+', ' ', match.group( 1 ) ).strip( )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'extract_title( self, html: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def truncate_text( self, text: str, limit: int=12000 ) -> str:
		"""Limit text to a character count.

		Purpose:
		    Return the original text when it fits; otherwise slice to the requested count and append a truncation marker.

		Args:
		    text (str): Nonblank text to shorten.
		    limit (int): Requested character count before the marker. Defaults to 12000.

		Returns:
		    str: Original or truncated text with a marker when shortened.

		Raises:
		    Error: Raised when validation or truncation fails.
		"""
		try:

			throw_if( 'limit', limit )

			throw_if( 'text', text )
			self.text = text
			self.limit = limit
			if len( self.text ) <= self.limit:
				return self.text
			
			return self.text[ : self.limit ] + '\n\n... [truncated]'
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'truncate_text( self, text: str, limit: int=12000 ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_url( self, base_url: str, href: str ) -> str:
		"""Resolve a hyperlink to an HTTP URL.

		Purpose:
		    Resolve relative references against the base URL, remove fragments, and supply a root path when absent. Reject unsupported schemes and hostless URLs.

		Args:
		    base_url (str): Base URL for resolving the reference.
		    href (str): Hyperlink reference to resolve.

		Returns:
		    str: Absolute HTTP or HTTPS URL, or an empty string for rejected references or errors.
		"""
		try:
			throw_if( 'base_url', base_url )
			throw_if( 'href', href )
			self.source_url = base_url.strip( )
			self.href = href.strip( )
			if self.href.startswith( ('mailto:', 'tel:', 'javascript:', '#') ):
				return ''
			
			self.absolute_url = urllib.parse.urljoin( self.source_url, self.href )
			self.parsed_url = urllib.parse.urlparse( self.absolute_url )
			if self.parsed_url.scheme not in ('http', 'https'):
				return ''
			
			if not self.parsed_url.netloc:
				return ''
			
			self.normalized_path = self.parsed_url.path or '/'
			self.normalized_url = self.parsed_url._replace( path=self.normalized_path,
				fragment='' )
			
			return self.normalized_url.geturl( )
		
		except Exception:
			return ''
	
	def same_domain( self, left_url: str, right_url: str ) -> bool:
		"""Compare the network locations of two URLs.

		Purpose:
		    Compare parsed network locations case-insensitively; ports remain part of the comparison.

		Args:
		    left_url (str): First URL.
		    right_url (str): Second URL.

		Returns:
		    bool: True when both have the same nonempty network location; otherwise False, including errors.
		"""
		try:
			throw_if( 'left_url', left_url )
			throw_if( 'right_url', right_url )
			self.left_url = left_url.strip( )
			self.right_url = right_url.strip( )
			self.left_host = urllib.parse.urlparse( self.left_url ).netloc.lower( )
			self.right_host = urllib.parse.urlparse( self.right_url ).netloc.lower( )
			
			return self.left_host != '' and self.left_host == self.right_host
		
		except Exception:
			return False
	
	def extract_links( self, base_url: str, html: str ) -> List[ str ]:
		"""Extract unique page hyperlinks.

		Purpose:
		    Parse anchor elements, resolve their references, and retain valid distinct URLs in discovery order.

		Args:
		    base_url (str): Base URL for resolving links.
		    html (str): Nonblank HTML content.

		Returns:
		    List[str]: Unique absolute HTTP or HTTPS hyperlinks.

		Raises:
		    Error: Raised when validation or extraction fails.
		"""
		try:
			throw_if( 'base_url', base_url )
			throw_if( 'html', html )
			self.source_url = base_url.strip( )
			self.source_html = html
			self.soup = BeautifulSoup( self.source_html, 'html.parser' )
			self.results = [ ]
			self.seen = set( )
			for tag in self.soup.find_all( 'a', href=True ):
				self.candidate = self.normalize_url( self.source_url, tag.get( 'href', '' ) )
				if self.candidate and self.candidate not in self.seen:
					self.seen.add( self.candidate )
					self.results.append( self.candidate )
			
			return self.results
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'extract_links( self, base_url: str, html: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def extract_structured_data( self, url: str, html: str, selected_methods: Optional[ List[ str ] ]=None ) -> Dict[ str, List[ str ] ]:
		"""Extract selected HTML element groups.

		Purpose:
		    Run registered extractors for requested method names, ignore unknown names, and deduplicate each group while retaining discovery order.

		Args:
		    url (str): Page URL used to resolve links and images.
		    html (str): Nonblank HTML content.
		    selected_methods (Optional[List[str]]): Extractor method names. Defaults to None, which selects no groups.

		Returns:
		    Dict[str, List[str]]: Display labels mapped to distinct text or URL values.

		Raises:
		    Error: Raised when validation, parsing, or extraction fails.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'html', html )
			self.source_url = url.strip( )
			self.source_html = html
			self.selected_methods = selected_methods or [ ]
			self.results = { }
			self.soup = BeautifulSoup( self.source_html, 'html.parser' )
			self.registry = { 'scrape_headings': ('Headings',
					lambda: [ tag.get_text( ' ', strip=True ) for tag in
							self.soup.find_all( [ 'h1', 'h2', 'h3', 'h4', 'h5', 'h6' ] ) if
							tag.get_text( ' ', strip=True ) ]), 'scrape_paragraphs': ('Paragraphs',
					lambda: [ tag.get_text( ' ', strip=True ) for tag in self.soup.find_all( 'p' )
							if tag.get_text( ' ', strip=True ) ]), 'scrape_lists': ('Lists',
					lambda: [ tag.get_text( ' ', strip=True ) for tag in self.soup.find_all( 'li' )
							if tag.get_text( ' ', strip=True ) ]), 'scrape_tables': ('Tables',
					lambda: [ cell.get_text( ' ', strip=True ) for table in
							self.soup.find_all( 'table' ) for row in table.find_all( 'tr' ) for
							cell
							in row.find_all( [ 'td', 'th' ] ) if
							cell.get_text( ' ', strip=True ) ]), 'scrape_articles': ('Articles',
					lambda: [ tag.get_text( ' ', strip=True ) for tag in
							self.soup.find_all( 'article' ) if tag.get_text( ' ', strip=True ) ]),
					'scrape_sections': ('Sections',
							lambda: [ tag.get_text( ' ', strip=True ) for tag in
									self.soup.find_all( 'section' ) if
									tag.get_text( ' ', strip=True ) ]),
					'scrape_divisions': ('Divisions',
							lambda: [ tag.get_text( ' ', strip=True ) for tag in
									self.soup.find_all( 'div' ) if
									tag.get_text( ' ', strip=True ) ]),
					'scrape_blockquotes': ('Blockquotes',
							lambda: [ tag.get_text( ' ', strip=True ) for tag in
									self.soup.find_all( 'blockquote' ) if
									tag.get_text( ' ', strip=True ) ]),
					'scrape_hyperlinks': ('Hyperlinks',
							lambda: [ self.normalize_url( self.source_url, tag.get( 'href', '' ) )
									for tag in self.soup.find_all( 'a', href=True ) if
									self.normalize_url( self.source_url, tag.get( 'href',
										'' ) ) ]),
					'scrape_images': ('Images',
							lambda: [ self.normalize_url( self.source_url, tag.get( 'src', '' ) )
									for tag in self.soup.find_all( 'img', src=True ) if
									self.normalize_url( self.source_url, tag.get( 'src', '' ) ) ]),}
			
			for self.method_name in self.selected_methods:
				if self.method_name not in self.registry:
					continue
				
				self.label, self.extractor = self.registry[ self.method_name ]
				self.values = self.coerce_items( self.extractor( ) )
				self.deduped = [ ]
				self.seen = set( )
				for self.value in self.values:
					if self.value not in self.seen:
						self.seen.add( self.value )
						self.deduped.append( self.value )
				
				self.results[ self.label ] = self.deduped
			
			return self.results
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'extract_structured_data( self, url: str, html: str, selected_methods: Optional[ List[ str ] ]=None ) -> Dict[ str, List[ str ] ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_paragraphs( self, uri: str ) -> List[ str ] | None:
		"""Extract paragraphs from a web page.

		Purpose:
		    Fetch the page and return distinct paragraph text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted paragraph text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url, 
				self.html or '', ['scrape_paragraphs' ] ).get( 'Paragraphs', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_paragraphs( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_lists( self, uri: str ) -> List[ str ] | None:
		"""Extract lists from a web page.

		Purpose:
		    Fetch the page and return distinct individual list-item text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted individual list-item text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url,
				self.html or '', [ 'scrape_lists' ] ).get( 'Lists', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_lists( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_tables( self, uri: str ) -> List[ str ] | None:
		"""Extract tables from a web page.

		Purpose:
		    Fetch the page and return distinct individual table header and data cell text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted individual table header and data cell text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url, self.html or '',
				[ 'scrape_tables' ] ).get( 'Tables', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_tables( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_articles( self, uri: str ) -> List[ str ] | None:
		"""Extract articles from a web page.

		Purpose:
		    Fetch the page and return distinct article element text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted article element text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url,
				self.html or '', [ 'scrape_articles' ] ).get( 'Articles', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_articles( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_headings( self, uri: str ) -> List[ str ] | None:
		"""Extract headings from a web page.

		Purpose:
		    Fetch the page and return distinct heading text from h1 through h6 elements in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted heading text from h1 through h6 elements, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url, self.html or '', [
					'scrape_headings' ] ).get( 'Headings', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_headings( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_divisions( self, uri: str ) -> List[ str ] | None:
		"""Extract divisions from a web page.

		Purpose:
		    Fetch the page and return distinct div element text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted div element text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url, self.html or '',
				[ 'scrape_divisions' ] ).get( 'Divisions', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_divisions( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_sections( self, uri: str ) -> List[ str ] | None:
		"""Extract sections from a web page.

		Purpose:
		    Fetch the page and return distinct section element text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted section element text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url, self.html or '',
				[ 'scrape_sections' ] ).get( 'Sections', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_sections( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_blockquotes( self, uri: str ) -> List[ str ] | None:
		"""Extract blockquotes from a web page.

		Purpose:
		    Fetch the page and return distinct blockquote text in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted blockquote text, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url,
				self.html or '', [ 'scrape_blockquotes' ] ).get( 'Blockquotes', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_blockquotes( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_hyperlinks( self, uri: str ) -> List[ str ] | None:
		"""Extract hyperlinks from a web page.

		Purpose:
		    Fetch the page and return distinct absolute HTTP or HTTPS anchor URLs in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted absolute HTTP or HTTPS anchor URLs, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url,
				self.html or '', [ 'scrape_hyperlinks' ] ).get( 'Hyperlinks', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_hyperlinks( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def scrape_images( self, uri: str ) -> List[ str ] | None:
		"""Extract images from a web page.

		Purpose:
		    Fetch the page and return distinct absolute HTTP or HTTPS image source URLs in discovery order.

		Args:
		    uri (str): Nonblank target page URL.

		Returns:
		    List[str]: Extracted absolute HTTP or HTTPS image source URLs, or an empty list when none are found.

		Raises:
		    Error: Raised when validation, retrieval, or extraction fails.
		"""
		try:
			throw_if( 'uri', uri )
			self.url = uri.strip( )
			self.fetch( self.url, time=self.timeout or 10 )
			return self.extract_structured_data( self.url,
				self.html or '', [ 'scrape_images' ] ).get( 'Images', [ ] )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'scrape_images( self, uri: str ) -> List[ str ] | None'
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible tool schema with a service-qualified description, object properties, and required parameter names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str]): Nonempty list of required argument names.

		Returns:
		    Dict[str, Any]: Function name, description, and nested parameter schema.

		Raises:
		    Error: Raised when required inputs are missing or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			throw_if( 'required', required )
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetcher'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class WebCrawler( WebFetcher ):
	"""HTTP and browser-based web crawler.

	Purpose:
	    Extends WebFetcher with optional Playwright rendering, per-page extraction, and breadth-first link traversal governed by page and depth limits.

	Attributes:
	    use_playwright (Optional[bool]): Whether browser rendering is enabled.
	    browser_context (Optional[Any]): Reserved browser context state, initialized to None.
	    raw_url (Optional[str]): URL supplied to browser retrieval.
	    raw_html (Optional[str]): HTML returned by browser rendering.
	    pages (Optional[List[Dict[str, Any]]]): Page dictionaries from the latest crawl.
	    summary (Optional[Dict[str, Any]]): Latest traversal statistics and configuration.
	    response (Optional[Response]): Inherited HTTP response state.
	    headers (Optional[Dict[str, Any]]): Inherited or replacement HTTP headers.
	"""
	use_playwright: Optional[ bool ]
	browser_context: Optional[ Any ]
	raw_url: Optional[ str ]
	raw_html: Optional[ str ]
	pages: Optional[ List[ Dict[ str, Any ] ] ]
	summary: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, headers: Optional[ Dict[ str, str ] ]=None, use_playwright: bool=False ) -> None:
		"""Initialize crawler configuration.

		Purpose:
		    Initialize inherited HTTP and extraction state, configure optional browser rendering, and initialize page and summary collections.

		Args:
		    headers (Optional[Dict[str, str]]): Replacement HTTP headers. Defaults to None to retain inherited headers; a missing User-Agent is supplied from configuration.
		    use_playwright (bool): Enable Chromium rendering instead of HTTP retrieval. Defaults to False.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.browser_context = None
		self.raw_url = None
		self.raw_html = None
		self.response = None
		self.pages = [ ]
		self.summary = { }
		self.use_playwright = use_playwright
		if headers is not None:
			self.headers = headers
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = cfg.AGENTS
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed crawler members.

		Purpose:
		    Provide the explicit member list used by dir(), including inherited extraction helpers and crawler operations.

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'use_playwright', 'browser_context', 'raw_url', 'raw_html', 'pages', 'summary',
				'fetch', 'html_to_text', 'coerce_items', 'extract_title', 'truncate_text',
				'normalize_url', 'same_domain', 'extract_links', 'extract_structured_data',
				'render_with_playwright', 'scrape_page', 'crawl' ]
	
	def fetch( self, url: str, time: int=10 ) -> Result | None:
		"""Retrieve a page using HTTP or Chromium.

		Purpose:
		    Use Playwright when enabled, storing rendered HTML and a parsed document; otherwise delegate to WebFetcher.fetch.

		Args:
		    url (str): Nonblank page URL.
		    time (int): Timeout in seconds. Defaults to 10.

		Returns:
		    Result | None: HTTP response wrapper in HTTP mode; None in browser mode.

		Raises:
		    Error: Raised when validation, retrieval, or rendering fails.
		"""
		try:
			throw_if( 'url', url )
			if self.use_playwright:
				self.url = url.strip( )

				throw_if( 'time', time )
				self.timeout = time
				self.raw_url = self.url
				self.raw_html = self.render_with_playwright( self.url, timeout=self.timeout )
				self.html = self.raw_html or ''
				self.soup = BeautifulSoup( self.html, 'html.parser' )
				return None
			
			return super( ).fetch( url=url, time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebCrawler'
			exception.method = 'fetch( self, url: str, time: int=10 ) -> Result | None'
			Logger( ).write( exception )
			raise exception
	
	def render_with_playwright( self, url: str, timeout: int=15 ) -> str:
		"""Render a web page using Chromium.

		Purpose:
		    Launch Chromium, navigate to the requested URL, wait for network idle, obtain the rendered HTML, and close the browser on success.

		Args:
		    url (str): Nonblank page URL.
		    timeout (int): Navigation and network-idle timeout in seconds for each operation. Defaults to 15.

		Returns:
		    str: Rendered page HTML.

		Raises:
		    Error: Raised when validation or browser rendering fails.
		"""
		try:


			throw_if( 'url', url )

			throw_if( 'timeout', timeout )

			self.url = url
			self.timeout = timeout
			with sync_playwright( ) as p:
				browser = p.chromium.launch( )
				page = browser.new_page( )
				page.goto( self.url, timeout=self.timeout * 1000 )
				page.wait_for_load_state( 'networkidle', timeout=self.timeout * 1000 )
				html = page.content( )
				browser.close( )
				return html
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebCrawler'
			exception.method = 'render_with_playwright( self, url: str, timeout: int=15 ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def scrape_page( self, url: str, include_title: bool=True, include_basic_text: bool=True, include_raw_html: bool=False, selected_methods: Optional[ List[ str ] ]=None, request_timeout: int=10, max_bytes: int=1000000 ) -> Dict[ str, Any ]:
		"""Retrieve and extract a single page.

		Purpose:
		    Collect response metadata, links, selected text and HTML groups, and content size. Truncate HTML by UTF-8 byte count before extraction when the positive limit is exceeded; record failures in the result.

		Args:
		    url (str): Page URL.
		    include_title (bool): Include page title. Defaults to True.
		    include_basic_text (bool): Include extracted plain text. Defaults to True.
		    include_raw_html (bool): Include raw or truncated HTML. Defaults to False.
		    selected_methods (Optional[List[str]]): Registered extraction method names. Defaults to None, selecting no groups.
		    request_timeout (int): Retrieval timeout in seconds. Defaults to 10.
		    max_bytes (int): Maximum retained HTML bytes before extraction; nonpositive values disable truncation. Defaults to 1000000.

		Returns:
		    Dict[str, Any]: Page URL, status, encoding, title, plain text, HTML, links, structured data, errors, original byte count, and truncation flag.
		"""
		page_result: Dict[ str, Any ] = { 'url': url, 'status_code': None, 'encoding': None,
				'title': '', 'plain_text': '', 'raw_html': '', 'links_discovered': [ ], 'data': { },
				'errors': [ ], 'content_bytes': 0, 'truncated_by_max_bytes': False, }
		
		try:
			throw_if( 'include_title', include_title )

			throw_if( 'include_basic_text', include_basic_text )

			throw_if( 'include_raw_html', include_raw_html )

			methods = selected_methods or [ ]
			self.fetch( url=url, time=request_timeout )
			raw_html = self.html or ''
			if self.response is not None:
				page_result[ 'status_code' ] = self.response.status_code
				page_result[ 'encoding' ] = self.response.encoding
			else:
				page_result[ 'status_code' ] = 200
				page_result[ 'encoding' ] = 'rendered'
			raw_bytes = raw_html.encode( 'utf-8', errors='ignore' )
			page_result[ 'content_bytes' ] = len( raw_bytes )
			if max_bytes > 0 and len( raw_bytes ) > max_bytes:
				raw_html = raw_bytes[ : max_bytes ].decode( 'utf-8', errors='ignore' )
				page_result[ 'truncated_by_max_bytes' ] = True
				page_result[ 'errors' ].append(
					f'Response exceeded max bytes and was truncated to {max_bytes} bytes.' )
			
			page_result[ 'links_discovered' ] = self.extract_links( url, raw_html )
			if include_title:
				page_result[ 'title' ] = self.extract_title( raw_html )
			
			if include_basic_text:
				try:
					page_result[ 'plain_text' ] = self.html_to_text( raw_html ) or ''
				except Exception as exc:
					page_result[ 'errors' ].append( f'Basic Text: {str( exc )}' )
			
			if include_raw_html:
				page_result[ 'raw_html' ] = raw_html
			
			page_result[ 'data' ] = self.extract_structured_data( url=url, html=raw_html,
				selected_methods=methods )
			return page_result
		except Exception as exc:
			page_result[ 'errors' ].append( f'Fetch: {str( exc )}' )
			return page_result
	
	def crawl( self, seed_url: str, include_title: bool=True, include_basic_text: bool=True, include_raw_html: bool=False, selected_methods: Optional[ List[ str ] ]=None, recursive: bool=False, max_depth: int=1, max_pages: int=10, same_domain_only: bool=True, request_timeout: int=10, delay_seconds: float=0.25, max_bytes: int=1000000 ) -> Dict[ str, Any ]:
		"""Traverse and extract linked pages.

		Purpose:
		    Process the normalized seed and optionally discovered links in breadth-first order, respecting depth, page, and host limits. Store collected pages and a traversal summary.

		Args:
		    seed_url (str): Nonblank absolute HTTP or HTTPS starting URL.
		    include_title (bool): Include page titles. Defaults to True.
		    include_basic_text (bool): Include plain text. Defaults to True.
		    include_raw_html (bool): Include HTML. Defaults to False.
		    selected_methods (Optional[List[str]]): Registered extraction method names. Defaults to None, selecting no groups.
		    recursive (bool): Follow discovered links. Defaults to False.
		    max_depth (int): Maximum link depth from the seed at depth zero. Defaults to 1.
		    max_pages (int): Maximum number of processed pages. Defaults to 10.
		    same_domain_only (bool): Restrict discovered links to the seed network location. Defaults to True.
		    request_timeout (int): Per-page retrieval timeout in seconds. Defaults to 10.
		    delay_seconds (float): Delay in seconds when queued pages remain; nonpositive values disable waiting. Defaults to 0.25.
		    max_bytes (int): Per-page retained HTML byte limit; nonpositive values disable truncation. Defaults to 1000000.

		Returns:
		    Dict[str, Any]: Pages with traversal depths and summary metrics, configuration, timing, and visited and skipped URLs.

		Raises:
		    Error: Raised when crawl setup or traversal fails; individual page failures are recorded in page results.
		"""
		try:
			throw_if( 'include_title', include_title )

			throw_if( 'include_basic_text', include_basic_text )

			throw_if( 'include_raw_html', include_raw_html )

			throw_if( 'recursive', recursive )

			throw_if( 'same_domain_only', same_domain_only )

			throw_if( 'seed_url', seed_url )
			started_at = dt.datetime.now( )
			normalized_seed = self.normalize_url( seed_url, seed_url )
			if not normalized_seed:
				raise ValueError( 'A valid absolute URL is required.' )
			
			methods = selected_methods or [ ]
			queue: List[ Tuple[ str, int ] ] = [ (normalized_seed, 0) ]
			visited: set[ str ] = set( )
			enqueued: set[ str ] = { normalized_seed }
			skipped_urls: List[ str ] = [ ]
			pages: List[ Dict[ str, Any ] ] = [ ]
			index = 0
			while index < len( queue ) and len( pages ) < max_pages:
				current_url, depth = queue[ index ]
				index += 1
				if current_url in visited:
					continue
				visited.add( current_url )
				page_result = self.scrape_page( url=current_url, include_title=include_title,
					include_basic_text=include_basic_text, include_raw_html=include_raw_html,
					selected_methods=methods, request_timeout=request_timeout,
					max_bytes=max_bytes )
				
				page_result[ 'depth' ] = depth
				pages.append( page_result )
				if delay_seconds > 0 and index < len( queue ):
					time.sleep( delay_seconds )
				
				if not recursive:
					continue
				
				if depth >= max_depth:
					continue
				
				discovered_links = page_result.get( 'links_discovered', [ ] ) or [ ]
				for next_url in discovered_links:
					if len( pages ) + (len( queue ) - index) >= max_pages:
						break
					
					if not next_url or next_url in visited or next_url in enqueued:
						continue
					
					if same_domain_only and not self.same_domain( normalized_seed, next_url ):
						skipped_urls.append( next_url )
						continue
					
					queue.append( (next_url, depth + 1) )
					enqueued.add( next_url )
			
			finished_at = dt.datetime.now( )
			error_count = sum( len( page.get( 'errors', [ ] ) or [ ] ) for page in pages )
			total_bytes = sum( page.get( 'content_bytes', 0 ) or 0 for page in pages )
			self.pages = pages
			self.summary = { 'mode': 'recursive' if recursive else 'single-page',
					'seed_url': normalized_seed, 'pages_processed': len( pages ),
					'pages_visited': len( visited ), 'pages_skipped': len( skipped_urls ),
					'pages_enqueued_remaining': max( 0, len( queue ) - index ),
					'errors': error_count, 'total_content_bytes': total_bytes,
					'recursive_requested': recursive, 'max_depth': max_depth,
					'max_pages': max_pages, 'same_domain_only': same_domain_only,
					'request_timeout': request_timeout,
					'delay_seconds': delay_seconds, 'max_bytes_per_page': (
						max_bytes ),
					'use_playwright': self.use_playwright,
					'started_at': started_at.isoformat( ), 'finished_at': finished_at.isoformat( ),
					'elapsed_seconds': round( (finished_at - started_at).total_seconds( ), 3 ),
					'visited_urls': list( visited ), 'skipped_urls': skipped_urls, }
			
			return { 'pages': self.pages, 'summary': self.summary }
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebCrawler'
			exception.method = 'crawl( self, seed_url: str, include_title: bool=True, include_basic_text: bool=True, include_raw_html: bool=False, selected_methods: Optional[ List[ str ] ]=None, recursive: bool=False, max_depth: int=1, max_pages: int=10, same_domain_only: bool=True, request_timeout: int=10, delay_seconds: float=0.25, max_bytes: int=1000000 ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception

class ArXiv( Fetcher ):
	"""arXiv scholarly document retriever.

	Purpose:
	    Configure and invoke LangChain ArxivRetriever for scholarly search, using constructor defaults unless per-request overrides are supplied.

	Attributes:
	    fetcher (Optional[ArxivRetriever]): Active LangChain arXiv retriever.
	    documents (Optional[List[Document]]): Documents returned by the latest search.
	    max_documents (Optional[int]): Default document limit.
	    full_documents (Optional[bool]): Default full-document retrieval flag.
	    include_metadata (Optional[bool]): Default all-metadata flag.
	    query (Optional[str]): Latest scholarly search query.
	    load_max_docs (int): Effective document limit for the current search.
	    get_full_documents (bool): Effective full-document retrieval flag.
	    load_all_available_meta (bool): Effective all-metadata flag.
	"""
	fetcher: Optional[ ArxivRetriever ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	full_documents: Optional[ bool ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	load_max_docs: int
	get_full_documents: bool
	load_all_available_meta: bool
	
	def __init__( self, max_documents: int=5, full_documents: bool=False, include_metadata: bool=False ) -> None:
		"""Initialize arXiv retrieval defaults.

		Purpose:
		    Store the default document limit, full-document and metadata options, and initialize retriever and response state without issuing a search.

		Args:
		    max_documents (int): Default maximum documents, 5.
		    full_documents (bool): Request full documents by default. Defaults to False.
		    include_metadata (bool): Load all available metadata by default. Defaults to False.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.fetcher = None
		self.documents = None
		self.query = None
		self.max_documents = max_documents
		self.full_documents = full_documents
		self.include_metadata = include_metadata
		self.load_max_docs = self.max_documents
		self.get_full_documents = self.full_documents
		self.load_all_available_meta = self.include_metadata
	
	def fetch( self, question: str, max_documents: int | None=None, full_documents: bool | None=None, include_metadata: bool | None=None ) -> List[ Document ] | None:
		"""Retrieve scholarly documents from arXiv.

		Purpose:
		    Resolve optional overrides against constructor defaults, construct ArxivRetriever, invoke it with the query, and store the resulting documents.

		Args:
		    question (str): Nonblank scholarly search query.
		    max_documents (int | None): Document limit; defaults to None to use the constructor setting.
		    full_documents (bool | None): Full-document flag; defaults to None to use the constructor setting.
		    include_metadata (bool | None): All-metadata flag; defaults to None to use the constructor setting.

		Returns:
		    List[Document]: Documents returned by the LangChain retriever.

		Raises:
		    Error: Raised when validation, retriever construction, or retrieval fails.
		"""
		try:
			throw_if( 'question', question )
			self.query = question
			
			if max_documents is None:
				self.load_max_docs = self.max_documents
			else:
				throw_if( 'max_documents', max_documents )
				self.load_max_docs = max_documents
			
			if full_documents is None:
				self.get_full_documents = self.full_documents
			else:
				throw_if( 'full_documents', full_documents )
				self.get_full_documents = full_documents
			
			if include_metadata is None:
				self.load_all_available_meta = self.include_metadata
			else:
				throw_if( 'include_metadata', include_metadata )
				self.load_all_available_meta = include_metadata
			
			self.fetcher = ArxivRetriever( load_max_docs=self.load_max_docs,
				get_full_documents=self.get_full_documents,
				load_all_available_meta=self.load_all_available_meta )
			
			self.documents = self.fetcher.invoke( self.query )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ArXiv'
			exception.method = 'fetch( self, question: str, max_documents: int | None=None, full_documents: bool | None=None, include_metadata: bool | None=None ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class GoogleDrive( Fetcher ):
	"""Google Drive document retriever.

	Purpose:
	    Construct and invoke LangChain GoogleDriveRetriever using configured credential paths, query templates, folder and MIME filters, and document or snippet mode.

	Attributes:
	    fetcher (Optional[GoogleDriveRetriever]): Active LangChain Google Drive retriever.
	    documents (Optional[List[Document]]): Documents returned by the latest retrieval.
	    num_results (Optional[int]): Requested maximum results.
	    folder_id (Optional[str]): Target Drive folder identifier.
	    template (Optional[str]): Selected Drive query template.
	    query (Optional[str]): Original search text.
	    mime_type (Optional[str]): Optional MIME filter.
	    mode (Optional[str]): Document or snippet retrieval mode.
	    credentials_path (Optional[str]): Configured Google account credential file path.
	    token_path (Optional[str]): Configured Drive token file path.
	    retriever_kwargs (Optional[Dict[str, Any]]): Options assembled for retriever construction.
	    invoke_query (Optional[str]): Effective query, including wildcard fallback where supported.
	"""
	fetcher: Optional[ GoogleDriveRetriever ]
	documents: Optional[ List[ Document ] ]
	num_results: Optional[ int ]
	folder_id: Optional[ str ]
	template: Optional[ str ]
	query: Optional[ str ]
	mime_type: Optional[ str ]
	mode: Optional[ str ]
	credentials_path: Optional[ str ]
	token_path: Optional[ str ]
	retriever_kwargs: Optional[ Dict[ str, Any ] ]
	invoke_query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize Google Drive retrieval state.

		Purpose:
		    Load configured folder and credential paths, initialize a ten-result document search using the gdrive-query template, and prepare empty retriever state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.fetcher = None
		self.documents = [ ]
		self.query = ''
		self.template = 'gdrive-query'
		self.folder_id = cfg.GOOGLE_DRIVE_FOLDER_ID or 'root'
		self.num_results = 10
		self.mime_type = None
		self.mode = 'documents'
		self.credentials_path = cfg.GOOGLE_ACCOUNT_FILE
		self.token_path = cfg.GOOGLE_DRIVE_TOKEN_PATH
		self.retriever_kwargs = { }
		self.invoke_query = ''
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed Google Drive members.

		Purpose:
		    Provide the explicit member names returned by dir(), including option properties and retrieval.

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'fetcher', 'documents', 'num_results', 'folder_id', 'template', 'query',
				'mime_type', 'mode', 'credentials_path', 'token_path', 'retriever_kwargs',
				'invoke_query', 'mime_options', 'template_options', 'mode_options', 'fetch' ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Return MIME filter options.

		Purpose:
		    Expose MIME types supported by the wrapper, including a blank option for no filter.

		Returns:
		    List[str]: Supported MIME filter strings.
		"""
		return [ '', 'text/text', 'text/plain', 'text/html', 'text/csv', 'text/markdown',
				'image/png', 'image/jpeg', 'application/epub+zip', 'application/pdf',
				'application/rtf', 'application/vnd.google-apps.document',
				'application/vnd.google-apps.presentation',
				'application/vnd.google-apps.spreadsheet', 'application/vnd.google.colaboratory',
				'application/vnd.openxmlformats-officedocument.presentationml.presentation',
				'application/vnd.openxmlformats-officedocument.wordprocessingml.document', ]
	
	@property
	def template_options( self ) -> List[ str ]:
		"""Return Drive query templates.

		Purpose:
		    Expose the supported folder, text, name, and MIME query templates.

		Returns:
		    List[str]: Supported template identifiers.
		"""
		return [ 'gdrive-all-in-folder', 'gdrive-query', 'gdrive-by-name',
				'gdrive-query-in-folder', 'gdrive-mime-type', 'gdrive-mime-type-in-folder',
				'gdrive-query-with-mime-type', 'gdrive-query-with-mime-type-and-folder', ]
	
	@property
	def mode_options( self ) -> List[ str ]:
		"""Return retrieval mode options.

		Purpose:
		    Expose the document and snippet modes accepted by fetch.

		Returns:
		    List[str]: documents and snippets.
		"""
		return [ 'documents', 'snippets' ]
	
	def fetch( self, question: str, folder_id: str='root', results: int=10, template: str='gdrive-query', mime_type: str=None, mode: str='documents' ) -> List[ Document ] | None:
		"""Retrieve matching Google Drive documents.

		Purpose:
		    Validate the selected template, mode, result range, and optional MIME filter. Use a wildcard for an empty query in supported folder or MIME templates, construct the retriever with configured credential paths, and invoke it.

		Args:
		    question (str): Search text; may be empty for all-in-folder or MIME-only templates.
		    folder_id (str): Nonblank target folder identifier. Defaults to root.
		    results (int): Requested result count from 1 through 100. Defaults to 10.
		    template (str): Supported query template identifier. Defaults to gdrive-query.
		    mime_type (str | None): Optional supported MIME filter. Defaults to None.
		    mode (str): documents or snippets. Defaults to documents.

		Returns:
		    List[Document]: Documents returned by the LangChain retriever.

		Raises:
		    Error: Raised when validation, retriever construction, or retrieval fails.
		"""
		try:
			throw_if( 'template', template )
			throw_if( 'folder_id', folder_id )
			throw_if( 'results', results )
			throw_if( 'mode', mode )
			self.query = question
			self.folder_id = folder_id
			self.num_results = results
			self.template = template
			self.mime_type = mime_type
			self.mode = mode
			if self.num_results < 1 or self.num_results > 100:
				raise ValueError( 'results must be between 1 and 100.' )
			
			if self.template not in self.template_options:
				raise ValueError( f'Unsupported Google Drive template: {self.template}' )
			
			if self.mode not in self.mode_options:
				raise ValueError( f'Unsupported Google Drive mode: {self.mode}' )
			
			if self.mime_type and self.mime_type not in self.mime_options:
				raise ValueError( f'Unsupported Google Drive MIME type: {self.mime_type}' )
			
			self.invoke_query = self.query
			if not self.invoke_query:
				if self.template in ('gdrive-all-in-folder', 'gdrive-mime-type',
						'gdrive-mime-type-in-folder'):
					self.invoke_query = '*'
				else:
					throw_if( 'question', question )
			
			self.retriever_kwargs = { 'folder_id': self.folder_id, 'template': self.template,
					'num_results': self.num_results, 'mode': self.mode, }
			
			if self.mime_type:
				self.retriever_kwargs[ 'mime_type' ] = self.mime_type
			
			if self.credentials_path:
				self.retriever_kwargs[ 'credentials_path' ] = self.credentials_path
			
			if self.token_path:
				self.retriever_kwargs[ 'token_path' ] = self.token_path
			
			self.fetcher = GoogleDriveRetriever( **self.retriever_kwargs )
			self.documents = self.fetcher.invoke( self.invoke_query )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleDrive'
			exception.method = "fetch( self, question: str, folder_id: str='root', results: int=10, template: str='gdrive-query', mime_type: str=None, mode: str='documents' ) -> List[ Document ] | None"
			Logger( ).write( exception )
			raise exception

class Wikipedia( Fetcher ):
	"""Wikipedia article retriever.

	Purpose:
	    Construct and invoke LangChain WikipediaRetriever to obtain article documents with configurable language, document limit, and metadata options.

	Attributes:
	    fetcher (Optional[WikipediaRetriever]): Active LangChain Wikipedia retriever.
	    documents (Optional[List[Document]]): Documents returned by the latest search.
	    max_documents (Optional[int]): Default document limit.
	    include_metadata (Optional[bool]): Default all-metadata flag.
	    language (Optional[str]): Default Wikipedia language code.
	    query (Optional[str]): Latest search query.
	    lang (str): Effective language code for the current search.
	    load_max_docs (int): Effective document limit for the current search.
	    load_all_available_meta (bool): Effective all-metadata flag.
	"""
	fetcher: Optional[ WikipediaRetriever ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	include_metadata: Optional[ bool ]
	language: Optional[ str ]
	query: Optional[ str ]
	lang: str
	load_max_docs: int
	load_all_available_meta: bool
	
	def __init__( self, language: str='en', max_documents: int=5, include_metadata: bool=False ) -> None:
		"""Initialize Wikipedia retrieval defaults.

		Purpose:
		    Store default language, document limit, and metadata options without issuing a search.

		Args:
		    language (str): Default Wikipedia language code, en.
		    max_documents (int): Default maximum documents, 5.
		    include_metadata (bool): Load all available metadata by default. Defaults to False.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.fetcher = None
		self.documents = None
		self.query = None
		self.language = language
		self.max_documents = max_documents
		self.include_metadata = include_metadata
		self.lang = self.language
		self.load_max_docs = self.max_documents
		self.load_all_available_meta = self.include_metadata
	
	def fetch( self, question: str, language: str | None=None, max_documents: int | None=None, include_metadata: bool | None=None ) -> List[ Document ] | None:
		"""Retrieve Wikipedia article documents.

		Purpose:
		    Resolve per-request overrides against constructor defaults, construct WikipediaRetriever, invoke it with the search text, and store the resulting documents.

		Args:
		    question (str): Nonblank Wikipedia search query.
		    language (str | None): Wikipedia language code; defaults to None to use the constructor setting.
		    max_documents (int | None): Document limit; defaults to None to use the constructor setting.
		    include_metadata (bool | None): All-metadata flag; defaults to None to use the constructor setting.

		Returns:
		    List[Document]: Documents returned by the LangChain retriever.

		Raises:
		    Error: Raised when validation, retriever construction, or retrieval fails.
		"""
		try:
			throw_if( 'question', question )
			self.query = question
			
			if language is None:
				self.lang = self.language
			else:
				throw_if( 'language', language )
				self.lang = language
			
			if max_documents is None:
				self.load_max_docs = self.max_documents
			else:
				throw_if( 'max_documents', max_documents )
				self.load_max_docs = max_documents
			
			if include_metadata is None:
				self.load_all_available_meta = self.include_metadata
			else:
				throw_if( 'include_metadata', include_metadata )
				self.load_all_available_meta = include_metadata
			
			self.fetcher = WikipediaRetriever( lang=self.lang, load_max_docs=self.load_max_docs,
				load_all_available_meta=self.load_all_available_meta )
			
			self.documents = self.fetcher.invoke( input=self.query )
			return self.documents
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Wikipedia'
			exception.method = 'fetch( self, question: str, language: str | None=None, max_documents: int | None=None, include_metadata: bool | None=None ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class TheNews( Fetcher ):
	"""The News API article and source retriever.

	Purpose:
	    Construct endpoint-specific news queries using configured credentials and return decoded
	    JSON for article searches, top news, grouped headlines, and source listings.

	Attributes:
	    agents (Optional[str]): Configured HTTP user-agent.
	    url (Optional[str]): Base news API URL.
	    response (Optional[Response]): Latest HTTP response.
	    api_key (Optional[str]): News API token loaded from configuration.
	    params (Optional[Dict[str, Any]]): Parameters submitted to the latest endpoint, including
	        the API token.
	    endpoint (Optional[str]): Selected news endpoint.
	    limit (Optional[int]): Requested article count.
	    page (Optional[int]): Requested result page.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    result (Optional[Result]): Inherited result placeholder; fetch returns decoded JSON
	        directly.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    query (str): Article search expression.
	    language (str): Language filter.
	    categories (str): Included article or source categories.
	    exclude_categories (str): Excluded categories.
	    locale (str): Locale filter for top news and headlines.
	    domains (str): Included publisher domains.
	    exclude_domains (str): Excluded publisher domains.
	    source_ids (str): Included source identifiers.
	    exclude_source_ids (str): Excluded source identifiers.
	    published_after (str): Lower publication-date filter.
	    published_before (str): Upper publication-date filter.
	    published_on (str): Exact publication-date filter.
	    sort (str): Article ordering expression.
	    include_similar (bool): Whether grouped headlines include similar articles.
	    headlines_per_category (int): Requested headlines per category.
	    request_url (str): Complete URL of the selected news endpoint.
	"""
	agents: Optional[ str ]
	url: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	endpoint: Optional[ str ]
	limit: Optional[ int ]
	page: Optional[ int ]
	query: str
	language: str
	categories: str
	exclude_categories: str
	locale: str
	domains: str
	exclude_domains: str
	source_ids: str
	exclude_source_ids: str
	published_after: str
	published_before: str
	published_on: str
	sort: str
	include_similar: bool
	headlines_per_category: int
	request_url: str
	
	def __init__( self ) -> None:
		"""Initialize news retrieval configuration.

		Purpose:
		    Load the API token and user-agent from configuration, set the news endpoint defaults, and
		    prepare HTTP headers and request state without sending a request.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.timeout = 10
		self.url = 'https://api.thenewsapi.com/v1/news'
		self.response = None
		self.result = None
		self.headers = { }
		self.params = { }
		self.endpoint = 'all'
		self.limit = 10
		self.page = 1
		self.query = ''
		self.language = 'en'
		self.categories = ''
		self.exclude_categories = ''
		self.locale = ''
		self.domains = ''
		self.exclude_domains = ''
		self.source_ids = ''
		self.exclude_source_ids = ''
		self.published_after = ''
		self.published_before = ''
		self.published_on = ''
		self.sort = 'published_at'
		self.include_similar = True
		self.headlines_per_category = 6
		self.request_url = ''
		self.api_key = cfg.THENEWS_API_KEY
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed news retrieval members.

		Purpose:
		    Provide the explicit attribute and method names returned by dir().

		Returns:
		    List[str]: Ordered news retrieval member names.
		"""
		return [ 'api_key', 'url', 'timeout', 'headers', 'endpoint', 'limit', 'page', 'params',
				'query', 'language', 'categories', 'exclude_categories', 'locale', 'domains', 'exclude_domains', 'source_ids', 'exclude_source_ids', 'published_after', 'published_before', 'published_on', 'sort', 'include_similar', 'headlines_per_category', 'request_url',
				'fetch', ]
	
	def fetch( self, endpoint: str='all', query: str='', language: str='en', categories: str='', exclude_categories: str='', locale: str='', domains: str='', exclude_domains: str='', source_ids: str='', exclude_source_ids: str='', published_after: str='', published_before: str='', published_on: str='', sort: str='published_at', limit: int=10, page: int=1, include_similar: bool=True, headlines_per_category: int=6, time: int=10 ) -> Dict[ str, Any ]:
		"""Retrieve news articles, headlines, or sources.

		Purpose:
		    Select the requested endpoint, assemble only its applicable filters with the configured
		    API token, send an HTTP GET request, check its status, and return decoded JSON.

		Args:
		    endpoint (str): Supported endpoint: all, top, headlines, or sources. Defaults to 'all'.
		    query (str): Article search expression for all and top. Defaults to an empty string.
		    language (str): Language filter. Defaults to 'en'.
		    categories (str): Included categories for all, top, and sources. Defaults to an empty
		        string.
		    exclude_categories (str): Excluded categories for all, top, and sources. Defaults to an
		        empty string.
		    locale (str): Locale filter for top and headlines. Defaults to an empty string.
		    domains (str): Included publisher domains for article and headline retrieval. Defaults to
		        an empty string.
		    exclude_domains (str): Excluded publisher domains for article and headline retrieval.
		        Defaults to an empty string.
		    source_ids (str): Included source identifiers for article and headline retrieval. Defaults
		        to an empty string.
		    exclude_source_ids (str): Excluded source identifiers for article and headline retrieval.
		        Defaults to an empty string.
		    published_after (str): Lower publication-date filter for all and top. Defaults to an empty
		        string.
		    published_before (str): Upper publication-date filter for all and top. Defaults to an
		        empty string.
		    published_on (str): Exact publication-date filter for article and headline retrieval.
		        Defaults to an empty string.
		    sort (str): Article sort expression for all and top. Defaults to 'published_at'.
		    limit (int): Requested article count for all and top. Defaults to 10.
		    page (int): Requested result page for all, top, and sources. Defaults to 1.
		    include_similar (bool): Include similar articles in headlines mode. Defaults to True.
		    headlines_per_category (int): Requested headlines per category in headlines mode. Defaults
		        to 6.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Decoded provider JSON, whose structure depends on the endpoint.

		Raises:
		    Error: Raised when validation, HTTP retrieval, or JSON decoding fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			self.endpoint = endpoint
			throw_if( 'time', time )
			self.timeout = time
			throw_if( 'limit', limit )
			self.limit = limit
			throw_if( 'page', page )
			self.page = page
			self.query = query
			self.language = language
			self.categories = categories
			self.exclude_categories = exclude_categories
			self.locale = locale
			self.domains = domains
			self.exclude_domains = exclude_domains
			self.source_ids = source_ids
			self.exclude_source_ids = exclude_source_ids
			self.published_after = published_after
			self.published_before = published_before
			self.published_on = published_on
			self.sort = sort
			
			valid_endpoints = { 'all', 'top', 'headlines', 'sources' }
			if self.endpoint not in valid_endpoints:
				raise ValueError( f"Unsupported endpoint '{self.endpoint}'. "
				                  f"Supported endpoints: "
				                  f"{', '.join( sorted( valid_endpoints ) )}." )
			
			self.params = { 'api_token': self.api_key }
			if self.endpoint in ('all', 'top'):
				if self.query:
					self.params[ 'search' ] = self.query
				
				if self.language:
					self.params[ 'language' ] = self.language
				
				if self.categories:
					self.params[ 'categories' ] = self.categories
				
				if self.exclude_categories:
					self.params[ 'exclude_categories' ] = self.exclude_categories
				
				if self.domains:
					self.params[ 'domains' ] = self.domains
				
				if self.exclude_domains:
					self.params[ 'exclude_domains' ] = self.exclude_domains
				
				if self.source_ids:
					self.params[ 'source_ids' ] = self.source_ids
				
				if self.exclude_source_ids:
					self.params[ 'exclude_source_ids' ] = self.exclude_source_ids
				
				if self.published_after:
					self.params[ 'published_after' ] = self.published_after
				
				if self.published_before:
					self.params[ 'published_before' ] = self.published_before
				
				if self.published_on:
					self.params[ 'published_on' ] = self.published_on
				
				if self.sort:
					self.params[ 'sort' ] = self.sort
				
				self.params[ 'limit' ] = self.limit
				self.params[ 'page' ] = self.page
				if self.endpoint == 'top' and self.locale:
					self.params[ 'locale' ] = self.locale
			
			elif self.endpoint == 'headlines':
				if self.locale:
					self.params[ 'locale' ] = self.locale
				
				if self.domains:
					self.params[ 'domains' ] = self.domains
				
				if self.exclude_domains:
					self.params[ 'exclude_domains' ] = self.exclude_domains
				
				if self.source_ids:
					self.params[ 'source_ids' ] = self.source_ids
				
				if self.exclude_source_ids:
					self.params[ 'exclude_source_ids' ] = self.exclude_source_ids
				
				if self.language:
					self.params[ 'language' ] = self.language
				
				if self.published_on:
					self.params[ 'published_on' ] = self.published_on
				
				throw_if( 'headlines_per_category', headlines_per_category )
				self.headlines_per_category = headlines_per_category
				self.params[ 'headlines_per_category' ] = self.headlines_per_category
				
				throw_if( 'include_similar', include_similar )
				self.include_similar = include_similar
				self.params[ 'include_similar' ] = 'true' if self.include_similar else 'false'
			
			elif self.endpoint == 'sources':
				if self.categories:
					self.params[ 'categories' ] = self.categories
				
				if self.exclude_categories:
					self.params[ 'exclude_categories' ] = self.exclude_categories
				
				if self.language:
					self.params[ 'language' ] = self.language
				
				self.params[ 'page' ] = self.page
			
			self.request_url = f'{self.url}/{self.endpoint}'
			self.response = requests.get( url=self.request_url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			
			self.response.raise_for_status( )
			return self.response.json( )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'TheNews'
			exception.method = "fetch( self, endpoint: str='all', query: str='', language: str='en', categories: str='', exclude_categories: str='', locale: str='', domains: str='', exclude_domains: str='', source_ids: str='', exclude_source_ids: str='', published_after: str='', published_before: str='', published_on: str='', sort: str='published_at', limit: int=10, page: int=1, include_similar: bool=True, headlines_per_category: int=6, time: int=10 ) -> Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception

class GoogleSearch( Fetcher ):
	"""Google Custom Search JSON API retriever.

	Purpose:
	    Submit web or image search requests to the configured programmable search engine and retain
	    the decoded provider payload without transforming search records.

	Attributes:
	    keywords (Optional[str]): Latest search query.
	    url (Optional[str]): Custom Search JSON API endpoint.
	    re_tag (Optional[Pattern]): Compiled markup-removal expression retained by the wrapper.
	    re_ws (Optional[Pattern]): Compiled whitespace expression retained by the wrapper.
	    response (Optional[Response]): Latest HTTP response.
	    api_key (Optional[str]): Google API key loaded from configuration.
	    cse_id (Optional[str]): Configured or subsequently supplied programmable search engine
	        identifier.
	    params (Optional[Dict[str, Any]]): Latest request parameters, including credentials and
	        filters.
	    payload (Optional[Any]): Latest decoded JSON response.
	    results (Optional[int]): Requested number of results.
	    start (Optional[int]): One-based result start index.
	    exact_terms (Optional[str]): Required exact phrase.
	    exclude_terms (Optional[str]): Terms excluded from results.
	    file_type (Optional[str]): Requested file extension.
	    date_restrict (Optional[str]): Relative date restriction expression.
	    gl (Optional[str]): Country code used for geographic relevance.
	    lr (Optional[str]): Language restriction expression.
	    safe (Optional[str]): SafeSearch setting.
	    search_type (Optional[str]): Search type, including image search.
	    site_search (Optional[str]): Site to include or exclude.
	    site_search_filter (Optional[str]): Site inclusion or exclusion setting.
	    sort (Optional[str]): Provider sort expression.
	    img_size (Optional[str]): Image size filter.
	    img_type (Optional[str]): Image type filter.
	    img_color_type (Optional[str]): Image color type filter.
	    img_dominant_color (Optional[str]): Dominant image color filter.
	    agents (Optional[str]): Configured HTTP user-agent.
	    result (Optional[Dict[str, Any]]): Decoded JSON response assigned by fetch.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	"""
	keywords: Optional[ str ]
	url: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	cse_id: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	results: Optional[ int ]
	start: Optional[ int ]
	exact_terms: Optional[ str ]
	exclude_terms: Optional[ str ]
	file_type: Optional[ str ]
	date_restrict: Optional[ str ]
	gl: Optional[ str ]
	lr: Optional[ str ]
	safe: Optional[ str ]
	search_type: Optional[ str ]
	site_search: Optional[ str ]
	site_search_filter: Optional[ str ]
	sort: Optional[ str ]
	img_size: Optional[ str ]
	img_type: Optional[ str ]
	img_color_type: Optional[ str ]
	img_dominant_color: Optional[ str ]
	agents: Optional[ str ]
	result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize Custom Search configuration.

		Purpose:
		    Load the Google API key and search engine identifier from configuration, initialize the
		    endpoint, regular expressions, headers, and search defaults without issuing a request.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.cse_id = cfg.GOOGLE_CSE_ID
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = 'https://customsearch.googleapis.com/customsearch/v1'
		self.headers = { }
		self.timeout = 10
		self.keywords = None
		self.params = { }
		self.payload = { }
		self.response = None
		self.result = { }
		self.results = 10
		self.start = 1
		self.exact_terms = None
		self.exclude_terms = None
		self.file_type = None
		self.date_restrict = None
		self.gl = None
		self.lr = None
		self.safe = 'off'
		self.search_type = None
		self.site_search = None
		self.site_search_filter = None
		self.sort = None
		self.img_size = None
		self.img_type = None
		self.img_color_type = None
		self.img_dominant_color = None
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed Custom Search members.

		Purpose:
		    Provide the explicit member names returned by dir().

		Returns:
		    List[str]: Ordered search configuration and retrieval member names.
		"""
		return [ 'keywords', 'url', 'timeout', 'headers', 'fetch', 'api_key', 'response',
				'payload', 'result', 'cse_id', 'params', 'agents', 'results',
				'start', 'exact_terms', 'exclude_terms', 'file_type', 'date_restrict',
				'gl', 'lr', 'safe', 'search_type', 'site_search', 'site_search_filter',
				'sort', 'img_size', 'img_type', 'img_color_type', 'img_dominant_color' ]
	
	def fetch( self, keywords: str, results: int=10, start: int=1, exact_terms: str='', exclude_terms: str='', file_type: str='', date_restrict: str='', gl: str='', lr: str='', safe: str='off', search_type: str='', site_search: str='', site_search_filter: str='', sort: str='', img_size: str='', img_type: str='', img_color_type: str='', img_dominant_color: str='', time: int=10, cse_id: str=None ) ->  Dict[ str, Any ] | None:
		"""Execute a Google Custom Search request.

		Purpose:
		    Validate the nonblank query and requested result bounds, retain any supplied search engine
		    override, assemble search filters, send an HTTP GET request, and store the decoded JSON
		    response. Image filters are included only for image searches.

		Args:
		    keywords (str): Nonblank search query.
		    results (int): Requested number of results, from 1 through 10. Defaults to 10.
		    start (int): One-based start index, from 1 through 91. Defaults to 1.
		    exact_terms (str): Required exact phrase. Defaults to an empty string.
		    exclude_terms (str): Excluded search terms. Defaults to an empty string.
		    file_type (str): File extension restriction. Defaults to an empty string.
		    date_restrict (str): Relative date restriction expression. Defaults to an empty string.
		    gl (str): Country code for geographic relevance. Defaults to an empty string.
		    lr (str): Language restriction expression. Defaults to an empty string.
		    safe (str): SafeSearch setting. Defaults to 'off'.
		    search_type (str): Search type; 'image' enables image filters. Defaults to an empty string
		        for web search.
		    site_search (str): Site to include or exclude. Defaults to an empty string.
		    site_search_filter (str): Site inclusion or exclusion setting. Defaults to an empty
		        string.
		    sort (str): Provider sort expression. Defaults to an empty string.
		    img_size (str): Image size filter. Defaults to an empty string.
		    img_type (str): Image type filter. Defaults to an empty string.
		    img_color_type (str): Image color type filter. Defaults to an empty string.
		    img_dominant_color (str): Dominant image color filter. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 10.
		    cse_id (str | None): Optional search engine override retained by the instance; defaults to
		        None to keep the current identifier.

		Returns:
		    Dict[str, Any]: Decoded Custom Search JSON response, without record normalization.

		Raises:
		    Error: Raised when validation, HTTP retrieval, or JSON decoding fails.
		"""
		try:

			throw_if( 'results', results )


			throw_if( 'start', start )

			throw_if( 'keywords', keywords )

			
			
			if cse_id:
				throw_if( 'cse_id', cse_id )
				self.cse_id = cse_id
			
			throw_if( 'time', time )
			self.timeout = time
			self.keywords = keywords
			self.results = results
			self.start = start
			if self.results < 1:
				raise ValueError( 'results must be greater than or equal to 1.' )
			
			if self.results > 10:
				raise ValueError( 'results cannot exceed 10 for one Google Custom Search '
				                  'request.' )
			
			if self.start < 1:
				raise ValueError( 'start must be greater than or equal to 1.' )
			
			if self.start > 91:
				raise ValueError( 'start cannot exceed 91 when requesting up to 10 results.' )
			
			self.exact_terms = exact_terms
			self.exclude_terms = exclude_terms
			self.file_type = file_type
			self.date_restrict = date_restrict
			self.gl = gl
			self.lr = lr
			self.safe = safe
			self.search_type = search_type
			self.site_search = site_search
			self.site_search_filter = site_search_filter
			self.sort = sort
			self.img_size = img_size
			self.img_type = img_type
			self.img_color_type = img_color_type
			self.img_dominant_color = img_dominant_color
			self.params = { 'q': self.keywords, 'key': self.api_key, 'cx': self.cse_id,
					'num': self.results, 'start': self.start, 'safe': self.safe }
			
			if self.exact_terms:
				self.params[ 'exactTerms' ] = self.exact_terms
			
			if self.exclude_terms:
				self.params[ 'excludeTerms' ] = self.exclude_terms
			
			if self.file_type:
				self.params[ 'fileType' ] = self.file_type
			
			if self.date_restrict:
				self.params[ 'dateRestrict' ] = self.date_restrict
			
			if self.gl:
				self.params[ 'gl' ] = self.gl
			
			if self.lr:
				self.params[ 'lr' ] = self.lr
			
			if self.search_type:
				self.params[ 'searchType' ] = self.search_type
			
			if self.site_search:
				self.params[ 'siteSearch' ] = self.site_search
			
			if self.site_search_filter:
				self.params[ 'siteSearchFilter' ] = self.site_search_filter
			
			if self.sort:
				self.params[ 'sort' ] = self.sort
			
			if self.search_type.lower( ) == 'image':
				if self.img_size:
					self.params[ 'imgSize' ] = self.img_size
				
				if self.img_type:
					self.params[ 'imgType' ] = self.img_type
				
				if self.img_color_type:
					self.params[ 'imgColorType' ] = self.img_color_type
				
				if self.img_dominant_color:
					self.params[ 'imgDominantColor' ] = self.img_dominant_color
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			self.result = self.payload
			return self.result
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleSearch'
			exception.method = "fetch( self, keywords: str, results: int=10, start: int=1, exact_terms: str='', exclude_terms: str='', file_type: str='', date_restrict: str='', gl: str='', lr: str='', safe: str='off', search_type: str='', site_search: str='', site_search_filter: str='', sort: str='', img_size: str='', img_type: str='', img_color_type: str='', img_dominant_color: str='', time: int=10, cse_id: str=None ) ->  Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception

class GoogleMaps( Fetcher ):
	"""Google Maps geocoding, address validation, and directions adapter.

	Purpose:
	    Perform forward and reverse geocoding, validate postal address lines, and retrieve the
	    first available route using configured Google credentials.

	Attributes:
	    file_path (Optional[str]): Reserved filesystem path, initialized to None.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    num_results (Optional[int]): Reserved result count, initialized to None.
	    api_key (Optional[str]): Google API key loaded from configuration.
	    mode (Optional[str]): Requested directions travel mode.
	    latitude (Optional[float]): Current geographic latitude in decimal degrees.
	    longitude (Optional[float]): Current geographic longitude in decimal degrees.
	    coordinates (Optional[Tuple[float, float]]): Current latitude and longitude pair.
	    address (Optional[str]): Address submitted to or returned by geocoding.
	    address_lines (Optional[List[str]]): Postal address lines submitted for address validation.
	    origin (Optional[str]): Directions starting location.
	    destination (Optional[str]): Directions destination.
	    directions (Optional[Dict[str, Any]]): First route returned by the directions API, or an
	        empty dictionary.
	    params (Optional[Dict[str, Any]]): Latest HTTP query parameters, including the API key.
	    payload (Optional[Any]): Latest decoded provider response or address-validation request
	        body.
	    result (Optional[Any]): Latest coordinates, formatted address, validation response, or
	        route.
	    agents (Optional[str]): Configured HTTP user-agent.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    url (Optional[str]): Latest service endpoint URL.
	    response (Optional[Response]): Latest HTTP response.
	"""
	file_path: Optional[ str ]
	headers: Optional[ Dict[ str, Any ] ]
	num_results: Optional[ int ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	address: Optional[ str ]
	address_lines: Optional[ List[ str ] ]
	origin: Optional[ str ]
	destination: Optional[ str ]
	directions: Optional[ Dict[ str, Any ] ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Any ]
	agents: Optional[ str ]
	timeout: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize Google Maps service configuration.

		Purpose:
		    Load the configured Google API key and user-agent, establish JSON headers and a ten-second
		    timeout, and initialize geocoding, address validation, and route state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.headers = { }
		self.params = { }
		self.longitude = None
		self.latitude = None
		self.mode = None
		self.url = None
		self.file_path = None
		self.coordinates = None
		self.address = None
		self.address_lines = [ ]
		self.origin = None
		self.destination = None
		self.directions = { }
		self.response = None
		self.payload = { }
		self.result = None
		self.timeout = 10
		self.num_results = None
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed Google Maps members.

		Purpose:
		    Provide the explicit attribute and service method names returned by dir().

		Returns:
		    List[str]: Ordered Google Maps member names.
		"""
		return [ 'file_path', 'headers', 'num_results', 'api_key', 'mode', 'latitude', 'longitude',
				'coordinates', 'address', 'address_lines', 'origin', 'destination', 'directions',
				'params', 'payload', 'result', 'agents', 'timeout', 'response', 'url',
				'geocode_location', 'geocode_coordinates', 'validate_address',
				'request_directions',
				'create_schema' ]
	
	def geocode_location( self, address: str ) -> Tuple[ float, float ]:
		"""Resolve an address to geographic coordinates.

		Purpose:
		    Submit an address to the Geocoding API, check HTTP and provider status, select the first
		    result, and store its latitude and longitude.

		Args:
		    address (str): Nonblank address or location text.

		Returns:
		    Tuple[float, float]: Latitude followed by longitude in decimal degrees.

		Raises:
		    Error: Raised when validation, HTTP retrieval, provider status, response structure, or
		        coordinate extraction fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			self.url = 'https://maps.googleapis.com/maps/api/geocode/json'
			self.params = { 'address': self.address, 'key': self.api_key }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'Google Geocoding returned an invalid response payload.' )
			status = str( self.payload.get( 'status', '' ) or '' ).strip( )
			error_message = str( self.payload.get( 'error_message', '' ) or '' ).strip( )
			if status and status != 'OK':
				detail = f' — {error_message}' if error_message else ''
				raise ValueError( f'Google Geocoding failed: {status}{detail}' )
			results = self.payload.get( 'results', [ ] )
			
			if not results:
				raise ValueError( 'Google Geocoding returned no results for the supplied '
				                  'address.' )
			
			location = results[ 0 ].get( 'geometry', { } ).get( 'location', { } )
			self.latitude = location[ 'lat' ]
			self.longitude = location[ 'lng' ]
			self.coordinates = (self.latitude, self.longitude)
			self.result = self.coordinates
			return self.coordinates
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = 'geocode_location( self, address: str ) -> Tuple[ float, float ]'
			Logger( ).write( exception )
			raise exception
	
	def geocode_coordinates( self, lat: float, long: float ) -> str | None:
		"""Resolve geographic coordinates to an address.

		Purpose:
		    Submit the latitude and longitude to the Geocoding API and retain the formatted address
		    from the first returned result.

		Args:
		    lat (float): Latitude in decimal degrees.
		    long (float): Longitude in decimal degrees.

		Returns:
		    str: Formatted address from the first result, or an empty string if that result has no
		        formatted address.

		Raises:
		    Error: Raised when validation, HTTP retrieval, JSON decoding, or address extraction fails,
		        including when no results are returned.
		"""
		try:
			throw_if( 'lat', lat )
			self.latitude = lat
			throw_if( 'long', long )
			self.longitude = long
			self.coordinates = (self.latitude, self.longitude)
			self.url = 'https://maps.googleapis.com/maps/api/geocode/json'
			self.params = { 'latlng': f'{self.latitude},{self.longitude}', 'key': self.api_key }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			results = self.payload.get( 'results', [ ] ) if isinstance( self.payload, dict ) else\
				[ ]
			
			if not results:
				raise ValueError( 'No address results were returned for the supplied '
				                  'coordinates.' )
			
			self.address = str( results[ 0 ].get( 'formatted_address', '' ) )
			self.result = self.address
			
			return self.address
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = 'geocode_coordinates( self, lat: float, long: float ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def validate_address( self, address: List[ str ] ) -> Dict[ Any, Any ] | None:
		"""Submit postal address lines for validation.

		Purpose:
		    Send the supplied address lines as JSON to the Address Validation API and retain its
		    decoded response.

		Args:
		    address (List[str]): Nonempty list of postal address lines.

		Returns:
		    Dict[Any, Any]: Decoded Address Validation API response, including any provider validation
		        findings.

		Raises:
		    Error: Raised when validation, HTTP retrieval, or JSON decoding fails.
		"""
		try:
			throw_if( 'address', address )
			self.address_lines = address
			self.url = 'https://addressvalidation.googleapis.com/v1:validateAddress'
			
			self.params = { 'key': self.api_key }
			self.payload = { 'address': { 'addressLines': self.address_lines } }
			
			self.response = requests.post( url=self.url, params=self.params, json=self.payload,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.result = self.response.json( )
			
			return self.result
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = 'validate_address( self, address: List[ str ] ) -> Dict[ Any, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def request_directions( self, origin: str, destination: str, mode: str='driving' ) -> Dict[ str, Any ] | None:
		"""Retrieve directions between two locations.

		Purpose:
		    Submit origin, destination, and travel mode to the Directions API and retain the first
		    route when one is available.

		Args:
		    origin (str): Nonblank starting location.
		    destination (str): Nonblank destination location.
		    mode (str): Nonblank provider travel mode. Defaults to 'driving'.

		Returns:
		    Dict[str, Any]: First route dictionary, or an empty dictionary when no route is returned.

		Raises:
		    Error: Raised when validation, HTTP retrieval, JSON decoding, or route extraction fails.
		"""
		try:
			throw_if( 'origin', origin )
			self.origin = origin
			throw_if( 'destination', destination )
			self.destination = destination
			throw_if( 'mode', mode )
			self.mode = mode
			self.url = 'https://maps.googleapis.com/maps/api/directions/json'
			self.params = { 'origin': self.origin, 'destination': self.destination,
					'mode': self.mode, 'key': self.api_key }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			routes = self.payload.get( 'routes', [ ] ) if isinstance( self.payload, dict ) else [ ]
			self.directions = routes[ 0 ] if routes else { }
			self.result = self.directions
			return self.result
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = "request_directions( self, origin: str, destination: str, mode: str='driving' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible tool schema with the function name, service-qualified
		    description, parameter properties, and required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, description, and nested parameter schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class GoogleWeather( Fetcher ):
	"""Google Weather API retrieval adapter.

	Purpose:
	    Geocode addresses through GoogleMaps and retrieve current conditions, hourly and daily
	    forecasts, hourly history, or public weather alerts. Preserve the decoded provider object
	    and request metadata while omitting the API key from returned parameters.

	Attributes:
	    gmaps (Optional[GoogleMaps]): GoogleMaps instance used for forward geocoding.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    api_key (Optional[str]): Weather API key loaded from configuration.
	    mode (Optional[str]): Selected weather retrieval operation.
	    latitude (Optional[float]): Resolved latitude in decimal degrees.
	    longitude (Optional[float]): Resolved longitude in decimal degrees.
	    coordinates (Optional[Tuple[float, float]]): Resolved latitude and longitude pair.
	    address (Optional[str]): Latest address submitted for geocoding.
	    params (Optional[Dict[str, Any]]): Latest request parameters, including the API key.
	    response (Optional[Response]): Latest weather HTTP response.
	    payload (Optional[Dict[str, Any]]): Latest decoded weather JSON object.
	    result (Optional[Dict[str, Any]]): Packaged weather response with request metadata.
	    units_system (Optional[str]): Requested measurement system.
	    language_code (Optional[str]): Requested response language.
	    hours (Optional[int]): Requested forecast or historical hours.
	    days (Optional[int]): Requested forecast days.
	    path (Optional[str]): Weather API endpoint path.
	    request_url (str): Complete weather endpoint URL.
	    agents (Optional[str]): Configured HTTP user-agent.
	    url (Optional[str]): Base Google Weather API URL.
	    timeout (Optional[int]): Weather HTTP request timeout in seconds.
	"""
	gmaps: Optional[ GoogleMaps ]
	headers: Optional[ Dict[ str, Any ] ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	address: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	payload: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	units_system: Optional[ str ]
	language_code: Optional[ str ]
	hours: Optional[ int ]
	days: Optional[ int ]
	path: Optional[ str ]
	request_url: str
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize weather retrieval configuration.

		Purpose:
		    Load the configured weather API key, create the GoogleMaps geocoder, and initialize JSON
		    headers, a ten-second timeout, metric units, English language, and empty response state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_WEATHER_API_KEY
		self.headers = { }
		self.gmaps = GoogleMaps( )
		self.mode = None
		self.url = 'https://weather.googleapis.com/v1'
		self.longitude = 0.0
		self.latitude = 0.0
		self.coordinates = (0.0, 0.0)
		self.address = None
		self.params = { }
		self.response = None
		self.payload = { }
		self.result = { }
		self.timeout = 10
		self.units_system = 'METRIC'
		self.language_code = 'en'
		self.hours = None
		self.days = None
		self.path = None
		self.request_url = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'url', 'timeout', 'headers', 'gmaps', 'mode', 'latitude', 'longitude',
				'coordinates', 'address', 'params', 'response', 'payload', 'result',
				'units_system',
				'language_code', 'hours', 'days', 'path', 'request_url', 'agents', 'resolve_coordinates',
				'request', 'package_response', 'fetch_current', 'fetch_hourly_forecast',
				'fetch_daily_forecast', 'fetch_hourly_history', 'fetch_alerts' ]
	
	def resolve_coordinates( self, address: str ) -> Tuple[ float, float ]:
		"""Resolve an address to coordinates.

		Purpose:
		    Delegate forward geocoding to GoogleMaps and retain the resolved geographic coordinates.

		Args:
		    address (str): Nonblank location or address to geocode.

		Returns:
		    Tuple[float, float]: Latitude followed by longitude in decimal degrees.

		Raises:
		    Error: Raised when validation or geocoding fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			lat, lng = self.gmaps.geocode_location( address=self.address )
			self.latitude = lat
			self.longitude = lng
			self.coordinates = (self.latitude, self.longitude)
			return self.coordinates
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = 'resolve_coordinates( self, address: str ) -> Tuple[ float, float ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, path: str, params: Dict[ str, Any ], time: int=10 ) -> Dict[ str, Any ]:
		"""Execute a weather endpoint request.

		Purpose:
		    Remove None and blank string parameters, add the configured API key, perform an HTTP GET
		    request, check its status, and package the decoded JSON object.

		Args:
		    path (str): Nonblank weather API endpoint path relative to the base URL.
		    params (Dict[str, Any]): Nonempty dictionary of endpoint query parameters.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'path', path )
			self.path = path
			throw_if( 'params', params )

			throw_if( 'time', time )
			self.timeout = time
			self.params = { }
			
			for key, value in params.items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.params[ 'key' ] = self.api_key
			self.request_url = f'{self.url}/{self.path.lstrip( "/" )}'
			self.response = requests.get( url=self.request_url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'Google Weather response must be a JSON object.' )

			self.result = { 'mode': self.mode, 'url': self.request_url,
				'params': { key: value for key, value in self.params.items( )
					if key != 'key' }, 'data': self.payload }
			
			return self.result
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = 'request( self, path: str, params: Dict[ str, Any ], time: int=10 ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self ) -> Dict[ str, Any ]:
		"""Return the packaged weather response.

		Purpose:
		    Return the existing result when it contains data; otherwise reconstruct it from the
		    retained endpoint, request parameters, and payload.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when response packaging fails.
		"""
		try:
			if not isinstance( self.result, dict ):
				self.result = { }
			
			if 'data' not in self.result:
				self.result = { 'mode': self.mode,
					'url': f'{self.url}/{self.path}' if self.path else self.url,
					'params': { key: value for key, value in self.params.items( )
						if key != 'key' }, 'data': self.payload or { } }
			
			return self.result
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = 'package_response( self ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_current( self, address: str, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve current weather conditions.

		Purpose:
		    Resolve the address and request current conditions using the specified measurement system
		    and language.

		Args:
		    address (str): Nonblank location or address to geocode.
		    units_system (str): Nonblank provider measurement-system setting. Defaults to 'METRIC'.
		    language_code (str): Nonblank response language code. Defaults to 'en'.
		    time (int): Weather HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			throw_if( 'units_system', units_system )
			throw_if( 'language_code', language_code )

			self.mode = 'current'
			self.units_system = units_system
			self.language_code = language_code
			throw_if( 'time', time )
			self.timeout = time
			self.latitude, self.longitude = self.resolve_coordinates( self.address )
			
			self.params = { 'location.latitude': self.latitude,
					'location.longitude': self.longitude, 'unitsSystem': self.units_system,
					'languageCode': self.language_code }
			
			self.request( path='currentConditions:lookup', params=self.params, time=self.timeout )
			
			return self.package_response( )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = "fetch_current( self, address: str, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_hourly_forecast( self, address: str, hours: int=24, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve an hourly weather forecast.

		Purpose:
		    Validate the forecast horizon, geocode the address, and request hourly forecast data.

		Args:
		    address (str): Nonblank location or address to geocode.
		    hours (int): Forecast horizon from 1 through 240 hours. Defaults to 24.
		    units_system (str): Nonblank provider measurement-system setting. Defaults to 'METRIC'.
		    language_code (str): Nonblank response language code. Defaults to 'en'.
		    time (int): Weather HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			throw_if( 'hours', hours )
			throw_if( 'units_system', units_system )
			throw_if( 'language_code', language_code )

			self.mode = 'hourly_forecast'
			self.hours = hours
			self.units_system = units_system
			self.language_code = language_code
			throw_if( 'time', time )
			self.timeout = time
			if self.hours < 1 or self.hours > 240:
				raise ValueError( 'hours must be between 1 and 240.' )
			
			self.latitude, self.longitude = self.resolve_coordinates( self.address )
			self.params = { 'location.latitude': self.latitude,
					'location.longitude': self.longitude, 'hours': self.hours,
					'unitsSystem': self.units_system, 'languageCode': self.language_code }
			
			self.request( path='forecast/hours:lookup', params=self.params, time=self.timeout )
			
			return self.package_response( )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = "fetch_hourly_forecast( self, address: str, hours: int=24, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_daily_forecast( self, address: str, days: int=5, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve a daily weather forecast.

		Purpose:
		    Validate the forecast horizon, geocode the address, and request daily forecast data.

		Args:
		    address (str): Nonblank location or address to geocode.
		    days (int): Forecast horizon from 1 through 10 days. Defaults to 5.
		    units_system (str): Nonblank provider measurement-system setting. Defaults to 'METRIC'.
		    language_code (str): Nonblank response language code. Defaults to 'en'.
		    time (int): Weather HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			throw_if( 'days', days )
			throw_if( 'units_system', units_system )
			throw_if( 'language_code', language_code )

			self.mode = 'daily_forecast'
			self.days = days
			self.units_system = units_system
			self.language_code = language_code
			throw_if( 'time', time )
			self.timeout = time
			if self.days < 1 or self.days > 10:
				raise ValueError( 'days must be between 1 and 10.' )
			
			self.latitude, self.longitude = self.resolve_coordinates( self.address )
			self.params = { 'location.latitude': self.latitude,
					'location.longitude': self.longitude, 'days': self.days,
					'unitsSystem': self.units_system, 'languageCode': self.language_code }
			
			self.request( path='forecast/days:lookup', params=self.params, time=self.timeout )
			
			return self.package_response( )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = "fetch_daily_forecast( self, address: str, days: int=5, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_hourly_history( self, address: str, hours: int=24, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve historical hourly weather.

		Purpose:
		    Validate the historical horizon, geocode the address, and request hourly weather history.

		Args:
		    address (str): Nonblank location or address to geocode.
		    hours (int): Historical horizon from 1 through 24 hours. Defaults to 24.
		    units_system (str): Nonblank provider measurement-system setting. Defaults to 'METRIC'.
		    language_code (str): Nonblank response language code. Defaults to 'en'.
		    time (int): Weather HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			throw_if( 'hours', hours )
			throw_if( 'units_system', units_system )
			throw_if( 'language_code', language_code )

			self.mode = 'hourly_history'
			self.hours = hours
			self.units_system = units_system
			self.language_code = language_code
			throw_if( 'time', time )
			self.timeout = time
			if self.hours < 1 or self.hours > 24:
				raise ValueError( 'hours must be between 1 and 24 for hourly history.' )
			
			self.latitude, self.longitude = self.resolve_coordinates( self.address )
			self.params = { 'location.latitude': self.latitude,
					'location.longitude': self.longitude, 'hours': self.hours,
					'unitsSystem': self.units_system, 'languageCode': self.language_code }
			
			self.request( path='history/hours:lookup', params=self.params, time=self.timeout )
			
			return self.package_response( )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = "fetch_hourly_history( self, address: str, hours: int=24, units_system: str='METRIC', language_code: str='en', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_alerts( self, address: str, language_code: str='en', time: int=10 ) -> (Dict[ str, Any ] | None):
		"""Retrieve public weather alerts.

		Purpose:
		    Resolve the address and request public alerts in the selected language.

		Args:
		    address (str): Nonblank location or address to geocode.
		    language_code (str): Nonblank response language code. Defaults to 'en'.
		    time (int): Weather HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Response containing mode, endpoint URL, request parameters without the API
		        key, and decoded weather data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			throw_if( 'language_code', language_code )

			
			self.mode = 'alerts'
			self.language_code = language_code
			throw_if( 'time', time )
			self.timeout = time
			self.latitude, self.longitude = self.resolve_coordinates( self.address )
			self.params = { 'location.latitude': self.latitude,
					'location.longitude': self.longitude, 'languageCode': self.language_code }
			
			self.request( path='publicAlerts:lookup', params=self.params, time=self.timeout )
			
			return self.package_response( )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = "fetch_alerts( self, address: str, language_code: str='en', time: int=10 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception

class NavalObservatory( Fetcher ):
	"""U.S. Naval Observatory celestial-navigation adapter.

	Purpose:
	    Validate the observation date, time, and geographic coordinates, retrieve
	    celestial-navigation data, and package the provider response with the submitted parameters.

	Attributes:
	    base_url (Optional[str]): Base USNO API URL.
	    url (Optional[str]): Latest celestial-navigation endpoint URL.
	    params (Optional[Dict[str, Any]]): Submitted date, time, and coordinate parameters.
	    date_value (Optional[str]): Validated observation date.
	    time_value (Optional[str]): Validated observation time.
	    latitude (Optional[float]): Observation latitude in decimal degrees.
	    longitude (Optional[float]): Observation longitude in decimal degrees.
	    location_label (Optional[str]): Descriptive location label included in the result.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Dict[str, Any]): Decoded celestial-navigation JSON object assigned after
	        retrieval.
	    result (Optional[Dict[str, Any]]): Packaged celestial-navigation response.
	"""
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	date_value: Optional[ str ]
	time_value: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	location_label: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize celestial-navigation state.

		Purpose:
		    Configure the USNO API base URL and user-agent, initialize date and time placeholders, and
		    set the initial coordinates to 38.9072 latitude and -77.0369 longitude.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://aa.usno.navy.mil/api'
		self.url = None
		self.params = { }
		self.date_value = ''
		self.time_value = ''
		self.latitude = 38.9072
		self.longitude = -77.0369
		self.location_label = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'url', 'params', 'date_value', 'time_value', 'latitude', 'longitude',
				'location_label', 'fetch_celnav', 'fetch', 'create_schema' ]
	
	def validate_date( self, date_value: str ) -> str:
		"""Validate an observation date.

		Purpose:
		    Parse the supplied date and enforce the supported year range of 1800 through 2050.

		Args:
		    date_value (str): Observation date parsed as YYYY-MM-DD, with year 1800 through 2050.

		Returns:
		    str: Original date string after validation.

		Raises:
		    Error: Raised when the date is missing, invalid, or outside the supported year range.
		"""
		try:
			throw_if( 'date_value', date_value )
			calendar_date = dt.datetime.strptime( date_value, '%Y-%m-%d' )
			if calendar_date.year < 1800 or calendar_date.year > 2050:
				raise ValueError( 'Celestial navigation dates must be between 1800 and 2050.' )
			return date_value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = 'validate_date( self, date_value: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_time( self, time_value: str ) -> str:
		"""Validate an observation time.

		Purpose:
		    Match a 24-hour time with optional seconds and one to six fractional second digits.

		Args:
		    time_value (str): Observation time in UTC, formatted HH:MM or HH:MM:SS with optional one
		        to six fractional second digits.

		Returns:
		    str: Original time string after validation.

		Raises:
		    Error: Raised when the time is missing or does not match the accepted format.
		"""
		try:
			throw_if( 'time_value', time_value )
			pattern = (r'^(?:[01]\d|2[0-3]):[0-5]\d'
			           r'(?:'
			           r':[0-5]\d(?:\.\d{1,6})?'
			           r')?$')
			
			if not re.fullmatch( pattern, time_value ):
				raise ValueError( "Invalid time format. Use HH:MM, HH:MM:SS, or HH:MM:SS.S" )
			
			return time_value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = 'validate_time( self, time_value: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_coordinates( self, latitude: float, longitude: float ) -> tuple[ float, float ]:
		"""Validate geographic coordinates.

		Purpose:
		    Check that latitude and longitude are finite and within their respective geographic
		    bounds.

		Args:
		    latitude (float): Finite latitude in decimal degrees, from -90 through 90.
		    longitude (float): Finite longitude in decimal degrees, from -180 through 180.

		Returns:
		    tuple[float, float]: Original latitude and longitude values in that order.

		Raises:
		    Error: Raised when either coordinate is missing, nonfinite, or outside its geographic
		        bounds.
		"""
		try:
			throw_if( 'latitude', latitude )
			throw_if( 'longitude', longitude )
			lat = latitude
			lon = longitude
			if not math.isfinite( lat ) or not -90.0 <= lat <= 90.0:
				raise ValueError( 'Latitude must be between -90 and 90 and finite.' )
			if not math.isfinite( lon ) or not -180.0 <= lon <= 180.0:
				raise ValueError( 'Longitude must be between -180 and 180 and finite.' )
			return lat, lon
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = 'validate_coordinates( self, latitude: float, longitude: float ) -> tuple[ float, float ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_celnav( self, date_value: str, time_value: str, latitude: float, longitude: float, location_label: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve celestial-navigation data.

		Purpose:
		    Validate observation inputs, send the date, time, and coordinate pair to the USNO celnav
		    endpoint, check HTTP status and provider errors, and package the JSON object.

		Args:
		    date_value (str): Observation date parsed as YYYY-MM-DD, with year 1800 through 2050.
		    time_value (str): Observation time in UTC, formatted HH:MM or HH:MM:SS with optional one
		        to six fractional second digits.
		    latitude (float): Finite latitude in decimal degrees, from -90 through 90.
		    longitude (float): Finite longitude in decimal degrees, from -180 through 180.
		    location_label (str): Optional descriptive location label retained in the result. Defaults
		        to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, location label, and decoded
		        celestial-navigation data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'time', time )
			self.timeout = time

			self.date_value = self.validate_date( date_value )
			self.time_value = self.validate_time( time_value )
			self.latitude, self.longitude = self.validate_coordinates( latitude=latitude,
				longitude=longitude )
			self.location_label = location_label
			self.url = f'{self.base_url}/celnav'
			self.params = { 'date': self.date_value, 'time': self.time_value,
			                'coords': f'{self.latitude},{self.longitude}' }
			
			self.response = requests.get( url=self.url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'USNO celestial navigation response must be a JSON object.' )

			if self.payload.get( 'error' ):
				raise ValueError( f"USNO celestial navigation error: {self.payload[ 'error' ]}" )

			self.result = { 'mode': 'celnav', 'url': self.url, 'params': self.params,
				'location_label': self.location_label, 'data': self.payload }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = "fetch_celnav( self, date_value: str, time_value: str, latitude: float, longitude: float, location_label: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='celnav', date_value: str='', time_value: str='', latitude: float=0.0, longitude: float=0.0, location_label: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a celestial-navigation request.

		Purpose:
		    Normalize the mode to lowercase and delegate to fetch_celnav. Reject modes other than
		    celnav.

		Args:
		    mode (str): Retrieval mode. Defaults to 'celnav'; a blank mode also selects celnav.
		    date_value (str): Observation date parsed as YYYY-MM-DD, with year 1800 through 2050.
		        Defaults to an empty string, which requires replacement for retrieval.
		    time_value (str): Observation time in UTC, formatted HH:MM or HH:MM:SS with optional one
		        to six fractional second digits. Defaults to an empty string, which requires
		        replacement for retrieval.
		    latitude (float): Finite latitude in decimal degrees, from -90 through 90. Defaults to
		        0.0.
		    longitude (float): Finite longitude in decimal degrees, from -180 through 180. Defaults to
		        0.0.
		    location_label (str): Optional descriptive location label retained in the result. Defaults
		        to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, location label, and decoded
		        celestial-navigation data.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			active_mode = ( mode or 'celnav' ).strip( ).lower( )
			if active_mode == 'celnav':
				return self.fetch_celnav( date_value=date_value, time_value=time_value,
					latitude=latitude, longitude=longitude, location_label=location_label,
					time=time )
			
			raise ValueError( 'Unsupported mode.' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = "fetch( self, mode: str='celnav', date_value: str='', time_value: str='', latitude: float=0.0, longitude: float=0.0, location_label: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a celestial-navigation function schema.

		Purpose:
		    Construct a JSON-compatible function schema with a service-qualified description and
		    object parameter definitions.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, description, and nested parameter schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ), 'description': (f'{description.strip( )} '
			                                                    f'This function uses the '
			                                                    f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class SatelliteCenter( Fetcher ):
	"""NASA Satellite Situation Center retrieval adapter.

	Purpose:
	    Retrieve observatory and ground-station inventories through SscWs and spacecraft location
	    data through the SSC REST service, retaining the returned dictionaries and checking
	    reported provider status.

	Attributes:
	    ssc (Optional[SscWs]): SscWs client used for inventory retrieval.
	    url (Optional[str]): Base SSC REST service URL.
	    params (Optional[Dict[str, Any]]): Latest spacecraft-location query parameters.
	    observatories (Optional[List[Dict[str, Any]]]): Observatory records extracted from the
	        latest observatory response.
	    ground_stations (Optional[List[Dict[str, Any]]]): Ground-station records extracted from the
	        latest station response.
	    timeout (Optional[int]): Current request timeout in seconds.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    observatory_ids (str): Comma-separated SSC spacecraft identifiers.
	    start_time (str): Beginning UTC timestamp for location retrieval.
	    end_time (str): Ending UTC timestamp for location retrieval.
	    coordinate_systems (str): Comma-separated SSC coordinate system identifiers.
	    resolution_factor (int): Requested sampling resolution factor.
	    request_url (str): Complete spacecraft-location endpoint URL.
	    response (Optional[Response]): Latest direct HTTP response from location retrieval.
	    result (Optional[Dict[str, Any]]): Latest SSC inventory or spacecraft-location response
	        dictionary.
	"""
	ssc: Optional[ SscWs ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	observatories: Optional[ List[ Dict[ str, Any ] ] ]
	ground_stations: Optional[ List[ Dict[ str, Any ] ] ]
	timeout: Optional[ int ]
	observatory_ids: str
	start_time: str
	end_time: str
	coordinate_systems: str
	resolution_factor: int
	request_url: str
	
	def __init__( self ) -> None:
		"""Initialize SSC retrieval configuration.

		Purpose:
		    Configure the SSC REST endpoint, JSON headers, twenty-second timeout, default GSE
		    coordinate system, resolution factor of one, and empty inventory state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.ssc = None
		self.url = 'https://sscweb.gsfc.nasa.gov/WS/sscr/2'
		self.params = { }
		self.observatories = [ ]
		self.ground_stations = [ ]
		self.timeout = 20
		self.observatory_ids = ''
		self.start_time = ''
		self.end_time = ''
		self.coordinate_systems = 'gse'
		self.resolution_factor = 1
		self.request_url = ''
		self.headers = { }
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'url', 'timeout', 'headers', 'fetch_observatories', 'fetch_ground_stations',
				'fetch_locations', 'fetch', ]
	
	def fetch_observatories( self ) -> Dict[ str, Any ] | None:
		"""Retrieve the spacecraft observatory inventory.

		Purpose:
		    Construct SscWs with the current user-agent and timeout, request observatory metadata,
		    check the returned dictionary and status, and retain its Observatory records.

		Returns:
		    Dict[str, Any]: Complete SSC observatory response dictionary.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			self.ssc = SscWs( user_agent=self.agents, timeout=self.timeout )
			self.result = self.ssc.get_observatories( )
			if not isinstance( self.result, dict ):
				raise ValueError( 'SSC observatories response must be a dictionary.' )
			if self.result.get( 'HttpStatus', 200 ) != 200:
				raise ValueError( self.result.get( 'ErrorMessage',
					self.result.get( 'HttpText', 'SSC observatories request failed.' ) ) )
			self.observatories = self.result.get( 'Observatory', [ ] )
			return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_observatories( self ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_ground_stations( self ) -> Dict[ str, Any ] | None:
		"""Retrieve the ground-station inventory.

		Purpose:
		    Construct SscWs with the current user-agent and timeout, request ground-station metadata,
		    check the returned dictionary and status, and retain its GroundStation records.

		Returns:
		    Dict[str, Any]: Complete SSC ground-station response dictionary.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			self.ssc = SscWs( user_agent=self.agents, timeout=self.timeout )
			self.result = self.ssc.get_ground_stations( )
			if not isinstance( self.result, dict ):
				raise ValueError( 'SSC ground stations response must be a dictionary.' )
			if self.result.get( 'HttpStatus', 200 ) != 200:
				raise ValueError( self.result.get( 'ErrorMessage',
					self.result.get( 'HttpText', 'SSC ground stations request failed.' ) ) )
			self.ground_stations = self.result.get( 'GroundStation', [ ] )
			return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_ground_stations( self ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_locations( self, observatories: str, start_time: str, end_time: str, coordinate_systems: str='gse', resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve spacecraft location data.

		Purpose:
		    Validate SSC identifiers, coordinate systems, UTC interval, and resolution factor;
		    construct the REST endpoint and return its decoded response after checking HTTP and
		    provider status.

		Args:
		    observatories (str): Nonblank comma-separated SSC identifiers containing letters, digits,
		        underscores, or hyphens.
		    start_time (str): Beginning UTC timestamp in YYYYMMDDTHHMMSSZ format, strictly before
		        end_time.
		    end_time (str): Ending UTC timestamp in YYYYMMDDTHHMMSSZ format.
		    coordinate_systems (str): Comma-separated coordinate systems: geo, gm, gse, gsm, sm,
		        geitod, or geij2000. Defaults to 'gse'.
		    resolution_factor (int): Sampling resolution factor of at least 1. Defaults to 1.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Complete decoded SSC spacecraft-location response.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			throw_if( 'observatories', observatories )
			self.observatory_ids = observatories
			throw_if( 'start_time', start_time )
			self.start_time = start_time
			throw_if( 'end_time', end_time )
			self.end_time = end_time
			throw_if( 'coordinate_systems', coordinate_systems )
			self.coordinate_systems = coordinate_systems
			throw_if( 'resolution_factor', resolution_factor )
			self.resolution_factor = resolution_factor

			throw_if( 'time', time )
			self.timeout = time
			if self.resolution_factor < 1:
				raise ValueError( 'resolution_factor must be greater than or equal to 1.' )
			if not re.fullmatch( r'[A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*',
					self.observatory_ids ):
				raise ValueError( 'observatories must contain comma-separated SSC identifiers.' )

			supported = { 'geo', 'gm', 'gse', 'gsm', 'sm', 'geitod', 'geij2000' }
			requested = self.coordinate_systems.split( ',' )
			if any( item not in supported for item in requested ):
				raise ValueError( 'Unsupported SSC coordinate system.' )

			pattern = r'\d{8}T\d{6}Z'
			if not re.fullmatch( pattern, self.start_time ) or not re.fullmatch(
					pattern, self.end_time ):
				raise ValueError(
					'SSC location times must use YYYYMMDDTHHMMSSZ (UTC).' )
			start_date = dt.datetime.strptime( self.start_time, '%Y%m%dT%H%M%SZ' )
			end_date = dt.datetime.strptime( self.end_time, '%Y%m%dT%H%M%SZ' )
			if start_date >= end_date:
				raise ValueError( 'start_time must precede end_time.' )

			self.request_url = (f'{self.url}/locations/{self.observatory_ids}/'
				f'{self.start_time},{self.end_time}/{self.coordinate_systems}/')
			self.params = { 'resolutionFactor': self.resolution_factor }
			self.response = requests.get( url=self.request_url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.result = self.response.json( )
			if not isinstance( self.result, dict ):
				raise ValueError( 'SSC locations response must be a dictionary.' )
			if self.result.get( 'HttpStatus', 200 ) != 200:
				raise ValueError( self.result.get( 'ErrorMessage',
					self.result.get( 'HttpText', 'SSC locations request failed.' ) ) )
			return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = "fetch_locations( self, observatories: str, start_time: str, end_time: str, coordinate_systems: str='gse', resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='observatories', query: str='', start_time: str='', end_time: str = '', coordinate_systems: str='gse', resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an SSC retrieval operation.

		Purpose:
		    Normalize the requested mode and delegate to observatory, ground-station, or
		    spacecraft-location retrieval. Inventory modes use the current instance timeout; location
		    mode uses the supplied timeout.

		Args:
		    mode (str): observatories, ground_stations, or locations. Defaults to 'observatories'; a
		        blank mode selects observatories.
		    query (str): Comma-separated SSC observatory identifiers for locations mode. Defaults to
		        an empty string.
		    start_time (str): Beginning UTC timestamp in YYYYMMDDTHHMMSSZ format, strictly before
		        end_time. Defaults to an empty string; required in locations mode.
		    end_time (str): Ending UTC timestamp in YYYYMMDDTHHMMSSZ format. Defaults to an empty
		        string; required in locations mode.
		    coordinate_systems (str): Comma-separated coordinate systems: geo, gm, gse, gsm, sm,
		        geitod, or geij2000. Defaults to 'gse'.
		    resolution_factor (int): Sampling resolution factor of at least 1. Defaults to 1.
		    time (int): Location request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Complete SSC response for the selected operation.

		Raises:
		    Error: Raised when input validation, provider retrieval, or response processing fails.
		"""
		try:
			active_mode = (mode or 'observatories').strip( ).lower( )
			if active_mode == 'observatories':
				return self.fetch_observatories( )
			
			if active_mode == 'ground_stations':
				return self.fetch_ground_stations( )
			
			if active_mode == 'locations':
				return self.fetch_locations( observatories=query, start_time=start_time,
					end_time=end_time, coordinate_systems=coordinate_systems,
					resolution_factor=resolution_factor, time=time )
			
			raise ValueError( "Use 'observatories', 'ground_stations', or 'locations'." )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = "fetch( self, mode: str='observatories', query: str='', start_time: str='', end_time: str = '', coordinate_systems: str='gse', resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception

class EarthObservatory( Fetcher ):
	"""NASA EONET event and reference-data adapter.

	Purpose:
	    Retrieve natural events, categories, sources, and imagery-layer metadata through the NASA
	    EONET v3 API, packaging the selected collection with request metadata.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    base_url (Optional[str]): Base provider API URL.
	    url (Optional[str]): Latest endpoint or request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Any): Latest decoded EONET JSON object.
	    result (Optional[Dict[str, Any]]): Latest packaged event or reference collection.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    mode (Optional[str]): Selected retrieval operation.
	    status (Optional[str]): Requested event status.
	    category (Optional[str]): Event or imagery-layer category filter.
	    source (Optional[str]): Event source filter.
	    days (Optional[int]): Event lookback period in days.
	    limit (Optional[int]): Requested record count.
	    start_date (Optional[str]): Start date of the requested interval.
	    end_date (Optional[str]): End date of the requested interval.
	    agents (Optional[str]): Configured HTTP user-agent.
	"""
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	status: Optional[ str ]
	category: Optional[ str ]
	source: Optional[ str ]
	days: Optional[ int ]
	limit: Optional[ int ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider configuration.

		Purpose:
		    Configure the EONET v3 base URL, user-agent, twenty-second timeout, and open-event
		    defaults without issuing a request.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://eonet.gsfc.nasa.gov/api/v3'
		self.url = None
		self.timeout = 20
		self.response = None
		self.payload = { }
		self.result = { }
		self.params = { }
		self.mode = 'events'
		self.status = 'open'
		self.category = ''
		self.source = ''
		self.days = 30
		self.limit = 20
		self.start_date = ''
		self.end_date = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'url', 'params', 'mode', 'status', 'category', 'source', 'days',
				'limit', 'start_date', 'end_date', 'request', 'fetch_events', 'fetch_categories',
				'fetch_sources', 'fetch_layers', 'fetch', 'create_schema' ]
	
	def request( self, time: int=20 ) -> Dict[ str, Any ]:
		"""Execute the configured endpoint request.

		Purpose:
		    Send an HTTP GET using the retained URL, parameters, and headers, check HTTP status,
		    decode JSON, and reject provider error responses.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Decoded provider JSON object.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:

			throw_if( 'time', time )
			self.timeout = time
			self.response = requests.get( url=self.url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'NASA EONET response must be a JSON object.' )
			if self.payload.get( 'error' ):
				raise ValueError( f"NASA EONET error: {self.payload[ 'error' ]}" )
			return self.payload
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'request( self, time: int=20 ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_events( self, status: str='open', category: str='', source: str='', limit: int=20, days: int=30, start_date: str='', end_date: str='', time: int=20 ) -> Dict[ str, Any ]:
		"""Retrieve natural events.

		Purpose:
		    Validate status, positive limits and paired dates; request events using an explicit date
		    range when present, otherwise use the lookback period.

		Args:
		    status (str): Event status: open, closed, or all. Defaults to 'open'.
		    category (str): Optional category filter. Defaults to an empty string.
		    source (str): Optional source filter. Defaults to an empty string.
		    limit (int): Positive requested event count. Defaults to 20.
		    days (int): Positive lookback in days when no explicit dates are supplied. Defaults to
		        30.
		    start_date (str): Start date parsed as YYYY-MM-DD; supply with end_date. Defaults to an
		        empty string.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date. Defaults to an
		        empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, events collection, title, and description.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:


			self.mode = 'events'
			throw_if( 'status', status )
			self.status = status
			self.category = category
			self.source = source
			throw_if( 'limit', limit )
			self.limit = limit
			throw_if( 'days', days )
			self.days = days
			self.start_date = start_date
			self.end_date = end_date
			if self.status not in ( 'open', 'closed', 'all' ):
				raise ValueError( 'status must be open, closed, or all.' )
			if self.limit < 1 or self.days < 1:
				raise ValueError( 'limit and days must be positive integers.' )
			if ( self.start_date != '' ) != ( self.end_date != '' ):
				raise ValueError( 'start_date and end_date must be supplied together.' )
			if self.start_date:
				start = dt.datetime.strptime( self.start_date, '%Y-%m-%d' ).date( )
				end = dt.datetime.strptime( self.end_date, '%Y-%m-%d' ).date( )
				if start > end:
					raise ValueError( 'start_date must be on or before end_date.' )
			self.url = f'{self.base_url}/events'
			self.params = { 'status': self.status, 'limit': self.limit }
			if self.category:
				self.params[ 'category' ] = self.category
			if self.source:
				self.params[ 'source' ] = self.source
			if self.start_date:
				self.params[ 'start' ] = self.start_date
				self.params[ 'end' ] = self.end_date
			else:
				self.params[ 'days' ] = self.days
			self.payload = self.request( time=time )
			if not isinstance( self.payload.get( 'events', [ ] ), list ):
				raise ValueError( 'NASA EONET events must be a list.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'events': self.payload.get( 'events', [ ] ),
				'title': self.payload.get( 'title', '' ),
				'description': self.payload.get( 'description', '' ) }
			return self.result

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = "fetch_events( self, status: str='open', category: str='', source: str='', limit: int=20, days: int=30, start_date: str='', end_date: str='', time: int=20 ) -> Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception
	
	def fetch_categories( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve event categories.

		Purpose:
		    Request the categories endpoint and validate that the returned category collection is a
		    list.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, categories collection, title, and description.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'categories'
			self.url = f'{self.base_url}/categories'
			self.params = { }
			self.payload = self.request( time=time )
			if not isinstance( self.payload.get( 'categories', [ ] ), list ):
				raise ValueError( 'NASA EONET categories must be a list.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'categories': self.payload.get( 'categories', [ ] ),
				'title': self.payload.get( 'title', '' ),
				'description': self.payload.get( 'description', '' ) }
			return self.result

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_categories( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_sources( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve event sources.

		Purpose:
		    Request the sources endpoint and validate that the returned source collection is a list.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, sources collection, title, and description.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'sources'
			self.url = f'{self.base_url}/sources'
			self.params = { }
			self.payload = self.request( time=time )
			if not isinstance( self.payload.get( 'sources', [ ] ), list ):
				raise ValueError( 'NASA EONET sources must be a list.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'sources': self.payload.get( 'sources', [ ] ),
				'title': self.payload.get( 'title', '' ),
				'description': self.payload.get( 'description', '' ) }
			return self.result

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_sources( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_layers( self, category: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve imagery-layer metadata.

		Purpose:
		    Optionally validate a category identifier, request the layers endpoint, and flatten
		    nested category layer lists or use the top-level layers list.

		Args:
		    category (str): Optional category identifier containing letters, digits, underscores, or
		        hyphens. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, category, flattened layer list, title, and
		        description.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'layers'
			self.category = category
			if self.category and not re.fullmatch( r'[A-Za-z0-9_-]+', self.category ):
				raise ValueError( 'category must be a valid EONET category identifier.' )
			self.url = f'{self.base_url}/layers'
			if self.category:
				self.url = f'{self.url}/{self.category}'
			self.params = { }
			self.payload = self.request( time=time )
			categories = self.payload.get( 'categories', [ ] )
			if not isinstance( categories, list ):
				raise ValueError( 'NASA EONET layer categories must be a list.' )
			layers = [ ]
			for item in categories:
				if not isinstance( item, dict ) or not isinstance( item.get( 'layers', [ ] ), list ):
					raise ValueError( 'NASA EONET layer category is malformed.' )
				layers.extend( item.get( 'layers', [ ] ) )
			if not categories:
				layers = self.payload.get( 'layers', [ ] )
			if not isinstance( layers, list ):
				raise ValueError( 'NASA EONET layers must be a list.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'category': self.category, 'layers': layers,
				'title': self.payload.get( 'title', '' ),
				'description': self.payload.get( 'description', '' ) }
			return self.result

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = "fetch_layers( self, category: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='events', status: str='open', category: str='', source: str='', limit: int=20, days: int=30, start_date: str='', end_date: str='', time: int=20 ) ->  Dict[ str, Any ]:
		"""Dispatch an EONET operation.

		Purpose:
		    Select the events, categories, sources, or layers retrieval method and pass the
		    applicable arguments.

		Args:
		    mode (str): events, categories, sources, or layers. Defaults to 'events'.
		    status (str): Event status: open, closed, or all. Defaults to 'open'.
		    category (str): Optional category filter. Defaults to an empty string.
		    source (str): Optional source filter. Defaults to an empty string.
		    limit (int): Positive requested event count. Defaults to 20.
		    days (int): Positive lookback in days when no explicit dates are supplied. Defaults to
		        30.
		    start_date (str): Start date parsed as YYYY-MM-DD; supply with end_date. Defaults to an
		        empty string.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date. Defaults to an
		        empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged collection and metadata for the selected operation.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode

			throw_if( 'time', time )
			self.timeout = time
			if self.mode == 'events':
				self.status = status
				self.category = category
				self.source = source
				self.limit = limit
				self.days = days
				self.start_date = start_date
				self.end_date = end_date
				return self.fetch_events( status=self.status, category=self.category,
					source=self.source, limit=self.limit, days=self.days,
					start_date=self.start_date, end_date=self.end_date, time=self.timeout )
			if self.mode == 'categories':
				return self.fetch_categories( time=self.timeout )
			if self.mode == 'sources':
				return self.fetch_sources( time=self.timeout )
			if self.mode == 'layers':
				self.category = category
				return self.fetch_layers( category=self.category, time=self.timeout )
			raise ValueError( "Use 'events', 'categories', 'sources', or 'layers'." )

		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = "fetch( self, mode: str='events', status: str='open', category: str='', source: str='', limit: int=20, days: int=30, start_date: str='', end_date: str='', time: int=20 ) ->  Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the '
					               f'{tool.strip( )} '
					               f'service.',
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class GlobalImagery( Fetcher ):
	"""NASA GIBS Web Map Service imagery adapter.

	Purpose:
	    Construct WMS GetCapabilities and GetMap URLs, account for WMS version and projection
	    axis ordering, and retrieve image files with request and output metadata.

	Attributes:
	    api_key (Optional[str]): Provider API key loaded from configuration.
	    mode (Optional[str]): Selected retrieval operation.
	    layer (str): Selected WMS layer identifier.
	    image_date (str): WMS TIME parameter.
	    bbox (Optional[Tuple[float, float, float, float]]): West, south, east, north bounding
	        coordinates.
	    projection (str): Normalized GIBS projection identifier.
	    quality (str): GIBS service quality identifier.
	    version (str): Selected WMS protocol version.
	    image_format (str): Requested image MIME type.
	    transparent (bool): Requested background transparency.
	    width (int): Image width in pixels.
	    height (int): Image height in pixels.
	    wms_url (str): Projection-specific WMS service URL.
	    epsg (str): EPSG coordinate reference identifier.
	    url (Optional[str]): Latest endpoint or request URL.
	    file_path (Optional[str]): Path of the downloaded image file.
	    longitude (Optional[float]): Reserved longitude state, initialized to None.
	    latitude (Optional[float]): Reserved latitude state, initialized to None.
	    coordinates (Optional[Tuple[float, float]]): Reserved coordinate pair, initialized to
	        None.
	    fetcher (Any): Reserved fetcher reference, initialized to None.
	    calendar_date (Optional[dt.datetime]): Reserved calendar date, initialized to None.
	    julian_date (Optional[float]): Reserved Julian date, initialized to None.
	    sidereal_time (Optional[str]): Reserved sidereal time, initialized to None.
	    local_time (Optional[dt.time]): Reserved local time, initialized to None.
	    utc_time (Optional[dt.time]): Reserved UTC time, initialized to None.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    response (Optional[Response]): Latest HTTP response.
	    result (Optional[Dict[str, Any]]): Latest downloaded image metadata.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    era (Optional[str]): Reserved era, initialized to None.
	    year (Optional[str]): Reserved year, initialized to None.
	    month (Optional[str]): Reserved month, initialized to None.
	    day (Optional[str]): Reserved day, initialized to None.
	"""
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	calendar_date: Optional[ dt.datetime ]
	julian_date: Optional[ float ]
	sidereal_time: Optional[ str ]
	utc_time: Optional[ dt.time ]
	local_time: Optional[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	era: Optional[ str ]
	year: Optional[ str ]
	month: Optional[ str ]
	day: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider configuration.

		Purpose:
		    Initialize GIBS service defaults, including EPSG:4326, best quality, WMS 1.1.1, PNG
		    output, transparency, and a 1200 by 600 image size.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.NASA_API_KEY
		self.mode = None
		self.layer = ''
		self.image_date = ''
		self.bbox = None
		self.projection = 'epsg4326'
		self.quality = 'best'
		self.version = '1.1.1'
		self.image_format = 'image/png'
		self.transparent = True
		self.width = 1200
		self.height = 600
		self.wms_url = ''
		self.epsg = 'EPSG:4326'
		self.url = 'https://gibs.earthdata.nasa.gov/wms'
		self.file_path = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.fetcher = None
		self.calendar_date = None
		self.julian_date = None
		self.sidereal_time = None
		self.local_time = None
		self.utc_time = None
		self.params = { }
		self.response = None
		self.result = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { }
		self.headers[ 'User-Agent' ] = self.agents
		self.era = None
		self.year = None
		self.month = None
		self.day = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'file_path', 'api_key', 'url', 'latitude', 'longitude', 'coordinates',
				'calendar_date', 'julian_date', 'sidereal_time', 'utc_time', 'local_time',
				'params', 'response', 'result', 'mode', 'timeout', 'headers',
				'layer', 'image_date', 'bbox', 'projection', 'quality', 'version',
				'image_format', 'transparent', 'width', 'height', 'wms_url', 'epsg',
				'get_capabilities_url', 'build_wms_url', 'fetch_wms_map', 'fetch_map_services',
				'fetch_mercator_map', 'create_schema' ]
	
	def get_capabilities_url( self, projection: str='epsg4326', quality: str='best', version: str='1.1.1' ) -> str:
		"""Build a WMS capabilities URL.

		Purpose:
		    Validate projection, quality and version, then assemble the GetCapabilities query
		    without issuing a request.

		Args:
		    projection (str): epsg4326, epsg3857, epsg3413, or epsg3031; case and colon are
		        normalized. Defaults to 'epsg4326'.
		    quality (str): Nonblank service quality identifier containing lowercase letters, digits,
		        underscores, or hyphens. Defaults to 'best'.
		    version (str): WMS version 1.1.1 or 1.3.0. Defaults to '1.1.1'.

		Returns:
		    str: Complete URL for the GetCapabilities operation.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'projection', projection )
			self.projection = projection.strip( ).lower( ).replace( ':', '' )
			throw_if( 'quality', quality )
			self.quality = quality
			throw_if( 'version', version )
			self.version = version
			if self.projection not in ( 'epsg4326', 'epsg3857', 'epsg3413', 'epsg3031' ):
				raise ValueError( 'Unsupported NASA GIBS WMS projection.' )
			if not re.fullmatch( r'[a-z0-9_-]+', self.quality ):
				raise ValueError( 'Invalid NASA GIBS service quality identifier.' )
			if self.version not in ( '1.1.1', '1.3.0' ):
				raise ValueError( 'GIBS WMS version must be 1.1.1 or 1.3.0.' )
			self.wms_url = (f'https://gibs.earthdata.nasa.gov/wms/'
				f'{self.projection}/{self.quality}/wms.cgi')
			self.params = { 'SERVICE': 'WMS', 'REQUEST': 'GetCapabilities',
				'VERSION': self.version }
			return f'{self.wms_url}?{urllib.parse.urlencode( self.params )}'

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = "get_capabilities_url( self, projection: str='epsg4326', quality: str='best', version: str='1.1.1' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def build_wms_url( self, layer: str, image_date: str, bbox: Tuple[ float, float, float, float ], width: int=1200, height: int=600, projection: str='epsg4326', quality: str='best', image_format: str='image/png', transparent: bool=True, version: str='1.1.1' ) -> str:
		"""Build a WMS map URL.

		Purpose:
		    Validate bounds, dimensions, projection, layer, format and transparency; use CRS and
		    latitude-first EPSG:4326 bounds for WMS 1.3.0, otherwise the applicable SRS or CRS
		    ordering.

		Args:
		    layer (str): Nonblank WMS layer identifier using letters, digits, underscores, commas,
		        or hyphens.
		    image_date (str): Nonblank imagery time value passed to the WMS TIME parameter.
		    bbox (Tuple[float, float, float, float]): Finite west, south, east, north bounds in the
		        selected projection, with increasing axes.
		    width (int): Positive image width in pixels. Defaults to 1200.
		    height (int): Positive image height in pixels. Defaults to 600.
		    projection (str): epsg4326, epsg3857, epsg3413, or epsg3031; case and colon are
		        normalized. Defaults to 'epsg4326'.
		    quality (str): Nonblank service quality identifier containing lowercase letters, digits,
		        underscores, or hyphens. Defaults to 'best'.
		    image_format (str): image/png or image/jpeg. Defaults to 'image/png'.
		    transparent (bool): Request a transparent background. Defaults to True.
		    version (str): WMS version 1.1.1 or 1.3.0. Defaults to '1.1.1'.

		Returns:
		    str: Complete URL for the GetMap operation.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'layer', layer )
			throw_if( 'image_date', image_date )
			throw_if( 'bbox', bbox )
			self.layer = layer
			self.image_date = image_date
			throw_if( 'projection', projection )
			self.projection = projection.strip( ).lower( ).replace( ':', '' )
			throw_if( 'quality', quality )
			self.quality = quality
			throw_if( 'version', version )
			self.version = version
			throw_if( 'image_format', image_format )
			self.image_format = image_format
			throw_if( 'width', width )
			self.width = width
			throw_if( 'height', height )
			self.height = height
			throw_if( 'transparent', transparent )
			self.transparent = transparent
			if self.projection not in ( 'epsg4326', 'epsg3857', 'epsg3413', 'epsg3031' ):
				raise ValueError( 'Unsupported NASA GIBS WMS projection.' )
			if not re.fullmatch( r'[a-z0-9_-]+', self.quality ):
				raise ValueError( 'Invalid NASA GIBS service quality identifier.' )
			if self.version not in ( '1.1.1', '1.3.0' ):
				raise ValueError( 'GIBS WMS version must be 1.1.1 or 1.3.0.' )
			if len( bbox ) != 4:
				raise ValueError( 'bbox must contain west, south, east, north.' )
			self.bbox = bbox
			if not all( float( '-inf' ) < value < float( 'inf' ) for value in self.bbox ):
				raise ValueError( 'bbox coordinates must be finite.' )
			west, south, east, north = self.bbox
			if west >= east or south >= north:
				raise ValueError( 'bbox bounds must increase from west to east and south to north.' )
			if self.width < 1 or self.height < 1:
				raise ValueError( 'width and height must be positive.' )
			if self.image_format not in ( 'image/png', 'image/jpeg' ):
				raise ValueError( 'GIBS WMS image_format must be image/png or image/jpeg.' )
			if not re.fullmatch( r'[A-Za-z0-9_,-]+', self.layer ):
				raise ValueError( 'Invalid WMS layer identifier.' )
			if not isinstance( self.transparent, bool ):
				raise ValueError( 'transparent must be a boolean.' )
			self.wms_url = (f'https://gibs.earthdata.nasa.gov/wms/'
				f'{self.projection}/{self.quality}/wms.cgi')
			self.epsg = f'EPSG:{self.projection[ 4: ]}'
			self.params = { 'SERVICE': 'WMS', 'VERSION': self.version,
				'REQUEST': 'GetMap', 'LAYERS': self.layer, 'STYLES': '',
				'FORMAT': self.image_format,
				'TRANSPARENT': str( self.transparent ).lower( ),
				'WIDTH': self.width, 'HEIGHT': self.height, 'TIME': self.image_date }
			if self.version == '1.3.0':
				self.params[ 'CRS' ] = self.epsg
				if self.projection == 'epsg4326':
					self.params[ 'BBOX' ] = f'{south},{west},{north},{east}'
				else:
					self.params[ 'BBOX' ] = f'{west},{south},{east},{north}'
			else:
				self.params[ 'SRS' ] = self.epsg
				self.params[ 'BBOX' ] = f'{west},{south},{east},{north}'
			return f'{self.wms_url}?{urllib.parse.urlencode( self.params )}'

		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = "build_wms_url( self, layer: str, image_date: str, bbox: Tuple[ float, float, float, float ], width: int=1200, height: int=600, projection: str='epsg4326', quality: str='best', image_format: str='image/png', transparent: bool=True, version: str='1.1.1' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def fetch_wms_map( self, layer: str, image_date: str, bbox: Tuple[ float, float, float, float ], width: int=1200, height: int=600, projection: str='epsg4326', quality: str='best', image_format: str='image/png', transparent: bool=True, output_dir: str='python-examples', output_name: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Download a GIBS map image.

		Purpose:
		    Construct a GetMap URL using the builder's default WMS version, create the output
		    directory, request the image, require an image content type, and write the response
		    bytes to disk.

		Args:
		    layer (str): Nonblank WMS layer identifier using letters, digits, underscores, commas,
		        or hyphens.
		    image_date (str): Nonblank imagery time value passed to the WMS TIME parameter.
		    bbox (Tuple[float, float, float, float]): Finite west, south, east, north bounds in the
		        selected projection, with increasing axes.
		    width (int): Positive image width in pixels. Defaults to 1200.
		    height (int): Positive image height in pixels. Defaults to 600.
		    projection (str): epsg4326, epsg3857, epsg3413, or epsg3031; case and colon are
		        normalized. Defaults to 'epsg4326'.
		    quality (str): Nonblank service quality identifier containing lowercase letters, digits,
		        underscores, or hyphens. Defaults to 'best'.
		    image_format (str): image/png or image/jpeg. Defaults to 'image/png'.
		    transparent (bool): Request a transparent background. Defaults to True.
		    output_dir (str): Destination directory. Defaults to 'python-examples'; a blank value
		        uses the same directory.
		    output_name (str): Destination filename. Defaults to an empty string, generating a
		        layer-and-date filename.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Image path, URL, request parameters, content type, HTTP status, byte
		        count, layer, date, bounds, and summary.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'layer', layer )
			throw_if( 'image_date', image_date )
			throw_if( 'bbox', bbox )
			self.layer = layer
			self.image_date = image_date
			self.bbox = bbox
			self.mode = 'wms_map'

			throw_if( 'time', time )
			self.timeout = time
			self.url = self.build_wms_url( layer=self.layer, image_date=self.image_date,
				bbox=self.bbox,
				width=width, height=height, projection=projection, quality=quality,
				image_format=image_format, transparent=transparent )
			
			directory = Path( output_dir or 'python-examples' )
			directory.mkdir( parents=True, exist_ok=True )
			if output_name:
				filename = output_name
			else:
				safe_layer = re.sub( r'[^A-Za-z0-9_\-]+', '_', self.layer )
				safe_date = re.sub( r'[^0-9\-]+', '_', self.image_date )
				extension = '.jpg' if self.image_format == 'image/jpeg' else '.png'
				filename = f'{safe_layer}_{safe_date}{extension}'
			
			self.file_path = str( directory / filename )
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			content_type = self.response.headers.get( 'Content-Type', '' )
			if 'image' not in content_type.lower( ):
				message = ('NASA GIBS did not return an image. '
				           f'Content-Type: {content_type}. '
				           f'Response preview: {self.response.text[ :500 ]}')
				raise ValueError( message )
			
			Path( self.file_path ).write_bytes( self.response.content )
			
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'image_path': self.file_path, 'content_type': content_type,
					'status_code': self.response.status_code, 'bytes': len(
						self.response.content ),
					'layer': self.layer, 'image_date': self.image_date,
					'bbox': { 'west': self.bbox[ 0 ], 'south': self.bbox[ 1 ],
							'east': self.bbox[ 2 ], 'north': self.bbox[ 3 ] },
					'summary': { 'rows': 1, 'columns': 8,
							'description': 'NASA GIBS WMS image written to disk.' } }
			
			return self.result
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = "fetch_wms_map( self, layer: str, image_date: str, bbox: Tuple[ float, float, float, float ], width: int=1200, height: int=600, projection: str='epsg4326', quality: str='best', image_format: str='image/png', transparent: bool=True, output_dir: str='python-examples', output_name: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_map_services( self ) -> Dict[ str, Any ] | None:
		"""Download the predefined global MODIS image.

		Purpose:
		    Request the global MODIS Terra true-color image for 2021-09-21 in EPSG:4326 at 1200 by
		    600 pixels and save the PNG in python-examples.

		Returns:
		    Dict[str, Any]: Image path, URL, request parameters, content type, HTTP status, byte
		        count, layer, date, bounds, and summary.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			return self.fetch_wms_map( layer='MODIS_Terra_CorrectedReflectance_TrueColor',
				image_date='2021-09-21', bbox=( -180.0, -90.0, 180.0, 90.0), width=1200,
				height=600, projection='epsg4326', quality='best', image_format='image/png',
				transparent=True, output_dir='python-examples',
				output_name='MODIS_Terra_CorrectedReflectance_TrueColor.png', time=20 )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = 'fetch_map_services( self ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_mercator_map( self, ccrs=None ) -> Dict[ str, Any ] | None:
		"""Download the predefined Mercator Landsat image.

		Purpose:
		    Request the Landsat WELD annual reflectance image for 2000-12-01 in EPSG:3857 at 600 by
		    600 pixels and save the PNG in python-examples.

		Args:
		    ccrs (Any): Reserved compatibility argument; unused. Defaults to None.

		Returns:
		    Dict[str, Any]: Image path, URL, request parameters, content type, HTTP status, byte
		        count, layer, date, bounds, and summary.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			return self.fetch_wms_map(
				layer='Landsat_WELD_CorrectedReflectance_Bands157_Global_Annual',
				image_date='2000-12-01', bbox=(
					-8000000.0, -8000000.0, 8000000.0,
					8000000.0), width=600, height=600, projection='epsg3857', quality='best',
				image_format='image/png', transparent=True, output_dir='python-examples',
				output_name='Landsat_WELD_CorrectedReflectance_Bands157_Global_Annual.png',
				time=20 )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = 'fetch_mercator_map( self, ccrs=None ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of parameter schema definitions.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the '
					               f'{tool.strip( )} '
					               f'service.',
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class NearbyObjects( Fetcher ):
	"""NASA JPL small-body and fireball data adapter.

	Purpose:
	    Retrieve close approaches, small-body object details, human-accessible target summaries
	    and individual objects, and fireball observations through JPL SSD endpoints.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    base_url (Optional[str]): Base provider API URL.
	    url (Optional[str]): Latest endpoint or request URL.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    mode (Optional[str]): Selected retrieval operation.
	    start_date (Optional[str]): Start date of the requested interval.
	    end_date (Optional[str]): End date of the requested interval.
	    query (Optional[str]): Latest search value.
	    dist_max (Optional[str]): Maximum close-approach distance with unit suffix.
	    body (Optional[str]): Close-approach target body.
	    sort (Optional[str]): Close-approach sort expression.
	    limit (Optional[int]): Requested record count.
	    agents (Optional[str]): Configured HTTP user-agent.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Any): Latest decoded SSD JSON object.
	    dv (float): NHATS delta-v threshold.
	    dur (int): NHATS mission duration in days.
	    stay (int): NHATS stay duration in days.
	    launch (str): NHATS launch window.
	    start (dt.date): Parsed close-approach start date.
	    end (dt.date): Parsed close-approach end date.
	    result (Optional[Dict[str, Any]]): Latest packaged SSD response.
	    query_type (str): Small-body search selector.
	    include_physical (bool): Whether physical parameters are requested.
	    include_close_approaches (bool): Whether close-approach details are requested.
	    include_discovery (bool): Whether discovery details are requested.
	    ca_body (str): Body filter for object close-approach details.
	    h (float): NHATS absolute magnitude threshold.
	    occ (int): NHATS orbit condition code.
	"""
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	query: Optional[ str ]
	dist_max: Optional[ str ]
	body: Optional[ str ]
	sort: Optional[ str ]
	limit: Optional[ int ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider configuration.

		Purpose:
		    Configure the JPL SSD base URL, request headers, Earth close-approach defaults, and
		    empty query state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://ssd-api.jpl.nasa.gov'
		self.url = None
		self.params = { }
		self.mode = 'close_approaches'
		self.start_date = ''
		self.end_date = ''
		self.query = ''
		self.dist_max = '10LD'
		self.body = 'Earth'
		self.sort = 'date'
		self.limit = 20
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'url', 'params', 'mode', 'start_date', 'end_date', 'query',
				'dist_max',
				'body', 'sort', 'limit', 'fetch_close_approaches', 'fetch_object_lookup',
				'request', 'validate_nhats_options', 'fetch_nhats_summary', 'fetch_nhats_object', 'fetch_fireballs', 'fetch',
				'create_schema' ]
	
	def request( self, time: int=20 ) -> Dict[ str, Any ]:
		"""Execute the configured endpoint request.

		Purpose:
		    Send an HTTP GET using the retained URL, parameters, and headers, check HTTP status,
		    decode JSON, and reject provider error responses.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Decoded provider JSON object.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:

			throw_if( 'time', time )
			self.timeout = time
			self.response = requests.get( url=self.url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'JPL SSD response must be a JSON object.' )
			if self.payload.get( 'error' ):
				raise ValueError( f"JPL SSD API error: {self.payload[ 'error' ]}" )
			return self.payload
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = 'request( self, time: int=20 ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception


	def validate_nhats_options( self, dv: float, dur: int, stay: int, launch: str ) -> None:
		"""Validate NHATS mission filters.

		Purpose:
		    Store the supplied mission filters and check allowed delta-v, duration, stay, and
		    launch-window values.

		Args:
		    dv (float): Delta-v threshold in km/s; accepted whole-number values 4 through 12.
		    dur (int): Mission duration in days, 60 through 450 in 30-day increments.
		    stay (int): Stay duration in days: 8, 16, 24, or 32.
		    launch (str): Launch window: 2020-2025, 2025-2030, 2030-2035, 2035-2040, 2040-2045, or
		        2020-2045.

		Returns:
		    None: Stores validated mission filter state; no value is returned.

		Raises:
		    Error: Raised when required mission filters are missing or outside the accepted values.
		"""
		try:
			throw_if( 'dv', dv )
			throw_if( 'dur', dur )
			throw_if( 'stay', stay )
			throw_if( 'launch', launch )
			self.dv = dv
			self.dur = dur
			self.stay = stay
			self.launch = launch
			if self.dv not in range( 4, 13 ):
				raise ValueError( 'NHATS dv must be an integer from 4 through 12.' )
			if self.dur not in range( 60, 451, 30 ):
				raise ValueError( 'NHATS dur must be between 60 and 450 in 30-day steps.' )
			if self.stay not in (8, 16, 24, 32):
				raise ValueError( 'NHATS stay must be 8, 16, 24, or 32 days.' )
			if self.launch not in ('2020-2025', '2025-2030', '2030-2035',
					'2035-2040', '2040-2045', '2020-2045'):
				raise ValueError( 'Unsupported NHATS launch window.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = 'validate_nhats_options( self, dv: float, dur: int, stay: int, launch: str ) -> None'
			Logger( ).write( exception )
			raise exception


	def fetch_close_approaches( self, start_date: str, end_date: str, dist_max: str='10LD', body: str='Earth', sort: str='date', limit: int=20, time: int=20 ) -> Dict[ str, Any ]:
		"""Retrieve small-body close approaches.

		Purpose:
		    Validate dates, record limit, distance expression and sort field, then query cad.api and
		    package its tabular response.

		Args:
		    start_date (str): Start date parsed as YYYY-MM-DD.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date.
		    dist_max (str): Maximum distance with numeric magnitude and LD, au, AU, km, or KM
		        suffix. Defaults to '10LD'.
		    body (str): Close-approach body filter. Defaults to 'Earth'.
		    sort (str): date, dist, dist-min, v-inf, v-rel, h, or object; a leading minus requests
		        descending order. Defaults to 'date'.
		    limit (int): Positive requested record count. Defaults to 20.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, count, fields, data rows, and provider signature.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:

			throw_if( 'limit', limit )

			throw_if( 'start_date', start_date )
			throw_if( 'end_date', end_date )
			self.mode = 'close_approaches'
			self.start_date = start_date
			self.end_date = end_date
			self.dist_max = dist_max
			self.body = body
			self.sort = sort
			self.limit = limit
			self.start = dt.datetime.strptime( self.start_date, '%Y-%m-%d' ).date( )
			self.end = dt.datetime.strptime( self.end_date, '%Y-%m-%d' ).date( )
			if self.start > self.end:
				raise ValueError( 'start_date must be on or before end_date.' )
			if self.limit < 1:
				raise ValueError( 'limit must be positive.' )
			if self.sort.lstrip( '-' ) not in ('date', 'dist', 'dist-min',
					'v-inf', 'v-rel', 'h', 'object'):
				raise ValueError( 'Unsupported JPL CAD sort field.' )
			if not re.fullmatch( r'[0-9]+(?:\.[0-9]+)?(?:LD|au|AU|km|KM)',
					self.dist_max ):
				raise ValueError( 'dist_max must contain a supported distance unit.' )
			self.url = f'{self.base_url}/cad.api'
			self.params = { 'date-min': self.start_date, 'date-max': self.end_date,
				'dist-max': self.dist_max, 'body': self.body, 'sort': self.sort,
				'limit': self.limit }
			payload = self.request( time=time )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'count': payload.get( 'count', 0 ), 'fields': payload.get( 'fields', [ ] ),
				'data': payload.get( 'data', [ ] ), 'signature': payload.get( 'signature', { } ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch_close_approaches( self, start_date: str, end_date: str, dist_max: str='10LD', body: str='Earth', sort: str='date', limit: int=20, time: int=20 ) -> Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception
	
	def fetch_object_lookup( self, query: str, query_type: str='sstr', include_physical: bool = True, include_close_approaches: bool=True, ca_body: str='Earth', include_discovery: bool = True, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve small-body object details.

		Purpose:
		    Validate the search selector, assemble optional physical, close-approach and discovery
		    flags, and query sbdb.api.

		Args:
		    query (str): Nonblank object search value.
		    query_type (str): Search selector: sstr, spk, or des. Defaults to 'sstr'.
		    include_physical (bool): Include physical parameters. Defaults to True.
		    include_close_approaches (bool): Include close-approach data. Defaults to True.
		    ca_body (str): Body filter for included close approaches. Defaults to 'Earth'.
		    include_discovery (bool): Include discovery data. Defaults to True.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, and decoded provider data.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'include_physical', include_physical )

			throw_if( 'include_close_approaches', include_close_approaches )

			throw_if( 'include_discovery', include_discovery )

			throw_if( 'query', query )
			self.mode = 'object_lookup'
			self.query = query
			throw_if( 'query_type', query_type )
			self.query_type = query_type
			if self.query_type not in ('sstr', 'spk', 'des'):
				raise ValueError( 'query_type must be sstr, spk, or des.' )
			self.include_physical = include_physical
			self.include_close_approaches = include_close_approaches
			self.include_discovery = include_discovery
			self.ca_body = ca_body
			self.url = f'{self.base_url}/sbdb.api'
			self.params = { self.query_type: self.query,
				'phys-par': '1' if self.include_physical else '0',
				'ca-data': '1' if self.include_close_approaches else '0',
				'discovery': '1' if self.include_discovery else '0' }
			if self.include_close_approaches and self.ca_body:
				self.params[ 'ca-body' ] = self.ca_body
			payload = self.request( time=time )
			self.result = { 'mode': self.mode, 'url': self.url,
				'params': self.params, 'data': payload }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch_object_lookup( self, query: str, query_type: str='sstr', include_physical: bool = True, include_close_approaches: bool=True, ca_body: str='Earth', include_discovery: bool = True, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_nhats_summary( self, dv: float=6.0, dur: int=360, stay: int=8, launch: str = '2020-2045', h: float=26.0, occ: int=7, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve human-accessible target summaries.

		Purpose:
		    Validate mission and magnitude filters and orbit condition code, then request NHATS
		    summary data.

		Args:
		    dv (float): Delta-v threshold in km/s; accepted whole-number values 4 through 12.
		        Defaults to 6.0.
		    dur (int): Mission duration in days, 60 through 450 in 30-day increments. Defaults to
		        360.
		    stay (int): Stay duration in days: 8, 16, 24, or 32. Defaults to 8.
		    launch (str): Launch window: 2020-2025, 2025-2030, 2030-2035, 2035-2040, 2040-2045, or
		        2020-2045. Defaults to '2020-2045'.
		    h (float): Absolute magnitude threshold; accepted whole-number values 16 through 30.
		        Defaults to 26.0.
		    occ (int): Orbit condition code from 0 through 8. Defaults to 7.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, and decoded provider data.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			self.validate_nhats_options( dv=dv, dur=dur, stay=stay, launch=launch )
			self.h = h
			self.occ = occ
			if self.h not in range( 16, 31 ):
				raise ValueError( 'NHATS h must be an integer from 16 through 30.' )
			if self.occ not in range( 9 ):
				raise ValueError( 'NHATS occ must be between 0 and 8.' )
			self.mode = 'nhats_summary'
			self.url = f'{self.base_url}/nhats.api'
			self.params = { 'dv': self.dv, 'dur': self.dur, 'stay': self.stay,
				'launch': self.launch, 'h': self.h, 'occ': self.occ }
			payload = self.request( time=time )
			self.result = { 'mode': self.mode, 'url': self.url,
				'params': self.params, 'data': payload }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch_nhats_summary( self, dv: float=6.0, dur: int=360, stay: int=8, launch: str = '2020-2045', h: float=26.0, occ: int=7, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_nhats_object( self, designation: str, dv: float=6.0, dur: int=360, stay: int = 8, launch: str='2020-2045', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve accessibility data for a designated object.

		Purpose:
		    Validate the object designation and mission filters and request individual NHATS data.

		Args:
		    designation (str): Nonblank small-body designation.
		    dv (float): Delta-v threshold in km/s; accepted whole-number values 4 through 12.
		        Defaults to 6.0.
		    dur (int): Mission duration in days, 60 through 450 in 30-day increments. Defaults to
		        360.
		    stay (int): Stay duration in days: 8, 16, 24, or 32. Defaults to 8.
		    launch (str): Launch window: 2020-2025, 2025-2030, 2030-2035, 2035-2040, 2040-2045, or
		        2020-2045. Defaults to '2020-2045'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, and decoded provider data.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'designation', designation )
			self.validate_nhats_options( dv=dv, dur=dur, stay=stay, launch=launch )
			self.mode = 'nhats_object'
			self.query = designation
			self.url = f'{self.base_url}/nhats.api'
			self.params = { 'des': self.query, 'dv': self.dv, 'dur': self.dur,
				'stay': self.stay, 'launch': self.launch }
			payload = self.request( time=time )
			self.result = { 'mode': self.mode, 'url': self.url,
				'params': self.params, 'data': payload }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch_nhats_object( self, designation: str, dv: float=6.0, dur: int=360, stay: int = 8, launch: str='2020-2045', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_fireballs( self, date_min: str='', limit: int=20, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve fireball observations.

		Purpose:
		    Validate the record limit and optional minimum date, then query fireball.api and package
		    its tabular response.

		Args:
		    date_min (str): Optional earliest date parsed as YYYY-MM-DD. Defaults to an empty
		        string.
		    limit (int): Positive requested record count. Defaults to 20.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, count, fields, data rows, and provider signature.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:

			throw_if( 'limit', limit )

			self.limit = limit
			if self.limit < 1:
				raise ValueError( 'limit must be positive.' )
			self.start_date = date_min
			if self.start_date:
				dt.datetime.strptime( self.start_date, '%Y-%m-%d' )
			self.mode = 'fireballs'
			self.url = f'{self.base_url}/fireball.api'
			self.params = { 'limit': self.limit }
			if self.start_date:
				self.params[ 'date-min' ] = self.start_date
			payload = self.request( time=time )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
				'count': payload.get( 'count', 0 ), 'fields': payload.get( 'fields', [ ] ),
				'data': payload.get( 'data', [ ] ), 'signature': payload.get( 'signature', { } ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch_fireballs( self, date_min: str='', limit: int=20, time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='close_approaches', start_date: str='', end_date: str='', query: str='', query_type: str='sstr', dist_max: str='10LD', body: str = 'Earth', sort: str='date', limit: int=20, dv: float=6.0, dur: int=360, stay: int=8, launch: str='2020-2045', h: float=26.0, occ: int=7, include_physical: bool=True, include_close_approaches: bool=True, ca_body: str = 'Earth', include_discovery: bool=True, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a JPL SSD retrieval operation.

		Purpose:
		    Select close approaches, object lookup, NHATS summary, NHATS object, or fireball
		    retrieval and forward the corresponding filters.

		Args:
		    mode (str): close_approaches, object_lookup, nhats_summary, nhats_object, or fireballs.
		        Defaults to 'close_approaches'.
		    start_date (str): Start date parsed as YYYY-MM-DD. Defaults to an empty string; required
		        for close approaches and optional for fireballs.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date. Defaults to an
		        empty string; required for close approaches.
		    query (str): Object search value or NHATS designation. Defaults to an empty string.
		    query_type (str): Search selector: sstr, spk, or des. Defaults to 'sstr'.
		    dist_max (str): Maximum distance with numeric magnitude and LD, au, AU, km, or KM
		        suffix. Defaults to '10LD'.
		    body (str): Close-approach body filter. Defaults to 'Earth'.
		    sort (str): date, dist, dist-min, v-inf, v-rel, h, or object; a leading minus requests
		        descending order. Defaults to 'date'.
		    limit (int): Positive requested record count. Defaults to 20.
		    dv (float): Delta-v threshold in km/s; accepted whole-number values 4 through 12.
		        Defaults to 6.0.
		    dur (int): Mission duration in days, 60 through 450 in 30-day increments. Defaults to
		        360.
		    stay (int): Stay duration in days: 8, 16, 24, or 32. Defaults to 8.
		    launch (str): Launch window: 2020-2025, 2025-2030, 2030-2035, 2035-2040, 2040-2045, or
		        2020-2045. Defaults to '2020-2045'.
		    h (float): Absolute magnitude threshold; accepted whole-number values 16 through 30.
		        Defaults to 26.0.
		    occ (int): Orbit condition code from 0 through 8. Defaults to 7.
		    include_physical (bool): Include physical parameters. Defaults to True.
		    include_close_approaches (bool): Include close-approach data. Defaults to True.
		    ca_body (str): Body filter for included close approaches. Defaults to 'Earth'.
		    include_discovery (bool): Include discovery data. Defaults to True.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged response for the selected SSD operation.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			
			if self.mode == 'close_approaches':
				return self.fetch_close_approaches( start_date=start_date, end_date=end_date,
					dist_max=dist_max, body=body, sort=sort, limit=limit, time=time )
			
			if self.mode == 'object_lookup':
				return self.fetch_object_lookup( query=query, query_type=query_type,
					include_physical=include_physical,
					include_close_approaches=include_close_approaches, ca_body=ca_body,
					include_discovery=include_discovery, time=time )
			
			if self.mode == 'nhats_summary':
				return self.fetch_nhats_summary( dv=dv, dur=dur, stay=stay, launch=launch, h=h,
					occ=occ, time=time )
			
			if self.mode == 'nhats_object':
				return self.fetch_nhats_object( designation=query, dv=dv, dur=dur, stay=stay,
					launch=launch, time=time )
			
			if self.mode == 'fireballs':
				return self.fetch_fireballs( date_min=start_date, limit=limit, time=time )
			
			raise ValueError( "Unsupported mode." )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = "fetch( self, mode: str='close_approaches', start_date: str='', end_date: str='', query: str='', query_type: str='sstr', dist_max: str='10LD', body: str = 'Earth', sort: str='date', limit: int=20, dv: float=6.0, dur: int=360, stay: int=8, launch: str='2020-2045', h: float=26.0, occ: int=7, include_physical: bool=True, include_close_approaches: bool=True, ca_body: str = 'Earth', include_discovery: bool=True, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the '
					               f'{tool.strip( )} '
					               f'service.',
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class OpenScience( Fetcher ):
	"""NASA Open Science Data Repository adapter.

	Purpose:
	    Retrieve dataset accession records and metadata, assay, or data queries from the OSDR
	    biodata API. Package JSON or textual responses with their endpoint and query parameters.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    base_url (Optional[str]): Base provider API URL.
	    url (Optional[str]): Latest endpoint or request URL.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    query_text (Optional[str]): Latest query string or encoded query parameters.
	    format_value (Optional[str]): Requested OSDR response format.
	    size (Optional[int]): Reserved result size, initialized to 100; not submitted by
	        retrieval methods.
	    endpoint (Optional[str]): Selected relative OSDR API path.
	    agents (Optional[str]): Configured HTTP user-agent.
	    accession (str): Latest dataset accession identifier.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Any): Latest decoded JSON or text response.
	    result (Optional[Dict[str, Any]]): Latest packaged OSDR response.
	"""
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	query_text: Optional[ str ]
	format_value: Optional[ str ]
	size: Optional[ int ]
	endpoint: Optional[ str ]
	agents: Optional[ str ]
	
	accession: str
	
	def __init__( self ) -> None:
		"""Initialize provider configuration.

		Purpose:
		    Configure the OSDR biodata base URL, user-agent, JSON format default, and empty query
		    and accession state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://visualization.osdr.nasa.gov/biodata/api'
		self.url = None
		self.params = { }
		self.query_text = ''
		self.format_value = 'json'
		self.size = 100
		self.endpoint = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		self.accession = ''
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'url', 'params', 'query_text', 'format_value', 'size', 'endpoint',
				'request', 'fetch_dataset', 'fetch_metadata', 'fetch_assays', 'fetch_data', 'fetch',
				'create_schema' ]
	
	def validate_format( self, format_value: str ) -> str:
		"""Validate the requested output format.

		Purpose:
		    Check that the supplied format is one of json, csv, tsv, or browser.

		Args:
		    format_value (str): Response format: json, csv, tsv, or browser.

		Returns:
		    str: Original accepted format string.

		Raises:
		    Error: Raised when the format is missing or unsupported.
		"""
		try:
			throw_if( 'format_value', format_value )
			
			allowed = { 'json', 'csv', 'tsv', 'browser' }
			if format_value not in allowed:
				raise ValueError( "Unsupported format. Use one of: json, csv, tsv, browser." )
			
			return format_value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'validate_format( self, format_value: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def coerce_response( self, response: requests.Response ) -> Dict[ str, Any ] | str:
		"""Decode an OSDR HTTP response.

		Purpose:
		    Decode JSON directly for application/json content; otherwise attempt JSON decoding and
		    fall back to response text.

		Args:
		    response (requests.Response): HTTP response to decode.

		Returns:
		    Dict[str, Any] | str: Decoded JSON value or response text; JSON may also have other
		        JSON-compatible types.

		Raises:
		    Error: Raised when response processing fails outside the text fallback.
		"""
		try:
			content_type = str( response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				return response.json( )
			
			try:
				return response.json( )
			except Exception:
				return response.text
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'coerce_response( self, response: requests.Response ) -> Dict[ str, Any ] | str'
			Logger( ).write( exception )
			raise exception
	
	def request( self, time: int=20 ) -> Dict[ str, Any ] | str:
		"""Execute an OSDR request.

		Purpose:
		    Send an HTTP GET using retained URL, parameters and headers, check status, decode the
		    response, and reject JSON objects containing provider errors.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any] | str: Decoded JSON value or response text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:

			throw_if( 'time', time )
			self.timeout = time
			self.response = requests.get( url=self.url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.coerce_response( self.response )
			if isinstance( self.payload, dict ) and self.payload.get( 'error' ):
				raise ValueError( f"NASA OSDR API error: {self.payload[ 'error' ]}" )
			return self.payload
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'request( self, time: int=20 ) -> Dict[ str, Any ] | str'
			Logger( ).write( exception )
			raise exception


	def fetch_dataset( self, accession: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve a dataset by accession.

		Purpose:
		    Validate the accession identifier, build the dataset endpoint, and package the retrieved
		    data.

		Args:
		    accession (str): Nonblank accession containing letters, digits, underscores, or hyphens.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, and response data, which may be decoded JSON or
		        text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'accession', accession )
			self.accession = accession
			if not re.fullmatch( r'[A-Za-z0-9_-]+', self.accession ):
				raise ValueError( 'Invalid OSDR dataset accession.' )
			self.endpoint = f'/v2/dataset/{self.accession}/'
			self.url = f'{self.base_url}{self.endpoint}'
			self.params = { }
			self.result = { 'mode': 'dataset', 'url': self.url,
				'params': self.params, 'data': self.request( time=time ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'fetch_dataset( self, accession: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_metadata( self, query: str, format_value: str='json', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve dataset metadata.

		Purpose:
		    Parse the query into parameters, validate and attach the format, and request the
		    metadata endpoint.

		Args:
		    query (str): Nonblank query text or URL-encoded key/value pairs; text containing '=' is
		        parsed into query parameters.
		    format_value (str): Response format: json, csv, tsv, or browser. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, and response data, which may be decoded JSON or
		        text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'query', query )
			self.query_text = query
			self.format_value = self.validate_format( format_value )
			self.endpoint = '/v2/query/metadata/'
			self.url = f'{self.base_url}{self.endpoint}'
			self.params = dict( urllib.parse.parse_qsl( self.query_text,
				keep_blank_values=True ) ) if '=' in self.query_text else {
				'query': self.query_text }
			self.params[ 'format' ] = self.format_value
			self.result = { 'mode': 'metadata', 'url': self.url,
				'params': self.params, 'data': self.request( time=time ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = "fetch_metadata( self, query: str, format_value: str='json', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_assays( self, query: str, format_value: str='json', time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve assay information.

		Purpose:
		    Parse the query into parameters, validate and attach the format, and request the assays
		    endpoint.

		Args:
		    query (str): Nonblank query text or URL-encoded key/value pairs; text containing '=' is
		        parsed into query parameters.
		    format_value (str): Response format: json, csv, tsv, or browser. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, and response data, which may be decoded JSON or
		        text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'query', query )
			self.query_text = query
			self.format_value = self.validate_format( format_value )
			self.endpoint = '/v2/query/assays/'
			self.url = f'{self.base_url}{self.endpoint}'
			self.params = dict( urllib.parse.parse_qsl( self.query_text,
				keep_blank_values=True ) ) if '=' in self.query_text else {
				'query': self.query_text }
			self.params[ 'format' ] = self.format_value
			self.result = { 'mode': 'assays', 'url': self.url,
				'params': self.params, 'data': self.request( time=time ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = "fetch_assays( self, query: str, format_value: str='json', time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def fetch_data( self, query: str, format_value: str='json', time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve OSDR data.

		Purpose:
		    Parse the query into parameters, validate and attach the format, and request the data
		    endpoint.

		Args:
		    query (str): Nonblank query text or URL-encoded key/value pairs; text containing '=' is
		        parsed into query parameters.
		    format_value (str): Response format: json, csv, tsv, or browser. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, and response data, which may be decoded JSON or
		        text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'query', query )
			self.query_text = query
			self.format_value = self.validate_format( format_value )
			self.endpoint = '/v2/query/data/'
			self.url = f'{self.base_url}{self.endpoint}'
			self.params = dict( urllib.parse.parse_qsl( self.query_text,
				keep_blank_values=True ) ) if '=' in self.query_text else {
				'query': self.query_text }
			self.params[ 'format' ] = self.format_value
			self.result = { 'mode': 'data', 'url': self.url,
				'params': self.params, 'data': self.request( time=time ) }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = "fetch_data( self, query: str, format_value: str='json', time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='dataset', query: str='', accession: str='', format_value: str = 'json', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an OSDR retrieval operation.

		Purpose:
		    Normalize the mode and delegate to dataset, metadata, assays, or data retrieval.

		Args:
		    mode (str): dataset, metadata, assays, or data. Defaults to 'dataset'; blank also
		        selects dataset.
		    query (str): Nonblank query text or URL-encoded key/value pairs; text containing '=' is
		        parsed into query parameters. Defaults to an empty string; required for query modes.
		    accession (str): Dataset accession identifier. Defaults to an empty string; required in
		        dataset mode.
		    format_value (str): Response format: json, csv, tsv, or browser. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters, and response data, which may be decoded JSON or
		        text.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			active_mode = ( mode or 'dataset' ).strip( ).lower( )
			
			if active_mode == 'dataset':
				return self.fetch_dataset( accession=accession, time=time )
			
			if active_mode == 'metadata':
				return self.fetch_metadata( query=query, format_value=format_value, time=time )
			
			if active_mode == 'assays':
				return self.fetch_assays( query=query, format_value=format_value, time=time )
			
			if active_mode == 'data':
				return self.fetch_data( query=query, format_value=format_value, time=time )
			
			raise ValueError( "Unsupported mode. Use one of: dataset, metadata, assays, data." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = "fetch( self, mode: str='dataset', query: str='', accession: str='', format_value: str = 'json', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ), 'description': (f'{description.strip( )} '
			                                                    f'This function uses the '
			                                                    f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class SpaceWeather( Fetcher ):
	"""NASA DONKI space-weather retrieval adapter.

	Purpose:
	    Map supported space-weather operations to DONKI endpoints, apply endpoint-specific
	    filters, and package decoded event or notification data without exposing the API key in
	    returned parameters.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    base_url (Optional[str]): Base provider API URL.
	    api_key (Optional[str]): Provider API key loaded from configuration.
	    url (Optional[str]): Latest endpoint or request URL.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    mode (Optional[str]): Selected retrieval operation.
	    start_date (Optional[str]): Start date of the requested interval.
	    end_date (Optional[str]): End date of the requested interval.
	    location (Optional[str]): IPS location filter.
	    catalog (Optional[str]): IPS or CMEAnalysis catalog filter.
	    notification_type (Optional[str]): Notification type filter.
	    limit_note (Optional[str]): Reserved rate-limit note, initialized to None.
	    agents (Optional[str]): Configured HTTP user-agent.
	    most_accurate_only (bool): CMEAnalysis accuracy filter flag.
	    complete_entry_only (bool): CMEAnalysis completeness filter flag.
	    speed (int): CMEAnalysis nonnegative speed filter.
	    half_angle (int): CMEAnalysis half-angle filter in degrees.
	    keyword (str): CMEAnalysis keyword filter.
	    endpoint (str): Selected DONKI endpoint.
	    active_key (str): Configured NASA key or DEMO_KEY fallback.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Any): Latest decoded DONKI JSON array or object.
	    result (Optional[Dict[str, Any]]): Latest packaged DONKI response.
	"""
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	location: Optional[ str ]
	catalog: Optional[ str ]
	notification_type: Optional[ str ]
	limit_note: Optional[ str ]
	agents: Optional[ str ]
	
	most_accurate_only: bool
	complete_entry_only: bool
	speed: int
	half_angle: int
	keyword: str
	endpoint: str
	active_key: str
	
	def __init__( self ) -> None:
		"""Initialize provider configuration.

		Purpose:
		    Configure the DONKI base URL, load the NASA API key or DEMO_KEY fallback, initialize
		    request headers and default CME filters.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://api.nasa.gov/DONKI'
		self.api_key = cfg.NASA_API_KEY
		self.url = None
		self.params = { }
		self.mode = 'cme'
		self.start_date = ''
		self.end_date = ''
		self.location = 'ALL'
		self.catalog = 'ALL'
		self.notification_type = 'all'
		self.limit_note = None
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		self.most_accurate_only = True
		self.complete_entry_only = True
		self.speed = 0
		self.half_angle = 0
		self.keyword = ''
		self.endpoint = ''
		self.active_key = self.api_key or 'DEMO_KEY'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'api_key', 'url', 'params', 'mode', 'start_date', 'end_date',
				'location', 'catalog', 'notification_type', 'fetch_endpoint', 'fetch',
				'create_schema' ]
	
	def fetch_endpoint( self, endpoint: str, start_date: str, end_date: str, time: int=20, location: str='', catalog: str='', notification_type: str='', most_accurate_only: bool=True, complete_entry_only: bool=True, speed: int=0, half_angle: int=0, keyword: str='' ) -> Dict[ str, Any ] | None:
		"""Retrieve a DONKI endpoint response.

		Purpose:
		    Validate the endpoint and paired date interval, add applicable IPS, CMEAnalysis, or
		    notification filters, request JSON, check provider errors, and exclude the API key from
		    returned parameters.

		Args:
		    endpoint (str): CME, CMEAnalysis, GST, IPS, FLR, SEP, MPC, RBE, HSS,
		        WSAEnlilSimulations, or notifications.
		    start_date (str): Start date parsed as YYYY-MM-DD; supply with end_date.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date.
		    time (int): HTTP request timeout in seconds. Defaults to 20.
		    location (str): Optional IPS location filter. Defaults to an empty string.
		    catalog (str): Optional IPS or CMEAnalysis catalog filter. Defaults to an empty string.
		    notification_type (str): Optional notifications type filter. Defaults to an empty
		        string.
		    most_accurate_only (bool): Restrict CME analyses to most accurate entries. Defaults to
		        True.
		    complete_entry_only (bool): Restrict CME analyses to complete entries. Defaults to True.
		    speed (int): Nonnegative CMEAnalysis speed filter. Defaults to 0.
		    half_angle (int): CMEAnalysis half-angle filter from 0 through 180 degrees. Defaults to
		        0.
		    keyword (str): Optional CMEAnalysis keyword. Defaults to an empty string.

		Returns:
		    Dict[str, Any]: Mode, endpoint, URL, parameters excluding the API key, and decoded JSON
		        array or object.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'most_accurate_only', most_accurate_only )
			self.most_accurate_only = most_accurate_only

			throw_if( 'complete_entry_only', complete_entry_only )
			self.complete_entry_only = complete_entry_only

			throw_if( 'endpoint', endpoint )

			throw_if( 'time', time )
			self.timeout = time
			self.endpoint = endpoint
			supported = { 'CME', 'CMEAnalysis', 'GST', 'IPS', 'FLR', 'SEP',
				'MPC', 'RBE', 'HSS', 'WSAEnlilSimulations', 'notifications' }
			if self.endpoint not in supported:
				raise ValueError( 'Unsupported NASA DONKI endpoint.' )
			self.start_date = start_date
			self.end_date = end_date
			if ( self.start_date != '' ) != ( self.end_date != '' ):
				raise ValueError( 'start_date and end_date must be supplied together.' )
			if self.start_date:
				start = dt.datetime.strptime( self.start_date, '%Y-%m-%d' ).date( )
				end = dt.datetime.strptime( self.end_date, '%Y-%m-%d' ).date( )
				if start > end:
					raise ValueError( 'start_date must precede or equal end_date.' )
			# NASA documents DEMO_KEY for limited unauthenticated use.
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { 'api_key': self.active_key }
			if self.start_date:
				self.params[ 'startDate' ] = self.start_date
				self.params[ 'endDate' ] = self.end_date
			self.location = location
			self.catalog = catalog
			self.notification_type = notification_type
			if self.endpoint == 'IPS':
				if self.location:
					self.params[ 'location' ] = self.location
				if self.catalog:
					self.params[ 'catalog' ] = self.catalog
			if self.endpoint == 'CMEAnalysis':
				self.speed = speed
				self.half_angle = half_angle
				if self.speed < 0 or not 0 <= self.half_angle <= 180:
					raise ValueError( 'CME speed and half_angle must be nonnegative; half_angle <= 180.' )
				self.params[ 'mostAccurateOnly' ] = str( self.most_accurate_only ).lower( )
				self.params[ 'completeEntryOnly' ] = str( self.complete_entry_only ).lower( )
				self.params[ 'speed' ] = self.speed
				self.params[ 'halfAngle' ] = self.half_angle
				if self.catalog:
					self.params[ 'catalog' ] = self.catalog
				self.keyword = keyword
				if self.keyword:
					self.params[ 'keyword' ] = self.keyword
			if self.endpoint == 'notifications' and self.notification_type:
				self.params[ 'type' ] = self.notification_type
			self.response = requests.get( url=self.url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, (list, dict) ):
				raise ValueError( 'NASA DONKI response must be JSON array or object.' )
			if isinstance( self.payload, dict ) and self.payload.get( 'error' ):
				raise ValueError( 'NASA DONKI returned an API error.' )
			self.result = { 'mode': self.mode, 'endpoint': self.endpoint,
				'url': self.url,
				'params': { key: value for key, value in self.params.items( )
					if key != 'api_key' }, 'data': self.payload }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = "fetch_endpoint( self, endpoint: str, start_date: str, end_date: str, time: int=20, location: str='', catalog: str='', notification_type: str='', most_accurate_only: bool=True, complete_entry_only: bool=True, speed: int=0, half_angle: int=0, keyword: str='' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='cme', start_date: str='', end_date: str='', time: int=20, location: str='ALL', catalog: str='ALL', notification_type: str='all', most_accurate_only: bool=True, complete_entry_only: bool=True, speed: int=0, half_angle: int=0, keyword: str='' ) -> Dict[ str, Any ] | None:
		"""Dispatch a DONKI retrieval operation.

		Purpose:
		    Map the selected operation to its provider endpoint and delegate with date and
		    endpoint-specific filters.

		Args:
		    mode (str): cme, cme_analysis, gst, ips, flr, sep, mpc, rbe, hss, wsa_enlil, or
		        notifications. Defaults to 'cme'.
		    start_date (str): Start date parsed as YYYY-MM-DD; supply with end_date. Defaults to an
		        empty string; omit both for provider defaults.
		    end_date (str): End date parsed as YYYY-MM-DD, on or after start_date. Defaults to an
		        empty string; omit both for provider defaults.
		    time (int): HTTP request timeout in seconds. Defaults to 20.
		    location (str): Optional IPS location filter. Defaults to 'ALL'.
		    catalog (str): Optional IPS or CMEAnalysis catalog filter. Defaults to 'ALL'.
		    notification_type (str): Optional notifications type filter. Defaults to 'all'.
		    most_accurate_only (bool): Restrict CME analyses to most accurate entries. Defaults to
		        True.
		    complete_entry_only (bool): Restrict CME analyses to complete entries. Defaults to True.
		    speed (int): Nonnegative CMEAnalysis speed filter. Defaults to 0.
		    half_angle (int): CMEAnalysis half-angle filter from 0 through 180 degrees. Defaults to
		        0.
		    keyword (str): Optional CMEAnalysis keyword. Defaults to an empty string.

		Returns:
		    Dict[str, Any]: Mode, endpoint, URL, parameters excluding the API key, and decoded JSON
		        array or object.

		Raises:
		    Error: Raised when input validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			
			endpoint_map = { 'cme': 'CME', 'cme_analysis': 'CMEAnalysis', 'gst': 'GST',
					'ips': 'IPS', 'flr': 'FLR', 'sep': 'SEP', 'mpc': 'MPC', 'rbe': 'RBE',
					'hss': 'HSS', 'wsa_enlil': 'WSAEnlilSimulations',
					'notifications': 'notifications' }
			
			if self.mode not in endpoint_map:
				raise ValueError( "Unsupported mode. Use 'cme', 'cme_analysis', 'gst', 'ips', "
				                  "'flr', 'sep', 'mpc', 'rbe', 'hss', 'wsa_enlil', "
				                  "or 'notifications'." )
			
			return self.fetch_endpoint( endpoint=endpoint_map[ self.mode ], start_date=start_date,
				end_date=end_date, time=time, location=location, catalog=catalog,
				notification_type=notification_type, most_accurate_only=most_accurate_only,
				complete_entry_only=complete_entry_only, speed=speed, half_angle=half_angle,
				keyword=keyword )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = "fetch( self, mode: str='cme', start_date: str='', end_date: str='', time: int=20, location: str='ALL', catalog: str='ALL', notification_type: str='all', most_accurate_only: bool=True, complete_entry_only: bool=True, speed: int=0, half_angle: int=0, keyword: str='' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} '
					               f'service.',
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class AstroCatalog( Fetcher ):
	"""SIMBAD and VizieR astronomical catalog adapter.

	Purpose:
	    Query named objects or ICRS coordinate regions through Astroquery, serialize Astropy
	    tables into row dictionaries, and optionally retain CSV or TSV representations.

	Attributes:
	    base_url (Optional[str]): Base service URL.
	    provider (str): Selected Astroquery provider; transient is mapped to VizieR.
	    catalog (str): VizieR catalog identifier or empty string.
	    route (str): Normalized object route or catalog region selector.
	    url (Optional[str]): Latest endpoint URL.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Any): Latest raw Astroquery table or table collection.
	    result (Dict[str, Any]): Latest serialized catalog response.
	    client (Any): Active Simbad or Vizier client.
	    coordinates (Optional[SkyCoord]): ICRS SkyCoord for the current cone search.
	    search_radius (Optional[u.Quantity]): Astropy angular quantity in arcminutes.
	    format (Optional[str]): Requested output representation.
	    name (Optional[str]): Astronomical object name.
	    right_ascension (Optional[str]): Latest right ascension value.
	    declination (Optional[str]): Latest declination value.
	    radius (Optional[int]): Latest search radius.
	    params (Optional[Dict[str, Any]]): Latest provider request parameters.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    timeout (Optional[int]): Request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    mode (str): Selected retrieval operation.
	    quantity (str): Legacy quantity path supplied to fetch.
	    attributes (str): Legacy attribute path supplied to fetch.
	    arguments (str): Provider-specific argument string supplied to fetch.
	"""
	base_url: Optional[ str ]
	format: Optional[ str ]
	name: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	radius: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Initialize SIMBAD as the default provider, JSON output, a twenty-second timeout, and
		    empty catalog, client, coordinate, and response state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://simbad.cds.unistra.fr'
		self.provider = 'simbad'
		self.catalog = ''
		self.route = ''
		self.url = self.base_url
		self.response = None
		self.payload = None
		self.result = { }
		self.client = None
		self.coordinates = None
		self.search_radius = None
		self.format = 'json'
		self.name = None
		self.right_ascension = None
		self.declination = None
		self.radius = None
		self.params = { }
		self.headers = { }
		self.timeout = 20
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'timeout', 'headers', 'provider', 'catalog',
				'route', 'format', 'name', 'right_ascension', 'declination', 'radius',
				'params', 'result', 'serialize_table', 'request',
				'fetch_object', 'cone_search', 'fetch' ]
	
	def normalize_attribute_path( self, quantity: str='', attributes: str='' ) -> str:
		"""Assemble a legacy attribute path.

		Purpose:
		    Strip nonblank quantity and comma-separated attribute components and join them with
		    slashes.

		Args:
		    quantity (str): Optional quantity path component. Defaults to an empty string.
		    attributes (str): Optional comma-separated attribute components. Defaults to an empty
		        string.

		Returns:
		    str: Slash-separated path, or an empty string when no components are supplied.
		"""
		parts: list[ str ] = [ ]
		if quantity and quantity.strip( ):
			parts.append( quantity.strip( ) )
		
		if attributes and attributes.strip( ):
			attr_parts = [ a.strip( ) for a in attributes.split( ',' ) if a.strip( ) ]
			parts.extend( attr_parts )
		
		return '/'.join( parts )
	
	def parse_argument( self, argument_string: str ) -> Dict[ str, Any ]:
		"""Parse provider argument text.

		Purpose:
		    Split text on commas and newlines, trim tokens, split each key/value pair on its first
		    equals sign, and use an empty value for bare keys.

		Args:
		    argument_string (str): Provider option text; blank text returns an empty dictionary.

		Returns:
		    Dict[str, Any]: Parsed option names mapped to string values.
		"""
		params: Dict[ str, Any ] = { }
		
		if not argument_string or not argument_string.strip( ):
			return params
		
		raw_items = re.split( r'[\n,]+', argument_string )
		items = [ item.strip( ) for item in raw_items if item and item.strip( ) ]
		
		for item in items:
			if '=' in item:
				k, v = item.split( '=', 1 )
				params[ k.strip( ) ] = v.strip( )
			else:
				params[ item ] = ''
		
		return params
	
	def serialize_table( self, table: Table, catalog: str ) -> Dict[ str, Any ]:
		"""Serialize an Astropy catalog table.

		Purpose:
		    Convert table rows to dictionaries, replace masked cells with None, unwrap NumPy
		    scalars, decode bytes as UTF-8, and stringify other unsupported cell values.

		Args:
		    table (Table): Astropy table to serialize.
		    catalog (str): Nonblank catalog label.

		Returns:
		    Dict[str, Any]: Catalog label, column names, serialized rows, and row count.

		Raises:
		    Error: Raised when validation or table serialization fails.
		"""
		try:
			throw_if( 'table', table )
			throw_if( 'catalog', catalog )
			rows: List[ Dict[ str, Any ] ] = [ ]
			for record in table:
				item: Dict[ str, Any ] = { }
				for column in table.colnames:
					value = record[ column ]
					if np.ma.is_masked( value ):
						item[ column ] = None
						continue
					if isinstance( value, np.generic ):
						value = value.item( )
					if isinstance( value, bytes ):
						value = value.decode( 'utf-8', errors='replace' )
					if isinstance( value, (str, int, float, bool) ) or value is None:
						item[ column ] = value
					else:
						item[ column ] = str( value )
				rows.append( item )
			return { 'catalog': catalog, 'columns': list( table.colnames ),
				'rows': rows, 'count': len( rows ) }
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = 'serialize_table( self, table: Table, catalog: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception

	def request( self, route: str, params: Dict[ str, Any ] | None=None, time: int=20 ) -> Any:
		"""Execute an astronomical catalog query.

		Purpose:
		    Select SIMBAD or VizieR (including the transient alias), resolve the object or cone
		    route, query Astroquery, serialize returned tables, and optionally produce delimited
		    text.

		Args:
		    route (str): Nonblank object route or the literal 'catalog' for a cone query.
		    params (Dict[str, Any] | None): Provider options; None uses an empty dictionary.
		        Supports format (json/csv/tsv), provider (simbad/vizier/transient), catalog
		        (required for VizieR), ra and dec (degrees, required for cones), radius (positive
		        arcminutes, default 2), and VizieR row_limit (default 50).
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Provider, catalog, retrieval mode, serialized tables, total row count,
		        and formatted_tables for CSV or TSV output.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'route', route )

			throw_if( 'time', time )
			self.timeout = time
			self.route = route.strip( ).lstrip( '/' )
			self.params = params if params is not None else { }
			self.format = str( self.params.get( 'format', 'json' ) ).strip( ).lower( )
			if self.format not in ('json', 'csv', 'tsv'):
				raise ValueError( 'Unsupported AstroCatalog output format.' )
			self.catalog = str( self.params.get( 'catalog', '' ) ).strip( )
			self.provider = str( self.params.get( 'provider',
				'vizier' if self.catalog else 'simbad' ) ).strip( ).lower( )
			if self.provider == 'transient':
				self.provider = 'vizier'
			if self.provider not in ('simbad', 'vizier'):
				raise ValueError( 'provider must be simbad, vizier, or transient.' )
			if self.provider == 'vizier' and not self.catalog:
				raise ValueError( 'A VizieR catalog identifier is required.' )
			self.base_url = ('https://simbad.cds.unistra.fr'
				if self.provider == 'simbad' else 'https://vizier.cds.unistra.fr')
			self.url = self.base_url
			self.response = None
			is_region = self.route == 'catalog'
			if is_region:
				throw_if( 'ra', self.params.get( 'ra' ) )
				throw_if( 'dec', self.params.get( 'dec' ) )
				self.right_ascension = str( self.params[ 'ra' ] )
				self.declination = str( self.params[ 'dec' ] )
				self.radius = int( self.params.get( 'radius', 2 ) )
				if self.radius < 1:
					raise ValueError( 'radius must be positive.' )
				self.coordinates = SkyCoord(
					self.right_ascension, self.declination,
					unit=(u.deg, u.deg), frame='icrs' )
				self.search_radius = self.radius * u.arcmin
			else:
				self.name = urllib.parse.unquote( self.route )
				throw_if( 'name', self.name )
			if self.provider == 'simbad':
				self.client = Simbad( )
				self.client.timeout = self.timeout
				self.payload = (self.client.query_region(
					self.coordinates, radius=self.search_radius )
					if is_region else self.client.query_object( self.name ))
				raw_tables = ([ ('SIMBAD', self.payload) ]
					if self.payload is not None else [ ])
			else:
				self.client = Vizier( row_limit=int( self.params.get( 'row_limit', 50 ) ) )
				self.client.TIMEOUT = self.timeout
				self.payload = (self.client.query_region(
					self.coordinates, radius=self.search_radius, catalog=self.catalog )
					if is_region else self.client.query_object(
					self.name, catalog=self.catalog ))
				raw_tables = [(str( key ), table)
					for key, table in self.payload.items( )]
			tables = [self.serialize_table( table, name )
				for name, table in raw_tables]
			self.result = { 'provider': self.provider, 'catalog': self.catalog,
				'mode': 'cone_search' if is_region else 'object_query',
				'tables': tables, 'count': sum( item[ 'count' ] for item in tables ) }
			if self.format in ('csv', 'tsv'):
				formatted: List[ Dict[ str, str ] ] = [ ]
				for name, table in raw_tables:
					output = io.StringIO( )
					table.write( output, format='ascii.csv' if self.format == 'csv'
						else 'ascii.tab', overwrite=True )
					formatted.append( { 'catalog': name, 'data': output.getvalue( ) } )
				self.result[ 'formatted_tables' ] = formatted
			return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = 'request( self, route: str, params: Dict[ str, Any ] | None=None, time: int=20 ) -> Any'
			Logger( ).write( exception )
			raise exception

	def fetch_object( self, name: str, quantity: str='', attributes: str='', arguments: str='', data_format: str='json', time: int=20 ) -> Any:
		"""Query an astronomical object by name.

		Purpose:
		    Encode the object name, reject unsupported legacy quantity and attribute paths, parse
		    provider options, and delegate to request.

		Args:
		    name (str): Nonblank astronomical object name or identifier.
		    quantity (str): Legacy quantity path; nonblank content is rejected. Defaults to an empty
		        string.
		    attributes (str): Legacy comma-separated attribute path; nonblank parsed content is
		        rejected. Defaults to an empty string.
		    arguments (str): Comma- or newline-separated provider options, such as provider,
		        catalog, and row_limit. Defaults to an empty string.
		    data_format (str): json, csv, or tsv. Defaults to 'json'.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Provider, catalog, retrieval mode, serialized tables, total row count,
		        and formatted_tables for CSV or TSV output.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'name', name )
			self.name = name
			throw_if( 'data_format', data_format )
			self.format = data_format
			if self.format not in ('json', 'csv', 'tsv'):
				raise ValueError( 'Unsupported AstroCatalog output format.' )
			route_parts = [ urllib.parse.quote( self.name, safe='' ) ]
			attr_path = self.normalize_attribute_path( quantity, attributes )
			if attr_path:
				raise ValueError( 'Legacy quantity/attributes paths are not supported by SIMBAD; '
					'use arguments="provider=vizier,catalog=<ID>" instead.' )
			
			self.route = '/'.join( route_parts )
			self.params = self.parse_argument( arguments )
			if self.format:
				self.params[ 'format' ] = self.format

			throw_if( 'time', time )
			self.timeout = time
			return self.request( route=self.route, params=self.params, time=self.timeout )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = "fetch_object( self, name: str, quantity: str='', attributes: str='', arguments: str='', data_format: str='json', time: int=20 ) -> Any"
			Logger( ).write( exception )
			raise exception
	
	def cone_search( self, ra: str, dec: str, radius: int=2, quantity: str='', attributes: str='', arguments: str='', data_format: str='json', time: int=20 ) -> Any:
		"""Query an astronomical coordinate cone.

		Purpose:
		    Validate coordinates, radius and format, reject unsupported legacy paths, and delegate
		    to request using the catalog route.

		Args:
		    ra (str): Nonblank right ascension interpreted in degrees.
		    dec (str): Nonblank declination interpreted in degrees.
		    radius (int): Positive cone radius in arcminutes. Defaults to 2.
		    quantity (str): Legacy quantity path; nonblank content is rejected. Defaults to an empty
		        string.
		    attributes (str): Legacy comma-separated attribute path; nonblank parsed content is
		        rejected. Defaults to an empty string.
		    arguments (str): Comma- or newline-separated provider options, such as provider,
		        catalog, and row_limit. Defaults to an empty string.
		    data_format (str): json, csv, or tsv. Defaults to 'json'.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Provider, catalog, retrieval mode, serialized tables, total row count,
		        and formatted_tables for CSV or TSV output.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			self.right_ascension = ra
			self.declination = dec
			self.radius = radius
			throw_if( 'data_format', data_format )
			self.format = data_format
			if self.radius < 1:
				raise ValueError( 'radius must be positive.' )
			if self.format not in ('json', 'csv', 'tsv'):
				raise ValueError( 'Unsupported AstroCatalog output format.' )
			self.route = 'catalog'
			attr_path = self.normalize_attribute_path( quantity, attributes )
			if attr_path:
				raise ValueError( 'Legacy quantity/attributes paths are not supported by SIMBAD; '
					'use arguments="provider=vizier,catalog=<ID>" instead.' )
			
			self.params = self.parse_argument( arguments )
			self.params[ 'ra' ] = self.right_ascension
			self.params[ 'dec' ] = self.declination
			self.params[ 'radius' ] = str( self.radius )
			if self.format:
				self.params[ 'format' ] = self.format

			throw_if( 'time', time )
			self.timeout = time
			return self.request( route=self.route, params=self.params, time=self.timeout )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = "cone_search( self, ra: str, dec: str, radius: int=2, quantity: str='', attributes: str='', arguments: str='', data_format: str='json', time: int=20 ) -> Any"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='object_query', query: str='', quantity: str='', attributes:str='', arguments: str='', ra: str='', dec: str='', radius: int=2, data_format: str='json', time: int=20 ) -> Any:
		"""Dispatch a catalog query.

		Purpose:
		    Select object_query or cone_search and delegate with the corresponding name or
		    coordinate arguments.

		Args:
		    mode (str): object_query or cone_search. Defaults to 'object_query'.
		    query (str): Object name for object_query. Defaults to an empty string; required in
		        object mode.
		    quantity (str): Legacy quantity path; nonblank content is rejected. Defaults to an empty
		        string.
		    attributes (str): Legacy comma-separated attribute path; nonblank parsed content is
		        rejected. Defaults to an empty string.
		    arguments (str): Comma- or newline-separated provider options, such as provider,
		        catalog, and row_limit. Defaults to an empty string.
		    ra (str): Nonblank right ascension interpreted in degrees. Defaults to an empty string;
		        required in cone mode.
		    dec (str): Nonblank declination interpreted in degrees. Defaults to an empty string;
		        required in cone mode.
		    radius (int): Positive cone radius in arcminutes. Defaults to 2.
		    data_format (str): json, csv, or tsv. Defaults to 'json'.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Provider, catalog, retrieval mode, serialized tables, total row count,
		        and formatted_tables for CSV or TSV output.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			self.quantity = quantity
			self.attributes = attributes
			self.arguments = arguments
			self.format = data_format

			throw_if( 'time', time )
			self.timeout = time
			if self.mode == 'object_query':
				throw_if( 'query', query )
				self.name = query
				return self.fetch_object( name=self.name, quantity=self.quantity,
					attributes=self.attributes, arguments=self.arguments,
					data_format=self.format, time=self.timeout )
			if self.mode == 'cone_search':
				throw_if( 'ra', ra )
				throw_if( 'dec', dec )
				self.right_ascension = ra
				self.declination = dec
				self.radius = radius
				return self.cone_search( ra=self.right_ascension,
					dec=self.declination, radius=self.radius, quantity=self.quantity,
					attributes=self.attributes, arguments=self.arguments,
					data_format=self.format, time=self.timeout )
			raise ValueError( "Unsupported mode. Use 'object_query' or 'cone_search'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = "fetch( self, mode: str='object_query', query: str='', quantity: str='', attributes:str='', arguments: str='', ra: str='', dec: str='', radius: int=2, data_format: str='json', time: int=20 ) -> Any"
			Logger( ).write( exception )
			raise exception

class AstroQuery( Fetcher ):
	"""SIMBAD object and region retrieval adapter.

	Purpose:
	    Use Astroquery Simbad to retrieve object records, alternate identifiers, and sky regions,
	    converting returned Astropy table cells to strings.

	Attributes:
	    url (Optional[str]): Reserved URL state, initialized to None.
	    radius (Optional[float]): Latest search radius.
	    name (Optional[str]): Astronomical object name.
	    right_ascension (Optional[str]): Latest right ascension value.
	    declination (Optional[str]): Latest declination value.
	    params (Optional[Dict[str, Any]]): Reserved parameter dictionary, initialized empty.
	    row_limit (Optional[int]): Requested SIMBAD row limit.
	    radius_unit (str): Angular unit used for region retrieval.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): Inherited HTTP header state consulted during
	        initialization.
	"""
	url: Optional[ str ]
	radius: Optional[ float ]
	name: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	row_limit: Optional[ int ]
	radius_unit: str
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Initialize object and coordinate placeholders, a row limit of 100, degree radius units,
		    and configured user-agent state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.url = None
		self.radius = None
		self.name = None
		self.right_ascension = None
		self.declination = None
		self.params = { }
		self.row_limit = 100
		self.radius_unit = 'deg'
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'headers', 'row_limit', 'object_search', 'object_ids', 'region_search', 'fetch', ]
	
	def table_to_records( self, table: Table | None ) -> List[ Dict[ str, Any ] ]:
		"""Convert an Astropy table to text records.

		Purpose:
		    Convert each cell to a string, substituting an empty string for individual conversion
		    failures; return no records for None.

		Args:
		    table (Table | None): Astropy table or None.

		Returns:
		    List[Dict[str, Any]]: Row dictionaries containing string values.

		Raises:
		    Error: Raised when table-level conversion fails.
		"""
		try:
			if table is None:
				return [ ]
			
			records: List[ Dict[ str, Any ] ] = [ ]
			for row in table:
				record: Dict[ str, Any ] = { }
				for col in table.colnames:
					try:
						value = row[ col ]
						record[ str( col ) ] = str( value )
					except Exception:
						record[ str( col ) ] = ''
				records.append( record )
			
			return records
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = 'table_to_records( self, table: Table | None ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def object_search( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ] | None:
		"""Retrieve SIMBAD object records.

		Purpose:
		    Set the SIMBAD row limit, query the named object, and convert the result table to text
		    records.

		Args:
		    name (str): Nonblank object name.
		    row_limit (int): SIMBAD row limit. Defaults to 100.

		Returns:
		    Dict[str, Any]: Mode, object query, row limit, column names, and string-valued row
		        dictionaries; empty columns and rows if no table is returned.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'name', name )
			self.name = name
			throw_if( 'row_limit', row_limit )
			self.row_limit = row_limit			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_object( self.name )
			
			return { 'mode': 'object_search', 'query': self.name, 'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self.table_to_records( result_table ), }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = 'object_search( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def object_ids( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ] | None:
		"""Retrieve alternate SIMBAD identifiers.

		Purpose:
		    Set the SIMBAD row limit, query identifiers for the named object, and convert the result
		    table to text records.

		Args:
		    name (str): Nonblank object name.
		    row_limit (int): SIMBAD row limit. Defaults to 100.

		Returns:
		    Dict[str, Any]: Mode, object query, row limit, column names, and string-valued row
		        dictionaries; empty columns and rows if no table is returned.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'name', name )
			self.name = name
			throw_if( 'row_limit', row_limit )
			self.row_limit = row_limit
			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_objectids( self.name )
			
			return { 'mode': 'object_ids', 'query': self.name, 'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self.table_to_records( result_table ), }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = 'object_ids( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def region_search( self, ra: str, dec: str, radius: float=0.5, radius_unit: str='deg',row_limit: int=100 ) -> Dict[ str, Any ] | None:
		"""Retrieve SIMBAD objects in a sky region.

		Purpose:
		    Construct a SkyCoord with hour-angle right ascension and degree declination, resolve the
		    radius unit, and query SIMBAD around the coordinate.

		Args:
		    ra (str): Nonblank right ascension interpreted as hour angle.
		    dec (str): Nonblank declination interpreted in degrees.
		    radius (float): Angular search radius. Defaults to 0.5.
		    radius_unit (str): deg, arcmin, or arcsec. Defaults to 'deg'.
		    row_limit (int): SIMBAD row limit. Defaults to 100.

		Returns:
		    Dict[str, Any]: Mode, coordinate values, radius, radius unit, row limit, column names,
		        and string-valued rows.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'ra', ra )
			self.right_ascension = ra
			throw_if( 'dec', dec )
			self.declination = dec
			throw_if( 'radius', radius )
			self.radius = radius
			throw_if( 'radius_unit', radius_unit )
			self.radius_unit = radius_unit
			throw_if( 'row_limit', row_limit )
			self.row_limit = row_limit
			
			unit_map = { 'deg': u.deg, 'arcmin': u.arcmin, 'arcsec': u.arcsec, }
			
			active_unit = unit_map[ self.radius_unit ]			
			coord = SkyCoord( ra=self.right_ascension, dec=self.declination, unit=(u.hourangle,
					u.deg) )
			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_region( coordinates=coord, radius=self.radius * active_unit)
			
			return { 'mode': 'region_search', 'ra': self.right_ascension, 'dec': self.declination,
					'radius': self.radius, 'radius_unit': self.radius_unit,
					'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self.table_to_records( result_table ), }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = "region_search( self, ra: str, dec: str, radius: float=0.5, radius_unit: str='deg',row_limit: int=100 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='object_search', query: str='', ra: str='', dec: str='', radius: float=0.5, radius_unit: str='deg', row_limit: int=100 ) -> Dict[ str, Any ] | None:
		"""Dispatch a SIMBAD retrieval operation.

		Purpose:
		    Normalize mode and select object_search, object_ids, or region_search.

		Args:
		    mode (str): object_search, object_ids, or region_search. Defaults to 'object_search';
		        blank selects the same mode.
		    query (str): Object name for object operations. Defaults to an empty string.
		    ra (str): Nonblank right ascension interpreted as hour angle. Defaults to an empty
		        string; required for region mode.
		    dec (str): Nonblank declination interpreted in degrees. Defaults to an empty string;
		        required for region mode.
		    radius (float): Angular search radius. Defaults to 0.5.
		    radius_unit (str): deg, arcmin, or arcsec. Defaults to 'deg'.
		    row_limit (int): SIMBAD row limit. Defaults to 100.

		Returns:
		    Dict[str, Any]: Query metadata, column names, and string-valued rows for the selected
		        operation.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			active_mode = (mode or 'object_search').strip( ).lower( )
			
			if active_mode == 'object_search':
				return self.object_search( name=query, row_limit=row_limit )
			
			if active_mode == 'object_ids':
				return self.object_ids( name=query, row_limit=row_limit )
			
			if active_mode == 'region_search':
				return self.region_search( ra=ra, dec=dec, radius=radius, radius_unit=radius_unit,
					row_limit=row_limit )
			
			raise ValueError( "Unsupported mode. Use 'object_search', 'object_ids', "
			                  "or 'region_search'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = "fetch( self, mode: str='object_search', query: str='', ra: str='', dec: str='', radius: float=0.5, radius_unit: str='deg', row_limit: int=100 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception

class StarMap( Fetcher ):
	"""Sky-Map interactive page and snapshot adapter.

	Purpose:
	    Request interactive sky-map pages by object or coordinates and retrieve snapshot pages
	    containing links to available image formats.

	Attributes:
	    base_url (Optional[str]): Interactive Sky-Map page URL.
	    snapshot_url (Optional[str]): Sky-Map snapshot page URL.
	    image_source (Optional[str]): Requested sky imagery source.
	    object (Optional[str]): Latest named object.
	    right_ascension (Optional[float]): Latest right ascension value.
	    declination (Optional[float]): Latest declination value.
	    box_color (Optional[str]): Marker box color.
	    show_box (Optional[bool]): Marker box visibility flag.
	    show_grid (Optional[bool]): Coordinate grid visibility flag.
	    show_lines (Optional[bool]): Constellation line visibility flag.
	    show_boundaries (Optional[bool]): Constellation boundary visibility flag.
	    show_const_names (Optional[bool]): Constellation name visibility flag.
	    zoom (Optional[int]): Requested map zoom level.
	    params (Optional[Dict[str, Any]]): Latest provider request parameters.
	    timeout (Optional[int]): Request timeout in seconds.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    response (Optional[Response]): Latest HTTP response.
	"""
	base_url: Optional[ str ]
	snapshot_url: Optional[ str ]
	image_source: Optional[ str ]
	object: Optional[ str ]
	right_ascension: Optional[ float ]
	declination: Optional[ float ]
	box_color: Optional[ str ]
	show_box: Optional[ bool ]
	show_grid: Optional[ bool ]
	show_lines: Optional[ bool ]
	show_boundaries: Optional[ bool ]
	show_const_names: Optional[ bool ]
	zoom: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure interactive and snapshot URLs, DSS2 imagery, yellow marker box, grid and
		    constellation overlays, zoom level five, and a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://www.sky-map.org/'
		self.snapshot_url = 'https://www.sky-map.org/snapshot'
		self.image_source = 'DSS2'
		self.object = None
		self.right_ascension = None
		self.declination = None
		self.box_color = 'yellow'
		self.show_box = True
		self.show_grid = True
		self.show_lines = True
		self.show_boundaries = True
		self.show_const_names = False
		self.zoom = 5
		self.params = { }
		self.timeout = 20
		self.headers = { }
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[
				'Accept' ] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'snapshot_url', 'object', 'right_ascension', 'declination',
				'image_source', 'zoom', 'show_box', 'show_grid', 'show_lines', 'show_boundaries',
				'show_const_names', 'box_color', 'params', 'fetch_object_link',
				'fetch_coordinate_link', 'fetch_snapshot', 'fetch', ]
	
	def normalize( self, value: bool ) -> str:
		"""Encode a display flag.

		Purpose:
		    Convert a truthy value to the string '1' and a falsey value to '0' after checking for a
		    missing argument.

		Args:
		    value (bool): Display flag.

		Returns:
		    str: '1' or '0'.

		Raises:
		    Error: Raised when validation or conversion fails.
		"""
		try:

			throw_if( 'value', value )

			return '1' if value else '0'
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = 'normalize( self, value: bool ) -> str'
			Logger( ).write( exception )
			raise exception

	def extract_links( self, html: str, base_url: str ) -> Dict[ str, str ]:
		"""Extract image-format links from HTML.

		Purpose:
		    Match anchor elements labeled JPEG, PNG, GIF, BMP, or TIFF, resolve their references
		    against the base URL, and retain the last link for each lowercase format.

		Args:
		    html (str): Snapshot HTML; empty or non-string content returns no links.
		    base_url (str): Base URL for relative references.

		Returns:
		    Dict[str, str]: Image-format labels mapped to resolved URLs.

		Raises:
		    Error: Raised when link extraction fails.
		"""
		try:
			links: Dict[ str, str ] = { }
			if not html or not isinstance( html, str ):
				return links
			
			pattern = re.compile( r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>\s*('
			                      r'jpeg|png|gif|bmp|tiff)\s*</a>', flags=re.IGNORECASE )
			
			for match in pattern.finditer( html ):
				href = match.group( 1 )
				label = match.group( 2 ).lower( )
				links[ label ] = urllib.parse.urljoin( base_url, href )
			
			return links
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = 'extract_links( self, html: str, base_url: str ) -> Dict[ str, str ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_object_link( self, name: str, zoom: int=5, box_color: str='yellow', show_box: bool=True, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve an interactive object map page.

		Purpose:
		    Construct a named-object URL with marker and zoom options, issue a GET request, and
		    return metadata and an HTML preview.

		Args:
		    name (str): Nonblank astronomical object name.
		    zoom (int): Map zoom level. Defaults to 5.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, target, zoom, submitted parameters, interactive URL, HTTP status,
		        and first 2000 characters of HTML.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'show_box', show_box )

			throw_if( 'name', name )
			
			self.object = name
			self.zoom = zoom
			self.box_color = box_color
			self.show_box = show_box

			throw_if( 'time', time )
			self.timeout = time
			
			self.params = { 'object': self.object, 'show_box': self.normalize( self.show_box ),
					'zoom': str( self.zoom ), 'box_color': self.box_color, 'box_width': '50',
					'box_height': '50', }
			
			interactive_url = f'{self.base_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get( url=self.base_url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			
			return { 'mode': 'object_link', 'object': self.object, 'zoom': self.zoom,
					'params': self.params, 'interactive_url': interactive_url,
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ], }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = "fetch_object_link( self, name: str, zoom: int=5, box_color: str='yellow', show_box: bool=True, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_coordinate_link( self, ra: float, dec: float, zoom: int=5, box_color: str = 'yellow', show_box: bool=True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve an interactive coordinate map page.

		Purpose:
		    Construct a coordinate-centered URL with marker, grid, and constellation options, issue
		    a GET request, and return metadata and an HTML preview.

		Args:
		    ra (float): Right ascension value passed to Sky-Map.
		    dec (float): Declination value passed to Sky-Map.
		    zoom (int): Map zoom level. Defaults to 5.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, target, zoom, submitted parameters, interactive URL, HTTP status,
		        and first 2000 characters of HTML.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'show_box', show_box )

			throw_if( 'show_grid', show_grid )

			throw_if( 'show_lines', show_lines )

			throw_if( 'show_boundaries', show_boundaries )

			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = ra
			self.declination = dec
			self.zoom = zoom
			self.box_color = box_color
			self.show_box = show_box
			self.show_grid = show_grid
			self.show_lines = show_lines
			self.show_boundaries = show_boundaries

			throw_if( 'time', time )
			self.timeout = time
			
			self.params = { 'ra': f'{self.right_ascension}', 'de': f'{self.declination}',
					'show_box': self.normalize( self.show_box ), 'zoom': str( self.zoom ),
					'box_color': self.box_color, 'show_grid': self.normalize( self.show_grid ),
					'show_constellation_lines': self.normalize( self.show_lines ),
					'show_constellation_boundaries': self.normalize( self.show_boundaries ), }
			
			interactive_url = f'{self.base_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get( url=self.base_url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			
			return { 'mode': 'coordinate_link', 'ra': self.right_ascension, 'dec':
				self.declination,
					'zoom': self.zoom, 'params': self.params, 'interactive_url': interactive_url,
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ], }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = "fetch_coordinate_link( self, ra: float, dec: float, zoom: int=5, box_color: str = 'yellow', show_box: bool=True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_snapshot( self, ra: float, dec: float, zoom: int=10, image_source: str='DSS2', show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve snapshot image links.

		Purpose:
		    Request the coordinate-centered snapshot page, extract image-format links, and prefer
		    PNG over JPEG without downloading the linked images.

		Args:
		    ra (float): Right ascension value passed to Sky-Map.
		    dec (float): Declination value passed to Sky-Map.
		    zoom (int): Map zoom level. Defaults to 10.
		    image_source (str): Sky imagery source identifier. Defaults to 'DSS2'.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    show_const_names (bool): Display constellation names. Defaults to False.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Coordinates, zoom, imagery source, parameters, snapshot page URL, image
		        links, preferred image URL, HTTP status, and HTML preview.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'show_grid', show_grid )

			throw_if( 'show_lines', show_lines )

			throw_if( 'show_boundaries', show_boundaries )

			throw_if( 'show_const_names', show_const_names )

			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = ra
			self.declination = dec
			self.zoom = zoom
			self.image_source = image_source
			self.show_grid = show_grid
			self.show_lines = show_lines
			self.show_boundaries = show_boundaries
			self.show_const_names = show_const_names

			throw_if( 'time', time )
			self.timeout = time
			
			self.params = { 'ra': f'{self.right_ascension}', 'de': f'{self.declination}',
					'zoom': str( self.zoom ), 'img_source': self.image_source,
					'show_grid': self.normalize( self.show_grid ),
					'show_constellation_lines': self.normalize( self.show_lines ),
					'show_constellation_boundaries': self.normalize( self.show_boundaries ),
					'show_const_names': self.normalize( self.show_const_names ), }
			
			page_url = f'{self.snapshot_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get( url=self.snapshot_url, params=self.params,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			
			image_links = self.extract_links( html=self.response.text, base_url=self.snapshot_url )
			
			return { 'mode': 'snapshot', 'ra': self.right_ascension, 'dec': self.declination,
					'zoom': self.zoom, 'image_source': self.image_source, 'params': self.params,
					'snapshot_page_url': page_url, 'image_links': image_links,
					'preferred_image_url': image_links.get( 'png' ) or image_links.get( 'jpeg',
						'' ),
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ], }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = "fetch_snapshot( self, ra: float, dec: float, zoom: int=10, image_source: str='DSS2', show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='object_link', query: str='', ra: float=0.0, dec: float = 0.0, zoom: int=5, image_source: str='DSS2', box_color: str='yellow', show_box: bool = True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a Sky-Map retrieval operation.

		Purpose:
		    Select object_link, coordinate_link, or snapshot and forward applicable map options.

		Args:
		    mode (str): object_link, coordinate_link, or snapshot. Defaults to 'object_link'.
		    query (str): Object name for object_link. Defaults to an empty string.
		    ra (float): Right ascension value passed to Sky-Map. Defaults to 0.0.
		    dec (float): Declination value passed to Sky-Map. Defaults to 0.0.
		    zoom (int): Map zoom level. Defaults to 5.
		    image_source (str): Sky imagery source identifier. Defaults to 'DSS2'.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    show_const_names (bool): Display constellation names. Defaults to False.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Interactive page or snapshot link metadata for the selected operation.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			active_mode = mode
			
			if active_mode == 'object_link':
				return self.fetch_object_link( name=query, zoom=zoom, box_color=box_color,
					show_box=show_box, time=time )
			
			if active_mode == 'coordinate_link':
				return self.fetch_coordinate_link( ra=ra, dec=dec, zoom=zoom, box_color=box_color,
					show_box=show_box, show_grid=show_grid, show_lines=show_lines,
					show_boundaries=show_boundaries, time=time )
			
			if active_mode == 'snapshot':
				return self.fetch_snapshot( ra=ra, dec=dec, zoom=zoom, image_source=image_source,
					show_grid=show_grid, show_lines=show_lines, show_boundaries=show_boundaries,
					show_const_names=show_const_names, time=time )
			
			raise ValueError( "Unsupported mode. Use 'object_link', 'coordinate_link', "
			                  "or 'snapshot'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = "fetch( self, mode: str='object_link', query: str='', ra: float=0.0, dec: float = 0.0, zoom: int=5, image_source: str='DSS2', box_color: str='yellow', show_box: bool = True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception

class GovData( Fetcher ):
	"""U.S. Government Publishing Office GovInfo API adapter.

	Purpose:
	    Search GovInfo publications, retrieve package summaries, and list collection packages
	    using the configured API key and provider pagination controls.

	Attributes:
	    api_key (Optional[str]): GovInfo API key loaded from configuration.
	    base_url (Optional[str]): Base service URL.
	    url (Optional[str]): Latest endpoint URL.
	    params (Optional[Dict[str, Any]]): Latest query parameters, including the API key.
	    payload (Optional[Dict[str, Any]]): Latest search POST body or empty dictionary for GET
	        requests.
	    result (Optional[Dict[str, Any]]): Latest packaged GovInfo response.
	    mode (Optional[str]): Selected retrieval operation.
	    query (Optional[str]): Latest search query.
	    page_size (Optional[int]): Requested page size.
	    offset_mark (Optional[str]): Provider pagination cursor.
	    sort_field (Optional[str]): Search sort field.
	    sort_order (Optional[str]): Search sort direction.
	    package_id (Optional[str]): Requested GovInfo package identifier.
	    collection (Optional[str]): Requested GovInfo collection code.
	    start_date (Optional[str]): Collection start-date path value.
	    response (Optional[Response]): Latest HTTP response.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    timeout (Optional[int]): Request timeout in seconds.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	page_size: Optional[ int ]
	offset_mark: Optional[ str ]
	sort_field: Optional[ str ]
	sort_order: Optional[ str ]
	package_id: Optional[ str ]
	collection: Optional[ str ]
	start_date: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the GovInfo API key from configuration and initialize JSON headers, search
		    defaults, pagination cursor, and empty request state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.base_url = 'https://api.govinfo.gov'
		self.url = None
		self.params = { }
		self.payload = { }
		self.result = { }
		self.mode = 'search'
		self.query = ''
		self.page_size = 10
		self.offset_mark = '*'
		self.sort_field = 'score'
		self.sort_order = 'DESC'
		self.package_id = ''
		self.collection = ''
		self.start_date = ''
		self.response = None
		self.headers = { 'Accept': 'application/json', 'Content-Type': 'application/json',
				'User-Agent': cfg.AGENTS }
		self.agents = cfg.AGENTS
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'url', 'params', 'payload', 'result', 'mode', 'query',
				'page_size', 'offset_mark', 'sort_field', 'sort_order', 'package_id', 'collection',
				'start_date', 'response', 'headers', 'agents', 'validate_page_size',
				'validate_sort_field', 'validate_sort_order', 'fetch_search',
				'fetch_package_summary', 'fetch_collection', 'fetch', 'create_schema' ]
	
	def validate_page_size( self, page_size: int ) -> int:
		"""Validate the requested page size.

		Purpose:
		    Check the requested page size against the inclusive range from 1 to 1000.

		Args:
		    page_size (int): Requested page size, from 1 through 1000.

		Returns:
		    int: Original accepted page size.

		Raises:
		    Error: Raised when the page size is missing or outside the range.
		"""
		try:
			

			throw_if( 'page_size', page_size )

			if page_size < 1 or page_size > 1000:
				raise ValueError( 'page_size must be between 1 and 1000.' )
			
			return page_size
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = 'validate_page_size( self, page_size: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_sort_field( self, sort_field: str ) -> str:
		"""Validate a GovInfo search sort field.

		Purpose:
		    Accept only score or lastModified.

		Args:
		    sort_field (str): score or lastModified.

		Returns:
		    str: Original accepted sort field.

		Raises:
		    Error: Raised when the field is missing or unsupported.
		"""
		try:
			throw_if( 'sort_field', sort_field )
			value = sort_field
			allowed = { 'score', 'lastModified' }
			
			if value not in allowed:
				raise ValueError( "Unsupported sort field. Use 'score' or 'lastModified'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = 'validate_sort_field( self, sort_field: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_sort_order( self, sort_order: str ) -> str:
		"""Validate a GovInfo search sort direction.

		Purpose:
		    Accept only ASC or DESC.

		Args:
		    sort_order (str): ASC or DESC.

		Returns:
		    str: Original accepted sort direction.

		Raises:
		    Error: Raised when the direction is missing or unsupported.
		"""
		try:
			throw_if( 'sort_order', sort_order )
			value = sort_order
			allowed = { 'ASC', 'DESC' }
			
			if value not in allowed:
				raise ValueError( "Unsupported sort order. Use 'ASC' or 'DESC'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = 'validate_sort_order( self, sort_order: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch_search( self, query: str, page_size: int=10, offset_mark: str='*', sort_field: str = 'score', sort_order: str='DESC', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search GovInfo publications.

		Purpose:
		    Validate query, paging and sorting options, submit a JSON POST to the search endpoint,
		    and package the decoded response.

		Args:
		    query (str): Nonblank GovInfo search expression.
		    page_size (int): Requested page size from 1 through 1000. Defaults to 10.
		    offset_mark (str): Nonblank pagination cursor. Defaults to '*'.
		    sort_field (str): score or lastModified. Defaults to 'score'.
		    sort_order (str): ASC or DESC. Defaults to 'DESC'.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, query parameters (including the API key), request payload,
		        and decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'offset_mark', offset_mark )

			
			self.mode = 'search'
			self.query = query
			self.page_size = self.validate_page_size( page_size )
			self.offset_mark = offset_mark
			self.sort_field = self.validate_sort_field( sort_field )
			self.sort_order = self.validate_sort_order( sort_order )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/search'
			self.params = { 'api_key': self.api_key }
			self.payload = { 'query': self.query, 'pageSize': self.page_size,
					'offsetMark': self.offset_mark,
					'sorts': [ { 'field': self.sort_field, 'sortOrder': self.sort_order } ] }
			
			self.response = requests.post( url=self.url, params=self.params, json=self.payload,
				headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'payload': self.payload, 'data': self.response.json( ) }
			
			return self.result
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = "fetch_search( self, query: str, page_size: int=10, offset_mark: str='*', sort_field: str = 'score', sort_order: str='DESC', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_package_summary( self, package_id: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve a GovInfo package summary.

		Purpose:
		    Request the package summary endpoint by package identifier and package its decoded
		    response.

		Args:
		    package_id (str): Nonblank GovInfo package identifier.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, query parameters (including the API key), request payload,
		        and decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'package_id', package_id )

			
			self.mode = 'package_summary'
			self.package_id = package_id
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/packages/{self.package_id}/summary'
			self.params = { 'api_key': self.api_key }
			self.payload = { }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'payload': self.payload, 'data': self.response.json( ) }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = 'fetch_package_summary( self, package_id: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_collection( self, collection: str, start_date: str, page_size: int=10, offset_mark: str='*', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve GovInfo collection packages.

		Purpose:
		    Build the collection endpoint using the collection code and start-date path value,
		    attach pagination parameters, and package the decoded response.

		Args:
		    collection (str): Nonblank collection code.
		    start_date (str): Nonblank start-date path value passed to GovInfo.
		    page_size (int): Requested page size from 1 through 1000. Defaults to 10.
		    offset_mark (str): Nonblank pagination cursor. Defaults to '*'.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, query parameters (including the API key), request payload,
		        and decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'collection', collection )
			throw_if( 'start_date', start_date )
			throw_if( 'offset_mark', offset_mark )

			
			self.mode = 'collection'
			self.collection = collection
			self.start_date = start_date
			self.page_size = self.validate_page_size( page_size )
			self.offset_mark = offset_mark
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/collections/{self.collection}/{self.start_date}'
			self.params = { 'pageSize': self.page_size, 'offsetMark': self.offset_mark,
					'api_key': self.api_key }
			self.payload = { }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'payload': self.payload, 'data': self.response.json( ) }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = "fetch_collection( self, collection: str, start_date: str, page_size: int=10, offset_mark: str='*', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='search', query: str='', page_size: int=10, offset_mark: str = '*', sort_field: str='score', sort_order: str='DESC', package_id: str='', collection: str = '', start_date: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a GovInfo retrieval operation.

		Purpose:
		    Select search, package_summary, or collection and delegate with applicable arguments.

		Args:
		    mode (str): search, package_summary, or collection. Defaults to 'search'.
		    query (str): Search expression. Defaults to an empty string; required in search mode.
		    page_size (int): Requested page size from 1 through 1000. Defaults to 10.
		    offset_mark (str): Nonblank pagination cursor. Defaults to '*'.
		    sort_field (str): score or lastModified. Defaults to 'score'.
		    sort_order (str): ASC or DESC. Defaults to 'DESC'.
		    package_id (str): Package identifier. Defaults to an empty string; required in
		        package_summary mode.
		    collection (str): Collection code. Defaults to an empty string; required in collection
		        mode.
		    start_date (str): Collection start-date path value. Defaults to an empty string;
		        required in collection mode.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, query parameters (including the API key), request payload,
		        and decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			
			if self.mode == 'search':
				return self.fetch_search( query=query, page_size=page_size,
					offset_mark=offset_mark, sort_field=sort_field, sort_order=sort_order,
					time=time )
			
			if self.mode == 'package_summary':
				return self.fetch_package_summary( package_id=package_id, time=time )
			
			if self.mode == 'collection':
				return self.fetch_collection( collection=collection, start_date=start_date,
					page_size=page_size, offset_mark=offset_mark, time=time )
			
			raise ValueError( "Unsupported GovInfo mode. Use 'search', 'package_summary', or "
			                  "'collection'." )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = "fetch( self, mode: str='search', query: str='', page_size: int=10, offset_mark: str = '*', sort_field: str='score', sort_order: str='DESC', package_id: str='', collection: str = '', start_date: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GovData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class StarChart( Fetcher ):
	"""Sky-Map object search and chart URL adapter.

	Purpose:
	    Resolve named astronomical objects through the XML search endpoint and construct
	    interactive or static chart URLs with configurable imagery and overlays.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    search_url (Optional[str]): Sky-Map XML object-search endpoint.
	    link_url (Optional[str]): Interactive chart base URL.
	    image_url (Optional[str]): Static chart image endpoint.
	    url (Optional[str]): Latest endpoint URL.
	    params (Optional[Dict[str, Any]]): Latest provider request parameters.
	    mode (Optional[str]): Selected retrieval operation.
	    query (Optional[str]): Latest search query.
	    ra (Optional[float]): Right ascension supplied or returned by Sky-Map.
	    dec (Optional[float]): Declination supplied or returned by Sky-Map.
	    zoom (Optional[int]): Requested map zoom level.
	    image_source (Optional[str]): Requested sky imagery source.
	    box_color (Optional[str]): Marker box color.
	    show_box (Optional[bool]): Marker box visibility flag.
	    show_grid (Optional[bool]): Coordinate grid visibility flag.
	    show_lines (Optional[bool]): Constellation line visibility flag.
	    show_boundaries (Optional[bool]): Constellation boundary visibility flag.
	    show_const_names (Optional[bool]): Constellation name visibility flag.
	    width (Optional[int]): Requested static image width in pixels.
	    height (Optional[int]): Requested static image height in pixels.
	    magnitude (Optional[float]): Requested stellar magnitude threshold.
	    agents (Optional[str]): Configured HTTP user-agent.
	    timeout (Optional[int]): Request timeout in seconds.
	    response (Optional[Response]): Latest HTTP response.
	"""
	search_url: Optional[ str ]
	link_url: Optional[ str ]
	image_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	ra: Optional[ float ]
	dec: Optional[ float ]
	zoom: Optional[ int ]
	image_source: Optional[ str ]
	box_color: Optional[ str ]
	show_box: Optional[ bool ]
	show_grid: Optional[ bool ]
	show_lines: Optional[ bool ]
	show_boundaries: Optional[ bool ]
	show_const_names: Optional[ bool ]
	width: Optional[ int ]
	height: Optional[ int ]
	magnitude: Optional[ float ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure XML search, interactive chart and static image endpoints, DSS2 imagery,
		    display defaults, and 900 by 450 static image dimensions.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.search_url = 'https://server1.sky-map.org/search'
		self.link_url = 'https://www.sky-map.org/'
		self.image_url = 'https://server2.sky-map.org/map'
		self.url = None
		self.params = { }
		self.mode = 'object_chart'
		self.query = ''
		self.ra = 0.0
		self.dec = 0.0
		self.zoom = 5
		self.image_source = 'DSS2'
		self.box_color = 'yellow'
		self.show_box = True
		self.show_grid = True
		self.show_lines = True
		self.show_boundaries = True
		self.show_const_names = False
		self.width = 900
		self.height = 450
		self.magnitude = 7.5
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'search_url', 'link_url', 'image_url', 'url', 'params', 'mode', 'query', 'ra',
				'dec', 'zoom', 'image_source', 'box_color', 'show_box', 'show_grid', 'show_lines',
				'show_boundaries', 'show_const_names', 'width', 'height', 'magnitude',
				'search_object', 'fetch_object_chart', 'fetch_coordinate_chart',
				'fetch_static_chart', 'fetch', 'create_schema' ]
	
	def flag( self, value: bool, invert: bool=False ) -> int:
		"""Encode a Sky-Map display flag.

		Purpose:
		    Return an integer flag based on the value's truthiness, reversing the mapping when
		    invert is enabled.

		Args:
		    value (bool): Display flag.
		    invert (bool): Reverse the numeric flag mapping. Defaults to False.

		Returns:
		    int: 1 for truthy values and 0 for falsey values, reversed when invert is enabled.

		Raises:
		    Error: Raised when validation or flag conversion fails.
		"""
		try:

			throw_if( 'value', value )

			throw_if( 'invert', invert )

			if invert:
				return 0 if value else 1
			
			return 1 if value else 0
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = 'flag( self, value: bool, invert: bool=False ) -> int'
			Logger( ).write( exception )
			raise exception

	def search_object( self, name: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search for an astronomical object.

		Purpose:
		    Request the XML search endpoint, parse status and star metadata, and retain returned
		    coordinates when a star element exists.

		Args:
		    name (str): Nonblank astronomical object name.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Search mode, endpoint, parameters, status, verbiage, and object metadata
		        or an empty data dictionary.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:


			throw_if( 'name', name )

			throw_if( 'time', time )

			self.query = name
			self.timeout = time
			self.url = self.search_url
			self.params = { 'star': self.query }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			root = ET.fromstring( self.response.text )
			status = root.findtext( 'status', default='' )
			verbiage = root.findtext( 'verbiage', default='' )
			star = root.find( 'star' )
			
			if star is None:
				return { 'mode': 'object_search', 'url': self.url, 'params': self.params,
						'status': status, 'verbiage': verbiage, 'data': { } }
			
			result = { 'id': star.attrib.get( 'id', '' ),
					'catalog_id': star.findtext( 'catId', default='' ),
					'constellation': star.findtext( 'constellation', default='' ),
					'ra': float( star.findtext( 'ra', default='0' ) ),
					'dec': float( star.findtext( 'de', default='0' ) ),
					'magnitude': star.findtext( 'mag', default='' ) }
			
			self.ra = result[ 'ra' ]
			self.dec = result[ 'dec' ]
			
			return { 'mode': 'object_search', 'url': self.url, 'params': self.params,
					'status': status, 'verbiage': verbiage, 'data': result }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = 'search_object( self, name: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_object_chart( self, name: str, zoom: int=5, box_color: str='yellow', show_box: bool = True, image_source: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Construct an interactive object chart and search for the object.

		Purpose:
		    Prepare the chart URL without fetching the chart page, then execute the XML object
		    search. The search updates instance URL and parameters before packaging the result.

		Args:
		    name (str): Nonblank astronomical object name.
		    zoom (int): Map zoom level. Defaults to 5.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    image_source (str): Optional imagery source identifier; omitted from the query when
		        blank. Defaults to an empty string.
		    time (int): Request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, latest search URL and parameters, prepared chart URL, and object
		        search result.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'show_box', show_box )

			throw_if( 'name', name )
			self.mode = 'object_chart'
			self.query = name
			self.zoom = zoom
			self.box_color = box_color
			self.show_box = show_box
			self.image_source = image_source
			
			self.url = self.link_url
			self.params = { 'object': self.query, 'zoom': self.zoom,
					'show_box': self.flag( self.show_box ), 'box_color': self.box_color }
			
			if self.image_source:
				self.params[ 'img_source' ] = self.image_source
			
			link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			search = self.search_object( name=self.query, time=time ) or { }
			
			return { 'mode': self.mode, 'url': self.url, 'params': self.params, 'chart_url': link,
					'search': search }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = "fetch_object_chart( self, name: str, zoom: int=5, box_color: str='yellow', show_box: bool = True, image_source: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_coordinate_chart( self, ra: float, dec: float, zoom: int=5, box_color: str = 'yellow', show_box: bool=True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, image_source: str='' ) -> Dict[ str, Any ] | None:
		"""Construct an interactive coordinate chart URL.

		Purpose:
		    Prepare a coordinate chart URL with marker and overlay settings; grid, line and boundary
		    flags use inverted numeric encoding. No HTTP request is issued.

		Args:
		    ra (float): Right ascension value passed to Sky-Map.
		    dec (float): Declination value passed to Sky-Map.
		    zoom (int): Map zoom level. Defaults to 5.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    image_source (str): Optional imagery source identifier; omitted from the query when
		        blank. Defaults to an empty string.

		Returns:
		    Dict[str, Any]: Mode, base URL, chart parameters, and prepared interactive chart URL.

		Raises:
		    Error: Raised when validation or URL construction fails.
		"""
		try:
			throw_if( 'show_box', show_box )

			throw_if( 'show_grid', show_grid )

			throw_if( 'show_lines', show_lines )

			throw_if( 'show_boundaries', show_boundaries )

			self.mode = 'coordinate_chart'
			self.ra = ra
			self.dec = dec
			self.zoom = zoom
			self.box_color = box_color
			self.show_box = show_box
			self.show_grid = show_grid
			self.show_lines = show_lines
			self.show_boundaries = show_boundaries
			self.image_source = image_source
			
			self.url = self.link_url
			self.params = { 'ra': self.ra, 'de': self.dec, 'zoom': self.zoom,
					'show_box': self.flag( self.show_box ), 'box_color': self.box_color,
					'show_grid': self.flag( self.show_grid, invert=True ),
					'show_constellation_lines': self.flag( self.show_lines, invert=True ),
					'show_constellation_boundaries': self.flag( self.show_boundaries, invert=True
					) }
			
			if self.image_source:
				self.params[ 'img_source' ] = self.image_source
			
			link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			return { 'mode': self.mode, 'url': self.url, 'params': self.params, 'chart_url': link }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = "fetch_coordinate_chart( self, ra: float, dec: float, zoom: int=5, box_color: str = 'yellow', show_box: bool=True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, image_source: str='' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_static_chart( self, ra: float, dec: float, zoom: int=5, image_source: str = 'DSS2', show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, width: int=900, height: int=450, magnitude: float = 7.5 ) -> Dict[ str, Any ] | None:
		"""Construct a static chart image URL.

		Purpose:
		    Prepare a static image URL with coordinates, dimensions, magnitude threshold, imagery
		    source, and overlays without downloading the image.

		Args:
		    ra (float): Right ascension value passed to Sky-Map.
		    dec (float): Declination value passed to Sky-Map.
		    zoom (int): Map zoom level. Defaults to 5.
		    image_source (str): Sky imagery source identifier. Defaults to 'DSS2'.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    show_const_names (bool): Display constellation names. Defaults to False.
		    width (int): Requested static image width in pixels. Defaults to 900.
		    height (int): Requested static image height in pixels. Defaults to 450.
		    magnitude (float): Requested stellar magnitude threshold. Defaults to 7.5.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, image parameters, and prepared image URL.

		Raises:
		    Error: Raised when validation or URL construction fails.
		"""
		try:
			throw_if( 'show_grid', show_grid )

			throw_if( 'show_lines', show_lines )

			throw_if( 'show_boundaries', show_boundaries )

			throw_if( 'show_const_names', show_const_names )

			self.mode = 'static_chart'
			self.ra = ra
			self.dec = dec
			self.zoom = zoom
			self.image_source = image_source
			self.show_grid = show_grid
			self.show_lines = show_lines
			self.show_boundaries = show_boundaries
			self.show_const_names = show_const_names
			self.width = width
			self.height = height
			self.magnitude = magnitude
			
			self.url = self.image_url
			self.params = { 'type': 'FULL', 'w': self.width, 'h': self.height, 'ra': self.ra,
					'de': self.dec, 'zoom': self.zoom, 'mag': self.magnitude,
					'show_grid': self.flag( self.show_grid ), 'grid_color': '404040',
					'grid_color_zero': '808080',
					'show_constellation_lines': self.flag( self.show_lines ),
					'show_constellation_boundaries': self.flag( self.show_boundaries ),
					'show_const_names': self.flag( self.show_const_names ),
					'img_source': self.image_source }
			
			image_link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			return { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'image_url': image_link }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = "fetch_static_chart( self, ra: float, dec: float, zoom: int=5, image_source: str = 'DSS2', show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, width: int=900, height: int=450, magnitude: float = 7.5 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='object_chart', query: str='', ra: float=0.0, dec: float = 0.0, zoom: int=5, image_source: str='DSS2', box_color: str='yellow', show_box: bool = True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, width: int=900, height: int=450, magnitude: float = 7.5, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a Sky-Map chart operation.

		Purpose:
		    Normalize mode and select XML object search, object chart, coordinate chart, or static
		    chart. The default DSS2 source is omitted from interactive chart requests.

		Args:
		    mode (str): object_search, object_chart, coordinate_chart, or static_chart. Defaults to
		        'object_chart'; blank selects the same mode.
		    query (str): Object name for object modes. Defaults to an empty string.
		    ra (float): Right ascension value passed to Sky-Map. Defaults to 0.0.
		    dec (float): Declination value passed to Sky-Map. Defaults to 0.0.
		    zoom (int): Map zoom level. Defaults to 5.
		    image_source (str): Sky imagery source identifier. Defaults to 'DSS2'.
		    box_color (str): Marker box color. Defaults to 'yellow'.
		    show_box (bool): Display the marker box. Defaults to True.
		    show_grid (bool): Display the coordinate grid. Defaults to True.
		    show_lines (bool): Display constellation lines. Defaults to True.
		    show_boundaries (bool): Display constellation boundaries. Defaults to True.
		    show_const_names (bool): Display constellation names. Defaults to False.
		    width (int): Requested static image width in pixels. Defaults to 900.
		    height (int): Requested static image height in pixels. Defaults to 450.
		    magnitude (float): Requested stellar magnitude threshold. Defaults to 7.5.
		    time (int): HTTP timeout for object search or object chart mode. Defaults to 20.

		Returns:
		    Dict[str, Any]: Search metadata or prepared chart URL and parameters for the selected
		        mode.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			active_mode = (mode or 'object_chart').strip( ).lower( )
			if active_mode == 'object_search':
				return self.search_object( name=query, time=time )
			if active_mode == 'object_chart':
				return self.fetch_object_chart( name=query, zoom=zoom, box_color=box_color,
					show_box=show_box, image_source=image_source if image_source != 'DSS2' else
					'', time=time )
			if active_mode == 'coordinate_chart':
				return self.fetch_coordinate_chart( ra=ra, dec=dec, zoom=zoom,
					box_color=box_color, show_box=show_box, show_grid=show_grid,
					show_lines=show_lines, show_boundaries=show_boundaries,
					image_source=image_source if image_source != 'DSS2' else '' )
			
			if active_mode == 'static_chart':
				return self.fetch_static_chart( ra=ra, dec=dec, zoom=zoom,
					image_source=image_source, show_grid=show_grid, show_lines=show_lines,
					show_boundaries=show_boundaries, show_const_names=show_const_names,
					width=width, height=height, magnitude=magnitude )
			
			raise ValueError( "Unsupported mode. Use 'object_search', 'object_chart', "
			                  "'coordinate_chart', or 'static_chart'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = "fetch( self, mode: str='object_chart', query: str='', ra: float=0.0, dec: float = 0.0, zoom: int=5, image_source: str='DSS2', box_color: str='yellow', show_box: bool = True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True, show_const_names: bool=False, width: int=900, height: int=450, magnitude: float = 7.5, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} '
					               f'service.',
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class Congress( Fetcher ):
	"""Congress.gov legislative data adapter.

	Purpose:
	    Retrieve Congress records, bills, laws, and committee reports through the Congress.gov v3
	    API, with pagination, type filters, and individual record identifiers.

	Attributes:
	    api_key (Optional[str]): Congress.gov API key loaded from configuration.
	    base_url (Optional[str]): Base provider API URL.
	    url (Optional[str]): Latest endpoint URL or client operation identifier.
	    params (Optional[Dict[str, Any]]): Latest request parameters, including the API key.
	    payload (Optional[Dict[str, Any]]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    mode (Optional[str]): Selected retrieval operation.
	    congress_number (Optional[int]): Selected Congress number.
	    bill_type (Optional[str]): Selected bill or resolution type.
	    bill_number (Optional[int]): Selected bill or resolution number.
	    law_type (Optional[str]): Selected public or private law type.
	    law_number (Optional[int]): Selected law number.
	    report_type (Optional[str]): Selected committee report type.
	    report_number (Optional[int]): Selected committee report number.
	    offset (Optional[int]): Requested nonnegative pagination offset.
	    limit (Optional[int]): Requested page size.
	    sort (Optional[str]): Provider sort expression.
	    from_date_time (Optional[str]): Lower bill update-date-time filter.
	    to_date_time (Optional[str]): Upper bill update-date-time filter.
	    conference (Optional[bool]): Conference-report filter.
	    response (Optional[Response]): Latest HTTP response.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	congress_number: Optional[ int ]
	bill_type: Optional[ str ]
	bill_number: Optional[ int ]
	law_type: Optional[ str ]
	law_number: Optional[ int ]
	report_type: Optional[ str ]
	report_number: Optional[ int ]
	offset: Optional[ int ]
	limit: Optional[ int ]
	sort: Optional[ str ]
	from_date_time: Optional[ str ]
	to_date_time: Optional[ str ]
	conference: Optional[ bool ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured Congress.gov API key and user-agent; initialize the v3 endpoint,
		    JSON headers, congresses mode, and pagination and sorting defaults.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.CONGRESS_API_KEY
		self.base_url = 'https://api.congress.gov/v3'
		self.url = None
		self.params = { }
		self.payload = { }
		self.result = { }
		self.mode = 'congresses'
		self.congress_number = None
		self.bill_type = ''
		self.bill_number = None
		self.law_type = ''
		self.law_number = None
		self.report_type = ''
		self.report_number = None
		self.offset = 0
		self.limit = 20
		self.sort = 'updateDate+desc'
		self.from_date_time = ''
		self.to_date_time = ''
		self.conference = False
		self.response = None
		self.headers = { 'Accept': 'application/json', 'User-Agent': cfg.AGENTS }
		self.agents = cfg.AGENTS
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'url', 'params', 'payload', 'result', 'mode',
				'congress_number', 'bill_type', 'bill_number', 'law_type', 'law_number',
				'report_type', 'report_number', 'offset', 'limit', 'sort', 'from_date_time',
				'to_date_time', 'conference', 'response', 'headers', 'agents', 'validate_limit',
				'validate_offset', 'normalize_bill_type', 'normalize_law_type',
				'normalize_report_type', 'build_params', 'request', 'fetch_congresses',
				'fetch_bills', 'fetch_bill', 'fetch_laws', 'fetch_law', 'fetch_reports',
				'fetch_report', 'fetch', 'create_schema' ]
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the requested record limit.

		Purpose:
		    Enforce the inclusive range from 1 through 250.

		Args:
		    limit (int): Requested record limit from 1 through 250.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the limit is missing or outside the range.
		"""
		try:
			

			throw_if( 'limit', limit )

			if limit < 1 or limit > 250:
				raise ValueError( 'limit must be between 1 and 250.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the pagination offset.

		Purpose:
		    Reject missing or negative offsets.

		Args:
		    offset (int): Nonnegative result offset.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the offset is missing or negative.
		"""
		try:
			

			throw_if( 'offset', offset )

			if offset < 0:
				raise ValueError( 'offset must be greater than or equal to 0.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def normalize_bill_type( self, bill_type: str ) -> str:
		"""Validate a bill type.

		Purpose:
		    Accept only supported lowercase bill and resolution type codes without changing their
		    values.

		Args:
		    bill_type (str): Bill type: hr, s, hjres, sjres, hconres, sconres, hres, or sres.

		Returns:
		    str: Original accepted bill type.

		Raises:
		    Error: Raised when the type is missing or unsupported.
		"""
		try:
			throw_if( 'bill_type', bill_type )
			value = bill_type
			
			allowed = { 'hr', 's', 'hjres', 'sjres', 'hconres', 'sconres', 'hres', 'sres' }
			
			if value not in allowed:
				raise ValueError( "Unsupported bill_type. Use 'hr', 's', 'hjres', 'sjres', "
				                  "'hconres', 'sconres', 'hres', or 'sres'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'normalize_bill_type( self, bill_type: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_law_type( self, law_type: str ) -> str:
		"""Validate a law type.

		Purpose:
		    Accept only pub or priv without changing the supplied value.

		Args:
		    law_type (str): Law type: pub or priv.

		Returns:
		    str: Original accepted law type.

		Raises:
		    Error: Raised when the type is missing or unsupported.
		"""
		try:
			throw_if( 'law_type', law_type )
			value = law_type
			
			allowed = { 'pub', 'priv' }
			
			if value not in allowed:
				raise ValueError( "Unsupported law_type. Use 'pub' or 'priv'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'normalize_law_type( self, law_type: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_report_type( self, report_type: str ) -> str:
		"""Validate a committee report type.

		Purpose:
		    Accept only hrpt, srpt, or erpt without changing the supplied value.

		Args:
		    report_type (str): Report type: hrpt, srpt, or erpt.

		Returns:
		    str: Original accepted report type.

		Raises:
		    Error: Raised when the type is missing or unsupported.
		"""
		try:
			throw_if( 'report_type', report_type )
			value = report_type
			
			allowed = { 'hrpt', 'srpt', 'erpt' }
			
			if value not in allowed:
				raise ValueError( "Unsupported report_type. Use 'hrpt', 'srpt', or 'erpt'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'normalize_report_type( self, report_type: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_params( self, limit: int=20, offset: int=0, sort: str='updateDate+desc' ) -> Dict[ str, Any ]:
		"""Construct common list-request parameters.

		Purpose:
		    Validate and retain paging and sorting values, then return the API key, JSON format,
		    limit, offset, and sort parameters.

		Args:
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    sort (str): Nonblank provider sort expression. Defaults to 'updateDate+desc'.

		Returns:
		    Dict[str, Any]: Common request parameters, including the API key.

		Raises:
		    Error: Raised when pagination or sort validation fails.
		"""
		try:
			
			self.limit = self.validate_limit( limit )
			self.offset = self.validate_offset( offset )
			throw_if( 'sort', sort )
			self.sort = sort
			
			return { 'api_key': self.api_key, 'format': 'json', 'limit': self.limit,
					'offset': self.offset, 'sort': self.sort }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = "build_params( self, limit: int=20, offset: int=0, sort: str='updateDate+desc' ) -> Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception
	
	def request( self, mode: str, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute a Congress.gov request.

		Purpose:
		    Retain the operation, URL, timeout and parameters; send a GET request, check HTTP
		    status, and package the decoded JSON response.

		Args:
		    mode (str): Nonblank operation label.
		    url (str): Nonblank endpoint URL.
		    params (Dict[str, Any]): Nonempty request parameter dictionary.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			throw_if( 'url', url )
			throw_if( 'params', params )

			
			self.mode = mode
			self.url = url
			throw_if( 'time', time )
			self.timeout = time
			self.params = params
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'request( self, mode: str, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_congresses( self, limit: int=20, offset: int=0, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve Congress records.

		Purpose:
		    Build the congress endpoint with common list parameters and descending update-date
		    sorting.

		Args:
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			self.url = f'{self.base_url}/congress'
			self.params = self.build_params( limit=limit, offset=offset, sort='updateDate+desc' )
			
			return self.request( mode='congresses', url=self.url, params=self.params, time=time )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'fetch_congresses( self, limit: int=20, offset: int=0, time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_bills( self, congress: int, bill_type: str='', offset: int=0, limit: int=20, sort: str='updateDate+desc', from_date_time: str='', to_date_time: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve bills for a Congress.

		Purpose:
		    Build the Congress-specific bill endpoint, optionally restrict bill type, and attach
		    paging, sort, and nonblank update-date-time filters.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    bill_type (str): Bill type: hr, s, hjres, sjres, hconres, sconres, hres, or sres.
		        Defaults to an empty string for all types.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    sort (str): Nonblank provider sort expression. Defaults to 'updateDate+desc'.
		    from_date_time (str): Optional lower update-date-time filter passed to the provider.
		        Defaults to an empty string.
		    to_date_time (str): Optional upper update-date-time filter passed to the provider.
		        Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'congress', congress )
			
			self.congress_number = congress
			self.bill_type = bill_type
			self.from_date_time = from_date_time
			self.to_date_time = to_date_time
			
			if self.bill_type:
				self.bill_type = self.normalize_bill_type( self.bill_type )
				self.url = f'{self.base_url}/bill/{self.congress_number}/{self.bill_type}'
			else:
				self.url = f'{self.base_url}/bill/{self.congress_number}'
			
			self.params = self.build_params( limit=limit, offset=offset, sort=sort )
			
			if self.from_date_time:
				self.params[ 'fromDateTime' ] = self.from_date_time
			
			if self.to_date_time:
				self.params[ 'toDateTime' ] = self.to_date_time
			
			return self.request( mode='bills', url=self.url, params=self.params, time=time )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = "fetch_bills( self, congress: int, bill_type: str='', offset: int=0, limit: int=20, sort: str='updateDate+desc', from_date_time: str='', to_date_time: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_bill( self, congress: int, bill_type: str, bill_number: int, time: int=20 ) -> ( Dict[ str, Any ] | None):
		"""Retrieve an individual bill.

		Purpose:
		    Validate the bill type and construct the endpoint for the supplied Congress and bill
		    number.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    bill_type (str): Bill type: hr, s, hjres, sjres, hconres, sconres, hres, or sres.
		    bill_number (int): Bill or resolution number.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'congress', congress )
			throw_if( 'bill_type', bill_type )
			throw_if( 'bill_number', bill_number )
			
			self.congress_number = congress
			self.bill_type = self.normalize_bill_type( bill_type )
			self.bill_number = bill_number
			self.url = (f'{self.base_url}/bill/{self.congress_number}/'
			            f'{self.bill_type}/{self.bill_number}')
			self.params = { 'api_key': self.api_key, 'format': 'json' }
			
			return self.request( mode='bill_detail', url=self.url, params=self.params, time=time )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'fetch_bill( self, congress: int, bill_type: str, bill_number: int, time: int=20 ) -> ( Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_laws( self, congress: int, law_type: str='', offset: int=0, limit: int=20, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve laws for a Congress.

		Purpose:
		    Build the law collection endpoint, optionally restrict law type, and attach common list
		    parameters.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    law_type (str): Law type: pub or priv. Defaults to an empty string for all types.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'congress', congress )
			
			self.congress_number = congress
			self.law_type = law_type
			
			if self.law_type:
				self.law_type = self.normalize_law_type( self.law_type )
				self.url = f'{self.base_url}/law/{self.congress_number}/{self.law_type}'
			else:
				self.url = f'{self.base_url}/law/{self.congress_number}'
			
			self.params = self.build_params( limit=limit, offset=offset, sort='updateDate+desc' )
			
			return self.request( mode='laws', url=self.url, params=self.params, time=time )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = "fetch_laws( self, congress: int, law_type: str='', offset: int=0, limit: int=20, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_law( self, congress: int, law_type: str, law_number: int, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve an individual law.

		Purpose:
		    Validate the law type and construct the endpoint for the supplied Congress and law
		    number.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    law_type (str): Law type: pub or priv.
		    law_number (int): Law number.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'congress', congress )
			throw_if( 'law_type', law_type )
			throw_if( 'law_number', law_number )
			
			self.congress_number = congress
			self.law_type = self.normalize_law_type( law_type )
			self.law_number = law_number
			self.url = (f'{self.base_url}/law/{self.congress_number}/'
			            f'{self.law_type}/{self.law_number}')
			self.params = { 'api_key': self.api_key, 'format': 'json' }
			
			return self.request( mode='law_detail', url=self.url, params=self.params, time=time )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'fetch_law( self, congress: int, law_type: str, law_number: int, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_reports( self, congress: int, report_type: str='', offset: int=0, limit: int = 20, conference: bool=False, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve committee reports for a Congress.

		Purpose:
		    Build the committee-report collection endpoint, optionally restrict report type, and
		    attach paging and conference filters.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    report_type (str): Report type: hrpt, srpt, or erpt. Defaults to an empty string for all
		        types.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    conference (bool): Conference-report filter. Defaults to False.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'conference', conference )

			throw_if( 'congress', congress )
			
			self.congress_number = congress
			self.report_type = report_type
			self.conference = conference
			
			if self.report_type:
				self.report_type = self.normalize_report_type( self.report_type )
				self.url = (f'{self.base_url}/committee-report/'
				            f'{self.congress_number}/{self.report_type}')
			else:
				self.url = f'{self.base_url}/committee-report/{self.congress_number}'
			
			self.params = self.build_params( limit=limit, offset=offset, sort='updateDate+desc' )
			self.params[ 'conference' ] = str( self.conference ).lower( )
			
			return self.request( mode='reports', url=self.url, params=self.params, time=time )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = "fetch_reports( self, congress: int, report_type: str='', offset: int=0, limit: int = 20, conference: bool=False, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_report( self, congress: int, report_type: str, report_number: int, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve an individual committee report.

		Purpose:
		    Validate report type and construct the endpoint for the supplied Congress and report
		    number.

		Args:
		    congress (int): Congress number used in the endpoint path.
		    report_type (str): Report type: hrpt, srpt, or erpt.
		    report_number (int): Committee report number.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'congress', congress )
			throw_if( 'report_type', report_type )
			throw_if( 'report_number', report_number )
			
			self.congress_number = congress
			self.report_type = self.normalize_report_type( report_type )
			self.report_number = report_number
			self.url = (f'{self.base_url}/committee-report/{self.congress_number}/'
			            f'{self.report_type}/{self.report_number}')
			self.params = { 'api_key': self.api_key, 'format': 'json' }
			
			return self.request( mode='report_detail', url=self.url, params=self.params,
				time=time )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'fetch_report( self, congress: int, report_type: str, report_number: int, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='congresses', congress: int=0, bill_type: str='', bill_number: int=0, law_type: str='', law_number: int=0, report_type: str='', report_number: int=0, offset: int=0, limit: int=20, sort: str = 'updateDate+desc', from_date_time: str='', to_date_time: str='', conference: bool = False, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a legislative retrieval operation.

		Purpose:
		    Select the requested list or individual-record method and forward its applicable
		    identifiers and filters.

		Args:
		    mode (str): congresses, bills, bill_detail, laws, law_detail, reports, or report_detail.
		        Defaults to 'congresses'.
		    congress (int): Congress number used in the endpoint path. Defaults to 0; supply a valid
		        Congress for legislative record modes.
		    bill_type (str): Bill type: hr, s, hjres, sjres, hconres, sconres, hres, or sres.
		        Defaults to an empty string.
		    bill_number (int): Individual bill number. Defaults to 0.
		    law_type (str): Law type: pub or priv. Defaults to an empty string.
		    law_number (int): Individual law number. Defaults to 0.
		    report_type (str): Report type: hrpt, srpt, or erpt. Defaults to an empty string.
		    report_number (int): Individual report number. Defaults to 0.
		    offset (int): Nonnegative result offset. Defaults to 0.
		    limit (int): Requested record count from 1 through 250. Defaults to 20.
		    sort (str): Nonblank provider sort expression. Defaults to 'updateDate+desc'.
		    from_date_time (str): Optional lower update-date-time filter passed to the provider.
		        Defaults to an empty string.
		    to_date_time (str): Optional upper update-date-time filter passed to the provider.
		        Defaults to an empty string.
		    conference (bool): Conference-report filter. Defaults to False.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters (including the API key), and
		        decoded provider data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			
			if self.mode == 'congresses':
				return self.fetch_congresses( limit=limit, offset=offset, time=time )
			
			if self.mode == 'bills':
				return self.fetch_bills( congress=congress, bill_type=bill_type, offset=offset,
					limit=limit, sort=sort, from_date_time=from_date_time,
					to_date_time=to_date_time, time=time )
			
			if self.mode == 'bill_detail':
				return self.fetch_bill( congress=congress, bill_type=bill_type,
					bill_number=bill_number, time=time )
			
			if self.mode == 'laws':
				return self.fetch_laws( congress=congress, law_type=law_type, offset=offset,
					limit=limit, time=time )
			
			if self.mode == 'law_detail':
				return self.fetch_law( congress=congress, law_type=law_type,
					law_number=law_number, time=time )
			
			if self.mode == 'reports':
				return self.fetch_reports( congress=congress, report_type=report_type,
					offset=offset, limit=limit, conference=conference, time=time )
			
			if self.mode == 'report_detail':
				return self.fetch_report( congress=congress, report_type=report_type,
					report_number=report_number, time=time )
			
			raise ValueError( "Unsupported Congress mode. Use 'congresses', 'bills', "
			                  "'bill_detail', 'laws', 'law_detail', 'reports', "
			                  "or 'report_detail'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = "fetch( self, mode: str='congresses', congress: int=0, bill_type: str='', bill_number: int=0, law_type: str='', law_number: int=0, report_type: str='', report_number: int=0, offset: int=0, limit: int=20, sort: str = 'updateDate+desc', from_date_time: str='', to_date_time: str='', conference: bool = False, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class InternetArchive( Fetcher ):
	"""Internet Archive advanced-search adapter.

	Purpose:
	    Construct search expressions with optional media-type and collection filters, request
	    selected metadata fields, and return decoded advanced-search results.

	Attributes:
	    url (Optional[str]): Internet Archive advanced-search endpoint.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    keywords (Optional[str]): Latest search expression.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    fields (Optional[List[str]]): Default metadata field list.
	    active_fields (List[str]): Metadata fields selected for the current request.
	    rows (Optional[int]): Requested records per page.
	    page (Optional[int]): One-based result page.
	    sort (Optional[str]): Provider sort expression.
	    media_type (Optional[str]): Optional media-type filter.
	    collection (Optional[str]): Optional collection filter.
	    response (Optional[Response]): Latest HTTP response.
	    agents (Optional[str]): Configured HTTP user-agent.
	"""
	keywords: Optional[ str ]
	url: Optional[ str ]
	response: Optional[ Response ]
	fields: Optional[ List[ str ] ]
	active_fields: List[ str ]
	rows: Optional[ int ]
	page: Optional[ int ]
	sort: Optional[ str ]
	media_type: Optional[ str ]
	collection: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the advanced-search endpoint, JSON headers, default metadata fields, ten rows
		    per page, first page, descending download sort, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.url = 'https://archive.org/advancedsearch.php'
		self.headers = { }
		self.timeout = 20
		self.keywords = ''
		self.params = { }
		self.fields = [ 'identifier', 'title', 'creator', 'mediatype', 'collection', 'publicdate',
				'description' ]
		self.active_fields = self.fields
		self.rows = 10
		self.page = 1
		self.sort = 'downloads desc'
		self.media_type = ''
		self.collection = ''
		self.response = None
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'keywords', 'url', 'timeout', 'headers', 'fields', 'rows', 'page', 'sort',
				'media_type', 'collection', 'params', 'agents', 'fetch', 'create_schema' ]
	
	def validate_rows( self, rows: int ) -> int:
		"""Validate records per page.

		Purpose:
		    Enforce the inclusive range from 1 through 100.

		Args:
		    rows (int): Requested rows from 1 through 100.

		Returns:
		    int: Original accepted row count.

		Raises:
		    Error: Raised when the row count is missing or outside the range.
		"""
		try:
			throw_if( 'rows', rows )
			if rows < 1 or rows > 100:
				raise ValueError( 'rows must be between 1 and 100.' )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
			exception.method = 'validate_rows( self, rows: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_page( self, page: int ) -> int:
		"""Validate the result page.

		Purpose:
		    Require a page number of at least one.

		Args:
		    page (int): One-based result page.

		Returns:
		    int: Original accepted page number.

		Raises:
		    Error: Raised when the page is missing or below one.
		"""
		try:

			throw_if( 'page', page )

			if page < 1:
				raise ValueError( 'page must be greater than or equal to 1.' )
			
			return page
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
			exception.method = 'validate_page( self, page: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def build_query( self, keywords: str, media_type: str='', collection: str='' ) -> str:
		"""Construct an advanced-search expression.

		Purpose:
		    Wrap the stripped keyword expression in parentheses and append nonblank media-type and
		    collection clauses using AND.

		Args:
		    keywords (str): Nonblank Internet Archive search expression.
		    media_type (str): Optional media-type filter. Defaults to an empty string.
		    collection (str): Optional collection filter. Defaults to an empty string.

		Returns:
		    str: Combined Internet Archive search expression.

		Raises:
		    Error: Raised when validation or query construction fails.
		"""
		try:
			throw_if( 'keywords', keywords )
			
			parts: List[ str ] = [ f'({keywords.strip( )})' ]
			
			if media_type and media_type.strip( ):
				parts.append( f'AND mediatype:({media_type.strip( )})' )
			
			if collection and collection.strip( ):
				parts.append( f'AND collection:({collection.strip( )})' )
			
			return ' '.join( parts )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
			exception.method = "build_query( self, keywords: str, media_type: str='', collection: str='' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, keywords: str, fields: List[ str ] | None = None, rows: int=10, page: int = 1, sort: str='downloads desc', media_type: str='', collection: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search Internet Archive metadata.

		Purpose:
		    Validate paging, select default or caller-supplied fields, construct the search
		    expression, request JSON, and return it with operation metadata.

		Args:
		    keywords (str): Nonblank Internet Archive search expression.
		    fields (List[str] | None): Nonempty list of metadata fields; defaults to None to use
		        instance defaults.
		    rows (int): Records per page from 1 through 100. Defaults to 10.
		    page (int): One-based page number of at least 1. Defaults to 1.
		    sort (str): Provider sort expression. Defaults to 'downloads desc'.
		    media_type (str): Optional media-type filter. Defaults to an empty string.
		    collection (str): Optional collection filter. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Advanced-search mode, endpoint URL, request parameters, and decoded JSON
		        data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'keywords', keywords )
			self.keywords = keywords
			

			throw_if( 'time', time )
			self.timeout = time
			self.rows = self.validate_rows( rows )
			self.page = self.validate_page( page )
			self.sort = sort
			self.media_type = media_type
			self.collection = collection
			
			if fields is None:
				self.active_fields = self.fields
			else:
				throw_if( 'fields', fields )
				self.active_fields = fields
			
			query_text = self.build_query( keywords=self.keywords, media_type=self.media_type,
				collection=self.collection )
			
			self.params = { 'q': query_text, 'fl[]': self.active_fields, 'rows': self.rows,
					'page': self.page, 'sort[]': self.sort, 'output': 'json' }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return { 'mode': 'advanced_search', 'url': self.url, 'params': self.params,
					'data': payload }
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
			exception.method = "fetch( self, keywords: str, fields: List[ str ] | None = None, rows: int=10, page: int = 1, sort: str='downloads desc', media_type: str='', collection: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ), 'description': (f'{description.strip( )} '
			                                                    f'This function uses the '
			                                                    f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class OpenWeather( Fetcher ):
	"""Open-Meteo current and forecast weather adapter.

	Purpose:
	    Resolve locations through Open-Meteo geocoding and retrieve current, hourly, or daily
	    weather variables using configured measurement units.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    location (Optional[str]): Latest location name supplied for geocoding.
	    latitude (Optional[float]): Current latitude in decimal degrees.
	    longitude (Optional[float]): Current longitude in decimal degrees.
	    timezone (Optional[str]): Resolved or explicitly supplied timezone.
	    mode (Optional[str]): Selected retrieval operation.
	    result_limit (Optional[int]): Requested number of geocoding candidates.
	    past_days (int): Requested preceding days.
	    forecast_days (int): Requested forecast days.
	    geocode_url (Optional[str]): Open-Meteo geocoding endpoint.
	    forecast_url (Optional[str]): Open-Meteo forecast endpoint.
	    temperature_unit (Optional[str]): Temperature unit, initialized to Fahrenheit.
	    windspeed_unit (Optional[str]): Wind speed unit, initialized to knots.
	    precipitation_unit (Optional[str]): Precipitation unit, initialized to inches.
	    geocode_params (Optional[Dict[str, Any]]): Latest geocoding query parameters.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    current_metrics (Optional[List[str]]): Configured current-weather variable identifiers.
	    hourly_metrics (Optional[List[str]]): Configured hourly weather variable identifiers.
	    daily_metrics (Optional[List[str]]): Configured daily weather variable identifiers.
	    url (Optional[str]): Latest endpoint URL or client operation identifier.
	    response (Optional[Response]): Latest HTTP response.
	"""
	geocode_url: Optional[ str ]
	forecast_url: Optional[ str ]
	location: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	mode: Optional[ str ]
	current_metrics: Optional[ List[ str ] ]
	hourly_metrics: Optional[ List[ str ] ]
	daily_metrics: Optional[ List[ str ] ]
	windspeed_unit: Optional[ str ]
	temperature_unit: Optional[ str ]
	precipitation_unit: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	geocode_params: Optional[ Dict[ str, Any ] ]
	result_limit: Optional[ int ]
	past_days: int
	forecast_days: int
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure Open-Meteo geocoding and forecast endpoints, weather variable lists,
		    Fahrenheit, knots, inches, automatic timezone, and default forecast settings.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.agents = cfg.AGENTS
		self.location = None
		self.latitude = None
		self.longitude = None
		self.timezone = 'auto'
		self.mode = 'current'
		self.result_limit = 10
		self.past_days = 0
		self.forecast_days = 7
		self.geocode_url = 'https://geocoding-api.open-meteo.com/v1/search'
		self.forecast_url = 'https://api.open-meteo.com/v1/forecast'
		self.temperature_unit = 'fahrenheit'
		self.windspeed_unit = 'kn'
		self.precipitation_unit = 'inch'
		self.geocode_params = None
		self.params = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		self.current_metrics = [ 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
				'is_day', 'precipitation', 'rain', 'showers', 'snowfall', 'weather_code',
				'cloud_cover', 'pressure_msl', 'surface_pressure', 'wind_speed_10m',
				'wind_direction_10m', 'wind_gusts_10m' ]
		
		self.hourly_metrics = [ 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
				'precipitation_probability', 'precipitation', 'rain', 'showers', 'snowfall',
				'weather_code', 'cloud_cover', 'pressure_msl', 'surface_pressure', 'visibility',
				'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m' ]
		
		self.daily_metrics = [ 'weather_code', 'temperature_2m_max', 'temperature_2m_min',
				'apparent_temperature_max', 'apparent_temperature_min', 'sunrise', 'sunset',
				'daylight_duration', 'sunshine_duration', 'precipitation_sum', 'rain_sum',
				'showers_sum', 'snowfall_sum', 'precipitation_hours',
				'precipitation_probability_max', 'wind_speed_10m_max', 'wind_gusts_10m_max',
				'wind_direction_10m_dominant' ]
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'geocode_url', 'forecast_url', 'location', 'latitude', 'longitude', 'timezone',
				'mode', 'current_metrics', 'hourly_metrics', 'daily_metrics', 'windspeed_unit',
				'temperature_unit', 'precipitation_unit', 'params', 'geocode_params',
				'result_limit', 'geocode_location', 'fetch', 'fetch_current', 'fetch_hourly',
				'fetch_daily', 'create_schema' ]
	
	def geocode_location( self, location: str, count: int=10 ) -> Dict[ str, Any ] | None:
		"""Resolve a location using Open-Meteo geocoding.

		Purpose:
		    Request location candidates with a fixed twenty-second timeout, validate the response
		    object and results list, retain coordinates and available timezone from the first
		    candidate.

		Args:
		    location (str): Nonblank location name to search.
		    count (int): Positive number of geocoding candidates requested. Defaults to 10.

		Returns:
		    Dict[str, Any] | None: First geocoding record, or None when no candidates are returned.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'location', location )
			self.location = location
			throw_if( 'count', count )
			self.result_limit = count
			if self.result_limit < 1:
				raise ValueError( 'count must be positive.' )
			self.geocode_params = { 'name': self.location, 'count': self.result_limit }
			
			self.url = self.geocode_url
			self.response = requests.get( self.url, params=self.geocode_params,
				headers=self.headers, timeout=20 )
			self.response.raise_for_status( )
			
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo geocoding response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo geocoding failed.' ) ) )
			if not isinstance( payload.get( 'results', [ ] ), list ):
				raise ValueError( 'Open-Meteo geocoding results must be a list.' )
			results = payload.get( 'results', [ ] ) or [ ]
			
			if not results:
				return None
			
			selected = results[ 0 ]
			self.latitude = selected.get( 'latitude', None )
			self.longitude = selected.get( 'longitude', None )
			
			resolved_timezone = selected.get( 'timezone', None )
			if resolved_timezone:
				self.timezone = resolved_timezone
			
			return selected
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'geocode_location( self, location: str, count: int=10 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_current( self, lat: float, long: float, zone: str='auto', past_days: int=0 ) -> Dict[ str, Any ] | None:
		"""Retrieve current weather by coordinates.

		Purpose:
		    Validate coordinates and timezone, submit current variables and preceding-day setting to
		    the forecast endpoint using a fixed thirty-second timeout, and package the response.

		Args:
		    lat (float): Finite latitude in decimal degrees, from -90 through 90.
		    long (float): Finite longitude in decimal degrees, from -180 through 180.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.
		    past_days (int): Requested number of preceding days. Defaults to 0.

		Returns:
		    Dict[str, Any]: Mode, location, latitude, longitude, timezone, endpoint URL, submitted
		        parameters, and decoded weather data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			self.mode = 'current'
			self.latitude = lat
			self.longitude = long
			if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0 or not math.isfinite( self.latitude ) or not math.isfinite( self.longitude ):
				raise ValueError( 'Invalid latitude or longitude.' )
			throw_if( 'zone', zone )
			self.timezone = zone
			throw_if( 'past_days', past_days )
			self.past_days = past_days
			
			self.params = { 'latitude': self.latitude, 'longitude': self.longitude,
					'current': self.current_metrics, 'timezone': self.timezone,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit, 'past_days': self.past_days }
			
			self.url = self.forecast_url
			self.response = requests.get( self.url, params=self.params, headers=self.headers,
				timeout=30 )
			self.response.raise_for_status( )
			
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo request failed.' ) ) )
			
			return { 'mode': self.mode, 'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ), 'url': self.url,
					'params': self.params, 'data': payload }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = "fetch_current( self, lat: float, long: float, zone: str='auto', past_days: int=0 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_hourly( self, lat: float, long: float, zone: str='auto', forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ] | None:
		"""Retrieve hourly weather by coordinates.

		Purpose:
		    Validate coordinates and timezone, submit configured hourly variables and requested
		    forecast and preceding days using a fixed thirty-second timeout.

		Args:
		    lat (float): Finite latitude in decimal degrees, from -90 through 90.
		    long (float): Finite longitude in decimal degrees, from -180 through 180.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.
		    forecast_days (int): Requested number of forecast days. Defaults to 7.
		    past_days (int): Requested number of preceding days. Defaults to 0.

		Returns:
		    Dict[str, Any]: Mode, location, latitude, longitude, timezone, endpoint URL, submitted
		        parameters, and decoded weather data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			self.mode = 'hourly'
			self.latitude = lat
			self.longitude = long
			if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0 or not math.isfinite( self.latitude ) or not math.isfinite( self.longitude ):
				raise ValueError( 'Invalid latitude or longitude.' )
			throw_if( 'zone', zone )
			self.timezone = zone
			throw_if( 'past_days', past_days )
			self.past_days = past_days
			throw_if( 'forecast_days', forecast_days )
			self.forecast_days = forecast_days
			self.params = { 'latitude': self.latitude, 'longitude': self.longitude,
					'hourly': self.hourly_metrics, 'timezone': self.timezone,
					'forecast_days': self.forecast_days, 'past_days': self.past_days,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit }
			
			self.url = self.forecast_url
			self.response = requests.get( self.url, params=self.params, headers=self.headers,
				timeout=30 )
			self.response.raise_for_status( )
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo request failed.' ) ) )
			
			return { 'mode': self.mode, 'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ), 'url': self.url,
					'params': self.params, 'data': payload }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = "fetch_hourly( self, lat: float, long: float, zone: str='auto', forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_daily( self, lat: float, long: float, zone: str='auto', forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ] | None:
		"""Retrieve daily weather by coordinates.

		Purpose:
		    Validate coordinates and timezone, submit configured daily variables and requested
		    forecast and preceding days using a fixed thirty-second timeout.

		Args:
		    lat (float): Finite latitude in decimal degrees, from -90 through 90.
		    long (float): Finite longitude in decimal degrees, from -180 through 180.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.
		    forecast_days (int): Requested number of forecast days. Defaults to 7.
		    past_days (int): Requested number of preceding days. Defaults to 0.

		Returns:
		    Dict[str, Any]: Mode, location, latitude, longitude, timezone, endpoint URL, submitted
		        parameters, and decoded weather data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			self.mode = 'daily'
			self.latitude = lat
			self.longitude = long
			if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0 or not math.isfinite( self.latitude ) or not math.isfinite( self.longitude ):
				raise ValueError( 'Invalid latitude or longitude.' )
			throw_if( 'zone', zone )
			self.timezone = zone
			throw_if( 'past_days', past_days )
			self.past_days = past_days
			throw_if( 'forecast_days', forecast_days )
			self.forecast_days = forecast_days
			self.params = { 'latitude': self.latitude, 'longitude': self.longitude,
					'daily': self.daily_metrics, 'timezone': self.timezone,
					'forecast_days': self.forecast_days, 'past_days': self.past_days,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit }
			
			self.url = self.forecast_url
			self.response = requests.get( self.url, params=self.params, headers=self.headers,
				timeout=30 )
			self.response.raise_for_status( )
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo request failed.' ) ) )
			return { 'mode': self.mode, 'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ), 'url': self.url,
					'params': self.params, 'data': payload }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = "fetch_daily( self, lat: float, long: float, zone: str='auto', forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, location: str, mode: str='current', zone: str='auto', forecast_days: int=7, past_days: int=0, count: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve weather for a named location.

		Purpose:
		    Geocode the location, return a diagnostic dictionary if no usable coordinates are
		    available, otherwise dispatch the selected weather mode and attach geocoding metadata.

		Args:
		    location (str): Nonblank location name to search.
		    mode (str): current, hourly, or daily. Defaults to 'current'.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.
		    forecast_days (int): Requested number of forecast days. Defaults to 7.
		    past_days (int): Requested number of preceding days. Defaults to 0.
		    count (int): Positive number of geocoding candidates requested. Defaults to 10.

		Returns:
		    Dict[str, Any]: Packaged weather response with geocoding metadata, or a diagnostic
		        message and empty data when geocoding fails to resolve coordinates.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'location', location )
			throw_if( 'mode', mode )
			selected = self.geocode_location( location=location, count=count )
			if not selected:
				return { 'location': location, 'mode': mode,
						'message': 'No geocoding results found.', 'data': { } }
			
			lat = selected.get( 'latitude', None )
			long = selected.get( 'longitude', None )
			if lat is None or long is None:
				return { 'location': location, 'mode': mode,
						'message': 'Geocoding returned no usable coordinates.',
						'geocoding': selected, 'data': { } }
			
			active_mode = mode
			if active_mode == 'current':
				result = self.fetch_current( lat=lat,
					long=long, zone=zone,
					past_days=past_days ) or { }
			
			elif active_mode == 'hourly':
				result = self.fetch_hourly( lat=lat, long=long, zone=zone, forecast_days=forecast_days, past_days=past_days ) or { }
			
			elif active_mode == 'daily':
				result = self.fetch_daily( lat=lat, long=long, zone=zone, forecast_days=forecast_days, past_days=past_days ) or { }
			
			else:
				raise ValueError( "Unsupported mode. Use 'current', 'hourly', or 'daily'." )
			
			result[ 'geocoding' ] = selected
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = "fetch( self, location: str, mode: str='current', zone: str='auto', forecast_days: int=7, past_days: int=0, count: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f"{description.strip( )} This function uses the {tool.strip( )} "
					               f"service.",
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class HistoricalWeather( Fetcher ):
	"""Open-Meteo historical weather adapter.

	Purpose:
	    Resolve a named location and retrieve daily and hourly historical weather variables for
	    one calendar date through the Open-Meteo archive endpoint.

	Attributes:
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    agents (Optional[str]): Configured HTTP user-agent.
	    location (Optional[str]): Latest location name supplied for geocoding.
	    latitude (Optional[float]): Current latitude in decimal degrees.
	    longitude (Optional[float]): Current longitude in decimal degrees.
	    timezone (Optional[str]): Resolved or explicitly supplied timezone.
	    target_date (Optional[dt.date]): Calendar date selected for historical retrieval.
	    result_limit (Optional[int]): Requested number of geocoding candidates.
	    geocode_url (Optional[str]): Open-Meteo geocoding endpoint.
	    archive_url (Optional[str]): Open-Meteo historical archive endpoint.
	    temperature_unit (Optional[str]): Temperature unit, initialized to Fahrenheit.
	    windspeed_unit (Optional[str]): Wind speed unit, initialized to knots.
	    precipitation_unit (Optional[str]): Precipitation unit, initialized to inches.
	    geocode_params (Optional[Dict[str, Any]]): Latest geocoding query parameters.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    daily_metrics (Optional[List[str]]): Configured daily weather variable identifiers.
	    hourly_metrics (Optional[List[str]]): Configured hourly weather variable identifiers.
	    url (Optional[str]): Latest endpoint URL or client operation identifier.
	    response (Optional[Response]): Latest HTTP response.
	"""
	geocode_url: Optional[ str ]
	archive_url: Optional[ str ]
	location: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	target_date: Optional[ dt.date ]
	daily_metrics: Optional[ List[ str ] ]
	hourly_metrics: Optional[ List[ str ] ]
	windspeed_unit: Optional[ str ]
	temperature_unit: Optional[ str ]
	precipitation_unit: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	geocode_params: Optional[ Dict[ str, Any ] ]
	result_limit: Optional[ int ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure Open-Meteo geocoding and archive endpoints, daily and hourly metric lists,
		    Fahrenheit, knots, inches, and automatic timezone.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.headers = { }
		self.agents = cfg.AGENTS
		self.location = None
		self.latitude = None
		self.longitude = None
		self.timezone = 'auto'
		self.target_date = None
		self.result_limit = 10
		self.geocode_url = 'https://geocoding-api.open-meteo.com/v1/search'
		self.archive_url = 'https://archive-api.open-meteo.com/v1/archive'
		self.temperature_unit = 'fahrenheit'
		self.windspeed_unit = 'kn'
		self.precipitation_unit = 'inch'
		self.geocode_params = None
		self.params = None
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		self.daily_metrics = [ 'weather_code', 'temperature_2m_max', 'temperature_2m_min',
				'apparent_temperature_max', 'apparent_temperature_min', 'sunrise', 'sunset',
				'daylight_duration', 'sunshine_duration', 'precipitation_sum', 'rain_sum',
				'showers_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
				'wind_gusts_10m_max', 'wind_direction_10m_dominant' ]
		
		self.hourly_metrics = [ 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
				'precipitation', 'rain', 'showers', 'snowfall', 'weather_code', 'cloud_cover',
				'pressure_msl', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
				'wind_gusts_10m' ]
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'geocode_url', 'archive_url', 'location', 'latitude', 'longitude', 'timezone',
				'target_date', 'daily_metrics', 'hourly_metrics', 'windspeed_unit',
				'temperature_unit', 'precipitation_unit', 'params', 'geocode_params',
				'result_limit', 'fetch', 'geocode_location', 'fetch_historical', 'create_schema' ]
	
	def geocode_location( self, location: str, count: int=10 ) -> Dict[ str, Any ] | None:
		"""Resolve a location using Open-Meteo geocoding.

		Purpose:
		    Request location candidates with a fixed twenty-second timeout, validate the response
		    object and results list, retain coordinates and available timezone from the first
		    candidate.

		Args:
		    location (str): Nonblank location name to search.
		    count (int): Positive number of geocoding candidates requested. Defaults to 10.

		Returns:
		    Dict[str, Any] | None: First geocoding record, or None when no candidates are returned.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'location', location )
			self.location = location
			self.result_limit = count
			if self.result_limit < 1:
				raise ValueError( 'count must be positive.' )
			self.geocode_params = { 'name': self.location, 'count': self.result_limit }
			self.url = self.geocode_url
			self.response = requests.get( self.url, params=self.geocode_params,
				headers=self.headers, timeout=20 )
			self.response.raise_for_status( )
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo geocoding response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo geocoding failed.' ) ) )
			if not isinstance( payload.get( 'results', [ ] ), list ):
				raise ValueError( 'Open-Meteo geocoding results must be a list.' )
			results = payload.get( 'results', [ ] ) or [ ]
			if not results:
				return None
			
			selected = results[ 0 ]
			self.latitude = selected.get( 'latitude', None )
			self.longitude = selected.get( 'longitude', None )
			resolved_timezone = selected.get( 'timezone', None )
			if resolved_timezone:
				self.timezone = resolved_timezone
			
			return selected
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = 'geocode_location( self, location: str, count: int=10 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_historical( self, lat: float, long: float, date: dt.date, zone: str='auto' ) -> Dict[ str, Any ] | None:
		"""Retrieve historical weather for one date.

		Purpose:
		    Validate geographic bounds and the datetime.date input, use its ISO representation as
		    both archive start and end date, and request daily and hourly metrics with a fixed
		    thirty-second timeout.

		Args:
		    lat (float): Finite latitude in decimal degrees, from -90 through 90.
		    long (float): Finite longitude in decimal degrees, from -180 through 180.
		    date (dt.date): Target calendar date.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.

		Returns:
		    Dict[str, Any]: Location, coordinates, timezone, target date, endpoint URL, submitted
		        parameters, and decoded archive data.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'date', date )
			self.latitude = lat
			self.longitude = long
			if (not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0 or not
			math.isfinite( self.latitude ) or not math.isfinite( self.longitude )):
				raise ValueError( 'Invalid latitude or longitude.' )
			if not isinstance( date, dt.date ):
				raise ValueError( 'date must be a datetime.date instance.' )
			self.target_date = date
			throw_if( 'zone', zone )
			self.timezone = zone
			self.params = { 'latitude': self.latitude, 'longitude': self.longitude,
					'start_date': self.target_date.isoformat( ),
					'end_date': self.target_date.isoformat( ), 'timezone': self.timezone,
					'daily': self.daily_metrics, 'hourly': self.hourly_metrics,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit }
			
			self.url = self.archive_url
			self.response = requests.get( self.url, params=self.params, headers=self.headers,
				timeout=30 )
			self.response.raise_for_status( )
			payload = self.response.json( )
			if not isinstance( payload, dict ):
				raise ValueError( 'Open-Meteo response must be a JSON object.' )
			if payload.get( 'error' ):
				raise ValueError( str( payload.get( 'reason', 'Open-Meteo request failed.' ) ) )
			return { 'location': self.location, 'latitude': self.latitude,
					'longitude': self.longitude,
					'timezone': payload.get( 'timezone', self.timezone ),
					'date': self.target_date.isoformat( ), 'url': self.url, 'params': self.params,
					'data': payload }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = "fetch_historical( self, lat: float, long: float, date: dt.date, zone: str='auto' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, location: str, date: dt.date, zone: str='auto', count: int=10 ) -> Dict[ str, Any ] | None:
		"""Retrieve historical weather for a named location.

		Purpose:
		    Resolve the location, return a diagnostic dictionary if no usable coordinates are
		    available, otherwise retrieve the requested date and attach geocoding metadata.

		Args:
		    location (str): Nonblank location name to search.
		    date (dt.date): Target calendar date.
		    zone (str): Nonblank provider timezone setting. Defaults to 'auto'.
		    count (int): Positive number of geocoding candidates requested. Defaults to 10.

		Returns:
		    Dict[str, Any]: Historical response with geocoding metadata, or a diagnostic message and
		        empty data when coordinates cannot be resolved.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'location', location )
			throw_if( 'date', date )
			selected = self.geocode_location( location=location, count=count )
			if not selected:
				return { 'location': location, 'date': date.isoformat( ),
						'message': 'No geocoding results found.', 'data': { } }
			
			lat = selected.get( 'latitude', None )
			long = selected.get( 'longitude', None )
			
			if lat is None or long is None:
				return { 'location': location, 'date': date.isoformat( ),
						'message': 'Geocoding returned no usable coordinates.',
						'geocoding': selected, 'data': { } }
			
			result = self.fetch_historical( lat=lat, long=long, date=date,
				zone=zone ) or { }
			
			result[ 'geocoding' ] = selected
			return result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = "fetch( self, location: str, date: dt.date, zone: str='auto', count: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': f"{description.strip( )} This function uses the {tool.strip( )} "
					               f"service.",
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class Grokipedia( Fetcher ):
	"""Grokipedia search and page retrieval adapter.

	Purpose:
	    Invoke GrokipediaClient search and get_page operations and package their responses with
	    request metadata and a configured-key indicator.

	Attributes:
	    api_key (Optional[str]): Configured xAI API key used only for the configuration indicator
	        in this wrapper.
	    url (Optional[str]): Symbolic Grokipedia client operation identifier.
	    client (Optional[GrokipediaClient]): GrokipediaClient instance created for the latest
	        operation.
	    query (Optional[str]): Latest search text.
	    page (Optional[str]): Latest requested page identifier.
	    response (Optional[Dict[str, Any]]): Latest raw GrokipediaClient response.
	    params (Optional[Dict[str, Any]]): Latest client operation arguments.
	    result (Optional[Dict[str, Any]]): Latest packaged client response.
	    limit (Optional[int]): Requested search result count.
	    offset (Optional[int]): Requested search offset.
	    include_content (Optional[bool]): Page content inclusion flag.
	    headers (Optional[Dict[str, Any]]): Reserved HTTP headers, initialized empty.
	    timeout (Optional[int]): Reserved timeout, initialized to 20; not forwarded to the
	        client.
	"""
	api_key: Optional[ str ]
	client: Optional[ GrokipediaClient ]
	query: Optional[ str ]
	page: Optional[ str ]
	limit: Optional[ int ]
	offset: Optional[ int ]
	include_content: Optional[ bool ]
	response: Optional[ Dict[ str, Any ] ]
	params: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured xAI key and initialize empty client and response state, search limit
		    of twelve, zero offset, and page-content inclusion.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.url = None
		self.client = None
		self.query = ''
		self.page = ''
		self.response = None
		self.params = { }
		self.result = { }
		self.limit = 12
		self.offset = 0
		self.include_content = True
		self.headers = { }
		self.timeout = 20
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'client', 'query', 'page', 'limit', 'offset', 'include_content',
				'response', 'params', 'result', 'fetch_search', 'fetch_page', 'fetch',
				'create_schema' ]
	
	def fetch_search( self, query: str, limit: int=12, offset: int=0 ) -> Dict[ str, Any ] | None:
		"""Search Grokipedia pages.

		Purpose:
		    Validate required search arguments, instantiate GrokipediaClient without explicit
		    credentials, invoke search, and package its response.

		Args:
		    query (str): Nonblank Grokipedia search text.
		    limit (int): Requested search result count. Defaults to 12.
		    offset (int): Search result offset. Defaults to 0.

		Returns:
		    Dict[str, Any]: Mode, client operation identifier, submitted arguments, API-key
		        configuration indicator, and client response.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			throw_if( 'limit', limit )
			self.limit = limit
			throw_if( 'offset', offset )
			self.offset = offset
			self.client = GrokipediaClient( )
			self.url = 'grokipedia.search'
			self.params = { 'query': self.query, 'limit': self.limit, 'offset': self.offset }
			self.response = self.client.search( query=self.query, limit=self.limit,
				offset=self.offset )
			
			self.result = { 'mode': 'search', 'url': self.url, 'params': self.params,
					'api_key_configured': bool( ( self.api_key or '' ).strip( ) ),
					'data': self.response }
			
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'fetch_search( self, query: str, limit: int=12, offset: int=0 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_page( self, page: str, include_content: bool=True ) -> Dict[ str, Any ] | None:
		"""Retrieve a Grokipedia page.

		Purpose:
		    Validate the page and content flag, instantiate GrokipediaClient without explicit
		    credentials, invoke get_page, and package its response.

		Args:
		    page (str): Nonblank page identifier passed to get_page.
		    include_content (bool): Request page content. Defaults to True.

		Returns:
		    Dict[str, Any]: Mode, client operation identifier, submitted arguments, API-key
		        configuration indicator, and client response.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'include_content', include_content )

			throw_if( 'page', page )
			self.page = page
			self.include_content = include_content
			self.client = GrokipediaClient( )
			self.url = 'grokipedia.get_page'
			self.params = { 'page': self.page, 'include_content': self.include_content }
			self.response = self.client.get_page( self.page, include_content=self.include_content )
			self.result = { 'mode': 'page', 'url': self.url, 'params': self.params,
					'api_key_configured': bool( ( self.api_key or '' ).strip( ) ),
					'data': self.response }
			
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'fetch_page( self, page: str, include_content: bool=True ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='search', query: str='', page: str='', limit: int=12, offset: int=0, include_content: bool=True ) -> Dict[ str, Any ] | None:
		"""Dispatch a Grokipedia retrieval operation.

		Purpose:
		    Select search or page retrieval and forward the relevant arguments.

		Args:
		    mode (str): search or page. Defaults to 'search'.
		    query (str): Nonblank Grokipedia search text. Defaults to an empty string; required in
		        search mode.
		    page (str): Nonblank page identifier passed to get_page. Defaults to an empty string;
		        required in page mode.
		    limit (int): Requested search result count. Defaults to 12.
		    offset (int): Search result offset. Defaults to 0.
		    include_content (bool): Request page content. Defaults to True.

		Returns:
		    Dict[str, Any]: Mode, client operation identifier, submitted arguments, API-key
		        configuration indicator, and client response.

		Raises:
		    Error: Raised when argument validation, provider retrieval, or response processing
		        fails.
		"""
		try:
			throw_if( 'mode', mode )
			mode_value = mode
			if mode_value == 'search':
				return self.fetch_search( query=query, limit=limit, offset=offset )
			
			if mode_value == 'page':
				return self.fetch_page( page=page, include_content=include_content )
			
			raise ValueError( "Unsupported Grokipedia mode. Use 'search' or 'page'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = "fetch( self, mode: str='search', query: str='', page: str='', limit: int=12, offset: int=0, include_content: bool=True ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ), 'description': (f'{description.strip( )} '
			                                                    f'This function uses the '
			                                                    f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class GoogleGeocoding( Fetcher ):
	"""Google Geocoding API adapter.

	Purpose:
	    Perform forward address, reverse coordinate, and place identifier geocoding, preserving
	    the complete decoded response and excluding the API key from returned request parameters.

	Attributes:
	    api_key (Optional[str]): Google API key loaded from configuration.
	    url (Optional[str]): Google Geocoding JSON endpoint.
	    params (Optional[Dict[str, Any]]): Latest submitted parameters, including the API key.
	    mode (Optional[str]): Forward, reverse, or place retrieval mode.
	    query (Optional[str]): Latest forward-geocoding address or location query.
	    latitude (Optional[float]): Latest reverse-geocoding latitude.
	    longitude (Optional[float]): Latest reverse-geocoding longitude.
	    place_id (Optional[str]): Latest requested Google place identifier.
	    language (Optional[str]): Requested response language.
	    region (Optional[str]): Optional region bias.
	    result_type (Optional[str]): Optional reverse-geocoding result type filter.
	    location_type (Optional[str]): Optional reverse-geocoding location type filter.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    response (Optional[Response]): Latest HTTP response.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	place_id: Optional[ str ]
	language: Optional[ str ]
	region: Optional[ str ]
	result_type: Optional[ str ]
	location_type: Optional[ str ]
	timeout: Optional[ int ]
	agents: Optional[ str ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the Google API key and user-agent, configure the geocoding endpoint and JSON
		    headers, and initialize forward mode, English language, and a ten-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.url = 'https://maps.googleapis.com/maps/api/geocode/json'
		self.params = { }
		self.mode = 'forward'
		self.query = ''
		self.latitude = None
		self.longitude = None
		self.place_id = ''
		self.language = 'en'
		self.region = ''
		self.result_type = ''
		self.location_type = ''
		self.timeout = 10
		self.agents = cfg.AGENTS
		self.response = None
		self.payload = { }
		self.result = { }
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'url', 'params', 'mode', 'query', 'latitude', 'longitude', 'place_id',
				'language', 'region', 'result_type', 'location_type', 'timeout', 'agents',
				'response', 'payload', 'result', 'request', 'fetch_forward', 'fetch_reverse',
				'fetch_place', 'fetch', 'create_schema' ]
	
	def request( self, params: Dict[ str, Any ], time: int=10 ) -> Dict[ str, Any ] | None:
		"""Execute a Google Geocoding request.

		Purpose:
		    Require a nonempty parameter dictionary, omit None and blank strings, append the
		    configured API key, issue a GET request, and reject non-object JSON or nonempty provider
		    statuses other than OK.

		Args:
		    params (Dict[str, Any]): Nonempty geocoding parameter dictionary.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, parameters excluding the key, provider status,
		        results list, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'params', params )

			
			
			
			throw_if( 'time', time )
			self.timeout = time
			if not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			self.params = { }
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.params[ 'key' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'Google Geocoding returned an invalid response payload.' )
			status = str( self.payload.get( 'status', '' ) or '' ).strip( )
			error_message = str( self.payload.get( 'error_message', '' ) or '' ).strip( )
			if status and status != 'OK':
				detail = f' — {error_message}' if error_message else ''
				raise ValueError( f'Google Geocoding failed: {status}{detail}' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': { key: value for key, value in self.params.items( ) if key != 'key' },
					'status': self.payload.get( 'status', '' ) if isinstance( self.payload,
						dict ) else '',
					'results': self.payload.get( 'results', [ ] ) if isinstance( self.payload,
						dict ) else [ ],
					'raw': self.payload }
			
			return self.result
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = 'request( self, params: Dict[ str, Any ], time: int=10 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_forward( self, query: str, language: str='en', region: str='', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Geocode an address or location query.

		Purpose:
		    Submit the query as the address parameter with the requested language and optional
		    region.

		Args:
		    query (str): Nonblank address or location query.
		    language (str): Requested response language. Defaults to 'en'.
		    region (str): Optional region bias passed to the provider. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, parameters excluding the key, provider status,
		        results list, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'query', query )

			
			self.mode = 'forward'
			self.query = query
			self.language = language
			self.region = region
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'address': self.query, 'language': self.language }
			
			if self.region:
				self.params[ 'region' ] = self.region
			
			return self.request( params=self.params, time=self.timeout )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = "fetch_forward( self, query: str, language: str='en', region: str='', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_reverse( self, latitude: float, longitude: float, language: str='en', result_type: str='', location_type: str='', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Reverse-geocode geographic coordinates.

		Purpose:
		    Validate finite geographic coordinates, construct the latlng parameter, and attach
		    optional result and location type filters.

		Args:
		    latitude (float): Finite latitude in decimal degrees, from -90 through 90.
		    longitude (float): Finite longitude in decimal degrees, from -180 through 180.
		    language (str): Requested response language. Defaults to 'en'.
		    result_type (str): Optional reverse-geocoding result type filter. Defaults to an empty
		        string.
		    location_type (str): Optional reverse-geocoding location type filter. Defaults to an
		        empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, parameters excluding the key, provider status,
		        results list, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'latitude', latitude )
			throw_if( 'longitude', longitude )

			
			self.mode = 'reverse'
			self.latitude = latitude
			self.longitude = longitude
			if not math.isfinite( self.latitude ) or not math.isfinite( self.longitude ) or not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
				raise ValueError( 'Latitude or longitude is outside valid geographic bounds.' )
			self.language = language
			self.result_type = result_type
			self.location_type = location_type
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'latlng': f'{self.latitude},{self.longitude}',
					'language': self.language }
			
			if self.result_type:
				self.params[ 'result_type' ] = self.result_type
			
			if self.location_type:
				self.params[ 'location_type' ] = self.location_type
			
			return self.request( params=self.params, time=self.timeout )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = "fetch_reverse( self, latitude: float, longitude: float, language: str='en', result_type: str='', location_type: str='', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_place( self, place_id: str, language: str='en', region: str='', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Geocode a Google place identifier.

		Purpose:
		    Submit a place identifier with the requested language and optional region.

		Args:
		    place_id (str): Nonblank Google place identifier.
		    language (str): Requested response language. Defaults to 'en'.
		    region (str): Optional region bias passed to the provider. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, parameters excluding the key, provider status,
		        results list, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'place_id', place_id )

			self.mode = 'place'
			self.place_id = place_id
			self.language = language
			self.region = region
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'place_id': self.place_id, 'language': self.language }
			if self.region:
				self.params[ 'region' ] = self.region
			
			return self.request( params=self.params, time=self.timeout )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = "fetch_place( self, place_id: str, language: str='en', region: str='', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='forward', query: str='', latitude: float=0.0, longitude: float=0.0, place_id: str='', language: str='en', region: str='', result_type: str='', location_type: str='', time: int=10 ) -> Dict[ str, Any ] | None:
		"""Dispatch a geocoding operation.

		Purpose:
		    Select forward, reverse, or place mode and delegate with the corresponding query,
		    coordinates, or place identifier.

		Args:
		    mode (str): forward, reverse, or place. Defaults to 'forward'.
		    query (str): Address or location query. Defaults to an empty string; required in forward
		        mode.
		    latitude (float): Finite latitude in decimal degrees, from -90 through 90. Defaults to
		        0.0.
		    longitude (float): Finite longitude in decimal degrees, from -180 through 180. Defaults
		        to 0.0.
		    place_id (str): Google place identifier. Defaults to an empty string; required in place
		        mode.
		    language (str): Requested response language. Defaults to 'en'.
		    region (str): Optional region bias passed to the provider. Defaults to an empty string.
		    result_type (str): Optional reverse-geocoding result type filter. Defaults to an empty
		        string.
		    location_type (str): Optional reverse-geocoding location type filter. Defaults to an
		        empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 10.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, parameters excluding the key, provider status,
		        results list, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			if self.mode == 'forward':
				return self.fetch_forward( query=query, language=language, region=region,
					time=time )
			
			if self.mode == 'reverse':
				return self.fetch_reverse( latitude=latitude, longitude=longitude,
					language=language, result_type=result_type, location_type=location_type,
					time=time )
			
			if self.mode == 'place':
				return self.fetch_place( place_id=place_id, language=language, region=region,
					time=time )
			
			raise ValueError( "Unsupported geocoding mode. Use 'forward', 'reverse', or 'place'." )
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = "fetch( self, mode: str='forward', query: str='', latitude: float=0.0, longitude: float=0.0, place_id: str='', language: str='en', region: str='', result_type: str='', location_type: str='', time: int=10 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Error:
			raise
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleGeocoding'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class CensusData( Fetcher ):
	"""U.S. Census Bureau dataset API adapter.

	Purpose:
	    Retrieve dataset variable metadata and geographic data tables, applying variable
	    selection and optional predicates and converting tabular JSON to column names and row
	    dictionaries.

	Attributes:
	    api_key (Optional[str]): Census API key loaded from configuration; attached to data
	        requests when present.
	    base_url (Optional[str]): Base provider API URL.
	    year (Optional[str]): Requested four-digit Census vintage.
	    dataset (Optional[str]): Requested dataset path.
	    mode (Optional[str]): Selected retrieval operation.
	    fields (Optional[List[str]]): Parsed list of requested variable names.
	    geography_for (Optional[str]): Required target geography selector for data retrieval.
	    geography_in (Optional[str]): Optional parent geography selector.
	    predicates (Optional[Dict[str, Any]]): Parsed additional query predicates.
	    params (Optional[Dict[str, Any]]): Latest query parameters, including the key when
	        configured.
	    payload (Optional[Any]): Latest decoded metadata object or tabular data array.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest endpoint URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	year: Optional[ str ]
	dataset: Optional[ str ]
	mode: Optional[ str ]
	fields: Optional[ List[ str ] ]
	geography_for: Optional[ str ]
	geography_in: Optional[ str ]
	predicates: Optional[ Dict[ str, Any ] ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured Census API key, configure the base data URL, JSON headers,
		    twenty-second timeout, and empty dataset, geography, and response state.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.CENSUS_API_KEY
		self.base_url = 'https://api.census.gov/data'
		self.year = None
		self.dataset = None
		self.mode = None
		self.fields = [ ]
		self.geography_for = None
		self.geography_in = None
		self.predicates = { }
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents, }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'year', 'dataset', 'mode', 'fields', 'geography_for',
				'geography_in', 'predicates', 'params', 'payload', 'result', 'response', 'url',
				'timeout', 'agents', 'headers', 'normalize_fields', 'parse_predicates',
				'shape_table', 'fetch_variables', 'fetch_data', 'fetch', 'create_schema' ]
	
	def normalize_fields( self, fields: str ) -> str:
		"""Parse Census variable selection.

		Purpose:
		    Split comma-separated names, trim and discard blank entries, retain the resulting list,
		    and reject an empty selection.

		Args:
		    fields (str): Nonblank comma-separated Census variable names.

		Returns:
		    str: Comma-separated nonblank variable names.

		Raises:
		    Error: Raised when field selection is missing or cannot produce any variables.
		"""
		try:
			throw_if( 'fields', fields )
			
			self.fields = [ field.strip( ) for field in fields.split( ',' ) if
					field and field.strip( ) ]
			
			if not self.fields:
				raise ValueError( 'At least one Census field is required.' )
			
			return ','.join( self.fields )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = 'normalize_fields( self, fields: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def parse_predicates( self, predicates: str='' ) -> Dict[ str, Any ]:
		"""Parse additional Census predicates.

		Purpose:
		    Reset the predicate dictionary, parse nonblank lines on the first equals sign, trim
		    names and values, and reject missing separators or blank keys and values.

		Args:
		    predicates (str): Optional newline-separated key=value filters. Defaults to an empty
		        string.

		Returns:
		    Dict[str, Any]: Parsed predicate names mapped to string values; duplicate names retain
		        the last value.

		Raises:
		    Error: Raised when a predicate line is malformed.
		"""
		try:
			self.predicates = { }
			text = ( predicates or '' ).strip( )
			
			if not text:
				return self.predicates
			
			for line in text.splitlines( ):
				clean = str( line or '' ).strip( )
				
				if not clean:
					continue
				
				if '=' not in clean:
					raise ValueError( 'Each Census predicate line must use key=value format.' )
				
				key, value = clean.split( '=', 1 )
				key = key.strip( )
				value = value.strip( )
				
				throw_if( 'predicate key', key )
				throw_if( 'predicate value', value )
				
				self.predicates[ key ] = value
			
			return self.predicates
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = "parse_predicates( self, predicates: str='' ) -> Dict[ str, Any ]"
			Logger( ).write( exception )
			raise exception
	
	def shape_table( self, rows: List[ Any ] ) -> Dict[ str, Any ]:
		"""Convert Census table rows to records.

		Purpose:
		    Use the first list row as column names, convert subsequent list rows to dictionaries,
		    fill missing cells with empty strings, and ignore extra cells and non-list data rows.

		Args:
		    rows (List[Any]): Census tabular response containing a header row followed by records.

		Returns:
		    Dict[str, Any]: Column names, row dictionaries, and record count; empty collections for
		        empty input.

		Raises:
		    Error: Raised when table shaping fails.
		"""
		try:
			if not rows:
				return { 'columns': [ ], 'rows': [ ], 'count': 0, }
			
			headers = rows[ 0 ] if isinstance( rows[ 0 ], list ) else [ ]
			data_rows = rows[ 1: ] if len( rows ) > 1 else [ ]
			records: List[ Dict[ str, Any ] ] = [ ]
			
			for row in data_rows:
				if isinstance( row, list ) and headers:
					records.append( { headers[ index ]: row[ index ] if index < len( row ) else ''
							for index in range( len( headers ) ) } )
			
			return { 'columns': headers, 'rows': records, 'count': len( records ), }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = 'shape_table( self, rows: List[ Any ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_variables( self, year: str, dataset: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve Census variable metadata.

		Purpose:
		    Validate vintage and dataset path, request variables.json without an API key, and reject
		    non-object or error-bearing JSON.

		Args:
		    year (str): Four-digit Census dataset vintage.
		    dataset (str): Dataset path containing slash-separated segments of letters, digits,
		        underscores, or hyphens.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters excluding the key, and decoded variable metadata.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'year', year )
			throw_if( 'dataset', dataset )

			self.mode = 'variables'
			self.year = year
			self.dataset = dataset
			throw_if( 'time', time )
			self.timeout = time
			if not re.fullmatch( r'\d{4}', self.year ):
				raise ValueError( 'year must be a four-digit Census vintage.' )
			if not re.fullmatch( r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', self.dataset ):
				raise ValueError( 'dataset must contain valid Census dataset path segments.' )
			self.url = f'{self.base_url}/{self.year}/{self.dataset}/variables.json'
			self.params = { }
			
			# Census metadata does not require an API key.
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ) or self.payload.get( 'error' ):
				raise ValueError( 'Census variables response must be a valid JSON object.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': { key: value for key, value in self.params.items( ) if key != 'key' },
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = 'fetch_variables( self, year: str, dataset: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_data( self, year: str, dataset: str, fields: str, geography_for: str='', geography_in: str='', predicates: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve Census geographic data.

		Purpose:
		    Validate dataset identifiers, parse predicates and variables, require target geography,
		    and attach optional parent geography and configured API key. Require a JSON table with a
		    list header row and shape its records.

		Args:
		    year (str): Four-digit Census dataset vintage.
		    dataset (str): Dataset path containing slash-separated segments of letters, digits,
		        underscores, or hyphens.
		    fields (str): Nonblank comma-separated Census variable names.
		    geography_for (str): Required Census geography selector for the 'for' parameter.
		    geography_in (str): Optional parent geography selector for the 'in' parameter. Defaults
		        to an empty string.
		    predicates (str): Optional newline-separated key=value filters. Defaults to an empty
		        string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, parameters excluding the key, shaped columns and records, and
		        raw tabular response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'year', year )
			throw_if( 'dataset', dataset )
			throw_if( 'fields', fields )
			throw_if( 'geography_for', geography_for )

			
			self.mode = 'data'
			self.year = year
			self.dataset = dataset
			self.geography_for = geography_for
			self.geography_in = geography_in
			self.predicates = self.parse_predicates( predicates )
			throw_if( 'time', time )
			self.timeout = time
			if not re.fullmatch( r'\d{4}', self.year ):
				raise ValueError( 'year must be a four-digit Census vintage.' )
			if not re.fullmatch( r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', self.dataset ):
				raise ValueError( 'dataset must contain valid Census dataset path segments.' )
			self.url = f'{self.base_url}/{self.year}/{self.dataset}'
			self.params = { 'get': self.normalize_fields( fields ), 'for': self.geography_for, }
			
			if self.geography_in:
				self.params[ 'in' ] = self.geography_in
			
			if self.predicates:
				self.params.update( self.predicates )
			
			if self.api_key:
				self.params[ 'key' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, list ) or not self.payload or not isinstance( self.payload[ 0 ], list ):
				raise ValueError( 'Census data response must be a table with a header row.' )
			self.result = { 'mode': self.mode, 'url': self.url, 'params': { key: value for key, value in self.params.items( ) if key != 'key' },
					'data': self.shape_table( self.payload ), 'raw': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = "fetch_data( self, year: str, dataset: str, fields: str, geography_for: str='', geography_in: str='', predicates: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='variables', year: str='2022', dataset: str='acs/acs5', fields: str='NAME,B01001_001E', geography_for: str='state:*', geography_in: str='', predicates: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a Census retrieval operation.

		Purpose:
		    Select variables metadata or geographic data retrieval and forward the applicable
		    arguments.

		Args:
		    mode (str): variables or data. Defaults to 'variables'.
		    year (str): Four-digit Census dataset vintage. Defaults to '2022'.
		    dataset (str): Dataset path containing slash-separated segments of letters, digits,
		        underscores, or hyphens. Defaults to 'acs/acs5'.
		    fields (str): Nonblank comma-separated Census variable names. Defaults to
		        'NAME,B01001_001E'.
		    geography_for (str): Required Census geography selector for the 'for' parameter.
		        Defaults to 'state:*'.
		    geography_in (str): Optional parent geography selector for the 'in' parameter. Defaults
		        to an empty string.
		    predicates (str): Optional newline-separated key=value filters. Defaults to an empty
		        string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged variable metadata or shaped geographic data response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'variables':
				return self.fetch_variables( year=year, dataset=dataset, time=time )
			
			if self.mode == 'data':
				return self.fetch_data( year=year, dataset=dataset, fields=fields,
					geography_for=geography_for, geography_in=geography_in, predicates=predicates,
					time=time )
			
			raise ValueError( "Unsupported Census mode. Use 'variables' or 'data'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = "fetch( self, mode: str='variables', year: str='2022', dataset: str='acs/acs5', fields: str='NAME,B01001_001E', geography_for: str='state:*', geography_in: str='', predicates: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'CensusData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class Socrata( Fetcher ):
	"""Socrata open-data portal adapter.

	Purpose:
	    Retrieve dataset metadata or rows through Socrata-style portal endpoints, applying
	    optional SoQL clauses and pagination and sending the configured application token when
	    present.

	Attributes:
	    api_key (Optional[str]): Socrata application token loaded from configuration; sent in
	        X-App-Token when present.
	    base_url (Optional[str]): Template URL for dataset row retrieval.
	    metadata_url (Optional[str]): Template URL for dataset metadata retrieval.
	    domain (Optional[str]): Normalized portal domain.
	    dataset_id (Optional[str]): Normalized dataset identifier.
	    mode (Optional[str]): Selected retrieval operation.
	    select_clause (Optional[str]): Latest SoQL selection expression.
	    where_clause (Optional[str]): Latest SoQL filtering expression.
	    order_clause (Optional[str]): Latest SoQL ordering expression.
	    group_clause (Optional[str]): Latest SoQL grouping expression.
	    limit_value (Optional[int]): Requested row limit.
	    offset_value (Optional[int]): Requested row offset.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest endpoint URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	metadata_url: Optional[ str ]
	domain: Optional[ str ]
	dataset_id: Optional[ str ]
	mode: Optional[ str ]
	select_clause: Optional[ str ]
	where_clause: Optional[ str ]
	order_clause: Optional[ str ]
	group_clause: Optional[ str ]
	limit_value: Optional[ int ]
	offset_value: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure row and metadata endpoint templates, load the Socrata application token,
		    initialize JSON headers, a twenty-second timeout, and paging defaults of 25 rows and
		    zero offset.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.SOCRATA_API_KEY
		self.base_url = 'https://{domain}/resource/{dataset}.json'
		self.metadata_url = 'https://{domain}/api/views/{dataset}.json'
		self.domain = None
		self.dataset_id = None
		self.mode = 'rows'
		self.select_clause = ''
		self.where_clause = ''
		self.order_clause = ''
		self.group_clause = ''
		self.limit_value = 25
		self.offset_value = 0
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
		
		if self.api_key:
			self.headers[ 'X-App-Token' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'metadata_url', 'domain', 'dataset_id', 'mode',
				'select_clause', 'where_clause', 'order_clause', 'group_clause', 'limit_value',
				'offset_value', 'params', 'payload', 'result', 'response', 'url', 'timeout',
				'agents', 'headers', 'normalize_domain', 'normalize_dataset_id', 'validate_limit',
				'validate_offset', 'fetch_metadata', 'fetch_rows', 'fetch', 'create_schema' ]
	
	def normalize_domain( self, domain: str ) -> str:
		"""Normalize a portal domain.

		Purpose:
		    Strip surrounding whitespace, remove HTTP and HTTPS scheme strings and surrounding
		    slashes, then require a nonempty value containing a dot.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.

		Returns:
		    str: Normalized domain string.

		Raises:
		    Error: Raised when the supplied domain is missing or fails the nonempty and dot checks.
		"""
		try:
			throw_if( 'domain', domain )
			
			value = domain.strip( )
			value = value.replace( 'https://', '' ).replace( 'http://', '' ).strip( '/' )
			
			if not value or '.' not in value:
				raise ValueError( 'domain must be a valid Socrata portal domain.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'normalize_domain( self, domain: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_dataset_id( self, dataset_id: str ) -> str:
		"""Normalize a dataset identifier.

		Purpose:
		    Strip whitespace, remove .json substrings and surrounding slashes, and reject an empty
		    result.

		Args:
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.

		Returns:
		    str: Normalized dataset identifier.

		Raises:
		    Error: Raised when the identifier is missing or becomes empty.
		"""
		try:
			throw_if( 'dataset_id', dataset_id )
			
			value = dataset_id.strip( )
			value = value.replace( '.json', '' ).strip( '/' )
			
			if not value:
				raise ValueError( 'dataset_id cannot be empty.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'normalize_dataset_id( self, dataset_id: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the requested row limit.

		Purpose:
		    Enforce the inclusive range from 1 through 50000.

		Args:
		    limit (int): Requested row count from 1 through 50000.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the limit is missing or outside the range.
		"""
		try:
			

			throw_if( 'limit', limit )

			if limit < 1 or limit > 50000:
				raise ValueError( 'limit must be between 1 and 50000.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the requested row offset.

		Purpose:
		    Reject missing or negative offsets.

		Args:
		    offset (int): Nonnegative row offset.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the offset is missing or negative.
		"""
		try:
			

			throw_if( 'offset', offset )

			if offset < 0:
				raise ValueError( 'offset must be greater than or equal to 0.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def fetch_metadata( self, domain: str, dataset_id: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve dataset metadata.

		Purpose:
		    Normalize the domain and dataset identifier, construct the metadata endpoint, attach the
		    application token if configured, issue a GET request, and decode JSON.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, empty query parameters, and decoded metadata.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'metadata'
			self.domain = self.normalize_domain( domain )
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			throw_if( 'time', time )
			self.timeout = time
			self.url = self.metadata_url.format( domain=self.domain, dataset=self.dataset_id )
			self.params = { }
			
			
			if self.api_key:
				self.headers[ 'X-App-Token' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'fetch_metadata( self, domain: str, dataset_id: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_rows( self, domain: str, dataset_id: str, select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve dataset rows.

		Purpose:
		    Normalize endpoint identifiers, validate pagination, assemble optional SoQL clauses,
		    issue a GET request with the configured application token when present, and package the
		    decoded response.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.
		    select (str): Optional SoQL selection expression. Defaults to an empty string.
		    where (str): Optional SoQL filtering expression. Defaults to an empty string.
		    order (str): Optional SoQL ordering expression. Defaults to an empty string.
		    group (str): Optional SoQL grouping expression. Defaults to an empty string.
		    limit (int): Requested row limit from 1 through 50000. Defaults to 25.
		    offset (int): Nonnegative row offset. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, decoded data, and count equal
		        to the list length or zero for non-list payloads.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'rows'
			self.domain = self.normalize_domain( domain )
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			self.select_clause = select
			self.where_clause = where
			self.order_clause = order
			self.group_clause = group
			self.limit_value = self.validate_limit( limit )
			self.offset_value = self.validate_offset( offset )
			throw_if( 'time', time )
			self.timeout = time
			
			
			self.url = self.base_url.format( domain=self.domain, dataset=self.dataset_id )
			self.params = { '$limit': self.limit_value, '$offset': self.offset_value, }
			
			if self.select_clause:
				self.params[ '$select' ] = self.select_clause
			
			if self.where_clause:
				self.params[ '$where' ] = self.where_clause
			
			if self.order_clause:
				self.params[ '$order' ] = self.order_clause
			
			if self.group_clause:
				self.params[ '$group' ] = self.group_clause
			
			if self.api_key:
				self.headers[ 'X-App-Token' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'count': len( self.payload ) if isinstance( self.payload, list ) else 0,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = "fetch_rows( self, domain: str, dataset_id: str, select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='rows', domain: str='data.cdc.gov', dataset_id: str='', select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a dataset retrieval operation.

		Purpose:
		    Select metadata or rows retrieval and forward the corresponding endpoint identifiers and
		    filters.

		Args:
		    mode (str): metadata or rows. Defaults to 'rows'.
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required. Defaults to 'data.cdc.gov'.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed. Defaults to an empty string; supply a dataset identifier for either mode.
		    select (str): Optional SoQL selection expression. Defaults to an empty string.
		    where (str): Optional SoQL filtering expression. Defaults to an empty string.
		    order (str): Optional SoQL ordering expression. Defaults to an empty string.
		    group (str): Optional SoQL grouping expression. Defaults to an empty string.
		    limit (int): Requested row limit from 1 through 50000. Defaults to 25.
		    offset (int): Nonnegative row offset. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged metadata or row response for the selected operation.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'metadata':
				return self.fetch_metadata( domain=domain, dataset_id=dataset_id, time=time )
			
			if self.mode == 'rows':
				return self.fetch_rows( domain=domain, dataset_id=dataset_id, select=select,
					where=where, order=order, group=group, limit=limit, offset=offset, time=time )
			
			raise ValueError( "Unsupported Socrata mode. Use 'metadata' or 'rows'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = "fetch( self, mode: str='rows', domain: str='data.cdc.gov', dataset_id: str='', select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Socrata'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class HealthData( Fetcher ):
	"""HealthData.gov Socrata-style dataset adapter.

	Purpose:
	    Retrieve dataset metadata or rows through Socrata-style portal endpoints, applying
	    optional SoQL clauses and pagination and sending the configured application token when
	    present.

	Attributes:
	    api_key (Optional[str]): HealthData application token loaded from configuration; sent in
	        X-App-Token when present.
	    base_url (Optional[str]): Template URL for dataset row retrieval.
	    metadata_url (Optional[str]): Template URL for dataset metadata retrieval.
	    domain (Optional[str]): Normalized portal domain.
	    dataset_id (Optional[str]): Normalized dataset identifier.
	    mode (Optional[str]): Selected retrieval operation.
	    select_clause (Optional[str]): Latest SoQL selection expression.
	    where_clause (Optional[str]): Latest SoQL filtering expression.
	    order_clause (Optional[str]): Latest SoQL ordering expression.
	    group_clause (Optional[str]): Latest SoQL grouping expression.
	    limit_value (Optional[int]): Requested row limit.
	    offset_value (Optional[int]): Requested row offset.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest endpoint URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	metadata_url: Optional[ str ]
	domain: Optional[ str ]
	dataset_id: Optional[ str ]
	mode: Optional[ str ]
	select_clause: Optional[ str ]
	where_clause: Optional[ str ]
	order_clause: Optional[ str ]
	group_clause: Optional[ str ]
	limit_value: Optional[ int ]
	offset_value: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure row and metadata endpoint templates, load the HealthData application token,
		    initialize JSON headers, a twenty-second timeout, and paging defaults of 25 rows and
		    zero offset.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.HEALTHDATA_API_KEY
		self.base_url = 'https://{domain}/resource/{dataset}.json'
		self.metadata_url = 'https://{domain}/api/views/{dataset}.json'
		self.domain = 'healthdata.gov'
		self.dataset_id = ''
		self.mode = 'rows'
		self.select_clause = ''
		self.where_clause = ''
		self.order_clause = ''
		self.group_clause = ''
		self.limit_value = 25
		self.offset_value = 0
		self.params = { }
		self.payload = [ ]
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents, }
		
		if self.api_key:
			self.headers[ 'X-App-Token' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'metadata_url', 'domain', 'dataset_id', 'mode',
				'select_clause', 'where_clause', 'order_clause', 'group_clause', 'limit_value',
				'offset_value', 'params', 'payload', 'result', 'response', 'url', 'timeout',
				'agents', 'headers', 'normalize_domain', 'normalize_dataset_id', 'validate_limit',
				'validate_offset', 'fetch_metadata', 'fetch_rows', 'fetch', 'create_schema' ]
	
	def normalize_domain( self, domain: str ) -> str:
		"""Normalize a portal domain.

		Purpose:
		    Strip surrounding whitespace, remove HTTP and HTTPS scheme strings and surrounding
		    slashes, then require a nonempty value containing a dot.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.

		Returns:
		    str: Normalized domain string.

		Raises:
		    Error: Raised when the supplied domain is missing or fails the nonempty and dot checks.
		"""
		try:
			throw_if( 'domain', domain )
			
			value = domain.strip( )
			value = value.replace( 'https://', '' ).replace( 'http://', '' ).strip( '/' )
			
			if not value or '.' not in value:
				raise ValueError( 'domain must be a valid HealthData.gov portal domain.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'normalize_domain( self, domain: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_dataset_id( self, dataset_id: str ) -> str:
		"""Normalize a dataset identifier.

		Purpose:
		    Strip whitespace, remove .json substrings and surrounding slashes, and reject an empty
		    result.

		Args:
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.

		Returns:
		    str: Normalized dataset identifier.

		Raises:
		    Error: Raised when the identifier is missing or becomes empty.
		"""
		try:
			throw_if( 'dataset_id', dataset_id )
			
			value = dataset_id.strip( )
			value = value.replace( '.json', '' ).strip( '/' )
			
			if not value:
				raise ValueError( 'dataset_id cannot be empty.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'normalize_dataset_id( self, dataset_id: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the requested row limit.

		Purpose:
		    Enforce the inclusive range from 1 through 50000.

		Args:
		    limit (int): Requested row count from 1 through 50000.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the limit is missing or outside the range.
		"""
		try:
			

			throw_if( 'limit', limit )

			if limit < 1 or limit > 50000:
				raise ValueError( 'limit must be between 1 and 50000.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the requested row offset.

		Purpose:
		    Reject missing or negative offsets.

		Args:
		    offset (int): Nonnegative row offset.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the offset is missing or negative.
		"""
		try:
			

			throw_if( 'offset', offset )

			if offset < 0:
				raise ValueError( 'offset must be greater than or equal to 0.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def fetch_metadata( self, domain: str, dataset_id: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve dataset metadata.

		Purpose:
		    Normalize the domain and dataset identifier, construct the metadata endpoint, attach the
		    application token if configured, issue a GET request, and decode JSON.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, empty query parameters, and decoded metadata.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'metadata'
			self.domain = self.normalize_domain( domain )
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			throw_if( 'time', time )
			self.timeout = time
			self.url = self.metadata_url.format( domain=self.domain, dataset=self.dataset_id )
			self.params = { }
			
			
			if self.api_key:
				self.headers[ 'X-App-Token' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'fetch_metadata( self, domain: str, dataset_id: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_rows( self, domain: str, dataset_id: str, select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve dataset rows.

		Purpose:
		    Normalize endpoint identifiers, validate pagination, assemble optional SoQL clauses,
		    issue a GET request with the configured application token when present, and package the
		    decoded response.

		Args:
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed.
		    select (str): Optional SoQL selection expression. Defaults to an empty string.
		    where (str): Optional SoQL filtering expression. Defaults to an empty string.
		    order (str): Optional SoQL ordering expression. Defaults to an empty string.
		    group (str): Optional SoQL grouping expression. Defaults to an empty string.
		    limit (int): Requested row limit from 1 through 50000. Defaults to 25.
		    offset (int): Nonnegative row offset. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, decoded data, and count equal
		        to the list length or zero for non-list payloads.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'rows'
			self.domain = self.normalize_domain( domain )
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			self.select_clause = select
			self.where_clause = where
			self.order_clause = order
			self.group_clause = group
			self.limit_value = self.validate_limit( limit )
			self.offset_value = self.validate_offset( offset )
			throw_if( 'time', time )
			self.timeout = time
			
			
			self.url = self.base_url.format( domain=self.domain, dataset=self.dataset_id )
			self.params = { '$limit': self.limit_value, '$offset': self.offset_value, }
			
			if self.select_clause:
				self.params[ '$select' ] = self.select_clause
			
			if self.where_clause:
				self.params[ '$where' ] = self.where_clause
			
			if self.order_clause:
				self.params[ '$order' ] = self.order_clause
			
			if self.group_clause:
				self.params[ '$group' ] = self.group_clause
			
			if self.api_key:
				self.headers[ 'X-App-Token' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'count': len( self.payload ) if isinstance( self.payload, list ) else 0,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = "fetch_rows( self, domain: str, dataset_id: str, select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='rows', domain: str='healthdata.gov', dataset_id: str='', select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a dataset retrieval operation.

		Purpose:
		    Select metadata or rows retrieval and forward the corresponding endpoint identifiers and
		    filters.

		Args:
		    mode (str): metadata or rows. Defaults to 'rows'.
		    domain (str): Nonblank portal domain; scheme and surrounding slashes are removed, and a
		        dot is required. Defaults to 'healthdata.gov'.
		    dataset_id (str): Nonblank dataset identifier; .json and surrounding slashes are
		        removed. Defaults to an empty string; supply a dataset identifier for either mode.
		    select (str): Optional SoQL selection expression. Defaults to an empty string.
		    where (str): Optional SoQL filtering expression. Defaults to an empty string.
		    order (str): Optional SoQL ordering expression. Defaults to an empty string.
		    group (str): Optional SoQL grouping expression. Defaults to an empty string.
		    limit (int): Requested row limit from 1 through 50000. Defaults to 25.
		    offset (int): Nonnegative row offset. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged metadata or row response for the selected operation.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'metadata':
				return self.fetch_metadata( domain=domain, dataset_id=dataset_id, time=time )
			
			if self.mode == 'rows':
				return self.fetch_rows( domain=domain, dataset_id=dataset_id, select=select,
					where=where, order=order, group=group, limit=limit, offset=offset, time=time )
			
			raise ValueError( "Unsupported HealthData mode. Use 'metadata' or 'rows'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = "fetch( self, mode: str='rows', domain: str='healthdata.gov', dataset_id: str='', select: str='', where: str='', order: str='', group: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HealthData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class GlobalHealthData( Fetcher ):
	"""WHO Global Health Observatory retrieval adapter.

	Purpose:
	    Retrieve the WHO indicator metadata registry page or query the configured GHO API
	    endpoint, decoding JSON responses and retaining non-JSON content as HTML or text.

	Attributes:
	    api_key (Optional[str]): WHO API key loaded from configuration; sent in X-API-Key when
	        present.
	    base_url (Optional[str]): WHO Global Health Observatory website base URL.
	    athena_base_url (Optional[str]): Configured GHO API base URL used by athena mode.
	    mode (Optional[str]): Selected retrieval operation.
	    query_path (Optional[str]): Normalized relative GHO API query path.
	    fmt (Optional[str]): Requested provider output format.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded JSON response or HTML/text dictionary.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest endpoint URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	athena_base_url: Optional[ str ]
	mode: Optional[ str ]
	query_path: Optional[ str ]
	fmt: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured WHO API key, initialize the registry and GHO API base URLs, JSON
		    headers, twenty-second timeout, registry mode, and JSON format default.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.WHO_API_KEY
		self.base_url = 'https://www.who.int/data/gho'
		self.athena_base_url = 'https://ghoapi.azureedge.net/api'
		self.mode = 'indicator_registry'
		self.query_path = ''
		self.fmt = 'json'
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents, }
		
		if self.api_key:
			self.headers[ 'X-API-Key' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'athena_base_url', 'mode', 'query_path', 'fmt', 'params',
				'payload', 'result', 'response', 'url', 'timeout', 'agents', 'headers',
				'normalize_query_path', 'fetch_indicator_registry', 'fetch_athena', 'fetch',
				'create_schema' ]
	
	def normalize_query_path( self, query_path: str ) -> str:
		"""Normalize a relative GHO query path.

		Purpose:
		    Trim whitespace and leading slashes, then reject empty paths, full HTTP or HTTPS URLs,
		    and any parent-directory markers.

		Args:
		    query_path (str): Nonblank relative query path; leading slashes and surrounding
		        whitespace are stripped. Full HTTP(S) URLs and parent-directory markers are
		        rejected.

		Returns:
		    str: Accepted normalized relative query path.

		Raises:
		    Error: Raised when the path is missing or fails the path checks.
		"""
		try:
			throw_if( 'query_path', query_path )
			
			value = query_path.strip( ).lstrip( '/' )
			
			if not value:
				raise ValueError( "Argument 'query_path' cannot be empty!" )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'query_path must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'query_path cannot contain parent-directory markers.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalHealthData'
			exception.method = 'normalize_query_path( self, query_path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch_indicator_registry( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve the WHO indicator metadata registry.

		Purpose:
		    Request the registry webpage with the configured key header when present. Decode JSON
		    when the content type includes application/json; otherwise retain response text under
		    html.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, empty parameters, and decoded JSON or an HTML
		        dictionary.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'indicator_registry'
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/indicator-metadata-registry'
			self.params = { }
			
			
			if self.api_key:
				self.headers[ 'X-API-Key' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'html': self.response.text, }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalHealthData'
			exception.method = 'fetch_indicator_registry( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_athena( self, query_path: str, fmt: str='json', time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve GHO API data using a relative path.

		Purpose:
		    Normalize the query path, attach the requested $format parameter and configured key
		    header when present, and issue a GET request. Decode JSON for application/json
		    responses; otherwise retain response text.

		Args:
		    query_path (str): Nonblank relative query path; leading slashes and surrounding
		        whitespace are stripped. Full HTTP(S) URLs and parent-directory markers are
		        rejected.
		    fmt (str): Nonblank provider output format passed as $format. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or a text
		        dictionary.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'athena'
			self.query_path = self.normalize_query_path( query_path )
			throw_if( 'fmt', fmt )
			self.fmt = fmt
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.athena_base_url}/{self.query_path}'
			self.params = { }
			
			
			if self.fmt:
				self.params[ '$format' ] = self.fmt
			
			if self.api_key:
				self.headers[ 'X-API-Key' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'text': self.response.text, }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalHealthData'
			exception.method = "fetch_athena( self, query_path: str, fmt: str='json', time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='indicator_registry', query_path: str='', fmt: str='json', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a WHO GHO retrieval operation.

		Purpose:
		    Select indicator_registry or athena and delegate with applicable query options.

		Args:
		    mode (str): indicator_registry or athena. Defaults to 'indicator_registry'.
		    query_path (str): Nonblank relative query path; leading slashes and surrounding
		        whitespace are stripped. Full HTTP(S) URLs and parent-directory markers are
		        rejected. Defaults to an empty string; required for athena mode.
		    fmt (str): Nonblank provider output format passed as $format. Defaults to 'json'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Packaged registry or GHO API response for the selected mode.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'indicator_registry':
				return self.fetch_indicator_registry( time=time )
			
			if self.mode == 'athena':
				return self.fetch_athena( query_path=query_path, fmt=fmt, time=time )
			
			raise ValueError( "Unsupported WHO Global Health mode. Use 'indicator_registry' or "
			                  "'athena'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalHealthData'
			exception.method = "fetch( self, mode: str='indicator_registry', query_path: str='', fmt: str='json', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GlobalHealthData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class UnitedNations( Fetcher ):
	"""UNdata download-handler retrieval adapter.

	Purpose:
	    Retrieve content from the configured UNdata download handler or an appended relative
	    query path, decoding JSON responses and preserving other content as text.

	Attributes:
	    base_url (Optional[str]): Base provider endpoint URL.
	    catalog_url (Optional[str]): Configured download-handler URL used in datasets mode.
	    mode (Optional[str]): Selected retrieval operation.
	    query_path (Optional[str]): Normalized relative path appended in sdmx_query mode.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded JSON or text dictionary.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	catalog_url: Optional[ str ]
	mode: Optional[ str ]
	query_path: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the UNdata download handler as both base and catalog URL, initialize datasets
		    mode, content negotiation headers, and a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://data.un.org/Handlers/DownloadHandler.ashx'
		self.catalog_url = 'https://data.un.org/Handlers/DownloadHandler.ashx'
		self.mode = 'datasets'
		self.query_path = ''
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json, text/json, text/csv, text/xml, */*',
				'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'catalog_url', 'mode', 'query_path', 'params', 'payload', 'result',
				'response', 'url', 'timeout', 'agents', 'headers', 'normalize_query_path',
				'fetch_datasets', 'fetch_sdmx_query', 'fetch', 'create_schema' ]
	
	def normalize_query_path( self, query_path: str ) -> str:
		"""Normalize a relative resource path.

		Purpose:
		    Strip surrounding whitespace and leading slashes, then reject an empty result, full HTTP
		    or HTTPS URL, or parent-directory markers.

		Args:
		    query_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.

		Returns:
		    str: Accepted normalized relative path.

		Raises:
		    Error: Raised when the path is missing or fails the path checks.
		"""
		try:
			throw_if( 'query_path', query_path )
			
			value = query_path.strip( ).lstrip( '/' )
			
			if not value:
				raise ValueError( 'query_path cannot be empty.' )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'query_path must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'query_path cannot contain parent-directory markers.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UnitedNations'
			exception.method = 'normalize_query_path( self, query_path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch_datasets( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve content from the configured UNdata catalog URL.

		Purpose:
		    Send a GET request to the download handler without query parameters; decode
		    application/json or text/json responses, otherwise retain response text.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'datasets'
			throw_if( 'time', time )
			self.timeout = time
			self.url = self.catalog_url
			self.params = { }
			
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type or 'text/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'text': self.response.text }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UnitedNations'
			exception.method = 'fetch_datasets( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_sdmx_query( self, query_path: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve content using an appended UNdata query path.

		Purpose:
		    Normalize the relative query path, append it to the configured download-handler URL, and
		    issue a GET request. Decode JSON content or retain response text.

		Args:
		    query_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'sdmx_query'
			self.query_path = self.normalize_query_path( query_path )
			throw_if( 'time', time )
			self.timeout = time
			
			
			self.url = f'{self.base_url}/{self.query_path}'
			self.params = { }
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type or 'text/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'text': self.response.text }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UnitedNations'
			exception.method = 'fetch_sdmx_query( self, query_path: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='datasets', query_path: str='', time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Dispatch a UNdata retrieval operation.

		Purpose:
		    Select datasets or sdmx_query mode and forward applicable arguments.

		Args:
		    mode (str): datasets or sdmx_query. Defaults to 'datasets'.
		    query_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.
		        Defaults to an empty string; required in sdmx_query mode.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'datasets':
				return self.fetch_datasets( time=time )
			
			if self.mode == 'sdmx_query':
				return self.fetch_sdmx_query( query_path=query_path, time=time )
			
			raise ValueError( "Unsupported United Nations mode. Use 'datasets' or 'sdmx_query'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UnitedNations'
			exception.method = "fetch( self, mode: str='datasets', query_path: str='', time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UnitedNations'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class WorldPopulation( Fetcher ):
	"""WorldPop REST catalog and raster metadata adapter.

	Purpose:
	    Request the configured WorldPop REST catalog, catalog search, or relative asset endpoint
	    and retain JSON or textual content with response metadata.

	Attributes:
	    base_url (Optional[str]): Base provider endpoint URL.
	    stac_url (Optional[str]): Reserved WorldPop STAC base URL; not used by retrieval methods.
	    mode (Optional[str]): Selected retrieval operation.
	    query_text (Optional[str]): Latest catalog search text.
	    asset_path (Optional[str]): Latest normalized relative asset path.
	    page (Optional[int]): Requested one-based search page.
	    page_size (Optional[int]): Requested search page size.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded JSON or HTML/text dictionary.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	stac_url: Optional[ str ]
	mode: Optional[ str ]
	query_text: Optional[ str ]
	asset_path: Optional[ str ]
	page: Optional[ int ]
	page_size: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the WorldPop REST and STAC URLs, catalog mode, first page, page size of 25,
		    content negotiation headers, and a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://www.worldpop.org/rest'
		self.stac_url = 'https://worldpop.github.io/worldpop-stac'
		self.mode = 'catalog'
		self.query_text = ''
		self.asset_path = ''
		self.page = 1
		self.page_size = 25
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json, text/html, text/plain, */*',
				'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'stac_url', 'mode', 'query_text', 'asset_path', 'page', 'page_size',
				'params', 'payload', 'result', 'response', 'url', 'timeout', 'agents', 'headers',
				'normalize_asset_path', 'validate_page', 'validate_page_size', 'fetch_catalog',
				'search_catalog', 'fetch_raster_metadata', 'fetch', 'create_schema' ]
	
	def normalize_asset_path( self, asset_path: str ) -> str:
		"""Normalize a relative resource path.

		Purpose:
		    Strip surrounding whitespace and leading slashes, then reject an empty result, full HTTP
		    or HTTPS URL, or parent-directory markers.

		Args:
		    asset_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.

		Returns:
		    str: Accepted normalized relative path.

		Raises:
		    Error: Raised when the path is missing or fails the path checks.
		"""
		try:
			throw_if( 'asset_path', asset_path )
			
			value = asset_path.strip( ).lstrip( '/' )
			
			if not value:
				raise ValueError( "Argument 'asset_path' cannot be empty!" )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'asset_path must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'asset_path cannot contain parent-directory markers.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'normalize_asset_path( self, asset_path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_page( self, page: int ) -> int:
		"""Validate a search page.

		Purpose:
		    Require a one-based page number of at least one.

		Args:
		    page (int): One-based search page.

		Returns:
		    int: Original accepted page number.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'page', page )

			if page < 1:
				raise ValueError( 'page must be greater than or equal to 1.' )
			
			return page
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'validate_page( self, page: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_page_size( self, page_size: int ) -> int:
		"""Validate the search page size.

		Purpose:
		    Enforce the inclusive range from 1 through 500.

		Args:
		    page_size (int): Requested page size from 1 through 500.

		Returns:
		    int: Original accepted page size.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'page_size', page_size )

			if page_size < 1 or page_size > 500:
				raise ValueError( 'page_size must be between 1 and 500.' )
			
			return page_size
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'validate_page_size( self, page_size: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def fetch_catalog( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve WorldPop REST catalog content.

		Purpose:
		    Request the REST base URL without query parameters and decode JSON or retain the
		    response under html.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'catalog'
			throw_if( 'time', time )
			self.timeout = time
			self.url = self.base_url
			self.params = { }
			
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'html': self.response.text }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'fetch_catalog( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def search_catalog( self, query: str='', page: int=1, page_size: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search the configured WorldPop catalog.

		Purpose:
		    Validate nonblank search text and pagination, request the REST search path with q, page,
		    and page_size, and decode JSON or retain text.

		Args:
		    query (str): Nonblank catalog search text. Defaults to an empty string, which must be
		        replaced.
		    page (int): One-based page number. Defaults to 1.
		    page_size (int): Page size from 1 through 500. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'query', query )

			
			self.mode = 'search'
			self.query_text = query
			self.page = self.validate_page( page )
			self.page_size = self.validate_page_size( page_size )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/search'
			self.params = { 'q': self.query_text, 'page': self.page, 'page_size': self.page_size }
			
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'text': self.response.text }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = "search_catalog( self, query: str='', page: int=1, page_size: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_raster_metadata( self, asset_path: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve content for a relative raster asset path.

		Purpose:
		    Normalize the asset path, append it to the REST base URL, and request JSON or textual
		    metadata.

		Args:
		    asset_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'raster_metadata'
			self.asset_path = self.normalize_asset_path( asset_path )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/{self.asset_path}'
			self.params = { }
			
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			
			if 'application/json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = { 'text': self.response.text }
			
			self.result = { 'mode': self.mode, 'url': self.response.url, 'params': self.params,
					'data': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'fetch_raster_metadata( self, asset_path: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='catalog', query: str='', asset_path: str='', page: int=1, page_size: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a WorldPop retrieval operation.

		Purpose:
		    Select catalog, search, or raster_metadata mode and forward relevant arguments.

		Args:
		    mode (str): catalog, search, or raster_metadata. Defaults to 'catalog'.
		    query (str): Search text. Defaults to an empty string; required in search mode.
		    asset_path (str): Nonblank relative path; surrounding whitespace and leading slashes are
		        removed. Full lowercase HTTP(S) URLs and parent-directory markers are rejected.
		        Defaults to an empty string; required in raster_metadata mode.
		    page (int): One-based search page. Defaults to 1.
		    page_size (int): Search page size from 1 through 500. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, response URL, submitted parameters, and decoded JSON or textual
		        response dictionary.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			if self.mode == 'catalog':
				return self.fetch_catalog( time=time )
			
			if self.mode == 'search':
				return self.search_catalog( query=query, page=page, page_size=page_size,
					time=time )
			
			if self.mode == 'raster_metadata':
				return self.fetch_raster_metadata( asset_path=asset_path, time=time )
			
			raise ValueError( "Unsupported World Population mode. Use 'catalog', 'search', "
			                  "or 'raster_metadata'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = "fetch( self, mode: str='catalog', query: str='', asset_path: str='', page: int=1, page_size: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'WorldPopulation'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class Wonder( Fetcher ):
	"""CDC WONDER XML request adapter.

	Purpose:
	    Build local XML request templates for CDC WONDER datasets and submit caller-supplied XML
	    as form data to the configured datarequest endpoint.

	Attributes:
	    base_url (Optional[str]): CDC WONDER datarequest endpoint base URL.
	    mode (Optional[str]): Selected retrieval operation.
	    dataset_id (Optional[str]): Selected CDC WONDER database identifier.
	    request_xml (Optional[str]): Generated template or submitted XML request string.
	    params (Optional[Dict[str, Any]]): Latest template metadata or submitted form parameters.
	    payload (Optional[Any]): Generated template metadata or raw XML response dictionary.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	dataset_id: Optional[ str ]
	request_xml: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the CDC WONDER datarequest URL, D76 dataset default, metadata_template mode,
		    form-encoded request headers, and a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://wonder.cdc.gov/controller/datarequest'
		self.mode = 'metadata_template'
		self.dataset_id = 'D76'
		self.request_xml = ''
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/xml, text/xml, text/plain',
				'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': self.agents, }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'dataset_id', 'request_xml', 'params', 'payload', 'result',
				'response', 'url', 'timeout', 'agents', 'headers', 'normalize_dataset_id',
				'build_template', 'fetch_template', 'submit_query', 'fetch', 'create_schema' ]
	
	def normalize_dataset_id( self, dataset_id: str ) -> str:
		"""Validate a CDC WONDER dataset identifier.

		Purpose:
		    Require uppercase D followed by one to four digits; preserve the accepted identifier.

		Args:
		    dataset_id (str): CDC WONDER identifier matching uppercase D followed by one to four
		        digits.

		Returns:
		    str: Original accepted dataset identifier.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'dataset_id', dataset_id )
			
			value = dataset_id
			
			if not re.fullmatch( r'D\d{1,4}', value ):
				raise ValueError( 'CDC WONDER Dataset ID must use the format D followed by '
				                  'digits, '
				                  'such as D76.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = 'normalize_dataset_id( self, dataset_id: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_template( self, dataset_id: str='D76' ) -> str:
		"""Build a local XML request template.

		Purpose:
		    Validate the dataset identifier and construct XML containing data-use acceptance, B_1,
		    M_1, and totals parameters. No network request is issued.

		Args:
		    dataset_id (str): CDC WONDER identifier matching uppercase D followed by one to four
		        digits. Defaults to 'D76'.

		Returns:
		    str: Generated XML request template.

		Raises:
		    Error: Raised when dataset validation or template construction fails.
		"""
		try:
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			self.request_xml = (f'<request>\n'
			                    f'  <request-parameters>\n'
			                    f'    <parameter>\n'
			                    f'      <name>accept_datause_restrictions</name>\n'
			                    f'      <value>true</value>\n'
			                    f'    </parameter>\n'
			                    f'    <parameter>\n'
			                    f'      <name>B_1</name>\n'
			                    f'      <value>{self.dataset_id}.V1</value>\n'
			                    f'    </parameter>\n'
			                    f'    <parameter>\n'
			                    f'      <name>M_1</name>\n'
			                    f'      <value>{self.dataset_id}.M1</value>\n'
			                    f'    </parameter>\n'
			                    f'    <parameter>\n'
			                    f'      <name>O_show_totals</name>\n'
			                    f'      <value>true</value>\n'
			                    f'    </parameter>\n'
			                    f'  </request-parameters>\n'
			                    f'</request>')
			
			return self.request_xml
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = "build_template( self, dataset_id: str='D76' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def fetch_template( self, dataset_id: str='D76' ) -> Dict[ str, Any ] | None:
		"""Package a local CDC WONDER request template.

		Purpose:
		    Build the dataset URL, generate the XML template, and package it with notes about POST
		    submission. No metadata is retrieved from the provider.

		Args:
		    dataset_id (str): CDC WONDER identifier matching uppercase D followed by one to four
		        digits. Defaults to 'D76'.

		Returns:
		    Dict[str, Any]: Template mode, endpoint URL, dataset identifier, generated XML, and
		        submission notes.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'metadata_template'
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			self.url = f'{self.base_url}/{self.dataset_id}'
			self.params = { 'dataset_id': self.dataset_id }
			self.payload = { 'dataset_id': self.dataset_id,
					'request_xml': self.build_template( dataset_id=self.dataset_id ),
					'notes': ('CDC WONDER expects POST requests to '
					          'https://wonder.cdc.gov/controller/datarequest/[database ID] '
					          'with a request_xml parameter and acceptance of data-use '
					          'restrictions.'), }
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = "fetch_template( self, dataset_id: str='D76' ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def submit_query( self, dataset_id: str, request_xml: str, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Submit a CDC WONDER XML query.

		Purpose:
		    Validate the dataset and nonblank XML, POST form data with acceptance of data-use
		    restrictions set to true, and retain the raw response text without parsing XML.

		Args:
		    dataset_id (str): CDC WONDER identifier matching uppercase D followed by one to four
		        digits.
		    request_xml (str): Nonblank XML request body supplied by the caller.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Query mode, endpoint URL, dataset identifier and acceptance indicator,
		        and raw response text under data.xml.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'request_xml', request_xml )

			
			self.mode = 'query_xml'
			self.dataset_id = self.normalize_dataset_id( dataset_id )
			self.request_xml = request_xml
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/{self.dataset_id}'
			self.params = { 'request_xml': self.request_xml,
					'accept_datause_restrictions': 'true', }
			
			
			self.response = requests.post( url=self.url, data=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = { 'xml': self.response.text, }
			self.result = { 'mode': self.mode, 'url': self.url,
					'params': { 'dataset_id': self.dataset_id,
							'accept_datause_restrictions': 'true', }, 'data': self.payload, }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = 'submit_query( self, dataset_id: str, request_xml: str, time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='metadata_template', dataset_id: str='D76', request_xml: str = '', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a CDC WONDER operation.

		Purpose:
		    Select local template generation or XML query submission; timeout applies only to query
		    submission.

		Args:
		    mode (str): metadata_template or query_xml. Defaults to 'metadata_template'.
		    dataset_id (str): CDC WONDER identifier matching uppercase D followed by one to four
		        digits. Defaults to 'D76'.
		    request_xml (str): Nonblank XML request body supplied by the caller. Defaults to an
		        empty string; required in query_xml mode.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Local template metadata or submitted query response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'metadata_template':
				return self.fetch_template( dataset_id=dataset_id )
			
			if self.mode == 'query_xml':
				return self.submit_query( dataset_id=dataset_id, request_xml=request_xml,
					time=time )
			
			raise ValueError( "Unsupported CDC WONDER mode. Use 'metadata_template' or "
			                  "'query_xml'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = "fetch( self, mode: str='metadata_template', dataset_id: str='D76', request_xml: str = '', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wonder'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class USGSEarthquakes( Fetcher ):
	"""USGS earthquake GeoJSON feed and event search adapter.

	Purpose:
	    Retrieve predefined earthquake summary feeds or filtered FDSN event searches, shape
	    GeoJSON features into rows, and package raw data and summary statistics.

	Attributes:
	    feed_url (Optional[str]): Base URL for predefined USGS summary feeds.
	    search_url (Optional[str]): FDSN event query endpoint.
	    mode (Optional[str]): Selected retrieval operation.
	    feed (Optional[str]): Selected summary-feed filename; cleared for search mode.
	    start_date (Optional[str]): Search interval start date or datetime.
	    end_date (Optional[str]): Search interval end date or datetime.
	    min_magnitude (Optional[float]): Minimum event magnitude filter.
	    max_magnitude (Optional[float]): Maximum event magnitude filter.
	    limit (Optional[int]): Requested search event limit.
	    order_by (Optional[str]): Search sort expression.
	    event_type (Optional[str]): Requested event type.
	    latitude (Optional[float]): Optional radial-search center latitude.
	    longitude (Optional[float]): Optional radial-search center longitude.
	    max_radius_km (Optional[float]): Optional radial-search radius in kilometers.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Dict[str, Any]]): Latest successful GeoJSON object.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	feed_url: Optional[ str ]
	search_url: Optional[ str ]
	mode: Optional[ str ]
	feed: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	min_magnitude: Optional[ float ]
	max_magnitude: Optional[ float ]
	limit: Optional[ int ]
	order_by: Optional[ str ]
	event_type: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	max_radius_km: Optional[ float ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure feed and FDSN search URLs, all-day feed, default search order and limit,
		    GeoJSON headers, and a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.feed_url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary'
		self.search_url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
		self.mode = 'feed'
		self.feed = 'all_day.geojson'
		self.start_date = ''
		self.end_date = ''
		self.min_magnitude = None
		self.max_magnitude = None
		self.limit = 25
		self.order_by = 'time'
		self.event_type = 'earthquake'
		self.latitude = None
		self.longitude = None
		self.max_radius_km = None
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/geo+json, application/json',
				'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'feed_url', 'search_url', 'mode', 'feed', 'start_date', 'end_date',
				'min_magnitude', 'max_magnitude', 'limit', 'order_by', 'event_type', 'latitude',
				'longitude', 'max_radius_km', 'params', 'payload', 'result', 'response', 'url',
				'timeout', 'agents', 'headers', 'validate_feed', 'validate_limit',
				'validate_magnitude', 'validate_order_by', 'validate_latitude',
				'validate_longitude', 'validate_radius', 'to_iso_date', 'epoch_millis_to_iso',
				'shape_feature_rows', 'summarize_features', 'package_response', 'fetch_feed',
				'fetch_search', 'fetch', 'create_schema' ]
	
	def validate_feed( self, feed: str ) -> str:
		"""Validate a predefined feed filename.

		Purpose:
		    Accept only the predefined combinations of magnitude/category prefix and hour, day,
		    week, or month periods.

		Args:
		    feed (str): GeoJSON feed filename with prefix all, significant, 4.5, 2.5, or 1.0 and
		        period hour, day, week, or month.

		Returns:
		    str: Original accepted feed filename.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'feed', feed )
			
			value = feed
			allowed = { 'all_hour.geojson', 'all_day.geojson', 'all_week.geojson',
					'all_month.geojson', 'significant_hour.geojson', 'significant_day.geojson',
					'significant_week.geojson', 'significant_month.geojson', '4.5_hour.geojson',
					'4.5_day.geojson', '4.5_week.geojson', '4.5_month.geojson', '2.5_hour.geojson',
					'2.5_day.geojson', '2.5_week.geojson', '2.5_month.geojson', '1.0_hour.geojson',
					'1.0_day.geojson', '1.0_week.geojson', '1.0_month.geojson' }
			
			if value not in allowed:
				raise ValueError( f'Unsupported USGS earthquake feed: {value}' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_feed( self, feed: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the search event limit.

		Purpose:
		    Enforce the inclusive range from 1 through 20000.

		Args:
		    limit (int): Requested event count from 1 through 20000.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'limit', limit )

			if limit < 1 or limit > 20000:
				raise ValueError( 'limit must be between 1 and 20000.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_magnitude( self, name: str, value: float ) -> float:
		"""Validate an earthquake magnitude filter.

		Purpose:
		    Require a finite magnitude from zero through ten.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (float): Finite magnitude from 0 through 10.

		Returns:
		    float: Original accepted magnitude.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'name', name )
			throw_if( name, value )
			
			number = value
			if not math.isfinite( number ):
				raise ValueError( 'validate_magnitude must be finite.' )
			if number < 0.0 or number > 10.0:
				raise ValueError( f'{name} must be between 0.0 and 10.0.' )
			
			return number
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_magnitude( self, name: str, value: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_order_by( self, order_by: str ) -> str:
		"""Validate event ordering.

		Purpose:
		    Accept only supported time or magnitude sort expressions.

		Args:
		    order_by (str): time, time-asc, magnitude, or magnitude-asc.

		Returns:
		    str: Original accepted ordering expression.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'order_by', order_by )
			
			value = order_by
			allowed = { 'time', 'time-asc', 'magnitude', 'magnitude-asc' }
			
			if value not in allowed:
				raise ValueError( "order_by must be 'time', 'time-asc', 'magnitude', or "
				                  "'magnitude-asc'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_order_by( self, order_by: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_latitude( self, latitude: float ) -> float:
		"""Validate radial-search latitude.

		Purpose:
		    Require a finite latitude within geographic bounds.

		Args:
		    latitude (float): Finite latitude from -90 through 90 degrees.

		Returns:
		    float: Original accepted latitude.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'latitude', latitude )
			
			value = latitude
			if not math.isfinite( value ):
				raise ValueError( 'validate_latitude must be finite.' )
			if value < -90.0 or value > 90.0:
				raise ValueError( 'latitude must be between -90 and 90.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_latitude( self, latitude: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_longitude( self, longitude: float ) -> float:
		"""Validate radial-search longitude.

		Purpose:
		    Require a finite longitude within geographic bounds.

		Args:
		    longitude (float): Finite longitude from -180 through 180 degrees.

		Returns:
		    float: Original accepted longitude.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'longitude', longitude )
			
			value = longitude
			if not math.isfinite( value ):
				raise ValueError( 'validate_longitude must be finite.' )
			if value < -180.0 or value > 180.0:
				raise ValueError( 'longitude must be between -180 and 180.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_longitude( self, longitude: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_radius( self, radius: float ) -> float:
		"""Validate radial-search distance.

		Purpose:
		    Require a finite radius from zero through 20001.6 kilometers.

		Args:
		    radius (float): Finite radius from 0 through 20001.6 km.

		Returns:
		    float: Original accepted radius.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'max_radius_km', radius )
			
			value = radius
			if not math.isfinite( value ):
				raise ValueError( 'validate_radius must be finite.' )
			if value < 0.0 or value > 20001.6:
				raise ValueError( 'max_radius_km must be between 0.0 and 20001.6.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'validate_radius( self, radius: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def to_iso_date( self, value: str ) -> str:
		"""Trim a date argument.

		Purpose:
		    Check for a missing value and strip surrounding whitespace; ISO parsing occurs in
		    fetch_search.

		Args:
		    value (str): Nonblank date or datetime string.

		Returns:
		    str: Trimmed input string.

		Raises:
		    Error: Raised when validation or string processing fails.
		"""
		try:
			throw_if( 'date', value )
			return value.strip( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'to_iso_date( self, value: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def epoch_millis_to_iso( self, value: Any ) -> str:
		"""Convert an epoch timestamp to UTC ISO text.

		Purpose:
		    Convert milliseconds since the Unix epoch to seconds and produce a timezone-aware UTC
		    ISO string; return an empty string on missing or invalid input.

		Args:
		    value (Any): Epoch milliseconds convertible to float, or None.

		Returns:
		    str: UTC ISO 8601 datetime or an empty string.
		"""
		try:
			if value is None:
				return ''
			
			return dt.datetime.fromtimestamp( float( value ) / 1000.0, tz=dt.timezone.utc
			).isoformat( )
		
		except Exception:
			return ''
	
	def shape_feature_rows( self, features: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape earthquake GeoJSON features.

		Purpose:
		    Extract event properties, longitude, latitude, depth, and UTC timestamps into
		    display-oriented dictionaries.

		Args:
		    features (List[Dict[str, Any]]): GeoJSON feature dictionaries.

		Returns:
		    List[Dict[str, Any]]: Event rows containing identifiers, magnitude, location, times,
		        alert, status, tsunami flag, felt reports, depth, coordinates, event type, and URL.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for feature in features or [ ]:
				properties = feature.get( 'properties', { } ) or { }
				geometry = feature.get( 'geometry', { } ) or { }
				coordinates = geometry.get( 'coordinates', [ ] ) or [ ]
				
				longitude = coordinates[ 0 ] if len( coordinates ) > 0 else None
				latitude = coordinates[ 1 ] if len( coordinates ) > 1 else None
				depth = coordinates[ 2 ] if len( coordinates ) > 2 else None
				event_time = self.epoch_millis_to_iso( properties.get( 'time' ) )
				updated_time = self.epoch_millis_to_iso( properties.get( 'updated' ) )
				
				rows.append( { 'Id': feature.get( 'id', '' ),
						'Magnitude': properties.get( 'mag', None ),
						'Place': properties.get( 'place', '' ), 'Time': event_time,
						'Updated': updated_time, 'Alert': properties.get( 'alert', '' ),
						'Status': properties.get( 'status', '' ),
						'Tsunami': properties.get( 'tsunami', None ),
						'Felt Reports': properties.get( 'felt', None ), 'Depth (km)': depth,
						'Latitude': latitude, 'Longitude': longitude,
						'Event Type': properties.get( 'type', '' ),
						'URL': properties.get( 'url', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'shape_feature_rows( self, features: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_features( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped earthquake rows.

		Purpose:
		    Count rows, find the greatest convertible magnitude and associated place, and retain the
		    first nonempty event time encountered.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, maximum magnitude, strongest place, and most_recent (first
		        nonempty event time, not necessarily chronological maximum).

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			max_magnitude = None
			strongest_place = ''
			most_recent = ''
			
			for row in rows or [ ]:
				magnitude = row.get( 'Magnitude', None )
				
				if magnitude is not None:
					try:
						if max_magnitude is None or float( magnitude ) > float( max_magnitude ):
							max_magnitude = float( magnitude )
							strongest_place = str( row.get( 'Place', '' ) )
					except Exception:
						pass
				
				if not most_recent and row.get( 'Time', '' ):
					most_recent = str( row.get( 'Time', '' ) )
			
			return { 'count': count, 'max_magnitude': max_magnitude,
					'strongest_place': strongest_place, 'most_recent': most_recent }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'summarize_features( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self ) -> Dict[ str, Any ]:
		"""Package the retained GeoJSON response.

		Purpose:
		    Extract metadata and features, shape event rows, calculate the summary, and retain raw
		    GeoJSON.

		Returns:
		    Dict[str, Any]: Mode, feed, URL, parameters, title, generation timestamp, count,
		        bounding box, summary, shaped rows, and raw GeoJSON.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			metadata = self.payload.get( 'metadata', { } ) or { }
			features = self.payload.get( 'features', [ ] ) or [ ]
			rows = self.shape_feature_rows( features )
			
			self.result = { 'mode': self.mode, 'feed': self.feed if self.mode == 'feed' else '',
					'url': self.url, 'params': self.params, 'title': metadata.get( 'title', '' ),
					'generated': metadata.get( 'generated', None ),
					'count': metadata.get( 'count', len( rows ) ),
					'bbox': self.payload.get( 'bbox', [ ] ),
					'summary': self.summarize_features( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'package_response( self ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_feed( self, feed: str='all_day.geojson', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve a predefined earthquake feed.

		Purpose:
		    Validate the feed filename, request the summary feed, require a successful GeoJSON
		    object with a features list, and package the result.

		Args:
		    feed (str): GeoJSON feed filename with prefix all, significant, 4.5, 2.5, or 1.0 and
		        period hour, day, week, or month. Defaults to 'all_day.geojson'.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, feed, URL, parameters, title, generation timestamp, count,
		        bounding box, summary, shaped rows, and raw GeoJSON.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'feed'
			self.feed = self.validate_feed( feed )
			throw_if( 'time', time )
			self.timeout = time
			self.params = { }
			self.url = f'{self.feed_url}/{self.feed}'
			
			
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ) or self.payload.get( 'error' ):
				raise ValueError( 'USGS earthquake response must be a successful GeoJSON object.' )
			if not isinstance( self.payload.get( 'features' ), list ):
				raise ValueError( 'USGS earthquake response requires a features array.' )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = "fetch_feed( self, feed: str='all_day.geojson', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_search( self, start_date: str, end_date: str, min_magnitude: float=1.0, max_magnitude: float=10.0, limit: int=25, order_by: str='time', event_type: str = 'earthquake', latitude: float | None = None, longitude: float | None = None, max_radius_km: float | None = None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search USGS earthquake events.

		Purpose:
		    Parse and compare date bounds, validate magnitude and ordering filters, require paired
		    radial coordinates when supplied, request GeoJSON, and package its features.

		Args:
		    start_date (str): Nonblank ISO 8601 start date or datetime.
		    end_date (str): Nonblank ISO 8601 end date or datetime, not earlier than start_date;
		        both dates must consistently include or omit timezone information.
		    min_magnitude (float): Finite minimum magnitude from 0 through 10. Defaults to 1.0.
		    max_magnitude (float): Finite maximum magnitude from 0 through 10, at least
		        min_magnitude. Defaults to 10.0.
		    limit (int): Requested event count from 1 through 20000. Defaults to 25.
		    order_by (str): time, time-asc, magnitude, or magnitude-asc. Defaults to 'time'.
		    event_type (str): Nonblank event type passed to USGS. Defaults to 'earthquake'.
		    latitude (float | None): Finite radial-search latitude from -90 through 90 degrees;
		        supply with longitude. Defaults to None.
		    longitude (float | None): Finite radial-search longitude from -180 through 180 degrees;
		        supply with latitude. Defaults to None.
		    max_radius_km (float | None): Finite radial-search radius from 0 through 20001.6 km;
		        requires both coordinates. Defaults to None.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, feed, URL, parameters, title, generation timestamp, count,
		        bounding box, summary, shaped rows, and raw GeoJSON.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'event_type', event_type )

			
			self.mode = 'search'
			self.feed = ''
			self.start_date = self.to_iso_date( start_date )
			self.end_date = self.to_iso_date( end_date )
			try:
				start_dt = dt.datetime.fromisoformat( self.start_date.replace( 'Z', '+00:00' ) )
				end_dt = dt.datetime.fromisoformat( self.end_date.replace( 'Z', '+00:00' ) )
			except ValueError as error:
				raise ValueError( 'start_date and end_date must be ISO 8601 dates.' ) from error
			if start_dt.tzinfo is None and end_dt.tzinfo is not None or start_dt.tzinfo is not None and end_dt.tzinfo is None:
				raise ValueError( 'start_date and end_date must use consistent timezones.' )
			if start_dt > end_dt:
				raise ValueError( 'start_date must precede or equal end_date.' )
			self.latitude = None
			self.longitude = None
			self.max_radius_km = None
			self.min_magnitude = self.validate_magnitude( 'min_magnitude', min_magnitude )
			self.max_magnitude = self.validate_magnitude( 'max_magnitude', max_magnitude )
			self.limit = self.validate_limit( limit )
			self.order_by = self.validate_order_by( order_by )
			self.event_type = event_type
			throw_if( 'time', time )
			self.timeout = time
			self.url = self.search_url
			
			if self.min_magnitude > self.max_magnitude:
				raise ValueError( 'min_magnitude must be less than or equal to max_magnitude.' )
			
			
			self.params = { 'format': 'geojson', 'starttime': self.start_date,
					'endtime': self.end_date, 'minmagnitude': self.min_magnitude,
					'maxmagnitude': self.max_magnitude, 'limit': self.limit,
					'orderby': self.order_by, 'eventtype': self.event_type }
			
			if (latitude is None) != (longitude is None):
				raise ValueError( 'latitude and longitude must both be supplied for radial '
				                  'search.' )
			
			if latitude is not None and longitude is not None:
				self.latitude = self.validate_latitude( latitude )
				self.longitude = self.validate_longitude( longitude )
				self.params[ 'latitude' ] = self.latitude
				self.params[ 'longitude' ] = self.longitude
			
			if max_radius_km is not None:
				if latitude is None or longitude is None:
					raise ValueError( 'max_radius_km requires both latitude and longitude.' )
				
				self.max_radius_km = self.validate_radius( max_radius_km )
				self.params[ 'maxradiuskm' ] = self.max_radius_km
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ) or self.payload.get( 'error' ):
				raise ValueError( 'USGS earthquake response must be a successful GeoJSON object.' )
			if not isinstance( self.payload.get( 'features' ), list ):
				raise ValueError( 'USGS earthquake response requires a features array.' )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = "fetch_search( self, start_date: str, end_date: str, min_magnitude: float=1.0, max_magnitude: float=10.0, limit: int=25, order_by: str='time', event_type: str = 'earthquake', latitude: float | None = None, longitude: float | None = None, max_radius_km: float | None = None, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='feed', feed: str='all_day.geojson', start_date: str='', end_date: str='', min_magnitude: float=1.0, max_magnitude: float=10.0, limit: int=25, order_by: str='time', event_type: str='earthquake', latitude: float | None = None, longitude: float | None = None, max_radius_km: float | None = None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an earthquake retrieval operation.

		Purpose:
		    Select feed or search mode and pass the applicable filters.

		Args:
		    mode (str): feed or search. Defaults to 'feed'.
		    feed (str): GeoJSON feed filename with prefix all, significant, 4.5, 2.5, or 1.0 and
		        period hour, day, week, or month. Defaults to 'all_day.geojson'.
		    start_date (str): Nonblank ISO 8601 start date or datetime. Defaults to an empty string;
		        required in search mode.
		    end_date (str): Nonblank ISO 8601 end date or datetime, not earlier than start_date;
		        both dates must consistently include or omit timezone information. Defaults to an
		        empty string; required in search mode.
		    min_magnitude (float): Finite minimum magnitude from 0 through 10. Defaults to 1.0.
		    max_magnitude (float): Finite maximum magnitude from 0 through 10, at least
		        min_magnitude. Defaults to 10.0.
		    limit (int): Requested event count from 1 through 20000. Defaults to 25.
		    order_by (str): time, time-asc, magnitude, or magnitude-asc. Defaults to 'time'.
		    event_type (str): Nonblank event type passed to USGS. Defaults to 'earthquake'.
		    latitude (float | None): Finite radial-search latitude from -90 through 90 degrees;
		        supply with longitude. Defaults to None.
		    longitude (float | None): Finite radial-search longitude from -180 through 180 degrees;
		        supply with latitude. Defaults to None.
		    max_radius_km (float | None): Finite radial-search radius from 0 through 20001.6 km;
		        requires both coordinates. Defaults to None.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, feed, URL, parameters, title, generation timestamp, count,
		        bounding box, summary, shaped rows, and raw GeoJSON.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'feed':
				return self.fetch_feed( feed=feed, time=time )
			
			if self.mode == 'search':
				return self.fetch_search( start_date=start_date, end_date=end_date,
					min_magnitude=min_magnitude, max_magnitude=max_magnitude, limit=limit,
					order_by=order_by, event_type=event_type, latitude=latitude,
					longitude=longitude, max_radius_km=max_radius_km, time=time )
			
			raise ValueError( "Unsupported USGS Earthquakes mode. Use 'feed' or 'search'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = "fetch( self, mode: str='feed', feed: str='all_day.geojson', start_date: str='', end_date: str='', min_magnitude: float=1.0, max_magnitude: float=10.0, limit: int=25, order_by: str='time', event_type: str='earthquake', latitude: float | None = None, longitude: float | None = None, max_radius_km: float | None = None, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSEarthquakes'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class USGSWaterData( Fetcher ):
	"""USGS Water Data OGC API retrieval adapter.

	Purpose:
	    Retrieve monitoring locations, time-series metadata, and latest continuous or daily
	    observations through collection items endpoints, shape records, and package summaries and
	    raw responses.

	Attributes:
	    api_key (Optional[str]): Configured USGS Water Data key, sent in X-API-Key when present.
	    base_url (Optional[str]): OGC API collections base URL.
	    mode (Optional[str]): Selected retrieval operation.
	    collection (Optional[str]): Selected collection identifier.
	    monitoring_location_id (Optional[str]): Monitoring location identifier filter.
	    state_code (Optional[str]): State code filter.
	    county_code (Optional[str]): County code filter.
	    site_type (Optional[str]): Site type filter.
	    parameter_code (Optional[str]): Optional five-digit parameter code filter.
	    limit (Optional[int]): Requested record limit.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded JSON object.
	    result (Optional[Dict[str, Any]]): Latest packaged response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	mode: Optional[ str ]
	collection: Optional[ str ]
	monitoring_location_id: Optional[ str ]
	state_code: Optional[ str ]
	county_code: Optional[ str ]
	site_type: Optional[ str ]
	parameter_code: Optional[ str ]
	limit: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured USGS Water Data key, configure the collections endpoint and JSON
		    headers, and initialize monitoring-locations mode, limit of 25, and twenty-second
		    timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.api_key = cfg.USGS_WATERDATA_API_KEY
		self.base_url = 'https://api.waterdata.usgs.gov/ogcapi/v0/collections'
		self.mode = 'monitoring-locations'
		self.collection = None
		self.monitoring_location_id = ''
		self.state_code = ''
		self.county_code = ''
		self.site_type = ''
		self.parameter_code = ''
		self.limit = 25
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
		
		if self.api_key:
			self.headers[ 'X-API-Key' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'api_key', 'base_url', 'mode', 'collection', 'monitoring_location_id',
				'state_code', 'county_code', 'site_type', 'parameter_code', 'limit', 'params',
				'payload', 'result', 'response', 'url', 'timeout', 'agents', 'headers',
				'validate_collection', 'validate_limit', 'validate_parameter_code',
				'coalesce_records', 'shape_monitoring_locations', 'shape_time_series_metadata',
				'shape_latest_values', 'summarize_rows', 'package_response', 'request',
				'fetch_monitoring_locations', 'fetch_time_series_metadata',
				'fetch_latest_continuous', 'fetch_latest_daily', 'fetch', 'create_schema' ]
	
	def validate_collection( self, collection: str ) -> str:
		"""Validate an OGC collection identifier.

		Purpose:
		    Accept only the four supported monitoring and observation collection identifiers.

		Args:
		    collection (str): monitoring-locations, time-series-metadata, latest-continuous, or
		        latest-daily.

		Returns:
		    str: Original accepted collection identifier.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'collection', collection )
			
			value = collection
			allowed = { 'monitoring-locations', 'time-series-metadata', 'latest-continuous',
					'latest-daily' }
			
			if value not in allowed:
				raise ValueError( 'Unsupported USGS Water Data collection. Use '
				                  'monitoring-locations, time-series-metadata, latest-continuous, '
				                  'or latest-daily.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'validate_collection( self, collection: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the requested record limit.

		Purpose:
		    Enforce the inclusive range from 1 through 10000.

		Args:
		    limit (int): Requested row limit from 1 through 10000.

		Returns:
		    int: Original accepted record limit.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'limit', limit )

			if limit < 1 or limit > 10000:
				raise ValueError( 'limit must be between 1 and 10000.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_parameter_code( self, parameter_code: str ) -> str:
		"""Validate an optional parameter code.

		Purpose:
		    Accept a blank value or a truthy string matching exactly five digits; return the
		    original value.

		Args:
		    parameter_code (str): Optional five-digit USGS parameter code.

		Returns:
		    str: Original accepted parameter code.

		Raises:
		    Error: Raised when a nonblank code does not match the required format.
		"""
		try:
			value = parameter_code
			
			if value and not re.fullmatch( r'\d{5}', value ):
				raise ValueError( 'parameter_code must be a 5-digit USGS parameter code.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'validate_parameter_code( self, parameter_code: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract dictionary records from a response.

		Purpose:
		    Filter list payloads to dictionaries, select the first list under supported collection
		    keys in dictionary payloads, or wrap a standalone dictionary; return an empty list for
		    other payloads.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Extracted dictionary records.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			if payload is None:
				return [ ]
			
			if isinstance( payload, list ):
				return [ item for item in payload if isinstance( item, dict ) ]
			
			if not isinstance( payload, dict ):
				return [ ]
			
			for key in [ 'features', 'items', 'results', 'data', 'value', 'timeSeries',
					'observations' ]:
				value = payload.get( key, None )
				
				if isinstance( value, list ):
					return [ item for item in value if isinstance( item, dict ) ]
			
			return [ payload ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_monitoring_locations( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape monitoring location records.

		Purpose:
		    Read nested properties or the record itself, prefer geometry coordinates, and fall back
		    to alternate provider property names for location metadata.

		Args:
		    records (List[Dict[str, Any]]): Provider records, optionally containing nested
		        properties and geometry.

		Returns:
		    List[Dict[str, Any]]: Location ID, name, site type, state and county codes, latitude,
		        and longitude rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				properties = item.get( 'properties', { } ) or item
				geometry = item.get( 'geometry', { } ) or { }
				coordinates = geometry.get( 'coordinates', [ ] ) or [ ]
				
				longitude = coordinates[ 0 ] if len( coordinates ) > 0 else (
						properties.get( 'longitude', None ) or properties.get( 'dec_long_va',
					None ))
				latitude = coordinates[ 1 ] if len( coordinates ) > 1 else (
						properties.get( 'latitude', None ) or properties.get( 'dec_lat_va', None ))
				
				rows.append( { 'Monitoring Location ID': (
						properties.get( 'monitoring_location_id', None ) or properties.get(
					'monitoringLocationIdentifier', None ) or properties.get( 'site_no', '' )),
						'Name': (
								properties.get( 'monitoring_location_name', None ) or
								properties.get( 'monitoringLocationName', None ) or
								properties.get( 'station_nm', '' )),
						'Site Type': (
								properties.get( 'site_type', None ) or properties.get( 'siteType',
							None ) or properties.get( 'site_tp_cd', '' )),
						'State Code': (
								properties.get( 'state_code', None ) or properties.get(
							'stateCode', None ) or properties.get( 'state_cd', '' )),
						'County Code': (
								properties.get( 'county_code', None ) or properties.get(
							'countyCode', None ) or properties.get( 'county_cd', '' )),
						'Latitude': latitude, 'Longitude': longitude } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'shape_monitoring_locations( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_time_series_metadata( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape time-series metadata records.

		Purpose:
		    Extract location, parameter, statistic, unit, and time bounds using alternate provider
		    field names.

		Args:
		    records (List[Dict[str, Any]]): Provider records, optionally containing nested
		        properties and geometry.

		Returns:
		    List[Dict[str, Any]]: Location identifiers and names, parameter codes and names,
		        statistic IDs, units, and begin/end times.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				properties = item.get( 'properties', { } ) or item
				
				rows.append( { 'Monitoring Location ID': (
						properties.get( 'monitoring_location_id', None ) or properties.get(
					'monitoringLocationIdentifier', None ) or properties.get( 'site_no', '' )),
						'Name': (
								properties.get( 'monitoring_location_name', None ) or
								properties.get( 'monitoringLocationName', None ) or
								properties.get( 'site_name', '' )),
						'Parameter Code': (
								properties.get( 'parameter_code', None ) or properties.get(
							'parameterCode', '' )),
						'Parameter Name': (
								properties.get( 'parameter_name', None ) or properties.get(
							'parameterName', '' )),
						'Statistic ID': (
								properties.get( 'statistic_id', None ) or properties.get(
							'statisticId', '' )),
						'Unit': (
								properties.get( 'unit_of_measure', None ) or properties.get(
							'unitOfMeasure', '' )),
						'Begin Time': (
								properties.get( 'begin_time', None ) or properties.get(
							'beginTime', '' )),
						'End Time': (
								properties.get( 'end_time', None ) or properties.get( 'endTime',
							'' )) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'shape_time_series_metadata( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_latest_values( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape latest observation records.

		Purpose:
		    Extract location, parameter, statistic, measured value, unit, time, and approval status
		    using alternate provider field names and truthiness-based fallbacks.

		Args:
		    records (List[Dict[str, Any]]): Provider records, optionally containing nested
		        properties and geometry.

		Returns:
		    List[Dict[str, Any]]: Display-oriented latest observation rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				properties = item.get( 'properties', { } ) or item
				
				rows.append( { 'Monitoring Location ID': (
						properties.get( 'monitoring_location_id', None ) or properties.get(
					'monitoringLocationIdentifier', None ) or properties.get( 'site_no', '' )),
						'Name': (
								properties.get( 'monitoring_location_name', None ) or
								properties.get( 'monitoringLocationName', None ) or
								properties.get( 'site_name', '' )),
						'Parameter Code': (
								properties.get( 'parameter_code', None ) or properties.get(
							'parameterCode', '' )),
						'Parameter Name': (
								properties.get( 'parameter_name', None ) or properties.get(
							'parameterName', '' )),
						'Statistic ID': (
								properties.get( 'statistic_id', None ) or properties.get(
							'statisticId', '' )),
						'Value': (
								properties.get( 'result_value', None ) or properties.get( 'value',
							None ) or properties.get( 'primary_value', '' )),
						'Unit': (
								properties.get( 'unit_of_measure', None ) or properties.get(
							'unitOfMeasure', '' )),
						'Time': (
								properties.get( 'time', None ) or properties.get( 'result_time',
							None ) or properties.get( 'resultTime', '' )),
						'Approval Status': (
								properties.get( 'approval_status', None ) or properties.get(
							'approvalStatus', '' )) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'shape_latest_values( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped water data rows.

		Purpose:
		    Count records and retain the first row's location, parameter, value, and time as
		    strings, using empty strings when missing or falsey.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_location, first_parameter, first_value, and first_time.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_location = ''
			first_parameter = ''
			first_value = ''
			first_time = ''
			
			if rows:
				first_location = str( rows[ 0 ].get( 'Name', '' ) or rows[
					0 ].get( 'Monitoring Location ID', '' ) or '' )
				first_parameter = str( rows[ 0 ].get( 'Parameter Name', '' ) or '' )
				first_value = str( rows[ 0 ].get( 'Value', '' ) or '' )
				first_time = str(
					rows[ 0 ].get( 'Time', '' ) or rows[ 0 ].get( 'Begin Time', '' ) or '' )
			
			return { 'count': count, 'first_location': first_location,
					'first_parameter': first_parameter, 'first_value': first_value,
					'first_time': first_time }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped water data.

		Purpose:
		    Combine row summaries, retained mode and request metadata, shaped rows, and raw JSON.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, collection: str, params: Dict[ str, Any ], time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Execute a collection items request.

		Purpose:
		    Validate collection and parameters, omit None and blank string values, attach the
		    configured key header, request JSON, and reject non-object payloads and recognized
		    provider error responses.

		Args:
		    collection (str): monitoring-locations, time-series-metadata, latest-continuous, or
		        latest-daily.
		    params (Dict[str, Any]): Nonempty collection query parameter dictionary.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters, and raw JSON object.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'collection', collection )
			throw_if( 'params', params )

			
			if not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			self.collection = self.validate_collection( collection )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/{self.collection}/items'
			self.params = { }
			
			
			for key, value in params.items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			if self.api_key:
				self.headers[ 'X-API-Key' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'USGS Water Data response must be a JSON object.' )
			if self.payload.get( 'error' ) or self.payload.get( 'type' ) == 'https://www.opengis.net/def/exceptions/ogcapi-features-1/1.0':
				raise ValueError( str( self.payload.get( 'detail', self.payload.get( 'error', 'USGS Water Data API request failed.' ) ) ) )
			if isinstance( self.payload.get( 'status' ), int ) and self.payload[ 'status' ] >= 400:
				raise ValueError( 'USGS Water Data API returned an error status.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'request( self, collection: str, params: Dict[ str, Any ], time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_monitoring_locations( self, monitoring_location_id: str='', state_code: str='', county_code: str='', site_type: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve monitoring locations.

		Purpose:
		    Request the monitoring-locations collection using optional geographic and site filters,
		    extract records, and shape location metadata.

		Args:
		    monitoring_location_id (str): Optional monitoring location identifier. Defaults to an
		        empty string.
		    state_code (str): Optional state code. Defaults to an empty string.
		    county_code (str): Optional county code. Defaults to an empty string.
		    site_type (str): Optional site type. Defaults to an empty string.
		    limit (int): Requested row limit from 1 through 10000. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'limit', limit )

			
			self.mode = 'monitoring-locations'
			self.monitoring_location_id = monitoring_location_id
			self.state_code = state_code
			self.county_code = county_code
			self.site_type = site_type
			self.limit = self.validate_limit( limit )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( collection='monitoring-locations', params={
					'monitoring_location_id': self.monitoring_location_id,
					'state_code': self.state_code, 'county_code': self.county_code,
					'site_type': self.site_type, 'limit': self.limit }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_monitoring_locations( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = "fetch_monitoring_locations( self, monitoring_location_id: str='', state_code: str='', county_code: str='', site_type: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_time_series_metadata( self, monitoring_location_id: str='', parameter_code: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve time-series metadata.

		Purpose:
		    Request the time-series-metadata collection using a required monitoring location
		    identifier, optional parameter code and validated limit, then shape time-series metadata
		    records.

		Args:
		    monitoring_location_id (str): Monitoring location identifier. Defaults to an empty
		        string; required for time-series metadata and latest-value modes.
		    parameter_code (str): Optional five-digit USGS parameter code. Defaults to an empty
		        string.
		    limit (int): Requested row limit from 1 through 10000. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'monitoring_location_id', monitoring_location_id )
			throw_if( 'limit', limit )

			
			self.mode = 'time-series-metadata'
			self.monitoring_location_id = monitoring_location_id
			self.parameter_code = self.validate_parameter_code( parameter_code )
			self.limit = self.validate_limit( limit )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( collection='time-series-metadata', params={
					'monitoring_location_id': self.monitoring_location_id,
					'parameter_code': self.parameter_code,
					'limit': self.limit }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_time_series_metadata( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = "fetch_time_series_metadata( self, monitoring_location_id: str='', parameter_code: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_latest_continuous( self, monitoring_location_id: str='', parameter_code: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve latest continuous observations.

		Purpose:
		    Request the latest-continuous collection using a required monitoring location
		    identifier, optional parameter code and validated limit, then shape latest observation
		    records.

		Args:
		    monitoring_location_id (str): Monitoring location identifier. Defaults to an empty
		        string; required for time-series metadata and latest-value modes.
		    parameter_code (str): Optional five-digit USGS parameter code. Defaults to an empty
		        string.
		    limit (int): Requested row limit from 1 through 10000. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'monitoring_location_id', monitoring_location_id )
			throw_if( 'limit', limit )

			
			self.mode = 'latest-continuous'
			self.monitoring_location_id = monitoring_location_id
			self.parameter_code = self.validate_parameter_code( parameter_code )
			self.limit = self.validate_limit( limit )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( collection='latest-continuous', params={
					'monitoring_location_id': self.monitoring_location_id,
					'parameter_code': self.parameter_code,
					'limit': self.limit }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_latest_values( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = "fetch_latest_continuous( self, monitoring_location_id: str='', parameter_code: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_latest_daily( self, monitoring_location_id: str='', parameter_code: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve latest daily observations.

		Purpose:
		    Request the latest-daily collection using a required monitoring location identifier,
		    optional parameter code and validated limit, then shape latest observation records.

		Args:
		    monitoring_location_id (str): Monitoring location identifier. Defaults to an empty
		        string; required for time-series metadata and latest-value modes.
		    parameter_code (str): Optional five-digit USGS parameter code. Defaults to an empty
		        string.
		    limit (int): Requested row limit from 1 through 10000. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'monitoring_location_id', monitoring_location_id )
			throw_if( 'limit', limit )

			
			self.mode = 'latest-daily'
			self.monitoring_location_id = monitoring_location_id
			self.parameter_code = self.validate_parameter_code( parameter_code )
			self.limit = self.validate_limit( limit )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( collection='latest-daily', params={
					'monitoring_location_id': self.monitoring_location_id,
					'parameter_code': self.parameter_code,
					'limit': self.limit }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_latest_values( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = "fetch_latest_daily( self, monitoring_location_id: str='', parameter_code: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='monitoring-locations', monitoring_location_id: str='', state_code: str='', county_code: str='', site_type: str='', parameter_code: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a USGS Water Data retrieval operation.

		Purpose:
		    Select a supported collection and forward the applicable location, geographic,
		    parameter, and pagination filters.

		Args:
		    mode (str): monitoring-locations, time-series-metadata, latest-continuous, or
		        latest-daily. Defaults to 'monitoring-locations'.
		    monitoring_location_id (str): Monitoring location identifier. Defaults to an empty
		        string; required for time-series metadata and latest-value modes.
		    state_code (str): Optional state code. Defaults to an empty string.
		    county_code (str): Optional county code. Defaults to an empty string.
		    site_type (str): Optional site type. Defaults to an empty string.
		    parameter_code (str): Optional five-digit USGS parameter code. Defaults to an empty
		        string.
		    limit (int): Requested row limit from 1 through 10000. Defaults to 25.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, first-row summary, shaped
		        rows, and raw provider payload.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'monitoring-locations':
				return self.fetch_monitoring_locations(
					monitoring_location_id=monitoring_location_id, state_code=state_code,
					county_code=county_code, site_type=site_type, limit=limit, time=time )
			
			if self.mode == 'time-series-metadata':
				return self.fetch_time_series_metadata(
					monitoring_location_id=monitoring_location_id, parameter_code=parameter_code,
					limit=limit, time=time )
			
			if self.mode == 'latest-continuous':
				return self.fetch_latest_continuous(
					monitoring_location_id=monitoring_location_id, parameter_code=parameter_code,
					limit=limit, time=time )
			
			if self.mode == 'latest-daily':
				return self.fetch_latest_daily( monitoring_location_id=monitoring_location_id,
					parameter_code=parameter_code, limit=limit, time=time )
			
			raise ValueError( 'Unsupported USGS Water Data mode. Use monitoring-locations, '
			                  'time-series-metadata, latest-continuous, or latest-daily.' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = "fetch( self, mode: str='monitoring-locations', monitoring_location_id: str='', state_code: str='', county_code: str='', site_type: str='', parameter_code: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSWaterData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class USGSTheNationalMap( Fetcher ):
	"""USGS The National Map TNMAccess adapter.

	Purpose:
	    Retrieve dataset metadata and product search results through TNMAccess v1, apply
	    geographic and paging filters, and shape records with download and metadata URLs.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    dataset (Optional[str]): Requested dataset identifier or filter.
	    query_text (Optional[str]): Latest search expression.
	    bbox (Optional[str]): Normalized geographic bounding-box string.
	    prod_formats (Optional[str]): Requested product format filter.
	    max_items (Optional[int]): Requested maximum number of records.
	    offset (Optional[int]): Requested pagination offset.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	dataset: Optional[ str ]
	query_text: Optional[ str ]
	bbox: Optional[ str ]
	prod_formats: Optional[ str ]
	max_items: Optional[ int ]
	offset: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the TNMAccess v1 base URL, products mode, default maximum of 25 records, zero
		    offset, JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://tnmaccess.nationalmap.gov/api/v1'
		self.mode = 'products'
		self.endpoint = None
		self.dataset = ''
		self.query_text = ''
		self.bbox = ''
		self.prod_formats = ''
		self.max_items = 25
		self.offset = 0
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'endpoint', 'dataset', 'query_text', 'bbox', 'prod_formats',
				'max_items', 'offset', 'params', 'payload', 'result', 'response', 'url', 'timeout',
				'agents', 'headers', 'validate_endpoint', 'validate_max_items', 'validate_offset',
				'validate_bbox', 'coalesce_records', 'shape_dataset_rows', 'shape_product_rows',
				'summarize_rows', 'package_response', 'request', 'fetch_datasets',
				'fetch_products',
				'fetch', 'create_schema' ]
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate a TNMAccess endpoint.

		Purpose:
		    Accept only datasets or products without changing the supplied value.

		Args:
		    endpoint (str): datasets or products.

		Returns:
		    str: Original accepted endpoint.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			
			value = endpoint
			allowed = { 'datasets', 'products' }
			
			if value not in allowed:
				raise ValueError( "endpoint must be 'datasets' or 'products'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_max_items( self, max_items: int ) -> int:
		"""Validate the maximum record count.

		Purpose:
		    Enforce the inclusive range from 1 through 500.

		Args:
		    max_items (int): Requested maximum record count from 1 through 500.

		Returns:
		    int: Original accepted record count.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'max_items', max_items )

			if max_items < 1 or max_items > 500:
				raise ValueError( 'max_items must be between 1 and 500.' )
			
			return max_items
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'validate_max_items( self, max_items: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the pagination offset.

		Purpose:
		    Enforce the inclusive range from 0 through 10000.

		Args:
		    offset (int): Pagination offset from 0 through 10000.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'offset', offset )

			if offset < 0 or offset > 10000:
				raise ValueError( 'offset must be between 0 and 10000.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_bbox( self, bbox: str ) -> str:
		"""Validate and serialize geographic bounds.

		Purpose:
		    Allow an empty selection; otherwise parse four finite coordinates, check geographic
		    ranges and strict axis ordering, and serialize the numeric values.

		Args:
		    bbox (str): Optional comma-separated minx,miny,maxx,maxy bounds; finite longitude within
		        -180 through 180 and latitude within -90 through 90, with strictly increasing
		        bounds.

		Returns:
		    str: Comma-separated floating-point bounds, or an empty string.

		Raises:
		    Error: Raised when the bounds cannot be parsed or violate geographic constraints.
		"""
		try:
			value = ( bbox or '' ).strip( )
			
			if not value:
				return ''
			
			parts = [ part.strip( ) for part in value.split( ',' ) ]
			
			if len( parts ) != 4:
				raise ValueError( 'bbox must use minx,miny,maxx,maxy format.' )
			
			numbers = [ float( part ) for part in parts ]
			if not all( math.isfinite( number ) for number in numbers ):
				raise ValueError( 'bbox coordinates must be finite.' )
			minx, miny, maxx, maxy = numbers
			if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
				raise ValueError( 'bbox longitude must be between -180 and 180.' )
			if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
				raise ValueError( 'bbox latitude must be between -90 and 90.' )
			
			if minx >= maxx:
				raise ValueError( 'bbox minx must be less than maxx.' )
			
			if miny >= maxy:
				raise ValueError( 'bbox miny must be less than maxy.' )
			
			return ','.join( [ str( number ) for number in numbers ] )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'validate_bbox( self, bbox: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract dictionary records from a response.

		Purpose:
		    Filter list payloads to dictionaries, select the first list under items, datasets, data,
		    results, or products in a dictionary payload, or wrap a standalone dictionary. Return an
		    empty list for other payloads.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Extracted dictionary records.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			if payload is None:
				return [ ]
			
			if isinstance( payload, list ):
				return [ item for item in payload if isinstance( item, dict ) ]
			
			if not isinstance( payload, dict ):
				return [ ]
			
			for key in [ 'items', 'datasets', 'data', 'results', 'products' ]:
				value = payload.get( key, None )
				
				if isinstance( value, list ):
					return [ item for item in value if isinstance( item, dict ) ]
			
			return [ payload ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_dataset_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape dataset metadata records.

		Purpose:
		    Extract dataset identifier, tag, name, category, type, and description using alternate
		    provider field names.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented dataset metadata rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				rows.append( { 'Id': item.get( 'id', '' ), 'Dataset': (
						item.get( 'sbDatasetTag', None ) or item.get( 'dataset', None ) or
						item.get( 'tag', '' )),
						'Name': (item.get( 'sbDatasetName', None ) or item.get( 'name', '' )),
						'Category': item.get( 'category', '' ), 'Type': item.get( 'type', '' ),
						'Description': (
								item.get( 'description', None ) or item.get( 'summary', '' )) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'shape_dataset_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_product_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape product metadata records.

		Purpose:
		    Extract title, dataset, format, publication date, size, download and metadata URLs, and
		    bounding box using alternate provider field names.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented product rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				download_url = (
						item.get( 'downloadURL', None ) or item.get( 'downloadUrl', None ) or
						item.get( 'url', '' ))
				meta_url = (
						item.get( 'metaUrl', None ) or item.get( 'metadataUrl', None ) or
						item.get( 'metaURL', '' ))
				bbox = (
						item.get( 'bbox', None ) or item.get( 'boundingBox', None ) or item.get(
					'extent', '' ))
				
				rows.append( { 'Title': (
						item.get( 'title', None ) or item.get( 'name', None ) or item.get(
					'displayName', '' )),
						'Dataset': (
								item.get( 'dataset', None ) or item.get( 'datasets', None ) or
								item.get( 'sourceDataSet', '' )),
						'Product Format': (
								item.get( 'format', None ) or item.get( 'prodFormat', None ) or
								item.get( 'productFormat', '' )),
						'Publication Date': (
								item.get( 'publicationDate', None ) or item.get( 'dateCreated',
							None ) or item.get( 'lastUpdated', '' )),
						'Size': (
								item.get( 'size', None ) or item.get( 'filesize', None ) or
								item.get( 'fileSize', '' )),
						'Download URL': download_url, 'Metadata URL': meta_url,
						'Bounding Box': bbox } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'shape_product_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped TNM records.

		Purpose:
		    Count rows and retain the first row's title, name or dataset and its dataset field.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_title, and first_dataset.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_title = ''
			first_dataset = ''
			
			if rows:
				first_title = str(
					rows[ 0 ].get( 'Title', '' ) or rows[ 0 ].get( 'Name', '' ) or rows[
						0 ].get( 'Dataset', '' ) or '' )
				first_dataset = str( rows[ 0 ].get( 'Dataset', '' ) or '' )
			
			return { 'count': count, 'first_title': first_title, 'first_dataset': first_dataset }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped provider records.

		Purpose:
		    Combine the retained operation and request metadata with row summary statistics, shaped
		    rows, and the raw response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Dict[ str, Any ], time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Execute a TNMAccess request.

		Purpose:
		    Validate endpoint and parameter dictionary, omit None and blank strings, issue a GET
		    request, require JSON list or object, and reject reported provider errors.

		Args:
		    endpoint (str): datasets or products.
		    params (Dict[str, Any]): Request parameter dictionary; an empty dictionary is used for
		        dataset retrieval.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			throw_if( 'params', params )

			if not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			self.endpoint = self.validate_endpoint( endpoint )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			
			for key, value in params.items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, (dict, list) ):
				raise ValueError( 'The National Map returned an invalid JSON payload.' )
			if isinstance( self.payload, dict ) and (self.payload.get( 'error' ) or self.payload.get( 'errors' )):
				raise ValueError( 'The National Map returned an API error.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'request( self, endpoint: str, params: Dict[ str, Any ], time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_datasets( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve TNMAccess dataset metadata.

		Purpose:
		    Request the datasets endpoint, extract records, shape dataset rows, and package the
		    result.

		Args:
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'datasets'
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='datasets', params={ }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_dataset_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'fetch_datasets( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_products( self, dataset: str='', q: str='', bbox: str='', prod_formats: str = '', max_items: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search TNMAccess products.

		Purpose:
		    Validate geographic bounds and pagination, attach optional dataset, search and format
		    filters, retrieve products, and shape metadata rows.

		Args:
		    dataset (str): Optional dataset filter. Defaults to an empty string.
		    q (str): Optional search expression. Defaults to an empty string.
		    bbox (str): Optional comma-separated minx,miny,maxx,maxy bounds; finite longitude within
		        -180 through 180 and latitude within -90 through 90, with strictly increasing
		        bounds. Defaults to an empty string.
		    prod_formats (str): Optional product format filter. Defaults to an empty string.
		    max_items (int): Requested maximum record count from 1 through 500. Defaults to 25.
		    offset (int): Pagination offset from 0 through 10000. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'products'
			self.dataset = dataset
			self.query_text = q
			self.bbox = self.validate_bbox( bbox )
			self.prod_formats = prod_formats
			self.max_items = self.validate_max_items( max_items )
			self.offset = self.validate_offset( offset )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='products', params={ 'datasets': self.dataset,
					'q': self.query_text, 'bbox': self.bbox, 'prodFormats': self.prod_formats,
					'max': self.max_items, 'offset': self.offset }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_product_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = "fetch_products( self, dataset: str='', q: str='', bbox: str='', prod_formats: str = '', max_items: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='products', dataset: str='', q: str='', bbox: str='', prod_formats: str='', max_items: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a TNMAccess operation.

		Purpose:
		    Select datasets or products retrieval and forward applicable search filters.

		Args:
		    mode (str): datasets or products. Defaults to 'products'.
		    dataset (str): Optional dataset filter. Defaults to an empty string.
		    q (str): Optional search expression. Defaults to an empty string.
		    bbox (str): Optional comma-separated minx,miny,maxx,maxy bounds; finite longitude within
		        -180 through 180 and latitude within -90 through 90, with strictly increasing
		        bounds. Defaults to an empty string.
		    prod_formats (str): Optional product format filter. Defaults to an empty string.
		    max_items (int): Requested maximum record count from 1 through 500. Defaults to 25.
		    offset (int): Pagination offset from 0 through 10000. Defaults to 0.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = mode
			
			if self.mode == 'datasets':
				return self.fetch_datasets( time=time )
			
			if self.mode == 'products':
				return self.fetch_products( dataset=dataset, q=q, bbox=bbox,
					prod_formats=prod_formats, max_items=max_items, offset=offset, time=time )
			
			raise ValueError( "Unsupported TNMAccess mode. Use 'datasets' or 'products'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = "fetch( self, mode: str='products', dataset: str='', q: str='', bbox: str='', prod_formats: str='', max_items: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSTheNationalMap'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class USGSScienceBase( Fetcher ):
	"""USGS ScienceBase catalog retrieval adapter.

	Purpose:
	    Search ScienceBase items or retrieve a single item, validate catalog paths and
	    pagination, and package shaped metadata and raw records.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    item_id (Optional[str]): Requested ScienceBase item identifier.
	    query_text (Optional[str]): Latest search expression.
	    max_items (Optional[int]): Requested maximum number of records.
	    offset (Optional[int]): Requested pagination offset.
	    fields (Optional[str]): Requested provider field-selection expression.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Dict[str, Any]]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	item_id: Optional[ str ]
	query_text: Optional[ str ]
	max_items: Optional[ int ]
	offset: Optional[ int ]
	fields: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Dict[ str, Any ] ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the ScienceBase catalog base URL, items mode, default maximum of 25 records,
		    zero offset, JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://www.sciencebase.gov/catalog'
		self.mode = 'items'
		self.endpoint = None
		self.item_id = ''
		self.query_text = ''
		self.max_items = 25
		self.offset = 0
		self.fields = ''
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'endpoint', 'item_id', 'query_text', 'max_items', 'offset',
				'fields', 'params', 'payload', 'result', 'response', 'url', 'timeout', 'agents',
				'headers', 'validate_endpoint', 'validate_max_items', 'validate_offset',
				'coalesce_records', 'shape_item_rows', 'shape_single_item', 'summarize_rows',
				'package_response', 'request', 'fetch_items', 'fetch_item', 'fetch',
				'create_schema' ]
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate a relative ScienceBase endpoint.

		Purpose:
		    Trim surrounding whitespace and slashes, reject full HTTP(S) URLs and parent-directory
		    markers, and require items or a single nonempty item ID path without query or fragment.

		Args:
		    endpoint (str): Relative path 'items' or 'item/{id}', with one nonempty ID segment and
		        no query, fragment, or parent-directory marker.

		Returns:
		    str: Normalized accepted endpoint.

		Raises:
		    Error: Raised when endpoint validation fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			
			value = endpoint.strip( ).strip( '/' )
			
			if not value:
				raise ValueError( 'endpoint cannot be empty.' )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'endpoint must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'endpoint cannot contain parent-directory markers.' )
			
			if value != 'items' and (not value.startswith( 'item/' ) or not value[ 5: ] or '/' in value[ 5: ] or '?' in value or '#' in value):
				raise ValueError( "endpoint must be 'items' or 'item/{id}'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_max_items( self, max_items: int ) -> int:
		"""Validate the requested maximum record count.

		Purpose:
		    Require a multiple of five from 5 through 1000.

		Args:
		    max_items (int): Requested maximum record count, a multiple of five from 5 through 1000.

		Returns:
		    int: Original accepted record count.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'max_items', max_items )

			if max_items < 5 or max_items > 1000 or max_items % 5 != 0:
				raise ValueError( 'max_items must be a multiple of 5 between 5 and 1000.' )
			
			return max_items
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'validate_max_items( self, max_items: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the pagination offset.

		Purpose:
		    Require a multiple of five from 0 through 150000.

		Args:
		    offset (int): Pagination offset, a multiple of five from 0 through 150000.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'offset', offset )

			if offset < 0 or offset > 150000 or offset % 5 != 0:
				raise ValueError( 'offset must be a multiple of 5 between 0 and 150000.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract dictionary records from a response.

		Purpose:
		    Filter list payloads to dictionaries, select the first list under items, results, or
		    data in a dictionary payload, or wrap a standalone dictionary. Return an empty list for
		    other payloads.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Extracted dictionary records.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			if payload is None:
				return [ ]
			
			if isinstance( payload, list ):
				return [ item for item in payload if isinstance( item, dict ) ]
			
			if not isinstance( payload, dict ):
				return [ ]
			
			for key in [ 'items', 'results', 'data' ]:
				value = payload.get( key, None )
				
				if isinstance( value, list ):
					return [ item for item in value if isinstance( item, dict ) ]
			
			return [ payload ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_single_item( self, item: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Shape a ScienceBase item.

		Purpose:
		    Extract item metadata, flag non-None spatial metadata, count file, web-link and contact
		    lists, and retain the raw item. Non-dictionary input is replaced with an empty
		    dictionary.

		Args:
		    item (Dict[str, Any]): Decoded ScienceBase item dictionary.

		Returns:
		    Dict[str, Any]: Item ID, title, type, updated date, summary, spatial flag, counts, and
		        raw item.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			if not isinstance( item, dict ):
				item = { }
			
			files = item.get( 'files', [ ] ) or [ ]
			file_count = len( files ) if isinstance( files, list ) else 0
			
			web_links = item.get( 'webLinks', [ ] ) or [ ]
			link_count = len( web_links ) if isinstance( web_links, list ) else 0
			
			contacts = item.get( 'contacts', [ ] ) or [ ]
			contact_count = len( contacts ) if isinstance( contacts, list ) else 0
			
			return { 'Id': item.get( 'id', '' ), 'Title': item.get( 'title', '' ),
					'Type': item.get( 'itemType', '' ),
					'Updated': (item.get( 'dateUpdated', None ) or item.get( 'lastUpdated', '' )),
					'Summary': (item.get( 'summary', None ) or item.get( 'body', '' )),
					'Has Spatial Metadata': item.get( 'spatial', None ) is not None,
					'File Count': file_count, 'Web Link Count': link_count,
					'Contact Count': contact_count, 'Raw Item': item }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'shape_single_item( self, item: Dict[ str, Any ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_item_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape ScienceBase item records.

		Purpose:
		    Apply single-item shaping to each supplied record.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Shaped ScienceBase item rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				rows.append( self.shape_single_item( item ) )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'shape_item_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped ScienceBase records.

		Purpose:
		    Count rows and rows with spatial metadata, and retain the first title and type.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_title, first_type, and spatial_count.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_title = ''
			first_type = ''
			spatial_count = 0
			
			for row in rows or [ ]:
				if row.get( 'Has Spatial Metadata', False ):
					spatial_count += 1
			
			if rows:
				first_title = str( rows[ 0 ].get( 'Title', '' ) or '' )
				first_type = str( rows[ 0 ].get( 'Type', '' ) or '' )
			
			return { 'count': count, 'first_title': first_title, 'first_type': first_type,
					'spatial_count': spatial_count }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped provider records.

		Purpose:
		    Combine the retained operation and request metadata with row summary statistics, shaped
		    rows, and the raw response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute a ScienceBase catalog request.

		Purpose:
		    Validate the relative endpoint and optional dictionary, omit None and blank strings,
		    issue a GET request, require a JSON object, and reject provider errors or integer status
		    values of at least 400.

		Args:
		    endpoint (str): Relative path 'items' or 'item/{id}', with one nonempty ID segment and
		        no query, fragment, or parent-directory marker.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )

			
			self.endpoint = self.validate_endpoint( endpoint )
			throw_if( 'time', time )
			self.timeout = time
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			
			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'ScienceBase returned an invalid JSON payload.' )
			if self.payload.get( 'error' ) or self.payload.get( 'errors' ) or (isinstance( self.payload.get( 'status' ), int ) and self.payload[ 'status' ] >= 400):
				raise ValueError( 'ScienceBase returned an API error.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_items( self, q: str='', max_items: int=25, offset: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search ScienceBase catalog items.

		Purpose:
		    Validate pagination, request JSON with optional query and fields, extract records, and
		    package shaped item rows.

		Args:
		    q (str): Optional search expression. Defaults to an empty string.
		    max_items (int): Requested maximum record count, a multiple of five from 5 through 1000.
		        Defaults to 25.
		    offset (int): Pagination offset, a multiple of five from 0 through 150000. Defaults to
		        0.
		    fields (str): Optional provider field-selection expression. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'items'
			self.query_text = q
			self.max_items = self.validate_max_items( max_items )
			self.offset = self.validate_offset( offset )
			self.fields = fields
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='items', params={ 'q': self.query_text, 'max': self.max_items,
					'offset': self.offset, 'fields': self.fields, 'format': 'json' }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_item_rows( records )
			return self.package_response( rows )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = "fetch_items( self, q: str='', max_items: int=25, offset: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_item( self, item_id: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve a single ScienceBase catalog item.

		Purpose:
		    Require an item identifier, request its catalog path, shape the decoded item, and
		    package one row.

		Args:
		    item_id (str): Nonblank ScienceBase item identifier.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'item_id', item_id )

			self.mode = 'item'
			self.item_id = item_id
			throw_if( 'time', time )
			self.timeout = time
			self.request( endpoint=f'item/{self.item_id}', params={ }, time=self.timeout )
			detail = self.shape_single_item( self.payload or { } )
			return self.package_response( [ detail ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'fetch_item( self, item_id: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='items', q: str='', item_id: str='', max_items: int=25, offset: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a ScienceBase retrieval operation.

		Purpose:
		    Select items search or individual item retrieval and forward relevant parameters.

		Args:
		    mode (str): items or item. Defaults to 'items'.
		    q (str): Optional search expression. Defaults to an empty string.
		    item_id (str): Item identifier. Defaults to an empty string; required in item mode.
		    max_items (int): Requested maximum record count, a multiple of five from 5 through 1000.
		        Defaults to 25.
		    offset (int): Pagination offset, a multiple of five from 0 through 150000. Defaults to
		        0.
		    fields (str): Optional provider field-selection expression. Defaults to an empty string.
		    time (int): HTTP request timeout in seconds. Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode
			if self.mode == 'items':
				return self.fetch_items( q=q, max_items=max_items, offset=offset, fields=fields,
					time=time )
			
			if self.mode == 'item':
				return self.fetch_item( item_id=item_id, time=time )
			
			raise ValueError( "Unsupported ScienceBase mode. Use 'items' or 'item'." )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = "fetch( self, mode: str='items', q: str='', item_id: str='', max_items: int=25, offset: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'USGSScienceBase'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class AirNow( Fetcher ):
	"""AirNow observation and forecast retrieval adapter.

	Purpose:
	    Retrieve current air-quality observations or forecasts by ZIP code or geographic
	    coordinates, shape AQI records, and summarize maximum AQI while excluding the API key
	    from returned parameters.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    api_key (Optional[str]): AirNow API key loaded from configuration.
	    mode (Optional[str]): Selected retrieval operation.
	    params (Optional[Dict[str, Any]]): Latest submitted parameters, including API_KEY.
	    payload (Optional[Any]): Latest decoded list of observation or forecast dictionaries.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    zip_code (Optional[str]): Latest five-digit ZIP code.
	    latitude (Optional[float]): Latest search latitude.
	    longitude (Optional[float]): Latest search longitude.
	    date (Optional[str]): Latest forecast date string.
	    distance (Optional[int]): Requested positive integer search distance.
	    response (Optional[Response]): Latest HTTP response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	    url (Optional[str]): Latest request URL.
	"""
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	timeout: Optional[ int ]
	agents: Optional[ str ]
	endpoint: Optional[ str ]
	zip_code: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	date: Optional[ str ]
	distance: Optional[ int ]
	result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured AirNow key and user-agent, configure the API base URL and JSON
		    headers, initialize current-zip mode and empty observation state, and set a
		    twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://www.airnowapi.org/aq'
		self.api_key = cfg.AIRNOW_API_KEY
		self.mode = 'current-zip'
		self.params = { }
		self.payload = [ ]
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.endpoint = None
		self.zip_code = None
		self.latitude = None
		self.longitude = None
		self.date = None
		self.distance = None
		self.response = None
		self.result = { }
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'api_key', 'mode', 'params', 'payload', 'timeout', 'agents',
				'endpoint', 'zip_code', 'latitude', 'longitude', 'date', 'distance', 'response',
				'result', 'request', 'shape_rows', 'summarize_rows', 'package_response',
				'fetch_current_zip', 'fetch_current_latlon', 'fetch_forecast_zip',
				'fetch_forecast_latlon', 'fetch', 'create_schema' ]
	
	def request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an AirNow observation or forecast request.

		Purpose:
		    Validate endpoint, positive integer timeout and optional dictionary; omit None and blank
		    strings, attach format and API_KEY, issue a GET request, and require a list of
		    dictionaries.

		Args:
		    endpoint (str): Nonblank relative AirNow API endpoint path.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters excluding API_KEY, and raw record
		        list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )

			
			self.endpoint = endpoint.strip( )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.params[ 'format' ] = 'application/json'
			self.params[ 'API_KEY' ] = self.api_key
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, list ) or any( not isinstance( item, dict ) for item in self.payload ):
				raise ValueError( 'AirNow returned an invalid observation or forecast payload.' )
			self.result = { 'url': self.url, 'params': { key: value for key, value in self.params.items( ) if key != 'API_KEY' }, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape AirNow observation or forecast records.

		Purpose:
		    Extract observed date and hour, reporting area, state, coordinates, pollutant, AQI,
		    category name, action day, and discussion.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented air-quality rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				category = item.get( 'Category', { } ) or { }
				category_name = category.get( 'Name', '' ) if isinstance( category, dict ) else ''
				
				rows.append( { 'Date Observed': item.get( 'DateObserved', '' ),
						'Hour Observed': item.get( 'HourObserved', '' ),
						'Local Time Zone': item.get( 'LocalTimeZone', '' ),
						'Reporting Area': item.get( 'ReportingArea', '' ),
						'State Code': item.get( 'StateCode', '' ),
						'Latitude': item.get( 'Latitude', None ),
						'Longitude': item.get( 'Longitude', None ),
						'Parameter Name': item.get( 'ParameterName', '' ),
						'AQI': item.get( 'AQI', None ), 'Category': category_name,
						'Action Day': item.get( 'ActionDay', '' ),
						'Discussion': item.get( 'Discussion', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize air-quality rows.

		Purpose:
		    Count rows, retain the first nonempty reporting area, and find the maximum convertible
		    AQI with its associated pollutant and category.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, max_aqi, dominant_parameter, top_category, and reporting_area.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			max_aqi = None
			dominant_parameter = ''
			top_category = ''
			reporting_area = ''
			
			for row in rows or [ ]:
				if not reporting_area and row.get( 'Reporting Area', '' ):
					reporting_area = str( row.get( 'Reporting Area', '' ) )
				
				aqi_value = row.get( 'AQI', None )
				
				try:
					if aqi_value is not None:
						if max_aqi is None or float( aqi_value ) > float( max_aqi ):
							max_aqi = float( aqi_value )
							dominant_parameter = str( row.get( 'Parameter Name', '' ) or '' )
							top_category = str( row.get( 'Category', '' ) or '' )
				
				except Exception:
					pass
			
			return { 'count': count, 'max_aqi': max_aqi, 'dominant_parameter': dominant_parameter,
					'top_category': top_category, 'reporting_area': reporting_area }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self ) -> Dict[ str, Any ]:
		"""Package the latest AirNow records.

		Purpose:
		    Extract raw records from the current result, shape and summarize them, and return
		    request metadata without API_KEY.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			records = self.result.get( 'raw', [ ] ) if isinstance( self.result, dict ) else [ ]
			records = records or [ ]
			rows = self.shape_rows( records )
			
			self.result = { 'mode': self.mode, 'url': self.url,
					'params': { key: value for key, value in self.params.items( ) if key != 'API_KEY' },
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': records }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'package_response( self ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def fetch_current_zip( self, zip_code: str, distance: int=25, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve current observations by ZIP code.

		Purpose:
		    Trim and validate the ZIP code, validate positive integer distance, request current
		    observations, and package the records.

		Args:
		    zip_code (str): ZIP code containing exactly five digits after trimming whitespace.
		    distance (int): Positive integer search distance passed to AirNow; booleans are
		        rejected. Defaults to 25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'zip_code', zip_code )
			throw_if( 'distance', distance )

			
			self.mode = 'current-zip'
			self.zip_code = zip_code.strip( )
			if not re.fullmatch( r'\d{5}', self.zip_code ):
				raise ValueError( 'zip_code must contain exactly five digits.' )
			self.distance = distance
			if type( self.distance ) is not int or self.distance < 1:
				raise ValueError( 'distance must be a positive integer.' )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='observation/zipCode/current/', params={
					'zipCode': self.zip_code, 'distance': self.distance }, time=self.timeout )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'fetch_current_zip( self, zip_code: str, distance: int=25, time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_current_latlon( self, latitude: float, longitude: float, distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve current observations by coordinates.

		Purpose:
		    Validate finite geographic coordinates and positive integer distance, request current
		    observations, and package the records.

		Args:
		    latitude (float): Finite latitude from -90 through 90 degrees; booleans are rejected.
		    longitude (float): Finite longitude from -180 through 180 degrees; booleans are
		        rejected.
		    distance (int): Positive integer search distance passed to AirNow; booleans are
		        rejected. Defaults to 25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'latitude', latitude )
			throw_if( 'longitude', longitude )
			throw_if( 'distance', distance )

			
			self.mode = 'current-latlon'
			self.latitude = latitude
			self.longitude = longitude
			if isinstance( self.latitude, bool ) or not (math.isfinite( self.latitude ) and -90 <= self.latitude <= 90):
				raise ValueError( 'latitude must be finite and between -90 and 90.' )
			if isinstance( self.longitude, bool ) or not (math.isfinite( self.longitude ) and -180 <= self.longitude <= 180):
				raise ValueError( 'longitude must be finite and between -180 and 180.' )
			self.distance = distance
			if type( self.distance ) is not int or self.distance < 1:
				raise ValueError( 'distance must be a positive integer.' )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='observation/latLong/current/', params={
					'latitude': self.latitude, 'longitude': self.longitude,
					'distance': self.distance }, time=self.timeout )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'fetch_current_latlon( self, latitude: float, longitude: float, distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_forecast_zip( self, zip_code: str, date: str, distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve forecasts by ZIP code.

		Purpose:
		    Validate ZIP code and positive integer distance, trim the required forecast date,
		    request forecasts, and package the records.

		Args:
		    zip_code (str): ZIP code containing exactly five digits after trimming whitespace.
		    date (str): Nonblank forecast date string, trimmed before submission; no calendar
		        parsing is performed.
		    distance (int): Positive integer search distance passed to AirNow; booleans are
		        rejected. Defaults to 25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'zip_code', zip_code )
			throw_if( 'date', date )
			throw_if( 'distance', distance )

			
			self.mode = 'forecast-zip'
			self.zip_code = zip_code.strip( )
			if not re.fullmatch( r'\d{5}', self.zip_code ):
				raise ValueError( 'zip_code must contain exactly five digits.' )
			self.date = date.strip( )
			self.distance = distance
			if type( self.distance ) is not int or self.distance < 1:
				raise ValueError( 'distance must be a positive integer.' )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='forecast/zipCode/', params={ 'zipCode': self.zip_code,
					'date': self.date, 'distance': self.distance }, time=self.timeout )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'fetch_forecast_zip( self, zip_code: str, date: str, distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_forecast_latlon( self, latitude: float, longitude: float, date: str, distance: int = 25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve forecasts by coordinates.

		Purpose:
		    Validate geographic coordinates and positive integer distance, trim the required
		    forecast date, request forecasts, and package the records.

		Args:
		    latitude (float): Finite latitude from -90 through 90 degrees; booleans are rejected.
		    longitude (float): Finite longitude from -180 through 180 degrees; booleans are
		        rejected.
		    date (str): Nonblank forecast date string, trimmed before submission; no calendar
		        parsing is performed.
		    distance (int): Positive integer search distance passed to AirNow; booleans are
		        rejected. Defaults to 25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'latitude', latitude )
			throw_if( 'longitude', longitude )
			throw_if( 'date', date )
			throw_if( 'distance', distance )

			
			self.mode = 'forecast-latlon'
			self.latitude = latitude
			self.longitude = longitude
			if isinstance( self.latitude, bool ) or not (math.isfinite( self.latitude ) and -90 <= self.latitude <= 90):
				raise ValueError( 'latitude must be finite and between -90 and 90.' )
			if isinstance( self.longitude, bool ) or not (math.isfinite( self.longitude ) and -180 <= self.longitude <= 180):
				raise ValueError( 'longitude must be finite and between -180 and 180.' )
			self.date = date.strip( )
			self.distance = distance
			if type( self.distance ) is not int or self.distance < 1:
				raise ValueError( 'distance must be a positive integer.' )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='forecast/latLong/', params={ 'latitude': self.latitude,
					'longitude': self.longitude, 'date': self.date,
					'distance': self.distance }, time=self.timeout )
			
			return self.package_response( )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'fetch_forecast_latlon( self, latitude: float, longitude: float, date: str, distance: int = 25, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='current-zip', zip_code: str='', latitude: float | None = None, longitude: float | None = None, date: str='', distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an AirNow retrieval operation.

		Purpose:
		    Normalize the mode to lowercase and select current or forecast retrieval by ZIP code or
		    coordinates.

		Args:
		    mode (str): current-zip, current-latlon, forecast-zip, or forecast-latlon. Defaults to
		        'current-zip'.
		    zip_code (str): ZIP code containing exactly five digits after trimming whitespace.
		        Defaults to an empty string; required in ZIP modes.
		    latitude (float | None): Finite latitude from -90 through 90 degrees; booleans are
		        rejected. Defaults to None; required in coordinate modes.
		    longitude (float | None): Finite longitude from -180 through 180 degrees; booleans are
		        rejected. Defaults to None; required in coordinate modes.
		    date (str): Nonblank forecast date string, trimmed before submission; no calendar
		        parsing is performed. Defaults to an empty string; required in forecast modes.
		    distance (int): Positive integer search distance passed to AirNow; booleans are
		        rejected. Defaults to 25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, URL, request parameters excluding API_KEY, AQI summary, shaped
		        rows, and raw record list.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			self.mode = mode.strip( ).lower( )
			
			if self.mode == 'current-zip':
				return self.fetch_current_zip( zip_code, distance, time )
			
			if self.mode == 'current-latlon':
				return self.fetch_current_latlon( latitude, longitude, distance, time )
			
			if self.mode == 'forecast-zip':
				return self.fetch_forecast_zip( zip_code, date, distance, time )
			
			if self.mode == 'forecast-latlon':
				return self.fetch_forecast_latlon( latitude, longitude, date, distance, time )
			
			raise ValueError( "Unsupported AirNow mode. Expected 'current-zip', 'current-latlon', "
			                  "'forecast-zip', or 'forecast-latlon'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = "fetch( self, mode: str='current-zip', zip_code: str='', latitude: float | None = None, longitude: float | None = None, date: str='', distance: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f"{description.strip( )} This function uses the "
					                f"{tool.strip( )} service."),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'AirNow'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class ClimateData( Fetcher ):
	"""NOAA NCEI Access Services climate retrieval adapter.

	Purpose:
	    Search NCEI datasets and retrieve climate data through separate search and data
	    endpoints, validate time intervals and pagination, and shape decoded records.

	Attributes:
	    data_url (Optional[str]): NCEI Access Data Service endpoint.
	    search_url (Optional[str]): NCEI Access Search Service base URL.
	    mode (Optional[str]): Selected retrieval operation.
	    keyword (Optional[str]): Dataset search keyword.
	    dataset (Optional[str]): Requested climate dataset identifier.
	    start_date (Optional[str]): Latest lower date or datetime bound.
	    end_date (Optional[str]): Latest upper date or datetime bound.
	    stations (Optional[str]): Requested station selection.
	    data_types (Optional[str]): Requested data-type selection.
	    limit (Optional[int]): Requested record limit.
	    offset (Optional[int]): Requested dataset-search offset.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded JSON object or list.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	data_url: Optional[ str ]
	search_url: Optional[ str ]
	mode: Optional[ str ]
	keyword: Optional[ str ]
	dataset: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	stations: Optional[ str ]
	data_types: Optional[ str ]
	limit: Optional[ int ]
	offset: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure NCEI search and data URLs, datasets mode, default limit of 25, zero offset,
		    JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.data_url = 'https://www.ncei.noaa.gov/access/services/data/v1'
		self.search_url = 'https://www.ncei.noaa.gov/access/services/search/v1'
		self.mode = 'datasets'
		self.keyword = ''
		self.dataset = ''
		self.start_date = ''
		self.end_date = ''
		self.stations = ''
		self.data_types = ''
		self.limit = 25
		self.offset = 0
		self.params = { }
		self.payload = [ ]
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'data_url', 'search_url', 'mode', 'keyword', 'dataset', 'start_date', 'end_date',
				'stations', 'data_types', 'limit', 'offset', 'params', 'payload', 'result',
				'response', 'url', 'timeout', 'agents', 'headers', 'validate_limit',
				'validate_offset', 'validate_date_range', 'coalesce_records', 'shape_dataset_rows',
				'shape_data_rows', 'summarize_rows', 'package_response', 'request',
				'fetch_datasets', 'fetch_data', 'fetch', 'create_schema' ]
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the record limit.

		Purpose:
		    Require an integer from 1 through 500, excluding booleans.

		Args:
		    limit (int): Integer record limit from 1 through 500; booleans are rejected.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'limit', limit )

			if type( limit ) is not int or limit < 1 or limit > 500:
				raise ValueError( 'limit must be between 1 and 500.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_offset( self, offset: int ) -> int:
		"""Validate the pagination offset.

		Purpose:
		    Require an integer from 0 through 10000, excluding booleans.

		Args:
		    offset (int): Integer offset from 0 through 10000; booleans are rejected.

		Returns:
		    int: Original accepted offset.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'offset', offset )

			if type( offset ) is not int or offset < 0 or offset > 10000:
				raise ValueError( 'offset must be between 0 and 10000.' )
			
			return offset
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'validate_offset( self, offset: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_date_range( self, start_date: str, end_date: str ) -> Tuple[ str, str ]:
		"""Validate a climate observation date range.

		Purpose:
		    Trim both values, parse ISO 8601 dates or datetimes with Z/z interpreted as UTC, reject
		    mixed timezone-aware and naive values, and require a nondecreasing interval.

		Args:
		    start_date (str): ISO 8601 start date or datetime.
		    end_date (str): ISO 8601 end date or datetime, no earlier than the start when both are
		        supplied; both must consistently include or omit timezone information.

		Returns:
		    Tuple[str, str]: Trimmed original start and end date strings.

		Raises:
		    Error: Raised when dates are missing, invalid, incompatible, or reversed.
		"""
		try:
			throw_if( 'start_date', start_date )
			throw_if( 'end_date', end_date )
			start_value = start_date.strip( )
			end_value = end_date.strip( )
			try:
				start_time = dt.datetime.fromisoformat( start_value.replace( 'Z', '+00:00' ).replace( 'z', '+00:00' ) )
				end_time = dt.datetime.fromisoformat( end_value.replace( 'Z', '+00:00' ).replace( 'z', '+00:00' ) )
			except ValueError as exc:
				raise ValueError( 'Dates must be valid ISO 8601 values.' ) from exc
			if (start_time.tzinfo is None) != (end_time.tzinfo is None):
				raise ValueError( 'Both dates must use compatible time zones.' )
			if start_time > end_time:
				raise ValueError( 'start_date must be on or before end_date.' )
			
			return start_value, end_value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'validate_date_range( self, start_date: str, end_date: str ) -> Tuple[ str, str ]'
			Logger( ).write( exception )
			raise exception
	
	def coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract dictionary records from a response.

		Purpose:
		    Filter list payloads to dictionaries, select the first list under results, data, items,
		    or records in a dictionary payload, or wrap a standalone dictionary. Return an empty
		    list for other payloads.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Extracted dictionary records.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			if payload is None:
				return [ ]
			
			if isinstance( payload, list ):
				return [ item for item in payload if isinstance( item, dict ) ]
			
			if not isinstance( payload, dict ):
				return [ ]
			
			for key in [ 'results', 'data', 'items', 'records' ]:
				value = payload.get( key, None )
				
				if isinstance( value, list ):
					return [ item for item in value if isinstance( item, dict ) ]
			
			return [ payload ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'coalesce_records( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_dataset_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape climate dataset metadata.

		Purpose:
		    Extract identifier, dataset, title, description, and date bounds using alternate
		    provider field names.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented climate dataset rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				rows.append( { 'Id': item.get( 'id', '' ),
						'Dataset': (item.get( 'dataset', None ) or item.get( 'name', '' )),
						'Title': (item.get( 'title', None ) or item.get( 'name', '' )),
						'Description': (
								item.get( 'description', None ) or item.get( 'summary', '' )),
						'Start Date': item.get( 'startDate', '' ),
						'End Date': item.get( 'endDate', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'shape_dataset_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_data_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape climate data records.

		Purpose:
		    Replace underscores in each field name with spaces, apply title case, and preserve the
		    associated values.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Climate data dictionaries with display-oriented field names.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			for item in records or [ ]:
				row: Dict[ str, Any ] = { }
				
				for key, value in item.items( ):
					friendly_key = str( key ).replace( '_', ' ' ).title( )
					row[ friendly_key ] = value
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'shape_data_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped climate records.

		Purpose:
		    Count rows and derive the first title and dataset from the first row's available
		    metadata fields.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_title, and first_dataset.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_title = ''
			first_dataset = ''
			
			if rows:
				first_title = str(
					rows[ 0 ].get( 'Title', '' ) or rows[ 0 ].get( 'Station', '' ) or rows[
						0 ].get( 'Date', '' ) or rows[ 0 ].get( 'Start Date', '' ) or '' )
				first_dataset = str(
					rows[ 0 ].get( 'Dataset', '' ) or rows[ 0 ].get( 'Datatype', '' ) or rows[
						0 ].get( 'Data Type', '' ) or '' )
			
			return { 'count': count, 'first_title': first_title, 'first_dataset': first_dataset }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped provider records.

		Purpose:
		    Combine the retained operation and request metadata with row summary statistics, shaped
		    rows, and the raw response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an NCEI Access Services request.

		Purpose:
		    Validate URL, parameter dictionary and positive integer timeout, omit None and blank
		    strings, issue a GET request, require a JSON list or object, and reject provider error
		    fields.

		Args:
		    url (str): Nonblank request URL.
		    params (Dict[str, Any]): Request parameter dictionary.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'params', params )

			if not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			self.url = url.strip( )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.params = { }
			
			
			for key, value in params.items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			content_type = str( self.response.headers.get( 'Content-Type', '' ) ).lower( )
			if 'application/json' in content_type or '+json' in content_type:
				self.payload = self.response.json( )
			else:
				self.payload = self.response.text or ''
			if not isinstance( self.payload, (dict, list) ):
				raise ValueError( 'NOAA returned a non-JSON response.' )
			if isinstance( self.payload, dict ) and any( self.payload.get( key ) for key in ('error', 'errors', 'errorMessage', 'errorCode') ):
				raise ValueError( 'NOAA reported a provider error.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'request( self, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_datasets( self, keyword: str='', start_date: str='', end_date: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Search NCEI climate datasets.

		Purpose:
		    Validate integer paging and optional ISO date filters, query the datasets search path
		    with keyword and date parameters, and package shaped metadata.

		Args:
		    keyword (str): Optional dataset-search keyword. Defaults to an empty string.
		    start_date (str): ISO 8601 start date or datetime. Defaults to an empty string.
		    end_date (str): ISO 8601 end date or datetime, no earlier than the start when both are
		        supplied; both must consistently include or omit timezone information. Defaults to
		        an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    offset (int): Integer offset from 0 through 10000; booleans are rejected. Defaults to 0.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'limit', limit )
			throw_if( 'offset', offset )

			
			self.mode = 'datasets'
			self.keyword = ( keyword or '' ).strip( )
			self.start_date = ( start_date or '' ).strip( )
			self.end_date = ( end_date or '' ).strip( )
			self.limit = self.validate_limit( limit )
			self.offset = self.validate_offset( offset )
			throw_if( 'time', time )
			self.timeout = time
			if self.start_date and self.end_date:
				self.start_date, self.end_date = self.validate_date_range( self.start_date,
					self.end_date )
			else:
				for date_value in (self.start_date, self.end_date):
					if date_value:
						try:
							dt.datetime.fromisoformat( date_value.replace( 'Z', '+00:00' ).replace( 'z', '+00:00' ) )
						except ValueError as exc:
							raise ValueError( 'Dates must be valid ISO 8601 values.' ) from exc
			
			self.request( url=f'{self.search_url}/datasets', params={ 'keyword': self.keyword,
					'startDate': self.start_date, 'endDate': self.end_date, 'limit': self.limit,
					'offset': self.offset }, time=self.timeout )
			records = self.coalesce_records( self.payload )
			rows = self.shape_dataset_rows( records )
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = "fetch_datasets( self, keyword: str='', start_date: str='', end_date: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_data( self, dataset: str, start_date: str, end_date: str, stations: str='', data_types: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve NCEI climate observations.

		Purpose:
		    Validate dataset and required date range, attach optional station and data-type
		    selectors and limit, request JSON, and package shaped data.

		Args:
		    dataset (str): Nonblank dataset identifier.
		    start_date (str): ISO 8601 start date or datetime.
		    end_date (str): ISO 8601 end date or datetime, no earlier than the start when both are
		        supplied; both must consistently include or omit timezone information.
		    stations (str): Optional provider station selection. Defaults to an empty string.
		    data_types (str): Optional provider data-type selection. Defaults to an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'dataset', dataset )
			throw_if( 'start_date', start_date )
			throw_if( 'end_date', end_date )
			throw_if( 'limit', limit )

			
			self.mode = 'data'
			self.dataset = dataset.strip( )
			self.start_date, self.end_date = self.validate_date_range( start_date, end_date )
			self.stations = ( stations or '' ).strip( )
			self.data_types = ( data_types or '' ).strip( )
			self.limit = self.validate_limit( limit )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( url=self.data_url, params={ 'dataset': self.dataset,
					'startDate': self.start_date, 'endDate': self.end_date,
					'stations': self.stations, 'dataTypes': self.data_types, 'format': 'json',
					'limit': self.limit }, time=self.timeout )
			
			records = self.coalesce_records( self.payload )
			rows = self.shape_data_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = "fetch_data( self, dataset: str, start_date: str, end_date: str, stations: str='', data_types: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='datasets', keyword: str='', dataset: str='', start_date: str = '', end_date: str='', stations: str='', data_types: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an NCEI retrieval operation.

		Purpose:
		    Normalize mode and select datasets search or data retrieval; offset applies only to
		    dataset search.

		Args:
		    mode (str): datasets or data. Defaults to 'datasets'.
		    keyword (str): Optional dataset-search keyword. Defaults to an empty string.
		    dataset (str): Nonblank dataset identifier. Defaults to an empty string; required in
		        data mode.
		    start_date (str): ISO 8601 start date or datetime. Defaults to an empty string. Required
		        in data mode.
		    end_date (str): ISO 8601 end date or datetime, no earlier than the start when both are
		        supplied; both must consistently include or omit timezone information. Defaults to
		        an empty string. Required in data mode.
		    stations (str): Optional provider station selection. Defaults to an empty string.
		    data_types (str): Optional provider data-type selection. Defaults to an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    offset (int): Integer offset from 0 through 10000; booleans are rejected. Defaults to 0.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = ( mode or 'datasets' ).strip( ).lower( )
			
			if self.mode == 'datasets':
				return self.fetch_datasets( keyword=keyword, start_date=start_date,
					end_date=end_date, limit=limit, offset=offset, time=time )
			
			if self.mode == 'data':
				return self.fetch_data( dataset=dataset, start_date=start_date, end_date=end_date,
					stations=stations, data_types=data_types, limit=limit, time=time )
			
			raise ValueError( "Unsupported ClimateData mode. Use 'datasets' or 'data'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = "fetch( self, mode: str='datasets', keyword: str='', dataset: str='', start_date: str = '', end_date: str='', stations: str='', data_types: str='', limit: int=25, offset: int=0, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ClimateData'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class EoNet( Fetcher ):
	"""NASA EONET natural-event and category adapter.

	Purpose:
	    Retrieve NASA EONET v3 event and category records using optional source, category,
	    status, date and bounding-box filters, then shape and summarize records.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    source (Optional[str]): Requested event source filter.
	    category (Optional[str]): Requested event category filter.
	    status (Optional[str]): Requested normalized event status.
	    limit (Optional[int]): Requested event limit.
	    days (Optional[int]): Requested lookback period in days.
	    start_date (Optional[str]): Optional event start date.
	    end_date (Optional[str]): Optional event end date.
	    bbox (Optional[str]): Normalized bounding-box string in EONET coordinate order.
	    params (Optional[Dict[str, Any]]): Latest request parameters.
	    payload (Optional[Any]): Latest decoded EONET JSON object.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	source: Optional[ str ]
	category: Optional[ str ]
	status: Optional[ str ]
	limit: Optional[ int ]
	days: Optional[ int ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	bbox: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the EONET v3 base URL, events mode, open status, default limit of 25,
		    thirty-day lookback, JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://eonet.gsfc.nasa.gov/api/v3'
		self.mode = 'events'
		self.endpoint = None
		self.source = ''
		self.category = ''
		self.status = 'open'
		self.limit = 25
		self.days = 30
		self.start_date = ''
		self.end_date = ''
		self.bbox = ''
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'endpoint', 'source', 'category', 'status', 'limit', 'days',
				'start_date', 'end_date', 'bbox', 'params', 'payload', 'result', 'response', 'url',
				'timeout', 'agents', 'headers', 'validate_endpoint', 'validate_status',
				'validate_limit', 'validate_days', 'validate_bbox', 'validate_date_pair',
				'shape_event_rows', 'shape_category_rows', 'summarize_rows', 'package_response',
				'request', 'fetch_events', 'fetch_categories', 'fetch', 'create_schema' ]
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate an EONET endpoint.

		Purpose:
		    Trim and lowercase the endpoint, then accept only events or categories.

		Args:
		    endpoint (str): events or categories.

		Returns:
		    str: Normalized accepted endpoint.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			
			value = endpoint.strip( ).lower( )
			allowed = { 'events', 'categories' }
			
			if value not in allowed:
				raise ValueError( "endpoint must be 'events' or 'categories'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_status( self, status: str ) -> str:
		"""Validate an event status.

		Purpose:
		    Trim and lowercase the status, then accept open, closed, or all.

		Args:
		    status (str): open, closed, or all; trimmed and converted to lowercase.

		Returns:
		    str: Normalized accepted status.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'status', status )
			
			value = status.strip( ).lower( )
			allowed = { 'open', 'closed', 'all' }
			
			if value not in allowed:
				raise ValueError( "status must be 'open', 'closed', or 'all'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'validate_status( self, status: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the event limit.

		Purpose:
		    Require an integer from 1 through 500, excluding booleans.

		Args:
		    limit (int): Integer record limit from 1 through 500; booleans are rejected.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			

			throw_if( 'limit', limit )

			if type( limit ) is not int or limit < 1 or limit > 500:
				raise ValueError( 'limit must be between 1 and 500.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_days( self, days: int ) -> int:
		"""Validate the event lookback period.

		Purpose:
		    Require an integer from 1 through 3650, excluding booleans.

		Args:
		    days (int): Integer lookback period from 1 through 3650; booleans are rejected.

		Returns:
		    int: Original accepted lookback period.

		Raises:
		    Error: Raised when the supplied value is missing, invalid, or outside the accepted
		        constraints.
		"""
		try:
			throw_if( 'days', days )
			
			value = days
			if type( value ) is not int or value < 1 or value > 3650:
				raise ValueError( 'days must be between 1 and 3650.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'validate_days( self, days: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_bbox( self, bbox: str ) -> str:
		"""Validate and serialize EONET bounding coordinates.

		Purpose:
		    Allow empty bounds; otherwise parse four finite values in EONET coordinate order,
		    enforce geographic ranges and strict axis ordering, and serialize the numeric values.

		Args:
		    bbox (str): Optional comma-separated min_lon,max_lat,max_lon,min_lat bounds; finite
		        longitude within -180 through 180 and latitude within -90 through 90, with strictly
		        increasing west-to-east and south-to-north bounds.

		Returns:
		    str: Comma-separated floating-point bounds in EONET order, or an empty string.

		Raises:
		    Error: Raised when bounding coordinates cannot be parsed or violate constraints.
		"""
		try:
			value = ( bbox or '' ).strip( )
			
			if not value:
				return ''
			
			parts = [ part.strip( ) for part in value.split( ',' ) ]
			
			if len( parts ) != 4:
				raise ValueError( 'bbox must use min_lon,max_lat,max_lon,min_lat format.' )
			
			numbers = [ float( part ) for part in parts ]
			if not all( math.isfinite( number ) for number in numbers ):
				raise ValueError( 'bbox coordinates must be finite.' )
			min_lon, max_lat, max_lon, min_lat = numbers
			
			if min_lon < -180.0 or min_lon > 180.0:
				raise ValueError( 'bbox min_lon must be between -180 and 180.' )
			
			if max_lon < -180.0 or max_lon > 180.0:
				raise ValueError( 'bbox max_lon must be between -180 and 180.' )
			
			if min_lat < -90.0 or min_lat > 90.0:
				raise ValueError( 'bbox min_lat must be between -90 and 90.' )
			
			if max_lat < -90.0 or max_lat > 90.0:
				raise ValueError( 'bbox max_lat must be between -90 and 90.' )
			
			if min_lon >= max_lon:
				raise ValueError( 'bbox min_lon must be less than max_lon.' )
			
			if min_lat >= max_lat:
				raise ValueError( 'bbox min_lat must be less than max_lat.' )
			
			return ','.join( [ str( number ) for number in numbers ] )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'validate_bbox( self, bbox: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_date_pair( self, start_date: str='', end_date: str='' ) -> Tuple[ str, str ]:
		"""Validate optional EONET date bounds.

		Purpose:
		    Trim both dates, require both or neither, parse nonempty dates using YYYY-MM-DD, and
		    reject reversed intervals.

		Args:
		    start_date (str): Optional start date parsed with YYYY-MM-DD; supply with end_date.
		        Defaults to an empty string.
		    end_date (str): Optional end date parsed with YYYY-MM-DD, on or after start_date.
		        Defaults to an empty string.

		Returns:
		    Tuple[str, str]: Trimmed start and end values, both empty when omitted.

		Raises:
		    Error: Raised when dates are incomplete, invalid, or reversed.
		"""
		try:
			start_value = ( start_date or '' ).strip( )
			end_value = ( end_date or '' ).strip( )
			
			if ( start_value != '' ) != ( end_value != '' ):
				raise ValueError( 'start_date and end_date must either both be supplied or both be '
				                  'blank.' )
			
			if start_value and end_value:
				try:
					start_time = dt.datetime.strptime( start_value, '%Y-%m-%d' )
					end_time = dt.datetime.strptime( end_value, '%Y-%m-%d' )
				except ValueError as exc:
					raise ValueError( 'EONET dates must use valid YYYY-MM-DD values.' ) from exc
				if start_time > end_time:
					raise ValueError( 'start_date must be on or before end_date.' )
			
			return start_value, end_value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = "validate_date_pair( self, start_date: str='', end_date: str='' ) -> Tuple[ str, str ]"
			Logger( ).write( exception )
			raise exception
	
	def shape_event_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape EONET natural-event records.

		Purpose:
		    Extract event metadata and derive status from the closed date. Use the last geometry
		    list entry and extract coordinates only for point geometry.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Event IDs, titles, status, closed dates, categories, sources,
		        geometry counts and metadata, point coordinates, links, and descriptions.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				categories = item.get( 'categories', [ ] ) or [ ]
				sources = item.get( 'sources', [ ] ) or [ ]
				geometry = item.get( 'geometry', [ ] ) or [ ]
				last_geometry = geometry[ -1 ] if isinstance( geometry, list ) and geometry else { }
				if not isinstance( last_geometry, dict ):
					last_geometry = { }
				coordinates = last_geometry.get( 'coordinates', [ ] ) or [ ]
				geometry_type = str( last_geometry.get( 'type', '' ) or '' )
				latitude = None
				longitude = None
				
				if geometry_type.lower( ) == 'point' and isinstance( coordinates, list ):
					longitude = coordinates[ 0 ] if len( coordinates ) > 0 else None
					latitude = coordinates[ 1 ] if len( coordinates ) > 1 else None
				
				category_titles = ', '.join( [
						str( category.get( 'title', category.get( 'id', '' ) ) ) for category in
						categories if isinstance( category, dict ) ] )
				source_ids = ', '.join( [ str( source.get( 'id', source.get( 'title', '' ) ) ) for
						source in sources if isinstance( source, dict ) ] )
				closed_date = str( item.get( 'closed', '' ) or '' )
				status = 'closed' if closed_date else 'open'
				
				rows.append( { 'Id': item.get( 'id', '' ), 'Title': item.get( 'title', '' ),
						'Status': status, 'Closed Date': closed_date, 'Categories':
							category_titles,
						'Sources': source_ids, 'Geometry Count': len( geometry ) if isinstance( geometry, list ) else 0,
						'Last Geometry Type': geometry_type,
						'Last Geometry Date': last_geometry.get( 'date', '' ), 'Latitude':
							latitude,
						'Longitude': longitude, 'Link': item.get( 'link', '' ),
						'Description': item.get( 'description', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'shape_event_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_category_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape EONET category records.

		Purpose:
		    Extract category identifier, title, description, and link.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented category metadata rows.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				rows.append( { 'Id': item.get( 'id', '' ), 'Title': item.get( 'title', '' ),
						'Description': item.get( 'description', '' ),
						'Link': item.get( 'link', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'shape_category_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped EONET records.

		Purpose:
		    Count rows, count rows with open status, and retain the first title and categories.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, open_count, first_title, and first_categories.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			open_count = 0
			first_title = ''
			first_categories = ''
			
			for row in rows or [ ]:
				if str( row.get( 'Status', '' ) ).lower( ) == 'open':
					open_count += 1
			
			if rows:
				first_title = str( rows[ 0 ].get( 'Title', '' ) or '' )
				first_categories = str( rows[ 0 ].get( 'Categories', '' ) or '' )
			
			return { 'count': count, 'open_count': open_count, 'first_title': first_title,
					'first_categories': first_categories }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped provider records.

		Purpose:
		    Combine the retained operation and request metadata with row summary statistics, shaped
		    rows, and the raw response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an EONET API request.

		Purpose:
		    Validate endpoint, optional parameter dictionary and positive integer timeout; omit None
		    and blank strings, issue a GET request, reject provider errors, and require the
		    endpoint-specific list of dictionaries.

		Args:
		    endpoint (str): events or categories.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Endpoint URL, submitted parameters, and raw decoded response.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )

			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			self.endpoint = self.validate_endpoint( endpoint )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'EONET returned an invalid JSON object.' )
			if self.payload.get( 'error' ) or self.payload.get( 'errors' ):
				raise ValueError( 'EONET reported a provider error.' )
			record_key = 'events' if self.endpoint == 'events' else 'categories'
			if not isinstance( self.payload.get( record_key ), list ) or any( not isinstance( item, dict ) for item in self.payload[ record_key ] ):
				raise ValueError( 'EONET returned an invalid record collection.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_events( self, source: str='', category: str='', status: str='open', limit: int = 25, days: int=30, start_date: str='', end_date: str='', bbox: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve EONET natural events.

		Purpose:
		    Validate status, integer limit and lookback, optional date pair and bounding box, submit
		    event filters including days even when explicit dates are supplied, and package shaped
		    events.

		Args:
		    source (str): Optional event source filter. Defaults to an empty string.
		    category (str): Optional event category filter. Defaults to an empty string.
		    status (str): open, closed, or all; trimmed and converted to lowercase. Defaults to
		        'open'.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    days (int): Integer lookback period from 1 through 3650; booleans are rejected. Defaults
		        to 30.
		    start_date (str): Optional start date parsed with YYYY-MM-DD; supply with end_date.
		        Defaults to an empty string.
		    end_date (str): Optional end date parsed with YYYY-MM-DD, on or after start_date.
		        Defaults to an empty string.
		    bbox (str): Optional comma-separated min_lon,max_lat,max_lon,min_lat bounds; finite
		        longitude within -180 through 180 and latitude within -90 through 90, with strictly
		        increasing west-to-east and south-to-north bounds. Defaults to an empty string.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'status', status )
			throw_if( 'limit', limit )
			throw_if( 'days', days )

			
			self.mode = 'events'
			self.source = ( source or '' ).strip( )
			self.category = ( category or '' ).strip( )
			self.status = self.validate_status( status )
			self.limit = self.validate_limit( limit )
			self.days = self.validate_days( days )
			self.start_date, self.end_date = self.validate_date_pair( start_date=start_date,
				end_date=end_date )
			self.bbox = self.validate_bbox( bbox )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='events', params={ 'source': self.source,
					'category': self.category, 'status': self.status, 'limit': self.limit,
					'days': self.days, 'start': self.start_date, 'end': self.end_date,
					'bbox': self.bbox }, time=self.timeout )
			
			records = (
					self.payload.get( 'events', [ ] ) if isinstance( self.payload, dict ) else [ ])
			rows = self.shape_event_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = "fetch_events( self, source: str='', category: str='', status: str='open', limit: int = 25, days: int=30, start_date: str='', end_date: str='', bbox: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_categories( self, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve EONET event categories.

		Purpose:
		    Request the categories endpoint without filters and package shaped category metadata.

		Args:
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:

			
			self.mode = 'categories'
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='categories', params={ }, time=self.timeout )
			
			records = (
					self.payload.get( 'categories', [ ] ) if isinstance( self.payload, dict ) else
					[ ])
			rows = self.shape_category_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'fetch_categories( self, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='events', source: str='', category: str='', status: str = 'open', limit: int=25, days: int=30, start_date: str='', end_date: str='', bbox: str = '', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an EONET retrieval operation.

		Purpose:
		    Normalize mode and select events or categories retrieval, forwarding applicable event
		    filters.

		Args:
		    mode (str): events or categories. Defaults to 'events'.
		    source (str): Optional event source filter. Defaults to an empty string.
		    category (str): Optional event category filter. Defaults to an empty string.
		    status (str): open, closed, or all; trimmed and converted to lowercase. Defaults to
		        'open'.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    days (int): Integer lookback period from 1 through 3650; booleans are rejected. Defaults
		        to 30.
		    start_date (str): Optional start date parsed with YYYY-MM-DD; supply with end_date.
		        Defaults to an empty string.
		    end_date (str): Optional end date parsed with YYYY-MM-DD, on or after start_date.
		        Defaults to an empty string.
		    bbox (str): Optional comma-separated min_lon,max_lat,max_lon,min_lat bounds; finite
		        longitude within -180 through 180 and latitude within -90 through 90, with strictly
		        increasing west-to-east and south-to-north bounds. Defaults to an empty string.
		    time (int): Positive integer HTTP request timeout in seconds; booleans are rejected.
		        Defaults to 20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, submitted parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when argument validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'mode', mode )
			
			self.mode = ( mode or 'events' ).strip( ).lower( )
			
			if self.mode == 'events':
				return self.fetch_events( source=source, category=category, status=status,
					limit=limit, days=days, start_date=start_date, end_date=end_date, bbox=bbox,
					time=time )
			
			if self.mode == 'categories':
				return self.fetch_categories( time=time )
			
			raise ValueError( "Unsupported EONET mode. Use 'events' or 'categories'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = "fetch( self, mode: str='events', source: str='', category: str='', status: str = 'open', limit: int=25, days: int=30, start_date: str='', end_date: str='', bbox: str = '', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter definitions and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property definitions.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EoNet'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class EnviroFacts( Fetcher ):
	"""EPA Envirofacts table retrieval adapter.

	Purpose:
	    Retrieve supported EPA table records through path-based Envirofacts queries, apply
	    optional state and facility filters, and package display-oriented records.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    mode (Optional[str]): Selected retrieval operation.
	    table_name (Optional[str]): Selected normalized Envirofacts table identifier.
	    state_code (Optional[str]): Optional normalized two-letter state filter.
	    facility_name (Optional[str]): Optional facility-name filter.
	    limit (Optional[int]): Requested record limit.
	    path (Optional[str]): Encoded relative table query path.
	    params (Optional[Dict[str, Any]]): Descriptive table filters retained for output; not
	        sent as query-string parameters.
	    payload (Optional[Any]): Latest decoded list of table records.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	table_name: Optional[ str ]
	state_code: Optional[ str ]
	facility_name: Optional[ str ]
	limit: Optional[ int ]
	path: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure the Envirofacts base URL, TRI_FACILITY table, default limit of 25, JSON
		    headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://data.epa.gov/efservice'
		self.mode = 'table'
		self.table_name = 'TRI_FACILITY'
		self.state_code = ''
		self.facility_name = ''
		self.limit = 25
		self.path = ''
		self.params = { }
		self.payload = [ ]
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'table_name', 'state_code', 'facility_name', 'limit', 'path',
				'params', 'payload', 'result', 'response', 'url', 'timeout', 'agents', 'headers',
				'validate_table_name', 'validate_state_code', 'validate_limit',
				'resolve_table_path', 'shape_rows', 'summarize_rows', 'package_response',
				'request',
				'fetch_table', 'fetch', 'create_schema' ]
	
	def validate_table_name( self, table_name: str ) -> str:
		"""Validate an Envirofacts table name.

		Purpose:
		    Trim and uppercase the name and require one of the supported tables.

		Args:
		    table_name (str): TRI_FACILITY, TRI_RELEASE, or EF_W_EMISSIONS_SOURCE_GHG; trimmed and
		        uppercased.

		Returns:
		    str: Normalized accepted table name.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'table_name', table_name )
			
			value = table_name.strip( ).upper( )
			allowed = { 'TRI_FACILITY', 'TRI_RELEASE', 'EF_W_EMISSIONS_SOURCE_GHG' }
			
			if value not in allowed:
				raise ValueError( 'Unsupported Envirofacts table. Use TRI_FACILITY, TRI_RELEASE, '
				                  'or EF_W_EMISSIONS_SOURCE_GHG.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'validate_table_name( self, table_name: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_state_code( self, state_code: str='' ) -> str:
		"""Validate an optional state code.

		Purpose:
		    Trim and uppercase the supplied code; allow blank or exactly two alphabetic characters.

		Args:
		    state_code (str): Optional two-letter alphabetic code, trimmed and uppercased. Defaults
		        to an empty string.

		Returns:
		    str: Normalized two-letter code or an empty string.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			value = ( state_code or '' ).strip( ).upper( )
			
			if not value:
				return ''
			
			if not re.fullmatch( r'[A-Z]{2}', value ):
				raise ValueError( 'state_code must be blank or a two-letter code.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = "validate_state_code( self, state_code: str='' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def validate_limit( self, limit: int ) -> int:
		"""Validate the record limit.

		Purpose:
		    Require an integer from 1 through 500, excluding booleans.

		Args:
		    limit (int): Integer record limit from 1 through 500; booleans are rejected.

		Returns:
		    int: Original accepted limit.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			

			throw_if( 'limit', limit )

			if type( limit ) is not int or limit < 1 or limit > 500:
				raise ValueError( 'limit must be between 1 and 500.' )
			
			return limit
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'validate_limit( self, limit: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def resolve_table_path( self, table_name: str, state_code: str='', facility_name: str='', limit: int=25 ) -> str:
		"""Construct an Envirofacts table query path.

		Purpose:
		    Validate table, state and limit, select STATE or STATE_ABBR according to the table,
		    append optional FACILITY_NAME and inclusive zero-based row bounds, and URL-encode each
		    segment.

		Args:
		    table_name (str): TRI_FACILITY, TRI_RELEASE, or EF_W_EMISSIONS_SOURCE_GHG; trimmed and
		        uppercased.
		    state_code (str): Optional two-letter alphabetic code, trimmed and uppercased. Defaults
		        to an empty string.
		    facility_name (str): Optional facility-name filter. Defaults to an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.

		Returns:
		    str: Encoded relative table path ending in JSON.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.table_name = self.validate_table_name( table_name )
			self.state_code = self.validate_state_code( state_code )
			self.facility_name = ( facility_name or '' ).strip( )
			self.limit = self.validate_limit( limit )
			
			segments: List[ str ] = [ self.table_name ]
			
			if self.state_code:
				if self.table_name == 'EF_W_EMISSIONS_SOURCE_GHG':
					segments.extend( [ 'STATE', self.state_code ] )
				else:
					segments.extend( [ 'STATE_ABBR', self.state_code ] )
			
			if self.facility_name:
				if self.table_name == 'EF_W_EMISSIONS_SOURCE_GHG':
					segments.extend( [ 'FACILITY_NAME', self.facility_name ] )
				else:
					segments.extend( [ 'FACILITY_NAME', self.facility_name ] )
			
			segments.extend( [ 'ROWS', f'0:{self.limit - 1}', 'JSON' ] )
			
			self.path = '/'.join( [ urllib.parse.quote( str( segment ), safe='' ) for segment in
					segments ] )
			
			return self.path
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = "resolve_table_path( self, table_name: str, state_code: str='', facility_name: str='', limit: int=25 ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape provider records for display.

		Purpose:
		    Skip non-dictionary records and replace underscores in field names with spaces before
		    applying title case; preserve associated values unless transformed keys collide.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented row dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				if not isinstance( item, dict ):
					continue
				
				row: Dict[ str, Any ] = { }
				
				for key, value in item.items( ):
					friendly_key = str( key ).replace( '_', ' ' ).title( )
					row[ friendly_key ] = value
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped table records.

		Purpose:
		    Count rows and extract the first row's facility and state using alternative display
		    field names.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_facility, and first_state.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_facility = ''
			first_state = ''
			
			if rows:
				first_facility = str(
					rows[ 0 ].get( 'Facility Name', '' ) or rows[ 0 ].get( 'Primary Name', '' ) or
					rows[ 0 ].get( 'Name', '' ) or '' )
				
				first_state = str(
					rows[ 0 ].get( 'State', '' ) or rows[ 0 ].get( 'State Abbr', '' ) or rows[
						0 ].get( 'St', '' ) or '' )
			
			return { 'count': count, 'first_facility': first_facility, 'first_state': first_state }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped table records.

		Purpose:
		    Combine table name, mode, retained request metadata, summary, shaped rows, and raw
		    response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, table name, URL, descriptive parameters, summary, shaped rows, and
		        raw record list.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'table_name': self.table_name, 'url': self.url,
					'params': self.params, 'summary': self.summarize_rows( rows ), 'rows': rows,
					'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, url: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an Envirofacts request.

		Purpose:
		    Trim the URL, validate the timeout, issue a GET without query-string parameters, and
		    require a JSON list of dictionaries.

		Args:
		    url (str): Nonblank request URL, trimmed before retrieval.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Endpoint URL, request parameters, and raw decoded response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'url', url )

			
			self.url = url.strip( )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			
			
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, list ) or any( not isinstance( row, dict ) for row in self.payload ):
				raise ValueError( 'Envirofacts returned an invalid table response.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'request( self, url: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_table( self, table_name: str, state_code: str='', facility_name: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve a supported Envirofacts table.

		Purpose:
		    Build the encoded table path, retain descriptive filter parameters, retrieve records,
		    and shape and summarize the result.

		Args:
		    table_name (str): TRI_FACILITY, TRI_RELEASE, or EF_W_EMISSIONS_SOURCE_GHG; trimmed and
		        uppercased.
		    state_code (str): Optional two-letter alphabetic code, trimmed and uppercased. Defaults
		        to an empty string.
		    facility_name (str): Optional facility-name filter. Defaults to an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, table name, URL, descriptive parameters, summary, shaped rows, and
		        raw record list.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'table_name', table_name )
			throw_if( 'limit', limit )
			throw_if( 'time', time )
			
			self.mode = 'table'
			self.path = self.resolve_table_path( table_name=table_name, state_code=state_code,
				facility_name=facility_name, limit=limit )
			self.params = { 'table_name': self.table_name, 'state_code': self.state_code,
					'facility_name': self.facility_name, 'limit': self.limit }
			
			self.request( url=f'{self.base_url}/{self.path}', time=time )
			
			records = self.payload if isinstance( self.payload, list ) else [ ]
			rows = self.shape_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = "fetch_table( self, table_name: str, state_code: str='', facility_name: str='', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, table_name: str='TRI_FACILITY', state_code: str='', facility_name: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve Envirofacts records using the public entry point.

		Purpose:
		    Delegate to fetch_table with the selected table and optional filters.

		Args:
		    table_name (str): TRI_FACILITY, TRI_RELEASE, or EF_W_EMISSIONS_SOURCE_GHG; trimmed and
		        uppercased. Defaults to 'TRI_FACILITY'.
		    state_code (str): Optional two-letter alphabetic code, trimmed and uppercased. Defaults
		        to an empty string.
		    facility_name (str): Optional facility-name filter. Defaults to an empty string.
		    limit (int): Integer record limit from 1 through 500; booleans are rejected. Defaults to
		        25.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, table name, URL, descriptive parameters, summary, shaped rows, and
		        raw record list.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			return self.fetch_table( table_name=table_name, state_code=state_code,
				facility_name=facility_name, limit=limit, time=time )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = "fetch( self, table_name: str='TRI_FACILITY', state_code: str='', facility_name: str = '', limit: int=25, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EnviroFacts'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class TidesAndCurrents( Fetcher ):
	"""NOAA CO-OPS Tides and Currents retrieval adapter.

	Purpose:
	    Retrieve station metadata, water-level observations, or tide predictions using NOAA
	    metadata and data endpoints, validated station and date arguments, and selectable datum,
	    units and time zone.

	Attributes:
	    data_url (Optional[str]): NOAA CO-OPS data getter endpoint.
	    metadata_url (Optional[str]): NOAA CO-OPS station metadata base URL.
	    mode (Optional[str]): Selected retrieval operation.
	    station_id (Optional[str]): Selected normalized station identifier.
	    begin_date (Optional[str]): Validated beginning calendar date string.
	    end_date (Optional[str]): Validated ending calendar date string.
	    datum (Optional[str]): Selected normalized vertical datum.
	    units (Optional[str]): Selected measurement system.
	    time_zone (Optional[str]): Selected provider timezone setting.
	    interval (Optional[str]): Selected tide-prediction sampling interval.
	    params (Optional[Dict[str, Any]]): Latest request parameters or descriptive request
	        metadata.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	data_url: Optional[ str ]
	metadata_url: Optional[ str ]
	mode: Optional[ str ]
	station_id: Optional[ str ]
	begin_date: Optional[ str ]
	end_date: Optional[ str ]
	datum: Optional[ str ]
	units: Optional[ str ]
	time_zone: Optional[ str ]
	interval: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure NOAA metadata and data endpoints, water-level mode, MLLW datum, metric units,
		    GMT timezone, hilo prediction interval, JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.data_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
		self.metadata_url = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi'
		self.mode = 'water-level'
		self.station_id = ''
		self.begin_date = ''
		self.end_date = ''
		self.datum = 'MLLW'
		self.units = 'metric'
		self.time_zone = 'gmt'
		self.interval = 'hilo'
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'data_url', 'metadata_url', 'mode', 'station_id', 'begin_date', 'end_date',
				'datum', 'units', 'time_zone', 'interval', 'params', 'payload', 'result',
				'response', 'url', 'timeout', 'agents', 'headers', 'validate_mode',
				'validate_station_id', 'validate_date_range', 'validate_datum', 'validate_units',
				'validate_time_zone', 'validate_interval', 'shape_station_rows', 'shape_data_rows',
				'summarize_rows', 'package_response', 'request', 'fetch_station',
				'fetch_water_level', 'fetch_tide_predictions', 'fetch', 'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate the retrieval mode.

		Purpose:
		    Trim and lowercase the mode and reject unsupported operations.

		Args:
		    mode (str): station, water-level, or tide-predictions; trimmed and lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			
			value = mode.strip( ).lower( )
			allowed = { 'station', 'water-level', 'tide-predictions' }
			
			if value not in allowed:
				raise ValueError( "Unsupported NOAA Tides & Currents mode. Use 'station', "
				                  "'water-level', or 'tide-predictions'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_station_id( self, station_id: str ) -> str:
		"""Validate a NOAA station identifier.

		Purpose:
		    Trim the identifier and require the permitted characters and length.

		Args:
		    station_id (str): Station identifier containing 3–20 letters, digits, underscores, or
		        hyphens after trimming.

		Returns:
		    str: Trimmed station identifier.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'station_id', station_id )
			
			value = station_id.strip( )
			
			if not re.fullmatch( r'[A-Za-z0-9_-]{3,20}', value ):
				raise ValueError( 'station_id must contain 3 to 20 letters, digits, underscores, '
				                  'or hyphens.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_station_id( self, station_id: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_date_range( self, begin_date: str, end_date: str ) -> Tuple[ str, str ]:
		"""Validate a NOAA calendar date interval.

		Purpose:
		    Require eight digits for each date, parse valid calendar dates, and reject reversed
		    intervals.

		Args:
		    begin_date (str): Valid calendar date in YYYYMMDD format.
		    end_date (str): Valid calendar date in YYYYMMDD format, on or after begin_date.

		Returns:
		    Tuple[str, str]: Trimmed begin and end date strings.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'begin_date', begin_date )
			throw_if( 'end_date', end_date )
			
			begin_value = begin_date.strip( )
			end_value = end_date.strip( )
			
			if not re.fullmatch( r'\d{8}', begin_value ):
				raise ValueError( 'begin_date must use YYYYMMDD format.' )
			
			if not re.fullmatch( r'\d{8}', end_value ):
				raise ValueError( 'end_date must use YYYYMMDD format.' )
			
			try:
				begin_time = dt.datetime.strptime( begin_value, '%Y%m%d' )
				end_time = dt.datetime.strptime( end_value, '%Y%m%d' )
			except ValueError as exc:
				raise ValueError( 'begin_date and end_date must be valid calendar dates.' ) from exc
			if begin_time > end_time:
				raise ValueError( 'begin_date must be on or before end_date.' )
			
			return begin_value, end_value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_date_range( self, begin_date: str, end_date: str ) -> Tuple[ str, str ]'
			Logger( ).write( exception )
			raise exception
	
	def validate_datum( self, datum: str ) -> str:
		"""Validate the vertical datum.

		Purpose:
		    Trim and uppercase the datum and accept only the listed codes.

		Args:
		    datum (str): MLLW, MHHW, MHW, MTL, MSL, MLW, NAVD, or STND; trimmed and uppercased.

		Returns:
		    str: Normalized accepted datum.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'datum', datum )
			
			value = datum.strip( ).upper( )
			allowed = { 'MLLW', 'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'NAVD', 'STND' }
			
			if value not in allowed:
				raise ValueError( 'datum must be one of MLLW, MHHW, MHW, MTL, MSL, MLW, NAVD, '
				                  'or STND.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_datum( self, datum: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_units( self, units: str ) -> str:
		"""Validate the measurement system.

		Purpose:
		    Trim and lowercase the units and accept metric or english.

		Args:
		    units (str): metric or english; trimmed and lowercased.

		Returns:
		    str: Normalized accepted units.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'units', units )
			
			value = units.strip( ).lower( )
			allowed = { 'metric', 'english' }
			
			if value not in allowed:
				raise ValueError( "units must be 'metric' or 'english'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_units( self, units: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_time_zone( self, time_zone: str ) -> str:
		"""Validate the provider timezone.

		Purpose:
		    Trim and lowercase the timezone and accept the supported codes.

		Args:
		    time_zone (str): gmt, lst, or lst_ldt; trimmed and lowercased.

		Returns:
		    str: Normalized accepted timezone.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'time_zone', time_zone )
			
			value = time_zone.strip( ).lower( )
			allowed = { 'gmt', 'lst', 'lst_ldt' }
			
			if value not in allowed:
				raise ValueError( "time_zone must be 'gmt', 'lst', or 'lst_ldt'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_time_zone( self, time_zone: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_interval( self, interval: str ) -> str:
		"""Validate the prediction interval.

		Purpose:
		    Trim and lowercase the interval and accept only the supported codes.

		Args:
		    interval (str): hilo, h, 1, 5, 6, 10, 15, 30, or 60; trimmed and lowercased.

		Returns:
		    str: Normalized accepted interval.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'interval', interval )
			
			value = interval.strip( ).lower( )
			allowed = { 'hilo', 'h', '1', '5', '6', '10', '15', '30', '60' }
			
			if value not in allowed:
				raise ValueError( "interval must be one of 'hilo', 'h', '1', '5', '6', '10', "
				                  "'15', '30', or '60'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'validate_interval( self, interval: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def shape_station_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape NOAA station metadata.

		Purpose:
		    Use the first stations list entry, a stations dictionary, or the top-level payload;
		    construct one display-oriented station row even when metadata is empty.

		Args:
		    payload (Dict[str, Any]): Decoded provider response object.

		Returns:
		    List[Dict[str, Any]]: One row containing station ID, name, state, coordinates, timezone,
		        Great Lakes flag, SHEF code, and details link.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			station = payload.get( 'stations', None ) if isinstance( payload, dict ) else None
			
			if isinstance( station, list ):
				station = station[ 0 ] if station else { }
			
			if not isinstance( station, dict ):
				station = payload if isinstance( payload, dict ) else { }
			
			row: Dict[ str, Any ] = { 'Station Id': station.get( 'id', '' ),
					'Name': station.get( 'name', '' ), 'State': station.get( 'state', '' ),
					'Latitude': station.get( 'lat', '' ), 'Longitude': station.get( 'lng', '' ),
					'Time Zone': station.get( 'timezone', '' ),
					'Great Lakes': station.get( 'greatlakes', '' ),
					'Shef Id': station.get( 'shefcode', '' ),
					'Details Link': station.get( 'self', '' ) }
			
			return [ row ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'shape_station_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_data_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape observations or predictions.

		Purpose:
		    Read data or, when data is None, predictions; skip non-dictionaries and convert field
		    names to title case with spaces.

		Args:
		    payload (Dict[str, Any]): Decoded provider response object.

		Returns:
		    List[Dict[str, Any]]: Display-oriented observation or prediction rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			if not isinstance( payload, dict ):
				return rows
			
			records = payload.get( 'data', None )
			
			if records is None:
				records = payload.get( 'predictions', [ ] )
			
			for item in records or [ ]:
				if not isinstance( item, dict ):
					continue
				
				row: Dict[ str, Any ] = { }
				
				for key, value in item.items( ):
					friendly_key = str( key ).replace( '_', ' ' ).title( )
					row[ friendly_key ] = value
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'shape_data_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped NOAA records.

		Purpose:
		    Count rows and extract first-row station, value and time with truthiness-based
		    alternative-field selection.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_station, first_value, and first_time.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_station = ''
			first_value = ''
			first_time = ''
			
			if rows:
				first_station = str( rows[ 0 ].get( 'Name', '' ) or rows[
					0 ].get( 'Station Id', '' ) or self.station_id or '' )
				first_value = str( rows[ 0 ].get( 'V', '' ) or rows[ 0 ].get( 'Value', '' ) or
				                   rows[
					0 ].get( 'Prediction', '' ) or rows[ 0 ].get( 'Predicted Wl', '' ) or '' )
				first_time = str( rows[ 0 ].get( 'T', '' ) or rows[ 0 ].get( 'Time', '' ) or rows[
					0 ].get( 'Date Time', '' ) or '' )
			
			return { 'count': count, 'first_station': first_station, 'first_value': first_value,
					'first_time': first_time }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package NOAA station or data records.

		Purpose:
		    Combine mode, station identifier, request metadata, summary, shaped rows, and raw JSON.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, station identifier, URL, request parameters, summary, shaped rows,
		        and raw response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'station_id': self.station_id, 'url': self.url,
					'params': self.params, 'summary': self.summarize_rows( rows ), 'rows': rows,
					'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, url: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute a NOAA request.

		Purpose:
		    Validate URL, timeout and optional dictionary, omit None and blank strings, issue a GET,
		    and validate the JSON object and mode-dependent data collection or station metadata.

		Args:
		    url (str): Nonblank request URL, trimmed before retrieval.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Endpoint URL, request parameters, and raw decoded response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'url', url )

			
			self.url = url.strip( )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.params = { }
			
			
			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ):
				raise ValueError( 'NOAA returned an invalid JSON response.' )
			if self.payload.get( 'error' ) or self.payload.get( 'errors' ):
				raise ValueError( 'NOAA returned a provider error.' )
			if self.url == self.data_url:
				key = 'predictions' if self.mode == 'tide-predictions' else 'data'
				if not isinstance( self.payload.get( key ), list ):
					raise ValueError( 'NOAA returned an invalid data collection.' )
			else:
				if not isinstance( self.payload.get( 'stations', self.payload ), (dict, list) ):
					raise ValueError( 'NOAA returned invalid station metadata.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'request( self, url: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_station( self, station_id: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve NOAA station metadata.

		Purpose:
		    Validate station ID, request its metadata JSON endpoint, shape one station row, and
		    package the response.

		Args:
		    station_id (str): Station identifier containing 3–20 letters, digits, underscores, or
		        hyphens after trimming.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, station identifier, URL, request parameters, summary, shaped rows,
		        and raw response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'station_id', station_id )

			
			self.mode = 'station'
			self.station_id = self.validate_station_id( station_id )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( url=f'{self.metadata_url}/stations/{self.station_id}.json', params={ },
				time=self.timeout )
			
			rows = self.shape_station_rows( self.payload )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'fetch_station( self, station_id: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_water_level( self, station_id: str, begin_date: str, end_date: str, datum: str = 'MLLW', units: str='metric', time_zone: str='gmt', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve water-level observations.

		Purpose:
		    Validate station, dates, datum, units and timezone; request the water_level product and
		    shape the returned observation records.

		Args:
		    station_id (str): Station identifier containing 3–20 letters, digits, underscores, or
		        hyphens after trimming.
		    begin_date (str): Valid calendar date in YYYYMMDD format.
		    end_date (str): Valid calendar date in YYYYMMDD format, on or after begin_date.
		    datum (str): MLLW, MHHW, MHW, MTL, MSL, MLW, NAVD, or STND; trimmed and uppercased.
		        Defaults to 'MLLW'.
		    units (str): metric or english; trimmed and lowercased. Defaults to 'metric'.
		    time_zone (str): gmt, lst, or lst_ldt; trimmed and lowercased. Defaults to 'gmt'.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, station identifier, URL, request parameters, summary, shaped rows,
		        and raw response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'station_id', station_id )
			throw_if( 'begin_date', begin_date )
			throw_if( 'end_date', end_date )

			
			self.mode = 'water-level'
			self.station_id = self.validate_station_id( station_id )
			self.begin_date, self.end_date = self.validate_date_range( begin_date, end_date )
			self.datum = self.validate_datum( datum )
			self.units = self.validate_units( units )
			self.time_zone = self.validate_time_zone( time_zone )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( url=self.data_url, params={ 'product': 'water_level',
					'application': 'Foo', 'station': self.station_id, 'begin_date':
						self.begin_date,
					'end_date': self.end_date, 'datum': self.datum, 'units': self.units,
					'time_zone': self.time_zone, 'format': 'json' }, time=self.timeout )
			
			rows = self.shape_data_rows( self.payload )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = "fetch_water_level( self, station_id: str, begin_date: str, end_date: str, datum: str = 'MLLW', units: str='metric', time_zone: str='gmt', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_tide_predictions( self, station_id: str, begin_date: str, end_date: str, datum: str = 'MLLW', units: str='metric', time_zone: str='gmt', interval: str='hilo', time: int = 20 ) -> Dict[ str, Any ] | None:
		"""Retrieve tide predictions.

		Purpose:
		    Validate station, dates, datum, units, timezone and interval; request the predictions
		    product and shape the returned records.

		Args:
		    station_id (str): Station identifier containing 3–20 letters, digits, underscores, or
		        hyphens after trimming.
		    begin_date (str): Valid calendar date in YYYYMMDD format.
		    end_date (str): Valid calendar date in YYYYMMDD format, on or after begin_date.
		    datum (str): MLLW, MHHW, MHW, MTL, MSL, MLW, NAVD, or STND; trimmed and uppercased.
		        Defaults to 'MLLW'.
		    units (str): metric or english; trimmed and lowercased. Defaults to 'metric'.
		    time_zone (str): gmt, lst, or lst_ldt; trimmed and lowercased. Defaults to 'gmt'.
		    interval (str): hilo, h, 1, 5, 6, 10, 15, 30, or 60; trimmed and lowercased. Defaults to
		        'hilo'.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, station identifier, URL, request parameters, summary, shaped rows,
		        and raw response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'station_id', station_id )
			throw_if( 'begin_date', begin_date )
			throw_if( 'end_date', end_date )

			
			self.mode = 'tide-predictions'
			self.station_id = self.validate_station_id( station_id )
			self.begin_date, self.end_date = self.validate_date_range( begin_date, end_date )
			self.datum = self.validate_datum( datum )
			self.units = self.validate_units( units )
			self.time_zone = self.validate_time_zone( time_zone )
			self.interval = self.validate_interval( interval )
			throw_if( 'time', time )
			self.timeout = time
			
			self.request( url=self.data_url, params={ 'product': 'predictions',
					'application': 'Foo', 'station': self.station_id, 'begin_date':
						self.begin_date,
					'end_date': self.end_date, 'datum': self.datum, 'units': self.units,
					'time_zone': self.time_zone, 'interval': self.interval,
					'format': 'json' }, time=self.timeout )
			
			rows = self.shape_data_rows( self.payload )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = "fetch_tide_predictions( self, station_id: str, begin_date: str, end_date: str, datum: str = 'MLLW', units: str='metric', time_zone: str='gmt', interval: str='hilo', time: int = 20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='water-level', station_id: str='', begin_date: str='', end_date: str='', datum: str='MLLW', units: str='metric', time_zone: str = 'gmt', interval: str='hilo', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a NOAA CO-OPS retrieval operation.

		Purpose:
		    Validate mode and delegate to station metadata, water-level observations or tide
		    predictions; dates are required for data products and interval applies only to
		    predictions.

		Args:
		    mode (str): station, water-level, or tide-predictions; trimmed and lowercased. Defaults
		        to 'water-level'.
		    station_id (str): Station identifier containing 3–20 letters, digits, underscores, or
		        hyphens after trimming. Defaults to an empty string; required in all modes.
		    begin_date (str): Valid calendar date in YYYYMMDD format. Defaults to an empty string;
		        required in data modes.
		    end_date (str): Valid calendar date in YYYYMMDD format, on or after begin_date. Defaults
		        to an empty string; required in data modes.
		    datum (str): MLLW, MHHW, MHW, MTL, MSL, MLW, NAVD, or STND; trimmed and uppercased.
		        Defaults to 'MLLW'.
		    units (str): metric or english; trimmed and lowercased. Defaults to 'metric'.
		    time_zone (str): gmt, lst, or lst_ldt; trimmed and lowercased. Defaults to 'gmt'.
		    interval (str): hilo, h, 1, 5, 6, 10, 15, 30, or 60; trimmed and lowercased. Defaults to
		        'hilo'.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, station identifier, URL, request parameters, summary, shaped rows,
		        and raw response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = self.validate_mode( mode )
			
			if self.mode == 'station':
				return self.fetch_station( station_id=station_id, time=time )
			
			if self.mode == 'water-level':
				return self.fetch_water_level( station_id=station_id, begin_date=begin_date,
					end_date=end_date, datum=datum, units=units, time_zone=time_zone, time=time )
			
			if self.mode == 'tide-predictions':
				return self.fetch_tide_predictions( station_id=station_id, begin_date=begin_date,
					end_date=end_date, datum=datum, units=units, time_zone=time_zone,
					interval=interval, time=time )
			
			raise ValueError( "Unsupported mode. Use 'station', 'water-level', or "
			                  "'tide-predictions'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = "fetch( self, mode: str='water-level', station_id: str='', begin_date: str='', end_date: str='', datum: str='MLLW', units: str='metric', time_zone: str = 'gmt', interval: str='hilo', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TidesAndCurrents'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class UvIndex( Fetcher ):
	"""EPA daily and hourly UV Index forecast adapter.

	Purpose:
	    Retrieve EPA UV Index forecasts by ZIP code or city and state using encoded location
	    paths, shape records and summarize the maximum finite UV value.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    mode (Optional[str]): Selected retrieval operation.
	    zip_code (Optional[str]): Latest normalized ZIP or ZIP+4 selection.
	    city (Optional[str]): Latest trimmed city selection.
	    state (Optional[str]): Latest normalized two-letter state selection.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    params (Optional[Dict[str, Any]]): Descriptive location parameters retained for output;
	        not sent as query-string parameters.
	    payload (Optional[Any]): Latest decoded forecast record list.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	mode: Optional[ str ]
	zip_code: Optional[ str ]
	city: Optional[ str ]
	state: Optional[ str ]
	endpoint: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Configure EPA dmapservice base URL, daily-zip mode, empty location and response state,
		    JSON headers, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://data.epa.gov/dmapservice'
		self.mode = 'daily-zip'
		self.zip_code = ''
		self.city = ''
		self.state = ''
		self.endpoint = ''
		self.params = { }
		self.payload = [ ]
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'mode', 'zip_code', 'city', 'state', 'endpoint', 'params', 'payload',
				'result', 'response', 'url', 'timeout', 'agents', 'headers', 'validate_mode',
				'validate_zip_code', 'validate_city', 'validate_state', 'shape_rows',
				'summarize_rows', 'package_response', 'request', 'fetch_daily_zip',
				'fetch_daily_city_state', 'fetch_hourly_zip', 'fetch_hourly_city_state', 'fetch',
				'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate a UV Index retrieval mode.

		Purpose:
		    Trim and lowercase the mode and reject unsupported operations.

		Args:
		    mode (str): daily-zip, daily-city-state, hourly-zip, or hourly-city-state; trimmed and
		        lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			value = mode.strip( ).lower( )
			allowed = { 'daily-zip', 'daily-city-state', 'hourly-zip', 'hourly-city-state' }
			if value not in allowed:
				raise ValueError( "Unsupported EPA UV Index mode. Use 'daily-zip', "
				                  "'daily-city-state', 'hourly-zip', or 'hourly-city-state'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_zip_code( self, zip_code: str ) -> str:
		"""Validate ZIP or ZIP+4 format.

		Purpose:
		    Trim the value and require five digits with an optional hyphen and four additional
		    digits.

		Args:
		    zip_code (str): Five-digit ZIP or ZIP+4 string after trimming.

		Returns:
		    str: Trimmed accepted ZIP code.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'zip_code', zip_code )
			
			value = zip_code.strip( )
			
			if not re.fullmatch( r'\d{5}(?:-\d{4})?', value ):
				raise ValueError( 'zip_code must be a valid 5-digit ZIP or ZIP+4 value.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'validate_zip_code( self, zip_code: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_city( self, city: str ) -> str:
		"""Validate a city path value.

		Purpose:
		    Trim the city and reject empty values or path separators.

		Args:
		    city (str): Nonblank city name without forward or backward path separators.

		Returns:
		    str: Trimmed city name.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'city', city )
			
			value = city.strip( )
			
			if not value:
				raise ValueError( 'city cannot be empty.' )
			
			if '/' in value or '\\' in value:
				raise ValueError( 'city cannot contain path separators.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'validate_city( self, city: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_state( self, state: str ) -> str:
		"""Validate a state code.

		Purpose:
		    Trim and uppercase the code and require exactly two letters.

		Args:
		    state (str): Two-letter alphabetic state code after trimming and uppercasing.

		Returns:
		    str: Normalized two-letter code.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'state', state )
			
			value = state.strip( ).upper( )
			
			if not re.fullmatch( r'[A-Z]{2}', value ):
				raise ValueError( 'state must be a two-letter abbreviation.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'validate_state( self, state: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape provider records for display.

		Purpose:
		    Skip non-dictionary records and replace underscores in field names with spaces before
		    applying title case; preserve associated values unless transformed keys collide.

		Args:
		    records (List[Dict[str, Any]]): Decoded provider record dictionaries.

		Returns:
		    List[Dict[str, Any]]: Display-oriented row dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in records or [ ]:
				if not isinstance( item, dict ):
					continue
				
				row: Dict[ str, Any ] = { }
				
				for key, value in item.items( ):
					friendly_key = str( key ).replace( '_', ' ' ).title( )
					row[ friendly_key ] = value
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'shape_rows( self, records: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize UV forecast rows.

		Purpose:
		    Count rows, retain the first nonempty location and alert, and find the maximum finite
		    convertible UV value using supported alternate field names.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, max_uv, first_location, and first_alert.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			max_uv = None
			first_location = ''
			first_alert = ''
			
			for row in rows or [ ]:
				if not first_location:
					first_location = str( row.get( 'City', '' ) or row.get( 'Zip', '' ) or
					                      row.get( 'State', '' ) or '' )
				
				if not first_alert:
					first_alert = str( row.get( 'Uv Alert', '' ) or row.get( 'Alert', '' ) or '' )
				
				uv_value = row.get( 'Uv Value' )
				if uv_value is None:
					uv_value = row.get( 'Index' )
				if uv_value is None:
					uv_value = row.get( 'Uv Index' )
				
				try:
					if uv_value is not None and str( uv_value ).strip( ):
						value = float( uv_value )
						if math.isfinite( value ) and (max_uv is None or value > max_uv):
							max_uv = value
				except Exception:
					pass
			
			return { 'count': count, 'max_uv': max_uv, 'first_location': first_location,
					'first_alert': first_alert }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped provider records.

		Purpose:
		    Combine retained operation and request metadata with row summary statistics, shaped
		    rows, and raw response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an EPA UV Index request.

		Purpose:
		    Validate URL, parameter dictionary and timeout; retain descriptive parameters but issue
		    a GET using only the URL and headers, and require a JSON list of dictionaries.

		Args:
		    url (str): Nonblank request URL, trimmed before retrieval.
		    params (Dict[str, Any]): Descriptive request parameter dictionary.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Endpoint URL, request parameters, and raw decoded response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'params', params )

			
			self.url = url.strip( )
			if not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			self.params = params
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			
			
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, list ) or any( not isinstance( item, dict ) for item in self.payload ):
				raise ValueError( 'EPA returned an invalid UV Index forecast response.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'request( self, url: str, params: Dict[ str, Any ], time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_daily_zip( self, zip_code: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve daily UV forecasts by ZIP code.

		Purpose:
		    Validate the location, encode its values as URL path segments, request the EPA daily
		    endpoint, and shape and summarize the returned records.

		Args:
		    zip_code (str): Five-digit ZIP or ZIP+4 string after trimming.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'zip_code', zip_code )

			
			self.mode = 'daily-zip'
			self.zip_code = self.validate_zip_code( zip_code )
			self.endpoint = 'getEnvirofactsUVDAILY'
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'zip_code': self.zip_code }
			self.url = (f'{self.base_url}/{self.endpoint}/ZIP/'
			            f'{urllib.parse.quote( self.zip_code, safe="" )}/JSON')
			
			self.request( url=self.url, params=self.params, time=self.timeout )
			
			records = self.payload if isinstance( self.payload, list ) else [ ]
			rows = self.shape_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'fetch_daily_zip( self, zip_code: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_daily_city_state( self, city: str, state: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve daily UV forecasts by city and state.

		Purpose:
		    Validate the location, encode its values as URL path segments, request the EPA daily
		    endpoint, and shape and summarize the returned records.

		Args:
		    city (str): Nonblank city name without forward or backward path separators.
		    state (str): Two-letter alphabetic state code after trimming and uppercasing.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'city', city )
			throw_if( 'state', state )

			
			self.mode = 'daily-city-state'
			self.city = self.validate_city( city )
			self.state = self.validate_state( state )
			self.endpoint = 'getEnvirofactsUVDAILY'
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'city': self.city, 'state': self.state }
			self.url = (f'{self.base_url}/{self.endpoint}/CITY/'
			            f'{urllib.parse.quote( self.city, safe="" )}/STATE/'
			            f'{urllib.parse.quote( self.state, safe="" )}/JSON')
			
			self.request( url=self.url, params=self.params, time=self.timeout )
			
			records = self.payload if isinstance( self.payload, list ) else [ ]
			rows = self.shape_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'fetch_daily_city_state( self, city: str, state: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_hourly_zip( self, zip_code: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve hourly UV forecasts by ZIP code.

		Purpose:
		    Validate the location, encode its values as URL path segments, request the EPA hourly
		    endpoint, and shape and summarize the returned records.

		Args:
		    zip_code (str): Five-digit ZIP or ZIP+4 string after trimming.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'zip_code', zip_code )

			
			self.mode = 'hourly-zip'
			self.zip_code = self.validate_zip_code( zip_code )
			self.endpoint = 'getEnvirofactsUVHOURLY'
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'zip_code': self.zip_code }
			self.url = (f'{self.base_url}/{self.endpoint}/ZIP/'
			            f'{urllib.parse.quote( self.zip_code, safe="" )}/JSON')
			
			self.request( url=self.url, params=self.params, time=self.timeout )
			
			records = self.payload if isinstance( self.payload, list ) else [ ]
			rows = self.shape_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'fetch_hourly_zip( self, zip_code: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_hourly_city_state( self, city: str, state: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve hourly UV forecasts by city and state.

		Purpose:
		    Validate the location, encode its values as URL path segments, request the EPA hourly
		    endpoint, and shape and summarize the returned records.

		Args:
		    city (str): Nonblank city name without forward or backward path separators.
		    state (str): Two-letter alphabetic state code after trimming and uppercasing.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'city', city )
			throw_if( 'state', state )

			
			self.mode = 'hourly-city-state'
			self.city = self.validate_city( city )
			self.state = self.validate_state( state )
			self.endpoint = 'getEnvirofactsUVHOURLY'
			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'city': self.city, 'state': self.state }
			self.url = (f'{self.base_url}/{self.endpoint}/CITY/'
			            f'{urllib.parse.quote( self.city, safe="" )}/STATE/'
			            f'{urllib.parse.quote( self.state, safe="" )}/JSON')
			
			self.request( url=self.url, params=self.params, time=self.timeout )
			
			records = self.payload if isinstance( self.payload, list ) else [ ]
			rows = self.shape_rows( records )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'fetch_hourly_city_state( self, city: str, state: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='daily-zip', zip_code: str='', city: str='', state: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an EPA UV Index retrieval operation.

		Purpose:
		    Validate mode and select daily or hourly retrieval by ZIP or city/state.

		Args:
		    mode (str): daily-zip, daily-city-state, hourly-zip, or hourly-city-state; trimmed and
		        lowercased. Defaults to 'daily-zip'.
		    zip_code (str): Five-digit ZIP or ZIP+4 string after trimming. Defaults to an empty
		        string; required in ZIP modes.
		    city (str): Nonblank city name without forward or backward path separators. Defaults to
		        an empty string; required in city/state modes.
		    state (str): Two-letter alphabetic state code after trimming and uppercasing. Defaults
		        to an empty string; required in city/state modes.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = self.validate_mode( mode )
			
			if self.mode == 'daily-zip':
				return self.fetch_daily_zip( zip_code=zip_code, time=time )
			
			if self.mode == 'daily-city-state':
				return self.fetch_daily_city_state( city=city, state=state, time=time )
			
			if self.mode == 'hourly-zip':
				return self.fetch_hourly_zip( zip_code=zip_code, time=time )
			
			if self.mode == 'hourly-city-state':
				return self.fetch_hourly_city_state( city=city, state=state, time=time )
			
			raise ValueError( "Unsupported mode. Use 'daily-zip', 'daily-city-state', "
			                  "'hourly-zip', or 'hourly-city-state'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = "fetch( self, mode: str='daily-zip', zip_code: str='', city: str='', state: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'UvIndex'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class PurpleAir( Fetcher ):
	"""PurpleAir v1 sensor retrieval adapter.

	Purpose:
	    Retrieve sensor lists within a geographic bounding box or individual sensor details,
	    validate filters and field selection, and package shaped air-quality records and PM2.5
	    summaries.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    api_key (Optional[str]): Provider credential loaded from configuration; sent in
	        X-API-Key.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    sensor_index (Optional[int]): Selected individual sensor index.
	    nwlng (Optional[float]): Northwest bounding longitude.
	    nwlat (Optional[float]): Northwest bounding latitude.
	    selng (Optional[float]): Southeast bounding longitude.
	    selat (Optional[float]): Southeast bounding latitude.
	    location_type (Optional[int]): Requested integer location type.
	    max_age (Optional[int]): Requested nonnegative age filter.
	    modified_since (Optional[int]): Requested nonnegative modification filter.
	    fields (Optional[str]): Normalized comma-separated field selection.
	    params (Optional[Dict[str, Any]]): Latest request parameters or descriptive request
	        metadata.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	sensor_index: Optional[ int ]
	nwlng: Optional[ float ]
	nwlat: Optional[ float ]
	selng: Optional[ float ]
	selat: Optional[ float ]
	location_type: Optional[ int ]
	max_age: Optional[ int ]
	modified_since: Optional[ int ]
	fields: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured PurpleAir key, configure v1 endpoint and JSON headers, initialize
		    sensors mode, zero-valued filters, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://api.purpleair.com/v1'
		self.api_key = cfg.PURPLEAIR_API_KEY
		self.mode = 'sensors'
		self.endpoint = ''
		self.sensor_index = None
		self.nwlng = None
		self.nwlat = None
		self.selng = None
		self.selat = None
		self.location_type = 0
		self.max_age = 0
		self.modified_since = 0
		self.fields = ''
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
		
		if self.api_key:
			self.headers[ 'X-API-Key' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'api_key', 'mode', 'endpoint', 'sensor_index', 'nwlng', 'nwlat',
				'selng', 'selat', 'location_type', 'max_age', 'modified_since', 'fields', 'params',
				'payload', 'result', 'response', 'url', 'timeout', 'agents', 'headers',
				'validate_mode', 'validate_endpoint', 'validate_sensor_index',
				'validate_longitude', 'validate_latitude', 'validate_bbox',
				'validate_location_type', 'validate_non_negative_integer', 'normalize_fields',
				'shape_sensor_list_rows', 'shape_sensor_detail_rows', 'summarize_rows',
				'package_response', 'request', 'fetch_sensors', 'fetch_sensor', 'fetch',
				'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate the sensor retrieval mode.

		Purpose:
		    Trim and lowercase the mode and accept sensors or sensor.

		Args:
		    mode (str): sensors or sensor; trimmed and lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			
			value = mode.strip( ).lower( )
			allowed = { 'sensors', 'sensor' }
			
			if value not in allowed:
				raise ValueError( "Unsupported PurpleAir mode. Use 'sensors' or 'sensor'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate a PurpleAir relative endpoint.

		Purpose:
		    Trim surrounding whitespace and slashes, reject full URLs and parent-directory markers,
		    and require a supported sensor path.

		Args:
		    endpoint (str): Relative path sensors or sensors/{sensor_index}, with a digit-only
		        index; surrounding whitespace and slashes are removed.

		Returns:
		    str: Normalized accepted endpoint.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			
			value = endpoint.strip( ).strip( '/' )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'endpoint must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'endpoint cannot contain parent-directory markers.' )
			
			if value != 'sensors' and not re.fullmatch( r'sensors/\d+', value ):
				raise ValueError( "endpoint must be 'sensors' or 'sensors/{sensor_index}'." )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_sensor_index( self, sensor_index: int ) -> int:
		"""Validate an individual sensor index.

		Purpose:
		    Require a positive integer excluding booleans.

		Args:
		    sensor_index (int): Positive integer sensor index; booleans are rejected.

		Returns:
		    int: Original accepted sensor index.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'sensor_index', sensor_index )
			
			value = sensor_index
			if type( value ) is not int or value < 1:
				raise ValueError( 'sensor_index must be greater than or equal to 1.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_sensor_index( self, sensor_index: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_longitude( self, name: str, value: float ) -> float:
		"""Validate a longitude value.

		Purpose:
		    Reject booleans and nonfinite or out-of-range longitude values.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (float): Finite longitude from -180 through 180 degrees.

		Returns:
		    float: Original accepted longitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )
			throw_if( name, value )
			
			number = value
			if isinstance( number, bool ) or not math.isfinite( number ) or number < -180.0 or number > 180.0:
				raise ValueError( f'{name} must be between -180 and 180.' )
			
			return number
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_longitude( self, name: str, value: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_latitude( self, name: str, value: float ) -> float:
		"""Validate a latitude value.

		Purpose:
		    Reject booleans and nonfinite or out-of-range latitude values.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (float): Finite latitude from -90 through 90 degrees.

		Returns:
		    float: Original accepted latitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )
			throw_if( name, value )
			
			number = value
			if isinstance( number, bool ) or not math.isfinite( number ) or number < -90.0 or number > 90.0:
				raise ValueError( f'{name} must be between -90 and 90.' )
			
			return number
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_latitude( self, name: str, value: float ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_bbox( self, nwlng: float, nwlat: float, selng: float, selat: float ) -> Tuple[ float, float, float, float ]:
		"""Validate a northwest/southeast bounding box.

		Purpose:
		    Validate all four coordinates and require west longitude less than east longitude and
		    north latitude greater than south latitude.

		Args:
		    nwlng (float): Finite northwest longitude from -180 through 180 degrees; less than
		        selng; booleans are rejected.
		    nwlat (float): Finite northwest latitude from -90 through 90 degrees; greater than
		        selat; booleans are rejected.
		    selng (float): Finite southeast longitude from -180 through 180 degrees; booleans are
		        rejected.
		    selat (float): Finite southeast latitude from -90 through 90 degrees; booleans are
		        rejected.

		Returns:
		    Tuple[float, float, float, float]: Original accepted northwest longitude, northwest
		        latitude, southeast longitude, and southeast latitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			nw_lng = self.validate_longitude( 'nwlng', nwlng )
			nw_lat = self.validate_latitude( 'nwlat', nwlat )
			se_lng = self.validate_longitude( 'selng', selng )
			se_lat = self.validate_latitude( 'selat', selat )
			
			if nw_lng >= se_lng:
				raise ValueError( 'nwlng must be less than selng.' )
			
			if nw_lat <= se_lat:
				raise ValueError( 'nwlat must be greater than selat.' )
			
			return nw_lng, nw_lat, se_lng, se_lat
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_bbox( self, nwlng: float, nwlat: float, selng: float, selat: float ) -> Tuple[ float, float, float, float ]'
			Logger( ).write( exception )
			raise exception
	
	def validate_location_type( self, location_type: int ) -> int:
		"""Validate the location type.

		Purpose:
		    Require an integer equal to zero or one, excluding booleans.

		Args:
		    location_type (int): Integer location type, 0 or 1; booleans are rejected.

		Returns:
		    int: Original accepted location type.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'location_type', location_type )
			
			value = location_type
			if type( value ) is not int or value not in (0, 1):
				raise ValueError( 'location_type must be 0 or 1.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_location_type( self, location_type: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_non_negative_integer( self, name: str, value: int ) -> int:
		"""Validate a nonnegative integer filter.

		Purpose:
		    Require an integer at least zero, excluding booleans.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (int): Nonnegative integer filter.

		Returns:
		    int: Original accepted value.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )
			
			throw_if( name, value )
			
			number = value
			if type( number ) is not int or number < 0:
				raise ValueError( f'{name} must be greater than or equal to 0.' )
			
			return number
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'validate_non_negative_integer( self, name: str, value: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def normalize_fields( self, fields: str, default_fields: str ) -> str:
		"""Normalize a sensor field selection.

		Purpose:
		    Use default fields when the requested string is blank, trim comma-separated names,
		    discard blank entries, require at least one field, and retain the joined selection.

		Args:
		    fields (str): Requested comma-separated fields, or blank to use defaults.
		    default_fields (str): Nonblank fallback comma-separated fields.

		Returns:
		    str: Normalized comma-separated field selection.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'default_fields', default_fields )
			
			selected = ( fields or '' ).strip( )
			
			if not selected:
				selected = default_fields.strip( )
			
			field_list = [ field.strip( ) for field in selected.split( ',' ) if
					field and field.strip( ) ]
			
			if not field_list:
				raise ValueError( 'At least one PurpleAir field is required.' )
			
			self.fields = ','.join( field_list )
			
			return self.fields
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'normalize_fields( self, fields: str, default_fields: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def shape_sensor_list_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape PurpleAir sensor list records.

		Purpose:
		    Map each array record to provider field names by position, fill missing cells with None,
		    and extract fixed display columns.

		Args:
		    payload (Dict[str, Any]): Decoded provider response object.

		Returns:
		    List[Dict[str, Any]]: Sensor index, name, PM2.5, temperature, humidity, coordinates,
		        last seen, and location type rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			if not isinstance( payload, dict ):
				return rows
			
			fields = payload.get( 'fields', [ ] ) or [ ]
			data = payload.get( 'data', [ ] ) or [ ]
			
			for record in data:
				if not isinstance( record, list ):
					continue
				
				sensor = { fields[ index ]: record[ index ] if index < len( record ) else None for
						index in range( len( fields ) ) }
				
				rows.append( { 'Sensor Index': sensor.get( 'sensor_index', None ),
						'Name': sensor.get( 'name', '' ), 'PM2.5': sensor.get( 'pm2.5', None ),
						'Temperature': sensor.get( 'temperature', None ),
						'Humidity': sensor.get( 'humidity', None ),
						'Latitude': sensor.get( 'latitude', None ),
						'Longitude': sensor.get( 'longitude', None ),
						'Last Seen': sensor.get( 'last_seen', None ),
						'Location Type': sensor.get( 'location_type', None ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'shape_sensor_list_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_sensor_detail_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]:
		"""Shape an individual sensor record.

		Purpose:
		    Extract the sensor object and fixed detail fields, using the retained sensor index when
		    the response omits it.

		Args:
		    payload (Dict[str, Any]): Decoded provider response object.

		Returns:
		    List[Dict[str, Any]]: One sensor detail row for a dictionary payload, otherwise an empty
		        list.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			if not isinstance( payload, dict ):
				return [ ]
			
			sensor = payload.get( 'sensor', { } ) or { }
			
			row = { 'Sensor Index': sensor.get( 'sensor_index', self.sensor_index ),
					'Name': sensor.get( 'name', '' ), 'Model': sensor.get( 'model', '' ),
					'Hardware': sensor.get( 'hardware', '' ),
					'PM2.5 Cf 1 A': sensor.get( 'pm2.5_cf_1_a', None ),
					'PM2.5 Cf 1 B': sensor.get( 'pm2.5_cf_1_b', None ),
					'Temperature': sensor.get( 'temperature', None ),
					'Humidity': sensor.get( 'humidity', None ),
					'Pressure': sensor.get( 'pressure', None ),
					'Latitude': sensor.get( 'latitude', None ),
					'Longitude': sensor.get( 'longitude', None ),
					'Last Seen': sensor.get( 'last_seen', None ),
					'Firmware Version': sensor.get( 'firmware_version', '' ),
					'RSSI': sensor.get( 'rssi', None ) }
			
			return [ row ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'shape_sensor_detail_rows( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped sensor records.

		Purpose:
		    Count rows, retain the first nonempty sensor name or index, and compute maximum finite
		    PM2.5 using the list field or CF 1 A detail field.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, max_pm25, and first_name.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			max_pm25 = None
			first_name = ''
			
			for row in rows or [ ]:
				if not first_name:
					first_name = str( row.get( 'Name', '' ) or row.get( 'Sensor Index', '' ) or
					                  '' )
				
				pm25_value = row.get( 'PM2.5', None )
				
				if pm25_value is None:
					pm25_value = row.get( 'PM2.5 Cf 1 A', None )
				
				try:
					if pm25_value is not None and str( pm25_value ).strip( ):
						value = float( pm25_value )
						if math.isfinite( value ) and (max_pm25 is None or value > max_pm25):
							max_pm25 = value
				except Exception:
					pass
			
			return { 'count': count, 'max_pm25': max_pm25, 'first_name': first_name }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ], params: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Package shaped PurpleAir records.

		Purpose:
		    Combine mode, URL, caller-supplied descriptive parameters, row summary, shaped rows, and
		    raw provider response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.
		    params (Dict[str, Any]): Request metadata to include in the result.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url, 'params': params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ], params: Dict[ str, Any ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute a PurpleAir request.

		Purpose:
		    Validate endpoint, optional parameters and timeout, omit None and blank strings, set
		    X-API-Key, issue a GET, and validate list fields/data arrays or an individual sensor
		    object.

		Args:
		    endpoint (str): Relative path sensors or sensors/{sensor_index}, with a digit-only
		        index; surrounding whitespace and slashes are removed.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Endpoint URL, request parameters, and raw decoded response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.endpoint = self.validate_endpoint( endpoint )

			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.headers[ 'X-API-Key' ] = self.api_key
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ) or self.payload.get( 'error' ) or self.payload.get( 'errors' ):
				raise ValueError( 'PurpleAir returned an invalid or error response.' )
			if self.endpoint == 'sensors':
				if not isinstance( self.payload.get( 'fields' ), list ) or not isinstance( self.payload.get( 'data' ), list ):
					raise ValueError( 'PurpleAir sensor list must contain fields and data arrays.' )
				if any( not isinstance( row, list ) for row in self.payload[ 'data' ] ):
					raise ValueError( 'PurpleAir sensor rows must be arrays.' )
			else:
				if not isinstance( self.payload.get( 'sensor' ), dict ):
					raise ValueError( 'PurpleAir sensor detail must contain a sensor object.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_sensors( self, nwlng: float, nwlat: float, selng: float, selat: float, location_type: int=0, max_age: int=0, modified_since: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve sensors within a geographic bounding box.

		Purpose:
		    Validate bounding coordinates and integer filters, normalize requested fields, request
		    sensors, and shape the returned array records.

		Args:
		    nwlng (float): Finite northwest longitude from -180 through 180 degrees; less than
		        selng; booleans are rejected.
		    nwlat (float): Finite northwest latitude from -90 through 90 degrees; greater than
		        selat; booleans are rejected.
		    selng (float): Finite southeast longitude from -180 through 180 degrees; booleans are
		        rejected.
		    selat (float): Finite southeast latitude from -90 through 90 degrees; booleans are
		        rejected.
		    location_type (int): Integer location type, 0 or 1; booleans are rejected. Defaults to
		        0.
		    max_age (int): Nonnegative integer age filter passed to PurpleAir. Defaults to 0.
		    modified_since (int): Nonnegative integer modification filter passed to PurpleAir.
		        Defaults to 0.
		    fields (str): Comma-separated field selection; blank uses operation-specific defaults.
		        Defaults to an empty string.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			default_fields = ('name,pm2.5,temperature,humidity,latitude,longitude,last_seen,'
			                  'location_type')
			
			self.mode = 'sensors'
			self.nwlng, self.nwlat, self.selng, self.selat = self.validate_bbox( nwlng=nwlng,
				nwlat=nwlat, selng=selng, selat=selat )
			self.location_type = self.validate_location_type( location_type )
			self.max_age = self.validate_non_negative_integer( 'max_age', max_age )
			self.modified_since = self.validate_non_negative_integer( 'modified_since',
				modified_since )
			self.fields = self.normalize_fields( fields=fields, default_fields=default_fields )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='sensors', params={ 'fields': self.fields,
					'location_type': self.location_type, 'nwlng': self.nwlng, 'nwlat': self.nwlat,
					'selng': self.selng, 'selat': self.selat, 'max_age': self.max_age,
					'modified_since': self.modified_since }, time=self.timeout )
			
			rows = self.shape_sensor_list_rows( self.payload )
			
			return self.package_response( rows=rows, params=self.params )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = "fetch_sensors( self, nwlng: float, nwlat: float, selng: float, selat: float, location_type: int=0, max_age: int=0, modified_since: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_sensor( self, sensor_index: int, fields: str='', time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve an individual PurpleAir sensor.

		Purpose:
		    Validate sensor index, normalize fields, request its endpoint, and shape detail
		    metadata.

		Args:
		    sensor_index (int): Positive integer sensor index; booleans are rejected.
		    fields (str): Comma-separated field selection; blank uses operation-specific defaults.
		        Defaults to an empty string.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			default_fields = ('name,model,hardware,pm2.5_cf_1_a,pm2.5_cf_1_b,temperature,'
			                  'humidity,pressure,latitude,longitude,last_seen,firmware_version,'
			                  'rssi')
			
			self.mode = 'sensor'
			self.sensor_index = self.validate_sensor_index( sensor_index )
			self.fields = self.normalize_fields( fields=fields, default_fields=default_fields )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint=f'sensors/{self.sensor_index}', params={
					'fields': self.fields }, time=self.timeout )
			
			rows = self.shape_sensor_detail_rows( self.payload )
			
			return self.package_response( rows=rows, params={ 'sensor_index': self.sensor_index,
					'fields': self.fields } )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = "fetch_sensor( self, sensor_index: int, fields: str='', time: int=20 ) -> (Dict[ str, Any ] | None)"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='sensors', sensor_index: int=None, nwlng: float | None = None, nwlat: float | None = None, selng: float | None = None, selat: float | None = None, location_type: int=0, max_age: int=0, modified_since: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a PurpleAir retrieval operation.

		Purpose:
		    Validate mode and require four bounding coordinates for sensors mode or an index for
		    sensor mode.

		Args:
		    mode (str): sensors or sensor; trimmed and lowercased. Defaults to 'sensors'.
		    sensor_index (int | None): Positive integer sensor index; booleans are rejected.
		        Defaults to None; required in sensor mode.
		    nwlng (float | None): Finite northwest longitude from -180 through 180 degrees; less
		        than selng; booleans are rejected. Defaults to None; required in sensors mode.
		    nwlat (float | None): Finite northwest latitude from -90 through 90 degrees; greater
		        than selat; booleans are rejected. Defaults to None; required in sensors mode.
		    selng (float | None): Finite southeast longitude from -180 through 180 degrees; booleans
		        are rejected. Defaults to None; required in sensors mode.
		    selat (float | None): Finite southeast latitude from -90 through 90 degrees; booleans
		        are rejected. Defaults to None; required in sensors mode.
		    location_type (int): Integer location type, 0 or 1; booleans are rejected. Defaults to
		        0.
		    max_age (int): Nonnegative integer age filter passed to PurpleAir. Defaults to 0.
		    modified_since (int): Nonnegative integer modification filter passed to PurpleAir.
		        Defaults to 0.
		    fields (str): Comma-separated field selection; blank uses operation-specific defaults.
		        Defaults to an empty string.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = self.validate_mode( mode )
			
			if self.mode == 'sensors':
				throw_if( 'nwlng', nwlng )
				throw_if( 'nwlat', nwlat )
				throw_if( 'selng', selng )
				throw_if( 'selat', selat )
				self.nwlng = nwlng
				self.nwlat = nwlat
				self.selng = selng
				self.selat = selat
				self.location_type = location_type
				self.max_age = max_age
				self.modified_since = modified_since
				self.fields = fields

				throw_if( 'time', time )
				self.timeout = time
				return self.fetch_sensors( nwlng=self.nwlng, nwlat=self.nwlat,
					selng=self.selng, selat=self.selat, location_type=self.location_type,
					max_age=self.max_age, modified_since=self.modified_since,
					fields=self.fields, time=self.timeout )
			if self.mode == 'sensor':
				throw_if( 'sensor_index', sensor_index )
				self.sensor_index = sensor_index
				self.fields = fields
				self.timeout = time
				return self.fetch_sensor( sensor_index=self.sensor_index,
					fields=self.fields, time=self.timeout )
			
			raise ValueError( "Unsupported PurpleAir mode. Use 'sensors' or 'sensor'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = "fetch( self, mode: str='sensors', sensor_index: int=None, nwlng: float | None = None, nwlat: float | None = None, selng: float | None = None, selat: float | None = None, location_type: int=0, max_age: int=0, modified_since: int=0, fields: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'PurpleAir'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class OpenAQ( Fetcher ):
	"""OpenAQ v3 air-quality retrieval adapter.

	Purpose:
	    Retrieve country, provider, parameter, location, and latest-measurement resources through
	    OpenAQ v3, validate geographic and pagination arguments, and package shaped records.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    api_key (Optional[str]): Provider credential loaded from configuration; sent in
	        X-API-Key.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected relative API endpoint.
	    location_id (Optional[int]): Selected location identifier.
	    parameter_id (Optional[int]): Selected measurement parameter identifier.
	    country_id (Optional[int]): Optional country identifier filter.
	    coordinates (Optional[str]): Normalized latitude,longitude selection.
	    radius (Optional[int]): Requested integer search radius.
	    providers_id (Optional[str]): Optional provider identifier filter text.
	    parameters_id (Optional[str]): Optional parameter identifier filter text.
	    limit (Optional[int]): Requested page size.
	    page (Optional[int]): Requested one-based page number.
	    params (Optional[Dict[str, Any]]): Latest request parameters or descriptive request
	        metadata.
	    payload (Optional[Any]): Latest decoded provider response.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	location_id: Optional[ int ]
	parameter_id: Optional[ int ]
	country_id: Optional[ int ]
	coordinates: Optional[ str ]
	radius: Optional[ int ]
	providers_id: Optional[ str ]
	parameters_id: Optional[ str ]
	limit: Optional[ int ]
	page: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured OpenAQ key, configure v3 endpoint and JSON headers, initialize
		    locations mode, radius of 25000, page size of 25, page one, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://api.openaq.org/v3'
		self.api_key = cfg.OPENAQ_API_KEY
		self.mode = 'locations'
		self.endpoint = ''
		self.location_id = None
		self.parameter_id = None
		self.country_id = None
		self.coordinates = ''
		self.radius = 25000
		self.providers_id = ''
		self.parameters_id = ''
		self.limit = 25
		self.page = 1
		self.params = { }
		self.payload = { }
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents }
		
		if self.api_key:
			self.headers[ 'X-API-Key' ] = self.api_key
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'api_key', 'mode', 'endpoint', 'location_id', 'parameter_id',
				'country_id', 'coordinates', 'radius', 'providers_id', 'parameters_id', 'limit',
				'page', 'params', 'payload', 'result', 'response', 'url', 'timeout', 'agents',
				'headers', 'validate_mode', 'validate_endpoint',
				'validate_positive_integer', 'validate_non_negative_integer',
				'validate_coordinates', 'validate_radius', 'coalesce_results',
				'shape_resource_rows', 'shape_location_rows', 'shape_latest_rows',
				'summarize_rows',
				'package_response', 'request', 'fetch_countries', 'fetch_providers',
				'fetch_parameters', 'fetch_parameter_latest', 'fetch_locations', 'fetch_latest',
				'fetch', 'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate an OpenAQ retrieval mode.

		Purpose:
		    Trim and lowercase the mode and accept only supported operations.

		Args:
		    mode (str): countries, providers, parameters, parameter_latest, locations, or latest;
		        trimmed and lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			
			value = mode.strip( ).lower( )
			allowed = { 'countries', 'providers', 'parameters', 'parameter_latest', 'locations',
					'latest' }
			
			if value not in allowed:
				raise ValueError( 'Unsupported OpenAQ mode. Use countries, providers, parameters, '
				                  'parameter_latest, locations, or latest.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate an OpenAQ relative endpoint.

		Purpose:
		    Trim whitespace and surrounding slashes, reject full URLs and parent-directory markers,
		    and require a supported resource path.

		Args:
		    endpoint (str): Relative path countries, providers, parameters, locations,
		        parameters/{id}/latest, or locations/{id}/latest; IDs contain digits.

		Returns:
		    str: Normalized accepted endpoint.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			
			value = endpoint.strip( ).strip( '/' )
			
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'endpoint must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'endpoint cannot contain parent-directory markers.' )
			
			if not re.fullmatch( r'(countries|providers|parameters|locations|'
			                     r'parameters/\d+/latest|locations/\d+/latest)', value ):
				raise ValueError( 'Unsupported OpenAQ endpoint path.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_positive_integer( self, name: str, value: Any, maximum: int | None = None ) -> ( int):
		"""Validate a positive integer.

		Purpose:
		    Require a positive integer excluding booleans and enforce an optional validated positive
		    integer maximum.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (Any): Positive integer to validate.
		    maximum (int | None): Optional positive integer upper bound. Defaults to None.

		Returns:
		    int: Original accepted integer.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )

			
			throw_if( name, value )

			if maximum is not None:
				throw_if( 'maximum', maximum )
				if type( maximum ) is not int or maximum < 1:
					raise ValueError( 'maximum must be a positive integer.' )

			
			if type( value ) is not int or value < 1:
				raise ValueError( f'{name} must be greater than or equal to 1.' )
			
			if maximum is not None and value > maximum:
				raise ValueError( f'{name} must be less than or equal to {maximum}.' )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'validate_positive_integer( self, name: str, value: Any, maximum: int | None = None ) -> ( int)'
			Logger( ).write( exception )
			raise exception
	
	def validate_non_negative_integer( self, name: str, value: Any ) -> int:
		"""Validate a nonnegative integer.

		Purpose:
		    Require an integer at least zero, excluding booleans.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (Any): Nonnegative integer to validate.

		Returns:
		    int: Original accepted integer.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )
			
			throw_if( name, value )
			
			number = value
			
			if type( number ) is not int or number < 0:
				raise ValueError( f'{name} must be greater than or equal to 0.' )
			
			return number
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'validate_non_negative_integer( self, name: str, value: Any ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_coordinates( self, coordinates: str='' ) -> str:
		"""Validate and serialize a coordinate pair.

		Purpose:
		    Allow blank or parse two finite numeric coordinates, validate geographic ranges, and
		    serialize latitude then longitude.

		Args:
		    coordinates (str): Optional comma-separated latitude,longitude pair; finite latitude
		        from -90 through 90 and longitude from -180 through 180 degrees. Defaults to an
		        empty string.

		Returns:
		    str: Comma-separated floating-point coordinates or an empty string.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			value = ( coordinates or '' ).strip( )
			
			if not value:
				return ''
			
			parts = [ part.strip( ) for part in value.split( ',' ) ]
			
			if len( parts ) != 2:
				raise ValueError( 'coordinates must use latitude,longitude format.' )
			
			latitude = float( parts[ 0 ] )
			longitude = float( parts[ 1 ] )
			if not math.isfinite( latitude ) or not math.isfinite( longitude ):
				raise ValueError( 'coordinates must contain finite values.' )
			
			if latitude < -90.0 or latitude > 90.0:
				raise ValueError( 'coordinates latitude must be between -90 and 90.' )
			
			if longitude < -180.0 or longitude > 180.0:
				raise ValueError( 'coordinates longitude must be between -180 and 180.' )
			
			return f'{latitude},{longitude}'
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = "validate_coordinates( self, coordinates: str='' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def validate_radius( self, radius: int ) -> int:
		"""Validate the geographic search radius.

		Purpose:
		    Delegate positive integer validation with maximum 100000.

		Args:
		    radius (int): Integer search radius from 1 through 100000; booleans are rejected.

		Returns:
		    int: Original accepted radius.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			value = self.validate_positive_integer( name='radius', value=radius, maximum=100000 )
			
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'validate_radius( self, radius: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def coalesce_results( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract OpenAQ result objects.

		Purpose:
		    Filter a list payload or a dictionary's results list to dictionaries; return an empty
		    list for other input.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Extracted result dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			if payload is None:
				return [ ]
			
			if isinstance( payload, list ):
				return [ item for item in payload if isinstance( item, dict ) ]
			
			if not isinstance( payload, dict ):
				return [ ]
			
			results = payload.get( 'results', [ ] )
			
			if isinstance( results, list ):
				return [ item for item in results if isinstance( item, dict ) ]
			
			return [ ]
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'coalesce_results( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_resource_rows( self, payload: Any, resource_name: str ) -> List[ Dict[ str, Any ] ]:
		"""Shape OpenAQ resource metadata.

		Purpose:
		    Extract identifier, name, display name, code, resource label and date bounds; include
		    parameter count and first parameter for nonempty parameter lists.

		Args:
		    payload (Any): Decoded provider response.
		    resource_name (str): Nonblank resource label.

		Returns:
		    List[Dict[str, Any]]: Display-oriented resource metadata rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'resource_name', resource_name )
			
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in self.coalesce_results( payload ):
				row = { 'Id': item.get( 'id', '' ), 'Name': item.get( 'name', '' ),
						'Display Name': (
								item.get( 'displayName', None ) or item.get( 'display_name',
							None ) or item.get( 'name', '' )),
						'Code': item.get( 'code', '' ), 'Resource': resource_name,
						'First Datetime': item.get( 'datetimeFirst', '' ),
						'Last Datetime': item.get( 'datetimeLast', '' ) }
				
				parameters = item.get( 'parameters', [ ] )
				if isinstance( parameters, list ) and parameters:
					row[ 'Parameter Count' ] = len( parameters )
					row[ 'First Parameter' ] = (parameters[ 0 ].get( 'name', '' ) if isinstance(
						parameters[ 0 ], dict ) else '')
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'shape_resource_rows( self, payload: Any, resource_name: str ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_location_rows( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Shape monitoring location records.

		Purpose:
		    Extract location identifiers, names, locality, timezone, country, provider or owner,
		    parameter metadata, coordinates and date bounds.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Display-oriented monitoring location rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in self.coalesce_results( payload ):
				coordinates = item.get( 'coordinates', { } ) or { }
				country = item.get( 'country', { } ) or { }
				provider = item.get( 'provider', { } ) or item.get( 'owner', { } ) or { }
				parameters = item.get( 'parameters', [ ] ) or [ ]
				
				latitude = (
						coordinates.get( 'latitude', None ) if isinstance( coordinates, dict )
						else None)
				longitude = (
						coordinates.get( 'longitude', None ) if isinstance( coordinates,
							dict ) else None)
				
				rows.append( { 'Location Id': item.get( 'id', '' ), 'Name': item.get( 'name', '' ),
						'Locality': item.get( 'locality', '' ),
						'Timezone': item.get( 'timezone', '' ), 'Country': (
								country.get( 'name', '' ) if isinstance( country, dict ) else
								country),
						'Country Code': (
								country.get( 'code', '' ) if isinstance( country, dict ) else ''),
						'Provider': (
								provider.get( 'name', '' ) if isinstance( provider, dict ) else
								provider),
						'Parameter Count': (
								len( parameters ) if isinstance( parameters, list ) else 0),
						'First Parameter': (parameters[
							0 ].get( 'name', '' ) if isinstance( parameters, list ) and parameters
						                             and isinstance(
							parameters[ 0 ], dict ) else ''), 'Latitude': latitude,
						'Longitude': longitude, 'First Datetime': item.get( 'datetimeFirst', '' ),
						'Last Datetime': item.get( 'datetimeLast', '' ) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'shape_location_rows( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def shape_latest_rows( self, payload: Any ) -> List[ Dict[ str, Any ] ]:
		"""Shape latest measurement records.

		Purpose:
		    Extract location and sensor identifiers, parameter metadata, measured value, unit, UTC
		    datetime or supplied date value, period label, and coordinates.

		Args:
		    payload (Any): Decoded provider response.

		Returns:
		    List[Dict[str, Any]]: Display-oriented latest measurement rows.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for item in self.coalesce_results( payload ):
				location = item.get( 'location', { } ) or { }
				parameter = item.get( 'parameter', { } ) or { }
				coordinates = item.get( 'coordinates', { } ) or { }
				period = item.get( 'period', { } ) or { }
				datetime_value = item.get( 'datetime', { } ) or item.get( 'date', { } ) or { }
				
				rows.append( { 'Location Id': (
						location.get( 'id', item.get( 'locationsId', '' ) ) if isinstance( location, dict ) else item.get( 'locationsId', '' )),
					'Sensor Id': item.get( 'sensorsId', '' ),
						'Name': (
						location.get( 'name', '' ) if isinstance( location, dict ) else ''),
						'Parameter': (
								parameter.get( 'name', '' ) if isinstance( parameter, dict ) else
								item.get( 'parameter', '' )),
						'Parameter Id': (
								parameter.get( 'id', '' ) if isinstance( parameter, dict ) else
								''),
						'Value': item.get( 'value', '' ), 'Unit': (
						item.get( 'unit', '' ) or (parameter.get( 'units', '' ) if isinstance( parameter, dict ) else '')),
						'Datetime': (
								datetime_value.get( 'utc', '' ) if isinstance( datetime_value,
									dict ) else datetime_value),
						'Period Label': (
								period.get( 'label', '' ) if isinstance( period, dict ) else ''),
						'Latitude': (
								coordinates.get( 'latitude', None ) if isinstance( coordinates,
									dict ) else None),
						'Longitude': (
								coordinates.get( 'longitude', None ) if isinstance( coordinates,
									dict ) else None) } )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'shape_latest_rows( self, payload: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped OpenAQ records.

		Purpose:
		    Count rows and extract first-row resource, parameter and country labels using alternate
		    display fields.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_result, first_parameter, and first_country.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_result = ''
			first_parameter = ''
			first_country = ''
			
			if rows:
				first_result = str(
					rows[ 0 ].get( 'Name', '' ) or rows[ 0 ].get( 'Display Name', '' ) or rows[
						0 ].get( 'Location Id', '' ) or rows[ 0 ].get( 'Id', '' ) or '' )
				first_parameter = str( rows[ 0 ].get( 'First Parameter', '' ) or rows[
					0 ].get( 'Parameter', '' ) or '' )
				first_country = str(
					rows[ 0 ].get( 'Country', '' ) or rows[ 0 ].get( 'Country Code', '' ) or '' )
			
			return { 'count': count, 'first_result': first_result,
					'first_parameter': first_parameter, 'first_country': first_country }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ], params: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""Package shaped OpenAQ records.

		Purpose:
		    Combine mode, URL, explicit parameter metadata or retained parameters, summary, shaped
		    rows, and raw provider response.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url,
					'params': params if params is not None else self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ], params: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute an OpenAQ v3 request.

		Purpose:
		    Validate endpoint, optional dictionary and timeout, omit None and blank strings, set
		    X-API-Key, issue a GET, and require a successful JSON object containing a results list
		    of dictionaries.

		Args:
		    endpoint (str): Relative path countries, providers, parameters, locations,
		        parameters/{id}/latest, or locations/{id}/latest; IDs contain digits.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Endpoint URL, request parameters, and raw decoded response.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			self.endpoint = endpoint.strip( ).strip( '/' )
			self.endpoint = self.validate_endpoint( self.endpoint )

			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.url = f'{self.base_url}/{self.endpoint}'
			self.params = { }
			
			
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.headers[ 'X-API-Key' ] = self.api_key
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if not isinstance( self.payload, dict ) or self.payload.get( 'error' ) or self.payload.get( 'errors' ):
				raise ValueError( 'OpenAQ returned an invalid or error response.' )
			if not isinstance( self.payload.get( 'results' ), list ) or any( not isinstance( item, dict ) for item in self.payload[ 'results' ] ):
				raise ValueError( 'OpenAQ results must be an array of objects.' )
			self.result = { 'url': self.url, 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'request( self, endpoint: str, params: Optional[ Dict[ str, Any ] ]=None, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_countries( self, providers_id: str='', parameters_id: str='', limit: int=100, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve OpenAQ country resources.

		Purpose:
		    Validate pagination, attach optional provider and parameter filters, request countries
		    and shape resource metadata.

		Args:
		    providers_id (str): Optional provider identifier filter text. Defaults to an empty
		        string.
		    parameters_id (str): Optional parameter identifier filter text. Defaults to an empty
		        string.
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        100.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'countries'
			self.providers_id = ( providers_id or '' ).strip( )
			self.parameters_id = ( parameters_id or '' ).strip( )
			self.limit = self.validate_positive_integer( 'limit', limit, maximum=1000 )
			self.page = self.validate_positive_integer( 'page', page )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='countries', params={ 'providers_id': self.providers_id,
					'parameters_id': self.parameters_id, 'limit': self.limit,
					'page': self.page }, time=self.timeout )
			
			rows = self.shape_resource_rows( self.payload, 'countries' )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = "fetch_countries( self, providers_id: str='', parameters_id: str='', limit: int=100, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_providers( self, limit: int=100, page: int=1, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve OpenAQ providers.

		Purpose:
		    Validate page size and page number, request the providers endpoint, and shape resource
		    metadata.

		Args:
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        100.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'providers'
			self.limit = self.validate_positive_integer( 'limit', limit, maximum=1000 )
			self.page = self.validate_positive_integer( 'page', page )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='providers', params={ 'limit': self.limit,
					'page': self.page }, time=self.timeout )
			
			rows = self.shape_resource_rows( self.payload, 'providers' )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'fetch_providers( self, limit: int=100, page: int=1, time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_parameters( self, limit: int=100, page: int=1, time: int=20 ) -> (Dict[ str, Any ] | None):
		"""Retrieve OpenAQ parameters.

		Purpose:
		    Validate page size and page number, request the parameters endpoint, and shape resource
		    metadata.

		Args:
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        100.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'parameters'
			self.limit = self.validate_positive_integer( 'limit', limit, maximum=1000 )
			self.page = self.validate_positive_integer( 'page', page )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='parameters', params={ 'limit': self.limit,
					'page': self.page }, time=self.timeout )
			
			rows = self.shape_resource_rows( self.payload, 'parameters' )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'fetch_parameters( self, limit: int=100, page: int=1, time: int=20 ) -> (Dict[ str, Any ] | None)'
			Logger( ).write( exception )
			raise exception
	
	def fetch_parameter_latest( self, parameter_id: int, limit: int=100, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve latest measurements by parameter.

		Purpose:
		    Validate parameter identifier and pagination, request its latest endpoint, shape
		    measurement records, and retain descriptive parameter metadata.

		Args:
		    parameter_id (int): Positive integer parameter identifier.
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        100.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'parameter_latest'
			self.parameter_id = self.validate_positive_integer( name='parameter_id',
				value=parameter_id )
			self.limit = self.validate_positive_integer( 'limit', limit, maximum=1000 )
			self.page = self.validate_positive_integer( 'page', page )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint=f'parameters/{self.parameter_id}/latest', params={
					'limit': self.limit, 'page': self.page }, time=self.timeout )
			
			rows = self.shape_latest_rows( self.payload )
			
			return self.package_response( rows=rows, params={ 'parameter_id': self.parameter_id,
					'limit': self.limit, 'page': self.page } )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'fetch_parameter_latest( self, parameter_id: int, limit: int=100, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_locations( self, country_id: int=None, coordinates: str='', radius: int=25000, providers_id: str='', parameters_id: str='', limit: int=25, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve OpenAQ monitoring locations.

		Purpose:
		    Validate optional country identifier, coordinate pair, radius and pagination, attach
		    optional provider and parameter filters, and shape location results. Radius is submitted
		    even when coordinates are omitted.

		Args:
		    country_id (int | None): Optional positive integer country identifier. Defaults to None.
		    coordinates (str): Optional comma-separated latitude,longitude pair; finite latitude
		        from -90 through 90 and longitude from -180 through 180 degrees. Defaults to an
		        empty string.
		    radius (int): Integer search radius from 1 through 100000; booleans are rejected.
		        Defaults to 25000.
		    providers_id (str): Optional provider identifier filter text. Defaults to an empty
		        string.
		    parameters_id (str): Optional parameter identifier filter text. Defaults to an empty
		        string.
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        25.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'locations'
			self.country_id = (
					None if country_id is None else self.validate_positive_integer( 'country_id',
						country_id ))
			self.coordinates = self.validate_coordinates( coordinates )
			self.radius = self.validate_radius( radius )
			self.providers_id = ( providers_id or '' ).strip( )
			self.parameters_id = ( parameters_id or '' ).strip( )
			self.limit = self.validate_positive_integer( 'limit', limit, maximum=1000 )
			self.page = self.validate_positive_integer( 'page', page )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint='locations', params={ 'country_id': self.country_id,
					'coordinates': self.coordinates, 'radius': self.radius,
					'providers_id': self.providers_id, 'parameters_id': self.parameters_id,
					'limit': self.limit, 'page': self.page }, time=self.timeout )
			
			rows = self.shape_location_rows( self.payload )
			
			return self.package_response( rows )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = "fetch_locations( self, country_id: int=None, coordinates: str='', radius: int=25000, providers_id: str='', parameters_id: str='', limit: int=25, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_latest( self, location_id: int, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve latest measurements by location.

		Purpose:
		    Validate the location identifier, request its latest endpoint without pagination
		    arguments, and shape measurement records.

		Args:
		    location_id (int): Positive integer location identifier.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'latest'
			self.location_id = self.validate_positive_integer( name='location_id',
				value=location_id )

			throw_if( 'time', time )
			self.timeout = time
			
			self.request( endpoint=f'locations/{self.location_id}/latest', params={ },
				time=self.timeout )
			
			rows = self.shape_latest_rows( self.payload )
			
			return self.package_response( rows=rows, params={ 'location_id': self.location_id } )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'fetch_latest( self, location_id: int, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='locations', location_id: int=None, parameter_id: int=None, country_id: int=None, coordinates: str='', radius: int=25000, providers_id: str = '', parameters_id: str='', limit: int=25, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an OpenAQ retrieval operation.

		Purpose:
		    Validate mode and forward applicable resource, location, parameter, geographic and
		    pagination arguments.

		Args:
		    mode (str): countries, providers, parameters, parameter_latest, locations, or latest;
		        trimmed and lowercased. Defaults to 'locations'.
		    location_id (int | None): Positive integer location identifier. Defaults to None;
		        required in latest mode.
		    parameter_id (int | None): Positive integer parameter identifier. Defaults to None;
		        required in parameter_latest mode.
		    country_id (int | None): Optional positive integer country identifier. Defaults to None.
		    coordinates (str): Optional comma-separated latitude,longitude pair; finite latitude
		        from -90 through 90 and longitude from -180 through 180 degrees. Defaults to an
		        empty string.
		    radius (int): Integer search radius from 1 through 100000; booleans are rejected.
		        Defaults to 25000.
		    providers_id (str): Optional provider identifier filter text. Defaults to an empty
		        string.
		    parameters_id (str): Optional parameter identifier filter text. Defaults to an empty
		        string.
		    limit (int): Integer page size from 1 through 1000; booleans are rejected. Defaults to
		        25.
		    page (int): Positive integer page number; booleans are rejected. Defaults to 1.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, endpoint URL, request parameters, summary, shaped rows, and raw
		        provider data.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = self.validate_mode( mode )
			
			if self.mode == 'countries':
				return self.fetch_countries( providers_id=providers_id,
					parameters_id=parameters_id, limit=limit, page=page, time=time )
			
			if self.mode == 'providers':
				return self.fetch_providers( limit=limit, page=page, time=time )
			
			if self.mode == 'parameters':
				return self.fetch_parameters( limit=limit, page=page, time=time )
			
			if self.mode == 'parameter_latest':
				return self.fetch_parameter_latest( parameter_id=parameter_id, limit=limit,
					page=page, time=time )
			
			if self.mode == 'locations':
				return self.fetch_locations( country_id=country_id, coordinates=coordinates,
					radius=radius, providers_id=providers_id, parameters_id=parameters_id,
					limit=limit, page=page, time=time )
			
			if self.mode == 'latest':
				return self.fetch_latest( location_id=location_id, time=time )
			
			raise ValueError( 'Unsupported OpenAQ mode. Use countries, providers, parameters, '
			                  'parameter_latest, locations, or latest.' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = "fetch( self, mode: str='locations', location_id: int=None, parameter_id: int=None, country_id: int=None, coordinates: str='', radius: int=25000, providers_id: str = '', parameters_id: str='', limit: int=25, page: int=1, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a JSON-compatible function description with object parameter properties and
		    required argument names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Nonempty parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenAQ'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class Firms( Fetcher ):
	"""NASA FIRMS fire detection and data-availability adapter.

	Purpose:
	    Retrieve CSV area detections and sensor data-availability records through NASA FIRMS,
	    validate source and geographic filters, shape records, and redact the map-key path
	    segment in returned URLs.

	Attributes:
	    base_url (Optional[str]): Base provider API URL.
	    map_key (Optional[str]): FIRMS map key loaded from configuration.
	    mode (Optional[str]): Selected retrieval operation.
	    source (Optional[str]): Selected fire-detection product source.
	    area_coordinates (Optional[str]): Normalized world selector or west,south,east,north
	        bounds.
	    day_range (Optional[int]): Requested number of days.
	    date (Optional[str]): Optional requested calendar date string.
	    sensor (Optional[str]): Selected data-availability sensor.
	    params (Optional[Dict[str, Any]]): Descriptive request metadata; filters are encoded in
	        the URL.
	    payload (Optional[Any]): Latest raw CSV response text.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL, including the map key before output redaction.
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP request headers.
	"""
	base_url: Optional[ str ]
	map_key: Optional[ str ]
	mode: Optional[ str ]
	source: Optional[ str ]
	area_coordinates: Optional[ str ]
	day_range: Optional[ int ]
	date: Optional[ str ]
	sensor: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured FIRMS map key, configure the API base URL and CSV headers,
		    initialize area mode, VIIRS_SNPP_NRT, worldwide bounds, one-day range, ALL sensor
		    selection, and twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.base_url = 'https://firms.modaps.eosdis.nasa.gov/api'
		self.map_key = cfg.FIRMS_MAP_KEY
		self.mode = 'area'
		self.source = 'VIIRS_SNPP_NRT'
		self.area_coordinates = 'world'
		self.day_range = 1
		self.date = ''
		self.sensor = 'ALL'
		self.params = { }
		self.payload = ''
		self.result = { }
		self.response = None
		self.url = None
		self.timeout = 20
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'text/csv', 'User-Agent': self.agents }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'base_url', 'map_key', 'mode', 'source', 'area_coordinates', 'day_range', 'date',
				'sensor', 'params', 'payload', 'result', 'response', 'url', 'timeout', 'agents',
				'headers', 'validate_mode', 'validate_source',
				'validate_sensor', 'validate_day_range', 'validate_date',
				'validate_area_coordinates', 'csv_to_rows', 'summarize_rows', 'package_response',
				'request_csv', 'fetch_area', 'fetch_data_availability', 'fetch', 'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate the FIRMS retrieval mode.

		Purpose:
		    Trim and lowercase the supplied mode and accept only area or data-availability.

		Args:
		    mode (str): area or data-availability; trimmed and lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			value = mode.strip( ).lower( )
			allowed = { 'area', 'data-availability' }
			if value not in allowed:
				raise ValueError( "Unsupported FIRMS mode. Use 'area' or 'data-availability'." )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_source( self, source: str ) -> str:
		"""Validate a FIRMS product source.

		Purpose:
		    Trim and uppercase the supplied source and accept only the supported LANDSAT, MODIS and
		    VIIRS identifiers.

		Args:
		    source (str): LANDSAT_NRT, MODIS_NRT, MODIS_SP, VIIRS_NOAA20_NRT, VIIRS_NOAA20_SP,
		        VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT, or VIIRS_SNPP_SP; trimmed and uppercased.

		Returns:
		    str: Normalized accepted source.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'source', source )
			value = source.strip( ).upper( )
			allowed = { 'LANDSAT_NRT', 'MODIS_NRT', 'MODIS_SP', 'VIIRS_NOAA20_NRT',
					'VIIRS_NOAA20_SP', 'VIIRS_NOAA21_NRT', 'VIIRS_SNPP_NRT', 'VIIRS_SNPP_SP' }
			
			if value not in allowed:
				raise ValueError( 'Unsupported FIRMS source. Use one of the app-supported '
				                  'LANDSAT, MODIS, or VIIRS source identifiers.' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'validate_source( self, source: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_sensor( self, sensor: str ) -> str:
		"""Validate a FIRMS sensor selector.

		Purpose:
		    Trim and uppercase the supplied sensor and accept only supported sensor identifiers.

		Args:
		    sensor (str): ALL, LANDSAT, MODIS, VIIRS_SNPP, VIIRS_NOAA20, or VIIRS_NOAA21; trimmed
		        and uppercased.

		Returns:
		    str: Normalized accepted sensor.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'sensor', sensor )
			value = sensor.strip( ).upper( )
			allowed = { 'ALL', 'LANDSAT', 'MODIS', 'VIIRS_SNPP', 'VIIRS_NOAA20', 'VIIRS_NOAA21' }
			if value not in allowed:
				raise ValueError( 'Unsupported FIRMS sensor. Use ALL, LANDSAT, MODIS, '
				                  'VIIRS_SNPP, VIIRS_NOAA20, or VIIRS_NOAA21.' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'validate_sensor( self, sensor: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_day_range( self, day_range: int ) -> int:
		"""Validate the requested day range.

		Purpose:
		    Require an integer from one through five, excluding booleans.

		Args:
		    day_range (int): Integer from 1 through 5; booleans are rejected.

		Returns:
		    int: Original accepted day range.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'day_range', day_range )
			value = day_range
			if type( value ) is not int or value < 1 or value > 5:
				raise ValueError( 'day_range must be between 1 and 5.' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'validate_day_range( self, day_range: int ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_date( self, date: str='' ) -> str:
		"""Validate an optional FIRMS date.

		Purpose:
		    Trim the supplied value, allow an empty string, and parse nonempty values with the
		    YYYY-MM-DD format.

		Args:
		    date (str): Optional date parsed with YYYY-MM-DD after trimming. Defaults to an empty
		        string.

		Returns:
		    str: Trimmed date string or an empty string.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			value = ( date or '' ).strip( )
			if not value:
				return ''
			
			dt.datetime.strptime( value, '%Y-%m-%d' )
			return value
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = "validate_date( self, date: str='' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def validate_area_coordinates( self, area_coordinates: str='world' ) -> str:
		"""Validate the FIRMS geographic selector.

		Purpose:
		    Normalize world to lowercase or parse four finite geographic coordinates, require strict
		    west-to-east and south-to-north ordering, and serialize using general numeric format.

		Args:
		    area_coordinates (str): Case-insensitive world or comma-separated west,south,east,north
		        bounds; finite longitude from -180 through 180 and latitude from -90 through 90,
		        with strictly increasing axes. Defaults to 'world'.

		Returns:
		    str: World selector or comma-separated numeric bounds.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'area_coordinates', area_coordinates )
			value = area_coordinates.strip( )
			if value.lower( ) == 'world':
				return 'world'
			
			parts = [ part.strip( ) for part in value.split( ',' ) ]
			if len( parts ) != 4:
				raise ValueError( 'area_coordinates must be world or west,south,east,north.' )
			
			west, south, east, north = [ float( part ) for part in parts ]
			if not all( math.isfinite( x ) for x in (west, south, east, north) ):
				raise ValueError( 'area_coordinates must be finite.' )
			
			if west < -180.0 or west > 180.0:
				raise ValueError( 'west longitude must be between -180 and 180.' )
			
			if east < -180.0 or east > 180.0:
				raise ValueError( 'east longitude must be between -180 and 180.' )
			
			if south < -90.0 or south > 90.0:
				raise ValueError( 'south latitude must be between -90 and 90.' )
			
			if north < -90.0 or north > 90.0:
				raise ValueError( 'north latitude must be between -90 and 90.' )
			
			if west >= east:
				raise ValueError( 'west longitude must be less than east longitude.' )
			
			if south >= north:
				raise ValueError( 'south latitude must be less than north latitude.' )
			
			return f'{west:g},{south:g},{east:g},{north:g}'
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = "validate_area_coordinates( self, area_coordinates: str='world' ) -> str"
			Logger( ).write( exception )
			raise exception
	
	def csv_to_rows( self, csv_text: str ) -> List[ Dict[ str, Any ] ]:
		"""Parse FIRMS CSV records.

		Purpose:
		    Return no rows for blank text; require at least two nonblank header fields and reject
		    records containing extra cells. Convert field names to title case with spaces, retaining
		    CSV string values and None for missing cells.

		Args:
		    csv_text (str): Raw CSV response text.

		Returns:
		    List[Dict[str, Any]]: Display-oriented CSV record dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			text = csv_text or ''
			
			if not text.strip( ):
				return [ ]
			
			reader = csv.DictReader( io.StringIO( text ) )
			if not reader.fieldnames or len( reader.fieldnames ) < 2 or any( not name or not name.strip( ) for name in reader.fieldnames ):
				raise ValueError( 'FIRMS response is not valid CSV data.' )
			rows: List[ Dict[ str, Any ] ] = [ ]
			
			for record in reader:
				if not isinstance( record, dict ) or None in record:
					raise ValueError( 'FIRMS response contains malformed rows.' )
				
				row: Dict[ str, Any ] = { }
				
				for key, value in record.items( ):
					friendly_key = str( key ).replace( '_', ' ' ).title( )
					row[ friendly_key ] = value
				
				rows.append( row )
			
			return rows
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'csv_to_rows( self, csv_text: str ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Summarize shaped FIRMS records.

		Purpose:
		    Count rows and extract date, sensor, latitude and longitude from the first row using
		    alternative display field names and truthiness-based fallbacks.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Count, first_date, first_sensor, first_lat, and first_lon.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			count = len( rows or [ ] )
			first_date = ''
			first_sensor = ''
			first_lat = ''
			first_lon = ''
			
			if rows:
				first_date = str(
					rows[ 0 ].get( 'Acq Date', '' ) or rows[ 0 ].get( 'Date', '' ) or rows[
						0 ].get( 'Start Date', '' ) or '' )
				first_sensor = str(
					rows[ 0 ].get( 'Sensor', '' ) or rows[ 0 ].get( 'Satellite', '' ) or rows[
						0 ].get( 'Source', '' ) or '' )
				first_lat = str( rows[ 0 ].get( 'Latitude', '' ) or '' )
				first_lon = str( rows[ 0 ].get( 'Longitude', '' ) or '' )
			
			return { 'count': count, 'first_date': first_date, 'first_sensor': first_sensor,
					'first_lat': first_lat, 'first_lon': first_lon }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'summarize_rows( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]:
		"""Package shaped FIRMS records.

		Purpose:
		    Combine mode, redacted request URL, descriptive parameters, summary, rows and raw CSV
		    text.

		Args:
		    rows (List[Dict[str, Any]]): Shaped row dictionaries.

		Returns:
		    Dict[str, Any]: Mode, URL with map-key path segment redacted, descriptive parameters,
		        summary, shaped rows, and raw CSV text.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.result = { 'mode': self.mode, 'url': self.url.replace( f'/{self.map_key}/', '/[REDACTED]/' ), 'params': self.params,
					'summary': self.summarize_rows( rows ), 'rows': rows, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'package_response( self, rows: List[ Dict[ str, Any ] ] ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def request_csv( self, url: str, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Execute a FIRMS CSV request.

		Purpose:
		    Trim the URL, validate a positive integer timeout, issue a GET, check HTTP status, and
		    retain response text without parsing it. Redact the configured map-key path segment in
		    the returned URL.

		Args:
		    url (str): Nonblank FIRMS request URL.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Redacted URL, retained descriptive parameters, and raw response text.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'url', url )

			self.url = url.strip( )
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.text or ''
			self.result = { 'url': self.url.replace( f'/{self.map_key}/', '/[REDACTED]/' ), 'params': self.params, 'raw': self.payload }
			
			return self.result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'request_csv( self, url: str, time: int=20 ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_area( self, source: str, area_coordinates: str='world', day_range: int=1, date: str='', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve FIRMS area fire-detection records.

		Purpose:
		    Validate source, geographic selector, integer day range and optional date; construct the
		    authenticated area CSV path, retrieve and parse records, and package the response.

		Args:
		    source (str): LANDSAT_NRT, MODIS_NRT, MODIS_SP, VIIRS_NOAA20_NRT, VIIRS_NOAA20_SP,
		        VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT, or VIIRS_SNPP_SP; trimmed and uppercased.
		    area_coordinates (str): Case-insensitive world or comma-separated west,south,east,north
		        bounds; finite longitude from -180 through 180 and latitude from -90 through 90,
		        with strictly increasing axes. Defaults to 'world'.
		    day_range (int): Integer from 1 through 5; booleans are rejected. Defaults to 1.
		    date (str): Optional date parsed with YYYY-MM-DD after trimming. Defaults to an empty
		        string.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, URL with map-key path segment redacted, descriptive parameters,
		        summary, shaped rows, and raw CSV text.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'area'
			throw_if( 'source', source )
			self.source = source
			self.source = self.validate_source( self.source )
			self.area_coordinates = self.validate_area_coordinates( area_coordinates )
			self.day_range = self.validate_day_range( day_range )
			self.date = self.validate_date( date )

			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'source': self.source, 'area_coordinates': self.area_coordinates,
					'day_range': self.day_range, 'date': self.date }
			self.url = (f'{self.base_url}/area/csv/{self.map_key}/'
			            f'{self.source}/{self.area_coordinates}/{self.day_range}')
			
			if self.date:
				self.url = f'{self.url}/{self.date}'
			
			self.request_csv( url=self.url, time=self.timeout )
			
			rows = self.csv_to_rows( self.payload )
			
			return self.package_response( rows )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = "fetch_area( self, source: str, area_coordinates: str='world', day_range: int=1, date: str='', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_data_availability( self, sensor: str='ALL', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Retrieve FIRMS sensor data availability.

		Purpose:
		    Validate sensor, construct the authenticated data_availability CSV path, retrieve and
		    parse records, and package the result.

		Args:
		    sensor (str): ALL, LANDSAT, MODIS, VIIRS_SNPP, VIIRS_NOAA20, or VIIRS_NOAA21; trimmed
		        and uppercased. Defaults to 'ALL'.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, URL with map-key path segment redacted, descriptive parameters,
		        summary, shaped rows, and raw CSV text.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'data-availability'
			self.sensor = self.validate_sensor( sensor )

			throw_if( 'time', time )
			self.timeout = time
			self.params = { 'sensor': self.sensor }
			self.url = (f'{self.base_url}/data_availability/csv/'
			            f'{self.map_key}/{self.sensor}')
			
			self.request_csv( url=self.url, time=self.timeout )
			rows = self.csv_to_rows( self.payload )
			return self.package_response( rows )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = "fetch_data_availability( self, sensor: str='ALL', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='area', source: str='VIIRS_SNPP_NRT', area_coordinates: str = 'world', day_range: int=1, date: str='', sensor: str='ALL', time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch a FIRMS retrieval operation.

		Purpose:
		    Validate mode and select area detections or sensor data availability, forwarding
		    applicable arguments.

		Args:
		    mode (str): area or data-availability; trimmed and lowercased. Defaults to 'area'.
		    source (str): LANDSAT_NRT, MODIS_NRT, MODIS_SP, VIIRS_NOAA20_NRT, VIIRS_NOAA20_SP,
		        VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT, or VIIRS_SNPP_SP; trimmed and uppercased. Defaults
		        to 'VIIRS_SNPP_NRT'.
		    area_coordinates (str): Case-insensitive world or comma-separated west,south,east,north
		        bounds; finite longitude from -180 through 180 and latitude from -90 through 90,
		        with strictly increasing axes. Defaults to 'world'.
		    day_range (int): Integer from 1 through 5; booleans are rejected. Defaults to 1.
		    date (str): Optional date parsed with YYYY-MM-DD after trimming. Defaults to an empty
		        string.
		    sensor (str): ALL, LANDSAT, MODIS, VIIRS_SNPP, VIIRS_NOAA20, or VIIRS_NOAA21; trimmed
		        and uppercased. Defaults to 'ALL'.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Mode, URL with map-key path segment redacted, descriptive parameters,
		        summary, shaped rows, and raw CSV text.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = self.validate_mode( mode )
			if self.mode == 'area':
				return self.fetch_area( source=source, area_coordinates=area_coordinates,
					day_range=day_range, date=date, time=time )
			
			if self.mode == 'data-availability':
				return self.fetch_data_availability( sensor=sensor, time=time )
			
			raise ValueError( "Unsupported FIRMS mode. Use 'area' or 'data-availability'." )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = "fetch( self, mode: str='area', source: str='VIIRS_SNPP_NRT', area_coordinates: str = 'world', day_range: int=1, date: str='', sensor: str='ALL', time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a function description with object parameter definitions and required argument
		    names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Firms'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception

class OpenSky( Fetcher ):
	"""OpenSky Network state, flight, and aircraft track adapter.

	Purpose:
	    Retrieve aircraft state vectors, historical aircraft flights, airport arrivals and
	    departures, and aircraft tracks through OpenSky API endpoints, using OAuth2 client
	    credentials when configured and shaping provider responses into dictionaries.

	Attributes:
	    timeout (Optional[int]): HTTP request timeout in seconds.
	    base_url (Optional[str]): Base provider API URL.
	    token_url (Optional[str]): OAuth2 client-credentials token endpoint.
	    client_id (Optional[str]): Configured OpenSky OAuth2 client identifier.
	    client_secret (Optional[str]): Configured OAuth2 client secret or legacy credentials
	        fallback.
	    access_token (Optional[str]): Latest OAuth2 bearer access token, or None.
	    mode (Optional[str]): Selected retrieval operation.
	    endpoint (Optional[str]): Selected supported OpenSky API path.
	    icao24 (Optional[str]): Selected normalized six-character hexadecimal aircraft address.
	    airport (Optional[str]): Selected normalized four-character airport code.
	    begin (Optional[int]): Beginning Unix epoch seconds for flight queries.
	    end (Optional[int]): Ending Unix epoch seconds for flight queries.
	    time_value (Optional[int]): Optional snapshot or track Unix epoch seconds.
	    lamin (Optional[float]): Minimum bounding latitude.
	    lomin (Optional[float]): Minimum bounding longitude.
	    lamax (Optional[float]): Maximum bounding latitude.
	    lomax (Optional[float]): Maximum bounding longitude.
	    extended (Optional[bool]): Extended state-vector request flag.
	    params (Optional[Dict[str, Any]]): Latest request parameters or descriptive request
	        metadata.
	    payload (Optional[Any]): Latest decoded API response, which may be an object, list, or
	        None.
	    result (Optional[Dict[str, Any]]): Latest packaged provider response.
	    response (Optional[Response]): Latest HTTP response.
	    url (Optional[str]): Latest request URL.
	    agents (Optional[str]): Configured HTTP user-agent.
	    headers (Optional[Dict[str, Any]]): HTTP headers, including a bearer token after
	        successful authentication.
	"""
	token_url: Optional[ str ]
	base_url: Optional[ str ]
	client_id: Optional[ str ]
	client_secret: Optional[ str ]
	access_token: Optional[ str ]
	mode: Optional[ str ]
	endpoint: Optional[ str ]
	icao24: Optional[ str ]
	airport: Optional[ str ]
	begin: Optional[ int ]
	end: Optional[ int ]
	time_value: Optional[ int ]
	lamin: Optional[ float ]
	lomin: Optional[ float ]
	lamax: Optional[ float ]
	lomax: Optional[ float ]
	extended: Optional[ bool ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Any ]
	result: Optional[ Dict[ str, Any ] ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize provider state.

		Purpose:
		    Load the configured OpenSky client ID and client secret (or legacy credential fallback),
		    configure API and OAuth2 endpoints and JSON headers, initialize states_bbox mode and
		    empty filter state, and set a twenty-second timeout.

		Returns:
		    None: Initializes instance state.
		"""
		super( ).__init__( )
		self.timeout = 20
		self.base_url = 'https://opensky-network.org/api'
		self.token_url = ('https://auth.opensky-network.org/auth/realms/opensky-network/'
		                  'protocol/openid-connect/token')
		self.client_id = cfg.OPENSKY_API_CLIENT_ID
		self.client_secret = cfg.OPENSKY_API_CLIENT_SECRET or cfg.OPENSKY_API_CREDENTIALS
		self.access_token = None
		self.mode = 'states_bbox'
		self.endpoint = ''
		self.icao24 = ''
		self.airport = ''
		self.begin = None
		self.end = None
		self.time_value = None
		self.lamin = None
		self.lomin = None
		self.lamax = None
		self.lomax = None
		self.extended = False
		self.params = { }
		self.payload = None
		self.result = { }
		self.response = None
		self.url = None
		self.agents = cfg.AGENTS
		self.headers = { 'Accept': 'application/json', 'User-Agent': self.agents, }
	
	def __dir__( self ) -> List[ str ]:
		"""Return exposed member names.

		Purpose:
		    Provide the explicit member list returned by dir().

		Returns:
		    List[str]: Ordered member names.
		"""
		return [ 'timeout', 'headers', 'response', 'url', 'result', 'query', 'token_url',
				'base_url', 'client_id', 'client_secret', 'access_token', 'mode', 'endpoint',
				'icao24', 'airport', 'begin', 'end', 'time_value', 'lamin', 'lomin', 'lamax',
				'lomax', 'extended', 'params', 'payload', 'agents', 'validate_mode',
				'validate_endpoint', 'validate_icao24', 'validate_airport', 'validate_epoch',
				'validate_time_range', 'validate_latitude', 'validate_longitude', 'validate_bbox',
				'authenticate', 'request', 'normalize_states',
				'normalize_flights', 'normalize_track', 'fetch_states', 'fetch_flights_aircraft',
				'fetch_arrivals_airport', 'fetch_departures_airport', 'fetch_track_aircraft',
				'fetch', 'create_schema' ]
	
	def validate_mode( self, mode: str ) -> str:
		"""Validate an OpenSky retrieval mode.

		Purpose:
		    Trim and lowercase the supplied mode and accept only supported state, flight, arrival,
		    departure, and track operations.

		Args:
		    mode (str): states_bbox, flights_aircraft, arrivals_airport, departures_airport, or
		        track_aircraft; trimmed and lowercased.

		Returns:
		    str: Normalized accepted mode.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'mode', mode )
			value = mode.strip( ).lower( )
			allowed = { 'states_bbox', 'flights_aircraft', 'arrivals_airport',
					'departures_airport',
					'track_aircraft' }
			
			if value not in allowed:
				raise ValueError( "Unsupported OpenSky mode. Use 'states_bbox', "
				                  "'flights_aircraft', 'arrivals_airport', "
				                  "'departures_airport', or 'track_aircraft'." )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_mode( self, mode: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_endpoint( self, endpoint: str ) -> str:
		"""Validate an OpenSky endpoint.

		Purpose:
		    Trim whitespace, reject full URLs and parent-directory markers, and require an exact
		    supported API path.

		Args:
		    endpoint (str): One of /states/all, /flights/aircraft, /flights/arrival,
		        /flights/departure, or /tracks/all.

		Returns:
		    str: Trimmed accepted endpoint.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'endpoint', endpoint )
			value = endpoint.strip( )
			if value.startswith( 'http://' ) or value.startswith( 'https://' ):
				raise ValueError( 'endpoint must be a path segment, not a full URL.' )
			
			if '..' in value:
				raise ValueError( 'endpoint cannot contain parent-directory markers.' )
			
			allowed = { '/states/all', '/flights/aircraft', '/flights/arrival',
					'/flights/departure', '/tracks/all' }
			
			if value not in allowed:
				raise ValueError( 'Unsupported OpenSky endpoint path.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_endpoint( self, endpoint: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_icao24( self, icao24: str ) -> str:
		"""Validate an aircraft address.

		Purpose:
		    Trim and lowercase the input and require six hexadecimal characters.

		Args:
		    icao24 (str): Six-character hexadecimal aircraft address, trimmed and lowercased.

		Returns:
		    str: Normalized aircraft address.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'icao24', icao24 )
			value = icao24.strip( ).lower( )
			if not re.fullmatch( r'[0-9a-f]{6}', value ):
				raise ValueError( 'icao24 must be a six-character hexadecimal address.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_icao24( self, icao24: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_airport( self, airport: str ) -> str:
		"""Validate an airport code.

		Purpose:
		    Trim and uppercase the input and require four letters or digits.

		Args:
		    airport (str): Four-character alphanumeric ICAO airport code, trimmed and uppercased.

		Returns:
		    str: Normalized airport code.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'airport', airport )
			value = airport.strip( ).upper( )
			if not re.fullmatch( r'[A-Z0-9]{4}', value ):
				raise ValueError( 'airport must be a four-character ICAO airport code.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_airport( self, airport: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_epoch( self, name: str, value: Any ) -> int:
		"""Validate a Unix epoch argument.

		Purpose:
		    Require a nonnegative integer, excluding booleans.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (Any): Nonnegative Unix epoch seconds.

		Returns:
		    int: Original accepted epoch value.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )

			
			throw_if( name, value )

			if type( value ) is not int or value < 0:
				raise ValueError( f'{name} must be greater than or equal to 0.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_epoch( self, name: str, value: Any ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_time_range( self, begin: int, end: int ) -> Tuple[ int, int ]:
		"""Validate a flight query interval.

		Purpose:
		    Validate both epoch values and require begin strictly less than end.

		Args:
		    begin (int): Nonnegative Unix epoch seconds at the beginning of the interval; booleans
		        are rejected.
		    end (int): Nonnegative Unix epoch seconds strictly greater than begin; booleans are
		        rejected.

		Returns:
		    Tuple[int, int]: Original accepted beginning and ending epoch seconds.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			start = self.validate_epoch( 'begin', begin )
			stop = self.validate_epoch( 'end', end )
			
			if start >= stop:
				raise ValueError( 'begin must be less than end.' )
			
			return start, stop
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_time_range( self, begin: int, end: int ) -> Tuple[ int, int ]'
			Logger( ).write( exception )
			raise exception
	
	def validate_latitude( self, name: str, value: Any ) -> float:
		"""Validate a latitude value.

		Purpose:
		    Reject booleans, nonfinite values and values outside geographic latitude bounds.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (Any): Finite latitude from -90 through 90 degrees.

		Returns:
		    float: Original accepted numeric latitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )

			
			throw_if( name, value )

			if isinstance( value, bool ) or not math.isfinite( value ) or value < -90.0 or value > 90.0:
				raise ValueError( f'{name} must be between -90 and 90.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_latitude( self, name: str, value: Any ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_longitude( self, name: str, value: Any ) -> float:
		"""Validate a longitude value.

		Purpose:
		    Reject booleans, nonfinite values and values outside geographic longitude bounds.

		Args:
		    name (str): Nonblank argument label used in validation messages.
		    value (Any): Finite longitude from -180 through 180 degrees.

		Returns:
		    float: Original accepted numeric longitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			throw_if( 'name', name )

			
			throw_if( name, value )

			if isinstance( value, bool ) or not math.isfinite( value ) or value < -180.0 or value > 180.0:
				raise ValueError( f'{name} must be between -180 and 180.' )
			
			return value
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_longitude( self, name: str, value: Any ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def validate_bbox( self, lamin: float, lomin: float, lamax: float, lomax: float ) -> Tuple[ float, float, float, float ]:
		"""Validate a geographic bounding box.

		Purpose:
		    Validate all four coordinates and require strict increasing latitude and longitude
		    bounds.

		Args:
		    lamin (float): Finite minimum latitude from -90 through 90 degrees; booleans are
		        rejected.
		    lomin (float): Finite minimum longitude from -180 through 180 degrees; booleans are
		        rejected.
		    lamax (float): Finite maximum latitude from -90 through 90 degrees, greater than lamin;
		        booleans are rejected.
		    lomax (float): Finite maximum longitude from -180 through 180 degrees, greater than
		        lomin; booleans are rejected.

		Returns:
		    Tuple[float, float, float, float]: Original accepted minimum latitude, minimum
		        longitude, maximum latitude, and maximum longitude.

		Raises:
		    Error: Raised when the supplied value fails validation.
		"""
		try:
			min_lat = self.validate_latitude( 'lamin', lamin )
			min_lon = self.validate_longitude( 'lomin', lomin )
			max_lat = self.validate_latitude( 'lamax', lamax )
			max_lon = self.validate_longitude( 'lomax', lomax )
			
			if min_lat >= max_lat:
				raise ValueError( 'lamin must be less than lamax.' )
			
			if min_lon >= max_lon:
				raise ValueError( 'lomin must be less than lomax.' )
			
			return min_lat, min_lon, max_lat, max_lon
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'validate_bbox( self, lamin: float, lomin: float, lamax: float, lomax: float ) -> Tuple[ float, float, float, float ]'
			Logger( ).write( exception )
			raise exception
	
	def authenticate( self ) -> str | None:
		"""Obtain an OpenSky OAuth2 access token.

		Purpose:
		    Return None when either credential is absent; otherwise POST client credentials, check
		    HTTP status, require a JSON object with a truthy access_token, and retain the bearer
		    token and Authorization header.

		Returns:
		    str | None: Access token from the provider, or None when credentials are absent.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			if not self.client_id or not self.client_secret:
				return None
			
			self.response = requests.post( url=self.token_url, data={
					'grant_type': 'client_credentials', 'client_id': self.client_id,
					'client_secret': self.client_secret, }, headers={ 'Accept': 'application/json',
					'User-Agent': self.agents, }, timeout=self.timeout )
			self.response.raise_for_status( )
			token_payload = self.response.json( )
			if not isinstance( token_payload, dict ):
				raise ValueError( 'OpenSky OAuth2 returned invalid JSON.' )
			self.access_token = token_payload.get( 'access_token', None )
			if not self.access_token:
				raise ValueError( 'OpenSky OAuth2 response omitted access_token.' )
			if self.access_token:
				self.headers[ 'Authorization' ] = f'Bearer {self.access_token}'
			
			return self.access_token
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'authenticate( self ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def request( self, endpoint: str, params: Dict[ str, Any ] | None = None ) -> Any:
		"""Execute an OpenSky API request.

		Purpose:
		    Validate endpoint and optional parameters, clear previous bearer authorization,
		    authenticate again when credentials exist, omit None and blank strings, issue a GET
		    using the retained timeout, and validate endpoint-specific response structure.

		Args:
		    endpoint (str): One of /states/all, /flights/aircraft, /flights/arrival,
		        /flights/departure, or /tracks/all.
		    params (Dict[str, Any] | None): Optional query parameter dictionary. Defaults to None.

		Returns:
		    Any: Decoded provider JSON; state object, flight record list, or track object or None
		        depending on endpoint.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.endpoint = self.validate_endpoint( endpoint )
			if params is not None and not isinstance( params, dict ):
				raise ValueError( 'params must be a dictionary.' )
			self.headers.pop( 'Authorization', None )
			self.access_token = None
			self.authenticate( )
			self.url = f'{self.base_url}{self.endpoint}'
			self.params = { }
			for key, value in (params or { }).items( ):
				if value is None:
					continue
				
				if isinstance( value, str ) and not value.strip( ):
					continue
				
				self.params[ key ] = value
			
			self.response = requests.get( url=self.url, params=self.params, headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			if isinstance( self.payload, dict ) and (self.payload.get( 'error' ) or self.payload.get( 'errors' )):
				raise ValueError( 'OpenSky returned an API error.' )
			if self.endpoint == '/states/all' and (not isinstance( self.payload, dict ) or not isinstance( self.payload.get( 'states' ), (list, type( None )) )):
				raise ValueError( 'OpenSky states response is invalid.' )
			if self.endpoint.startswith( '/flights/' ) and (not isinstance( self.payload, list ) or any( not isinstance( item, dict ) for item in self.payload )):
				raise ValueError( 'OpenSky flights response is invalid.' )
			if self.endpoint == '/tracks/all' and self.payload is not None and (not isinstance( self.payload, dict ) or not isinstance( self.payload.get( 'path', [] ), list )):
				raise ValueError( 'OpenSky track response is invalid.' )
			return self.payload
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'request( self, endpoint: str, params: Dict[ str, Any ] | None = None ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def normalize_states( self, payload: Dict[ str, Any ] | None ) -> Dict[ str, Any ] | None:
		"""Shape OpenSky aircraft state vectors.

		Purpose:
		    Map the first seventeen positional state-vector fields to named dictionary keys, trim
		    callsigns, use None for missing positions, and skip non-list entries. Additional
		    positional fields are not retained.

		Args:
		    payload (Dict[str, Any] | None): Decoded state response object or None.

		Returns:
		    Dict[str, Any]: Mode, provider snapshot time, count, and normalized state-vector
		        dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			states = [ ]
			if isinstance( payload, dict ):
				states = payload.get( 'states', [ ] ) or [ ]
			
			items: List[ Dict[ str, Any ] ] = [ ]
			for row in states:
				if not isinstance( row, list ):
					continue
				
				items.append( { 'icao24': row[ 0 ] if len( row ) > 0 else None, 'callsign': (
					str( row[ 1 ] ).strip( ) if len( row ) > 1 and row[ 1 ] is not None else None),
						'origin_country': row[ 2 ] if len( row ) > 2 else None,
						'time_position': row[ 3 ] if len( row ) > 3 else None,
						'last_contact': row[ 4 ] if len( row ) > 4 else None,
						'longitude': row[ 5 ] if len( row ) > 5 else None,
						'latitude': row[ 6 ] if len( row ) > 6 else None,
						'baro_altitude_m': row[ 7 ] if len( row ) > 7 else None,
						'on_ground': row[ 8 ] if len( row ) > 8 else None,
						'velocity_mps': row[ 9 ] if len( row ) > 9 else None,
						'true_track_deg': row[ 10 ] if len( row ) > 10 else None,
						'vertical_rate_mps': row[ 11 ] if len( row ) > 11 else None,
						'sensors': row[ 12 ] if len( row ) > 12 else None,
						'geo_altitude_m': row[ 13 ] if len( row ) > 13 else None,
						'squawk': row[ 14 ] if len( row ) > 14 else None,
						'spi': row[ 15 ] if len( row ) > 15 else None,
						'position_source': row[ 16 ] if len( row ) > 16 else None, } )
			
			self.result = { 'mode': 'states_bbox',
					'time': payload.get( 'time', None ) if isinstance( payload, dict ) else None,
					'count': len( items ), 'items': items, }
			
			return self.result
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'normalize_states( self, payload: Dict[ str, Any ] | None ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def normalize_flights( self, payload: List[ Dict[ str, Any ] ] | None, mode: str ) -> Dict[ str, Any ] | None:
		"""Shape OpenSky flight records.

		Purpose:
		    Skip non-dictionary records and map provider fields to named keys for aircraft,
		    callsign, first and last seen times, estimated airports, distances and candidate counts.

		Args:
		    payload (List[Dict[str, Any]] | None): Decoded flight records or None.
		    mode (str): Operation label included in the result.

		Returns:
		    Dict[str, Any]: Mode, count, and normalized flight record dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			items: List[ Dict[ str, Any ] ] = [ ]
			
			for row in payload or [ ]:
				if not isinstance( row, dict ):
					continue
				
				items.append( { 'icao24': row.get( 'icao24' ), 'first_seen': row.get(
					'firstSeen' ),
						'est_departure_airport': row.get( 'estDepartureAirport' ),
						'last_seen': row.get( 'lastSeen' ),
						'est_arrival_airport': row.get( 'estArrivalAirport' ),
						'callsign': row.get( 'callsign' ),
						'est_departure_airport_horiz_distance_m': row.get(
							'estDepartureAirportHorizDistance' ),
						'est_departure_airport_vert_distance_m': row.get(
							'estDepartureAirportVertDistance' ),
						'est_arrival_airport_horiz_distance_m': row.get(
							'estArrivalAirportHorizDistance' ),
						'est_arrival_airport_vert_distance_m': row.get(
							'estArrivalAirportVertDistance' ),
						'departure_airport_candidates_count': row.get(
							'departureAirportCandidatesCount' ),
						'arrival_airport_candidates_count': row.get(
							'arrivalAirportCandidatesCount' ), } )
			
			self.result = { 'mode': mode, 'count': len( items ), 'items': items, }
			
			return self.result
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'normalize_flights( self, payload: List[ Dict[ str, Any ] ] | None, mode: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def normalize_track( self, payload: Dict[ str, Any ] | None ) -> Dict[ str, Any ] | None:
		"""Shape an OpenSky aircraft track.

		Purpose:
		    Map path array positions to time, coordinates, barometric altitude, heading and ground
		    status, skip non-list entries, and retain track metadata. Return empty items and None
		    metadata for falsey payloads.

		Args:
		    payload (Dict[str, Any] | None): Decoded track object or None.

		Returns:
		    Dict[str, Any]: Mode, aircraft address, callsign, start and end times, count, and
		        normalized track-point dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			if not payload:
				self.result = { 'mode': 'track_aircraft', 'icao24': None, 'callsign': None,
						'start_time': None, 'end_time': None, 'count': 0, 'items': [ ], }
				return self.result
			
			path = payload.get( 'path' ) or [ ]
			items: List[ Dict[ str, Any ] ] = [ ]
			
			for row in path:
				if not isinstance( row, list ):
					continue
				
				items.append( { 'time': row[ 0 ] if len( row ) > 0 else None,
						'latitude': row[ 1 ] if len( row ) > 1 else None,
						'longitude': row[ 2 ] if len( row ) > 2 else None,
						'baro_altitude_m': row[ 3 ] if len( row ) > 3 else None,
						'true_track_deg': row[ 4 ] if len( row ) > 4 else None,
						'on_ground': row[ 5 ] if len( row ) > 5 else None, } )
			
			self.result = { 'mode': 'track_aircraft', 'icao24': payload.get( 'icao24' ),
					'callsign': payload.get( 'callsign' ) or payload.get( 'calllsign' ),
					'start_time': payload.get( 'startTime' ), 'end_time': payload.get( 'endTime' ),
					'count': len( items ), 'items': items, }
			
			return self.result
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'normalize_track( self, payload: Dict[ str, Any ] | None ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_states( self, icao24: str='', time_value: int=None, lamin: float | None = None, lomin: float | None = None, lamax: float | None = None, lomax: float | None = None, extended: bool=False ) -> Dict[ str, Any ] | None:
		"""Retrieve OpenSky aircraft state vectors.

		Purpose:
		    Apply optional aircraft address and epoch filters, require all four geographic bounds
		    when any is supplied, submit extended=1 for truthy extended values, request state
		    vectors and normalize the response.

		Args:
		    icao24 (str): Six-character hexadecimal aircraft address, trimmed and lowercased.
		        Defaults to an empty string; omit for all aircraft.
		    time_value (int | None): Optional nonnegative Unix epoch seconds; booleans are rejected.
		        Defaults to None.
		    lamin (float | None): Finite minimum latitude from -90 through 90 degrees; booleans are
		        rejected. Defaults to None; all four bounds must be supplied together or omitted.
		    lomin (float | None): Finite minimum longitude from -180 through 180 degrees; booleans
		        are rejected. Defaults to None; all four bounds must be supplied together or
		        omitted.
		    lamax (float | None): Finite maximum latitude from -90 through 90 degrees, greater than
		        lamin; booleans are rejected. Defaults to None; all four bounds must be supplied
		        together or omitted.
		    lomax (float | None): Finite maximum longitude from -180 through 180 degrees, greater
		        than lomin; booleans are rejected. Defaults to None; all four bounds must be
		        supplied together or omitted.
		    extended (bool): When truthy, submit extended=1. Defaults to False.

		Returns:
		    Dict[str, Any]: Mode, provider snapshot time, count, and normalized state-vector
		        dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'states_bbox'
			self.icao24 = self.validate_icao24( icao24 ) if ( icao24 or '' ).strip( ) else ''
			self.time_value = (
					self.validate_epoch( 'time_value', time_value ) if time_value is not None else
					None)
			throw_if( 'extended', extended )

			self.extended = extended
			
			self.params = { }
			
			if self.time_value is not None:
				self.params[ 'time' ] = self.time_value
			
			if self.icao24:
				self.params[ 'icao24' ] = self.icao24
			
			has_bbox = all( value is not None for value in [ lamin, lomin, lamax, lomax ] )
			partial_bbox = any( value is not None for value in [ lamin, lomin, lamax, lomax ] )
			
			if partial_bbox and not has_bbox:
				raise ValueError( 'lamin, lomin, lamax, and lomax must all be supplied together.' )
			
			if has_bbox:
				self.lamin, self.lomin, self.lamax, self.lomax = self.validate_bbox( lamin=lamin,
					lomin=lomin, lamax=lamax, lomax=lomax )
				self.params[ 'lamin' ] = self.lamin
				self.params[ 'lomin' ] = self.lomin
				self.params[ 'lamax' ] = self.lamax
				self.params[ 'lomax' ] = self.lomax
			
			if self.extended:
				self.params[ 'extended' ] = 1
			
			self.payload = self.request( '/states/all', params=self.params )
			
			return self.normalize_states( self.payload )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = "fetch_states( self, icao24: str='', time_value: int=None, lamin: float | None = None, lomin: float | None = None, lamax: float | None = None, lomax: float | None = None, extended: bool=False ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def fetch_flights_aircraft( self, icao24: str, begin: int, end: int ) -> Dict[ str, Any ] | None:
		"""Retrieve historical flights for an aircraft.

		Purpose:
		    Validate aircraft address and epoch range, enforce a maximum interval of 172800 seconds,
		    request aircraft flights, and normalize records.

		Args:
		    icao24 (str): Six-character hexadecimal aircraft address, trimmed and lowercased.
		    begin (int): Nonnegative Unix epoch seconds at the beginning of the interval; booleans
		        are rejected.
		    end (int): Nonnegative Unix epoch seconds strictly greater than begin; booleans are
		        rejected. Interval cannot exceed two days.

		Returns:
		    Dict[str, Any]: Mode, count, and normalized flight record dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'flights_aircraft'
			self.icao24 = self.validate_icao24( icao24 )
			self.begin, self.end = self.validate_time_range( begin, end )
			if self.end - self.begin > 172800:
				raise ValueError( 'Aircraft flights interval cannot exceed two days.' )
			self.params = { 'icao24': self.icao24, 'begin': self.begin, 'end': self.end, }
			self.payload = self.request( '/flights/aircraft', params=self.params )
			
			return self.normalize_flights( self.payload, self.mode )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'fetch_flights_aircraft( self, icao24: str, begin: int, end: int ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_arrivals_airport( self, airport: str, begin: int, end: int ) -> Dict[ str, Any ] | None:
		"""Retrieve historical arrivals at an airport.

		Purpose:
		    Validate airport code and epoch range, enforce a maximum interval of 172800 seconds,
		    request arrivals, and normalize records.

		Args:
		    airport (str): Four-character alphanumeric ICAO airport code, trimmed and uppercased.
		    begin (int): Nonnegative Unix epoch seconds at the beginning of the interval; booleans
		        are rejected.
		    end (int): Nonnegative Unix epoch seconds strictly greater than begin; booleans are
		        rejected. Interval cannot exceed two days.

		Returns:
		    Dict[str, Any]: Mode, count, and normalized flight record dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'arrivals_airport'
			self.airport = self.validate_airport( airport )
			self.begin, self.end = self.validate_time_range( begin, end )
			if self.end - self.begin > 172800:
				raise ValueError( 'Airport arrival interval cannot exceed two days.' )
			self.params = { 'airport': self.airport, 'begin': self.begin, 'end': self.end, }
			self.payload = self.request( '/flights/arrival', params=self.params )
			
			return self.normalize_flights( self.payload, self.mode )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'fetch_arrivals_airport( self, airport: str, begin: int, end: int ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_departures_airport( self, airport: str, begin: int, end: int ) -> Dict[ str, Any ] | None:
		"""Retrieve historical departures from an airport.

		Purpose:
		    Validate airport code and epoch range, request departures, and normalize records. No
		    maximum interval is enforced locally.

		Args:
		    airport (str): Four-character alphanumeric ICAO airport code, trimmed and uppercased.
		    begin (int): Nonnegative Unix epoch seconds at the beginning of the interval; booleans
		        are rejected.
		    end (int): Nonnegative Unix epoch seconds strictly greater than begin; booleans are
		        rejected.

		Returns:
		    Dict[str, Any]: Mode, count, and normalized flight record dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'departures_airport'
			self.airport = self.validate_airport( airport )
			self.begin, self.end = self.validate_time_range( begin, end )
			self.params = { 'airport': self.airport, 'begin': self.begin, 'end': self.end, }
			
			self.payload = self.request( '/flights/departure', params=self.params )
			
			return self.normalize_flights( self.payload, self.mode )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'fetch_departures_airport( self, airport: str, begin: int, end: int ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch_track_aircraft( self, icao24: str, time_value: int=None ) -> Dict[ str, Any ] | None:
		"""Retrieve an aircraft track.

		Purpose:
		    Validate the aircraft address and optional epoch, use zero when time_value is None,
		    request the track endpoint, and normalize path records.

		Args:
		    icao24 (str): Six-character hexadecimal aircraft address, trimmed and lowercased.
		    time_value (int | None): Optional nonnegative Unix epoch seconds; booleans are rejected.
		        Defaults to None. None is submitted as zero.

		Returns:
		    Dict[str, Any]: Mode, aircraft address, callsign, start and end times, count, and
		        normalized track-point dictionaries.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			self.mode = 'track_aircraft'
			self.icao24 = self.validate_icao24( icao24 )
			self.time_value = (
				self.validate_epoch( 'time_value', time_value ) if time_value is not None else 0)
			self.params = { 'icao24': self.icao24, 'time': self.time_value, }
			self.payload = self.request( '/tracks/all', params=self.params )
			
			return self.normalize_track( self.payload )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'fetch_track_aircraft( self, icao24: str, time_value: int=None ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, mode: str='states_bbox', icao24: str='', airport: str='', begin: int = None, end: int=None, time_value: int=None, lamin: float | None = None, lomin: float | None = None, lamax: float | None = None, lomax: float | None = None, extended: bool=False, time: int=20 ) -> Dict[ str, Any ] | None:
		"""Dispatch an OpenSky retrieval operation.

		Purpose:
		    Validate and retain the HTTP timeout, validate mode, and delegate to the selected state,
		    flight, airport or track method with applicable arguments.

		Args:
		    mode (str): states_bbox, flights_aircraft, arrivals_airport, departures_airport, or
		        track_aircraft; trimmed and lowercased. Defaults to 'states_bbox'.
		    icao24 (str): Six-character hexadecimal aircraft address, trimmed and lowercased.
		        Defaults to an empty string; required for aircraft flight and track modes.
		    airport (str): Four-character alphanumeric ICAO airport code, trimmed and uppercased.
		        Defaults to an empty string; required for airport modes.
		    begin (int | None): Nonnegative Unix epoch seconds at the beginning of the interval;
		        booleans are rejected. Defaults to None; required for flight and airport modes.
		    end (int | None): Nonnegative Unix epoch seconds strictly greater than begin; booleans
		        are rejected. Defaults to None; required for flight and airport modes.
		    time_value (int | None): Optional nonnegative Unix epoch seconds; booleans are rejected.
		        Defaults to None.
		    lamin (float | None): Finite minimum latitude from -90 through 90 degrees; booleans are
		        rejected. Defaults to None; all four bounds must be supplied together or omitted.
		    lomin (float | None): Finite minimum longitude from -180 through 180 degrees; booleans
		        are rejected. Defaults to None; all four bounds must be supplied together or
		        omitted.
		    lamax (float | None): Finite maximum latitude from -90 through 90 degrees, greater than
		        lamin; booleans are rejected. Defaults to None; all four bounds must be supplied
		        together or omitted.
		    lomax (float | None): Finite maximum longitude from -180 through 180 degrees, greater
		        than lomin; booleans are rejected. Defaults to None; all four bounds must be
		        supplied together or omitted.
		    extended (bool): When truthy, submit extended=1. Defaults to False.
		    time (int): Positive integer HTTP timeout in seconds; booleans are rejected. Defaults to
		        20.

		Returns:
		    Dict[str, Any]: Normalized state, flight, or track result for the selected operation.

		Raises:
		    Error: Raised when validation, retrieval, or response processing fails.
		"""
		try:
			throw_if( 'time', time )
			if type( time ) is not int or time < 1:
				raise ValueError( 'time must be a positive integer.' )
			self.timeout = time
			self.mode = self.validate_mode( mode )
			
			if self.mode == 'states_bbox':
				return self.fetch_states( icao24=icao24, time_value=time_value, lamin=lamin,
					lomin=lomin, lamax=lamax, lomax=lomax, extended=extended )
			
			if self.mode == 'flights_aircraft':
				return self.fetch_flights_aircraft( icao24=icao24, begin=begin, end=end )
			
			if self.mode == 'arrivals_airport':
				return self.fetch_arrivals_airport( airport=airport, begin=begin, end=end )
			
			if self.mode == 'departures_airport':
				return self.fetch_departures_airport( airport=airport, begin=begin, end=end )
			
			if self.mode == 'track_aircraft':
				return self.fetch_track_aircraft( icao24=icao24, time_value=time_value )
			
			raise ValueError( "Unsupported mode. Use 'states_bbox', 'flights_aircraft', "
			                  "'arrivals_airport', 'departures_airport', or 'track_aircraft'." )
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = "fetch( self, mode: str='states_bbox', icao24: str='', airport: str='', begin: int = None, end: int=None, time_value: int=None, lamin: float | None = None, lomin: float | None = None, lamax: float | None = None, lomax: float | None = None, extended: bool=False, time: int=20 ) -> Dict[ str, Any ] | None"
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Build a function schema.

		Purpose:
		    Construct a function description with object parameter definitions and required argument
		    names.

		Args:
		    function (str): Nonblank function name.
		    tool (str): Nonblank service name.
		    description (str): Nonblank function description.
		    parameters (dict): Parameter-property dictionary.
		    required (list[str] | None): Required argument names; None selects all property names.

		Returns:
		    Dict[str, Any]: Function name, service-qualified description, and nested parameter
		        schema.

		Raises:
		    Error: Raised when validation or schema construction fails.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				raise ValueError( 'parameters must be a dict of param_name -> schema definition.' )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return { 'name': function.strip( ),
					'description': (f'{description.strip( )} This function uses the '
					                f'{tool.strip( )} service.'),
					'parameters': { 'type': 'object', 'properties': parameters,
							'required': required, } }
		
		except Error:
			raise
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'OpenSky'
			exception.method = 'create_schema( self, function: str, tool: str, description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None'
			Logger( ).write( exception )
			raise exception
