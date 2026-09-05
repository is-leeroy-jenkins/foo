'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                aap.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        08-25-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">

	     Foo is a python framework for web scraping information into ML pipelines.
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

	 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	 DEALINGS IN THE SOFTWARE.

	 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

	</copyright>
	<summary>
	    app.py — the UI
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
import altair
import inspect
from astroquery.simbad import Simbad
import base64

from bs4 import BeautifulSoup
import crawl4ai
import config as cfg
from collections import deque, Counter
import datetime as dt
import html as html_lib
import json
import numpy as np
import os
import re
import time
import types
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable
from langchain_core.documents import Document
from lxml import etree
from processors import PdfParser
from loaders import (TextLoader, CsvLoader, PdfLoader, ExcelLoader, WordLoader, MarkdownLoader, HtmlLoader, JsonLoader, PowerPointLoader, WikiLoader, GithubLoader, WebLoader, ArXivLoader, XmlLoader, PubMedSearchLoader, OpenCityLoader, OutlookLoader, JupyterNotebookLoader, AwsFileLoader, OneDriveDocLoader, GoogleCloudFileLoader, GoogleSpeechToTextLoader, GoogleBucketLoader, AwsBucketLoader, EmailLoader, SpfxLoader)

from generators import Chat, Claude, Grok, Mistral, Gemini
from fetchers import (Wikipedia, TheNews, SatelliteCenter, WebFetcher, GoogleWeather, Grokipedia, OpenWeather, NavalObservatory, GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience, EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap, GovData, Congress, InternetArchive, StarChart, HistoricalWeather, GoogleGeocoding, USGSEarthquakes, USGSWaterData, USGSTheNationalMap, USGSScienceBase, AirNow, ClimateData, EoNet, EnviroFacts, TidesAndCurrents, UvIndex, PurpleAir, OpenAQ, Firms, CensusData, Socrata, HealthData, GlobalHealthData, UnitedNations, WorldPopulation, Wonder, OpenSky, WebCrawler)

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pandas import DataFrame
import streamlit as st
import scrapers
import sqlite3
from sqlite3 import Connection
from urllib.parse import urljoin, urlparse

try:
	import textstat
	
	TEXTSTAT_AVAILABLE = True
except ImportError:
	TEXTSTAT_AVAILABLE = False

# =====================================================================
# SESSION STATE DEFINITIONS
# =====================================================================

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = list( cfg.MODE_MAP.keys( ) )[ 0 ]

# -------- GENERATIVE AI VARIABLES --------------------

if 'model' not in st.session_state:
	st.session_state[ 'model' ] = ''

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presence_penalty' not in st.session_state:
	st.session_state[ 'presence_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

# ------------ LOADER VARIABLES -----------

if 'loader_results' not in st.session_state:
	st.session_state[ 'loader_results' ] = { }

if 'documents' not in st.session_state:
	st.session_state[ 'documents' ] = None

if 'tokens' not in st.session_state:
	st.session_state[ 'tokens' ] = None

if 'vocabulary' not in st.session_state:
	st.session_state[ 'vocabulary' ] = None

if 'raw_text' not in st.session_state:
	st.session_state[ 'raw_text' ] = ''

if 'processed_text' not in st.session_state:
	st.session_state[ 'processed_text' ] = ''

if 'token_counts' not in st.session_state:
	st.session_state[ 'token_counts' ] = None

if 'loader_path' not in st.session_state:
	st.session_state[ 'loader_path' ] = ''

if 'loader_text' not in st.session_state:
	st.session_state[ 'loader_text' ] = ''

if 'loader_files' not in st.session_state:
	st.session_state[ 'loader_files' ] = ''

# ------------- SCRAPPER VARIABLES --------------

if 'target_url' not in st.session_state:
	st.session_state[ 'target_url' ] = ''

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )

# =====================================================================
# UTILITIES
# =====================================================================

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

def filter_kwargs_for_callable( fn: Any, kwargs: dict[ str, Any ] ) -> dict[ str, Any ]:
	try:
		sig = inspect.signature( fn )
		accepted = set( sig.parameters.keys( ) )
		return { k: v for k, v in kwargs.items( ) if k in accepted }
	except Exception:
		return kwargs

def _invoke_provider( fetcher: Any, prompt: str, params: dict[ str, Any ] ) -> Any:
	if hasattr( fetcher, "fetch" ) and callable( getattr( fetcher, "fetch" ) ):
		fn = getattr( fetcher, "fetch" )
		safe = filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = filter_kwargs_for_callable( fn, { **safe, "query": prompt } )
			return fn( **safe2 )
	
	if hasattr( fetcher, "chat" ) and callable( getattr( fetcher, "chat" ) ):
		fn = getattr( fetcher, "chat" )
		safe = filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = filter_kwargs_for_callable( fn, { **safe, "prompt": prompt } )
			return fn( **safe2 )
	
	if hasattr( fetcher, "invoke" ) and callable( getattr( fetcher, "invoke" ) ):
		fn = getattr( fetcher, "invoke" )
		safe = filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = filter_kwargs_for_callable( fn, { **safe, "input": prompt } )
			return fn( **safe2 )
	
	raise RuntimeError( f"Provider '{type( fetcher ).__name__}' does not expose fetch/chat/invoke." )

def _render_output( container: Any, result: Any ) -> None:
	if result is None:
		container.info( "No response returned." )
		return
	
	if isinstance( result, list ) and result and isinstance( result[ 0 ], Document ):
		with container.container( ):
			for idx, doc in enumerate( result, start=1 ):
				with st.expander( f"Document {idx}", expanded=False ):
					st.text_area( '', value=(doc.page_content or ""), height=300 )
					if doc.metadata:
						st.json( doc.metadata )
		return
	
	container.text_area( "Response", value=str( result ), height=320 )

def _render_result_metadata( result: Dict[ str, Any ] ) -> None:
	'''
		Purpose:
		--------
		Render common request metadata in a compact, readable format.

		Parameters:
		-----------
		result (Dict[str, Any]):
			Fetcher result dictionary.

		Returns:
		--------
		None

	'''
	if not isinstance( result, dict ) or not result:
		return
	
	meta = { 'Mode': result.get( 'mode', '' ), 'URL': result.get( 'url', '' ),
			'Params': result.get( 'params', { } ), }
	
	st.markdown( '#### Request Metadata' )
	st.json( meta )

def _render_summary_kv( title: str, data: Dict[ str, Any ] ) -> None:
	'''
		Purpose:
		--------
		Render a dictionary of scalar values as a two-column summary table.

		Parameters:
		-----------
		title (str):
			Section title.

		data (Dict[str, Any]):
			Scalar dictionary to render.

		Returns:
		--------
		None

	'''
	if not isinstance( data, dict ) or not data:
		return
	
	rows: List[ Dict[ str, Any ] ] = [ ]
	
	for key, value in data.items( ):
		if isinstance( value, (str, int, float, bool) ) or value is None:
			rows.append( { 'Field': key, 'Value': value, } )
	
	if rows:
		st.markdown( title )
		st.dataframe( pd.DataFrame( rows ), use_container_width=True, hide_index=True )

def _render_rows_table( title: str, rows: List[ Dict[ str, Any ] ] ) -> None:
	'''
		Purpose:
		--------
		Render a list of dictionaries as a dataframe.

		Parameters:
		-----------
		title (str):
			Section title.

		rows (List[Dict[str, Any]]):
			Rows to render.

		Returns:
		--------
		None

	'''
	if not isinstance( rows, list ) or not rows:
		st.info( 'No rows returned.' )
		return
	
	df_rows = pd.DataFrame( rows )
	
	if df_rows.empty:
		st.info( 'No rows returned.' )
		return
	
	st.markdown( title )
	st.dataframe( df_rows, use_container_width=True, hide_index=True, height=320 )

def _render_xml_preview( title: str, xml_text: str, max_chars: int = 12000 ) -> None:
	'''
		Purpose:
		--------
		Render XML content in a readable preview area.

		Parameters:
		-----------
		title (str):
			Section title.

		xml_text (str):
			XML payload text.

		max_chars (int):
			Maximum preview length.

		Returns:
		--------
		None

	'''
	if not str( xml_text or '' ).strip( ):
		st.info( 'No XML content returned.' )
		return
	
	st.markdown( title )
	st.code( str( xml_text )[ :int( max_chars ) ], language='xml' )

def _render_html_preview( title: str, html_text: str, max_chars: int = 5000 ) -> None:
	'''
		Purpose:
		--------
		Render a human-friendly HTML summary using title and hyperlinks when possible.

		Parameters:
		-----------
		title (str):
			Section title.

		html_text (str):
			HTML payload text.

		max_chars (int):
			Maximum preview length for fallback output.

		Returns:
		--------
		None

	'''
	text = str( html_text or '' )
	if not text.strip( ):
		st.info( 'No HTML content returned.' )
		return
	
	st.markdown( title )
	
	try:
		soup = BeautifulSoup( text, 'html.parser' )
		
		page_title = ''
		if soup.title and soup.title.string:
			page_title = str( soup.title.string ).strip( )
		
		if page_title:
			st.markdown( f'**Title:** {page_title}' )
		
		links: List[ Dict[ str, str ] ] = [ ]
		for anchor in soup.find_all( 'a', href=True )[ :15 ]:
			label = str( anchor.get_text( ' ', strip=True ) or anchor.get( 'href', '' ) ).strip( )
			href = str( anchor.get( 'href', '' ) ).strip( )
			if label or href:
				links.append( { 'Label': label, 'Link': href, } )
		
		if links:
			st.dataframe( pd.DataFrame( links ), use_container_width=True, hide_index=True, height=260 )
		else:
			st.code( text[ :int( max_chars ) ], language='html' )
	except Exception:
		st.code( text[ :int( max_chars ) ], language='html' )

def _render_fallback_raw( result: Any ) -> None:
	'''
		Purpose:
		--------
		Render a raw result payload inside a collapsed expander.

		Parameters:
		-----------
		result (Any):
			Result payload to render.

		Returns:
		--------
		None

	'''
	with st.expander( 'Raw Result', expanded=False ):
		if isinstance( result, (dict, list) ):
			st.json( result )
		else:
			st.text_area( 'Output', value=str( result ), height=320 )

def _model_selector( key_prefix: str, label: str, options: list[ str ], default_model: str ) -> str:
	base_options = options[ : ]
	if "Custom..." not in base_options:
		base_options.append( "Custom..." )
	
	idx_default = base_options.index( default_model ) if default_model in base_options else 0
	
	selected = st.selectbox( label=label, options=base_options, index=idx_default, key=f"{key_prefix}_model_select", )
	
	if selected == "Custom...":
		return st.text_input( "Custom Model", value=default_model, key=f"{key_prefix}_model_custom", )
	
	return selected

def rebuild_raw_text_from_documents( ) -> str | None:
	"""Rebuild raw text from loaded documents.

	Purpose:
		Reconstructs the shared raw-text buffer from the current Streamlit ``documents`` state
		after
		loader-specific clear operations. The helper preserves downstream processing compatibility
		by joining non-empty ``page_content`` values from remaining LangChain documents.

	Returns:
		str | None: Rebuilt raw text, or ``None`` when no usable document text remains.
	"""
	docs = st.session_state.get( "documents" ) or [ ]
	if not docs:
		return None
	text = '\n\n'.join( d.page_content for d in docs if
			hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) and d.page_content.strip( ) )
	return text if text.strip( ) else None

# -------- Text Utilities ---------------------

def render_google_results( response ) -> str:
	try:
		data = response.json( )
	except Exception:
		return "Failed to decode response."
	
	items = data.get( "items", [ ] )
	if not items:
		return "No results returned."
	
	lines = [ ]
	for idx, item in enumerate( items, start=1 ):
		title = item.get( "title", "Untitled" )
		snippet = item.get( "snippet", "" )
		link = item.get( "link", "" )
		
		lines.append( f"{idx}. {title}" )
		if snippet:
			lines.append( snippet )
		if link:
			lines.append( link )
		lines.append( "" )
	
	return "\n".join( lines )

def style_subheaders( ) -> None:
	"""

		Purpose:
		_________
		Sets the style of subheaders in the main UI

	"""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

def encode_image( path: str ) -> str:
	"""

		Purpose:
		_________

		Parametes:
		----------


		Returns:
		--------


	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def normalize_text( text: str ) -> str:
	"""

		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace

		Parameters
		----------
		text: str

		Returns
		-------
		str

	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", '', text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int = 400 ) -> list[ str ]:
	"""

		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed

		Parameters
		----------
		text: str
		max_tokens: int

		Returns
		-------
		list[str]

	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
	return chunks

def cosine_similarity( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""

		Purpose:
		_________


	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, 'model_dump' ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def metric_with_tooltip( label: str, value: str, tooltip: str ):
	"""
		Renders a metric with a hover tooltip using a two-column layout.
		Left column = the metric itself
		Right column = hoverable ℹ️ icon
	"""
	col_metric, col_info = st.columns( [ 0.5, 0.5 ] )
	
	with col_metric:
		st.metric( label, value )
	
	with col_info:
		if label not in [ 'Characters', 'Tokens', 'Unique Tokens', 'Avg Length' ]:
			st.markdown( f"""
	            <span style="
	                cursor: help;
	                font-size: 0.85rem;
	                color:#888;
	                vertical-align: super;
	            " title="{tooltip}">ℹ️ </span>
	            """, unsafe_allow_html=True, )

# ----------  Database Utilities --------------

def initialize_database( ) -> None:
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS chat_history
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          role
                          TEXT,
                          content
                          TEXT
                      )
		              """ )
		
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          chunk
                          TEXT,
                          vector
                          BLOB
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS Prompts
                      (
                          PromptsId
                          INTEGER
                          NOT
                          NULL
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          Caption
                          TEXT,
                          Name
                          TEXT,
                          Text
                          TEXT,
                          Version
                          TEXT,
                          ID
                          TEXT
                      )
		              """ )
		
		prompt_columns = [ row[ 1 ] for row in
				conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

def create_connection( ) -> Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int = 0 ) -> DataFrame:
	if not table:
		return DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			try:
				return str( normalize( value ) )
			except Exception:
				return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return DataFrame( normalized_rows, columns=columns )

def render_table( df: DataFrame ) -> None:
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		st.data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map( lambda x: x if isinstance( x, (str, int, float,
				bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: DataFrame ) -> DataFrame:
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map( lambda x: '' if x is None else str( x ) )
	
	return display_df

def drop_table( table: str ) -> None:
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: DataFrame ) -> DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: DataFrame ) -> None:
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map( lambda x: '' if x is None else str( x ) )
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_cols: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie',
			'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_cols:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_cols )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure( data=[
				go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure( data=[
				go.Heatmap( z=corr.values.tolist( ), x=corr.columns.tolist( ), y=corr.index.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )

def convert_dataframe( table_name: str, df: DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'attach ',
			'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute( f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table_name,) ).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute( f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";' )
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = { 'column': col, 'dtype': str( series.dtype ),
				'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
				'distinct_%': round( (
							                     distinct_count / total_rows) * 100, 2 ) if total_rows else 0, }
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ 'min' ] = series.min( )
			row[ 'max' ] = series.max( )
			row[ 'mean' ] = series.mean( )
		else:
			row[ 'min' ] = None
			row[ 'max' ] = None
			row[ 'mean' ] = None
		
		profile_rows.append( row )
	
	return DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( 'Table and column required.' )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (f'CREATE TABLE "{temp_table}" (' + ", ".join( new_defs ) + ");")
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [ c.split( )[ 0 ].strip( '"' ) for c in new_defs ]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute( f'INSERT INTO "{temp_table}" ({col_list}) '
		              f'SELECT {col_list} FROM "{table}";' )
		
		# Preserve indexes
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table,) ).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (old_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (old_name,) ).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute( f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";' )
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def clear_if_active( loader_name: str ) -> None:
	if st.session_state.active_loader == loader_name:
		st.session_state.documents = None
		st.session_state.active_loader = None
		st.session_state.tokens = None
		st.session_state.vocabulary = None
		st.session_state.token_counts = None
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.embeddings = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None
		st.session_state.df_chunks = None
		st.session_state.lines = None

def render_data_editor( df: pd.DataFrame, use_container_width: bool | None = None,
		key: str = '', height: int | str = 'auto', hide_index: bool | None = None,
		disabled: bool | Dict[ str, object ] = None, num_rows: str = 'fixed' ) -> pd.DataFrame:
	"""Render a consistently formatted Streamlit dataframe editor.

	Purpose:
	    Preserves the established data-editor arguments while merging automatic numeric formatting
	    with any specialized caller configuration. Floating-point values use the locale equivalent
	    of ``#,##0.00`` whenever Streamlit supports numeric column formatting.

	Args:
	    df (pd.DataFrame): Dataframe rendered by Streamlit.
	    use_container_width (bool | None): Existing Streamlit container-width setting.
	    key (str): Optional unique widget key.
	    height (int | str): Editor height setting.
	    hide_index (bool | None): Index visibility setting.
	    disabled (bool | List[str]): Editor or column disabled state.
	    column_config (Dict[str, object]): Optional specialized column configuration.
	    num_rows (str): Established fixed or dynamic row mode.

	Returns:
	    pd.DataFrame: Dataframe returned by ``st.data_editor``.
	"""
	throw_if( 'df', df )
	formatted_config = get_data_editor_column_config( df, column_config )
	widget_key = key if key else None
	return _streamlit_data_editor( df, use_container_width=use_container_width, key=widget_key,
		height=height, hide_index=hide_index, disabled=disabled,
		column_config=formatted_config, num_rows=num_rows )

def get_data_editor_column_config( df: pd.DataFrame,
		existing_config: Dict[ str, object ] = None ) -> Dict[ str, object ]:
	"""Build numeric display configuration for a dataframe editor.

	Purpose:
	    Adds locale-aware thousands separators and two-decimal precision for floating-point and
	    double-precision columns while retaining whole-number formatting for integer columns.
	    Explicit caller configurations take precedence over generated defaults.

	Args:
	    df (pd.DataFrame): Dataframe displayed by the editor.
	    existing_config (Dict[str, object]): Optional specialized caller configuration preserved in
	        the returned mapping.

	Returns:
	    Dict[str, object]: Streamlit column configuration keyed by dataframe column name.
	"""
	throw_if( 'df', df )
	column_config = dict( existing_config ) if existing_config else { }
	profiles = profile_dataframe_schema( df )
	for column in df.columns:
		if column in column_config:
			continue
		series = df[ column ]
		if profiles[ column ][ 'analytical_role' ] != 'numeric':
			continue
		if pd.api.types.is_float_dtype( series ):
			column_config[
				column ] = st.column_config.NumberColumn( str( column ), format='localized', step=0.01 )
		elif pd.api.types.is_integer_dtype( series ) and not pd.api.types.is_bool_dtype( series ):
			column_config[
				column ] = st.column_config.NumberColumn( str( column ), format='localized', step=1 )
	return column_config

def profile_dataframe_schema( df: pd.DataFrame,
		declared_schema: Dict[ str, str ] = None ) -> Dict[ str, Dict[ str, object ] ]:
	"""Profile every dataframe column using the authoritative schema classifier.

	Purpose:
	    Profiles dataframe columns with optional SQLite declared-type metadata while preserving the
	    established detailed profile contract and dataset-specific analytical-role overrides.

	Args:
	    df (pd.DataFrame): Dataframe whose storage types and analytical roles are profiled.
	    declared_schema (Dict[str, str]): Optional SQLite declared types keyed by column name.

	Returns:
	    Dict[str, Dict[str, object]]: Detailed profile keyed by dataframe column name.
	"""
	throw_if( 'df', df )
	schema_signature = repr( (tuple( df.columns.tolist( ) ),
			tuple( str( dtype ) for dtype in df.dtypes.tolist( ) )) )
	overrides_by_schema = st.session_state.get( 'column_type_overrides', { } )
	legacy_override_values = list( overrides_by_schema.values( ) ) if isinstance( overrides_by_schema, dict ) else [ ]
	if legacy_override_values and all(
			isinstance( value, str ) for value in legacy_override_values ):
		overrides = overrides_by_schema
	else:
		overrides = overrides_by_schema.get( schema_signature, { } ) if isinstance( overrides_by_schema, dict ) else { }
	declared_types = declared_schema if isinstance( declared_schema, dict ) else { }
	return { column: profile_column( str( column ),
		df[ column ], str( overrides.get( column, '' ) ), str( declared_types.get( column, '' ) ) )
			for column in df.columns }

def profile_column( column_name: str, series: pd.Series,
		override_role: str = '', declared_type: str = '' ) -> Dict[ str, object ]:
	"""Infer the physical type and analytical role of one dataframe column.

	Purpose:
	    Uses an authoritative SQLite declared type when available, then applies exact column-name
	    semantics to distinguish identifiers, categories, ordered variables, measures, booleans,
	    and datetimes. Dataframe value inference remains the fallback for non-database sources.
	    Explicit user overrides retain highest priority for the analytical role.

	Args:
	    column_name (str): Column name associated with the supplied series.
	    series (pd.Series): Column values evaluated by the profiler.
	    override_role (str): Optional user-selected analytical role.
	    declared_type (str): Optional SQLite declared type for database-backed data.

	Returns:
	    Dict[str, object]: Detailed profile containing storage type, inferred dtype, analytical role,
	        confidence, evidence, and cardinality statistics.
	"""
	throw_if( 'column_name', column_name )
	throw_if( 'series', series )
	valid_roles = { 'numeric', 'categorical', 'ordinal', 'identifier', 'datetime' }
	tokens = set( get_column_name_tokens( column_name ) )
	populated_mask = get_populated_mask( series )
	populated_count = int( populated_mask.sum( ) )
	distinct_count = int( series[ populated_mask ].nunique( dropna=True ) )
	unique_ratio = distinct_count / max( 1, populated_count )
	text_values = series.astype( 'string' ).str.strip( )
	numeric_values = parse_numeric_series( series )
	numeric_success = float( numeric_values[ populated_mask ].notna( ).mean( ) ) if (
			populated_count > 0) else 0.0
	integer_like = bool( numeric_values[ populated_mask ].dropna( ).mod( 1 ).eq( 0 ).all( ) ) if (
			numeric_values[ populated_mask ].notna( ).any( )) else False
	leading_zero_ratio = float( text_values[
		populated_mask ].str.match( r'^[+-]?0\d+$', na=False ).mean( ) ) if populated_count > 0 else 0.0
	
	date_tokens = { 'date', 'datetime', 'timestamp', 'time', 'effective', 'expiration', 'expiry' }
	temporal_context_tokens = { 'approved', 'approval', 'created', 'modified', 'updated',
			'submitted', 'posted', 'received', 'processed', 'opened', 'closed' }
	identifier_tokens = { 'id', 'identifier', 'key', 'uuid', 'guid', 'index', 'sequence' }
	ordinal_tokens = { 'rank', 'grade', 'level', 'rating', 'priority', 'stage', 'tier', 'fy' }
	category_tokens = { 'category', 'categories', 'class', 'type', 'status', 'agency', 'program',
			'symbol', 'split', 'authority', 'footnote', 'footnotes', 'code' }
	name_tokens = { 'name', 'title', 'description', 'label', 'caption' }
	boolean_values = { 'yes', 'no', 'true', 'false', 'y', 'n', '0', '1' }
	normalized_values = set( text_values[ populated_mask ].str.lower( ).unique( ).tolist( ) )
	boolean_candidate = bool( normalized_values ) and normalized_values.issubset( boolean_values )
	identifier_evidence = bool( tokens & identifier_tokens )
	ordinal_evidence = bool( tokens & ordinal_tokens ) or (
		{ 'fiscal', 'year' }.issubset( tokens )) or ({ 'calendar', 'year' }.issubset( tokens ))
	date_name_evidence = bool( tokens & date_tokens ) or (
		{ 'start', 'date' }.issubset( tokens )) or ({ 'end', 'date' }.issubset( tokens ))
	temporal_context_evidence = bool( tokens & temporal_context_tokens )
	category_evidence = bool( tokens & category_tokens )
	name_evidence = bool( tokens & name_tokens )
	
	date_lexical_mask = text_values[ populated_mask ].str.contains(
		r'[-/:T]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', case=False, regex=True, na=False )
	date_lexical_ratio = float( date_lexical_mask.mean( ) ) if populated_count > 0 else 0.0
	datetime_values = parse_datetime_series( series ) if (
			pd.api.types.is_datetime64_any_dtype( series ) or date_name_evidence \
			or temporal_context_evidence or date_lexical_ratio > 0.0) else pd.Series(
		pd.NaT, index=series.index, dtype='datetime64[ns]' )
	datetime_success = float( datetime_values[ populated_mask ].notna( ).mean( ) ) if (
			populated_count > 0) else 0.0
	datetime_value_evidence = bool( populated_count > 0 and datetime_success == 1.0 and (
			date_name_evidence or temporal_context_evidence or date_lexical_ratio > 0.0) )
	
	declared_family = get_declared_type_family( declared_type )
	storage_dtype = declared_type if declared_type else str( series.dtype )
	analytical_role = 'categorical'
	inferred_dtype = 'text'
	confidence = 0.60
	reason = 'Text values default to a categorical analytical role.'
	if override_role in valid_roles:
		analytical_role = override_role
		inferred_dtype = 'overridden'
		confidence = 1.0
		reason = f'User override selected the {override_role} analytical role.'
	elif declared_family != 'unknown':
		if declared_family == 'datetime':
			analytical_role = 'datetime'
			inferred_dtype = 'datetime'
			confidence = 1.0
			reason = f'SQLite declares the column as {declared_type}.'
		elif declared_family == 'boolean':
			analytical_role = 'categorical'
			inferred_dtype = 'boolean'
			confidence = 1.0
			reason = f'SQLite declares the column as {declared_type}; booleans are categorical.'
		elif declared_family == 'numeric':
			inferred_dtype = 'integer' if 'INT' in declared_type.upper( ) else 'float'
			if identifier_evidence:
				analytical_role = 'identifier'
				confidence = 0.99
				reason = 'SQLite declares a numeric type and the column name identifies a key or ID.'
			elif ordinal_evidence:
				analytical_role = 'ordinal'
				confidence = 0.99
				reason = 'SQLite declares a numeric type and the column name indicates an ordered variable.'
			elif category_evidence:
				analytical_role = 'categorical'
				confidence = 0.99
				reason = 'SQLite declares a numeric type but the column name indicates a category or code.'
			else:
				analytical_role = 'numeric'
				confidence = 1.0
				reason = f'SQLite declares the column as numeric type {declared_type}.'
		elif declared_family == 'text':
			inferred_dtype = 'text'
			if date_name_evidence and datetime_success > 0.0:
				analytical_role = 'datetime'
				inferred_dtype = 'datetime'
				confidence = 0.99
				reason = ('SQLite declares text storage, the column name indicates a date or time, '
				          'and populated values parse as datetimes.')
			elif datetime_value_evidence:
				analytical_role = 'datetime'
				inferred_dtype = 'datetime'
				confidence = 0.98
				reason = ('SQLite declares text storage and all populated values contain date-like '
				          'structure and parse as datetimes.')
			elif identifier_evidence:
				analytical_role = 'identifier'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name identifies a key or ID.'
			elif ordinal_evidence:
				analytical_role = 'ordinal'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name indicates an ordered variable.'
			elif category_evidence or name_evidence:
				analytical_role = 'categorical'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name indicates categorical text.'
			else:
				analytical_role = 'categorical'
				confidence = 0.95
				reason = f'SQLite declares the column as text type {declared_type}.'
	elif pd.api.types.is_datetime64_any_dtype( series ):
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 1.0
		reason = 'The pandas storage dtype is datetime.'
	elif pd.api.types.is_bool_dtype( series ) or boolean_candidate:
		analytical_role = 'categorical'
		inferred_dtype = 'boolean'
		confidence = 0.98
		reason = 'The values form a boolean or two-state category.'
	elif date_name_evidence and datetime_success > 0.0:
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 0.96
		reason = 'The column name indicates a date or time and populated values parse as datetimes.'
	elif datetime_value_evidence:
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 0.95
		reason = 'All populated values contain date-like structure and parse as datetimes.'
	elif identifier_evidence:
		analytical_role = 'identifier'
		inferred_dtype = 'integer' if integer_like else 'text'
		confidence = 0.95
		reason = 'The column name identifies a key or ID.'
	elif numeric_success == 1.0 and populated_count > 0:
		inferred_dtype = 'integer' if integer_like else 'float'
		if ordinal_evidence:
			analytical_role = 'ordinal'
			confidence = 0.95
			reason = 'Numeric values and the column name indicate an ordered variable.'
		elif category_evidence:
			analytical_role = 'categorical'
			confidence = 0.95
			reason = 'Numeric values are used by a column whose name indicates a category or code.'
		else:
			analytical_role = 'numeric'
			confidence = 0.95
			reason = 'All populated values parse as numeric values.'
	elif isinstance( series.dtype, pd.CategoricalDtype ):
		analytical_role = 'categorical'
		inferred_dtype = 'category'
		confidence = 1.0
		reason = 'The pandas storage dtype is categorical.'
	
	return { 'column': column_name, 'storage_dtype': storage_dtype,
			'inferred_dtype': inferred_dtype, 'analytical_role': analytical_role,
			'non_null_count': populated_count, 'distinct_count': distinct_count,
			'unique_ratio': unique_ratio, 'numeric_success_ratio': numeric_success,
			'datetime_success_ratio': datetime_success, 'leading_zero_ratio': leading_zero_ratio,
			'is_integer_like': integer_like, 'confidence': confidence, 'reason': reason }

def get_column_name_tokens( column_name: str ) -> List[ str ]:
	"""Return normalized semantic tokens from a dataframe column name.

	Purpose:
	    Splits camel-case, whitespace-delimited, and punctuation-delimited column names into exact
	    lowercase tokens so schema inference can use semantic evidence without unsafe substring
	    matches.

	Args:
	    column_name (str): Source dataframe column name.

	Returns:
	    List[str]: Ordered normalized name tokens.
	"""
	throw_if( 'column_name', column_name )
	spaced_name = re.sub( r'([a-z0-9])([A-Z])', r'\1 \2', str( column_name ) )
	return [ token for token in re.split( r'[^A-Za-z0-9]+', spaced_name.lower( ) ) if token ]

def get_populated_mask( series: pd.Series ) -> pd.Series:
	"""Return the populated-value mask for a series.

	Purpose:
	    Identifies values that are neither null nor blank text so conversion-success ratios use the
	    number of populated observations rather than the dataframe row count.

	Args:
	    series (pd.Series): Series evaluated for populated values.

	Returns:
	    pd.Series: Boolean mask aligned to the source series.
	"""
	throw_if( 'series', series )
	text_values = series.astype( 'string' ).str.strip( )
	return series.notna( ) & text_values.ne( '' )

def parse_numeric_series( series: pd.Series ) -> pd.Series:
	"""Parse numeric values without modifying the source series.

	Purpose:
	    Converts native numbers and common database or spreadsheet numeric text into numeric values.
	    Thousands separators, currency symbols, percentage suffixes, and accounting parentheses are
	    handled once in a dedicated conversion path while unparseable values become missing.

	Args:
	    series (pd.Series): Source values evaluated as a numeric candidate.

	Returns:
	    pd.Series: Numeric values aligned to the source index.
	"""
	throw_if( 'series', series )
	if pd.api.types.is_numeric_dtype( series ) and not pd.api.types.is_bool_dtype( series ):
		return pd.to_numeric( series, errors='coerce' )
	
	text_values = series.astype( 'string' ).str.strip( )
	accounting_mask = text_values.str.match( r'^\(.*\)$', na=False )
	cleaned_values = text_values.str.replace( r'^\((.*)\)$', r'-\1', regex=True )
	cleaned_values = cleaned_values.str.replace( r'[$£€¥,]', '', regex=True )
	cleaned_values = cleaned_values.str.replace( r'\s+', '', regex=True )
	cleaned_values = cleaned_values.str.replace( r'%$', '', regex=True )
	parsed_values = pd.to_numeric( cleaned_values, errors='coerce' )
	parsed_values.loc[ accounting_mask & parsed_values.gt( 0 ) ] *= -1
	return parsed_values

def get_declared_type_family( declared_type: str ) -> str:
	"""Return the physical type family for a declared SQLite column type.

	Purpose:
	    Maps SQLite declared type names into stable numeric, datetime, boolean, text, or unknown
	    families so database-backed datasets use authoritative schema metadata before analytical
	    column-name semantics are applied.

	Args:
	    declared_type (str): SQLite declared type returned by ``PRAGMA table_info``.

	Returns:
	    str: Normalized physical type family used by the schema classifier.
	"""
	if not declared_type:
		return 'unknown'
	
	type_name = declared_type.strip( ).upper( )
	if any( token in type_name for token in ('DATE', 'TIME', 'TIMESTAMP') ):
		return 'datetime'
	if 'BOOL' in type_name:
		return 'boolean'
	if any( token in type_name for token in ('INT', 'REAL', 'FLOA', 'DOUB', 'NUMERIC', 'DECIMAL') ):
		return 'numeric'
	if any( token in type_name for token in ('CHAR', 'CLOB', 'TEXT', 'VARCHAR', 'NVARCHAR') ):
		return 'text'
	return 'unknown'

def create_declared_schema_map( schema_rows: List[ Tuple ] ) -> Dict[ str, str ]:
	"""Create a declared-type mapping from SQLite schema metadata.

	Purpose:
	    Converts ``PRAGMA table_info`` rows into a column-to-declared-type mapping that can be
	    passed to the dataframe schema profiler without coupling the profiler to database I/O.

	Args:
	    schema_rows (List[Tuple]): SQLite ``PRAGMA table_info`` rows.

	Returns:
	    Dict[str, str]: Declared SQLite type keyed by column name.
	"""
	if not schema_rows:
		return { }
	return { str( row[ 1 ] ): str( row[ 2 ] or '' ) for row in schema_rows }

def parse_datetime_series( series: pd.Series ) -> pd.Series:
	"""Parse datetime values without modifying the source series.

	Purpose:
	    Converts native datetimes and populated date-like text into pandas datetime values while
	    rejecting unparseable values. Mixed-format parsing is used when supported by pandas, with a
	    compatibility fallback for earlier versions.

	Args:
	    series (pd.Series): Source values evaluated as a datetime candidate.

	Returns:
	    pd.Series: Datetime values aligned to the source index.
	"""
	throw_if( 'series', series )
	if pd.api.types.is_datetime64_any_dtype( series ):
		return pd.to_datetime( series, errors='coerce' )
	
	try:
		text_values = series.astype( 'string' ).str.strip( )
		parsed_values = pd.to_datetime( text_values, errors='coerce', format='mixed' )
		if parsed_values.notna( ).any( ) or not get_populated_mask( series ).any( ):
			return parsed_values
		return pd.to_datetime( text_values, errors='coerce' )
	except (TypeError, ValueError):
		return pd.to_datetime( series.astype( 'string' ).str.strip( ), errors='coerce' )

def apply_mathy_plotly_theme( figure: go.Figure, title: str = '', height: int = 500 ) -> go.Figure:
	"""Apply the Mathy Plotly presentation theme.

	Purpose:
	    Applies consistent typography, spacing, transparent backgrounds, hover styling, legends,
	    axes, and responsive dimensions to a Plotly figure before Streamlit renders it.

	Args:
	    figure (go.Figure): Plotly figure receiving the shared presentation settings.
	    title (str): Optional chart title displayed above the plotting area.
	    height (int): Figure height in pixels.

	Returns:
	    go.Figure: The themed Plotly figure.
	"""
	throw_if( 'figure', figure )
	figure.update_layout( template='plotly_dark', title={ 'text': title, 'x': 0.01,
			'xanchor': 'left' }, height=height, autosize=True, paper_bgcolor='#171A1F', plot_bgcolor='#1B2028', font={
			'family': 'Arial, sans-serif', 'size': 13, 'color': '#E2E8F0' }, margin={ 'l': 55,
			'r': 30, 't': 70 if title else 35, 'b': 55 }, legend={ 'orientation': 'h',
			'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1.0 }, hoverlabel={
			'bgcolor': '#0F172A', 'font_size': 13, 'font_family': 'Arial, sans-serif' }, colorway=[
			'#38BDF8', '#A78BFA', '#2DD4BF', '#F59E0B', '#F472B6', '#60A5FA', '#34D399',
			'#FB7185' ] )
	
	figure.update_xaxes( showgrid=True, gridcolor='rgba(148,163,184,0.16)', zeroline=False, showline=True, linecolor='rgba(148,163,184,0.35)' )
	figure.update_yaxes( showgrid=True, gridcolor='rgba(148,163,184,0.16)', zeroline=False, showline=True, linecolor='rgba(148,163,184,0.35)' )
	return figure

def render_mathy_plotly_chart( figure: go.Figure, key: str, filename: str, title: str = '', height: int = 500 ) -> None:
	"""Render a themed Plotly chart.

	Purpose:
	    Applies the shared Mathy Plotly theme and renders the figure with a responsive toolbar,
	    high-resolution PNG export, zooming, panning, autoscaling, and no Plotly branding.

	Args:
	    figure (go.Figure): Plotly figure rendered in Streamlit.
	    key (str): Unique Streamlit element key assigned to the chart.
	    filename (str): Base filename used by the Plotly image-download control.
	    title (str): Optional chart title.
	    height (int): Figure height in pixels.

	Returns:
	    None: This function renders the chart and does not return a value.
	"""
	throw_if( 'figure', figure )
	throw_if( 'key', key )
	throw_if( 'filename', filename )
	apply_mathy_plotly_theme( figure, title, height )
	chart_config = { 'displaylogo': False, 'responsive': True, 'scrollZoom': True,
			'toImageButtonOptions': { 'format': 'png', 'filename': filename, 'scale': 2 } }
	st.plotly_chart( figure, use_container_width=True, key=key, config=chart_config )

_streamlit_data_editor = st.data_editor

def clear_keys( keys: List[ str ] ) -> None:
	"""Remove specified keys from Streamlit session state.

	Purpose:
	    Deletes each supplied session-state key when present so mode-specific workflows can reset
	    their owned state without raising errors for keys that have not been initialized.

	Args:
	    keys (List[str]): Session-state keys to remove.

	Returns:
	    None: This function mutates ``st.session_state`` in place.
	"""
	for key in keys:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_session_keys( keys: List[ str ] ) -> None:
	"""Reset Streamlit widget state before controls are recreated.

	Purpose:
	    Removes an explicit collection of widget-owned session-state keys from a Streamlit callback
	    so reset operations occur before the affected controls are instantiated on the subsequent
	    application rerun.

	Args:
	    keys (List[str]): Widget session-state keys removed by the reset callback.

	Returns:
	    None: This function delegates session-state removal to ``clear_keys``.
	"""
	clear_keys( keys )

def reset_session_prefixes( prefixes: Tuple[ str, ... ] ) -> None:
	"""Reset Streamlit widget state matching one or more key prefixes.

	Purpose:
	    Finds dynamic CRUD widget keys owned by the supplied prefixes and clears them from a
	    Streamlit callback before the corresponding controls are recreated.

	Args:
	    prefixes (Tuple[str, ...]): Session-state key prefixes identifying controls to reset.

	Returns:
	    None: This function removes matching session-state keys in place.
	"""
	throw_if( 'prefixes', prefixes )
	keys = [ key for key in st.session_state.keys( ) if str( key ).startswith( prefixes ) ]
	clear_keys( keys )

def table_supports_rowid( table: str ) -> bool:
	"""Determine whether a SQLite table exposes the implicit rowid.

	Purpose:
	    Detects ``WITHOUT ROWID`` table definitions before CRUD workflows issue rowid-based
	    selection, update, delete, or verification statements.

	Args:
	    table (str): Existing SQLite table inspected for rowid support.

	Returns:
	    bool: ``True`` when the table exposes SQLite rowid semantics; otherwise ``False``.
	"""
	throw_if( 'table', table )
	with create_connection( ) as conn:
		row = conn.execute( "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (
				table,) ).fetchone( )
	if row is None or not row[ 0 ]:
		raise ValueError( f'Table definition for "{table}" could not be found.' )
	return 'WITHOUT ROWID' not in str( row[ 0 ] ).upper( )

def read_table_with_rowid( table: str ) -> pd.DataFrame:
	"""Read a SQLite table with its internal row identifier.

	Purpose:
	    Reads the selected SQLite table while exposing the internal ``rowid`` under an application-
	    owned column name. The internal identifier is used only by CRUD workflows so dataframe row
	    positions are never assumed to be SQLite row identifiers.

	Args:
	    table (str): SQLite table read into dataframe form.

	Returns:
	    pd.DataFrame: Table rows containing ``__mathy_rowid__`` followed by the stored columns.
	"""
	throw_if( 'table', table )
	with create_connection( ) as conn:
		return pd.read_sql_query( f'SELECT rowid AS "__mathy_rowid__", * FROM "{table}";', conn )

def read_database_record( table: str, rowid: int ) -> Dict[ str, object ] | None:
	"""Read one SQLite record by internal row identifier.

	Purpose:
	    Retrieves the current values of a selected database record so CRUD controls are populated
	    from authoritative SQLite state instead of placeholder values or dataframe row positions.

	Args:
	    table (str): SQLite table containing the requested record.
	    rowid (int): SQLite internal row identifier.

	Returns:
	    Dict[str, object] | None: Column/value mapping for the record, or ``None`` when the row does
	        not exist.
	"""
	throw_if( 'table', table )
	throw_if( 'rowid', rowid )
	with create_connection( ) as conn:
		cursor = conn.execute( f'SELECT * FROM "{table}" WHERE rowid=?;', (int( rowid ),) )
		row = cursor.fetchone( )
		if row is None:
			return None
		columns = [ description[ 0 ] for description in cursor.description or [ ] ]
		return { column: row[ index ] for index, column in enumerate( columns ) }

def coerce_sqlite_value( value: object, declared_type: str ) -> object:
	"""Coerce an edited control value to its declared SQLite storage family.

	Purpose:
	    Converts text-oriented CRUD inputs to stable Python values compatible with the selected
	    column's SQLite declared type. Empty non-text inputs become ``None`` while text values retain
	    an intentionally entered empty string.

	Args:
	    value (object): User-edited value prepared for persistence.
	    declared_type (str): SQLite declared type associated with the target column.

	Returns:
	    object: Converted SQLite-compatible scalar value.
	"""
	family = get_declared_type_family( declared_type )
	if value is None:
		return None
	if isinstance( value, str ):
		text = value.strip( )
		if family != 'text' and text == '':
			return None
	else:
		text = str( value )
	
	if family == 'numeric':
		parsed = pd.to_numeric( pd.Series( [ text ] ), errors='coerce' ).iloc[ 0 ]
		if pd.isna( parsed ):
			raise ValueError( f'Value "{value}" is not valid for SQLite type {declared_type}.' )
		if 'INT' in declared_type.upper( ):
			if float( parsed ) % 1 != 0:
				raise ValueError( f'Value "{value}" is not a whole number.' )
			return int( parsed )
		return float( parsed )
	if family == 'boolean':
		if isinstance( value, bool ):
			return int( value )
		normalized = text.lower( )
		if normalized in ('1', 'true', 'yes', 'y', 'on'):
			return 1
		if normalized in ('0', 'false', 'no', 'n', 'off'):
			return 0
		raise ValueError( f'Value "{value}" is not valid for SQLite type {declared_type}.' )
	if family == 'datetime':
		parsed = pd.to_datetime( text, errors='coerce' )
		if pd.isna( parsed ):
			raise ValueError( f'Value "{value}" is not a valid datetime.' )
		return parsed.isoformat( )
	if 'BLOB' in declared_type.upper( ):
		return value if isinstance( value, bytes ) else text.encode( 'utf-8' )
	return str( value )

def create_crud_database_table( table_name: str,
		columns: List[ Dict[ str, str ] ] ) -> Tuple[ str, bool ]:
	"""Create a SQLite table from CRUD Ops schema controls.

	Purpose:
	    Creates a new user-defined SQLite table with an automatic ``Id INTEGER PRIMARY KEY
	    AUTOINCREMENT`` column, validates and normalizes each requested column identifier, preserves
	    the selected SQLite data types, and optionally inserts one initial row when at least one
	    initial field value is supplied.

	Args:
	    table_name (str): User-entered table name normalized to a valid SQLite identifier.
	    columns (List[Dict[str, str]]): User-defined column specifications containing ``name``,
	        ``type``, and optional ``initial_value`` values.

	Returns:
	    Tuple[str, bool]: Created SQLite table name followed by a flag indicating whether an initial
	        row was inserted.
	"""
	throw_if( 'table_name', table_name )
	throw_if( 'columns', columns )
	safe_table_name = create_identifier( table_name )
	existing_tables = { name.lower( ) for name in list_tables( ) }
	if safe_table_name.lower( ) in existing_tables:
		raise ValueError( f'Table "{safe_table_name}" already exists.' )
	
	supported_types = { 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' }
	column_definitions: List[ str ] = [ '"Id" INTEGER PRIMARY KEY AUTOINCREMENT' ]
	column_names: List[ str ] = [ ]
	initial_values: List[ object ] = [ ]
	has_initial_values = False
	seen_columns = { 'id' }
	
	for index, column in enumerate( columns ):
		column_name = str( column.get( 'name', '' ) )
		throw_if( f'column_{index + 1}_name', column_name )
		safe_column_name = create_identifier( column_name )
		if safe_column_name.lower( ) in seen_columns:
			raise ValueError( f'Column name "{safe_column_name}" is duplicated or reserved.' )
		seen_columns.add( safe_column_name.lower( ) )
		
		declared_type = str( column.get( 'type', 'TEXT' ) ).upper( )
		if declared_type not in supported_types:
			raise ValueError( f'Unsupported SQLite type: {declared_type}' )
		
		raw_initial_value = column.get( 'initial_value', '' )
		initial_text = str( raw_initial_value ).strip( ) if raw_initial_value is not None else ''
		if initial_text:
			initial_value = coerce_sqlite_value( raw_initial_value, declared_type )
			has_initial_values = True
		else:
			initial_value = None
		
		column_definitions.append( f'"{safe_column_name}" {declared_type}' )
		column_names.append( safe_column_name )
		initial_values.append( initial_value )
	
	create_statement = (f'CREATE TABLE "{safe_table_name}" '
	                    f'({", ".join( column_definitions )});')
	
	with create_connection( ) as conn:
		try:
			conn.execute( 'BEGIN' )
			conn.execute( create_statement )
			if has_initial_values:
				column_list = ', '.join( f'"{name}"' for name in column_names )
				placeholders = ', '.join( [ '?' ] * len( column_names ) )
				conn.execute( f'INSERT INTO "{safe_table_name}" ({column_list}) '
				              f'VALUES ({placeholders});', initial_values )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise
	
	created_schema = create_schema( safe_table_name )
	created_columns = [ str( row[ 1 ] ) for row in created_schema ]
	expected_columns = [ 'Id' ] + column_names
	if created_columns != expected_columns:
		raise RuntimeError( 'Created table schema could not be verified.' )
	if has_initial_values:
		df_created = read_table( safe_table_name )
		if len( df_created ) != 1:
			raise RuntimeError( 'Initial table row could not be verified.' )
	
	return safe_table_name, has_initial_values

def database_record_exists( table: str,
		values: Dict[ str, object ], exclude_rowid: int = 0 ) -> bool:
	"""Determine whether an equivalent complete SQLite record already exists.

	Purpose:
	    Compares every supplied persisted field against existing rows so Add Row can prevent exact
	    duplicate records without incorrectly prohibiting legitimate reuse of individual categorical
	    or dimensional values.

	Args:
	    table (str): SQLite table checked for a duplicate record.
	    values (Dict[str, object]): Persisted column/value mapping for the proposed record.
	    exclude_rowid (int): Optional row identifier excluded from duplicate detection during edits.

	Returns:
	    bool: ``True`` when an equivalent record exists; otherwise ``False``.
	"""
	throw_if( 'table', table )
	throw_if( 'values', values )
	clauses = [ f'"{column}" IS ?' for column in values.keys( ) ]
	parameters = list( values.values( ) )
	if exclude_rowid > 0:
		clauses.append( 'rowid <> ?' )
		parameters.append( int( exclude_rowid ) )
	query = f'SELECT 1 FROM "{table}" WHERE {" AND ".join( clauses )} LIMIT 1;'
	with create_connection( ) as conn:
		return conn.execute( query, parameters ).fetchone( ) is not None

def database_record_matches( table: str, rowid: int, values: Dict[ str, object ] ) -> bool:
	"""Verify persisted values for one SQLite record.

	Purpose:
	    Confirms that a committed row contains the exact SQLite values submitted by a CRUD update or
	    insert operation so success messages are emitted only after database readback verification.

	Args:
	    table (str): SQLite table containing the record.
	    rowid (int): SQLite internal row identifier of the record.
	    values (Dict[str, object]): Expected persisted column/value mapping.

	Returns:
	    bool: ``True`` when the selected row contains all expected values; otherwise ``False``.
	"""
	throw_if( 'table', table )
	throw_if( 'rowid', rowid )
	throw_if( 'values', values )
	clauses = [ 'rowid=?' ] + [ f'"{column}" IS ?' for column in values.keys( ) ]
	parameters = [ int( rowid ) ] + list( values.values( ) )
	query = f'SELECT 1 FROM "{table}" WHERE {" AND ".join( clauses )} LIMIT 1;'
	with create_connection( ) as conn:
		return conn.execute( query, parameters ).fetchone( ) is not None

def refresh_crud_database_state( table: str ) -> pd.DataFrame:
	"""Refresh CRUD and active analytical state after a committed SQLite mutation.

	Purpose:
	    Re-reads the selected SQLite table after a successful mutation. When the table is also the
	    active Mathy database source, the loaded dataset, original baseline, declared schema, and
	    analytical column classifications are synchronized to the committed database state.

	Args:
	    table (str): SQLite table whose committed state is refreshed.

	Returns:
	    pd.DataFrame: Fresh dataframe read from the selected table.
	"""
	throw_if( 'table', table )
	df_database = read_table( table )
	if (
			st.session_state.get( 'active_source_kind', '' ) == 'database' \
			and st.session_state.get( 'active_database_table', '' ) == table):
		st.session_state[ 'active_declared_schema' ] = create_declared_schema_map(
			create_schema( table ) )
		if has_loaded_dataset( df_database ):
			store_loaded_dataset( df_database, df_database )
		else:
			st.session_state[ 'raw_df' ] = df_database.copy( )
			st.session_state[ 'df_original' ] = df_database.copy( )
			st.session_state[ 'df_dataset' ] = df_database.copy( )
			synchronize_dataset_columns( df_database )
	return df_database

def queue_database_success( message: str ) -> None:
	"""Queue a database success message for the next Streamlit render.

	Purpose:
	    Stores one verified database-operation result in session state so a success dialog can be
	    rendered after the operation-triggered rerun without losing the acknowledgement.

	Args:
	    message (str): Success message displayed in the database operation dialog.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	throw_if( 'message', message )
	st.session_state[ 'database_operation_result' ] = str( message )

def activate_database_table( table: str ) -> pd.DataFrame:
	"""Activate one persisted SQLite table as the Mathy dataset.

	Purpose:
	    Reads the selected table and declared SQLite schema, records the database-backed source
	    identity, and synchronizes the active, original, raw, typed, and analytical dataset state.

	Args:
	    table (str): Existing SQLite table activated as the current Mathy dataset.

	Returns:
	    pd.DataFrame: Fresh dataframe read from the activated SQLite table.
	"""
	throw_if( 'table', table )
	if table not in list_tables( ):
		raise ValueError( f'Table "{table}" does not exist.' )
	df_database = read_table( table )
	st.session_state[ 'active_source_kind' ] = 'database'
	st.session_state[ 'active_database_table' ] = table
	st.session_state[
		'active_declared_schema' ] = create_declared_schema_map( create_schema( table ) )
	st.session_state[ 'active_source_signature' ] = ('Database Data', table)
	if has_loaded_dataset( df_database ):
		store_loaded_dataset( df_database, df_database )
	else:
		st.session_state[ 'raw_df' ] = df_database.copy( )
		st.session_state[ 'df_original' ] = df_database.copy( )
		st.session_state[ 'df_dataset' ] = df_database.copy( )
		synchronize_dataset_columns( df_database )
	return df_database

def clear_active_database_table( ) -> None:
	"""Clear active database-backed dataset state.

	Purpose:
	    Removes a deleted database table from the active source contract and resets loaded dataset
	    state when no replacement table is available.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	st.session_state[ 'active_source_kind' ] = ''
	st.session_state[ 'active_database_table' ] = ''
	st.session_state[ 'active_declared_schema' ] = { }
	st.session_state[ 'active_source_signature' ] = None
	st.session_state[ 'raw_df' ] = pd.DataFrame( )
	st.session_state[ 'df_original' ] = pd.DataFrame( )
	st.session_state[ 'df_dataset' ] = pd.DataFrame( )
	synchronize_dataset_columns( pd.DataFrame( ) )

def create_available_table_names( requested_names: List[ str ],
		existing_tables: List[ str ] ) -> List[ str ]:
	"""Create collision-safe SQLite table names.

	Purpose:
	    Normalizes requested names to valid identifiers and appends numeric suffixes when a requested
	    name conflicts with an existing or already-reserved table name.

	Args:
	    requested_names (List[str]): Source names converted to database table identifiers.
	    existing_tables (List[str]): Current database tables reserved against collisions.

	Returns:
	    List[str]: Unique SQLite-safe table names in source order.
	"""
	throw_if( 'requested_names', requested_names )
	reserved = { str( table ).lower( ) for table in existing_tables }
	available: List[ str ] = [ ]
	for requested_name in requested_names:
		base_name = create_identifier( requested_name )
		candidate = base_name
		suffix = 2
		while candidate.lower( ) in reserved:
			candidate = f'{base_name}_{suffix}'
			suffix += 1
		reserved.add( candidate.lower( ) )
		available.append( candidate )
	return available

def write_dataframe_tables_to_database( df_tables: Dict[ str, pd.DataFrame ],
		overwrite: bool = False ) -> Dict[ str, pd.DataFrame ]:
	"""Persist dataframe tables to SQLite in one transaction.

	Purpose:
	    Normalizes uploaded dataframe column identifiers, creates each requested SQLite table, bulk
	    inserts normalized values, and rolls back the entire operation when any table fails.

	Args:
	    df_tables (Dict[str, pd.DataFrame]): Desired table names mapped to source dataframes.
	    overwrite (bool): Whether existing tables with matching names may be replaced.

	Returns:
	    Dict[str, pd.DataFrame]: Fresh persisted dataframes keyed by created table name.
	"""
	throw_if( 'df_tables', df_tables )
	with create_connection( ) as conn:
		try:
			conn.execute( 'BEGIN' )
			for requested_name, df_source in df_tables.items( ):
				throw_if( 'requested_name', requested_name )
				throw_if( 'df_source', df_source )
				if len( df_source.columns ) == 0:
					raise ValueError( f'Table "{requested_name}" does not contain any columns.' )
				table_name = create_identifier( requested_name )
				df_table = df_source.copy( )
				df_table.columns = create_unique_upload_identifiers( df_table.columns.tolist( ) )
				if overwrite:
					conn.execute( f'DROP TABLE IF EXISTS "{table_name}";' )
				elif table_name in list_tables( ):
					raise ValueError( f'Table "{table_name}" already exists.' )
				columns: List[ str ] = [ ]
				for column in df_table.columns:
					sql_type = get_sqlite_type( df_table[ column ].dtype )
					columns.append( f'"{column}" {sql_type}' )
				conn.execute( f'CREATE TABLE "{table_name}" ({", ".join( columns )});' )
				if not df_table.empty:
					placeholders = ', '.join( [ '?' ] * len( df_table.columns ) )
					rows = [ tuple( normalize_upload_sql_value( value ) for value in row ) for row
							in df_table.itertuples( index=False, name=None ) ]
					conn.executemany( f'INSERT INTO "{table_name}" VALUES ({placeholders});', rows )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise
	
	persisted_tables: Dict[ str, pd.DataFrame ] = { }
	for table_name, df_source in df_tables.items( ):
		created_name = create_identifier( table_name )
		df_persisted = read_table( created_name )
		if len( df_persisted ) != len( df_source ) or len( df_persisted.columns ) != len( df_source.columns ):
			raise RuntimeError( f'Table "{created_name}" could not be verified after import.' )
		persisted_tables[ created_name ] = df_persisted
	return persisted_tables

def reconcile_import_table_metadata( old_table: str, new_table: str = '' ) -> None:
	"""Reconcile upload metadata after a table rename or deletion.

	Purpose:
	    Updates sidebar and Data Upload table mappings so renamed or deleted SQLite tables do not
	    remain as stale session-state references.

	Args:
	    old_table (str): Previous table name removed from metadata mappings.
	    new_table (str): Replacement table name for a rename; empty for deletion.

	Returns:
	    None: This function updates upload-related session state in place.
	"""
	throw_if( 'old_table', old_table )
	for key in ('data_upload_tables', 'custom_upload_tables'):
		table_map = st.session_state.get( key, { } )
		if not isinstance( table_map, dict ) or old_table not in table_map:
			continue
		value = table_map.pop( old_table )
		if new_table:
			table_map[
				new_table ] = read_table( new_table ) if new_table in list_tables( ) else value
		st.session_state[ key ] = table_map
		if key == 'data_upload_tables':
			if st.session_state.get( 'data_upload_selected_table', None ) == old_table:
				if new_table:
					st.session_state[ 'data_upload_selected_table' ] = new_table
				else:
					clear_keys( [ 'data_upload_selected_table' ] )
			if not table_map and not new_table:
				st.session_state[ 'data_upload_signature' ] = None
		elif key == 'custom_upload_tables' and not table_map and not new_table:
			st.session_state[ 'custom_upload_signature' ] = None
	
	sheet_names = st.session_state.get( 'data_upload_sheet_names', { } )
	if isinstance( sheet_names, dict ) and old_table in sheet_names:
		value = sheet_names.pop( old_table )
		if new_table:
			sheet_names[ new_table ] = value
		st.session_state[ 'data_upload_sheet_names' ] = sheet_names

def split_sqlite_definitions( sql_text: str ) -> List[ str ]:
	"""Split a SQLite table-definition body at top-level commas.

	Purpose:
	    Separates column and table-constraint definitions while preserving commas contained inside
	    quoted text, parentheses, default expressions, and check constraints.

	Args:
	    sql_text (str): Text contained inside a SQLite ``CREATE TABLE`` parenthesized definition.

	Returns:
	    List[str]: Ordered top-level SQL definitions.
	"""
	throw_if( 'sql_text', sql_text )
	definitions: List[ str ] = [ ]
	buffer: List[ str ] = [ ]
	depth = 0
	quote = ''
	for char in sql_text:
		if quote:
			buffer.append( char )
			if char == quote:
				quote = ''
			continue
		if char in ('"', "'", '`'):
			quote = char
			buffer.append( char )
		elif char == '(':
			depth += 1
			buffer.append( char )
		elif char == ')':
			depth = max( 0, depth - 1 )
			buffer.append( char )
		elif char == ',' and depth == 0:
			definitions.append( ''.join( buffer ).strip( ) )
			buffer = [ ]
		else:
			buffer.append( char )
	if buffer:
		definitions.append( ''.join( buffer ).strip( ) )
	return definitions

def replace_sqlite_column_type( create_sql: str, table: str, column: str,
		new_type: str, temp_table: str ) -> str:
	"""Build a temporary CREATE TABLE statement with one revised declared type.

	Purpose:
	    Preserves the original SQLite column definitions and table-level constraints while replacing
	    only the selected column's declared type and temporary table name for an atomic schema rebuild.

	Args:
	    create_sql (str): Original SQLite ``CREATE TABLE`` statement.
	    table (str): Existing table name.
	    column (str): Column whose declared type is changed.
	    new_type (str): Replacement SQLite declared type.
	    temp_table (str): Temporary table name used during reconstruction.

	Returns:
	    str: Temporary ``CREATE TABLE`` statement containing the revised declared type.
	"""
	throw_if( 'create_sql', create_sql )
	throw_if( 'table', table )
	throw_if( 'column', column )
	throw_if( 'new_type', new_type )
	throw_if( 'temp_table', temp_table )
	open_paren = create_sql.find( '(' )
	close_paren = create_sql.rfind( ')' )
	if open_paren < 0 or close_paren <= open_paren:
		raise ValueError( 'Malformed CREATE TABLE statement.' )
	
	definitions = split_sqlite_definitions( create_sql[ open_paren + 1: close_paren ] )
	constraint_tokens = { 'PRIMARY', 'NOT', 'UNIQUE', 'CHECK', 'DEFAULT', 'COLLATE', 'REFERENCES',
			'GENERATED' }
	updated = False
	for index, definition in enumerate( definitions ):
		match = re.match( r'^\s*(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([^\s]+))\s+(.*)$', definition, flags=re.DOTALL )
		if not match:
			continue
		name = next( (group for group in match.groups( )[ :4 ] if group is not None), '' )
		if name != column:
			continue
		remainder = match.group( 5 ).strip( )
		parts = remainder.split( )
		constraint_index = next( (position for position, token in enumerate( parts ) if
		token.upper( ) in constraint_tokens), len( parts ) )
		constraints = ' '.join( parts[ constraint_index: ] )
		quoted_name = f'"{column}"'
		definitions[ index ] = f'{quoted_name} {new_type}' + (
				f' {constraints}' if constraints else '')
		updated = True
		break
	if not updated:
		raise ValueError( f'Column "{column}" was not found in the table definition.' )
	
	suffix = create_sql[ close_paren + 1: ].strip( ).rstrip( ';' )
	return f'CREATE TABLE "{temp_table}" ({", ".join( definitions )})' + (
			f' {suffix};' if suffix else ';')

def change_column_type( table: str, column: str, new_type: str ) -> None:
	"""Change a SQLite column's declared type with an atomic table rebuild.

	Purpose:
	    Validates convertibility of populated values, recreates the table using its original schema
	    and constraints with one revised declared type, casts the selected column during data copy,
	    recreates explicit indexes and triggers, and commits the change as one transaction.

	Args:
	    table (str): Existing SQLite table containing the target column.
	    column (str): Column whose declared type is changed.
	    new_type (str): Supported replacement SQLite type.

	Returns:
	    None: This function changes the SQLite schema and stored values in place.
	"""
	throw_if( 'table', table )
	throw_if( 'column', column )
	throw_if( 'new_type', new_type )
	supported_types = { 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' }
	target_type = new_type.upper( )
	if target_type not in supported_types:
		raise ValueError( f'Unsupported SQLite type: {new_type}' )
	
	df_table = read_table( table )
	if column not in df_table.columns:
		raise ValueError( f'Column "{column}" does not exist.' )
	series = df_table[ column ]
	populated = get_populated_mask( series )
	if target_type in ('INTEGER', 'REAL', 'NUMERIC') and populated.any( ):
		parsed = parse_numeric_series( series )
		if not parsed[ populated ].notna( ).all( ):
			raise ValueError( f'Column "{column}" contains values that cannot be converted to {target_type}.' )
		if target_type == 'INTEGER' and not parsed[ populated ].dropna( ).mod( 1 ).eq( 0 ).all( ):
			raise ValueError( f'Column "{column}" contains non-integer numeric values.' )
	
	temp_table = f'{table}__mathy_type_rebuild'
	with create_connection( ) as conn:
		row = conn.execute( "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (
				table,) ).fetchone( )
		if row is None or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		create_sql = str( row[ 0 ] )
		index_sql = [ item[ 0 ] for item in
				conn.execute( "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;", (
						table,) ).fetchall( ) if item[ 0 ] ]
		trigger_sql = [ item[ 0 ] for item in
				conn.execute( "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL;", (
						table,) ).fetchall( ) if item[ 0 ] ]
		columns = [ str( item[ 1 ] ) for item in
				conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( ) ]
		new_create_sql = replace_sqlite_column_type( create_sql, table, column, target_type, temp_table )
		select_expressions = [
				(f"""CASE WHEN "{name}" IS NULL OR TRIM(CAST("{name}" AS TEXT)) = '' THEN NULL """
				 f'ELSE CAST("{name}" AS {target_type}) END' if name == column and target_type in (
						'INTEGER', 'REAL',
						'NUMERIC') else f'CAST("{name}" AS {target_type})' if name == column else f'"{name}"')
				for name in columns ]
		column_list = ', '.join( f'"{name}"' for name in columns )
		select_list = ', '.join( select_expressions )
		
		try:
			conn.execute( 'BEGIN' )
			conn.execute( f'DROP TABLE IF EXISTS "{temp_table}";' )
			conn.execute( new_create_sql )
			conn.execute( f'INSERT INTO "{temp_table}" ({column_list}) SELECT {select_list} FROM "{table}";' )
			conn.execute( f'DROP TABLE "{table}";' )
			conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
			for statement in index_sql:
				conn.execute( statement )
			for statement in trigger_sql:
				conn.execute( statement )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise

# -------- Expander Utilities

def set_blue_divider( ) -> None:
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

def set_sidebar_mode( source_key: str, alternate_key: str ) -> None:
	"""Set the active application mode from one sidebar radio group.

	Purpose:
		Preserve one authoritative application mode while presenting functionality and data-management
		choices in separate sidebar expanders. Selecting a choice clears the alternate radio group so
		the sidebar never displays two active modes.

	Args:
		source_key (str): Session-state key containing the newly selected mode.
		alternate_key (str): Session-state key for the radio group that must be cleared.

	Returns:
		None: This function updates sidebar selection values in Streamlit session state.
	"""
	selected_mode = st.session_state.get( source_key )
	if selected_mode:
		st.session_state[ 'mode' ] = selected_mode
		st.session_state[ alternate_key ] = None

def _promote_loader_documents( documents: List[ Document ] | None, active_loader: str ) -> int:
	docs: List[ Document ] = list( documents or [ ] )
	
	st.session_state.documents = docs
	st.session_state.raw_documents = list( docs )
	st.session_state.raw_text = '\n\n'.join( doc.page_content for doc in docs if
			hasattr( doc, 'page_content' ) and isinstance( doc.page_content, str ) and doc.page_content.strip( ) )
	st.session_state.processed_text = ''
	st.session_state.tokens = None
	st.session_state.vocabulary = None
	st.session_state.token_counts = None
	st.session_state.active_loader = active_loader
	return len( docs )

def _clear_loader_documents( loader_name: str ) -> int:
	clear_if_active( loader_name )
	st.session_state.raw_text = rebuild_raw_text_from_documents( ) or ''
	st.session_state.processed_text = ''
	st.session_state.tokens = None
	st.session_state.vocabulary = None
	st.session_state.token_counts = None
	remaining = st.session_state.get( 'documents' ) or [ ]
	return len( remaining )

# =========================================================================
# SESSION-STATE INITIALIZATION
# =========================================================================
for key, default in cfg.SESSION_STATE_DEFAULTS.items( ):
	if key not in st.session_state:
		st.session_state[ key ] = default

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )

# =========================================================================
# APP SET-UP
# =========================================================================
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON )
style_subheaders( )
st.logo( cfg.LOGO, size='large' )
col_left, col_center, col_right = st.columns( [ 1, 2, 1 ], vertical_alignment='top' )

# ===========================================================================
# SIDEBAR
# ===========================================================================
functionality_modes = [ configured_mode for configured_mode in cfg.MODE_MAP if
		configured_mode not in cfg.DB_MODES ]
management_modes = list( cfg.DB_MODES )
active_mode = st.session_state[ 'mode' ]
if 'functionality_mode' not in st.session_state:
	st.session_state[ 'functionality_mode' ] = (
			active_mode if active_mode in functionality_modes else None)
if 'management_mode' not in st.session_state:
	st.session_state[ 'management_mode' ] = (
			active_mode if active_mode in management_modes else None)

with st.sidebar:
	st.divider( )
	
	# ------------- Modes -------------
	st.text( '🎮 Mode' )
	with st.expander( label='Functionality', expanded=True ):
		st.radio( label='Mode', options=functionality_modes, index=None,
			key='functionality_mode', label_visibility='collapsed',
			on_change=set_sidebar_mode, args=('functionality_mode', 'management_mode') )
	
	st.divider( )
	
	# ------------- Data -------------
	st.text( '🗄️ Data' )
	with st.expander( label='Management', expanded=False ):
		st.radio( label='Management Mode', options=management_modes, index=None,
			key='management_mode', label_visibility='collapsed',
			on_change=set_sidebar_mode, args=( 'management_mode', 'functionality_mode') )
	
	st.divider( )
	
	# ------------- API -------------
	st.text( '💻 API' )
	with st.expander( 'Keys', expanded=False ):
		for attr in dir( cfg ):
			if attr.endswith( '_API_KEY' ) or attr.endswith( '_TOKEN' ):
				current = getattr( cfg, attr, "" ) or ""
				val = st.text_input( attr, value=current, type='password' )
				if val:
					os.environ[ attr ] = val

mode = st.session_state[ 'mode' ]

# =============================================================================
# DOCUMENT LOADING MODE
# =============================================================================
if mode == 'Loading':
	tokens = st.session_state[ 'tokens' ]
	documents = st.session_state[ 'documents' ]
	raw_text = st.session_state[ 'raw_text' ]
	
	st.subheader( '📤 Document Loading' )
	st.divider( )
	# ------------------------------------------------------------------
	# LEFT COLUMN - LOADERS
	# ------------------------------------------------------------------
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	with left:
		_loader_msg = st.session_state.pop( '_loader_status', None )
		if isinstance( _loader_msg, str ) and _loader_msg.strip( ):
			st.success( _loader_msg )
		
		with st.expander( label='Local Documents', expanded=True ):
			# ----------------------------
			# ------- Expander NLTK Loader
			# ----------------------------
			with st.expander( label='Corpora Loader', icon='📚', expanded=False ):
				import nltk
				from nltk.corpus import (brown, gutenberg, reuters, webtext, inaugural, state_union)
				
				st.markdown( '###### NLTK Corpora' )
				
				corpus_name = st.selectbox( 'Select corpus',
					['Brown', 'Gutenberg', 'Reuters', 'WebText', 'Inaugural', 'State of the Union'],
					key='nltk_corpus_name', )
				
				file_ids = [ ]
				try:
					if corpus_name == 'Brown':
						file_ids = brown.fileids( )
					elif corpus_name == 'Gutenberg':
						file_ids = gutenberg.fileids( )
					elif corpus_name == 'Reuters':
						file_ids = reuters.fileids( )
					elif corpus_name == 'WebText':
						file_ids = webtext.fileids( )
					elif corpus_name == 'Inaugural':
						file_ids = inaugural.fileids( )
					elif corpus_name == 'State of the Union':
						file_ids = state_union.fileids( )
				except LookupError:
					st.error( "NLTK corpus not found. Run:\n\npython -m nltk.downloader all\n\n"
					          "or download individual corpora." )
				
				selected_files = st.multiselect( 'Select files (leave empty to load all)',
					options=file_ids, key='nltk_file_ids', )
				
				st.divider( )
				st.markdown( '###### Local Corpus' )
				local_corpus_dir = st.text_input( 'Local directory',
					placeholder='path/to/text/files', key='nltk_local_dir', )
				
				# ------------------------------------------------------------------
				# Load / Clear / Save controls
				# ------------------------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_nltk = col_load.button( label='Load', key='nltk_load',
					icon='📤', width='stretch' )
				
				clear_nltk = col_clear.button( label='Clear', key='nltk_clear',
					icon='🧹', width='stretch' )
				
				_docs = st.session_state.get( 'documents' ) or [ ]
				_nltk_docs = [ d for d in _docs if d.metadata.get( 'loader' ) == 'NLTKLoader' ]
				_nltk_text = "\n\n".join( d.page_content for d in _nltk_docs )
				_export_name = f"nltk_{corpus_name.lower( ).replace( ' ', '_' )}.txt"
				col_save.download_button( 'Save', data=_nltk_text, file_name=_export_name,
					mime='text/plain', disabled=not bool( _nltk_text.strip( ) ),
					icon='💾', width='stretch' )
				
				# ------------------------------------------------------------------
				# Clear
				# ------------------------------------------------------------------
				if clear_nltk and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( 'loader' ) != 'NLTKLoader' ]
					
					st.session_state.raw_text = ("\n\n".join( d.page_content for d in
							st.session_state.documents ) if st.session_state.documents else None)
					
					st.session_state.active_loader = None
					st.info( 'NLTKLoader documents removed.' )
				
				# ------------------------------------------------------------------
				# Load
				# ------------------------------------------------------------------
				if load_nltk:
					documents = [ ]
					if file_ids:
						files_to_load = selected_files or file_ids
						for fid in files_to_load:
							try:
								if corpus_name == 'Brown':
									text = ' '.join( brown.words( fid ) )
								elif corpus_name == 'Gutenberg':
									text = gutenberg.raw( fid )
								elif corpus_name == 'Reuters':
									text = reuters.raw( fid )
								elif corpus_name == 'WebText':
									text = webtext.raw( fid )
								elif corpus_name == 'Inaugural':
									text = inaugural.raw( fid )
								elif corpus_name == 'State of the Union':
									text = state_union.raw( fid )
								
								if text.strip( ):
									documents.append( Document( page_content=text, metadata={
											'loader': 'NLTKLoader', 'corpus': corpus_name,
											'file_id': fid, }, ) )
							except Exception:
								continue
					
					# Local corpus
					if local_corpus_dir and os.path.isdir( local_corpus_dir ):
						for fname in os.listdir( local_corpus_dir ):
							path = os.path.join( local_corpus_dir, fname )
							if os.path.isfile( path ) and fname.lower( ).endswith( '.txt' ):
								with open( path, 'r', encoding='utf-8', errors='ignore' ) as f:
									text = f.read( )
								
								if text.strip( ):
									documents.append( Document( page_content=text, metadata={
											'loader': 'NLTKLoader', 'source': path, }, ) )
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in st.session_state.documents )
						
						st.session_state.processed_text = None
						st.session_state.active_loader = 'NLTKLoader'
						
						st.success( f'Loaded {len( documents )} document(s) from NLTK.' )
					else:
						st.warning( 'No documents were loaded.' )
			
			# ----------------------------
			# ------ Expander Text Loader
			# ----------------------------
			with st.expander( label='Text Loader', icon='📝', expanded=False ):
				files = st.file_uploader( 'Upload Text File(s)', type=[ 'txt', 'text', 'log' ],
					accept_multiple_files=True, key='txt_upload' )
				
				# ------------------------------------------------------------------
				# Buttons: Load / Clear / Save
				# ------------------------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_txt = col_load.button( label='Load', key='txt_load',
					icon='📤', width='stretch' )
				
				clear_txt = col_clear.button( label='Clear', key='txt_clear',
					icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'TextLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( label='Save', data=st.session_state.get( 'raw_text' ),
						file_name='text_loader_output.txt', mime='text/plain',
						key='txt_save', icon='💾', width='stretch' )
				else:
					col_save.button( label='Save', key='txt_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# ------------------------------------------------------------------
				# Clear
				# ------------------------------------------------------------------
				if clear_txt:
					clear_if_active( 'TextLoader' )
					st.info( 'Text Loader state cleared.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# Load
				# ------------------------------------------------------------------
				if load_txt and files:
					documents: list[ Document ] = [ ]
					
					with tempfile.TemporaryDirectory( ) as tmp:
						for uploaded_file in files:
							path = os.path.join( tmp, uploaded_file.name )
							
							with open( path, 'wb' ) as handle:
								handle.write( uploaded_file.read( ) )
							
							loader = TextLoader( )
							loaded = loader.load( path ) or [ ]
							for document in loaded:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'TextLoader'
								document.metadata.setdefault( 'source', uploaded_file.name )
							
							documents.extend( loaded )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					
					st.session_state.active_loader = 'TextLoader'
					st.success( f'Loaded {len( documents )} text document(s).' )
			
			# ----------------------------
			# ------ Expander CSV Loader
			# ----------------------------
			with st.expander( label="CSV Loader", icon='📑', expanded=False ):
				csv_file = st.file_uploader( label="Upload CSV", type=[ "csv" ], key="csv_upload" )
				delimiter = st.text_input( "Delimiter", value=",", key="csv_delim", )
				quotechar = st.text_input( "Quote Character", value='"', key="csv_quote", )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_csv = col_load.button( 'Load', key='csv_load', icon='📤', width='stretch' )
				clear_csv = col_clear.button( 'Clear', key='csv_clear', icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( 'active_loader' ) == 'CsvLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='csv_loader_output.txt', mime='text/plain',
						key='csv_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='csv_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_csv:
					clear_if_active( "CsvLoader" )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "CSV Loader state cleared."
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_csv and csv_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, csv_file.name )
						with open( path, "wb" ) as f:
							f.write( csv_file.read( ) )
						
						loader = CsvLoader( )
						documents = loader.load( path, columns=None, delimiter=delimiter,
							quotechar=quotechar, ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
							hasattr( d, "page_content" ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.active_loader = "CsvLoader"
					
					st.session_state[
						"_loader_status" ] = f"Loaded {len( documents )} CSV document(s)."
			
			# ----------------------------
			# ---- XML Loader
			# ----------------------------
			with st.expander( label='XML Loader', icon='🧬', expanded=False ):
				# ------------------------------------------------------------------
				# Session-backed loader instance
				# ------------------------------------------------------------------
				if 'xml_loader' not in st.session_state or st.session_state.xml_loader is None:
					st.session_state.xml_loader = XmlLoader( )
				
				loader = st.session_state.xml_loader
				xml_file = st.file_uploader( label='Select XML file', type=[ 'xml' ],
					accept_multiple_files=False, key='xml_file_uploader' )
				st.text( 'Semantic XML Loading (Unstructured)' )
				col1, col2 = st.columns( 2 )
				with col1:
					chunk_size = st.number_input( 'Chunk Size', min_value=100, max_value=5000,
						value=1000, step=100 )
				
				with col2:
					overlap_amount = st.number_input( 'Chunk Overlap', min_value=0,
						max_value=1000, value=200, step=50 )
				
				# --------------------------------------------------
				# Semantic Load
				# --------------------------------------------------
				if st.button( 'Load XML (Semantic)', use_container_width=True, icon='📤', ):
					if xml_file is None:
						st.warning( 'Please select an XML file.' )
					else:
						with tempfile.TemporaryDirectory( ) as tmp:
							path = os.path.join( tmp, xml_file.name )
							with open( path, 'wb' ) as f:
								f.write( xml_file.read( ) )
							
							with st.spinner( 'Loading XML via UnstructuredXMLLoader...' ):
								documents = loader.load( path )
						
						if documents:
							raw_text = '\n\n'.join( d.page_content for d in documents if
									hasattr( d, 'page_content' ) and isinstance(
										d.page_content, str ) and d.page_content.strip( ) )
							
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
							st.session_state.raw_text = raw_text
							st.session_state.processed_text = None
							st.session_state.active_loader = 'XmlLoader'
							st.session_state[ 'xml_documents' ] = documents
							st.session_state[ 'xml_tree_loaded' ] = False
							st.session_state[ 'xml_xpath_results' ] = None
							st.session_state[ 'xml_namespaces' ] = None
						else:
							st.warning( 'No extractable text found in XML.' )
				
				# --------------------------------------------------
				# Split Semantic Documents
				# --------------------------------------------------
				if st.button( 'Split Semantic Documents', use_container_width=True, icon='➗', ):
					with st.spinner( 'Splitting documents...' ):
						split_docs = loader.split( size=int( chunk_size ),
							amount=int( overlap_amount ) )
					
					if split_docs:
						st.session_state[ 'xml_split_documents' ] = split_docs
						st.success( f'Produced {len( split_docs )} document chunks.' )
				
				# ------------------------------------------------------------------
				# Structured XML Tree Loading
				# ------------------------------------------------------------------
				st.divider( )
				st.text( 'Structured XML Tree Loading (XPath)' )
				if st.button( 'Load XML Tree', use_container_width=True, icon='🎄', ):
					if xml_file is None:
						st.warning( 'Please select an XML file.' )
					else:
						with tempfile.TemporaryDirectory( ) as tmp:
							path = os.path.join( tmp, xml_file.name )
							with open( path, 'wb' ) as f:
								f.write( xml_file.read( ) )
							
							with st.spinner( 'Parsing XML into ElementTree...' ):
								tree = loader.load_tree( path )
						
						if tree is not None:
							xml_text = etree.tostring( tree, pretty_print=True, encoding='unicode' )
							
							st.session_state.raw_text = xml_text
							st.session_state.processed_text = None
							st.session_state.active_loader = 'XmlLoader'
							st.session_state[ 'xml_tree_loaded' ] = True
							st.session_state[ 'xml_namespaces' ] = loader.xml_namespaces
							st.session_state[ 'xml_xpath_results' ] = None
							st.success( 'XML tree loaded successfully.' )
						else:
							st.warning( 'Failed to parse XML tree.' )
				
				# ------------------------------------------------------------------
				# XPath Query Interface
				# ------------------------------------------------------------------
				xml_loader = st.session_state.get( 'xml_loader' )
				if xml_loader is None:
					st.info( 'No loader initialized.' )
				elif not hasattr( xml_loader, 'xml_root' ):
					st.info( 'XML loader does not support XML tree operations.' )
				elif xml_loader.xml_root is None:
					st.info( 'XML loader initialized but no XML tree loaded.' )
				else:
					st.markdown( '**XPath Query**' )
					xpath_expr = st.text_input( 'XPath Expression', value='//*',
						help='Use namespace prefixes if applicable.' )
					
					if st.button( 'Run XPath Query', use_container_width=True, icon='🏃', ):
						with st.spinner( 'Executing XPath...' ):
							elements = xml_loader.get_elements( xpath_expr )
						
						if elements is not None:
							st.session_state[ 'xml_xpath_results' ] = elements
							st.success( f'Returned {len( elements )} elements.' )
					
					if 'xml_xpath_results' in st.session_state and st.session_state[
						'xml_xpath_results' ] is not None:
						preview_count = min( 10, len( st.session_state[ 'xml_xpath_results' ] ) )
						st.caption( f'Previewing first {preview_count} elements' )
						
						for el in st.session_state[ 'xml_xpath_results' ][ :preview_count ]:
							st.code( etree.tostring( el, pretty_print=True,
								encoding='unicode' ), language='xml' )
				
				# ------------------------------------------------------------------
				# Debug / Introspection
				# ------------------------------------------------------------------
				with st.expander( "ℹ Loader State" ):
					xml_loader = st.session_state.get( 'xml_loader' )
					
					if xml_loader is None:
						st.info( "No loader initialized." )
					else:
						st.json( { "file_path": getattr( xml_loader, 'file_path', None ),
								"documents_loaded": getattr( xml_loader, 'documents', None ) is not None,
								"xml_tree_loaded": getattr( xml_loader, 'xml_tree', None ) is not None,
								"namespaces": getattr( xml_loader, 'xml_namespaces', None ),
								"chunk_size": getattr( xml_loader, 'chunk_size', None ),
								"overlap_amount": getattr( xml_loader, 'overlap_amount', None ), } )
			
			# ----------------------------
			# ------- Expander Word Loader
			# ----------------------------
			with st.expander( label='Word Document Loader', icon='📘', expanded=False ):
				word_file = st.file_uploader( 'Upload Word Document', type=[ 'docx' ],
					key='word_upload', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_word = col_load.button( 'Load', key='word_load', icon='📤', width='stretch' )
				clear_word = col_clear.button( 'Clear', key='word_clear', icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( 'active_loader' ) == 'WordLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='word_loader_output.txt', mime='text/plain',
						key='word_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='word_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_word:
					clear_if_active( 'WordLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Word Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_word and word_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, word_file.name )
						with open( path, 'wb' ) as f:
							f.write( word_file.read( ) )
						
						loader = WordLoader( )
						documents = loader.load( path ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'WordLoader'
						document.metadata.setdefault( 'source', word_file.name )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'WordLoader'
					st.session_state[ '_loader_status' ] = \
						f'Loaded {len( documents )} Word document(s).'
			
			# ----------------------------
			# ------ Expander PDF Loader
			# ----------------------------
			with st.expander( label='PDF Loader', icon='📕', expanded=False ):
				pdf = st.file_uploader( 'Upload PDF', type=[ 'pdf' ], key='pdf_upload' )
				mode = st.selectbox( 'Mode', [ 'single', 'page' ], key='pdf_mode',
					help='Used only when legacy extraction is enabled.' )
				
				extract = st.selectbox( 'Extract', [ 'plain', 'layout' ], key='pdf_extract',
					help='Used only when legacy extraction is enabled.' )
				
				include = st.checkbox( 'Include Images', value=False, key='pdf_include',
					help='Used only when legacy extraction is enabled.' )
				
				fmt = st.selectbox( 'Format', [ 'markdown-img', 'html-img',
						'text-img' ], key='pdf_fmt', help='Used only when legacy extraction is enabled.' )
				
				use_geometry = st.checkbox( 'Use Geometry Extraction', value=True,
					key='pdf_use_geometry', help='Uses PyMuPDF block coordinates' )
				
				use_legacy_pdf_loader = st.checkbox( 'Use Legacy PdfLoader', value=False,
					key='pdf_use_legacy_loader', help='Falls back to the existing PdfLoader' )
				
				band_left, band_right = st.columns( 2, border=True )
				with band_left:
					header_band = st.slider( 'Header Band', min_value=0, max_value=30,
						value=8, step=1, key='pdf_header_band',
						help='Percentage of page height classified as the top candidate header' )
				
				with band_right:
					footer_band = st.slider( 'Footer Band', min_value=0, max_value=30, value=8,
						step=1, key='pdf_footer_band',
						help='Percentage of page height classified as the bottom candidate footer.')
				
				preserve_page_breaks = st.checkbox( 'Preserve Page Breaks', value=False,
					key='pdf_preserve_page_breaks', help='Adds explicit page-break markers between extracted pages.' )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_pdf = col_load.button( 'Load', key='pdf_load', icon='📤', width='stretch' )
				clear_pdf = col_clear.button( 'Clear', key='pdf_clear', icon='🧹', width='stretch' )
				save_pdf = col_save.empty( )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_pdf:
					clear_if_active( 'PdfLoader' )
					st.session_state.pdf_pages = None
					st.session_state[ '_loader_status' ] = 'PDF Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_pdf and pdf:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, pdf.name )
						
						with open( path, 'wb' ) as f:
							f.write( pdf.read( ) )
						
						if use_geometry and not use_legacy_pdf_loader:
							parser = PdfParser( )
							pdf_pages = parser.extract_pages( path=path,
								header_ratio=float( header_band ) / 100.0,
								footer_ratio=float( footer_band ) / 100.0 ) or [ ]
							
							raw_text = parser.rebuild_pages( pages=pdf_pages,
								preserve_page_breaks=preserve_page_breaks ) or ''
							
							documents = [ Document( page_content=raw_text, metadata={
									'loader': 'PdfLoader', 'source': pdf.name,
									'extract': 'geometry', 'header_band': int( header_band ),
									'footer_band': int( footer_band ),
									'preserve_page_breaks': preserve_page_breaks, } ) ]
							
							st.session_state.pdf_pages = pdf_pages
						else:
							loader = PdfLoader( )
							documents = loader.load( path, mode=mode, extract=extract,
								include=include, format=fmt ) or [ ]
							
							for document in documents:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'PdfLoader'
								document.metadata.setdefault( 'source', pdf.name )
								document.metadata.setdefault( 'extract', 'legacy' )
							
							raw_text = '\n\n'.join( d.page_content for d in documents if
									hasattr( d, 'page_content' ) \
									and isinstance( d.page_content, str ) and d.page_content.strip())
							
							st.session_state.pdf_pages = None
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = raw_text
					st.session_state.processed_text = None
					st.session_state.displayed_text = ''
					st.session_state.processed_text_display = ''
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'PdfLoader'
					
					st.session_state[ '_loader_status' ] = \
						f'Loaded {len( documents )} PDF document(s).'
				
				# --------------------------------------------------
				# Save
				# --------------------------------------------------
				can_save = ( st.session_state.get( 'active_loader' ) == 'PdfLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					save_pdf.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='pdf_loader_output.txt', mime='text/plain',
						key='pdf_save', icon='💾', width='stretch' )
				else:
					save_pdf.button( 'Save', key='pdf_save_disabled', disabled=True, icon='💾', width='stretch' )
			
			# ----------------------------
			# --- Expander Power Point Loader
			# ----------------------------
			with st.expander( label='Power Point Loader', icon='📽', expanded=False ):
				pptx = st.file_uploader( 'Upload PPTX', type=[ 'pptx' ], key='pptx_upload', )
				mode = st.selectbox( 'Mode', [ 'single', 'elements' ], key='pptx_mode', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save (same row, same style)
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_pptx = col_load.button( 'Load', key='pptx_load', icon='📤', width='stretch' )
				clear_pptx = col_clear.button( 'Clear', key='pptx_clear', icon='🧹', width='stretch' )
				
				# ---------- Save
				can_save = (
							st.session_state.get( 'active_loader' ) == 'PowerPointLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='powerpoint_loader_output.txt', mime='text/plain',
						key='pptx_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='pptx_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# ---------- Clear
				if clear_pptx:
					clear_if_active( 'PowerPointLoader' )
					st.info( 'PowerPoint Loader state cleared.' )
				
				# ---------- Load
				if load_pptx and pptx:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, pptx.name )
						with open( path, "wb" ) as f:
							f.write( pptx.read( ) )
						
						loader = PowerPointLoader( )
						documents = loader.load( path, mode=mode ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.active_loader = "PowerPointLoader"
					st.success( f"Loaded {len( documents )} PowerPoint document(s)." )
			
			# ----------------------------
			# ------ Expander Jupyter Notebook Loader
			# ----------------------------
			with st.expander( label='Jupyter Notebook Loader', icon='📓', expanded=False ):
				notebook_file = st.file_uploader( 'Upload Notebook', type=[
						'ipynb' ], key='ipynb_upload', )
				
				include_outputs = st.checkbox( 'Include Outputs',
					value=False, key='ipynb_include_outputs', )
				
				max_output_length = st.number_input( 'Max Output Length', min_value=1,
					value=10, step=1, key='ipynb_max_output_length', )
				
				remove_newline = st.checkbox( 'Remove Newline',
					value=False, key='ipynb_remove_newline', )
				
				include_traceback = st.checkbox( 'Include Traceback',
					value=False, key='ipynb_traceback', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_ipynb = col_load.button( 'Load', key='ipynb_load', icon='📤', width='stretch' )
				clear_ipynb = col_clear.button( 'Clear', key='ipynb_clear', icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'JupyterNotebookLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='jupyter_notebook_loader_output.txt', mime='text/plain',
						key='ipynb_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='ipynb_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_ipynb:
					clear_if_active( 'JupyterNotebookLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Jupyter Notebook Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_ipynb and notebook_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, notebook_file.name )
						with open( path, 'wb' ) as f:
							f.write( notebook_file.read( ) )
						
						loader = JupyterNotebookLoader( )
						documents = loader.load( path=path, include_outputs=include_outputs,
							max_output_length=int( max_output_length ),
							remove_newline=remove_newline, traceback=include_traceback, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'JupyterNotebookLoader'
						document.metadata.setdefault( 'source', notebook_file.name )
						document.metadata.setdefault( 'include_outputs', include_outputs )
						document.metadata.setdefault( 'max_output_length', int( max_output_length ))
						document.metadata.setdefault( 'remove_newline', remove_newline )
						document.metadata.setdefault( 'traceback', include_traceback )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'JupyterNotebookLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} notebook document(s).'
			
			# ----------------------------
			# ------- Expander Excel Loader
			# ----------------------------
			with st.expander( label='Excel Loader', icon='📊', expanded=False ):
				excel_file = st.file_uploader( 'Upload Excel file', type=[ 'xlsx', 'xls' ],
					key='excel_upload', )
				
				load_mode = st.selectbox( 'Load Mode', ['Tabular + SQLite', 'Unstructured Document'],
					index=0, key='excel_load_mode', help=(
						'Use "Tabular + SQLite" to preserve the current sheet-to-SQLite workflow. '
						'Use "Unstructured Document" to route through ExcelLoader.'), )
				
				sheet_name = st.text_input(
					'Sheet name (leave blank for all sheets)', key='excel_sheet' )
				
				table_prefix = st.text_input( 'table prefix', value='excel',
					help='Each sheet will be written as <prefix>_<sheetname>',
					key='excel_table_prefix' )
				
				unstructured_mode = st.selectbox( 'Document Mode', [ 'single', 'elements' ],
					index=0, key='excel_unstructured_mode',
					help='Used only with "Unstructured Documents".' )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_excel = col_load.button( 'Load', key='excel_load', icon='📤', width='stretch')
				clear_excel = col_clear.button( 'Clear', key='excel_clear',
					icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( 'active_loader' ) == 'ExcelLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='excel_loader_output.txt', mime='text/plain',
						key='excel_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='excel_save_disabled', disabled=True, icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear (remove only ExcelLoader documents)
				# --------------------------------------------------
				if clear_excel and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( 'loader' ) != 'ExcelLoader' ]
					
					st.session_state.raw_documents = [ d for d in st.session_state.documents if
							isinstance( getattr(
								d, 'metadata', None ), dict ) ] if st.session_state.documents else [ ]
					
					st.session_state.raw_text = ('\n\n'.join(
						d.page_content for d in st.session_state.documents if
								isinstance( d.page_content, str ) \
								and d.page_content.strip( ) ) if st.session_state.documents else None)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = None
					st.info( "ExcelLoader documents removed." )
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_excel and excel_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						excel_path = os.path.join( tmp, excel_file.name )
						with open( excel_path, "wb" ) as f:
							f.write( excel_file.read( ) )
						
						documents = [ ]
						if load_mode == 'Tabular + SQLite':
							sqlite_path = os.path.join( "stores", "sqlite", "data.db" )
							os.makedirs( os.path.dirname( sqlite_path ), exist_ok=True )
							if sheet_name.strip( ):
								dfs = { sheet_name: pd.read_excel( excel_path,
									sheet_name=sheet_name, ) }
							else:
								dfs = pd.read_excel( excel_path, sheet_name=None, )
							
							conn = sqlite3.connect( sqlite_path )
							try:
								for sheet, df in dfs.items( ):
									if df.empty:
										continue
									table_name = f"{table_prefix}_{sheet}".replace( " ", "_" ).lower( )
									df.to_sql( table_name, conn, if_exists="replace", index=False, )
									text = df.to_csv( index=False )
									documents.append( Document( page_content=text, metadata={
											'loader': 'ExcelLoader', 'source': excel_file.name,
											'sheet': sheet, 'table': table_name,
											'sqlite_db': sqlite_path,
											'load_mode': 'Tabular + SQLite', }, ) )
							finally:
								conn.close( )
						
						else:
							loader = ExcelLoader( )
							documents = loader.load( excel_path, mode=unstructured_mode,
								has_headers=True ) or [ ]
							
							for document in documents:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'ExcelLoader'
								document.metadata.setdefault( 'source', excel_file.name )
								document.metadata[ 'load_mode' ] = 'Unstructured Document'
								document.metadata[ 'document_mode' ] = unstructured_mode
					
					if documents:
						existing_documents = st.session_state.get( 'documents' )
						if isinstance( existing_documents, list ) and existing_documents:
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = list( documents )
						
						st.session_state.raw_documents = list( st.session_state.documents )
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in st.session_state.documents if
									isinstance( d.page_content, str ) and d.page_content.strip( ) )
						
						st.session_state.processed_text = None
						st.session_state.active_loader = 'ExcelLoader'
						
						if load_mode == 'Tabular + SQLite':
							st.success(
								f"Loaded {len( documents )} sheet(s) and stored in SQLite." )
						else:
							st.success(
								f"Loaded {len( documents )} Excel {unstructured_mode!r} mode." )
					else:
						if load_mode == 'Tabular + SQLite':
							st.warning( "No data loaded (empty sheets or invalid selection)." )
						else:
							st.warning( "No Excel document content was loaded." )
			
			# ----------------------------
			# ------ Expander Markdown Loader
			# ----------------------------
			with st.expander( label='Markdown Loader', icon='🧾', expanded=False ):
				md = st.file_uploader( 'Upload Markdown', type=[ 'md',
						'markdown' ], key='md_upload', )
				
				mode = st.selectbox( 'Mode', [ 'single', 'elements' ], index=0, key='md_mode',
					help='Use "single" for one combined document or "elements" for multiple.' )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_md = col_load.button( 'Load', key='md_load', icon='📤', width='stretch' )
				clear_md = col_clear.button( 'Clear', key='md_clear', icon='🧹', width='stretch' )
				
				can_save = (
						st.session_state.get( 'active_loader' ) == 'MarkdownLoader' \
						and isinstance( st.session_state.get( 'raw_text' ), str ) \
						and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='markdown_loader_output.txt', mime='text/plain',
						key='md_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='md_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_md:
					clear_if_active( 'MarkdownLoader' )
					st.info( "Markdown Loader state cleared." )
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_md and md:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, md.name )
						with open( path, "wb" ) as f:
							f.write( md.read( ) )
						
						loader = MarkdownLoader( )
						documents = loader.load( path, mode=mode ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.active_loader = "MarkdownLoader"
					st.success( f"Loaded {len( documents )} Markdown document(s)." )
			
			# ----------------------------
			# ---- Expander HTML Loader
			# ----------------------------
			with st.expander( label='HTML Loader', icon='🌐', expanded=False ):
				html = st.file_uploader( 'Upload HTML', type=[ 'html', 'htm' ], key='html_upload' )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save (same row, same style)
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_html = col_load.button( 'Load', key='html_load', icon='📤', width='stretch' )
				clear_html = col_clear.button( 'Clear', key='html_clear', icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'HtmlLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='html_loader_output.txt', mime='text/plain', key='html_save',
						icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='html_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear (UNCHANGED behavior)
				# --------------------------------------------------
				if clear_html:
					clear_if_active( "HtmlLoader" )
					st.info( "HTML Loader state cleared." )
				
				# --------------------------------------------------
				# Load (UNCHANGED behavior)
				# --------------------------------------------------
				if load_html and html:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, html.name )
						with open( path, "wb" ) as f:
							f.write( html.read( ) )
						
						loader = HtmlLoader( )
						documents = loader.load( path )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
					st.session_state.active_loader = "HtmlLoader"
					st.success( f"Loaded {len( documents )} HTML document(s)." )
			
			# ----------------------------
			# --------- Expander JSON Loader
			# ----------------------------
			with st.expander( label='JSON Loader', icon='🧩', expanded=False ):
				js = st.file_uploader( 'Upload JSON', type=[ 'json', 'jsonl' ], key='json_upload', )
				
				jq_schema = st.text_input( 'jq Schema', value='.', key='json_jq_schema',
					help='Examples: ., .[], .messages[], .content' )
				
				content_key = st.text_input( 'Content Key (optional)', value='',
					key='json_content_key', help='Use when jq_schema returns objects page_content.' )
				
				is_lines = st.checkbox( 'JSON Lines', value=False, key='json_lines', )
				is_text = st.checkbox( 'Extracted content is already text', value=True,
					key='json_text_content',
					help='Turn this off when jq_schema/content_key selects structured values' )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_json = col_load.button( 'Load', key='json_load', icon='📤', width='stretch' )
				clear_json = col_clear.button( 'Clear', key='json_clear', icon='🧹', width='stretch' )
				
				can_save = (
							st.session_state.get( 'active_loader' ) == 'JsonLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='json_loader_output.txt', mime='text/plain', key='json_save',
						icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='json_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_json:
					clear_if_active( 'JsonLoader' )
					st.info( 'JSON Loader state cleared.' )
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_json and js:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, js.name )
						with open( path, 'wb' ) as f:
							f.write( js.read( ) )
						
						loader = JsonLoader( )
						documents = loader.load( path, jq_schema=jq_schema, content_key=content_key,
							is_text=is_text, is_lines=is_lines, ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.active_loader = "JsonLoader"
					st.success( f"Loaded {len( documents )} JSON document(s)." )
		
		with st.expander( label='Web Documents', expanded=False ):
			# ----------------------------
			# ------- Expander ArXiv Loader
			# ----------------------------
			with st.expander( label='ArXiv Loader', icon='🧠', expanded=False ):
				arxiv_query = st.text_input( 'Query',
					placeholder='e.g., transformer OR llm', key='arxiv_query', )
				
				arxiv_max_chars = st.number_input( 'Max characters per document', min_value=250,
					max_value=100000, value=1000, step=250,
					key='arxiv_max_chars', help='Maximum characters read', )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				arxiv_fetch = col_fetch.button( 'Load', key='arxiv_fetch', icon='📤', width='stretch' )
				arxiv_clear = col_clear.button( 'Clear', key='arxiv_clear', icon='🧹', width='stretch' )
				can_save = (
							st.session_state.get( 'active_loader' ) == 'ArXivLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='arxiv_loader_output.txt', mime='text/plain',
						key='arxiv_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='arxiv_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				if arxiv_clear and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( 'loader' ) != 'ArXivLoader' ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'ArXivLoader documents removed.'
				
				if arxiv_fetch and arxiv_query:
					loader = ArXivLoader( )
					documents = loader.load( arxiv_query, max_chars=int( arxiv_max_chars ), ) or [ ]
					
					for d in documents:
						d.metadata[ 'loader' ] = 'ArXivLoader'
						d.metadata[ 'source' ] = arxiv_query
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = 'ArXivLoader'
						
						st.session_state[
							'_loader_status' ] = f'Fetched {len( documents )} document(s).'
			
			# ----------------------------
			# ---- Expander Wikipedia Loader
			# ----------------------------
			with st.expander( label='Wikipedia Loader', icon='📚', expanded=False ):
				wiki_query = st.text_input( 'Query',
					placeholder='e.g., Natural language processing', key='wiki_query', )
				
				wiki_max_docs = st.number_input( 'Max documents', min_value=1, max_value=250,
					value=25, step=1, key='wiki_max_docs',
					help='Maximum number of documents loaded', )
				
				wiki_max_chars = st.number_input( 'Max characters per document', min_value=250,
					max_value=100000, value=4000, step=250, key='wiki_max_chars',
					help='Upper limit on the number of characters', )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				wiki_fetch = col_fetch.button( 'Load', key='wiki_fetch', icon='📤', width='stretch' )
				wiki_clear = col_clear.button( 'Clear', key='wiki_clear', icon='🧹', width='stretch' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'WikiLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				            and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='wiki_loader_output.txt', mime='text/plain',
						key='wiki_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='wiki_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				if wiki_clear and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( 'loader' ) != 'WikiLoader' ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'WikiLoader documents removed.'
				
				if wiki_fetch and wiki_query:
					loader = WikiLoader( )
					documents = loader.load( wiki_query, max_docs=int( wiki_max_docs ),
						max_chars=int( wiki_max_chars ), ) or [ ]
					
					for d in documents:
						d.metadata[ 'loader' ] = 'WikiLoader'
						d.metadata[ 'source' ] = wiki_query
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = 'WikiLoader'
						
						st.session_state[ '_loader_status' ] = (
								f'Fetched {len( documents )} Wikipedia document(s).')
			
			# ----------------------------
			# ----- Expander GitHub Loader
			# ----------------------------
			with st.expander( label='GitHub Loader', icon='🐙', expanded=False ):
				gh_url = st.text_input( "GitHub API URL", placeholder="https://api.github.com",
					value="https://api.github.com", key="gh_url", help="GitHub REST API base URL.", )
				
				gh_repo = st.text_input( "Repo (owner/name)", placeholder="openai/openai-python",
					key="gh_repo", help="Name of the repository.", )
				
				gh_branch = st.text_input( "Branch", placeholder="main", value="main",
					key="gh_branch", help="The branch of the repository.", )
				
				gh_filetype = st.text_input( "File type filter", value=".md",
					key="gh_filetype", help="Filtering by file type. Example: .py, .md, .txt", )
				
				gh_access_token = st.text_input( "GitHub Access Token (optional)", value="",
					type="password", key="gh_access_token", help="Optional personal access token.")
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				gh_fetch = col_fetch.button( "Load", key="gh_fetch", icon='📤', width='stretch' )
				gh_clear = col_clear.button( "Clear", key="gh_clear", icon='🧹', width='stretch' )
				
				can_save = (
						st.session_state.get( "active_loader" ) == "GithubLoader" \
						and isinstance( st.session_state.get( "raw_text" ), str ) \
						and st.session_state.get( "raw_text" ).strip( ) )
				
				if can_save:
					col_save.download_button( "Save", data=st.session_state.get( "raw_text" ),
						file_name="github_loader_output.txt", mime="text/plain",
						key="gh_save", icon='💾', width='stretch' )
				else:
					col_save.button( "Save", key="gh_save_disabled", disabled=True,
						icon='💾', width='stretch' )
				
				if gh_clear and st.session_state.get( "documents" ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( "loader" ) != "GithubLoader" ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "GithubLoader documents removed."
				
				if gh_fetch and gh_repo and gh_branch:
					loader = GithubLoader( )
					documents = loader.load( gh_url, gh_repo, gh_branch, gh_filetype,
						gh_access_token, ) or [ ]
					
					for d in documents:
						if not isinstance( getattr( d, "metadata", None ), dict ):
							d.metadata = { }
						d.metadata[ "loader" ] = "GithubLoader"
						d.metadata[ "source" ] = f"{gh_repo}@{gh_branch}"
					
					if documents:
						if st.session_state.get( "documents" ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = "GithubLoader"
						
						st.session_state[ "_loader_status" ] = \
							f"Fetched {len( documents )} GitHub document(s)."
			
			# ----------------------------
			# -------- Expander Outlook Loader
			# ----------------------------
			with st.expander( label='Outlook Loader', icon='📨', expanded=False ):
				outlook_file = st.file_uploader( 'Upload Outlook Message', type=[
						'msg' ], key='outlook_upload', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_outlook = col_load.button( 'Load', key='outlook_load',
					icon='📤', width='stretch' )
				
				clear_outlook = col_clear.button( 'Clear', key='outlook_clear',
					icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'OutlookLoader' \
						and isinstance( st.session_state.get( 'raw_text' ), str ) \
						and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='outlook_loader_output.txt', mime='text/plain',
						key='outlook_save', icon='📥', width='stretch' )
				else:
					col_save.button( 'Save', key='outlook_save_disabled',
						disabled=True, icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_outlook:
					clear_if_active( 'OutlookLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Outlook Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_outlook and outlook_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, outlook_file.name )
						with open( path, 'wb' ) as f:
							f.write( outlook_file.read( ) )
						
						loader = OutlookLoader( )
						documents = loader.load( path ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OutlookLoader'
						document.metadata.setdefault( 'source', outlook_file.name )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OutlookLoader'
					st.session_state[ '_loader_status' ] = (
							f'Loaded {len( documents )} Outlook message document('
							f's).')
			
			# ----------------------------
			# ------- Expander Web Loader
			# ----------------------------
			with st.expander( label="Web Loader", icon='🌐', expanded=False ):
				urls = st.text_area( "Enter one URL per line",
					placeholder="https://example.com\nhttps://another.com", key="web_urls", )
				
				web_timeout = st.number_input( "Timeout (seconds)", min_value=1, max_value=120, value=10, step=1, key="web_timeout", )
				
				web_ignore = st.checkbox( "Continue On Failure", value=True,
					key="web_ignore", help="Keep loading remaining URLs if one page fails." )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				load_web = col_fetch.button( "Load", key="web_fetch", icon='📤', width='stretch' )
				clear_web = col_clear.button( "Clear", key="web_clear", icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( "active_loader" ) == "WebLoader" \
							and isinstance( st.session_state.get( "raw_text" ), str ) \
							and st.session_state.get( "raw_text" ).strip( ))
				
				if can_save:
					col_save.download_button( "Save", data=st.session_state.get( "raw_text" ),
						file_name="web_loader_output.txt", mime="text/plain",
						key="web_save", icon='💾', width='stretch' )
				else:
					col_save.button( "Save", key="web_save_disabled", disabled=True,
						icon='💾', width='stretch' )
				
				if clear_web and st.session_state.get( "documents" ):
					st.session_state.documents = [ d for d in st.session_state.documents if
							d.metadata.get( "loader" ) != "WebLoader" ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "WebLoader documents removed."
				
				if load_web and urls.strip( ):
					loader = WebLoader( recursive=False )
					new_docs = [ ]
					for url in [ u.strip( ) for u in urls.splitlines( ) if u.strip( ) ]:
						documents = loader.load( urls=url, timeout=int( web_timeout ),
							ignore=bool( web_ignore ), progress=True ) or [ ]
						
						for d in documents:
							if not isinstance( getattr( d, "metadata", None ), dict ):
								d.metadata = { }
							d.metadata[ "loader" ] = "WebLoader"
							d.metadata[ "source" ] = url
						
						new_docs.extend( documents )
					
					if new_docs:
						if st.session_state.get( "documents" ):
							st.session_state.documents.extend( new_docs )
						else:
							st.session_state.documents = new_docs
							st.session_state.raw_documents = list( new_docs )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = "WebLoader"
						st.session_state[ "_loader_status" ] = \
							f"Fetched {len( new_docs )} web document(s)."
			
			# ----------------------------
			# ----- Expander Web Crawler
			# ----------------------------
			with st.expander( label='Web Crawler', icon='🕷️', expanded=False ):
				start_url = st.text_input( 'Start URL',
					placeholder='https://example.com', key='crawl_start_url', )
				
				max_depth = st.number_input( 'Max crawl depth', min_value=1, max_value=5,
					value=2, step=1, key='crawl_depth', )
				
				crawl_timeout = st.number_input( 'Timeout (seconds)', min_value=1, max_value=120,
					value=10, step=1, key='crawl_timeout', )
				
				stay_on_domain = st.checkbox( 'Stay on starting domain', value=True,
					key='crawl_domain_lock', )
				
				col_run, col_clear, col_save = st.columns( 3 )
				run_crawl = col_run.button( 'Load', key='crawl_run', icon='🏃', width='stretch' )
				clear_crawl = col_clear.button( 'Clear', key='crawl_clear',
					icon='🧹', width='stretch' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'WebCrawler' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='web_crawler_output.txt', mime='text/plain', key='crawl_save',
						icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='crawl_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				if clear_crawl:
					clear_if_active( 'WebCrawler' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Web Crawler state cleared.'
				
				if run_crawl and isinstance( start_url, str ) and start_url.strip( ):
					loader = WebCrawler( url=start_url.strip( ), recursive=True,
						max_depth=int( max_depth ), prevent_outside=bool( stay_on_domain ),
						timeout=int( crawl_timeout ), ignore=True, progress=True, )
					
					documents = loader.load( urls=start_url.strip( ), depth=int( max_depth ),
						timeout=int( crawl_timeout ), ignore=True, progress=True,
						prevent_outside=bool( stay_on_domain ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'WebCrawler'
						document.metadata.setdefault( 'source', start_url.strip( ) )
						document.metadata.setdefault( 'max_depth', int( max_depth ) )
						document.metadata.setdefault( 'timeout', int( crawl_timeout ) )
						document.metadata.setdefault( 'prevent_outside', bool( stay_on_domain ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str )  \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'WebCrawler'
					st.session_state[
						'_loader_status' ] = f'Crawled {len( documents )} document(s).'
			
			# ----------------------------
			# ----- Expander Email Loader
			# ----------------------------
			with st.expander( label='E-mail Loader', icon='📧', expanded=False ):
				email_file = st.file_uploader( 'Upload Email File',
					type=[ 'eml' ], key='email_upload', )
				
				email_mode = st.selectbox( 'Mode', options=[ 'elements',
						'single' ], index=0, key='email_mode', )
				
				email_attachments = st.checkbox( 'Process Attachments',
					value=False, key='email_attachments', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_email = col_load.button( 'Load', key='email_load', icon='📤', width='stretch' )
				clear_email = col_clear.button( 'Clear', key='email_clear', icon='🧹', width='stretch' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'EmailLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='email_loader_output.txt', mime='text/plain',
						key='email_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='email_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_email:
					clear_if_active( 'EmailLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Email Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_email and email_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, email_file.name )
						with open( path, 'wb' ) as f:
							f.write( email_file.read( ) )
						
						loader = EmailLoader( )
						documents = loader.load( path=path, mode=email_mode,
							attachments=email_attachments, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'EmailLoader'
						document.metadata.setdefault( 'source', email_file.name )
						document.metadata.setdefault( 'mode', email_mode )
						document.metadata.setdefault( 'attachments', email_attachments )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'EmailLoader'
					st.session_state[ '_loader_status' ] = \
						f'Loaded {len( documents )} email document(s).'
			
			# ----------------------------
			# ---- Expander PubMed Loader
			# ----------------------------
			with st.expander( label='Pub Med Loader', icon='🧬', expanded=False ):
				pubmed_query = st.text_input( 'PubMed Query', value='',
					key='pubmed_query', placeholder='e.g. transformer models biomedical NLP', )
				
				pubmed_max_docs = st.number_input( 'Maximum Documents', min_value=1,
					value=5, step=1, key='pubmed_max_docs', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_pubmed = col_load.button( 'Load', key='pubmed_load', icon='📥', width='stretch' )
				clear_pubmed = col_clear.button( 'Clear', key='pubmed_clear', icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'PubMedSearchLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='pubmed_loader_output.txt', mime='text/plain',
						key='pubmed_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='pubmed_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_pubmed:
					clear_if_active( 'PubMedSearchLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'PubMed Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if load_pubmed and isinstance( pubmed_query, str ) and pubmed_query.strip( ):
					loader = PubMedSearchLoader( )
					documents = loader.load( query=pubmed_query.strip( ),
						max_docs=int( pubmed_max_docs ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'PubMedSearchLoader'
						document.metadata.setdefault( 'query', pubmed_query.strip( ) )
						document.metadata.setdefault( 'max_docs', int( pubmed_max_docs ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'PubMedSearchLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} PubMed document(s).'
			
			# ----------------------------
			# --- Expander Open City Loader
			# ----------------------------
			with st.expander( label='Open City Loader', icon='🏙️', expanded=False ):
				open_city_id = st.text_input( 'City Domain', value='', key='open_city_id',
					placeholder='e.g. data.sfgov.org',
					help='City domain identifier for the Socrata-backed portal.',
					icon='💾', width='stretch' )
				
				open_city_dataset_id = st.text_input( 'Dataset ID', value='',
					key='open_city_dataset_id', placeholder='e.g. vw6y-z8j6',
					help='Dataset identifier from the city portal.', icon='💾', width='stretch' )
				
				open_city_limit = st.number_input( 'Maximum Records', min_value=1, value=100,
					step=1, key='open_city_limit', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_open_city = col_load.button( 'Load', key='open_city_load',
					icon='📥', width='stretch' )
				clear_open_city = col_clear.button( 'Clear', key='open_city_clear',
					icon='🧹', width='stretch' )
				
				can_save = (
						st.session_state.get( 'active_loader' ) == 'OpenCityLoader' \
						and isinstance( st.session_state.get( 'raw_text' ), str ) \
						and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='open_city_loader_output.txt', mime='text/plain',
						key='open_city_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='open_city_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_open_city:
					clear_if_active( 'OpenCityLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Open City Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if (load_open_city and isinstance( open_city_id, str ) \
						and open_city_id.strip( ) and isinstance( open_city_dataset_id, str ) \
						and open_city_dataset_id.strip( )):
					loader = OpenCityLoader( )
					documents = loader.load( city_id=open_city_id.strip( ),
						dataset_id=open_city_dataset_id.strip( ),
						limit=int( open_city_limit ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OpenCityLoader'
						document.metadata.setdefault( 'city_id', open_city_id.strip( ) )
						document.metadata.setdefault( 'dataset_id', open_city_dataset_id.strip( ) )
						document.metadata.setdefault( 'limit', int( open_city_limit ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OpenCityLoader'
					st.session_state[ '_loader_status' ] = \
						f'Loaded {len( documents )} Open City document(s).'
		
		with st.expander( label='Cloud Documents', expanded=False ):
			# ----------------------------
			# ---- Expander OneDrive Loader
			# ----------------------------
			with st.expander( label='OneDrive Loader', icon='🟦', expanded=False ):
				onedrive_drive_id = st.text_input( 'Drive ID', value='', key='onedrive_drive_id',
					placeholder='OneDrive drive identifier', )
				
				onedrive_folder_path = st.text_input( 'Folder Path', value='',
					key='onedrive_folder_path', placeholder='Optional folder path within the drive',
					help='Leave blank to load the drive target directly.', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_onedrive = col_load.button( 'Load', key='onedrive_load', icon='📤', width='stretch' )
				clear_onedrive = col_clear.button( 'Clear', key='onedrive_clear', icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'OneDriveDocLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='onedrive_loader_output.txt', mime='text/plain',
						key='onedrive_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='onedrive_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_onedrive:
					clear_if_active( 'OneDriveDocLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'OneDrive Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if ( load_onedrive and isinstance( onedrive_drive_id, str ) \
						and onedrive_drive_id.strip( )):
					loader = OneDriveDocLoader( )
					if isinstance( onedrive_folder_path, str ) and onedrive_folder_path.strip( ):
						documents = loader.load_folder( id=onedrive_drive_id.strip( ),
							path=onedrive_folder_path.strip( ), ) or [ ]
					else:
						documents = loader.load( id=onedrive_drive_id.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OneDriveDocLoader'
						document.metadata.setdefault( 'drive_id', onedrive_drive_id.strip( ) )
						document.metadata.setdefault( 'folder_path',
							onedrive_folder_path.strip( ) or None, )
						
						if onedrive_folder_path.strip( ):
							document.metadata.setdefault( 'source',
								f"{onedrive_drive_id.strip( )}:{onedrive_folder_path.strip( )}" )
						else:
							document.metadata.setdefault( 'source', onedrive_drive_id.strip( ), )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OneDriveDocLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} OneDrive document(s).'
			
			# ----------------------------
			# ---- Expander Google Cloud File Loader
			# ----------------------------
			with st.expander( label='Google Cloud File Loader', icon='☁️', expanded=False ):
				gcs_project_name = st.text_input( 'Project Name', value='',
					key='gcs_file_project_name', placeholder='e.g. my-gcp-project', )
				
				gcs_bucket = st.text_input( 'Bucket', value='',
					key='gcs_file_bucket', placeholder='e.g. my-bucket', )
				
				gcs_blob = st.text_input( 'Blob', value='',
					key='gcs_file_blob', placeholder='e.g. documents/report.pdf', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_gcs_file = col_load.button( 'Load', key='gcs_file_load', icon='📤', width='stretch' )
				clear_gcs_file = col_clear.button( 'Clear', key='gcs_file_clear', icon='🧹', width='stretch' )
				can_save = (
							st.session_state.get( 'active_loader' ) == 'GoogleCloudFileLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
							and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='google_cloud_file_loader_output.txt', mime='text/plain',
						key='gcs_file_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='gcs_file_save_disabled',
						disabled=True, icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_gcs_file:
					clear_if_active( 'GoogleCloudFileLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = ('Google Cloud File Loader state '
					                                        'cleared.')
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if ( load_gcs_file and isinstance( gcs_project_name, str ) \
						and gcs_project_name.strip( ) and isinstance( gcs_bucket, str ) \
						and gcs_bucket.strip( ) and isinstance( gcs_blob, str ) and gcs_blob.strip( )):
					loader = GoogleCloudFileLoader( )
					documents = loader.load( project_name=gcs_project_name.strip( ),
						bucket=gcs_bucket.strip( ), blob=gcs_blob.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'GoogleCloudFileLoader'
						document.metadata.setdefault( 'project_name', gcs_project_name.strip( ) )
						document.metadata.setdefault( 'bucket', gcs_bucket.strip( ) )
						document.metadata.setdefault( 'blob', gcs_blob.strip( ) )
						document.metadata.setdefault( 'source',
							f"gs://{gcs_bucket.strip( )}/{gcs_blob.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'GoogleCloudFileLoader'
					st.session_state[ '_loader_status' ] = (
							f'Loaded {len( documents )} Google Cloud file '
							f'document(s).')
			
			# ----------------------------
			# ---- Expander AWS File Loader
			# ----------------------------
			with st.expander( label='AWS File Loader', icon='🪣', expanded=False ):
				aws_file_bucket = st.text_input( 'Bucket', value='', key='aws_file_bucket',
					placeholder='e.g. my-s3-bucket', )
				
				aws_file_key = st.text_input( 'Object Key', value='', key='aws_file_key',
					placeholder='e.g. documents/report.pdf', )
				
				aws_file_region = st.text_input( 'Region Name', value='',
					key='aws_file_region', placeholder='e.g. us-east-1', )
				
				aws_file_api_version = st.text_input( 'API Version', value='',
					key='aws_file_api_version', placeholder='Optional', )
				
				aws_file_use_ssl = st.checkbox( 'Use SSL', value=True, key='aws_file_use_ssl', )
				aws_file_verify = st.text_input( 'Verify', value='', key='aws_file_verify',
					placeholder='Optional path or True / False',
					help='Leave blank for default behavior, or provide a CA bundle path.', )
				
				aws_file_endpoint_url = st.text_input( 'Endpoint URL', value='',
					key='aws_file_endpoint_url', placeholder='Optional custom endpoint', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_aws_file = col_load.button( label='Load', key='aws_file_load',
					icon='📤', width='stretch' )
				clear_aws_file = col_clear.button( label='Clear', key='aws_file_clear',
					icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( 'active_loader' ) == 'AwsFileLoader' \
				             and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='aws_file_loader_output.txt', mime='text/plain',
						key='aws_file_save', icon='📥', width='stretch' )
				else:
					col_save.button( 'Save', key='aws_file_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_aws_file:
					clear_if_active( 'AwsFileLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'AWS File Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if (load_aws_file and isinstance( aws_file_bucket, str ) \
						and aws_file_bucket.strip( ) and isinstance( aws_file_key, str ) \
						and aws_file_key.strip( )):
					verify_value: str | bool | None = None
					if isinstance( aws_file_verify, str ) and aws_file_verify.strip( ):
						verify_text = aws_file_verify.strip( )
						if verify_text.lower( ) == 'true':
							verify_value = True
						elif verify_text.lower( ) == 'false':
							verify_value = False
						else:
							verify_value = verify_text
					
					loader = AwsFileLoader( )
					documents = loader.load( bucket=aws_file_bucket.strip( ),
						key=aws_file_key.strip( ), region_name=aws_file_region.strip( ) or None,
						api_version=aws_file_api_version.strip( ) or None,
						use_ssl=bool( aws_file_use_ssl ), verify=verify_value,
						endpoint_url=aws_file_endpoint_url.strip( ) or None, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'AwsFileLoader'
						document.metadata.setdefault( 'bucket', aws_file_bucket.strip( ) )
						document.metadata.setdefault( 'key', aws_file_key.strip( ) )
						document.metadata.setdefault( 'region_name',
							aws_file_region.strip( ) or None )
						document.metadata.setdefault( 'api_version',
							aws_file_api_version.strip( ) or None )
						document.metadata.setdefault( 'use_ssl', bool( aws_file_use_ssl ) )
						document.metadata.setdefault( 'verify', verify_value )
						document.metadata.setdefault( 'endpoint_url',
							aws_file_endpoint_url.strip( ) or None )
						document.metadata.setdefault( 'source',
							f"s3://{aws_file_bucket.strip( )}/{aws_file_key.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'AwsFileLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} AWS file document(s).'
			
			# ----------------------------
			# ----- Expander Google Bucket Loader
			# ----------------------------
			with st.expander( label='Google Bucket Loader', icon='🗂️', expanded=False ):
				gcs_bucket_project_name = st.text_input( 'Project Name', value='',
					key='gcs_bucket_project_name', placeholder='e.g. my-gcp-project', )
				
				gcs_bucket_name = st.text_input( 'Bucket', value='', key='gcs_bucket_name',
					placeholder='e.g. my-bucket', )
				
				gcs_bucket_prefix = st.text_input( 'Prefix', value='', key='gcs_bucket_prefix',
					placeholder='Optional folder / object prefix', )
				
				gcs_bucket_continue_on_failure = st.checkbox( 'Continue On Failure',
					value=False, key='gcs_bucket_continue_on_failure', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_gcs_bucket = col_load.button( 'Load', key='gcs_bucket_load',
					icon='📥', width='stretch' )
				clear_gcs_bucket = col_clear.button( 'Clear', key='gcs_bucket_clear',
					icon='🧹', width='stretch' )
				can_save = ( st.session_state.get( 'active_loader' ) == 'GoogleBucketLoader' and \
				             isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='google_bucket_loader_output.txt', mime='text/plain',
						key='gcs_bucket_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='gcs_bucket_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_gcs_bucket:
					clear_if_active( 'GoogleBucketLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Google Bucket Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if (
						load_gcs_bucket and isinstance( gcs_bucket_project_name, str ) \
						and gcs_bucket_project_name.strip( ) and isinstance( gcs_bucket_name, str )\
						and gcs_bucket_name.strip( )):
					loader = GoogleBucketLoader( )
					documents = loader.load( project_name=gcs_bucket_project_name.strip( ),
						bucket=gcs_bucket_name.strip( ), prefix=gcs_bucket_prefix.strip( ) or None,
						continue_on_failure=bool( gcs_bucket_continue_on_failure ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'GoogleBucketLoader'
						document.metadata.setdefault( 'project_name',
							gcs_bucket_project_name.strip( ), )
						
						document.metadata.setdefault( 'bucket', gcs_bucket_name.strip( ), )
						document.metadata.setdefault( 'prefix', gcs_bucket_prefix.strip( ) or None,)
						document.metadata.setdefault( 'continue_on_failure',
							bool( gcs_bucket_continue_on_failure ), )
						
						if gcs_bucket_prefix.strip( ):
							document.metadata.setdefault( 'source',
								f"gs://{gcs_bucket_name.strip( )}/{gcs_bucket_prefix.strip( )}" )
						else:
							document.metadata.setdefault( 'source',
								f"gs://{gcs_bucket_name.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) \
							and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'GoogleBucketLoader'
					st.session_state[ '_loader_status' ] = (
							f'Loaded {len( documents )} Google bucket document(s).')
			
			# ----------------------------
			# ---- Expander AWS Bucket Loader
			# ----------------------------
			with st.expander( label='AWS Bucket Loader', icon='🗃️', expanded=False ):
				aws_bucket_name = st.text_input( 'Bucket', value='',
					key='aws_bucket_name', placeholder='e.g. my-s3-bucket', )
				
				aws_bucket_prefix = st.text_input( 'Prefix', value='',
					key='aws_bucket_prefix', placeholder='Optional folder / object prefix', )
				
				aws_bucket_region = st.text_input( 'Region Name', value='',
					key='aws_bucket_region', placeholder='e.g. us-east-1', )
				
				aws_bucket_api_version = st.text_input( 'API Version', value='',
					key='aws_bucket_api_version', placeholder='Optional', )
				
				aws_bucket_use_ssl = st.checkbox( 'Use SSL', value=True, key='aws_bucket_use_ssl', )
				
				aws_bucket_verify = st.text_input( 'Verify', value='', key='aws_bucket_verify',
					placeholder='Optional path or True / False',
					help='Leave blank for default behavior, or provide a CA bundle path.', )
				
				aws_bucket_endpoint_url = st.text_input( 'Endpoint URL', value='',
					key='aws_bucket_endpoint_url', placeholder='Optional custom endpoint', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_aws_bucket = col_load.button( 'Load', key='aws_bucket_load',
					icon='📤', width='stretch' )
				
				clear_aws_bucket = col_clear.button( 'Clear', key='aws_bucket_clear',
					icon='🧹', width='stretch' )
				
				can_save = st.session_state.get( 'active_loader' ) == 'AwsBucketLoader' \
							and isinstance( st.session_state.get( 'raw_text' ), str ) \
				            and st.session_state.get( 'raw_text' ).strip( )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='aws_bucket_loader_output.txt', mime='text/plain',
						key='aws_bucket_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='aws_bucket_save_disabled',
						disabled=True, icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_aws_bucket:
					clear_if_active( 'AwsBucketLoader' )
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'AWS Bucket Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if ( load_aws_bucket and isinstance( aws_bucket_name, str ) \
						and aws_bucket_name.strip( )):
					verify_value: str = None
					
					if isinstance( aws_bucket_verify, str ) and aws_bucket_verify.strip( ):
						verify_text = aws_bucket_verify.strip( )
						if verify_text.lower( ) == 'true':
							verify_value = True
						elif verify_text.lower( ) == 'false':
							verify_value = False
						else:
							verify_value = verify_text
					
					loader = AwsBucketLoader( )
					documents = loader.load( bucket=aws_bucket_name.strip( ),
						prefix=aws_bucket_prefix.strip( ),
						region_name=aws_bucket_region.strip( ) or None,
						api_version=aws_bucket_api_version.strip( ) or None,
						use_ssl=bool( aws_bucket_use_ssl ), verify=verify_value,
						endpoint_url=aws_bucket_endpoint_url.strip( ) or None, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'AwsBucketLoader'
						document.metadata.setdefault( 'bucket', aws_bucket_name.strip( ) )
						document.metadata.setdefault( 'prefix', aws_bucket_prefix.strip( ) )
						document.metadata.setdefault( 'region_name', aws_bucket_region.strip( ) or None )
						document.metadata.setdefault( 'api_version', aws_bucket_api_version.strip( ) or None )
						document.metadata.setdefault( 'use_ssl', bool( aws_bucket_use_ssl ) )
						document.metadata.setdefault( 'verify', verify_value )
						document.metadata.setdefault( 'endpoint_url', aws_bucket_endpoint_url.strip( ) or None )
						
						if aws_bucket_prefix.strip( ):
							document.metadata.setdefault( 'source',
								f"s3://{aws_bucket_name.strip( )}/{aws_bucket_prefix.strip( )}" )
						else:
							document.metadata.setdefault( 'source',
								f"s3://{aws_bucket_name.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'AwsBucketLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} AWS bucket document(s).'
			
			# ---------------------------
			# ---- Expander SharePoint Loader
			# ---------------------------
			with st.expander( label='SharePoint Loader', icon='🟩', expanded=False ):
				spfx_library_id = st.text_input( 'Library ID', value='',
					key='spfx_library_id', placeholder='SharePoint document library identifier', )
				
				spfx_folder_id = st.text_input( 'Folder ID', value='', key='spfx_folder_id',
					placeholder='Optional folder identifier within the library',
					help='Leave blank to load the library directly.', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_spfx = col_load.button( 'Load', key='spfx_load', icon='📤', width='stretch' )
				clear_spfx = col_clear.button( 'Clear', key='spfx_clear', icon='🧹', width='stretch' )
				
				can_save = ( st.session_state.get( 'active_loader' ) == 'SpfxLoader' \
				             and isinstance( st.session_state.get( 'raw_text' ), str ) \
				             and st.session_state.get( 'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='sharepoint_loader_output.txt', mime='text/plain',
						key='spfx_save', icon='💾', width='stretch' )
				else:
					col_save.button( 'Save', key='spfx_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				
				# --------------------------------------------------
				# Clear
				# --------------------------------------------------
				if clear_spfx:
					clear_if_active( 'SpfxLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'SharePoint Loader state cleared.'
				
				# --------------------------------------------------
				# Load
				# --------------------------------------------------
				if (load_spfx and isinstance( spfx_library_id, str ) and spfx_library_id.strip( )):
					loader = SpfxLoader( )
					if isinstance( spfx_folder_id, str ) and spfx_folder_id.strip( ):
						documents = loader.load_folder( library_id=spfx_library_id.strip( ), folder_id=spfx_folder_id.strip( ), ) or [ ]
					else:
						documents = loader.load( library_id=spfx_library_id.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'SpfxLoader'
						document.metadata.setdefault( 'library_id', spfx_library_id.strip( ) )
						document.metadata.setdefault( 'folder_id', spfx_folder_id.strip( ) or None, )
						
						if spfx_folder_id.strip( ):
							document.metadata.setdefault( 'source', f"{spfx_library_id.strip( )}:{spfx_folder_id.strip( )}" )
						else:
							document.metadata.setdefault( 'source', spfx_library_id.strip( ), )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
							hasattr( d, 'page_content' ) and isinstance( d.page_content, str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'SpfxLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} SharePoint document(s).'
	
	# ------------------------------------------------------------------
	# RIGHT COLUMN — DOCUMENT RENDERING
	# ------------------------------------------------------------------
	with right:
		documents = st.session_state.documents
		if not documents:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f'Active Loader: {st.session_state.active_loader}' )
			st.write( f'Documents: {len( documents )}' )
			for i, d in enumerate( documents[ :5 ] ):
				with st.expander( f'Document {i + 1}', expanded=True ):
					st.json( d.metadata )
					st.text_area( 'Content', d.page_content[
						: ], height=450, key=f'preview_doc_{i}' )
	
	# ------------------------------------------------------------------
	# NLP METRIC CALCULATIONS
	# ------------------------------------------------------------------
	metrics_container = st.container( )
	with metrics_container:
		if documents is not None:
			# -------- Tokenization (session-cached)
			if st.session_state.tokens is None:
				try:
					raw_text = rebuild_raw_text_from_documents( )
					tokens = [ t.lower( ) for t in word_tokenize( raw_text ) if t.isalpha( ) ]
				except LookupError:
					st.error( 'NLTK resources missing' )
				
				if not tokens:
					st.warning( 'No valid alphabetic tokens found.' )
				
				st.session_state.tokens = tokens
				st.session_state.vocabulary = set( tokens )
				st.session_state.token_counts = Counter( tokens )
			
			tokens = st.session_state.tokens
			vocabulary = st.session_state.vocabulary
			counts = st.session_state.token_counts
			char_count = len( raw_text )
			token_count = len( tokens )
			vocab_size = len( vocabulary )
			hapax_count = sum( 1 for c in counts.values( ) if c == 1 )
			hapax_ratio = hapax_count / vocab_size if vocab_size else 0.0
			avg_word_len = sum( len( t ) for t in tokens ) / token_count
			ttr = vocab_size / token_count
			stopword_ratio = 0.0
			lexical_density = 0.0
			try:
				stop_words = set( stopwords.words( 'english' ) )
				stopword_ratio = sum( 1 for t in tokens if t in stop_words ) / token_count
				lexical_density = 1.0 - stopword_ratio
			except LookupError:
				pass
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			st.markdown( '#### Top Tokens' )
			
			# ------------ Top Tokens
			with st.expander( label='Tokens', icon='🉑', expanded=True ):
				top_tokens = counts.most_common( 10 )
				df_top = pd.DataFrame( top_tokens, columns=[ 'token',
						'count' ] ).set_index( 'token' )
				st.bar_chart( df_top, color='#01438A' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			st.markdown( '#### Corpus Metrics' )
			
			# ------------ Corpus Metrics
			with st.expander( label='Text', icon='📖', expanded=True ):
				col1, col2, col3, col4 = st.columns( 4, border=True )
				with col1:
					metric_with_tooltip( 'Characters', f'{char_count:,}', 'Total number of characters in the selected text.' )
				
				with col2:
					metric_with_tooltip( 'Tokens', f'{token_count:,}', 'Token Count: total number of tokenized words after cleanup.' )
				
				with col3:
					metric_with_tooltip( 'Unique Tokens', f'{vocab_size:,}', 'Vocabulary Size: number of distinct word types in the text.' )
				
				with col4:
					metric_with_tooltip( 'TTR', f'{ttr:.3f}', 'Type–Token Ratio: unique words ÷ total words.' )
				
				col5, col6, col7, col8 = st.columns( 4, border=True )
				with col5:
					metric_with_tooltip( 'Hapax Ratio', f'{hapax_ratio:.3f}', 'Hapax Ratio: proportion of words that occur only once.' )
				
				with col6:
					metric_with_tooltip( 'Avg Length', f'{avg_word_len:.2f}', 'Average number of characters per token.' )
				
				with col7:
					metric_with_tooltip( 'Stopword Ratio', f'{stopword_ratio:.2%}', 'Percentage of stopwords in the text.' )
				
				with col8:
					metric_with_tooltip( 'Lexical Density', f'{lexical_density:.2%}', 'Proportion of content-bearing words.' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			st.markdown( '#### Comprehension Metrics' )
			
			# ------------ Readability
			with st.expander( label='Words', icon='👀', expanded=True ):
				if TEXTSTAT_AVAILABLE:
					r1, r2, r3, r4 = st.columns( 4, border=True )
					with r1:
						metric_with_tooltip( 'Flesch Reading Ease', f'{textstat.flesch_reading_ease( raw_text ):.1f}', 'Higher scores indicate easier readability.' )
					
					with r2:
						metric_with_tooltip( 'Flesch–Kincaid Grade', f'{textstat.flesch_kincaid_grade( raw_text ):.1f}', 'Estimated U.S. grade level required.' )
					
					with r3:
						metric_with_tooltip( 'Gunning Fog', f'{textstat.gunning_fog( raw_text ):.1f}', 'Readability based on sentence length and complex words.' )
					
					with r4:
						metric_with_tooltip( 'Coleman–Liau Index', f'{textstat.coleman_liau_index( raw_text ):.1f}', 'Readability based on characters and sentences.' )
				else:
					st.caption( 'Install `textstat` to enable readability metrics.' )

# =============================================================================
# SCRAPING MODE
# ==============================================================================
elif mode == 'Scraping':
	st.subheader( f'🕷️ Web Scraping' )
	st.divider( )
	if 'webscrape_clear_request' not in st.session_state:
		st.session_state[ 'webscrape_clear_request' ] = False
	
	if st.session_state.get( 'webscrape_clear_request', False ):
		st.session_state[ 'webfetcher_url' ] = ''
		st.session_state[ 'webscrape_results' ] = [ ]
		st.session_state[ 'webscrape_summary' ] = { }
		st.session_state[ 'webscrape_clear_request' ] = False
	
	def _clear_webscrape_state( ) -> None:
		st.session_state[ 'webscrape_clear_request' ] = True
	
	def _coerce_items( value: Any ) -> list[ str ]:
		if value is None:
			return [ ]
		if isinstance( value, list ):
			return [ str( item ) for item in value if item is not None ]
		return [ str( value ) ]
	
	def _extract_title_from_html( html: str ) -> str:
		try:
			if not isinstance( html, str ) or not html.strip( ):
				return ''
			
			match = re.search( r'<title[^>]*>(.*?)</title>', html, flags=re.IGNORECASE | re.DOTALL )
			
			if not match:
				return ''
			
			title = re.sub( r'\s+', ' ', match.group( 1 ) ).strip( )
			return html_lib.unescape( title )
		except Exception:
			return ''
	
	def _truncate_text( text: str, limit: int = 12000 ) -> str:
		if not isinstance( text, str ):
			return ''
		if len( text ) <= limit:
			return text
		return text[ : limit ] + '\n\n... [truncated]'
	
	def _normalize_url( base_url: str, href: str ) -> str:
		try:
			if not href or not isinstance( href, str ):
				return ''
			
			href = href.strip( )
			if not href:
				return ''
			
			absolute = urljoin( base_url, href )
			parsed = urlparse( absolute )
			if parsed.scheme not in ('http', 'https'):
				return ''
			
			normalized = parsed._replace( fragment='' )
			return normalized.geturl( )
		except Exception:
			return ''
	
	def _same_domain( left: str, right: str ) -> bool:
		try:
			left_host = (urlparse( left ).netloc or '').lower( )
			right_host = (urlparse( right ).netloc or '').lower( )
			return bool( left_host ) and left_host == right_host
		except Exception:
			return False
	
	def _extract_links_from_html( base_url: str, html: str ) -> list[ str ]:
		try:
			if not isinstance( html, str ) or not html.strip( ):
				return [ ]
			
			soup = BeautifulSoup( html, 'html.parser' )
			results: list[ str ] = [ ]
			seen: set[ str ] = set( )
			for tag in soup.find_all( 'a', href=True ):
				candidate = _normalize_url( base_url, tag.get( 'href', '' ) )
				if candidate and candidate not in seen:
					seen.add( candidate )
					results.append( candidate )
			
			return results
		except Exception:
			return [ ]
	
	def _scrape_single_page( url: str, include_title: bool, include_basic_text: bool, include_raw_html: bool, selected_methods:
	list[ str ] ) -> dict[ str, Any ]:
		page_result: dict[ str, Any ] = { 'url': url, 'status_code': None, 'encoding': None,
				'title': '', 'plain_text': '', 'raw_html': '', 'links_discovered': [ ], 'data': { },
				'errors': [ ], }
		
		fetcher = WebFetcher( )
		
		try:
			response = fetcher.fetch( url )
			if response is None:
				page_result[ 'errors' ].append( 'No response returned.' )
				return page_result
			
			page_result[ 'status_code' ] = getattr( response, 'status_code', None )
			page_result[ 'encoding' ] = getattr( response, 'encoding', None )
			raw_html = getattr( response, 'text', '' ) or ''
			page_result[ 'links_discovered' ] = _extract_links_from_html( url, raw_html )
			if include_title:
				page_result[ 'title' ] = _extract_title_from_html( raw_html )
			
			if include_basic_text:
				try:
					page_result[ 'plain_text' ] = fetcher.html_to_text( raw_html ) or ''
				except Exception as exc:
					page_result[ 'errors' ].append( f'Basic Text: {str( exc )}' )
			
			if include_raw_html:
				page_result[ 'raw_html' ] = raw_html
		
		except Exception as exc:
			page_result[ 'errors' ].append( f'Fetch: {str( exc )}' )
			return page_result
		
		REGISTRY: dict[ str, tuple[ str, Callable ] ] = {
				'scrape_headings': ('Headings', fetcher.scrape_headings),
				'scrape_paragraphs': ('Paragraphs', fetcher.scrape_paragraphs),
				'scrape_lists': ('Lists', fetcher.scrape_lists),
				'scrape_tables': ('Tables', fetcher.scrape_tables),
				'scrape_articles': ('Articles', fetcher.scrape_articles),
				'scrape_sections': ('Sections', fetcher.scrape_sections),
				'scrape_divisions': ('Divisions', fetcher.scrape_divisions),
				'scrape_blockquotes': ('Blockquotes', fetcher.scrape_blockquotes),
				'scrape_hyperlinks': ('Hyperlinks', fetcher.scrape_hyperlinks),
				'scrape_images': ('Images', fetcher.scrape_images), }
		
		for method_name in selected_methods:
			if method_name not in REGISTRY:
				continue
			
			label, method = REGISTRY[ method_name ]
			try:
				data = method( url )
				page_result[ 'data' ][ label ] = _coerce_items( data )
			except Exception as exc:
				page_result[ 'data' ][ label ] = [ ]
				page_result[ 'errors' ].append( f'{label}: {str( exc )}' )
		
		return page_result
	
	def _crawl_pages( seed_url: str, include_title: bool, include_basic_text: bool, include_raw_html: bool, selected_methods:
	list[ str ], recursive: bool, max_depth: int, max_pages: int, same_domain_only: bool ) -> tuple[
		list[ dict[ str, Any ] ], dict[ str, Any ] ]:
		results: list[ dict[ str, Any ] ] = [ ]
		visited: set[ str ] = set( )
		enqueued: set[ str ] = set( )
		queue: deque[ tuple[ str, int ] ] = deque( )
		skipped_urls: list[ str ] = [ ]
		
		normalized_seed = _normalize_url( seed_url, seed_url )
		if not normalized_seed:
			raise ValueError( 'A valid absolute URL is required.' )
		
		queue.append( (normalized_seed, 0) )
		enqueued.add( normalized_seed )
		
		while queue and len( results ) < max_pages:
			current_url, depth = queue.popleft( )
			
			if current_url in visited:
				continue
			
			visited.add( current_url )
			
			page_result = _scrape_single_page( url=current_url, include_title=include_title, include_basic_text=include_basic_text, include_raw_html=include_raw_html, selected_methods=selected_methods )
			
			page_result[ 'depth' ] = depth
			results.append( page_result )
			
			if not recursive:
				continue
			
			if depth >= max_depth:
				continue
			
			discovered_links = page_result.get( 'links_discovered', [ ] ) or [ ]
			for next_url in discovered_links:
				if len( results ) + len( queue ) >= max_pages:
					break
				
				if not next_url or next_url in visited or next_url in enqueued:
					continue
				
				if same_domain_only and not _same_domain( normalized_seed, next_url ):
					skipped_urls.append( next_url )
					continue
				
				queue.append( (next_url, depth + 1) )
				enqueued.add( next_url )
		
		summary: dict[ str, Any ] = { 'mode': 'recursive' if recursive else 'single-page',
				'seed_url': normalized_seed, 'pages_processed': len( results ),
				'pages_visited': len( visited ), 'pages_skipped': len( skipped_urls ),
				'recursive_requested': bool( recursive ), 'max_depth': int( max_depth ),
				'max_pages': int( max_pages ), 'same_domain_only': bool( same_domain_only ),
				'visited_urls': list( visited ), 'skipped_urls': skipped_urls, }
		
		return results, summary
	
	col_left, col_right = st.columns( [ 0.35, 0.65 ], border=True, gap='xxsmall' )
	with col_left:
		target_url = st.text_input( 'Enter Target URL', placeholder='https://example.com', key='webfetcher_url' )
		
		st.markdown( '##### Core Output' )
		include_title = st.checkbox( 'Page Title', value=True, key='wf_page_title' )
		include_basic_text = st.checkbox( 'Basic Text', value=True, key='wf_basic_text' )
		include_raw_html = st.checkbox( 'Raw HTML', value=False, key='wf_raw_html' )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.markdown( '##### Structured Extraction' )
		col1, col2 = st.columns( [ 0.5, 0.5 ] )
		
		REGISTRY_LABELS: dict[ str, str ] = { 'scrape_headings': 'Headings',
				'scrape_paragraphs': 'Paragraphs', 'scrape_lists': 'Lists',
				'scrape_tables': 'Tables', 'scrape_articles': 'Articles',
				'scrape_sections': 'Sections', 'scrape_divisions': 'Divisions',
				'scrape_blockquotes': 'Blockquotes', 'scrape_hyperlinks': 'Hyperlinks',
				'scrape_images': 'Images', }
		
		selected_methods: list[ str ] = [ ]
		_registry_items: list[ tuple[ str, str ] ] = list( REGISTRY_LABELS.items( ) )
		_col1_items: list[ tuple[ str, str ] ] = _registry_items[ :5 ]
		_col2_items: list[ tuple[ str, str ] ] = _registry_items[ 5: ]
		
		with col1:
			for method_name, label in _col1_items:
				if st.checkbox( label, key=f'wf_{method_name}' ):
					selected_methods.append( method_name )
		
		with col2:
			for method_name, label in _col2_items:
				if st.checkbox( label, key=f'wf_{method_name}' ):
					selected_methods.append( method_name )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		st.markdown( '##### Crawl Controls' )
		enable_recursive = st.checkbox( 'Recursive Crawl', value=False, key='wf_recursive' )
		max_depth = st.number_input( 'Max Depth', min_value=0, max_value=10, value=1, step=1, key='wf_max_depth', disabled=(
			not enable_recursive) )
		
		max_pages = st.number_input( 'Max Pages', min_value=1, max_value=500, value=10, step=1, key='wf_max_pages' )
		
		same_domain_only = st.checkbox( 'Same Domain Only', value=True, key='wf_same_domain_only', disabled=(
			not enable_recursive) )
		
		b1, b2 = st.columns( 2 )
		with b1:
			run_scraper = st.button( 'Run Scraper', key='webfetcher_run' )
		
		with b2:
			st.button( 'Clear', key='webfetcher_clear', on_click=_clear_webscrape_state )
	
	with col_right:
		if run_scraper:
			try:
				if not target_url or not target_url.strip( ):
					raise ValueError( 'A target URL is required.' )
				
				results, summary = _crawl_pages( seed_url=target_url.strip( ), include_title=include_title, include_basic_text=include_basic_text, include_raw_html=include_raw_html, selected_methods=selected_methods, recursive=bool( enable_recursive ), max_depth=int( max_depth ), max_pages=int( max_pages ), same_domain_only=bool( same_domain_only ) )
				
				st.session_state[ 'webscrape_results' ] = results
				st.session_state[ 'webscrape_summary' ] = summary
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
		
		summary = st.session_state.get( 'webscrape_summary', { } )
		results = st.session_state.get( 'webscrape_results', [ ] )
		if summary:
			st.subheader( 'Summary' )
			st.json( summary )
		
		if not results:
			st.info( 'No results.' )
		else:
			st.subheader( 'Results' )
			for idx, page in enumerate( results, start=1 ):
				title = page.get( 'title', '' ) or page.get( 'url', f'Page {idx}' )
				depth = page.get( 'depth', 0 )
				with st.expander( f'Page {idx} [Depth {depth}]: {title}', expanded=(idx == 1) ):
					meta_col1, meta_col2 = st.columns( 2 )
					with meta_col1:
						st.markdown( f"**URL:** {page.get( 'url', '' )}" )
						st.markdown( f"**Status Code:** {page.get( 'status_code', '' )}" )
						st.markdown( f"**Depth:** {page.get( 'depth', 0 )}" )
					
					with meta_col2:
						st.markdown( f"**Encoding:** {page.get( 'encoding', '' )}" )
						st.markdown( f"**Title:** {page.get( 'title', '' )}" )
					
					plain_text = page.get( 'plain_text', '' )
					if isinstance( plain_text, str ) and plain_text.strip( ):
						st.subheader( 'Basic Text' )
						st.text_area( label='', value=_truncate_text( plain_text, limit=12000 ), height=280, key=f'webscrape_plain_text_{idx}' )
					
					raw_html = page.get( 'raw_html', '' )
					if isinstance( raw_html, str ) and raw_html.strip( ):
						st.subheader( 'Raw HTML' )
						st.text_area( label='', value=_truncate_text( raw_html, limit=12000 ), height=240, key=f'webscrape_raw_html_{idx}' )
					
					discovered_links = page.get( 'links_discovered', [ ] ) or [ ]
					if discovered_links:
						st.subheader( 'Links Discovered' )
						for link_idx, link in enumerate( discovered_links, start=1 ):
							st.write( f'{link_idx}. {link}' )
					
					data = page.get( 'data', { } ) or { }
					for label, items in data.items( ):
						st.subheader( f'{label}' )
						if not items:
							st.info( 'No results returned.' )
							continue
						
						for item_idx, item in enumerate( items, start=1 ):
							st.write( f'{item_idx}. {item}' )
					
					errors = page.get( 'errors', [ ] ) or [ ]
					if errors:
						st.subheader( 'Errors' )
						for err in errors:
							st.error( err )

# ==============================================================================
# FETCHING MODE
# =============================================================================
elif mode == 'Retrieval':
	st.session_state.setdefault( 'retrieval_active_source', '' )
	st.session_state.setdefault( 'arxiv_input', '' )
	st.session_state.setdefault( 'arxiv_results', [ ] )
	st.subheader( '🏛️ Public Collections & Archives' )
	st.divider( )
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	with left:
		# ----------------------------
		# -------- Expander (ArXiv)
		# ----------------------------
		with st.expander( label='ArXiv', icon='📘', expanded=False ):
			st.badge( label='Information', help=cfg.ARXIV )
			if 'arxiv_results' not in st.session_state:
				st.session_state[ 'arxiv_results' ] = [ ]
			
			if 'arxiv_clear_request' not in st.session_state:
				st.session_state[ 'arxiv_clear_request' ] = False
			
			if st.session_state.get( 'arxiv_clear_request', False ):
				st.session_state[ 'arxiv_input' ] = ''
				st.session_state[ 'arxiv_results' ] = [ ]
				st.session_state[ 'arxiv_max_docs' ] = 5
				st.session_state[ 'arxiv_full_documents' ] = False
				st.session_state[ 'arxiv_include_metadata' ] = False
				st.session_state[ 'arxiv_clear_request' ] = False
			
			def _clear_arxiv_state( ) -> None:
				st.session_state[ 'arxiv_clear_request' ] = True
			
			arxiv_input = st.text_area( 'Query', height=80, key='arxiv_input', placeholder=(
				'Examples:\n'
				'What is the ImageBind model?\n'
				'2401.01234\n'
				'graph neural networks for molecular property prediction'), )
			
			c1, c2 = st.columns( 2, vertical_alignment='center' )
			with c1:
				arxiv_max_docs = st.number_input( 'Max Docs', min_value=1, max_value=300,
					value=st.session_state.get( 'arxiv_max_docs', 5 ), step=1,
					key='arxiv_max_docs', help='Maximum number of ArXiv documents to retrieve.' )
			
			with c2:
				arxiv_full_documents = st.checkbox( 'Full Documents',
					value=st.session_state.get( 'arxiv_full_documents', False ),
					key='arxiv_full_documents', help='When checked, retrieves full document text.' )
			
			arxiv_include_metadata = st.checkbox( 'Include All Metadata',
				value=st.session_state.get( 'arxiv_include_metadata', False ),
				key='arxiv_include_metadata',
				help='Include additional metadata fields when available.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				do_submit = st.button( 'Submit', key='arxiv_submit', width='stretch', icon='✔️', )
			
			with b2:
				st.button( 'Clear', key='arxiv_clear', on_click=_clear_arxiv_state,
					width='stretch', icon='🧹', )
			
			if do_submit:
				try:
					queries = [ q.strip( ) for q in (arxiv_input or '').splitlines( ) if
							q.strip( ) ]
					
					if not queries:
						st.warning( 'No input provided.' )
					else:
						from fetchers import ArXiv
						
						fetcher = ArXiv( max_documents=int( arxiv_max_docs ),
							full_documents=bool( arxiv_full_documents ),
							include_metadata=bool( arxiv_include_metadata ) )
						
						results: list[ Document ] = [ ]
						for q in queries:
							docs = fetcher.fetch( q, max_documents=int( arxiv_max_docs ),
								full_documents=bool( arxiv_full_documents ),
								include_metadata=bool( arxiv_include_metadata ) )
							
							if isinstance( docs, list ):
								results.extend( docs )
						
						st.session_state[ 'arxiv_results' ] = results
						st.session_state[ 'retrieval_active_source' ] = 'ArXiv'
						st.rerun( )
				
				except Exception as exc:
					st.error( 'ArXiv request failed.' )
					st.exception( exc )
		
		# ----------------------------
		# -------- Expander (Google Drive)
		# ----------------------------
		with st.expander( label='Google Drive', icon='🛡️', expanded=False ):
			st.badge( label='Information', help=cfg.GOOGLE_DRIVE )
			if 'googledrive_results' not in st.session_state:
				st.session_state[ 'googledrive_results' ] = [ ]
			
			if 'googledrive_clear_request' not in st.session_state:
				st.session_state[ 'googledrive_clear_request' ] = False
			
			if st.session_state.get( 'googledrive_clear_request', False ):
				st.session_state[ 'googledrive_query' ] = ''
				st.session_state[ 'googledrive_folder_id' ] = cfg.GOOGLE_DRIVE_FOLDER_ID or 'root'
				st.session_state[ 'googledrive_results_limit' ] = 10
				st.session_state[ 'googledrive_template' ] = 'gdrive-query'
				st.session_state[ 'googledrive_mode' ] = 'documents'
				st.session_state[ 'googledrive_mime_type' ] = ''
				st.session_state[ 'googledrive_results' ] = [ ]
				st.session_state[ 'googledrive_clear_request' ] = False
			
			def _clear_googledrive_state( ) -> None:
				st.session_state[ 'googledrive_clear_request' ] = True
			
			gd_query = st.text_area( 'Google Drive Query', height=90, key='googledrive_query', placeholder=(
				'Examples:\n'
				'machine learning\n'
				'budget execution\n'
				'FY 2026 operating plan\n'
				'\n'
				'Use "*" only when the selected template supports folder-wide or '
				'mime-type retrieval.'), )
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				gd_folder_id = st.text_input( 'Folder ID', value=st.session_state.get( 'googledrive_folder_id', cfg.GOOGLE_DRIVE_FOLDER_ID or 'root' ), key='googledrive_folder_id', placeholder='root or a Google Drive folder id', help='Use "root" for your My Drive root, or provide a specific folder '
				                                                                                                                                                                                                                          'id.' )
			
			with c2:
				gd_results_limit = st.number_input( 'Max Docs', min_value=1, max_value=100, value=int( st.session_state.get( 'googledrive_results_limit', 10 ) ), step=1, key='googledrive_results_limit', )
			
			c3, c4 = st.columns( 2 )
			with c3:
				gd_template = st.selectbox( 'Template', options=[ 'gdrive-all-in-folder',
						'gdrive-query', 'gdrive-by-name', 'gdrive-query-in-folder',
						'gdrive-mime-type', 'gdrive-mime-type-in-folder',
						'gdrive-query-with-mime-type',
						'gdrive-query-with-mime-type-and-folder', ], index=1, key='googledrive_template', help='Select the Drive retrieval strategy.' )
			
			with c4:
				gd_mode = st.selectbox( 'Mode', options=[ 'documents',
						'snippets' ], index=0, key='googledrive_mode', help='Use snippets for short metadata-driven returns.' )
			
			gd_mime_type = st.selectbox( 'MIME Type Filter', options=[ '', 'text/text',
					'text/plain', 'text/html', 'text/csv', 'text/markdown', 'image/png',
					'image/jpeg', 'application/epub+zip', 'application/pdf', 'application/rtf',
					'application/vnd.google-apps.document',
					'application/vnd.google-apps.presentation',
					'application/vnd.google-apps.spreadsheet',
					'application/vnd.google.colaboratory',
					'application/vnd.openxmlformats-officedocument.presentationml'
					'.presentation',
					'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
					'', ], index=0, key='googledrive_mime_type', help='Optional MIME type restriction.' )
			
			st.caption( 'Expected auth: GOOGLE_ACCOUNT_FILE for credentials JSON. '
			            'Optional: GOOGLE_DRIVE_TOKEN_PATH for token persistence.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				gd_submit = st.button( 'Submit', key='googledrive_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='googledrive_clear', on_click=_clear_googledrive_state, width='stretch' )
			
			if gd_submit:
				try:
					from fetchers import GoogleDrive
					
					fetcher = GoogleDrive( )
					docs = fetcher.fetch( question=gd_query, folder_id=gd_folder_id or 'root', results=int( gd_results_limit ), template=gd_template, mime_type=gd_mime_type or None, mode=gd_mode, )
					
					st.session_state[ 'googledrive_results' ] = docs or [ ]
					st.session_state[ 'retrieval_active_source' ] = 'Google Drive'
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Drive request failed.' )
					st.exception( exc )
		
		# ----------------------------
		# -------- Expander (Wikipedia)
		# ----------------------------
		with st.expander( label='Wikipedia', icon='📖', expanded=False ):
			st.badge( label='Information', help=cfg.WIKIPEDIA )
			if 'wikipedia_results' not in st.session_state:
				st.session_state[ 'wikipedia_results' ] = [ ]
			
			if 'wikipedia_clear_request' not in st.session_state:
				st.session_state[ 'wikipedia_clear_request' ] = False
			
			if st.session_state.get( 'wikipedia_clear_request', False ):
				st.session_state[ 'wikipedia_query' ] = ''
				st.session_state[ 'wikipedia_language' ] = 'en'
				st.session_state[ 'wikipedia_max_docs' ] = 5
				st.session_state[ 'wikipedia_include_metadata' ] = False
				st.session_state[ 'wikipedia_results' ] = [ ]
				st.session_state[ 'wikipedia_clear_request' ] = False
			
			def _clear_wikipedia_state( ) -> None:
				st.session_state[ 'wikipedia_clear_request' ] = True
			
			wiki_query = st.text_area( 'Wikipedia Query', height=90, key='wikipedia_query', placeholder=(
				'Examples:\n'
				'Alan Turing\n'
				'History of machine learning\n'
				'Python (programming language)\n'
				'Battle of Midway'), )
			
			c1, c2 = st.columns( 2 )
			with c1:
				wiki_language = st.text_input( 'Language Code', value=st.session_state.get( 'wikipedia_language', 'en' ), key='wikipedia_language', placeholder='en', help='Wikipedia language code, e.g. en, fr, de, ja.' )
			
			with c2:
				wiki_max_docs = st.number_input( 'Max Docs', min_value=1, max_value=300, value=int( st.session_state.get( 'wikipedia_max_docs', 5 ) ), step=1, key='wikipedia_max_docs', help='Maximum number of Wikipedia documents to retrieve.' )
			
			wiki_include_metadata = st.checkbox( 'Include All Metadata', value=st.session_state.get( 'wikipedia_include_metadata', False ), key='wikipedia_include_metadata', help='Include additional metadata fields when available.' )
			
			st.caption( 'No API key is required for Wikipedia retrieval. '
			            'Optional only: LANGSMITH_API_KEY for tracing.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				wiki_submit = st.button( 'Submit', key='wikipedia_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='wikipedia_clear', on_click=_clear_wikipedia_state, width='stretch' )
			
			if wiki_submit:
				try:
					queries = [ q.strip( ) for q in (wiki_query or '').splitlines( ) if q.strip( ) ]
					
					if not queries:
						st.warning( 'No input provided.' )
					else:
						from fetchers import Wikipedia
						
						fetcher = Wikipedia( language=wiki_language or 'en', max_documents=int( wiki_max_docs ), include_metadata=bool( wiki_include_metadata ) )
						
						results: list[ Document ] = [ ]
						
						for q in queries:
							docs = fetcher.fetch( q, language=wiki_language or 'en', max_documents=int( wiki_max_docs ), include_metadata=bool( wiki_include_metadata ) )
							
							if isinstance( docs, list ):
								results.extend( docs )
						
						st.session_state[ 'wikipedia_results' ] = results
						st.session_state[ 'retrieval_active_source' ] = 'Wikipedia'
						st.rerun( )
				
				except Exception as exc:
					st.error( 'Wikipedia request failed.' )
					st.exception( exc )
			
			with st.expander( label='Information', expanded=False ):
				st.help( Wikipedia )
		
		# ----------------------------
		# -------- Expander (Google Search)
		# ----------------------------
		with st.expander( label='Google Search', icon='🔍', expanded=False ):
			st.badge( label='Information', help=cfg.GOOGLE_CSE )
			if 'googlesearch_results' not in st.session_state:
				st.session_state[ 'googlesearch_results' ] = { }
			
			if 'googlesearch_clear_request' not in st.session_state:
				st.session_state[ 'googlesearch_clear_request' ] = False
			
			if st.session_state.get( 'googlesearch_clear_request', False ):
				st.session_state[ 'googlesearch_query' ] = ''
				st.session_state[ 'googlesearch_num_results' ] = 10
				st.session_state[ 'googlesearch_start' ] = 1
				st.session_state[ 'googlesearch_exact_terms' ] = ''
				st.session_state[ 'googlesearch_exclude_terms' ] = ''
				st.session_state[ 'googlesearch_file_type' ] = ''
				st.session_state[ 'googlesearch_date_restrict' ] = ''
				st.session_state[ 'googlesearch_gl' ] = ''
				st.session_state[ 'googlesearch_lr' ] = ''
				st.session_state[ 'googlesearch_safe' ] = 'off'
				st.session_state[ 'googlesearch_search_type' ] = ''
				st.session_state[ 'googlesearch_site_search' ] = ''
				st.session_state[ 'googlesearch_site_search_filter' ] = ''
				st.session_state[ 'googlesearch_sort' ] = ''
				st.session_state[ 'googlesearch_img_size' ] = ''
				st.session_state[ 'googlesearch_img_type' ] = ''
				st.session_state[ 'googlesearch_img_color_type' ] = ''
				st.session_state[ 'googlesearch_img_dominant_color' ] = ''
				st.session_state[ 'googlesearch_api_key' ] = ''
				st.session_state[ 'googlesearch_cse_id' ] = ''
				st.session_state[ 'googlesearch_timeout' ] = 10
				st.session_state[ 'googlesearch_results' ] = { }
				st.session_state[ 'googlesearch_clear_request' ] = False
			
			def _clear_googlesearch_state( ) -> None:
				st.session_state[ 'googlesearch_clear_request' ] = True
			
			google_query = st.text_area( 'Query', height=90, key='googlesearch_query', placeholder=(
				'Examples:\n'
				'site:epa.gov budget execution\n'
				'OpenAI GPT-5 reasoning\n'
				'filetype:pdf appropriations law'), )
			
			c1, c2, c3 = st.columns( 3 )
			with c1:
				google_num_results = st.number_input( 'Results / Request', min_value=1, max_value=10, value=int( st.session_state.get( 'googlesearch_num_results', 10 ) ), step=1, key='googlesearch_num_results', help='Google Custom Search returns up to 10 results per request.' )
			
			with c2:
				google_start = st.number_input( 'Start Index', min_value=1, max_value=91, value=int( st.session_state.get( 'googlesearch_start', 1 ) ), step=1, key='googlesearch_start' )
			
			with c3:
				google_timeout = st.number_input( 'Timeout', min_value=1, max_value=60, value=int( st.session_state.get( 'googlesearch_timeout', 10 ) ), step=1, key='googlesearch_timeout' )
			
			c4, c5 = st.columns( 2 )
			with c4:
				google_exact_terms = st.text_input( 'Exact Terms', value=st.session_state.get( 'googlesearch_exact_terms', '' ), key='googlesearch_exact_terms' )
			
			with c5:
				google_exclude_terms = st.text_input( 'Exclude Terms', value=st.session_state.get( 'googlesearch_exclude_terms', '' ), key='googlesearch_exclude_terms' )
			
			c6, c7, c8 = st.columns( 3 )
			with c6:
				google_file_type = st.text_input( 'File Type', value=st.session_state.get( 'googlesearch_file_type', '' ), key='googlesearch_file_type', placeholder='pdf' )
			
			with c7:
				google_date_restrict = st.text_input( 'Date Restrict', value=st.session_state.get( 'googlesearch_date_restrict', '' ), key='googlesearch_date_restrict', placeholder='d7, m1, y1' )
			
			with c8:
				google_safe = st.selectbox( 'Safe Search', options=[ 'off', 'active' ], index=[
						'off',
						'active' ].index( st.session_state.get( 'googlesearch_safe', 'off' ) ), key='googlesearch_safe' )
			
			c9, c10, c11 = st.columns( 3 )
			with c9:
				google_gl = st.text_input( 'Country (gl)', value=st.session_state.get( 'googlesearch_gl', '' ), key='googlesearch_gl', placeholder='us' )
			
			with c10:
				google_lr = st.text_input( 'Language Restrict (lr)', value=st.session_state.get( 'googlesearch_lr', '' ), key='googlesearch_lr', placeholder='lang_en' )
			
			with c11:
				google_search_type = st.selectbox( 'Search Type', options=[ '', 'image' ], index=[
						'',
						'image' ].index( st.session_state.get( 'googlesearch_search_type', '' ) ), key='googlesearch_search_type' )
			
			c12, c13 = st.columns( 2 )
			with c12:
				google_site_search = st.text_input( 'Site Search', value=st.session_state.get( 'googlesearch_site_search', '' ), key='googlesearch_site_search', placeholder='example.gov' )
			
			with c13:
				google_site_search_filter = st.selectbox( 'Site Search Filter', options=[ '', 'i',
						'e' ], index=[ '', 'i',
						'e' ].index( st.session_state.get( 'googlesearch_site_search_filter', '' ) ), key='googlesearch_site_search_filter', help='i=include, e=exclude' )
			
			google_sort = st.text_input( 'Sort', value=st.session_state.get( 'googlesearch_sort', '' ), key='googlesearch_sort', placeholder='date' )
			
			c14, c15 = st.columns( 2 )
			with c14:
				google_img_size = st.selectbox( 'Image Size', options=[ '', 'icon', 'small',
						'medium', 'large', 'xlarge', 'xxlarge', 'huge' ], index=[ '', 'icon',
						'small', 'medium', 'large', 'xlarge', 'xxlarge',
						'huge' ].index( st.session_state.get( 'googlesearch_img_size', '' ) ), key='googlesearch_img_size', disabled=(
							google_search_type != 'image') )
			
			with c15:
				google_img_type = st.selectbox( 'Image Type', options=[ '', 'clipart', 'face',
						'lineart', 'stock', 'photo', 'animated' ], index=[ '', 'clipart', 'face',
						'lineart', 'stock', 'photo',
						'animated' ].index( st.session_state.get( 'googlesearch_img_type', '' ) ), key='googlesearch_img_type', disabled=(
							google_search_type != 'image') )
			
			c16, c17 = st.columns( 2 )
			with c16:
				google_img_color_type = st.selectbox( 'Image Color Type', options=[ '', 'color',
						'gray', 'mono', 'trans' ], index=[ '', 'color', 'gray', 'mono',
						'trans' ].index( st.session_state.get( 'googlesearch_img_color_type', '' ) ), key='googlesearch_img_color_type', disabled=(
							google_search_type != 'image') )
			
			with c17:
				google_img_dominant_color = st.selectbox( 'Image Dominant Color', options=[ '',
						'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
						'red', 'teal', 'white', 'yellow' ], index=[ '', 'black', 'blue', 'brown',
						'gray', 'green', 'orange', 'pink', 'purple', 'red', 'teal', 'white',
						'yellow' ].index( st.session_state.get( 'googlesearch_img_dominant_color', '' ) ), key='googlesearch_img_dominant_color', disabled=(
							google_search_type != 'image') )
			
			c18, c19 = st.columns( 2 )
			with c18:
				google_api_key = st.text_input( 'API Key', value='', type='password', key='googlesearch_api_key', placeholder='Uses GOOGLE_API_KEY when left blank.' )
			
			with c19:
				google_cse_id = st.text_input( 'CSE ID', value='', key='googlesearch_cse_id', placeholder='Uses GOOGLE_CSE_ID when left blank.' )
			
			st.caption( 'Required keys: GOOGLE_API_KEY and GOOGLE_CSE_ID. '
			            'Endpoint updated to customsearch.googleapis.com/customsearch/v1.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				google_submit = st.button( 'Submit', key='googlesearch_submit', width='stretch' )
			with b2:
				st.button( 'Clear', key='googlesearch_clear', on_click=_clear_googlesearch_state, width='stretch' )
		
		# ----------------------------
		# -------- Expander (Open Science)
		# ----------------------------
		with st.expander( label='Open Science', icon='🧪', expanded=False ):
			st.badge( label='Information', help=cfg.NASA_OPEN_SCIENCE )
			if 'openscience_results' not in st.session_state:
				st.session_state[ 'openscience_results' ] = { }
			
			if 'openscience_clear_request' not in st.session_state:
				st.session_state[ 'openscience_clear_request' ] = False
			
			if st.session_state.get( 'openscience_clear_request', False ):
				st.session_state[ 'openscience_mode' ] = 'dataset'
				st.session_state[ 'openscience_accession' ] = ''
				st.session_state[ 'openscience_query' ] = ''
				st.session_state[ 'openscience_format' ] = 'json'
				st.session_state[ 'openscience_timeout' ] = 20
				st.session_state[ 'openscience_results' ] = { }
				st.session_state[ 'openscience_clear_request' ] = False
			
			def _clear_openscience_state( ) -> None:
				st.session_state[ 'openscience_clear_request' ] = True
			
			openscience_mode = st.selectbox( 'Mode', options=[ 'dataset', 'metadata', 'assays',
					'data' ], index=[ 'dataset', 'metadata', 'assays',
					'data' ].index( st.session_state.get( 'openscience_mode', 'dataset' ) ), key='openscience_mode', help=(
				'dataset = fetch dataset metadata by accession; '
				'metadata/assays/data = query the corresponding '
				'OSDR API endpoint.') )
			
			openscience_accession = st.text_input( 'Dataset Accession', value=st.session_state.get( 'openscience_accession', '' ), key='openscience_accession', placeholder='Example: OSD-48' )
			
			openscience_query = st.text_area( 'Query', value=st.session_state.get( 'openscience_query', '' ), height=120, key='openscience_query', placeholder=(
				'Example: (id.accession=OSD-48) '
				'OR study.characteristics.organism=Mus '
				'musculus') )
			
			openscience_format = st.selectbox( 'Format', options=[ 'json', 'csv', 'tsv',
					'browser' ], index=[ 'json', 'csv', 'tsv',
					'browser' ].index( st.session_state.get( 'openscience_format', 'json' ) ), key='openscience_format' )
			
			openscience_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'openscience_timeout', 20 ) ), step=1, key='openscience_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				openscience_submit = st.button( 'Submit', key='openscience_submit', width='stretch' )
			with b2:
				st.button( 'Clear', key='openscience_clear', on_click=_clear_openscience_state, width='stretch' )
		
		# ----------------------------
		# -------- Expander (Gov Info)
		# ----------------------------
		with st.expander( label='Gov Info', icon='🏛️', expanded=False ):
			st.badge( label='Information', help=cfg.GOV_INFO )
			if 'govinfo_results' not in st.session_state:
				st.session_state[ 'govinfo_results' ] = { }
			
			if 'govinfo_clear_request' not in st.session_state:
				st.session_state[ 'govinfo_clear_request' ] = False
			
			if st.session_state.get( 'govinfo_clear_request', False ):
				st.session_state[ 'govinfo_mode' ] = 'search'
				st.session_state[ 'govinfo_query' ] = ''
				st.session_state[ 'govinfo_page_size' ] = 10
				st.session_state[ 'govinfo_offset_mark' ] = '*'
				st.session_state[ 'govinfo_sort_field' ] = 'score'
				st.session_state[ 'govinfo_sort_order' ] = 'DESC'
				st.session_state[ 'govinfo_package_id' ] = ''
				st.session_state[ 'govinfo_collection' ] = ''
				st.session_state[ 'govinfo_start_date' ] = '2025-01-01T00:00:00Z'
				st.session_state[ 'govinfo_timeout' ] = 20
				st.session_state[ 'govinfo_results' ] = { }
				st.session_state[ 'govinfo_clear_request' ] = False
			
			def _clear_govinfo_state( ) -> None:
				st.session_state[ 'govinfo_clear_request' ] = True
			
			govinfo_mode = st.selectbox( 'Mode', options=[ 'search', 'package_summary',
					'collection' ], index=[ 'search', 'package_summary',
					'collection' ].index( st.session_state.get( 'govinfo_mode', 'search' ) ), key='govinfo_mode', help=(
				'search = GovInfo Search Service; '
				'package_summary = package details by package ID; '
				'collection = browse a collection since an ISO timestamp.') )
			
			govinfo_query = st.text_area( 'Query', value=st.session_state.get( 'govinfo_query', '' ), height=120, key='govinfo_query', placeholder=(
				'Example: collection:BILLS AND congress:118 '
				'AND title:"appropriations"') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				govinfo_page_size = st.number_input( 'Page Size', min_value=1, max_value=1000, value=int( st.session_state.get( 'govinfo_page_size', 10 ) ), step=1, key='govinfo_page_size' )
			
			with c2:
				govinfo_offset_mark = st.text_input( 'Offset Mark', value=st.session_state.get( 'govinfo_offset_mark', '*' ), key='govinfo_offset_mark', placeholder='*' )
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				govinfo_sort_field = st.selectbox( 'Sort Field', options=[ 'score',
						'lastModified' ], index=[ 'score',
						'lastModified' ].index( st.session_state.get( 'govinfo_sort_field', 'score' ) ), key='govinfo_sort_field' )
			
			with c4:
				govinfo_sort_order = st.selectbox( 'Sort Order', options=[ 'DESC', 'ASC' ], index=[
						'DESC',
						'ASC' ].index( st.session_state.get( 'govinfo_sort_order', 'DESC' ) ), key='govinfo_sort_order' )
			
			govinfo_package_id = st.text_input( 'Package ID', value=st.session_state.get( 'govinfo_package_id', '' ), key='govinfo_package_id', placeholder='Example: CREC-2018-10-10' )
			
			c5, c6 = st.columns( 2 )
			with c5:
				govinfo_collection = st.text_input( 'Collection', value=st.session_state.get( 'govinfo_collection', '' ), key='govinfo_collection', placeholder='Example: CREC' )
			
			with c6:
				govinfo_start_date = st.text_input( 'Start Date (ISO)', value=st.session_state.get( 'govinfo_start_date', '2025-01-01T00:00:00Z' ), key='govinfo_start_date', placeholder='YYYY-MM-DDTHH:MM:SSZ' )
			
			govinfo_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'govinfo_timeout', 20 ) ), step=1, key='govinfo_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				govinfo_submit = st.button( 'Submit', key='govinfo_submit', use_container_width=True, width='stretch' )
			
			with b2:
				govinfo_clear = st.button( 'Clear', key='govinfo_clear', on_click=_clear_govinfo_state, use_container_width=True, width='stretch' )
			
			result = st.session_state.get( 'govinfo_results', { } )
			
			if govinfo_submit:
				try:
					f = GovData( )
					
					result = f.fetch( mode=str( govinfo_mode ), query=str( govinfo_query ), page_size=int( govinfo_page_size ), offset_mark=str( govinfo_offset_mark ), sort_field=str( govinfo_sort_field ), sort_order=str( govinfo_sort_order ), package_id=str( govinfo_package_id ), collection=str( govinfo_collection ), start_date=str( govinfo_start_date ), time=int( govinfo_timeout ) )
					
					st.session_state[ 'govinfo_results' ] = result or { }
					st.session_state[ 'retrieval_active_source' ] = 'Gov Info'
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
					
					result = st.session_state.get( 'govinfo_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				payload = result.get( 'payload', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': mode_value, 'url': result.get( 'url', '' ), 'params': params,
						'payload': payload, } )
				
				items: List[ Dict[ str, Any ] ] = [ ]
				
				if isinstance( data, dict ):
					for key in [ 'packages', 'results', 'items' ]:
						value = data.get( key, None )
						if isinstance( value, list ):
							items = [ item for item in value if isinstance( item, dict ) ]
							break
				elif isinstance( data, list ):
					items = [ item for item in data if isinstance( item, dict ) ]
				
				if mode_value == 'package_summary' and isinstance( data, dict ) and data:
					st.markdown( '#### Package Summary' )
					
					title_value = (
							data.get( 'title' ) or data.get( 'packageTitle' ) or data.get( 'packageId' ) or 'Package')
					
					st.markdown( f'### {title_value}' )
					
					meta_parts: List[ str ] = [ ]
					for key in [ 'packageId', 'collectionCode', 'lastModified', 'dateIssued' ]:
						if key in data and str( data.get( key ) ).strip( ):
							meta_parts.append( f'{key}: `{data.get( key )}`' )
					
					if meta_parts:
						st.caption( ' | '.join( meta_parts ) )
					
					for text_key in [ 'summary', 'description' ]:
						if text_key in data and str( data.get( text_key ) ).strip( ):
							st.write( str( data.get( text_key ) ) )
							break
					
					with st.expander( 'Raw Package JSON', expanded=False ):
						st.json( data )
				
				elif items:
					st.markdown( '#### Results' )
					
					for index, item in enumerate( items, start=1 ):
						title_value = (
								item.get( 'title' ) or item.get( 'packageTitle' ) or item.get( 'packageId' ) or item.get( 'granuleId' ) or f'Result {index}')
						
						package_value = (
								item.get( 'packageId' ) or item.get( 'granuleId' ) or item.get( 'id' ) or '')
						
						collection_value = (
									item.get( 'collectionCode' ) or item.get( 'collectionName' ) or item.get( 'collection' ) or '')
						
						date_value = (
									item.get( 'lastModified' ) or item.get( 'dateIssued' ) or item.get( 'publishDate' ) or '')
						
						summary_value = (
								item.get( 'summary' ) or item.get( 'description' ) or item.get( 'snippet' ) or '')
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							meta_parts: List[ str ] = [ ]
							
							if package_value:
								meta_parts.append( f'ID: `{package_value}`' )
							
							if collection_value:
								meta_parts.append( f'Collection: `{collection_value}`' )
							
							if date_value:
								meta_parts.append( f'Date: `{date_value}`' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if summary_value:
								st.write( str( summary_value ) )
							else:
								st.caption( 'No summary available.' )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Result' )
					st.json( data )
				elif data:
					st.markdown( '#### Result' )
					st.text_area( 'Output', value=str( data ), height=320 )
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# ----------------------------
		# -------- Expander (Congress)
		# ----------------------------
		with st.expander( label='US Congress', icon='⚖️', expanded=False ):
			st.badge( label='Information', help=cfg.CONGRESS )
			if 'congress_results' not in st.session_state:
				st.session_state[ 'congress_results' ] = { }
			
			if 'congress_clear_request' not in st.session_state:
				st.session_state[ 'congress_clear_request' ] = False
			
			if st.session_state.get( 'congress_clear_request', False ):
				st.session_state[ 'congress_mode' ] = 'congresses'
				st.session_state[ 'congress_number' ] = 119
				st.session_state[ 'congress_bill_type' ] = ''
				st.session_state[ 'congress_bill_number' ] = 0
				st.session_state[ 'congress_law_type' ] = ''
				st.session_state[ 'congress_law_number' ] = 0
				st.session_state[ 'congress_report_type' ] = ''
				st.session_state[ 'congress_report_number' ] = 0
				st.session_state[ 'congress_offset' ] = 0
				st.session_state[ 'congress_limit' ] = 20
				st.session_state[ 'congress_sort' ] = 'updateDate+desc'
				st.session_state[ 'congress_from_datetime' ] = ''
				st.session_state[ 'congress_to_datetime' ] = ''
				st.session_state[ 'congress_conference' ] = False
				st.session_state[ 'congress_timeout' ] = 20
				st.session_state[ 'congress_results' ] = { }
				st.session_state[ 'congress_clear_request' ] = False
			
			def _clear_congress_state( ) -> None:
				st.session_state[ 'congress_clear_request' ] = True
			
			congress_mode = st.selectbox( 'Mode', options=[ 'congresses', 'bills', 'bill_detail',
					'laws', 'law_detail', 'reports', 'report_detail' ], index=[ 'congresses',
					'bills', 'bill_detail', 'laws', 'law_detail', 'reports',
					'report_detail' ].index( st.session_state.get( 'congress_mode', 'congresses' ) ), key='congress_mode', help=(
				'Congress.gov is a structured endpoint API. '
				'Choose the specific operation you want to '
				'perform.') )
			
			congress_number = st.number_input( 'Congress Number', min_value=1, max_value=999, value=int( st.session_state.get( 'congress_number', 119 ) ), step=1, key='congress_number' )
			
			c1, c2 = st.columns( 2 )
			with c1:
				congress_bill_type = st.selectbox( 'Bill Type', options=[ '', 'hr', 's', 'hjres',
						'sjres', 'hconres', 'sconres', 'hres', 'sres' ], index=[ '', 'hr', 's',
						'hjres', 'sjres', 'hconres', 'sconres', 'hres',
						'sres' ].index( st.session_state.get( 'congress_bill_type', '' ) ), key='congress_bill_type' )
			
			with c2:
				congress_bill_number = st.number_input( 'Bill Number', min_value=0, max_value=999999, value=int( st.session_state.get( 'congress_bill_number', 0 ) ), step=1, key='congress_bill_number' )
			
			c3, c4 = st.columns( 2 )
			with c3:
				congress_law_type = st.selectbox( 'Law Type', options=[ '', 'pub', 'priv' ], index=[
						'', 'pub',
						'priv' ].index( st.session_state.get( 'congress_law_type', '' ) ), key='congress_law_type' )
			
			with c4:
				congress_law_number = st.number_input( 'Law Number', min_value=0, max_value=999999, value=int( st.session_state.get( 'congress_law_number', 0 ) ), step=1, key='congress_law_number' )
			
			c5, c6 = st.columns( 2 )
			with c5:
				congress_report_type = st.selectbox( 'Report Type', options=[ '', 'hrpt', 'srpt',
						'erpt' ], index=[ '', 'hrpt', 'srpt',
						'erpt' ].index( st.session_state.get( 'congress_report_type', '' ) ), key='congress_report_type' )
			
			with c6:
				congress_report_number = st.number_input( 'Report Number', min_value=0, max_value=999999, value=int( st.session_state.get( 'congress_report_number', 0 ) ), step=1, key='congress_report_number' )
			
			c7, c8 = st.columns( 2 )
			with c7:
				congress_offset = st.number_input( 'Offset', min_value=0, max_value=1000000, value=int( st.session_state.get( 'congress_offset', 0 ) ), step=1, key='congress_offset' )
			
			with c8:
				congress_limit = st.number_input( 'Limit', min_value=1, max_value=250, value=int( st.session_state.get( 'congress_limit', 20 ) ), step=1, key='congress_limit' )
			
			congress_sort = st.selectbox( 'Sort', options=[ 'updateDate+desc',
					'updateDate+asc' ], index=[ 'updateDate+desc',
					'updateDate+asc' ].index( st.session_state.get( 'congress_sort', 'updateDate+desc' ) ), key='congress_sort' )
			
			c9, c10 = st.columns( 2 )
			with c9:
				congress_from_datetime = st.text_input( 'From DateTime (ISO)', value=st.session_state.get( 'congress_from_datetime', '' ), key='congress_from_datetime', placeholder='YYYY-MM-DDTHH:MM:SSZ' )
			
			with c10:
				congress_to_datetime = st.text_input( 'To DateTime (ISO)', value=st.session_state.get( 'congress_to_datetime', '' ), key='congress_to_datetime', placeholder='YYYY-MM-DDTHH:MM:SSZ' )
			
			congress_conference = st.checkbox( 'Conference Reports', value=bool( st.session_state.get( 'congress_conference', False ) ), key='congress_conference' )
			
			congress_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'congress_timeout', 20 ) ), step=1, key='congress_timeout' )
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				congress_submit = st.button( 'Submit', key='congress_submit', use_container_width=True, width='stretch' )
			
			with b2:
				congress_clear = st.button( 'Clear', key='congress_clear', on_click=_clear_congress_state, use_container_width=True, width='stretch' )
			
			if congress_submit:
				try:
					f = Congress( )
					
					result = f.fetch( mode=str( congress_mode ), congress=int( congress_number ), bill_type=str( congress_bill_type ), bill_number=int( congress_bill_number ), law_type=str( congress_law_type ), law_number=int( congress_law_number ), report_type=str( congress_report_type ), report_number=int( congress_report_number ), offset=int( congress_offset ), limit=int( congress_limit ), sort=str( congress_sort ), from_date_time=str( congress_from_datetime ), to_date_time=str( congress_to_datetime ), conference=bool( congress_conference ), time=int( congress_timeout ) )
					
					st.session_state[ 'congress_results' ] = result or { }
					st.session_state[ 'retrieval_active_source' ] = 'US Congress'
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
			
			result = st.session_state.get( 'congress_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': mode_value, 'url': result.get( 'url', '' ), 'params': params, } )
				
				items: List[ Dict[ str, Any ] ] = [ ]
				
				if isinstance( data, dict ):
					for key in [ 'bills', 'laws', 'reports', 'committeeReports', 'congresses',
							'sessions', 'results', 'items' ]:
						value = data.get( key, None )
						if isinstance( value, list ):
							items = [ item for item in value if isinstance( item, dict ) ]
							break
				elif isinstance( data, list ):
					items = [ item for item in data if isinstance( item, dict ) ]
				
				if items:
					st.markdown( '#### Results' )
					
					for index, item in enumerate( items, start=1 ):
						title_value = (
									item.get( 'title' ) or item.get( 'name' ) or item.get( 'number' ) or item.get( 'billNumber' ) or item.get( 'lawNumber' ) or item.get( 'reportNumber' ) or f'Result {index}')
						
						id_parts: List[ str ] = [ ]
						for key in [ 'congress', 'type', 'number', 'billType', 'billNumber',
								'lawType', 'lawNumber', 'reportType', 'reportNumber' ]:
							if key in item and str( item.get( key ) ).strip( ):
								id_parts.append( f'{key}={item.get( key )}' )
						
						action_value = (
									item.get( 'latestAction' ) or item.get( 'latestActionText' ) or item.get( 'actionDate' ) or '')
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							if id_parts:
								st.caption( ' | '.join( id_parts ) )
							
							if isinstance( action_value, dict ):
								st.write( str( action_value ) )
							elif action_value:
								st.write( str( action_value ) )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Detail' )
					
					title_value = (
								data.get( 'title' ) or data.get( 'name' ) or data.get( 'number' ) or data.get( 'billNumber' ) or data.get( 'lawNumber' ) or data.get( 'reportNumber' ) or 'Result')
					
					st.markdown( f'### {title_value}' )
					
					summary_fields: Dict[ str, Any ] = { }
					for key in [ 'congress', 'type', 'number', 'billType', 'billNumber', 'lawType',
							'lawNumber', 'reportType', 'reportNumber', 'updateDate', 'actionDate' ]:
						if key in data:
							summary_fields[ key ] = data.get( key )
					
					if summary_fields:
						st.json( summary_fields )
					
					for key in [ 'summary', 'latestAction', 'description' ]:
						if key in data and str( data.get( key ) ).strip( ):
							st.markdown( f'#### {key}' )
							st.write( str( data.get( key ) ) )
					
					with st.expander( 'Raw Detail JSON', expanded=False ):
						st.json( data )
				
				elif data:
					st.markdown( '#### Result' )
					st.text_area( 'Output', value=str( data ), height=320 )
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# ----------------------------
		# -------- Expander (Internet Archive)
		# ----------------------------
		with st.expander( label='Internet Archive', icon='🌐', expanded=False ):
			st.badge( label='Information', help=cfg.INTERNET_ARCHIVE )
			if 'internetarchive_results' not in st.session_state:
				st.session_state[ 'internetarchive_results' ] = { }
			
			if 'internetarchive_clear_request' not in st.session_state:
				st.session_state[ 'internetarchive_clear_request' ] = False
			
			if st.session_state.get( 'internetarchive_clear_request', False ):
				st.session_state[ 'internetarchive_query' ] = ''
				st.session_state[ 'internetarchive_rows' ] = 10
				st.session_state[ 'internetarchive_page' ] = 1
				st.session_state[ 'internetarchive_sort' ] = 'downloads desc'
				st.session_state[ 'internetarchive_media_type' ] = ''
				st.session_state[ 'internetarchive_collection' ] = ''
				st.session_state[ 'internetarchive_timeout' ] = 20
				st.session_state[ 'internetarchive_results' ] = { }
				st.session_state[ 'internetarchive_clear_request' ] = False
			
			def _clear_internetarchive_state( ) -> None:
				st.session_state[ 'internetarchive_clear_request' ] = True
			
			ia_query = st.text_area( 'Query', value=st.session_state.get( 'internetarchive_query', '' ), height=80, key='internetarchive_query', placeholder=(
				'Examples:\n'
				'climate change\n'
				'title:"appropriations" AND '
				'creator:"Congress"\n'
				'budget execution') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				ia_rows = st.number_input( 'Rows', min_value=1, max_value=100, value=int( st.session_state.get( 'internetarchive_rows', 10 ) ), step=1, key='internetarchive_rows' )
			
			with c2:
				ia_page = st.number_input( 'Page', min_value=1, max_value=100000, value=int( st.session_state.get( 'internetarchive_page', 1 ) ), step=1, key='internetarchive_page' )
			
			ia_sort = st.selectbox( 'Sort', options=[ 'downloads desc', 'downloads asc',
					'publicdate desc', 'publicdate asc', 'titleSorter asc',
					'titleSorter desc' ], index=[ 'downloads desc', 'downloads asc',
					'publicdate desc', 'publicdate asc', 'titleSorter asc',
					'titleSorter desc' ].index( st.session_state.get( 'internetarchive_sort', 'downloads desc' ) ), key='internetarchive_sort' )
			
			c3, c4 = st.columns( 2 )
			with c3:
				ia_media_type = st.text_input( 'Mediatype', value=st.session_state.get( 'internetarchive_media_type', '' ), key='internetarchive_media_type', placeholder='Example: texts' )
			
			with c4:
				ia_collection = st.text_input( 'Collection', value=st.session_state.get( 'internetarchive_collection', '' ), key='internetarchive_collection', placeholder='Example: americana' )
			
			ia_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'internetarchive_timeout', 20 ) ), step=1, key='internetarchive_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				ia_submit = st.button( 'Submit', key='internetarchive_submit', use_container_width=True, width='stretch' )
			
			with b2:
				ia_clear = st.button( 'Clear', key='internetarchive_clear', on_click=_clear_internetarchive_state, use_container_width=True, width='stretch' )
			
			result = st.session_state.get( 'internetarchive_results', { } )
			
			if ia_submit:
				try:
					f = InternetArchive( )
					
					result = f.fetch( keywords=str( ia_query ), rows=int( ia_rows ), page=int( ia_page ), sort=str( ia_sort ), media_type=str( ia_media_type ), collection=str( ia_collection ), time=int( ia_timeout ) )
					
					st.session_state[ 'internetarchive_results' ] = result or { }
					st.session_state[ 'retrieval_active_source' ] = 'Internet Archive'
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
				
				result = st.session_state.get( 'internetarchive_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': mode_value, 'url': result.get( 'url', '' ), 'params': params, } )
				
				docs: List[ Dict[ str, Any ] ] = [ ]
				num_found = None
				
				if isinstance( data, dict ):
					response_block = data.get( 'response', { } )
					
					if isinstance( response_block, dict ):
						num_found = response_block.get( 'numFound', None )
						value = response_block.get( 'docs', [ ] )
						if isinstance( value, list ):
							docs = [ item for item in value if isinstance( item, dict ) ]
				
				if num_found is not None:
					st.markdown( f'#### Search Results ({num_found})' )
				else:
					st.markdown( '#### Search Results' )
				
				if docs:
					for index, item in enumerate( docs, start=1 ):
						title_value = (
									item.get( 'title' ) or item.get( 'identifier' ) or f'Result {index}')
						
						identifier_value = item.get( 'identifier', '' )
						mediatype_value = item.get( 'mediatype', '' )
						
						collection_value = ''
						collection_raw = item.get( 'collection', '' )
						if isinstance( collection_raw, list ) and collection_raw:
							collection_value = ', '.join( [ str( x ) for x in
									collection_raw[ :3 ] ] )
						elif collection_raw:
							collection_value = str( collection_raw )
						
						date_value = (
								item.get( 'publicdate' ) or item.get( 'date' ) or item.get( 'addeddate' ) or '')
						
						desc_value = item.get( 'description', '' )
						if isinstance( desc_value, list ):
							desc_value = ' '.join( [ str( x ) for x in desc_value[ :2 ] ] )
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							meta_parts: List[ str ] = [ ]
							
							if identifier_value:
								meta_parts.append( f'Identifier: `{identifier_value}`' )
							
							if mediatype_value:
								meta_parts.append( f'Mediatype: `{mediatype_value}`' )
							
							if collection_value:
								meta_parts.append( f'Collection: `{collection_value}`' )
							
							if date_value:
								meta_parts.append( f'Date: `{date_value}`' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if desc_value:
								st.write( str( desc_value ) )
							else:
								st.caption( 'No description available.' )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Result' )
					st.json( data )
				elif isinstance( data, list ) and data:
					df_ia = pd.DataFrame( data )
					if not df_ia.empty:
						st.dataframe( df_ia, use_container_width=True, hide_index=True )
					else:
						st.json( data )
				elif data:
					st.markdown( '#### Result' )
					st.text_area( 'Output', value=str( data ), height=320 )
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# ----------------------------
		# -------- Expander (Grokipedia)
		# ----------------------------
		with st.expander( label='Grokipedia', icon='🧠', expanded=False ):
			st.badge( label='Information', help=cfg.GROKIPEDIA )
			if 'grokipedia_results' not in st.session_state:
				st.session_state[ 'grokipedia_results' ] = { }
			
			if 'grokipedia_clear_request' not in st.session_state:
				st.session_state[ 'grokipedia_clear_request' ] = False
			
			if 'grokipedia_auto_fetch_page' not in st.session_state:
				st.session_state[ 'grokipedia_auto_fetch_page' ] = False
			
			if st.session_state.get( 'grokipedia_clear_request', False ):
				st.session_state[ 'grokipedia_mode' ] = 'search'
				st.session_state[ 'grokipedia_query' ] = ''
				st.session_state[ 'grokipedia_page' ] = ''
				st.session_state[ 'grokipedia_limit' ] = 12
				st.session_state[ 'grokipedia_offset' ] = 0
				st.session_state[ 'grokipedia_include_content' ] = True
				st.session_state[ 'grokipedia_results' ] = { }
				st.session_state[ 'grokipedia_auto_fetch_page' ] = False
				st.session_state[ 'grokipedia_clear_request' ] = False
			
			def _clear_grokipedia_state( ) -> None:
				st.session_state[ 'grokipedia_clear_request' ] = True
			
			def _load_grokipedia_page( slug: str ) -> None:
				st.session_state[ 'grokipedia_mode' ] = 'page'
				st.session_state[ 'grokipedia_page' ] = str( slug ).strip( )
				st.session_state[ 'grokipedia_include_content' ] = True
				st.session_state[ 'grokipedia_auto_fetch_page' ] = True
			
			grokipedia_mode = st.selectbox( 'Mode', options=[ 'search', 'page' ], index=[ 'search',
					'page' ].index( st.session_state.get( 'grokipedia_mode', 'search' ) ), key='grokipedia_mode', help='search = keyword search; page = fetch a specific page by slug.' )
			
			grokipedia_query = st.text_input( 'Query', value=st.session_state.get( 'grokipedia_query', '' ), key='grokipedia_query', placeholder='Example: machine learning' )
			
			grokipedia_page = st.text_input( 'Page Slug', value=st.session_state.get( 'grokipedia_page', '' ), key='grokipedia_page', placeholder='Example: United_Petroleum' )
			
			c1, c2 = st.columns( 2 )
			with c1:
				grokipedia_limit = st.number_input( 'Limit', min_value=1, max_value=100, value=int( st.session_state.get( 'grokipedia_limit', 12 ) ), step=1, key='grokipedia_limit' )
			
			with c2:
				grokipedia_offset = st.number_input( 'Offset', min_value=0, max_value=100000, value=int( st.session_state.get( 'grokipedia_offset', 0 ) ), step=1, key='grokipedia_offset' )
			
			grokipedia_include_content = st.checkbox( 'Include Content', value=bool( st.session_state.get( 'grokipedia_include_content', True ) ), key='grokipedia_include_content' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				grokipedia_submit = st.button( 'Submit', key='grokipedia_submit', use_container_width=True, width='stretch' )
			
			with b2:
				grokipedia_clear = st.button( 'Clear', key='grokipedia_clear', on_click=_clear_grokipedia_state, use_container_width=True, width='stretch' )
			
			should_fetch_grokipedia = False
			
			if grokipedia_submit:
				should_fetch_grokipedia = True
			
			if st.session_state.get( 'grokipedia_auto_fetch_page', False ):
				should_fetch_grokipedia = True
			
			if should_fetch_grokipedia:
				try:
					f = Grokipedia( )
					result = f.fetch( mode=str( st.session_state.get( 'grokipedia_mode', 'search' ) ), query=str( st.session_state.get( 'grokipedia_query', '' ) ), page=str( st.session_state.get( 'grokipedia_page', '' ) ), limit=int( st.session_state.get( 'grokipedia_limit', 12 ) ), offset=int( st.session_state.get( 'grokipedia_offset', 0 ) ), include_content=bool( st.session_state.get( 'grokipedia_include_content', True ) ) )
					
					st.session_state[ 'grokipedia_results' ] = result or { }
					st.session_state[ 'retrieval_active_source' ] = 'Grokipedia'
					st.session_state[ 'grokipedia_auto_fetch_page' ] = False
					st.rerun( )
				
				except Exception as exc:
					st.session_state[ 'grokipedia_auto_fetch_page' ] = False
					st.error( str( exc ) )
			
			result = st.session_state.get( 'grokipedia_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': result.get( 'mode', '' ), 'url': result.get( 'url', '' ),
						'params': result.get( 'params', { } ),
						'api_key_configured': result.get( 'api_key_configured', False ) } )
				
				if mode_value == 'search':
					st.markdown( '#### Search Results' )
					items: List[ Dict[ str, Any ] ] = [ ]
					if isinstance( data, list ):
						items = [ item for item in data if isinstance( item, dict ) ]
					elif isinstance( data, dict ):
						for key in [ 'results', 'items', 'pages', 'data' ]:
							value = data.get( key, None )
							if isinstance( value, list ):
								items = [ item for item in value if isinstance( item, dict ) ]
								break
					
					if not items:
						if data:
							st.info( 'No structured search hits were detected. '
							         'Showing raw output.' )
							st.json( data )
						else:
							st.info( 'No results returned.' )
					else:
						for index, item in enumerate( items, start=1 ):
							title_value = (
									item.get( 'title' ) or item.get( 'name' ) or item.get( 'slug' ) or item.get( 'id' ) or f'Result {index}')
							
							slug_value = (
										item.get( 'slug' ) or item.get( 'page' ) or item.get( 'path' ) or item.get( 'id' ) or '')
							
							summary_value = (
										item.get( 'summary' ) or item.get( 'description' ) or item.get( 'excerpt' ) or item.get( 'snippet' ) or '')
							
							score_value = (
									item.get( 'score' ) or item.get( 'rank' ) or item.get( 'relevance' ))
							
							with st.container( border=True ):
								st.markdown( f'**{index}. {title_value}**' )
								meta_parts: List[ str ] = [ ]
								
								if slug_value:
									meta_parts.append( f'Slug: `{slug_value}`' )
								
								if score_value is not None and str( score_value ).strip( ):
									meta_parts.append( f'Score: `{score_value}`' )
								
								if meta_parts:
									st.caption( ' | '.join( meta_parts ) )
								
								if summary_value:
									st.write( str( summary_value ) )
								else:
									st.caption( 'No summary available.' )
								
								ca, cb = st.columns( 2 )
								
								with ca:
									if slug_value:
										if st.button( 'Load Page', key=f'grokipedia_load_page_{index}_'
										                               f'{slug_value}', use_container_width=True ):
											_load_grokipedia_page( slug_value )
											st.rerun( )
								
								with cb:
									with st.expander( 'Raw Item', expanded=False ):
										st.json( item )
				
				elif mode_value == 'page':
					st.markdown( '#### Page Result' )
					
					page_item: Dict[ str, Any ] = data if isinstance( data, dict ) else { }
					
					if not page_item:
						if data:
							st.text_area( 'Output', value=str( data ), height=320 )
						else:
							st.info( 'No results returned.' )
					else:
						title_value = (
									page_item.get( 'title' ) or page_item.get( 'name' ) or page_item.get( 'slug' ) or page_item.get( 'id' ) or 'Untitled Page')
						
						slug_value = (
									page_item.get( 'slug' ) or page_item.get( 'page' ) or page_item.get( 'path' ) or page_item.get( 'id' ) or '')
						
						summary_value = (
									page_item.get( 'summary' ) or page_item.get( 'description' ) or page_item.get( 'excerpt' ) or '')
						
						content_value = (
									page_item.get( 'content' ) or page_item.get( 'text' ) or page_item.get( 'body' ) or '')
						
						st.markdown( f'### {title_value}' )
						
						if slug_value:
							st.caption( f'Slug: `{slug_value}`' )
						
						if summary_value:
							st.markdown( '#### Summary' )
							st.write( str( summary_value ) )
						
						if content_value:
							st.markdown( '#### Content' )
							st.text_area( 'Page Content', value=str( content_value ), height=360 )
						else:
							st.info( 'No page content was returned.' )
						
						with st.expander( 'Raw Page JSON', expanded=False ):
							st.json( page_item )
				
				else:
					st.markdown( '#### Result' )
					
					if isinstance( data, dict ) or isinstance( data, list ):
						st.json( data )
					elif data:
						st.text_area( 'Output', value=str( data ), height=320 )
					else:
						st.info( 'No results returned.' )
		
		# ----------------------------
		# -------- Expander (Jupyter Notebook)
		# ----------------------------
		with st.expander( label='Jupyter Notebook', icon='🪐', expanded=False ):
			st.badge( label='Information', help=cfg.JUPYTER_NOTEBOOK )
			if 'jupyter_notebook_results' not in st.session_state:
				st.session_state[ 'jupyter_notebook_results' ] = { }
			
			notebook_file = st.file_uploader( 'Upload Notebook', type=[ 'ipynb' ],
				key='jupyter_notebook_upload' )
			
			include_outputs = st.checkbox( 'Include Outputs', value=True,
				key='jupyter_notebook_include_outputs' )
			
			max_output_length = st.number_input( 'Max Output Length', min_value=1, max_value=20000,
				value=100, step=10, key='jupyter_notebook_max_output_length' )
			
			remove_newline = st.checkbox( 'Remove Newline', value=False, key='jupyter_notebook_remove_newline' )
			
			include_traceback = st.checkbox( 'Traceback', value=False, key='jupyter_notebook_traceback' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				jupyter_submit = st.button( 'Submit', key='jupyter_notebook_submit', use_container_width=True, width='stretch' )
			
			with b2:
				jupyter_clear = st.button( 'Clear', key='jupyter_notebook_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'JupyterNotebookLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='jupyter_notebook_loader_output.txt', mime='text/plain', key='jupyter_notebook_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='jupyter_notebook_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander (Google Cloud File)
		# ----------------------------
		with st.expander( label='Google Cloud File', icon='☁️', expanded=False ):
			st.badge( label='Information', help=cfg.GOOGLE_FILE )
			if 'google_cloud_file_results' not in st.session_state:
				st.session_state[ 'google_cloud_file_results' ] = { }
			
			project_name = st.text_input( 'Project Name', key='google_cloud_file_project_name' )
			
			bucket = st.text_input( 'Bucket', key='google_cloud_file_bucket' )
			
			blob = st.text_input( 'Blob', key='google_cloud_file_blob',
				help='The exact object name in the Google Cloud Storage bucket.' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				google_cloud_file_submit = st.button( 'Submit', key='google_cloud_file_submit', use_container_width=True, width='stretch' )
			
			with b2:
				google_cloud_file_clear = st.button( 'Clear', key='google_cloud_file_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'GoogleCloudStorageFileLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='google_cloud_file_loader_output.txt', mime='text/plain', key='google_cloud_file_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='google_cloud_file_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander (AWS S3 File)
		# ----------------------------
		with st.expander( label='AWS S3 File', icon='📗', expanded=False ):
			st.badge( label='Information', help=cfg.AWS_FILE )
			if 'aws_file_results' not in st.session_state:
				st.session_state[ 'aws_file_results' ] = { }
			
			bucket = st.text_input( 'Bucket', key='aws_file_bucket' )
			
			key_name = st.text_input( 'Key', key='aws_file_key', help='The exact S3 object key to load.' )
			
			region_name = st.text_input( 'Region (Optional)', key='aws_file_region_name' )
			
			aws_access_key_id = st.text_input( 'AWS Access Key ID (Optional)', type='password', key='aws_file_access_key' )
			
			aws_secret_access_key = st.text_input( 'AWS Secret Access Key (Optional)', type='password', key='aws_file_secret_key' )
			
			aws_session_token = st.text_input( 'AWS Session Token (Optional)', type='password', key='aws_file_session_token' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				aws_file_submit = st.button( 'Submit', key='aws_file_submit', use_container_width=True, width='stretch' )
			
			with b2:
				aws_file_clear = st.button( 'Clear', key='aws_file_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'AwsFileLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='aws_s3_file_loader_output.txt', mime='text/plain', key='aws_file_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='aws_file_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander (OneDrive)
		# ----------------------------
		with st.expander( label='OneDrive', icon='💻', expanded=False ):
			st.badge( label='Information', help=cfg.ONEDRIVE )
			if 'onedrive_results' not in st.session_state:
				st.session_state[ 'onedrive_results' ] = { }
			
			drive_id = st.text_input( 'Drive ID', key='onedrive_drive_id' )
			folder_path = st.text_input( 'Folder Path (Optional)', key='onedrive_folder_path',
				help='Example: Documents/clients' )
			
			object_ids_text = st.text_area( 'Object IDs (Optional)',
				key='onedrive_object_ids',
				help='Optional comma-separated OneDrive object IDs. Leave blank to load folder path.' )
			
			auth_with_token = st.checkbox( 'Authenticate With Cached Token', value=True, key='onedrive_auth_with_token' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				onedrive_submit = st.button( 'Submit', key='onedrive_submit', use_container_width=True, width='stretch' )
			
			with b2:
				onedrive_clear = st.button( 'Clear', key='onedrive_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'OneDriveDocLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='onedrive_loader_output.txt', mime='text/plain', key='onedrive_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='onedrive_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander (Google Speech-to-Text)
		# ----------------------------
		with st.expander( label='Google Speech-to-Text', icon='🗣️', expanded=False ):
			st.badge( label='Information', help=cfg.GOOGLE_STT )
			if 'google_speech_to_text_results' not in st.session_state:
				st.session_state[ 'google_speech_to_text_results' ] = { }
			
			project_id = st.text_input( 'Project ID', key='google_speech_to_text_project_id' )
			audio_file = st.file_uploader( 'Upload Audio File', type=[ 'wav', 'flac', 'mp3', 'm4a',
					'ogg' ], key='google_speech_to_text_audio_upload' )
			
			gcs_audio_uri = st.text_input( 'GCS Audio URI (Optional)', placeholder='gs://bucket/path/audio.flac', key='google_speech_to_text_gcs_uri', help='Use either a local upload or a gs:// URI.' )
			
			language_code = st.text_input( 'Language Code (Optional)', value='en-US', key='google_speech_to_text_language_code' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				google_speech_submit = st.button( 'Submit', key='google_speech_to_text_submit', use_container_width=True, width='stretch' )
			
			with b2:
				google_speech_clear = st.button( 'Clear', key='google_speech_to_text_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'GoogleSpeechToTextLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='google_speech_to_text_loader_output.txt', mime='text/plain', key='google_speech_to_text_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='google_speech_to_text_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander  (AWS S3 Bucket)
		# ----------------------------
		with st.expander( label='AWS S3 Bucket', icon='🗂️', expanded=False ):
			st.badge( label='Information', help=cfg.AWS_BUCKET )
			if 'aws_bucket_results' not in st.session_state:
				st.session_state[ 'aws_bucket_results' ] = { }
			
			bucket_name = st.text_input( 'Bucket', key='aws_bucket_name' )
			prefix = st.text_input( 'Prefix (Optional)', key='aws_bucket_prefix', help='Optional folder / key prefix inside the bucket.' )
			
			region_name = st.text_input( 'Region (Optional)', key='aws_bucket_region_name' )
			endpoint_url = st.text_input( 'Endpoint URL (Optional)', key='aws_bucket_endpoint_url', help='Optional S3-compatible endpoint URL.' )
			
			aws_access_key_id = st.text_input( 'AWS Access Key ID (Optional)', type='password', key='aws_bucket_access_key' )
			
			aws_secret_access_key = st.text_input( 'AWS Secret Access Key (Optional)', type='password', key='aws_bucket_secret_key' )
			
			aws_session_token = st.text_input( 'AWS Session Token (Optional)', type='password', key='aws_bucket_session_token' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				aws_bucket_submit = st.button( 'Submit', key='aws_bucket_submit', use_container_width=True, width='stretch' )
			
			with b2:
				aws_bucket_clear = st.button( 'Clear', key='aws_bucket_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'AmazonBucketLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='aws_s3_bucket_loader_output.txt', mime='text/plain', key='aws_bucket_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='aws_bucket_save_disabled', disabled=True, use_container_width=True, width='stretch' )
		
		# ----------------------------
		# -------- Expander (Google Cloud Bucket)
		# ----------------------------
		with st.expander( label='Google Cloud Bucket', icon='🧊', expanded=False ):
			st.badge( label='Information', help=cfg.GOOGLE_BUCKET )
			if 'google_bucket_results' not in st.session_state:
				st.session_state[ 'google_bucket_results' ] = { }
			
			project_name = st.text_input( 'Project Name', key='google_bucket_project_name' )
			
			bucket_name = st.text_input( 'Bucket', key='google_bucket_name' )
			
			prefix = st.text_input( 'Prefix (Optional)', key='google_bucket_prefix',
				help='Optional folder / object prefix filter inside the bucket.' )
			
			continue_on_failure = st.checkbox( 'Continue On Failure', value=False, key='google_bucket_continue_on_failure', help='Skip objects that fail to load instead of aborting the whole request.' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				google_bucket_submit = st.button( 'Submit', key='google_bucket_submit', use_container_width=True, width='stretch' )
			
			with b2:
				google_bucket_clear = st.button( 'Clear', key='google_bucket_clear', use_container_width=True, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'GoogleBucketLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='google_bucket_loader_output.txt', mime='text/plain', key='google_bucket_save', use_container_width=True, width='stretch' )
				else:
					st.button( 'Save', key='google_bucket_save_disabled', disabled=True, use_container_width=True, width='stretch' )
	
	# ------------------------------------------------------------------
	# Shared result selection
	# ------------------------------------------------------------------
	if do_submit:
		st.session_state[ 'retrieval_active_source' ] = 'ArXiv'
	elif gd_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Google Drive'
	elif wiki_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Wikipedia'
	elif google_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Google Search'
	elif openscience_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Open Science'
	elif govinfo_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Gov Info'
	elif congress_submit:
		st.session_state[ 'retrieval_active_source' ] = 'US Congress'
	elif ia_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Internet Archive'
	elif grokipedia_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Grokipedia'
	elif jupyter_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Jupyter Notebook'
	elif google_cloud_file_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Google Cloud File'
	elif aws_file_submit:
		st.session_state[ 'retrieval_active_source' ] = 'AWS S3 File'
	elif onedrive_submit:
		st.session_state[ 'retrieval_active_source' ] = 'OneDrive'
	elif google_speech_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Google Speech-to-Text'
	elif aws_bucket_submit:
		st.session_state[ 'retrieval_active_source' ] = 'AWS S3 Bucket'
	elif google_bucket_submit:
		st.session_state[ 'retrieval_active_source' ] = 'Google Cloud Bucket'
	
	with right:
		st.markdown( '##### Results' )
		active_source = st.session_state.get( 'retrieval_active_source', '' )
		if not active_source:
			st.info( 'Select a source, configure the request, and submit it to display results.' )
		else:
			st.caption( f'Active Source: {active_source}' )
		
		# -------- ArXiv
		if active_source == 'ArXiv':
			st.markdown( 'Results' )
			results = st.session_state.get( 'arxiv_results', [ ] )
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'Title', '' ) ) if doc.metadata else ''
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					with st.expander( label, expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								with meta_col1:
									if 'Title' in doc.metadata:
										st.markdown( f"**Title:** {doc.metadata.get( 'Title', '' )}" )
									if 'Authors' in doc.metadata:
										st.markdown( f"**Authors:** "
										             f"{doc.metadata.get( 'Authors', '' )}" )
								
								with meta_col2:
									if 'Published' in doc.metadata:
										st.markdown( f"**Published:** "
										             f"{doc.metadata.get( 'Published', '' )}" )
									if 'Entry ID' in doc.metadata:
										st.markdown( f"**Entry ID:** "
										             f"{doc.metadata.get( 'Entry ID', '' )}" )
							
							st.text_area( 'Content', value=doc.page_content or '', height=300, key=f'arxiv_doc_{idx}' )
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
		
		# -------- Google Drive
		if active_source == 'Google Drive':
			st.markdown( 'Results', help=cfg.GOOGLE_DRIVE )
			
			results = st.session_state.get( 'googledrive_results', [ ] )
			
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'name', '' ) ) if doc.metadata else ''
					
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					
					with st.expander( label, expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								
								with meta_col1:
									if 'name' in doc.metadata:
										st.markdown( f"**Name:** {doc.metadata.get( 'name', '' )}" )
									if 'id' in doc.metadata:
										st.markdown( f"**ID:** "
										             f"{doc.metadata.get( 'id', '' )}" )
								
								with meta_col2:
									if 'mimeType' in doc.metadata:
										st.markdown( f"**MIME Type:** {doc.metadata.get( 'mimeType', '' )}" )
									if 'modifiedTime' in doc.metadata:
										st.markdown( f"**Modified:** "
										             f"{doc.metadata.get( 'modifiedTime', '' )}" )
							
							st.text_area( 'Content', value=doc.page_content or '', height=300, key=f'googledrive_doc_{idx}' )
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
		
		# -------- Wikipedia
		if active_source == 'Wikipedia':
			st.markdown( 'Results' )
			
			results = st.session_state.get( 'wikipedia_results', [ ] )
			
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'title', '' ) ) if doc.metadata else ''
					
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					
					with st.expander( 'Search Results', expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								
								with meta_col1:
									if 'title' in doc.metadata:
										st.markdown( f"**Title:** {doc.metadata.get( 'title', '' )}" )
									if 'source' in doc.metadata:
										st.markdown( f"**Source:** {doc.metadata.get( 'source', '' )}" )
								
								with meta_col2:
									if 'categories' in doc.metadata:
										st.markdown( f"**Categories:** "
										             f"{doc.metadata.get( 'categories', '' )}" )
									if 'pageid' in doc.metadata:
										st.markdown( f"**Page ID:** "
										             f"{doc.metadata.get( 'pageid', '' )}" )
							
							st.text_area( 'Content', value=doc.page_content or '', height=300, key=f'wikipedia_doc_{idx}' )
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
		
		# -------- Google Search
		if active_source == 'Google Search':
			st.markdown( 'Results' )
			
			if google_submit:
				try:
					f = GoogleSearch( )
					result = f.fetch( keywords=google_query, results=int( google_num_results ), start=int( google_start ), exact_terms=google_exact_terms, exclude_terms=google_exclude_terms, file_type=google_file_type, date_restrict=google_date_restrict, gl=google_gl, lr=google_lr, safe=google_safe, search_type=google_search_type, site_search=google_site_search, site_search_filter=google_site_search_filter, sort=google_sort, img_size=google_img_size, img_type=google_img_type, img_color_type=google_img_color_type, img_dominant_color=google_img_dominant_color, time=int( google_timeout ), api_key=(
								google_api_key or None), cse_id=(google_cse_id or None) )
					
					st.session_state[ 'googlesearch_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Search request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googlesearch_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				queries = result.get( 'queries', { } ) if isinstance( result, dict ) else { }
				search_info = result.get( 'searchInformation', { } ) if isinstance( result, dict ) else { }
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if queries or search_info:
					st.markdown( '#### Search Metadata' )
					
					meta_summary: Dict[ str, Any ] = { }
					
					if isinstance( search_info, dict ):
						for key in [ 'searchTime', 'formattedSearchTime', 'totalResults',
								'formattedTotalResults' ]:
							if key in search_info:
								meta_summary[ key ] = search_info.get( key )
					
					if isinstance( queries, dict ) and 'request' in queries:
						requests = queries.get( 'request', [ ] )
						if isinstance( requests, list ) and requests:
							request_item = requests[ 0 ]
							if isinstance( request_item, dict ):
								for key in [ 'searchTerms', 'count', 'startIndex', 'inputEncoding',
										'outputEncoding' ]:
									if key in request_item:
										meta_summary[ key ] = request_item.get( key )
					
					if meta_summary:
						st.json( meta_summary )
					
					with st.expander( 'Raw Search Metadata', expanded=False ):
						st.json( { 'queries': queries, 'searchInformation': search_info, } )
				
				if not items:
					st.info( 'No results returned.' )
				else:
					for idx, item in enumerate( items, start=1 ):
						title = item.get( 'title', f'Result {idx}' )
						
						with st.container( border=True ):
							st.markdown( f'**{idx}. {title}**' )
							
							link_value = item.get( 'link', '' )
							display_link = item.get( 'displayLink', '' )
							snippet_value = item.get( 'snippet', '' )
							
							meta_parts: List[ str ] = [ ]
							if display_link:
								meta_parts.append( f'Domain: `{display_link}`' )
							
							pagemap = item.get( 'pagemap', { } ) if isinstance( item, dict ) else { }
							if isinstance( pagemap, dict ):
								if 'metatags' in pagemap:
									meta_parts.append( 'Has metatags' )
								if 'cse_image' in pagemap:
									meta_parts.append( 'Has image' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if link_value:
								st.markdown( f'**Link:** {link_value}' )
							
							if snippet_value:
								st.write( str( snippet_value ) )
							
							image_url = ''
							if isinstance( pagemap, dict ):
								cse_images = pagemap.get( 'cse_image', [ ] )
								if isinstance( cse_images, list ) and cse_images:
									first_img = cse_images[ 0 ]
									if isinstance( first_img, dict ):
										image_url = first_img.get( 'src', '' )
							
							if image_url and google_search_type == 'image':
								try:
									st.image( image_url, use_container_width=True )
								except Exception:
									pass
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
		
		# -------- Open Science
		if active_source == 'Open Science':
			st.markdown( 'Results' )
			
			if openscience_submit:
				try:
					f = OpenScience( )
					
					result = f.fetch( mode=str( openscience_mode ), query=str( openscience_query ), accession=str( openscience_accession ), format_value=str( openscience_format ), time=int( openscience_timeout ) )
					
					st.session_state[ 'openscience_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
			
			result = st.session_state.get( 'openscience_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': mode_value, 'url': result.get( 'url', '' ), 'params': params, } )
				
				if mode_value == 'dataset' and isinstance( data, dict ) and data:
					title_value = (
								data.get( 'title' ) or data.get( 'name' ) or data.get( 'accession' ) or params.get( 'accession', '' ) or 'Dataset')
					
					st.markdown( f'### {title_value}' )
					
					meta_fields: Dict[ str, Any ] = { }
					for key in [ 'accession', 'identifier', 'organism', 'platform', 'assay',
							'project', 'study' ]:
						if key in data:
							meta_fields[ key ] = data.get( key )
					
					if meta_fields:
						st.json( meta_fields )
					
					for key in [ 'summary', 'description', 'abstract' ]:
						if key in data and str( data.get( key ) ).strip( ):
							st.markdown( '#### Description' )
							st.write( str( data.get( key ) ) )
							break
					
					with st.expander( 'Raw Dataset JSON', expanded=False ):
						st.json( data )
				
				elif isinstance( data, list ) and data:
					df_os = pd.DataFrame( data )
					if not df_os.empty:
						st.markdown( f'#### Result Rows ({len( df_os )})' )
						st.dataframe( df_os, use_container_width=True, hide_index=True )
					else:
						st.json( data )
				
				elif isinstance( data, dict ) and data:
					table_candidates: List[ Dict[ str, Any ] ] = [ ]
					for key in [ 'results', 'items', 'rows', 'data' ]:
						value = data.get( key, None )
						if isinstance( value, list ) and value:
							table_candidates = [ item for item in value if
									isinstance( item, dict ) ]
							break
					
					if table_candidates:
						df_os = pd.DataFrame( table_candidates )
						if not df_os.empty:
							st.markdown( f'#### Result Rows ({len( df_os )})' )
							st.dataframe( df_os, use_container_width=True, hide_index=True )
						else:
							st.json( data )
					else:
						st.markdown( '#### Result' )
						st.json( data )
				
				elif data:
					st.text_area( 'Output', value=str( data ), height=320 )
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Jupyter Notebook
		if active_source == 'Jupyter Notebook':
			if jupyter_clear:
				st.session_state[ 'jupyter_notebook_results' ] = { }
				remaining = _clear_loader_documents( 'JupyterNotebookLoader' )
				st.info( f'Jupyter Notebook Loader state cleared. Remaining documen'
				         f'ts: {remaining}.' )
			
			if jupyter_submit:
				if not notebook_file:
					st.info( 'Upload a notebook file.' )
				else:
					try:
						with tempfile.TemporaryDirectory( ) as tmp:
							path = os.path.join( tmp, notebook_file.name )
							with open( path, 'wb' ) as f:
								f.write( notebook_file.read( ) )
							
							loader = JupyterNotebookLoader( )
							documents = loader.load( path=path, include_outputs=include_outputs, max_output_length=int( max_output_length ), remove_newline=remove_newline, traceback=include_traceback ) or [ ]
						
						count = _promote_loader_documents( documents, 'JupyterNotebookLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'Preview': content[ :200 ], 'Content': content,
									'Metadata': metadata, } )
						
						st.session_state[ 'jupyter_notebook_results' ] = {
								'mode': 'jupyter_notebook', 'include_outputs': include_outputs,
								'max_output_length': int( max_output_length ),
								'remove_newline': remove_newline, 'traceback': include_traceback,
								'count': count, 'items': items, }
						
						st.success( f'Loaded {count} notebook document(s).' )
					
					except Exception as exc:
						st.error( 'Jupyter Notebook request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'jupyter_notebook_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'IncludeOutputs': result.get( 'include_outputs', False ),
						'MaxOutputLength': result.get( 'max_output_length', 0 ),
						'RemoveNewline': result.get( 'remove_newline', False ),
						'Traceback': result.get( 'traceback', False ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '#### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First Notebook Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No notebook documents returned.' )
				
				_render_fallback_raw( result )
		
		# -------- Google Cloud FIle
		if active_source == 'Google Cloud File':
			if google_cloud_file_clear:
				st.session_state[ 'google_cloud_file_results' ] = { }
				remaining = _clear_loader_documents( 'GoogleCloudStorageFileLoader' )
				st.info( f'Google Cloud File Loader state cleared. Remaining documents: '
				         f'{remaining}.' )
			
			if google_cloud_file_submit:
				if not project_name or not project_name.strip( ):
					st.info( 'Enter a Project Name.' )
				elif not bucket or not bucket.strip( ):
					st.info( 'Enter a Bucket.' )
				elif not blob or not blob.strip( ):
					st.info( 'Enter a Blob.' )
				else:
					try:
						loader = GoogleCloudFileLoader( )
						documents = loader.load( project_name=project_name.strip( ), bucket=bucket.strip( ), blob=blob.strip( ) ) or [ ]
						
						count = _promote_loader_documents( documents, 'GoogleCloudStorageFileLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'Blob': blob.strip( ), 'Preview': content[ :200 ],
									'Content': content, 'Metadata': metadata, } )
						
						st.session_state[ 'google_cloud_file_results' ] = {
								'mode': 'google_cloud_file', 'project_name': project_name.strip( ),
								'bucket': bucket.strip( ), 'blob': blob.strip( ), 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} Google Cloud file document(s).' )
					
					except Exception as exc:
						st.error( 'Google Cloud File request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'google_cloud_file_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'ProjectName': result.get( 'project_name', '' ),
						'Bucket': result.get( 'bucket', '' ), 'Blob': result.get( 'blob', '' ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Blob': item.get( 'Blob', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '#### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First File Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No Google Cloud file documents returned.' )
			
			_render_fallback_raw( result )
		
		# -------- AWS S3 File
		if active_source == 'AWS S3 File':
			if aws_file_clear:
				st.session_state[ 'aws_file_results' ] = { }
				remaining = _clear_loader_documents( 'AwsFileLoader' )
				st.info( f'AWS S3 File Loader state cleared. Remaining documents: {remaining}.' )
			
			if aws_file_submit:
				if not bucket or not bucket.strip( ):
					st.info( 'Enter a Bucket.' )
				elif not key_name or not key_name.strip( ):
					st.info( 'Enter a Key.' )
				else:
					try:
						loader = AwsFileLoader( )
						documents = loader.load( bucket=bucket.strip( ), key=key_name.strip( ), aws_access_key_id=aws_access_key_id.strip( ) or None, aws_secret_access_key=aws_secret_access_key.strip( ) or None, aws_session_token=aws_session_token.strip( ) or None, region_name=region_name.strip( ) or None ) or [ ]
						
						count = _promote_loader_documents( documents, 'AwsFileLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'Bucket': bucket.strip( ), 'Key': key_name.strip( ),
									'Preview': content[ :200 ], 'Content': content,
									'Metadata': metadata, } )
						
						st.session_state[ 'aws_file_results' ] = { 'mode': 'aws_s3_file',
								'bucket': bucket.strip( ), 'key': key_name.strip( ),
								'region_name': region_name.strip( ) or '', 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} AWS S3 file document(s).' )
					
					except Exception as exc:
						st.error( 'AWS S3 File request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'aws_file_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'Bucket': result.get( 'bucket', '' ), 'Key': result.get( 'key', '' ),
						'Region': result.get( 'region_name', '' ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Bucket': item.get( 'Bucket', '' ),
									'Key': item.get( 'Key', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '##### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '##### First File Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No AWS S3 file documents returned.' )
				
				_render_fallback_raw( result )
		
		# -------- OneDrive
		if active_source == 'OneDrive':
			if onedrive_clear:
				st.session_state[ 'onedrive_results' ] = { }
				remaining = _clear_loader_documents( 'OneDriveDocLoader' )
				st.info( f'OneDrive Loader state cleared. Remaining documents: {remaining}.' )
			
			if onedrive_submit:
				if not drive_id or not drive_id.strip( ):
					st.info( 'Enter a Drive ID.' )
				else:
					try:
						object_ids: List[ str ] | None = None
						if object_ids_text and object_ids_text.strip( ):
							object_ids = [ item.strip( ) for item in object_ids_text.split( ',' ) if
									item and item.strip( ) ]
						
						loader = OneDriveDocLoader( )
						documents = loader.load( drive_id=drive_id.strip( ), folder_path=folder_path.strip( ) or None, object_ids=object_ids, auth_with_token=auth_with_token ) or [ ]
						
						count = _promote_loader_documents( documents, 'OneDriveDocLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'DriveId': drive_id.strip( ), 'Preview': content[ :200 ],
									'Content': content, 'Metadata': metadata, } )
						
						st.session_state[ 'onedrive_results' ] = { 'mode': 'onedrive',
								'drive_id': drive_id.strip( ),
								'folder_path': folder_path.strip( ) or '',
								'object_ids_count': len( object_ids or [ ] ),
								'auth_with_token': auth_with_token, 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} OneDrive document(s).' )
					
					except Exception as exc:
						st.error( 'OneDrive request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'onedrive_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'DriveId': result.get( 'drive_id', '' ),
						'FolderPath': result.get( 'folder_path', '' ),
						'ObjectIds': result.get( 'object_ids_count', 0 ),
						'AuthWithToken': result.get( 'auth_with_token', False ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'DriveId': item.get( 'DriveId', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '##### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First File Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No OneDrive documents returned.' )
				
				_render_fallback_raw( result )
		
		# -------- Google STT
		if active_source == 'Google Speech-to-Text':
			if google_speech_clear:
				st.session_state[ 'google_speech_to_text_results' ] = { }
				remaining = _clear_loader_documents( 'GoogleSpeechToTextLoader' )
				st.info( f'Google Speech-to-Text Loader state cleared. Remaining documents: '
				         f'{remaining}.' )
			
			if google_speech_submit:
				if not project_id or not project_id.strip( ):
					st.info( 'Enter a Project ID.' )
				elif not audio_file and not gcs_audio_uri.strip( ):
					st.info( 'Upload an audio file or enter a GCS Audio URI.' )
				else:
					try:
						config: Dict[ str, Any ] | None = None
						if language_code.strip( ):
							config = { 'language_code': language_code.strip( ) }
						
						if gcs_audio_uri.strip( ):
							file_path = gcs_audio_uri.strip( )
							loader = GoogleSpeechToTextLoader( )
							documents = loader.load( project_id=project_id.strip( ), file_path=file_path, config=config ) or [ ]
						else:
							with tempfile.TemporaryDirectory( ) as tmp:
								path = os.path.join( tmp, audio_file.name )
								with open( path, 'wb' ) as f:
									f.write( audio_file.read( ) )
								
								loader = GoogleSpeechToTextLoader( )
								documents = loader.load( project_id=project_id.strip( ), file_path=path, config=config ) or [ ]
						
						count = _promote_loader_documents( documents, 'GoogleSpeechToTextLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'LanguageCode': language_code.strip( ),
									'Preview': content[ :200 ], 'Content': content,
									'Metadata': metadata, } )
						
						st.session_state[ 'google_speech_to_text_results' ] = {
								'mode': 'google_speech_to_text', 'project_id': project_id.strip( ),
								'file_path': gcs_audio_uri.strip( ) or audio_file.name,
								'language_code': language_code.strip( ), 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} transcript document(s).' )
					
					except Exception as exc:
						st.error( 'Google Speech-to-Text request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'google_speech_to_text_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'ProjectId': result.get( 'project_id', '' ),
						'FilePath': result.get( 'file_path', '' ),
						'LanguageCode': result.get( 'language_code', '' ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'LanguageCode': item.get( 'LanguageCode', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '##### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '##### Transcript Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No transcript documents returned.' )
				
				_render_fallback_raw( result )
		
		# -------- AWS S3 Bucket
		if active_source == 'AWS S3 Bucket':
			if aws_bucket_clear:
				st.session_state[ 'aws_bucket_results' ] = { }
				remaining = _clear_loader_documents( 'AmazonBucketLoader' )
				st.info( f'AWS S3 Bucket Loader state cleared. Remaining documents: {remaining}.' )
			
			if aws_bucket_submit:
				if not bucket_name or not bucket_name.strip( ):
					st.info( 'Enter a Bucket.' )
				else:
					try:
						loader = AwsBucketLoader( )
						documents = loader.load( bucket=bucket_name.strip( ), prefix=prefix.strip( ) or None, aws_access_key_id=aws_access_key_id.strip( ) or None, aws_secret_access_key=aws_secret_access_key.strip( ) or None, aws_session_token=aws_session_token.strip( ) or None, region_name=region_name.strip( ) or None, endpoint_url=endpoint_url.strip( ) or None ) or [ ]
						
						count = _promote_loader_documents( documents, 'AmazonBucketLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'Bucket': bucket_name.strip( ), 'Prefix': prefix.strip( ) or '',
									'Preview': content[ :200 ], 'Content': content,
									'Metadata': metadata, } )
						
						st.session_state[ 'aws_bucket_results' ] = { 'mode': 'aws_s3_bucket',
								'bucket': bucket_name.strip( ), 'prefix': prefix.strip( ) or '',
								'region_name': region_name.strip( ) or '',
								'endpoint_url': endpoint_url.strip( ) or '', 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} AWS S3 bucket document(s).' )
					
					except Exception as exc:
						st.error( 'AWS S3 Bucket request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'aws_bucket_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'Bucket': result.get( 'bucket', '' ), 'Prefix': result.get( 'prefix', '' ),
						'Region': result.get( 'region_name', '' ),
						'EndpointUrl': result.get( 'endpoint_url', '' ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Bucket': item.get( 'Bucket', '' ),
									'Prefix': item.get( 'Prefix', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '#### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First File Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No AWS S3 bucket documents returned.' )
				
				_render_fallback_raw( result )
		
		# -------- Google Cloud Bucket
		if active_source == 'Google Cloud Bucket':
			if google_bucket_clear:
				st.session_state[ 'google_bucket_results' ] = { }
				remaining = _clear_loader_documents( 'GoogleBucketLoader' )
				st.info( f'Google Bucket Loader state cleared. Remaining documents: {remaining}.' )
			
			if google_bucket_submit:
				if not project_name or not project_name.strip( ):
					st.info( 'Enter a Project Name.' )
				elif not bucket_name or not bucket_name.strip( ):
					st.info( 'Enter a Bucket.' )
				else:
					try:
						loader = GoogleBucketLoader( )
						documents = loader.load( project_name=project_name.strip( ), bucket=bucket_name.strip( ), prefix=prefix.strip( ) or None, continue_on_failure=continue_on_failure ) or [ ]
						
						count = _promote_loader_documents( documents, 'GoogleBucketLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
									'Bucket': bucket_name.strip( ), 'Prefix': prefix.strip( ) or '',
									'Preview': content[ :200 ], 'Content': content,
									'Metadata': metadata, } )
						
						st.session_state[ 'google_bucket_results' ] = {
								'mode': 'google_cloud_bucket',
								'project_name': project_name.strip( ),
								'bucket': bucket_name.strip( ), 'prefix': prefix.strip( ) or '',
								'continue_on_failure': continue_on_failure, 'count': count,
								'items': items, }
						
						st.success( f'Loaded {count} Google bucket document(s).' )
					
					except Exception as exc:
						st.error( 'Google Cloud Bucket request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'google_bucket_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'ProjectName': result.get( 'project_name', '' ),
						'Bucket': result.get( 'bucket', '' ), 'Prefix': result.get( 'prefix', '' ),
						'ContinueOnFailure': result.get( 'continue_on_failure', False ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Bucket': item.get( 'Bucket', '' ),
									'Prefix': item.get( 'Prefix', '' ),
									'Preview': item.get( 'Preview', '' ), } for item in items ]
					
					st.markdown( '#### Results' )
					st.dataframe( table_rows, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First File Preview' )
					st.code( str( first.get( 'Content', '' ) )[ :8000 ] )
				else:
					st.info( 'No Google Cloud bucket documents returned.' )
				
				_render_fallback_raw( result )
		
		if not active_source:
			st.info( 'No retrieval results available.' )

# ==============================================================================
# GEOSPATIAL MODE
# ==============================================================================
elif mode == 'Geospatial':
	st.session_state.setdefault( 'geospatial_active_source', '' )
	st.session_state.setdefault( 'geospatial_documents', [ ] )
	st.session_state.setdefault( 'geospatial_raw_result', None )
	
	def _build_geospatial_documents( source: str, result: Any ) -> List[ Document ]:
		"""Build canonical documents from a Geospatial-mode result.

		Purpose:
			Converts heterogeneous geospatial responses into LangChain documents so every source
			writes to the shared document contract consumed by the right-side viewer.

		Args:
			source (str): Geospatial source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents representing the result payload.
		"""
		documents: List[ Document ] = [ ]
		
		def _append_document( value: Any, index: int = 1 ) -> None:
			if isinstance( value, Document ):
				metadata = dict( value.metadata or { } )
				metadata.setdefault( 'mode', 'Geospatial' )
				metadata.setdefault( 'source', source )
				documents.append( Document( page_content=value.page_content or '', metadata=metadata ) )
				return
			
			metadata: Dict[ str, Any ] = { 'mode': 'Geospatial', 'source': source, 'item': index, }
			if isinstance( value, dict ):
				content = json.dumps( normalize( value ), indent=2, ensure_ascii=False, default=str )
			elif isinstance( value, (list, tuple, set) ):
				content = json.dumps( normalize( list( value ) ), indent=2, ensure_ascii=False, default=str )
			else:
				content = str( value )
			if content.strip( ):
				documents.append( Document( page_content=content, metadata=metadata ) )
		
		if result is None:
			return documents
		if isinstance( result, list ):
			for index, item in enumerate( result, start=1 ):
				_append_document( item, index )
		else:
			_append_document( result )
		return documents
	
	def _promote_geospatial_result( source: str, result: Any ) -> List[ Document ]:
		"""Promote a source result into the shared Geospatial document state.

		Purpose:
			Writes the active geospatial result to canonical document, raw-text, and source state
			while preserving every source-specific result key and specialized renderer.

		Args:
			source (str): Geospatial source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents written to shared session state.
		"""
		documents = _build_geospatial_documents( source=source, result=result )
		st.session_state[ 'geospatial_raw_result' ] = result
		st.session_state[ 'geospatial_documents' ] = documents
		st.session_state[ 'documents' ] = documents
		st.session_state[ 'raw_documents' ] = list( documents )
		st.session_state[ 'raw_text' ] = '\n\n'.join(
			document.page_content for document in documents if
			isinstance( document.page_content, str ) and document.page_content.strip( ) )
		st.session_state[ 'processed_text' ] = ''
		st.session_state[ 'active_loader' ] = source
		return documents
	
	st.subheader( '📡 Geospatial & Weather' )
	st.divider( )
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	
	with left:
		# ----------------------------
		# ------ Expander Geocoding
		# ----------------------------
		with st.expander( label='Geocoding', icon='📍', expanded=False ):
			if 'googlegeocoding_results' not in st.session_state:
				st.session_state[ 'googlegeocoding_results' ] = { }
			
			if 'googlegeocoding_clear_request' not in st.session_state:
				st.session_state[ 'googlegeocoding_clear_request' ] = False
			
			if st.session_state.get( 'googlegeocoding_clear_request', False ):
				st.session_state[ 'googlegeocoding_mode' ] = 'forward'
				st.session_state[ 'googlegeocoding_query' ] = ''
				st.session_state[ 'googlegeocoding_latitude' ] = 38.8895
				st.session_state[ 'googlegeocoding_longitude' ] = -77.0353
				st.session_state[ 'googlegeocoding_place_id' ] = ''
				st.session_state[ 'googlegeocoding_language' ] = 'en'
				st.session_state[ 'googlegeocoding_region' ] = ''
				st.session_state[ 'googlegeocoding_result_type' ] = ''
				st.session_state[ 'googlegeocoding_location_type' ] = ''
				st.session_state[ 'googlegeocoding_api_key' ] = ''
				st.session_state[ 'googlegeocoding_timeout' ] = 10
				st.session_state[ 'googlegeocoding_results' ] = { }
				st.session_state[ 'googlegeocoding_clear_request' ] = False
			
			def _clear_googlegeocoding_state( ) -> None:
				st.session_state[ 'googlegeocoding_clear_request' ] = True
			
			googlegeocoding_mode = st.selectbox( 'Mode', options=[ 'forward', 'reverse',
					'place' ], index=[ 'forward', 'reverse',
					'place' ].index( st.session_state.get( 'googlegeocoding_mode', 'forward' ) ), key='googlegeocoding_mode', help='forward = address search; reverse = lat/lng to address; place = place_id '
			                                                                                                                       'lookup.' )
			
			googlegeocoding_query = st.text_area( 'Address Query', height=80, key='googlegeocoding_query', placeholder=(
				'Examples:\n'
				'1600 Amphitheatre Parkway, Mountain '
				'View, CA\n'
				'Arlington, VA\n'
				'10 Downing Street, London'), disabled=(googlegeocoding_mode != 'forward') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				googlegeocoding_latitude = st.number_input( 'Latitude', min_value=-90.0, max_value=90.0, value=float( st.session_state.get( 'googlegeocoding_latitude', 38.8895 ) ), step=0.0001, format='%.6f', key='googlegeocoding_latitude', disabled=(
							googlegeocoding_mode != 'reverse') )
			
			with c2:
				googlegeocoding_longitude = st.number_input( 'Longitude', min_value=-180.0, max_value=180.0, value=float( st.session_state.get( 'googlegeocoding_longitude', -77.0353 ) ), step=0.0001, format='%.6f', key='googlegeocoding_longitude', disabled=(
							googlegeocoding_mode != 'reverse') )
			
			googlegeocoding_place_id = st.text_input( 'Place ID', value=st.session_state.get( 'googlegeocoding_place_id', '' ), key='googlegeocoding_place_id', placeholder='ChIJ2eUgeAK6j4ARbn5u_wAGqWA', disabled=(
						googlegeocoding_mode != 'place') )
			
			c3, c4 = st.columns( 2 )
			with c3:
				googlegeocoding_language = st.text_input( 'Language', value=st.session_state.get( 'googlegeocoding_language', 'en' ), key='googlegeocoding_language', placeholder='en' )
			
			with c4:
				googlegeocoding_region = st.text_input( 'Region Bias', value=st.session_state.get( 'googlegeocoding_region', '' ), key='googlegeocoding_region', placeholder='us', disabled=(
							googlegeocoding_mode == 'reverse') )
			
			c5, c6 = st.columns( 2 )
			with c5:
				googlegeocoding_result_type = st.text_input( 'Result Type', value=st.session_state.get( 'googlegeocoding_result_type', '' ), key='googlegeocoding_result_type', placeholder='street_address|premise', disabled=(
							googlegeocoding_mode != 'reverse') )
			
			with c6:
				googlegeocoding_location_type = st.text_input( 'Location Type', value=st.session_state.get( 'googlegeocoding_location_type', '' ), key='googlegeocoding_location_type', placeholder='ROOFTOP|GEOMETRIC_CENTER', disabled=(
							googlegeocoding_mode != 'reverse') )
			
			c7, c8 = st.columns( 2 )
			with c7:
				googlegeocoding_api_key = st.text_input( 'API Key', value='', type='password', key='googlegeocoding_api_key', placeholder='Uses GOOGLE_API_KEY when left blank.' )
			
			with c8:
				googlegeocoding_timeout = st.number_input( 'Timeout', min_value=1, max_value=60, value=int( st.session_state.get( 'googlegeocoding_timeout', 10 ) ), step=1, key='googlegeocoding_timeout' )
			
			st.caption( 'Google Geocoding requires billing plus a Google API key. '
			            'Result filters apply to reverse geocoding only.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				googlegeocoding_submit = st.button( 'Submit', key='googlegeocoding_submit', use_container_width=True, width='stretch' )
			
			with b2:
				st.button( 'Clear', key='googlegeocoding_clear', on_click=_clear_googlegeocoding_state, use_container_width=True, width='stretch' )
			
			if googlegeocoding_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Geocoding'
		
		# ----------------------------
		# ------ Expander Google Maps
		# ----------------------------
		with st.expander( label='Google Maps', icon='🗺️', expanded=False ):
			GOOGLEMAPS_MODES = [ 'geocode_location', 'geocode_coordinates', 'validate_address',
					'request_directions' ]
			
			GOOGLEMAPS_TRAVEL_MODES = [ 'driving', 'walking', 'bicycling', 'transit' ]
			
			def _clear_googlemaps_state( ) -> None:
				st.session_state[ 'googlemaps_clear_request' ] = True
			
			def _split_googlemaps_address_lines( value: str ) -> List[ str ]:
				lines = [ str( line or '' ).strip( ) for line in str( value or '' ).splitlines( ) if
						str( line or '' ).strip( ) ]
				
				return lines
			
			if 'googlemaps_results' not in st.session_state:
				st.session_state[ 'googlemaps_results' ] = { }
			
			if 'googlemaps_clear_request' not in st.session_state:
				st.session_state[ 'googlemaps_clear_request' ] = False
			
			if st.session_state.get( 'googlemaps_mode', 'geocode_location' ) not in GOOGLEMAPS_MODES:
				st.session_state[ 'googlemaps_mode' ] = 'geocode_location'
			
			if st.session_state.get( 'googlemaps_travel_mode', 'driving' ) not in GOOGLEMAPS_TRAVEL_MODES:
				st.session_state[ 'googlemaps_travel_mode' ] = 'driving'
			
			if 'googlemaps_query' not in st.session_state:
				st.session_state[ 'googlemaps_query' ] = ''
			
			if 'googlemaps_radius' not in st.session_state:
				st.session_state[ 'googlemaps_radius' ] = 5000
			
			if 'googlemaps_latitude' not in st.session_state:
				st.session_state[ 'googlemaps_latitude' ] = 38.8895
			
			if 'googlemaps_longitude' not in st.session_state:
				st.session_state[ 'googlemaps_longitude' ] = -77.0353
			
			if 'googlemaps_address_lines' not in st.session_state:
				st.session_state[ 'googlemaps_address_lines' ] = ''
			
			if 'googlemaps_origin' not in st.session_state:
				st.session_state[ 'googlemaps_origin' ] = ''
			
			if 'googlemaps_destination' not in st.session_state:
				st.session_state[ 'googlemaps_destination' ] = ''
			
			if st.session_state.get( 'googlemaps_clear_request', False ):
				st.session_state[ 'googlemaps_mode' ] = 'geocode_location'
				st.session_state[ 'googlemaps_query' ] = ''
				st.session_state[ 'googlemaps_radius' ] = 5000
				st.session_state[ 'googlemaps_latitude' ] = 38.8895
				st.session_state[ 'googlemaps_longitude' ] = -77.0353
				st.session_state[ 'googlemaps_address_lines' ] = ''
				st.session_state[ 'googlemaps_origin' ] = ''
				st.session_state[ 'googlemaps_destination' ] = ''
				st.session_state[ 'googlemaps_travel_mode' ] = 'driving'
				st.session_state[ 'googlemaps_results' ] = { }
				st.session_state[ 'googlemaps_clear_request' ] = False
			
			googlemaps_mode = st.selectbox( 'Mode', options=GOOGLEMAPS_MODES, index=GOOGLEMAPS_MODES.index( st.session_state.get( 'googlemaps_mode', 'geocode_location' ) ), key='googlemaps_mode', help='Select the Google Maps wrapper operation to '
			                                                                                                                                                                                             'run.' )
			
			googlemaps_query = st.text_area( 'Address', value=st.session_state.get( 'googlemaps_query', '' ), height=70, key='googlemaps_query', disabled=(
						googlemaps_mode != 'geocode_location'), placeholder='1600 Pennsylvania Ave NW, Washington, DC' )
			
			coord_c1, coord_c2 = st.columns( 2 )
			with coord_c1:
				googlemaps_latitude = st.number_input( 'Latitude', value=float( st.session_state.get( 'googlemaps_latitude', 38.8895 ) ), step=0.0001, format='%.6f', key='googlemaps_latitude', disabled=(
							googlemaps_mode != 'geocode_coordinates') )
			
			with coord_c2:
				googlemaps_longitude = st.number_input( 'Longitude', value=float( st.session_state.get( 'googlemaps_longitude', -77.0353 ) ), step=0.0001, format='%.6f', key='googlemaps_longitude', disabled=(
							googlemaps_mode != 'geocode_coordinates') )
			
			googlemaps_address_lines = st.text_area( 'Address Lines', value=st.session_state.get( 'googlemaps_address_lines', '' ), height=90, key='googlemaps_address_lines', disabled=(
						googlemaps_mode != 'validate_address'), placeholder=(
				'1600 Pennsylvania Ave NW\n'
				'Washington, DC 20500') )
			
			route_c1, route_c2 = st.columns( 2 )
			with route_c1:
				googlemaps_origin = st.text_input( 'Origin', value=st.session_state.get( 'googlemaps_origin', '' ), key='googlemaps_origin', disabled=(
							googlemaps_mode != 'request_directions'), placeholder='Arlington, VA' )
			
			with route_c2:
				googlemaps_destination = st.text_input( 'Destination', value=st.session_state.get( 'googlemaps_destination', '' ), key='googlemaps_destination', disabled=(
							googlemaps_mode != 'request_directions'), placeholder='Washington, DC' )
			
			opt_c1, opt_c2 = st.columns( 2 )
			with opt_c1:
				googlemaps_travel_mode = st.selectbox( 'Travel Mode', options=GOOGLEMAPS_TRAVEL_MODES, index=GOOGLEMAPS_TRAVEL_MODES.index( st.session_state.get( 'googlemaps_travel_mode', 'driving' ) ), key='googlemaps_travel_mode', disabled=(
							googlemaps_mode != 'request_directions') )
			
			with opt_c2:
				googlemaps_radius = st.number_input( 'Radius (meters)', min_value=1, max_value=50000, value=int( st.session_state.get( 'googlemaps_radius', 5000 ) ), step=100, key='googlemaps_radius', help='Preserved legacy control. The current GoogleMaps wrapper does not '
				                                                                                                                                                                                              'consume radius.' )
			
			st.caption( 'Google Maps uses GOOGLE_API_KEY from config. This section now '
			            'exposes '
			            'forward geocoding, reverse geocoding, address validation, '
			            'and directions.' )
			
			m1, m2 = st.columns( 2 )
			with m1:
				googlemaps_submit = st.button( 'Submit', key='googlemaps_submit', use_container_width=True, width='stretch' )
			
			with m2:
				st.button( 'Clear', key='googlemaps_clear', on_click=_clear_googlemaps_state, use_container_width=True, width='stretch' )
			
			if googlemaps_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Google Maps'
		
		# ----------------------------
		# ------ Expander Google Weather
		# ----------------------------
		with st.expander( label='Google Weather', icon='🌤️', expanded=False ):
			GOOGLEWEATHER_MODES = [ 'current', 'hourly_forecast', 'daily_forecast',
					'hourly_history', 'alerts' ]
			
			GOOGLEWEATHER_UNITS = [ 'METRIC', 'IMPERIAL' ]
			
			def _clear_googleweather_state( ) -> None:
				st.session_state[ 'googleweather_clear_request' ] = True
			
			if 'googleweather_results' not in st.session_state:
				st.session_state[ 'googleweather_results' ] = { }
			
			if 'googleweather_clear_request' not in st.session_state:
				st.session_state[ 'googleweather_clear_request' ] = False
			
			if st.session_state.get( 'googleweather_mode', 'current' ) not in GOOGLEWEATHER_MODES:
				st.session_state[ 'googleweather_mode' ] = 'current'
			
			if st.session_state.get( 'googleweather_units', 'METRIC' ) not in GOOGLEWEATHER_UNITS:
				st.session_state[ 'googleweather_units' ] = 'METRIC'
			
			if 'googleweather_location' not in st.session_state:
				st.session_state[ 'googleweather_location' ] = ''
			
			if 'googleweather_language' not in st.session_state:
				st.session_state[ 'googleweather_language' ] = 'en'
			
			if 'googleweather_hours' not in st.session_state:
				st.session_state[ 'googleweather_hours' ] = 24
			
			if 'googleweather_history_hours' not in st.session_state:
				st.session_state[ 'googleweather_history_hours' ] = 24
			
			if 'googleweather_days' not in st.session_state:
				st.session_state[ 'googleweather_days' ] = 5
			
			if 'googleweather_timeout' not in st.session_state:
				st.session_state[ 'googleweather_timeout' ] = 10
			
			if st.session_state.get( 'googleweather_clear_request', False ):
				st.session_state[ 'googleweather_location' ] = ''
				st.session_state[ 'googleweather_mode' ] = 'current'
				st.session_state[ 'googleweather_units' ] = 'METRIC'
				st.session_state[ 'googleweather_language' ] = 'en'
				st.session_state[ 'googleweather_hours' ] = 24
				st.session_state[ 'googleweather_history_hours' ] = 24
				st.session_state[ 'googleweather_days' ] = 5
				st.session_state[ 'googleweather_timeout' ] = 10
				st.session_state[ 'googleweather_results' ] = { }
				st.session_state[ 'googleweather_clear_request' ] = False
			
			gw_location = st.text_area( 'Location', height=70, key='googleweather_location', placeholder=(
				'Examples:\n'
				'Washington, DC\n'
				'1600 Pennsylvania Ave NW, Washington, DC\n'
				'Arlington, VA'), )
			
			c1, c2 = st.columns( 2 )
			with c1:
				gw_mode = st.selectbox( 'Mode', options=GOOGLEWEATHER_MODES, index=GOOGLEWEATHER_MODES.index( st.session_state.get( 'googleweather_mode', 'current' ) ), key='googleweather_mode', help='Choose the Google Weather API operation to run.' )
			
			with c2:
				gw_units = st.selectbox( 'Units', options=GOOGLEWEATHER_UNITS, index=GOOGLEWEATHER_UNITS.index( st.session_state.get( 'googleweather_units', 'METRIC' ) ), key='googleweather_units' )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				gw_language = st.text_input( 'Language', value=st.session_state.get( 'googleweather_language', 'en' ), key='googleweather_language', placeholder='en' )
			
			with c4:
				gw_hours = st.number_input( 'Forecast Hours', min_value=1, max_value=240, value=int( st.session_state.get( 'googleweather_hours', 24 ) ), step=1, key='googleweather_hours', disabled=(
							gw_mode != 'hourly_forecast') )
			
			with c5:
				gw_days = st.number_input( 'Forecast Days', min_value=1, max_value=10, value=int( st.session_state.get( 'googleweather_days', 5 ) ), step=1, key='googleweather_days', disabled=(
							gw_mode != 'daily_forecast') )
			
			h1, h2 = st.columns( 2 )
			with h1:
				gw_history_hours = st.number_input( 'History Hours', min_value=1, max_value=24, value=int( st.session_state.get( 'googleweather_history_hours', 24 ) ), step=1, key='googleweather_history_hours', disabled=(
							gw_mode != 'hourly_history'), help='Google Weather hourly history supports up to 24 hours.' )
			
			with h2:
				gw_timeout = st.number_input( 'Timeout', min_value=1, max_value=60, value=int( st.session_state.get( 'googleweather_timeout', 10 ) ), step=1, key='googleweather_timeout' )
			
			st.caption( 'Required key: GOOGLE_WEATHER_API_KEY. '
			            'Weather API must be enabled in your Google Cloud project.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				gw_submit = st.button( 'Submit', key='googleweather_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='googleweather_clear', on_click=_clear_googleweather_state, width='stretch' )
			
			if gw_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Google Weather'
		
		# ----------------------------
		# ------ Expander Open Weather
		# ----------------------------
		with st.expander( label='Open Weather', icon='🌦️', expanded=False ):
			if 'openweather_results' not in st.session_state:
				st.session_state[ 'openweather_results' ] = { }
			
			if 'openweather_clear_request' not in st.session_state:
				st.session_state[ 'openweather_clear_request' ] = False
			
			if st.session_state.get( 'openweather_clear_request', False ):
				st.session_state[ 'openweather_location' ] = ''
				st.session_state[ 'openweather_mode' ] = 'current'
				st.session_state[ 'openweather_timezone' ] = 'auto'
				st.session_state[ 'openweather_forecast_days' ] = 7
				st.session_state[ 'openweather_past_days' ] = 0
				st.session_state[ 'openweather_count' ] = 10
				st.session_state[ 'openweather_results' ] = { }
				st.session_state[ 'openweather_clear_request' ] = False
			
			def _clear_openweather_state( ) -> None:
				st.session_state[ 'openweather_clear_request' ] = True
			
			openweather_location = st.text_area( 'Location', height=80, key='openweather_location', placeholder=(
				'Examples:\n'
				'Arlington, VA\n'
				'Paris, France\n'
				'90210') )
			
			openweather_mode = st.selectbox( 'Mode', options=[ 'current', 'hourly',
					'daily' ], index=[ 'current', 'hourly',
					'daily' ].index( st.session_state.get( 'openweather_mode', 'current' ) ), key='openweather_mode' )
			
			cfg_c1, cfg_c2 = st.columns( 2 )
			with cfg_c1:
				openweather_forecast_days = st.number_input( 'Forecast Days', min_value=1, max_value=16, value=int( st.session_state.get( 'openweather_forecast_days', 7 ) ), step=1, key='openweather_forecast_days', disabled=(
							openweather_mode == 'current') )
			
			with cfg_c2:
				openweather_past_days = st.number_input( 'Past Days', min_value=0, max_value=92, value=int( st.session_state.get( 'openweather_past_days', 0 ) ), step=1, key='openweather_past_days' )
			
			meta_c1, meta_c2 = st.columns( 2 )
			with meta_c1:
				openweather_timezone = st.text_input( 'Timezone', value=st.session_state.get( 'openweather_timezone', 'auto' ), key='openweather_timezone', placeholder='auto' )
			
			with meta_c2:
				openweather_count = st.number_input( 'Geocode Matches', min_value=1, max_value=20, value=int( st.session_state.get( 'openweather_count', 10 ) ), step=1, key='openweather_count' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				openweather_submit = st.button( 'Submit', key='openweather_submit', width='stretch' )
			
			with btn_c2:
				openweather_clear = st.button( 'Clear', key='openweather_clear', on_click=_clear_openweather_state, width='stretch' )
			
			if openweather_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Open Weather'
		
		# ----------------------------
		# ------ Expander Historical Weather
		# ----------------------------
		with st.expander( label='Historical Weather', icon='📈', expanded=False ):
			if 'historicalweather_results' not in st.session_state:
				st.session_state[ 'historicalweather_results' ] = { }
			
			if 'historicalweather_clear_request' not in st.session_state:
				st.session_state[ 'historicalweather_clear_request' ] = False
			
			if st.session_state.get( 'historicalweather_clear_request', False ):
				st.session_state[ 'historicalweather_location' ] = ''
				st.session_state[
					'historicalweather_date' ] = dt.date.today( ) - dt.timedelta( days=1 )
				st.session_state[ 'historicalweather_timezone' ] = 'auto'
				st.session_state[ 'historicalweather_count' ] = 10
				st.session_state[ 'historicalweather_results' ] = { }
				st.session_state[ 'historicalweather_clear_request' ] = False
			
			def _clear_historicalweather_state( ) -> None:
				st.session_state[ 'historicalweather_clear_request' ] = True
			
			historicalweather_location = st.text_area( 'Location', height=80, key='historicalweather_location', placeholder=(
				'Examples:\n'
				'Arlington, VA\n'
				'Tokyo, Japan\n'
				'90210') )
			
			date_c1, date_c2 = st.columns( 2 )
			with date_c1:
				historicalweather_date = st.date_input( 'Date', value=st.session_state.get( 'historicalweather_date', dt.date.today( ) - dt.timedelta( days=1 ) ), key='historicalweather_date' )
			
			with date_c2:
				historicalweather_count = st.number_input( 'Geocode Matches', min_value=1, max_value=20, value=int( st.session_state.get( 'historicalweather_count', 10 ) ), step=1, key='historicalweather_count' )
			
			historicalweather_timezone = st.text_input( 'Timezone', value=st.session_state.get( 'historicalweather_timezone', 'auto' ), key='historicalweather_timezone', placeholder='auto' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				historicalweather_submit = st.button( 'Submit', key='historicalweather_submit', width='stretch' )
			
			with btn_c2:
				historicalweather_clear = st.button( 'Clear', key='historicalweather_clear', on_click=_clear_historicalweather_state, width='stretch' )
			
			if historicalweather_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Historical Weather'
		
		# ----------------------------
		# ------ Expander USGS Earthquakes
		# ----------------------------
		with st.expander( label='USGS Earthquakes', icon='🌎', expanded=False ):
			USGSEARTHQUAKES_MODES = [ 'feed', 'search' ]
			
			USGSEARTHQUAKES_FEEDS = [ 'all_hour.geojson', 'all_day.geojson', 'all_week.geojson',
					'all_month.geojson', 'significant_hour.geojson', 'significant_day.geojson',
					'significant_week.geojson', 'significant_month.geojson', '4.5_hour.geojson',
					'4.5_day.geojson', '4.5_week.geojson', '4.5_month.geojson', '2.5_hour.geojson',
					'2.5_day.geojson', '2.5_week.geojson', '2.5_month.geojson', '1.0_hour.geojson',
					'1.0_day.geojson', '1.0_week.geojson', '1.0_month.geojson' ]
			
			USGSEARTHQUAKES_ORDER_BY = [ 'time', 'time-asc', 'magnitude', 'magnitude-asc' ]
			
			USGSEARTHQUAKES_EVENT_TYPES = [ 'earthquake', 'quarry blast', 'explosion', 'ice quake',
					'landslide', 'mining explosion', 'other event', 'rock burst',
					'volcanic eruption' ]
			
			def _clear_usgsearthquakes_state( ) -> None:
				st.session_state[ 'usgsearthquakes_clear_request' ] = True
			
			def _coerce_optional_float( name: str, value: object, min_value: float, max_value: float ) -> float | None:
				text = str( value or '' ).strip( )
				
				if not text:
					return None
				
				number = float( text )
				
				if number < min_value or number > max_value:
					raise ValueError( f'{name} must be between {min_value:g} and {max_value:g}.' )
				
				return number
			
			if 'usgsearthquakes_results' not in st.session_state:
				st.session_state[ 'usgsearthquakes_results' ] = { }
			
			if 'usgsearthquakes_clear_request' not in st.session_state:
				st.session_state[ 'usgsearthquakes_clear_request' ] = False
			
			if st.session_state.get( 'usgsearthquakes_mode', 'feed' ) not in USGSEARTHQUAKES_MODES:
				st.session_state[ 'usgsearthquakes_mode' ] = 'feed'
			
			if st.session_state.get( 'usgsearthquakes_feed', 'all_day.geojson' ) not in USGSEARTHQUAKES_FEEDS:
				st.session_state[ 'usgsearthquakes_feed' ] = 'all_day.geojson'
			
			if st.session_state.get( 'usgsearthquakes_order_by', 'time' ) not in USGSEARTHQUAKES_ORDER_BY:
				st.session_state[ 'usgsearthquakes_order_by' ] = 'time'
			
			if st.session_state.get( 'usgsearthquakes_event_type', 'earthquake' ) not in USGSEARTHQUAKES_EVENT_TYPES:
				st.session_state[ 'usgsearthquakes_event_type' ] = 'earthquake'
			
			if 'usgsearthquakes_start_date' not in st.session_state:
				st.session_state[ 'usgsearthquakes_start_date' ] = (
						dt.date.today( ) - dt.timedelta( days=7 ))
			
			if 'usgsearthquakes_end_date' not in st.session_state:
				st.session_state[ 'usgsearthquakes_end_date' ] = dt.date.today( )
			
			if 'usgsearthquakes_min_magnitude' not in st.session_state:
				st.session_state[ 'usgsearthquakes_min_magnitude' ] = 1.0
			
			if 'usgsearthquakes_max_magnitude' not in st.session_state:
				st.session_state[ 'usgsearthquakes_max_magnitude' ] = 10.0
			
			if 'usgsearthquakes_limit' not in st.session_state:
				st.session_state[ 'usgsearthquakes_limit' ] = 25
			
			if 'usgsearthquakes_latitude' not in st.session_state:
				st.session_state[ 'usgsearthquakes_latitude' ] = ''
			
			if 'usgsearthquakes_longitude' not in st.session_state:
				st.session_state[ 'usgsearthquakes_longitude' ] = ''
			
			if 'usgsearthquakes_max_radius_km' not in st.session_state:
				st.session_state[ 'usgsearthquakes_max_radius_km' ] = ''
			
			if 'usgsearthquakes_timeout' not in st.session_state:
				st.session_state[ 'usgsearthquakes_timeout' ] = 20
			
			if st.session_state.get( 'usgsearthquakes_clear_request', False ):
				st.session_state[ 'usgsearthquakes_mode' ] = 'feed'
				st.session_state[ 'usgsearthquakes_feed' ] = 'all_day.geojson'
				st.session_state[ 'usgsearthquakes_start_date' ] = (
						dt.date.today( ) - dt.timedelta( days=7 ))
				st.session_state[ 'usgsearthquakes_end_date' ] = dt.date.today( )
				st.session_state[ 'usgsearthquakes_min_magnitude' ] = 1.0
				st.session_state[ 'usgsearthquakes_max_magnitude' ] = 10.0
				st.session_state[ 'usgsearthquakes_limit' ] = 25
				st.session_state[ 'usgsearthquakes_order_by' ] = 'time'
				st.session_state[ 'usgsearthquakes_event_type' ] = 'earthquake'
				st.session_state[ 'usgsearthquakes_latitude' ] = ''
				st.session_state[ 'usgsearthquakes_longitude' ] = ''
				st.session_state[ 'usgsearthquakes_max_radius_km' ] = ''
				st.session_state[ 'usgsearthquakes_timeout' ] = 20
				st.session_state[ 'usgsearthquakes_results' ] = { }
				st.session_state[ 'usgsearthquakes_clear_request' ] = False
			
			usgseq_mode = st.selectbox( 'Mode', options=USGSEARTHQUAKES_MODES, index=USGSEARTHQUAKES_MODES.index( st.session_state.get( 'usgsearthquakes_mode', 'feed' ) ), key='usgsearthquakes_mode' )
			
			usgseq_feed = st.selectbox( 'Feed', options=USGSEARTHQUAKES_FEEDS, index=USGSEARTHQUAKES_FEEDS.index( st.session_state.get( 'usgsearthquakes_feed', 'all_day.geojson' ) ), key='usgsearthquakes_feed', disabled=(
						usgseq_mode != 'feed') )
			
			date_c1, date_c2 = st.columns( 2 )
			with date_c1:
				usgseq_start_date = st.date_input( 'Start Date', value=st.session_state.get( 'usgsearthquakes_start_date', dt.date.today( ) - dt.timedelta( days=7 ) ), key='usgsearthquakes_start_date', disabled=(
							usgseq_mode != 'search') )
			
			with date_c2:
				usgseq_end_date = st.date_input( 'End Date', value=st.session_state.get( 'usgsearthquakes_end_date', dt.date.today( ) ), key='usgsearthquakes_end_date', disabled=(
							usgseq_mode != 'search') )
			
			mag_c1, mag_c2 = st.columns( 2 )
			with mag_c1:
				usgseq_min_magnitude = st.number_input( 'Min Magnitude', min_value=0.0, max_value=10.0, value=float( st.session_state.get( 'usgsearthquakes_min_magnitude', 1.0 ) ), step=0.1, key='usgsearthquakes_min_magnitude', disabled=(
							usgseq_mode != 'search') )
			
			with mag_c2:
				usgseq_max_magnitude = st.number_input( 'Max Magnitude', min_value=0.0, max_value=10.0, value=float( st.session_state.get( 'usgsearthquakes_max_magnitude', 10.0 ) ), step=0.1, key='usgsearthquakes_max_magnitude', disabled=(
							usgseq_mode != 'search') )
			
			opt_c1, opt_c2 = st.columns( 2 )
			with opt_c1:
				usgseq_limit = st.number_input( 'Limit', min_value=1, max_value=20000, value=int( st.session_state.get( 'usgsearthquakes_limit', 25 ) ), step=1, key='usgsearthquakes_limit', disabled=(
							usgseq_mode != 'search'), help='USGS Earthquake Catalog supports limit values through 20000.' )
			
			with opt_c2:
				usgseq_order_by = st.selectbox( 'Order By', options=USGSEARTHQUAKES_ORDER_BY, index=USGSEARTHQUAKES_ORDER_BY.index( st.session_state.get( 'usgsearthquakes_order_by', 'time' ) ), key='usgsearthquakes_order_by', disabled=(
							usgseq_mode != 'search') )
			
			usgseq_event_type = st.selectbox( 'Event Type', options=USGSEARTHQUAKES_EVENT_TYPES, index=USGSEARTHQUAKES_EVENT_TYPES.index( st.session_state.get( 'usgsearthquakes_event_type', 'earthquake' ) ), key='usgsearthquakes_event_type', disabled=(
						usgseq_mode != 'search') )
			
			coord_c1, coord_c2 = st.columns( 2 )
			
			with coord_c1:
				usgseq_latitude = st.text_input( 'Latitude (optional)', value=st.session_state.get( 'usgsearthquakes_latitude', '' ), key='usgsearthquakes_latitude', disabled=(
							usgseq_mode != 'search') )
			
			with coord_c2:
				usgseq_longitude = st.text_input( 'Longitude (optional)', value=st.session_state.get( 'usgsearthquakes_longitude', '' ), key='usgsearthquakes_longitude', disabled=(
							usgseq_mode != 'search') )
			
			usgseq_max_radius_km = st.text_input( 'Max Radius KM (optional)', value=st.session_state.get( 'usgsearthquakes_max_radius_km', '' ), key='usgsearthquakes_max_radius_km', disabled=(
						usgseq_mode != 'search') )
			
			usgseq_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'usgsearthquakes_timeout', 5 ) ), step=1, key='usgsearthquakes_timeout' )
			
			st.caption( 'Feed mode is best for quick display. Search mode supports date, '
			            'magnitude, event type, and optional radial filtering.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				usgseq_submit = st.button( 'Submit', key='usgsearthquakes_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='usgsearthquakes_clear', on_click=_clear_usgsearthquakes_state, width='stretch' )
			
			if usgseq_submit:
				st.session_state[ 'geospatial_active_source' ] = 'USGS Earthquakes'
		
		# ----------------------------
		# ------ Expander NASA Earth Observatory
		# ----------------------------
		with st.expander( label='NASA Earth Observatory', icon='🛰️', expanded=False ):
			if 'earthobservatory_results' not in st.session_state:
				st.session_state[ 'earthobservatory_results' ] = { }
			
			if 'earthobservatory_clear_request' not in st.session_state:
				st.session_state[ 'earthobservatory_clear_request' ] = False
			
			if st.session_state.get( 'earthobservatory_clear_request', False ):
				st.session_state[ 'earthobservatory_mode' ] = 'events'
				st.session_state[ 'earthobservatory_status' ] = 'open'
				st.session_state[ 'earthobservatory_category' ] = ''
				st.session_state[ 'earthobservatory_source' ] = ''
				st.session_state[ 'earthobservatory_limit' ] = 20
				st.session_state[ 'earthobservatory_days' ] = 30
				st.session_state[ 'earthobservatory_start_date' ] = ''
				st.session_state[ 'earthobservatory_end_date' ] = ''
				st.session_state[ 'earthobservatory_timeout' ] = 20
				st.session_state[ 'earthobservatory_results' ] = { }
				st.session_state[ 'earthobservatory_clear_request' ] = False
			
			def _clear_earthobservatory_state( ) -> None:
				st.session_state[ 'earthobservatory_clear_request' ] = True
			
			earth_mode = st.selectbox( 'Mode', options=[ 'events', 'categories', 'sources',
					'layers' ], index=[ 'events', 'categories', 'sources',
					'layers' ].index( st.session_state.get( 'earthobservatory_mode', 'events' ) ), key='earthobservatory_mode', help='Choose the current documented EONET v3 endpoint.' )
			
			earth_status = st.selectbox( 'Status', options=[ 'open', 'closed', 'all' ], index=[
					'open', 'closed',
					'all' ].index( st.session_state.get( 'earthobservatory_status', 'open' ) ), key='earthobservatory_status', disabled=(
						earth_mode != 'events') )
			
			earth_category = st.text_input( 'Category', value=st.session_state.get( 'earthobservatory_category', '' ), key='earthobservatory_category', placeholder='Examples: wildfires, severe storms, volcanoes ', help='Used for events filtering and layers category path.', disabled=(
						earth_mode not in [ 'events', 'layers' ]) )
			
			earth_source = st.text_input( 'Source', value=st.session_state.get( 'earthobservatory_source', '' ), key='earthobservatory_source', placeholder=(
				'Examples: InciWeb, InciWeb, EO'), disabled=(earth_mode != 'events') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				earth_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'earthobservatory_limit', 20 ) ), step=1, key='earthobservatory_limit', disabled=(
							earth_mode != 'events') )
			
			with c2:
				earth_days = st.number_input( 'Days', min_value=1, max_value=3650, value=int( st.session_state.get( 'earthobservatory_days', 30 ) ), step=1, key='earthobservatory_days', disabled=(
							earth_mode != 'events') )
			
			d1, d2 = st.columns( 2 )
			with d1:
				earth_start_date = st.text_input( 'Start Date', value=st.session_state.get( 'earthobservatory_start_date', '' ), key='earthobservatory_start_date', placeholder='2026-03-01', disabled=(
							earth_mode != 'events') )
			
			with d2:
				earth_end_date = st.text_input( 'End Date', value=st.session_state.get( 'earthobservatory_end_date', '' ), key='earthobservatory_end_date', placeholder='2026-03-15', disabled=(
							earth_mode != 'events') )
			
			earth_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'earthobservatory_timeout', 20 ) ), step=1, key='earthobservatory_timeout' )
			
			st.caption( 'Examples: use events + category=wildfires, status=open; '
			            'use sources to list event source providers; '
			            'use layers + category=wildfires to inspect imagery layers.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				earth_submit = st.button( 'Submit', key='earthobservatory_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='earthobservatory_clear', on_click=_clear_earthobservatory_state, width='stretch' )
			
			if earth_submit:
				st.session_state[ 'geospatial_active_source' ] = 'NASA Earth Observatory'
		
		# ----------------------------
		# ------ Expander The National Map
		# ----------------------------
		with st.expander( label='The National Map', icon='🗺️', expanded=False ):
			USGSTNM_MODES = [ 'products', 'datasets' ]
			
			USGSTNM_PRODUCT_FORMATS = [ 'GeoTIFF', 'IMG', 'LAS', 'LAZ', 'GeoPackage', 'GeoJSON',
					'Shapefile', 'FileGDB', 'PDF', 'KMZ', 'TXT', 'CSV', 'JPEG', 'PNG', 'XML',
					'ZIP' ]
			
			def _clear_usgstnm_state( ) -> None:
				st.session_state[ 'usgstnm_clear_request' ] = True
			
			def _validate_usgstnm_bbox( value: str ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					return ''
				
				parts = [ item.strip( ) for item in text.split( ',' ) if item.strip( ) ]
				
				if len( parts ) != 4:
					raise ValueError( 'Bounding Box must contain four comma-separated values: '
					                  'minx,miny,maxx,maxy.' )
				
				minx, miny, maxx, maxy = [ float( item ) for item in parts ]
				
				if minx < -180 or maxx > 180:
					raise ValueError( 'Longitude values must be within -180 and 180.' )
				
				if miny < -90 or maxy > 90:
					raise ValueError( 'Latitude values must be within -90 and 90.' )
				
				if minx >= maxx:
					raise ValueError( 'Minimum longitude must be less than maximum longitude.' )
				
				if miny >= maxy:
					raise ValueError( 'Minimum latitude must be less than maximum latitude.' )
				
				return f'{minx:g},{miny:g},{maxx:g},{maxy:g}'
			
			if 'usgstnm_results' not in st.session_state:
				st.session_state[ 'usgstnm_results' ] = { }
			
			if 'usgstnm_clear_request' not in st.session_state:
				st.session_state[ 'usgstnm_clear_request' ] = False
			
			if st.session_state.get( 'usgstnm_mode', 'products' ) not in USGSTNM_MODES:
				st.session_state[ 'usgstnm_mode' ] = 'products'
			
			if 'usgstnm_dataset' not in st.session_state:
				st.session_state[ 'usgstnm_dataset' ] = ''
			
			if 'usgstnm_q' not in st.session_state:
				st.session_state[ 'usgstnm_q' ] = ''
			
			if 'usgstnm_bbox' not in st.session_state:
				st.session_state[ 'usgstnm_bbox' ] = ''
			
			if 'usgstnm_prod_formats' not in st.session_state:
				st.session_state[ 'usgstnm_prod_formats' ] = [ ]
			
			if isinstance( st.session_state.get( 'usgstnm_prod_formats' ), str ):
				raw_formats = str( st.session_state.get( 'usgstnm_prod_formats', '' ) )
				st.session_state[ 'usgstnm_prod_formats' ] = [ item.strip( ) for item in
						raw_formats.split( ',' ) if item.strip( ) in USGSTNM_PRODUCT_FORMATS ]
			
			if 'usgstnm_max_items' not in st.session_state:
				st.session_state[ 'usgstnm_max_items' ] = 25
			
			if 'usgstnm_offset' not in st.session_state:
				st.session_state[ 'usgstnm_offset' ] = 0
			
			if 'usgstnm_timeout' not in st.session_state:
				st.session_state[ 'usgstnm_timeout' ] = 20
			
			if st.session_state.get( 'usgstnm_clear_request', False ):
				st.session_state[ 'usgstnm_mode' ] = 'products'
				st.session_state[ 'usgstnm_dataset' ] = ''
				st.session_state[ 'usgstnm_q' ] = ''
				st.session_state[ 'usgstnm_bbox' ] = ''
				st.session_state[ 'usgstnm_prod_formats' ] = [ ]
				st.session_state[ 'usgstnm_max_items' ] = 25
				st.session_state[ 'usgstnm_offset' ] = 0
				st.session_state[ 'usgstnm_timeout' ] = 20
				st.session_state[ 'usgstnm_results' ] = { }
				st.session_state[ 'usgstnm_clear_request' ] = False
			
			usgstnm_mode = st.selectbox( 'Mode', options=USGSTNM_MODES, index=USGSTNM_MODES.index( st.session_state.get( 'usgstnm_mode', 'products' ) ), key='usgstnm_mode' )
			
			usgstnm_dataset = st.text_input( 'Dataset', value=st.session_state.get( 'usgstnm_dataset', '' ), key='usgstnm_dataset', disabled=(
						usgstnm_mode != 'products'), placeholder='Example: Digital Elevation Model (DEM) 1 meter' )
			
			usgstnm_q = st.text_input( 'Search Text', value=st.session_state.get( 'usgstnm_q', '' ), key='usgstnm_q', disabled=(
						usgstnm_mode != 'products'), placeholder='Optional keyword search' )
			
			usgstnm_bbox = st.text_input( 'Bounding Box', value=st.session_state.get( 'usgstnm_bbox', '' ), key='usgstnm_bbox', disabled=(
						usgstnm_mode != 'products'), placeholder='minx,miny,maxx,maxy' )
			
			usgstnm_prod_formats = st.multiselect( 'Product Formats', options=USGSTNM_PRODUCT_FORMATS, default=[
					value for value in st.session_state.get( 'usgstnm_prod_formats', [ ] ) if
					value in USGSTNM_PRODUCT_FORMATS ], key='usgstnm_prod_formats', disabled=(
						usgstnm_mode != 'products'), help='Optional TNM product-format filter.' )
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				usgstnm_max_items = st.number_input( 'Max Items', min_value=1, max_value=500, value=int( st.session_state.get( 'usgstnm_max_items', 25 ) ), step=1, key='usgstnm_max_items', disabled=(
							usgstnm_mode != 'products') )
			
			with page_c2:
				usgstnm_offset = st.number_input( 'Offset', min_value=0, max_value=10000, value=int( st.session_state.get( 'usgstnm_offset', 0 ) ), step=1, key='usgstnm_offset', disabled=(
							usgstnm_mode != 'products') )
			
			usgstnm_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'usgstnm_timeout', 20 ) ), step=1, key='usgstnm_timeout' )
			
			st.caption( 'Products mode searches downloadable TNM records. Datasets mode '
			            'returns the dataset catalog for discovery.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				usgstnm_submit = st.button( 'Submit', key='usgstnm_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='usgstnm_clear', on_click=_clear_usgstnm_state, width='stretch' )
			
			if usgstnm_submit:
				st.session_state[ 'geospatial_active_source' ] = 'The National Map'
		
		# ----------------------------
		# ------ Expander USGS Science Base
		# ----------------------------
		with st.expander( label='USGS Science Base', icon='🔬', expanded=False ):
			USGSSB_MODES = [ 'items', 'item' ]
			
			USGSSB_FIELD_OPTIONS = [ 'id', 'title', 'summary', 'body', 'itemType', 'dateCreated',
					'dateUpdated', 'lastUpdated', 'contacts', 'tags', 'facets', 'spatial', 'files',
					'webLinks', 'link', 'parentId', 'hasChildren', 'permissions' ]
			
			def _clear_usgssb_state( ) -> None:
				st.session_state[ 'usgssb_clear_request' ] = True
			
			if 'usgssb_results' not in st.session_state:
				st.session_state[ 'usgssb_results' ] = { }
			
			if 'usgssb_clear_request' not in st.session_state:
				st.session_state[ 'usgssb_clear_request' ] = False
			
			if st.session_state.get( 'usgssb_mode', 'items' ) not in USGSSB_MODES:
				st.session_state[ 'usgssb_mode' ] = 'items'
			
			if 'usgssb_q' not in st.session_state:
				st.session_state[ 'usgssb_q' ] = ''
			
			if 'usgssb_item_id' not in st.session_state:
				st.session_state[ 'usgssb_item_id' ] = ''
			
			if 'usgssb_max_items' not in st.session_state:
				st.session_state[ 'usgssb_max_items' ] = 25
			
			if 'usgssb_offset' not in st.session_state:
				st.session_state[ 'usgssb_offset' ] = 0
			
			if 'usgssb_fields' not in st.session_state:
				st.session_state[ 'usgssb_fields' ] = [ ]
			
			if isinstance( st.session_state.get( 'usgssb_fields' ), str ):
				raw_fields = str( st.session_state.get( 'usgssb_fields', '' ) )
				st.session_state[ 'usgssb_fields' ] = [ item.strip( ) for item in
						raw_fields.split( ',' ) if item.strip( ) in USGSSB_FIELD_OPTIONS ]
			
			if 'usgssb_timeout' not in st.session_state:
				st.session_state[ 'usgssb_timeout' ] = 20
			
			if st.session_state.get( 'usgssb_clear_request', False ):
				st.session_state[ 'usgssb_mode' ] = 'items'
				st.session_state[ 'usgssb_q' ] = ''
				st.session_state[ 'usgssb_item_id' ] = ''
				st.session_state[ 'usgssb_max_items' ] = 25
				st.session_state[ 'usgssb_offset' ] = 0
				st.session_state[ 'usgssb_fields' ] = [ ]
				st.session_state[ 'usgssb_timeout' ] = 20
				st.session_state[ 'usgssb_results' ] = { }
				st.session_state[ 'usgssb_clear_request' ] = False
			
			usgssb_mode = st.selectbox( 'Mode', options=USGSSB_MODES, index=USGSSB_MODES.index( st.session_state.get( 'usgssb_mode', 'items' ) ), key='usgssb_mode' )
			
			usgssb_q = st.text_input( 'Search Query', value=st.session_state.get( 'usgssb_q', '' ), key='usgssb_q', disabled=(
						usgssb_mode != 'items'), placeholder='Optional keyword search' )
			
			usgssb_item_id = st.text_input( 'Item ID', value=st.session_state.get( 'usgssb_item_id', '' ), key='usgssb_item_id', disabled=(
						usgssb_mode != 'item'), placeholder='ScienceBase item identifier' )
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				usgssb_max_items = st.number_input( 'Max Items', min_value=5, max_value=1000, value=int( st.session_state.get( 'usgssb_max_items', 25 ) ), step=5, key='usgssb_max_items', disabled=(
							usgssb_mode != 'items'), help='ScienceBase search guidance limits max to 1000 and uses increments '
				                                          'of 5.' )
			
			with page_c2:
				usgssb_offset = st.number_input( 'Offset', min_value=0, max_value=150000, value=int( st.session_state.get( 'usgssb_offset', 0 ) ), step=5, key='usgssb_offset', disabled=(
							usgssb_mode != 'items'), help='Paging offset. Use increments of 5 for ScienceBase search '
				                                          'requests.' )
			
			usgssb_fields = st.multiselect( 'Fields', options=USGSSB_FIELD_OPTIONS, default=[ value
					for value in st.session_state.get( 'usgssb_fields', [ ] ) if
					value in USGSSB_FIELD_OPTIONS ], key='usgssb_fields', disabled=(
						usgssb_mode != 'items'), help='Optional ScienceBase item fields to request.' )
			
			usgssb_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'usgssb_timeout', 20 ) ), step=1, key='usgssb_timeout' )
			
			st.caption( 'Items mode performs catalog discovery.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				usgssb_submit = st.button( 'Submit', key='usgssb_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='usgssb_clear', on_click=_clear_usgssb_state, width='stretch' )
			
			if usgssb_submit:
				st.session_state[ 'geospatial_active_source' ] = 'USGS Science Base'
		
		# ----------------------------
		# ------ Expander Open Sky
		# ----------------------------
		with st.expander( label='Open Sky', icon='✈️', expanded=False ):
			def _clear_opensky_state( ) -> None:
				st.session_state[ 'opensky_results' ] = { }
				st.session_state[ 'opensky_mode' ] = 'states_bbox'
				st.session_state[ 'opensky_icao24' ] = ''
				st.session_state[ 'opensky_airport' ] = ''
				st.session_state[ 'opensky_begin' ] = 0
				st.session_state[ 'opensky_end' ] = 0
				st.session_state[ 'opensky_time_value' ] = 0
				st.session_state[ 'opensky_lamin' ] = 39.0
				st.session_state[ 'opensky_lomin' ] = -77.5
				st.session_state[ 'opensky_lamax' ] = 40.5
				st.session_state[ 'opensky_lomax' ] = -75.0
				st.session_state[ 'opensky_extended' ] = False
				st.session_state[ 'opensky_client_id' ] = ''
				st.session_state[ 'opensky_client_secret' ] = ''
				st.session_state[ 'opensky_timeout' ] = 20
			
			def _coerce_opensky_time_value( mode_value: str, raw_value: int ) -> int | None:
				active_mode = str( mode_value or 'states_bbox' ).strip( ).lower( )
				value = int( raw_value or 0 )
				
				if active_mode == 'states_bbox' and value <= 0:
					return None
				
				if active_mode == 'track_aircraft':
					return value
				
				if value <= 0:
					return None
				
				return value
			
			if 'opensky_results' not in st.session_state:
				st.session_state[ 'opensky_results' ] = { }
			
			if 'opensky_mode' not in st.session_state:
				st.session_state[ 'opensky_mode' ] = 'states_bbox'
			
			if 'opensky_icao24' not in st.session_state:
				st.session_state[ 'opensky_icao24' ] = ''
			
			if 'opensky_airport' not in st.session_state:
				st.session_state[ 'opensky_airport' ] = ''
			
			if 'opensky_begin' not in st.session_state:
				st.session_state[ 'opensky_begin' ] = 0
			
			if 'opensky_end' not in st.session_state:
				st.session_state[ 'opensky_end' ] = 0
			
			if 'opensky_time_value' not in st.session_state:
				st.session_state[ 'opensky_time_value' ] = 0
			
			if 'opensky_lamin' not in st.session_state:
				st.session_state[ 'opensky_lamin' ] = 39.0
			
			if 'opensky_lomin' not in st.session_state:
				st.session_state[ 'opensky_lomin' ] = -77.5
			
			if 'opensky_lamax' not in st.session_state:
				st.session_state[ 'opensky_lamax' ] = 40.5
			
			if 'opensky_lomax' not in st.session_state:
				st.session_state[ 'opensky_lomax' ] = -75.0
			
			if 'opensky_extended' not in st.session_state:
				st.session_state[ 'opensky_extended' ] = False
			
			if 'opensky_client_id' not in st.session_state:
				st.session_state[ 'opensky_client_id' ] = ''
			
			if 'opensky_client_secret' not in st.session_state:
				st.session_state[ 'opensky_client_secret' ] = ''
			
			if 'opensky_timeout' not in st.session_state:
				st.session_state[ 'opensky_timeout' ] = 20
			
			mode = st.selectbox( 'Mode', options=[ 'states_bbox', 'flights_aircraft',
					'arrivals_airport', 'departures_airport',
					'track_aircraft', ], key='opensky_mode', help=(
				'states_bbox = live aircraft inside a bounding box; '
				'flights_aircraft = flights for one aircraft; '
				'arrivals_airport / departures_airport = airport traffic; '
				'track_aircraft = trajectory waypoints for one aircraft.') )
			
			icao24 = st.text_input( 'ICAO24 (Aircraft Hex ID)', value='', key='opensky_icao24', help='Required for flights_aircraft and track_aircraft.' )
			
			airport = st.text_input( 'Airport ICAO', value='', key='opensky_airport', help='Required for arrivals_airport and departures_airport.' )
			
			begin = st.number_input( 'Begin (Unix Time)', min_value=0, value=0, step=60, key='opensky_begin', help='Required for aircraft and airport history '
			                                                                                                       'queries.' )
			
			end = st.number_input( 'End (Unix Time)', min_value=0, value=0, step=60, key='opensky_end', help='Required for aircraft and airport history queries.' )
			
			time_value = st.number_input( 'Time (Unix Time / 0 for Live Track)', min_value=0, value=0, step=60, key='opensky_time_value', help=(
				'Optional for states_bbox. Leave at 0 for current live state '
				'vectors. Used by track_aircraft, where 0 requests the current '
				'or latest available track.') )
			
			lamin = st.number_input( 'Min Latitude', value=39.0, key='opensky_lamin' )
			lomin = st.number_input( 'Min Longitude', value=-77.5, key='opensky_lomin' )
			lamax = st.number_input( 'Max Latitude', value=40.5, key='opensky_lamax' )
			lomax = st.number_input( 'Max Longitude', value=-75.0, key='opensky_lomax' )
			extended = st.checkbox( 'Extended Aircraft Categories', value=False, key='opensky_extended', help='Only applies to states_bbox.' )
			
			client_id = st.text_input( 'Client ID (Optional)', value='', key='opensky_client_id' )
			
			client_secret = st.text_input( 'Client Secret (Optional)', value='', type='password', key='opensky_client_secret' )
			
			timeout = st.slider( 'Timeout (seconds)', min_value=5, max_value=60, value=20, step=1, key='opensky_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				opensky_submit = st.button( 'Submit', key='opensky_submit', use_container_width=True, width='stretch' )
			with b2:
				st.button( 'Clear', key='opensky_clear', on_click=_clear_opensky_state, use_container_width=True, width='stretch' )
			
			if opensky_submit:
				st.session_state[ 'geospatial_active_source' ] = 'Open Sky'
	
	with right:
		active_source = st.session_state.get( 'geospatial_active_source', '' )
		result_keys: Dict[ str, str ] = { 'Geocoding': 'googlegeocoding_results',
				'Google Maps': 'googlemaps_results', 'Google Weather': 'googleweather_results',
				'Open Weather': 'openweather_results',
				'Historical Weather': 'historicalweather_results',
				'USGS Earthquakes': 'usgsearthquakes_results',
				'NASA Earth Observatory': 'earthobservatory_results',
				'The National Map': 'usgstnm_results', 'USGS Science Base': 'usgssb_results',
				'Open Sky': 'opensky_results', }
		if active_source in result_keys:
			active_result = st.session_state.get( result_keys[ active_source ] )
			_promote_geospatial_result( source=active_source, result=active_result )
		
		st.markdown( '##### Results' )
		if not active_source:
			st.info( 'Select a source, configure the request, and submit it to display results.' )
		else:
			st.caption( f'Active Source: {active_source}' )
		
		if active_source == 'Geocoding':
			st.markdown( 'Results' )
			if googlegeocoding_submit:
				try:
					f = GoogleGeocoding( )
					result = f.fetch( mode=str( googlegeocoding_mode ), query=str( googlegeocoding_query ), latitude=float( googlegeocoding_latitude ), longitude=float( googlegeocoding_longitude ), place_id=str( googlegeocoding_place_id ), language=str( googlegeocoding_language or 'en' ).strip( ), region=str( googlegeocoding_region or '' ).strip( ), result_type=str( googlegeocoding_result_type or '' ).strip( ), location_type=str( googlegeocoding_location_type or '' ).strip( ), time=int( googlegeocoding_timeout ), api_key=(
								googlegeocoding_api_key or None) )
					
					st.session_state[ 'googlegeocoding_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Geocoding request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googlegeocoding_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': result.get( 'mode', '' ), 'url': result.get( 'url', '' ),
						'params': result.get( 'params', { } ),
						'status': result.get( 'status', '' ), } )
				
				results_list = result.get( 'results', [ ] ) if isinstance( result, dict ) else [ ]
				
				if not results_list:
					st.info( 'No geocoding results returned.' )
				else:
					for idx, item in enumerate( results_list, start=1 ):
						formatted_address = item.get( 'formatted_address', f'Result {idx}' )
						place_id_value = item.get( 'place_id', '' )
						types_value = item.get( 'types', [ ] )
						geometry = item.get( 'geometry', { } ) if isinstance( item, dict ) else { }
						location = geometry.get( 'location', { } ) if isinstance( geometry, dict ) else { }
						
						with st.container( border=True ):
							st.markdown( f'**{idx}. {formatted_address}**' )
							meta_parts: List[ str ] = [ ]
							if place_id_value:
								meta_parts.append( f'Place ID: `{place_id_value}`' )
							
							if isinstance( types_value, list ) and types_value:
								meta_parts.append( f"Types: `{', '.join( types_value[ :4 ] )}`" )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if isinstance( location, dict ):
								lat_value = location.get( 'lat', '' )
								lng_value = location.get( 'lng', '' )
								if str( lat_value ).strip( ) or str( lng_value ).strip( ):
									st.write( f'Coordinates: {lat_value}, {lng_value}' )
							
							address_components = item.get( 'address_components', [ ] )
							if isinstance( address_components, list ) and address_components:
								component_rows: List[ Dict[ str, Any ] ] = [ ]
								for component in address_components:
									if isinstance( component, dict ):
										component_rows.append( {
												'long_name': component.get( 'long_name', '' ),
												'short_name': component.get( 'short_name', '' ),
												'types': ', '.join( component.get( 'types', [ ] ) ) } )
								
								if component_rows:
									with st.expander( 'Address Components', expanded=False ):
										st.dataframe( pd.DataFrame( component_rows ), use_container_width=True, hide_index=True )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Google Maps
		if active_source == 'Google Maps':
			st.markdown( 'Results' )
			if googlemaps_submit:
				try:
					gm = GoogleMaps( )
					if googlemaps_mode == 'geocode_location':
						if not str( googlemaps_query or '' ).strip( ):
							raise ValueError( 'Address is required for geocode_location '
							                  'mode.' )
						
						coords = gm.geocode_location( str( googlemaps_query ).strip( ) )
						result = { 'mode': googlemaps_mode,
								'query': str( googlemaps_query ).strip( ),
								'radius': int( googlemaps_radius ), 'latitude': coords[ 0 ],
								'longitude': coords[ 1 ],
								'coordinates': f'{coords[ 0 ]}, {coords[ 1 ]}', }
					
					elif googlemaps_mode == 'geocode_coordinates':
						address = gm.geocode_coordinates( lat=float( googlemaps_latitude ), long=float( googlemaps_longitude ) )
						result = { 'mode': googlemaps_mode,
								'latitude': float( googlemaps_latitude ),
								'longitude': float( googlemaps_longitude ), 'address': address, }
					
					elif googlemaps_mode == 'validate_address':
						address_lines = _split_googlemaps_address_lines( googlemaps_address_lines )
						
						if not address_lines:
							raise ValueError( 'At least one address line is required for validate_address '
							                  'mode.' )
						
						payload = gm.validate_address( address_lines )
						result = { 'mode': googlemaps_mode, 'address_lines': address_lines,
								'data': payload or { }, }
					
					else:
						if not str( googlemaps_origin or '' ).strip( ):
							raise ValueError( 'Origin is required for request_directions mode.' )
						
						if not str( googlemaps_destination or '' ).strip( ):
							raise ValueError( 'Destination is required for request_directions mode.' )
						
						payload = gm.request_directions( origin=str( googlemaps_origin ).strip( ), destination=str( googlemaps_destination ).strip( ), mode=str( googlemaps_travel_mode ).strip( ) )
						result = { 'mode': googlemaps_mode,
								'origin': str( googlemaps_origin ).strip( ),
								'destination': str( googlemaps_destination ).strip( ),
								'travel_mode': str( googlemaps_travel_mode ).strip( ),
								'data': payload or { }, }
					
					st.session_state[ 'googlemaps_results' ] = result
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Maps request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googlemaps_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Summary' )
				summary_rows = [ ]
				
				for key, value in result.items( ):
					if key == 'data':
						continue
					
					if isinstance( value, (str, int, float, bool) ) or value is None:
						summary_rows.append( { 'Field': key, 'Value': value } )
				
				if summary_rows:
					st.dataframe( pd.DataFrame( summary_rows ), use_container_width=True, hide_index=True )
				
				if result.get( 'mode' ) == 'geocode_location':
					lat_value = result.get( 'latitude', None )
					lon_value = result.get( 'longitude', None )
					
					if lat_value is not None and lon_value is not None:
						st.markdown( '#### Coordinates' )
						st.text_area( 'Coords', value=str( result.get( 'coordinates', '' ) ), height=90 )
						
						st.markdown( '#### Map' )
						st.map( [ { 'lat': float( lat_value ), 'lon': float( lon_value ) } ] )
				
				elif result.get( 'mode' ) == 'geocode_coordinates':
					st.markdown( '#### Address' )
					st.text_area( 'Formatted Address', value=str( result.get( 'address', '' ) ), height=120 )
					
					st.markdown( '#### Map' )
					st.map( [ { 'lat': float( result.get( 'latitude', 0.0 ) ),
							'lon': float( result.get( 'longitude', 0.0 ) ) } ] )
				
				else:
					data = result.get( 'data', { } )
					
					if isinstance( data, dict ) and data:
						status_value = data.get( 'status', '' )
						if status_value:
							st.markdown( f"**Status:** `{status_value}`" )
						
						routes = data.get( 'routes', [ ] )
						if isinstance( routes, list ) and routes:
							route_rows = [ ]
							
							for idx, route in enumerate( routes, start=1 ):
								legs = route.get( 'legs', [ ] ) if isinstance( route, dict ) else [ ]
								first_leg = legs[ 0 ] if legs else { }
								
								route_rows.append( { 'Route': idx,
										'Summary': route.get( 'summary', '' ),
										'Start': first_leg.get( 'start_address', '' ),
										'End': first_leg.get( 'end_address', '' ), 'Distance': (
												first_leg.get( 'distance', { } ).get( 'text', '' )),
										'Duration': (
												first_leg.get( 'duration', { } ).get( 'text', '' )), } )
							
							st.markdown( '#### Routes' )
							st.dataframe( pd.DataFrame( route_rows ), use_container_width=True, hide_index=True )
						
						result_payload = data.get( 'result', { } )
						if isinstance( result_payload, dict ) and result_payload:
							st.markdown( '#### Address Validation Result' )
							st.json( result_payload )
						
						with st.expander( 'Raw Payload', expanded=False ):
							st.json( data )
					else:
						st.info( 'No Google Maps payload was returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Google Weather
		if active_source == 'Google Weather':
			st.markdown( 'Results' )
			
			if gw_submit:
				try:
					if not str( gw_location or '' ).strip( ):
						raise ValueError( 'Location is required for Google Weather.' )
					
					f = GoogleWeather( )
					
					if gw_mode == 'current':
						result = f.fetch_current( address=gw_location, units_system=gw_units, language_code=gw_language, time=int( gw_timeout ) )
					elif gw_mode == 'hourly_forecast':
						result = f.fetch_hourly_forecast( address=gw_location, hours=int( gw_hours ), units_system=gw_units, language_code=gw_language, time=int( gw_timeout ) )
					elif gw_mode == 'daily_forecast':
						result = f.fetch_daily_forecast( address=gw_location, days=int( gw_days ), units_system=gw_units, language_code=gw_language, time=int( gw_timeout ) )
					elif gw_mode == 'hourly_history':
						result = f.fetch_hourly_history( address=gw_location, hours=int( gw_history_hours ), units_system=gw_units, language_code=gw_language, time=int( gw_timeout ) )
					else:
						result = f.fetch_alerts( address=gw_location, language_code=gw_language, time=int( gw_timeout ) )
					
					st.session_state[ 'googleweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googleweather_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': mode_value, 'url': result.get( 'url', '' ), 'params': params, } )
				
				if isinstance( data, dict ) and data:
					current = (
								data.get( 'currentWeather' ) or data.get( 'current_weather' ) or data.get( 'currentConditions' ) or data.get( 'current_conditions' ) or { })
					
					hourly = (
								data.get( 'hourlyForecasts' ) or data.get( 'hourly_forecasts' ) or data.get( 'forecastHours' ) or data.get( 'hours' ) or [ ])
					
					history = (data.get( 'historyHours' ) or data.get( 'history_hours' ) or [ ])
					
					daily = (
								data.get( 'dailyForecasts' ) or data.get( 'daily_forecasts' ) or data.get( 'forecastDays' ) or data.get( 'days' ) or [ ])
					
					alerts = (
								data.get( 'weatherAlerts' ) or data.get( 'weather_alerts' ) or data.get( 'alerts' ) or [ ])
					
					if current:
						st.markdown( '#### Current Conditions' )
						st.json( current )
					
					if hourly:
						st.markdown( '#### Hourly Forecast' )
						df_googleweather_hourly = pd.DataFrame( hourly )
						st.dataframe( df_googleweather_hourly, use_container_width=True, hide_index=True )
					
					if history:
						st.markdown( '#### Hourly History' )
						df_googleweather_history = pd.DataFrame( history )
						st.dataframe( df_googleweather_history, use_container_width=True, hide_index=True )
					
					if daily:
						st.markdown( '#### Daily Forecast' )
						df_googleweather_daily = pd.DataFrame( daily )
						st.dataframe( df_googleweather_daily, use_container_width=True, hide_index=True )
					
					if alerts:
						st.markdown( '#### Weather Alerts' )
						df_googleweather_alerts = pd.DataFrame( alerts )
						st.dataframe( df_googleweather_alerts, use_container_width=True, hide_index=True )
					
					with st.expander( 'Raw Payload', expanded=False ):
						st.json( data )
				
				else:
					st.info( 'No Google Weather payload was returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Open Weather
		if active_source == 'Open Weather':
			if openweather_submit:
				try:
					f = OpenWeather( )
					result = f.fetch( location=str( openweather_location ), mode=str( openweather_mode ), zone=str( openweather_timezone or 'auto' ).strip( ), forecast_days=int( openweather_forecast_days ), past_days=int( openweather_past_days ), count=int( openweather_count ) )
					
					st.session_state[ 'openweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Open Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'openweather_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'location' in result:
						st.markdown( f"**Location:** {result.get( 'location', '' )}" )
					if 'latitude' in result:
						st.markdown( f"**Latitude:** {result.get( 'latitude', '' )}" )
				
				with meta_c2:
					if 'longitude' in result:
						st.markdown( f"**Longitude:** {result.get( 'longitude', '' )}" )
					if 'timezone' in result:
						st.markdown( f"**Timezone:** {result.get( 'timezone', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				geocoding = result.get( 'geocoding', { } ) or { }
				if geocoding:
					st.markdown( '#### Geocoding Result' )
					st.json( geocoding )
				
				params = result.get( 'params', { } ) or { }
				if params:
					st.markdown( '#### Request Parameters' )
					st.json( params )
				
				data = result.get( 'data', { } ) or { }
				if data:
					st.markdown( '#### Forecast Payload' )
					st.json( data )
				
				message = result.get( 'message', '' )
				if message:
					st.info( message )
		
		# -------- Historical Weather
		if active_source == 'Historical Weather':
			if historicalweather_submit:
				try:
					f = HistoricalWeather( )
					result = f.fetch( location=str( historicalweather_location ), date=historicalweather_date, zone=str( historicalweather_timezone or 'auto' ).strip( ), count=int( historicalweather_count ) )
					
					st.session_state[ 'historicalweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Historical Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'historicalweather_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'location' in result:
						st.markdown( f"**Location:** {result.get( 'location', '' )}" )
					if 'latitude' in result:
						st.markdown( f"**Latitude:** {result.get( 'latitude', '' )}" )
					if 'longitude' in result:
						st.markdown( f"**Longitude:** {result.get( 'longitude', '' )}" )
				
				with meta_c2:
					if 'date' in result:
						st.markdown( f"**Date:** {result.get( 'date', '' )}" )
					if 'timezone' in result:
						st.markdown( f"**Timezone:** {result.get( 'timezone', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				geocoding = result.get( 'geocoding', { } ) or { }
				if geocoding:
					st.markdown( '#### Geocoding Result' )
					st.json( geocoding )
				
				params = result.get( 'params', { } ) or { }
				if params:
					st.markdown( '#### Request Parameters' )
					st.json( params )
				
				data = result.get( 'data', { } ) or { }
				if data:
					st.markdown( '#### Historical Weather Payload' )
					st.json( data )
				
				message = result.get( 'message', '' )
				if message:
					st.info( message )
		
		# -------- USGS Earthquakes
		if active_source == 'USGS Earthquakes':
			if usgseq_submit:
				try:
					if usgseq_mode == 'search':
						if usgseq_start_date > usgseq_end_date:
							raise ValueError( 'Start Date must be on or before End Date.' )
						
						if float( usgseq_min_magnitude ) > float( usgseq_max_magnitude ):
							raise ValueError( 'Min Magnitude must be less than or equal to Max Magnitude.' )
					
					latitude_value = _coerce_optional_float( name='Latitude', value=usgseq_latitude, min_value=-90.0, max_value=90.0 )
					
					longitude_value = _coerce_optional_float( name='Longitude', value=usgseq_longitude, min_value=-180.0, max_value=180.0 )
					max_radius_value = _coerce_optional_float( name='Max Radius KM', value=usgseq_max_radius_km, min_value=0.0, max_value=20001.6 )
					
					if (latitude_value is None and longitude_value is not None) or (
							latitude_value is not None and longitude_value is None):
						raise ValueError( 'Latitude and Longitude must both be supplied for radial search.' )
					
					if max_radius_value is not None and (
							latitude_value is None or longitude_value is None):
						raise ValueError( 'Max Radius KM requires both Latitude and Longitude.' )
					
					f = USGSEarthquakes( )
					result = f.fetch( mode=str( usgseq_mode ), feed=str( usgseq_feed ), start_date=str( usgseq_start_date ), end_date=str( usgseq_end_date ), min_magnitude=float( usgseq_min_magnitude ), max_magnitude=float( usgseq_max_magnitude ), limit=int( usgseq_limit ), order_by=str( usgseq_order_by ), event_type=str( usgseq_event_type or 'earthquake' ), latitude=latitude_value, longitude=longitude_value, max_radius_km=max_radius_value, time=int( usgseq_timeout ) )
					
					st.session_state[ 'usgsearthquakes_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS Earthquakes request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgsearthquakes_results', { } )
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'feed' in result:
						st.markdown( f"**Feed:** {result.get( 'feed', '' )}" )
					if 'count' in result:
						st.markdown( f"**Events Returned:** {result.get( 'count', 0 )}" )
				
				with meta_c2:
					if 'title' in result:
						st.markdown( f"**Title:** {result.get( 'title', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '##### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_mag = summary.get( 'max_magnitude', None )
						st.metric( 'Strongest Magnitude', '' if max_mag is None else str( max_mag ) )
					
					with sum_c3:
						st.metric( 'Tsunami Flags', int( summary.get( 'tsunami_count', 0 ) or 0 ) )
					
					first_place = str( summary.get( 'first_place', '' ) or '' )
					if first_place:
						st.markdown( f"**First Event:** {first_place}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Earthquake Events' )
					df_usgsearthquakes = pd.DataFrame( rows )
					
					if not df_usgsearthquakes.empty:
						st.dataframe( df_usgsearthquakes, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = (
										item.get( 'Latitude', None ) or item.get( 'lat', None ) or item.get( 'latitude', None ))
							longitude = (
										item.get( 'Longitude', None ) or item.get( 'lon', None ) or item.get( 'longitude', None ))
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': float( latitude ),
										'lon': float( longitude ) } )
						
						if map_rows:
							st.markdown( '#### Event Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Place', '' ) or item.get( 'Title', '' ) or item.get( 'Id', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									if 'Magnitude' in item:
										st.markdown( f"**Magnitude:** {item.get( 'Magnitude', '' )}" )
									if 'Place' in item:
										st.markdown( f"**Place:** {item.get( 'Place', '' )}" )
									if 'Time' in item:
										st.markdown( f"**Time:** {item.get( 'Time', '' )}" )
								
								with right_c:
									if 'Depth' in item:
										st.markdown( f"**Depth:** {item.get( 'Depth', '' )}" )
									if 'Event Type' in item:
										st.markdown( f"**Event Type:** "
										             f"{item.get( 'Event Type', '' )}" )
									if 'URL' in item:
										st.markdown( f"**URL:** {item.get( 'URL', '' )}" )
								
								raw_item = item.get( 'Raw Feature', { } )
								if raw_item:
									with st.expander( 'Raw Feature', expanded=False ):
										st.json( raw_item )
					else:
						st.info( 'No displayable USGS earthquake rows were found.' )
				else:
					st.info( 'No USGS earthquake records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Earth Observatory
		if active_source == 'NASA Earth Observatory':
			if earth_submit:
				try:
					f = EarthObservatory( )
					result = f.fetch( mode=earth_mode, status=earth_status, category=earth_category, source=earth_source, limit=int( earth_limit ), days=int( earth_days ), start_date=str( earth_start_date ), end_date=str( earth_end_date ), time=int( earth_timeout ) )
					
					st.session_state[ 'earthobservatory_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Earth Observatory request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'earthobservatory_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					if result.get( 'params', { } ):
						st.markdown( f"**Parameters:** {len( result.get( 'params', { } ) )}" )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'events', [ ] ):
					st.markdown( '#### Events' )
					df_events = pd.DataFrame( result.get( 'events', [ ] ) )
					st.dataframe( df_events, use_container_width=True, hide_index=True )
				
				if result.get( 'categories', [ ] ):
					st.markdown( '#### Categories' )
					df_categories = pd.DataFrame( result.get( 'categories', [ ] ) )
					st.dataframe( df_categories, use_container_width=True, hide_index=True )
				
				if result.get( 'sources', [ ] ):
					st.markdown( '#### Sources' )
					df_sources = pd.DataFrame( result.get( 'sources', [ ] ) )
					st.dataframe( df_sources, use_container_width=True, hide_index=True )
				
				if result.get( 'layers', [ ] ):
					st.markdown( '#### Layers' )
					df_layers = pd.DataFrame( result.get( 'layers', [ ] ) )
					st.dataframe( df_layers, use_container_width=True, hide_index=True )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- USGS The National Map
		if active_source == 'The National Map':
			if usgstnm_submit:
				try:
					clean_bbox = _validate_usgstnm_bbox( usgstnm_bbox )
					clean_product_formats = ','.join( usgstnm_prod_formats or [ ] )
					
					f = USGSTheNationalMap( )
					result = f.fetch( mode=str( usgstnm_mode ), dataset=str( usgstnm_dataset ).strip( ), q=str( usgstnm_q ).strip( ), bbox=clean_bbox, prod_formats=clean_product_formats, max_items=int( usgstnm_max_items ), offset=int( usgstnm_offset ), time=int( usgstnm_timeout ) )
					
					st.session_state[ 'usgstnm_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS The National Map request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgstnm_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'datasets' in params:
						st.markdown( f"**Dataset Filter:** {params.get( 'datasets', '' )}" )
					if 'bbox' in params:
						st.markdown( f"**Bounding Box:** {params.get( 'bbox', '' )}" )
					if 'prodFormats' in params:
						st.markdown( f"**Formats:** {params.get( 'prodFormats', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_dataset = str( summary.get( 'first_dataset', '' ) or '' )
						if first_dataset:
							st.markdown( f"**Dataset:** {first_dataset}" )
						else:
							st.markdown( '**Dataset:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Results' )
					df_usgstnm = pd.DataFrame( rows )
					
					if not df_usgstnm.empty:
						st.dataframe( df_usgstnm, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Title', '' ) or item.get( 'Name', '' ) or item.get( 'Id', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable TNM rows were found.' )
				else:
					st.info( 'No TNM records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- USGS ScienceBase
		if active_source == 'USGS Science Base':
			if usgssb_submit:
				try:
					if usgssb_mode == 'item' and not str( usgssb_item_id or '' ).strip( ):
						raise ValueError( 'Item ID is required for ScienceBase item mode.' )
					
					selected_fields = ','.join( usgssb_fields or [ ] )
					
					f = USGSScienceBase( )
					result = f.fetch( mode=str( usgssb_mode ), q=str( usgssb_q ).strip( ), item_id=str( usgssb_item_id ).strip( ), max_items=int( usgssb_max_items ), offset=int( usgssb_offset ), fields=selected_fields, time=int( usgssb_timeout ) )
					
					st.session_state[ 'usgssb_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS ScienceBase request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgssb_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'q' in params:
						st.markdown( f"**Search Query:** {params.get( 'q', '' )}" )
					if 'fields' in params:
						st.markdown( f"**Fields:** {params.get( 'fields', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						st.metric( 'Spatial Records', int( summary.get( 'spatial_count', 0 ) or 0 ) )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Results' )
					df_usgssb = pd.DataFrame( rows )
					
					if not df_usgssb.empty:
						st.dataframe( df_usgssb, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Title', '' ) or item.get( 'Id', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									if 'Type' in item:
										st.markdown( f"**Type:** {item.get( 'Type', '' )}" )
									if 'Updated' in item:
										st.markdown( f"**Updated:** {item.get( 'Updated', '' )}" )
									if 'File Count' in item:
										st.markdown( f"**Files:** {item.get( 'File Count', '' )}" )
									if 'Web Link Count' in item:
										st.markdown( f"**Web Links:** "
										             f"{item.get( 'Web Link Count', '' )}" )
									if 'Contact Count' in item:
										st.markdown( f"**Contacts:** "
										             f"{item.get( 'Contact Count', '' )}" )
								
								with right_c:
									if 'Has Spatial Metadata' in item:
										st.markdown( f"**Spatial Metadata:** "
										             f"{item.get( 'Has Spatial Metadata', False )}" )
									if 'Id' in item:
										st.markdown( f"**ID:** `{item.get( 'Id', '' )}`" )
								
								summary_text = str( item.get( 'Summary', '' ) or '' )
								if summary_text:
									st.markdown( '##### Summary' )
									st.write( summary_text )
								
								raw_item = item.get( 'Raw Item', { } )
								if raw_item:
									with st.expander( 'Raw Item', expanded=False ):
										st.json( raw_item )
					else:
						st.info( 'No displayable ScienceBase rows were found.' )
				else:
					st.info( 'No ScienceBase records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Open Sky
		if active_source == 'Open Sky':
			if opensky_submit:
				try:
					normalized_time_value = _coerce_opensky_time_value( mode_value=mode, raw_value=int( time_value or 0 ) )
					
					client = OpenSky( )
					result = client.fetch( mode=mode, icao24=icao24, airport=airport, begin=int( begin ) if int( begin or 0 ) > 0 else None, end=int( end ) if int( end or 0 ) > 0 else None, time_value=normalized_time_value, lamin=float( lamin ) if lamin is not None else None, lomin=float( lomin ) if lomin is not None else None, lamax=float( lamax ) if lamax is not None else None, lomax=float( lomax ) if lomax is not None else None, extended=bool( extended ), client_id=client_id.strip( ) or None, client_secret=client_secret.strip( ) or None, time=int( timeout ), )
					st.session_state[ 'opensky_results' ] = result or { }
				except Exception as exc:
					st.error( 'OpenSky request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'opensky_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'Returned': int( result.get( 'count', 0 ) or 0 ),
						'Time': result.get( 'time', '' ) or result.get( 'start_time', '' ),
						'ICAO24': result.get( 'icao24', '' ),
						'Callsign': result.get( 'callsign', '' ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if result.get( 'mode' ) == 'states_bbox':
					if items:
						st.markdown( '#### Live State Vectors' )
						st.dataframe( items, use_container_width=True, hide_index=True )
						
						map_rows = [ { 'lat': x.get( 'latitude' ), 'lon': x.get( 'longitude' ) } for
								x in items if
								x.get( 'latitude' ) is not None and x.get( 'longitude' ) is not None ]
						if map_rows:
							st.markdown( '#### Aircraft Positions' )
							st.map( map_rows )
					else:
						st.info( 'No aircraft state vectors matched the requested filter.' )
				
				elif result.get( 'mode' ) in ('flights_aircraft', 'arrivals_airport',
						'departures_airport'):
					if items:
						st.markdown( '#### Flights' )
						st.dataframe( items, use_container_width=True, hide_index=True )
					else:
						st.info( 'No flight rows were returned for that query window.' )
				
				elif result.get( 'mode' ) == 'track_aircraft':
					if items:
						st.markdown( '#### Aircraft Track' )
						st.dataframe( items, use_container_width=True, hide_index=True )
						
						map_rows = [ { 'lat': x.get( 'latitude' ), 'lon': x.get( 'longitude' ) } for
								x in items if
								x.get( 'latitude' ) is not None and x.get( 'longitude' ) is not None ]
						if map_rows:
							st.markdown( '#### Track Map' )
							st.map( map_rows )
					else:
						st.info( 'No track waypoints were returned for that aircraft.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

# ==============================================================================
# ENVIRONMENTAL MODE
# ==============================================================================
elif mode == 'Environmental':
	st.session_state.setdefault( 'environmental_active_source', '' )
	st.session_state.setdefault( 'environmental_documents', [ ] )
	st.session_state.setdefault( 'environmental_raw_result', None )
	
	def _build_environmental_documents( source: str, result: Any ) -> List[ Document ]:
		"""Build canonical documents from an Environmental-mode result.

		Purpose:
			Converts heterogeneous environmental responses into LangChain documents so every
			source writes to the shared document contract consumed by the right-side viewer.

		Args:
			source (str): Environmental source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents representing the result payload.
		"""
		documents: List[ Document ] = [ ]
		
		def _append_document( value: Any, index: int = 1 ) -> None:
			if isinstance( value, Document ):
				metadata = dict( value.metadata or { } )
				metadata.setdefault( 'mode', 'Environmental' )
				metadata.setdefault( 'source', source )
				documents.append( Document( page_content=value.page_content or '', metadata=metadata ) )
				return
			
			metadata: Dict[ str, Any ] = { 'mode': 'Environmental', 'source': source,
					'item': index, }
			if isinstance( value, dict ):
				content = json.dumps( normalize( value ), indent=2, ensure_ascii=False, default=str )
			elif isinstance( value, (list, tuple, set) ):
				content = json.dumps( normalize( list( value ) ), indent=2, ensure_ascii=False, default=str )
			else:
				content = str( value )
			if content.strip( ):
				documents.append( Document( page_content=content, metadata=metadata ) )
		
		if result is None:
			return documents
		if isinstance( result, list ):
			for index, item in enumerate( result, start=1 ):
				_append_document( item, index )
		else:
			_append_document( result )
		return documents
	
	def _promote_environmental_result( source: str, result: Any ) -> List[ Document ]:
		"""Promote a source result into the shared Environmental document state.

		Purpose:
			Writes the active environmental result to canonical document, raw-text, and source
			state while preserving every source-specific result key and specialized renderer.

		Args:
			source (str): Environmental source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents written to shared session state.
		"""
		documents = _build_environmental_documents( source=source, result=result )
		st.session_state[ 'environmental_raw_result' ] = result
		st.session_state[ 'environmental_documents' ] = documents
		st.session_state[ 'documents' ] = documents
		st.session_state[ 'raw_documents' ] = list( documents )
		st.session_state[ 'raw_text' ] = '\n\n'.join(
			document.page_content for document in documents if
			isinstance( document.page_content, str ) and document.page_content.strip( ) )
		st.session_state[ 'processed_text' ] = ''
		st.session_state[ 'active_loader' ] = source
		return documents
	
	st.subheader( '🌍 Environmental Data' )
	st.divider( )
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	with left:
		# ----------------------------
		# -------- Expander Air Now
		# ----------------------------
		with st.expander( label='Air Now', icon='🌫️', expanded=False ):
			if 'airnow_results' not in st.session_state:
				st.session_state[ 'airnow_results' ] = { }
			
			if 'airnow_clear_request' not in st.session_state:
				st.session_state[ 'airnow_clear_request' ] = False
			
			if st.session_state.get( 'airnow_clear_request', False ):
				st.session_state[ 'airnow_mode' ] = 'current-zip'
				st.session_state[ 'airnow_zip_code' ] = ''
				st.session_state[ 'airnow_latitude' ] = ''
				st.session_state[ 'airnow_longitude' ] = ''
				st.session_state[ 'airnow_date' ] = dt.date.today( )
				st.session_state[ 'airnow_distance' ] = 25
				st.session_state[ 'airnow_timeout' ] = 20
				st.session_state[ 'airnow_results' ] = { }
				st.session_state[ 'airnow_clear_request' ] = False
			
			def _clear_airnow_state( ) -> None:
				st.session_state[ 'airnow_clear_request' ] = True
			
			airnow_mode = st.selectbox( 'Mode', options=[ 'current-zip', 'current-latlon',
					'forecast-zip', 'forecast-latlon' ], index=[ 'current-zip', 'current-latlon',
					'forecast-zip',
					'forecast-latlon' ].index( st.session_state.get( 'airnow_mode', 'current-zip' ) ), key='airnow_mode' )
			
			airnow_zip_code = st.text_input( 'Zip Code', value=st.session_state.get( 'airnow_zip_code', '' ), key='airnow_zip_code', disabled=(
						airnow_mode not in [ 'current-zip',
						'forecast-zip' ]), placeholder='Example: 22201' )
			
			coord_c1, coord_c2 = st.columns( 2 )
			with coord_c1:
				airnow_latitude = st.text_input( 'Latitude', value=st.session_state.get( 'airnow_latitude', '' ), key='airnow_latitude', disabled=(
							airnow_mode not in [ 'current-latlon',
							'forecast-latlon' ]), placeholder='Example: 38.8816' )
			
			with coord_c2:
				airnow_longitude = st.text_input( 'Longitude', value=st.session_state.get( 'airnow_longitude', '' ), key='airnow_longitude', disabled=(
							airnow_mode not in [ 'current-latlon',
							'forecast-latlon' ]), placeholder='Example: -77.0910' )
			
			airnow_date = st.date_input( 'Forecast Date', value=st.session_state.get( 'airnow_date', dt.date.today( ) ), key='airnow_date', disabled=(
						airnow_mode not in [ 'forecast-zip', 'forecast-latlon' ]) )
			
			airnow_distance = st.number_input( 'Distance (miles)', min_value=0, max_value=500, value=int( st.session_state.get( 'airnow_distance', 25 ) ), step=1, key='airnow_distance' )
			
			airnow_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'airnow_timeout', 20 ) ), step=1, key='airnow_timeout' )
			
			st.caption( 'AirNow supports current observations and forecasts by Zip code or '
			            'latitude/longitude.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				airnow_submit = st.button( 'Submit', key='airnow_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='airnow_clear', on_click=_clear_airnow_state, width='stretch' )
			
			if airnow_submit:
				st.session_state[ 'environmental_active_source' ] = 'Air Now'
		
		# ----------------------------
		# -------- Expander NOAA Climate Data
		# ----------------------------
		with st.expander( label='NOAA Climate Data', icon='🌡️', expanded=False ):
			CLIMATEDATA_MODES = [ 'datasets', 'data' ]
			CLIMATEDATA_DATASETS = [ 'daily-summaries', 'global-summary-of-the-day',
					'global-hourly', 'local-climatological-data', 'normals-daily',
					'normals-monthly', 'normals-annualseasonal' ]
			
			def _clear_climatedata_state( ) -> None:
				st.session_state[ 'climatedata_clear_request' ] = True
			
			if 'climatedata_results' not in st.session_state:
				st.session_state[ 'climatedata_results' ] = { }
			
			if 'climatedata_clear_request' not in st.session_state:
				st.session_state[ 'climatedata_clear_request' ] = False
			
			if st.session_state.get( 'climatedata_mode', 'datasets' ) not in CLIMATEDATA_MODES:
				st.session_state[ 'climatedata_mode' ] = 'datasets'
			
			if st.session_state.get( 'climatedata_dataset', 'daily-summaries' ) not in CLIMATEDATA_DATASETS:
				st.session_state[ 'climatedata_dataset' ] = 'daily-summaries'
			
			if 'climatedata_keyword' not in st.session_state:
				st.session_state[ 'climatedata_keyword' ] = ''
			
			if 'climatedata_start_date' not in st.session_state:
				st.session_state[
					'climatedata_start_date' ] = dt.date.today( ) - dt.timedelta( days=30 )
			
			if 'climatedata_end_date' not in st.session_state:
				st.session_state[ 'climatedata_end_date' ] = dt.date.today( )
			
			if 'climatedata_stations' not in st.session_state:
				st.session_state[ 'climatedata_stations' ] = ''
			
			if 'climatedata_data_types' not in st.session_state:
				st.session_state[ 'climatedata_data_types' ] = ''
			
			if 'climatedata_limit' not in st.session_state:
				st.session_state[ 'climatedata_limit' ] = 25
			
			if 'climatedata_offset' not in st.session_state:
				st.session_state[ 'climatedata_offset' ] = 0
			
			if 'climatedata_timeout' not in st.session_state:
				st.session_state[ 'climatedata_timeout' ] = 20
			
			if st.session_state.get( 'climatedata_clear_request', False ):
				st.session_state[ 'climatedata_mode' ] = 'datasets'
				st.session_state[ 'climatedata_keyword' ] = ''
				st.session_state[ 'climatedata_dataset' ] = 'daily-summaries'
				st.session_state[ 'climatedata_start_date' ] = (
						dt.date.today( ) - dt.timedelta( days=30 ))
				st.session_state[ 'climatedata_end_date' ] = dt.date.today( )
				st.session_state[ 'climatedata_stations' ] = ''
				st.session_state[ 'climatedata_data_types' ] = ''
				st.session_state[ 'climatedata_limit' ] = 25
				st.session_state[ 'climatedata_offset' ] = 0
				st.session_state[ 'climatedata_timeout' ] = 20
				st.session_state[ 'climatedata_results' ] = { }
				st.session_state[ 'climatedata_clear_request' ] = False
			
			climatedata_mode = st.selectbox( 'Mode', options=CLIMATEDATA_MODES, index=CLIMATEDATA_MODES.index( st.session_state.get( 'climatedata_mode', 'datasets' ) ), key='climatedata_mode' )
			
			climatedata_keyword = st.text_input( 'Keyword', value=st.session_state.get( 'climatedata_keyword', '' ), key='climatedata_keyword', disabled=(
						climatedata_mode != 'datasets'), placeholder='Example: precipitation' )
			
			climatedata_dataset = st.selectbox( 'Dataset', options=CLIMATEDATA_DATASETS, index=CLIMATEDATA_DATASETS.index( st.session_state.get( 'climatedata_dataset', 'daily-summaries' ) ), key='climatedata_dataset', disabled=(
						climatedata_mode != 'data'), help='NCEI dataset identifier passed to the Access Data Service.' )
			
			date_c1, date_c2 = st.columns( 2 )
			with date_c1:
				climatedata_start_date = st.date_input( 'Start Date', value=st.session_state.get( 'climatedata_start_date', dt.date.today( ) - dt.timedelta( days=30 ) ), key='climatedata_start_date' )
			
			with date_c2:
				climatedata_end_date = st.date_input( 'End Date', value=st.session_state.get( 'climatedata_end_date', dt.date.today( ) ), key='climatedata_end_date' )
			
			climatedata_stations = st.text_input( 'Stations', value=st.session_state.get( 'climatedata_stations', '' ), key='climatedata_stations', disabled=(
						climatedata_mode != 'data'), placeholder='Comma-separated station IDs' )
			
			climatedata_data_types = st.text_input( 'Data Types', value=st.session_state.get( 'climatedata_data_types', '' ), key='climatedata_data_types', disabled=(
						climatedata_mode != 'data'), placeholder='Comma-separated datatype IDs' )
			
			page_c1, page_c2 = st.columns( 2 )
			with page_c1:
				climatedata_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'climatedata_limit', 25 ) ), step=1, key='climatedata_limit' )
			
			with page_c2:
				climatedata_offset = st.number_input( 'Offset', min_value=0, max_value=10000, value=int( st.session_state.get( 'climatedata_offset', 0 ) ), step=1, key='climatedata_offset', disabled=(
							climatedata_mode != 'datasets') )
			
			climatedata_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'climatedata_timeout', 20 ) ), step=1, key='climatedata_timeout' )
			
			st.caption( 'Datasets mode discovers NOAA climate datasets. Data mode retrieves '
			            'subsetted climate records from a selected dataset.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				climatedata_submit = st.button( 'Submit', key='climatedata_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='climatedata_clear', on_click=_clear_climatedata_state, width='stretch' )
			
			if climatedata_submit:
				st.session_state[ 'environmental_active_source' ] = 'NOAA Climate Data'
		
		# ----------------------------
		# -------- Expander NASA EONET
		# ----------------------------
		with st.expander( label='NASA EONET', icon='🌎', expanded=False ):
			EONET_MODES = [ 'events', 'categories' ]
			EONET_STATUSES = [ 'open', 'closed', 'all' ]
			
			def _clear_eonet_state( ) -> None:
				st.session_state[ 'eonet_clear_request' ] = True
			
			def _validate_eonet_date( name: str, value: str ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					return ''
				
				dt.datetime.strptime( text, '%Y-%m-%d' )
				return text
			
			def _validate_eonet_bbox( value: str ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					return ''
				
				parts = [ item.strip( ) for item in text.split( ',' ) if item.strip( ) ]
				if len( parts ) != 4:
					raise ValueError( 'Bounding Box must contain four comma-separated values: '
					                  'min_lon,max_lat,max_lon,min_lat.' )
				
				min_lon, max_lat, max_lon, min_lat = [ float( item ) for item in parts ]
				if min_lon < -180 or max_lon > 180:
					raise ValueError( 'Longitude values must be within -180 and 180.' )
				
				if min_lat < -90 or max_lat > 90:
					raise ValueError( 'Latitude values must be within -90 and 90.' )
				
				if min_lon >= max_lon:
					raise ValueError( 'Minimum longitude must be less than maximum longitude.' )
				
				if min_lat >= max_lat:
					raise ValueError( 'Minimum latitude must be less than maximum latitude.' )
				
				return f'{min_lon:g},{max_lat:g},{max_lon:g},{min_lat:g}'
			
			if 'eonet_results' not in st.session_state:
				st.session_state[ 'eonet_results' ] = { }
			
			if 'eonet_clear_request' not in st.session_state:
				st.session_state[ 'eonet_clear_request' ] = False
			
			if st.session_state.get( 'eonet_mode', 'events' ) not in EONET_MODES:
				st.session_state[ 'eonet_mode' ] = 'events'
			
			if st.session_state.get( 'eonet_status', 'open' ) not in EONET_STATUSES:
				st.session_state[ 'eonet_status' ] = 'open'
			
			if 'eonet_source' not in st.session_state:
				st.session_state[ 'eonet_source' ] = ''
			
			if 'eonet_category' not in st.session_state:
				st.session_state[ 'eonet_category' ] = ''
			
			if 'eonet_limit' not in st.session_state:
				st.session_state[ 'eonet_limit' ] = 25
			
			if 'eonet_days' not in st.session_state:
				st.session_state[ 'eonet_days' ] = 30
			
			if 'eonet_start_date' not in st.session_state:
				st.session_state[ 'eonet_start_date' ] = ''
			
			if 'eonet_end_date' not in st.session_state:
				st.session_state[ 'eonet_end_date' ] = ''
			
			if 'eonet_bbox' not in st.session_state:
				st.session_state[ 'eonet_bbox' ] = ''
			
			if 'eonet_timeout' not in st.session_state:
				st.session_state[ 'eonet_timeout' ] = 20
			
			if st.session_state.get( 'eonet_clear_request', False ):
				st.session_state[ 'eonet_mode' ] = 'events'
				st.session_state[ 'eonet_source' ] = ''
				st.session_state[ 'eonet_category' ] = ''
				st.session_state[ 'eonet_status' ] = 'open'
				st.session_state[ 'eonet_limit' ] = 25
				st.session_state[ 'eonet_days' ] = 30
				st.session_state[ 'eonet_start_date' ] = ''
				st.session_state[ 'eonet_end_date' ] = ''
				st.session_state[ 'eonet_bbox' ] = ''
				st.session_state[ 'eonet_timeout' ] = 20
				st.session_state[ 'eonet_results' ] = { }
				st.session_state[ 'eonet_clear_request' ] = False
			
			eonet_mode = st.selectbox( 'Mode', options=EONET_MODES, index=EONET_MODES.index( st.session_state.get( 'eonet_mode', 'events' ) ), key='eonet_mode' )
			
			eonet_status = st.selectbox( 'Status', options=EONET_STATUSES, index=EONET_STATUSES.index( st.session_state.get( 'eonet_status', 'open' ) ), key='eonet_status', disabled=(
						eonet_mode != 'events') )
			
			filter_c1, filter_c2 = st.columns( 2 )
			with filter_c1:
				eonet_source = st.text_input( 'Source', value=st.session_state.get( 'eonet_source', '' ), key='eonet_source', disabled=(
							eonet_mode != 'events'), placeholder='Optional source ID' )
			
			with filter_c2:
				eonet_category = st.text_input( 'Category', value=st.session_state.get( 'eonet_category', '' ), key='eonet_category', disabled=(
							eonet_mode != 'events'), placeholder='Optional category ID' )
			
			count_c1, count_c2 = st.columns( 2 )
			with count_c1:
				eonet_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'eonet_limit', 25 ) ), step=1, key='eonet_limit', disabled=(
							eonet_mode != 'events') )
			
			with count_c2:
				eonet_days = st.number_input( 'Days', min_value=1, max_value=3650, value=int( st.session_state.get( 'eonet_days', 30 ) ), step=1, key='eonet_days', disabled=(
							eonet_mode != 'events') )
			
			date_c1, date_c2 = st.columns( 2 )
			with date_c1:
				eonet_start_date = st.text_input( 'Start Date', value=st.session_state.get( 'eonet_start_date', '' ), key='eonet_start_date', disabled=(
							eonet_mode != 'events'), placeholder='YYYY-MM-DD' )
			
			with date_c2:
				eonet_end_date = st.text_input( 'End Date', value=st.session_state.get( 'eonet_end_date', '' ), key='eonet_end_date', disabled=(
							eonet_mode != 'events'), placeholder='YYYY-MM-DD' )
			
			eonet_bbox = st.text_input( 'Bounding Box', value=st.session_state.get( 'eonet_bbox', '' ), key='eonet_bbox', disabled=(
						eonet_mode != 'events'), placeholder='min_lon,max_lat,max_lon,min_lat' )
			
			eonet_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'eonet_timeout', 20 ) ), step=1, key='eonet_timeout' )
			
			st.caption( 'Events mode retrieves natural events. Categories mode lists the '
			            'available EONET event categories.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				eonet_submit = st.button( 'Submit', key='eonet_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='eonet_clear', on_click=_clear_eonet_state, width='stretch' )
			
			if eonet_submit:
				st.session_state[ 'environmental_active_source' ] = 'NASA EONET'
		
		# ----------------------------
		# -------- Expander EPA Envirofacts
		# ----------------------------
		with st.expander( label='EPA Envirofacts', icon='♻️', expanded=False ):
			ENVIROFACTS_TABLES = [ 'TRI_FACILITY', 'TRI_RELEASE', 'EF_W_EMISSIONS_SOURCE_GHG' ]
			ENVIROFACTS_STATE_CODES = [ '', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
					'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
					'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
					'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
					'WV', 'WI', 'WY', 'AS', 'GU', 'MP', 'PR', 'VI' ]
			
			def _clear_envirofacts_state( ) -> None:
				st.session_state[ 'envirofacts_clear_request' ] = True
			
			if 'envirofacts_results' not in st.session_state:
				st.session_state[ 'envirofacts_results' ] = { }
			
			if 'envirofacts_clear_request' not in st.session_state:
				st.session_state[ 'envirofacts_clear_request' ] = False
			
			if st.session_state.get( 'envirofacts_table_name', 'TRI_FACILITY' ) not in ENVIROFACTS_TABLES:
				st.session_state[ 'envirofacts_table_name' ] = 'TRI_FACILITY'
			
			if st.session_state.get( 'envirofacts_state_code', '' ) not in ENVIROFACTS_STATE_CODES:
				st.session_state[ 'envirofacts_state_code' ] = ''
			
			if 'envirofacts_facility_name' not in st.session_state:
				st.session_state[ 'envirofacts_facility_name' ] = ''
			
			if 'envirofacts_limit' not in st.session_state:
				st.session_state[ 'envirofacts_limit' ] = 25
			
			if 'envirofacts_timeout' not in st.session_state:
				st.session_state[ 'envirofacts_timeout' ] = 20
			
			if st.session_state.get( 'envirofacts_clear_request', False ):
				st.session_state[ 'envirofacts_table_name' ] = 'TRI_FACILITY'
				st.session_state[ 'envirofacts_state_code' ] = ''
				st.session_state[ 'envirofacts_facility_name' ] = ''
				st.session_state[ 'envirofacts_limit' ] = 25
				st.session_state[ 'envirofacts_timeout' ] = 20
				st.session_state[ 'envirofacts_results' ] = { }
				st.session_state[ 'envirofacts_clear_request' ] = False
			
			envirofacts_table_name = st.selectbox( 'Table', options=ENVIROFACTS_TABLES, index=ENVIROFACTS_TABLES.index( st.session_state.get( 'envirofacts_table_name', 'TRI_FACILITY' ) ), key='envirofacts_table_name' )
			
			envirofacts_state_code = st.selectbox( 'State Code', options=ENVIROFACTS_STATE_CODES, index=ENVIROFACTS_STATE_CODES.index( st.session_state.get( 'envirofacts_state_code', '' ) ), key='envirofacts_state_code', format_func=lambda value: 'All' if value == '' else value, help='Optional state or territory filter.' )
			
			envirofacts_facility_name = st.text_input( 'Facility Name Prefix', value=st.session_state.get( 'envirofacts_facility_name', '' ), key='envirofacts_facility_name', placeholder='Optional facility-name prefix' )
			
			envirofacts_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'envirofacts_limit', 25 ) ), step=1, key='envirofacts_limit' )
			
			envirofacts_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'envirofacts_timeout', 20 ) ), step=1, key='envirofacts_timeout' )
			
			st.caption( 'This first-pass Envirofacts wrapper focuses on a constrained set of '
			            'common tables so the output remains human-readable and easy to use.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				envirofacts_submit = st.button( 'Submit', key='envirofacts_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='envirofacts_clear', on_click=_clear_envirofacts_state, width='stretch' )
			
			if envirofacts_submit:
				st.session_state[ 'environmental_active_source' ] = 'EPA Envirofacts'
		
		# ----------------------------
		# -------- Expander NOAA Tides & Currents
		# ----------------------------
		with st.expander( label='NOAA Tides & Currents', icon='🌊', expanded=False ):
			TIDESANDCURRENTS_MODES = [ 'station', 'water-level', 'tide-predictions' ]
			TIDESANDCURRENTS_DATUMS = [ 'MLLW', 'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'NAVD', 'STND' ]
			TIDESANDCURRENTS_UNITS = [ 'metric', 'english' ]
			TIDESANDCURRENTS_TIME_ZONES = [ 'gmt', 'lst', 'lst_ldt' ]
			TIDESANDCURRENTS_INTERVALS = [ 'hilo', 'h', '1', '5', '6', '10', '15', '30', '60' ]
			
			def _clear_tidesandcurrents_state( ) -> None:
				st.session_state[ 'tidesandcurrents_clear_request' ] = True
			
			if 'tidesandcurrents_results' not in st.session_state:
				st.session_state[ 'tidesandcurrents_results' ] = { }
			
			if 'tidesandcurrents_clear_request' not in st.session_state:
				st.session_state[ 'tidesandcurrents_clear_request' ] = False
			
			if st.session_state.get( 'tidesandcurrents_mode', 'water-level' ) not in TIDESANDCURRENTS_MODES:
				st.session_state[ 'tidesandcurrents_mode' ] = 'water-level'
			
			if st.session_state.get( 'tidesandcurrents_datum', 'MLLW' ) not in TIDESANDCURRENTS_DATUMS:
				st.session_state[ 'tidesandcurrents_datum' ] = 'MLLW'
			
			if st.session_state.get( 'tidesandcurrents_units', 'metric' ) not in TIDESANDCURRENTS_UNITS:
				st.session_state[ 'tidesandcurrents_units' ] = 'metric'
			
			if st.session_state.get( 'tidesandcurrents_time_zone', 'gmt' ) not in TIDESANDCURRENTS_TIME_ZONES:
				st.session_state[ 'tidesandcurrents_time_zone' ] = 'gmt'
			
			if st.session_state.get( 'tidesandcurrents_interval', 'hilo' ) not in TIDESANDCURRENTS_INTERVALS:
				st.session_state[ 'tidesandcurrents_interval' ] = 'hilo'
			
			if 'tidesandcurrents_station_id' not in st.session_state:
				st.session_state[ 'tidesandcurrents_station_id' ] = ''
			
			if 'tidesandcurrents_begin_date' not in st.session_state:
				st.session_state[ 'tidesandcurrents_begin_date' ] = (
						dt.date.today( ) - dt.timedelta( days=1 ))
			
			if 'tidesandcurrents_end_date' not in st.session_state:
				st.session_state[ 'tidesandcurrents_end_date' ] = dt.date.today( )
			
			if 'tidesandcurrents_timeout' not in st.session_state:
				st.session_state[ 'tidesandcurrents_timeout' ] = 20
			
			if st.session_state.get( 'tidesandcurrents_clear_request', False ):
				st.session_state[ 'tidesandcurrents_mode' ] = 'water-level'
				st.session_state[ 'tidesandcurrents_station_id' ] = ''
				st.session_state[ 'tidesandcurrents_begin_date' ] = (
						dt.date.today( ) - dt.timedelta( days=1 ))
				st.session_state[ 'tidesandcurrents_end_date' ] = dt.date.today( )
				st.session_state[ 'tidesandcurrents_datum' ] = 'MLLW'
				st.session_state[ 'tidesandcurrents_units' ] = 'metric'
				st.session_state[ 'tidesandcurrents_time_zone' ] = 'gmt'
				st.session_state[ 'tidesandcurrents_interval' ] = 'hilo'
				st.session_state[ 'tidesandcurrents_timeout' ] = 20
				st.session_state[ 'tidesandcurrents_results' ] = { }
				st.session_state[ 'tidesandcurrents_clear_request' ] = False
			
			tac_mode = st.selectbox( 'Mode', options=TIDESANDCURRENTS_MODES, index=TIDESANDCURRENTS_MODES.index( st.session_state.get( 'tidesandcurrents_mode', 'water-level' ) ), key='tidesandcurrents_mode' )
			
			tac_station_id = st.text_input( 'Station ID', value=st.session_state.get( 'tidesandcurrents_station_id', '' ), key='tidesandcurrents_station_id', placeholder='Example: 8724580' )
			
			date_c1, date_c2 = st.columns( 2 )
			with date_c1:
				tac_begin_date = st.date_input( 'Begin Date', value=st.session_state.get( 'tidesandcurrents_begin_date', dt.date.today( ) - dt.timedelta( days=1 ) ), key='tidesandcurrents_begin_date', disabled=(
							tac_mode == 'station') )
			
			with date_c2:
				tac_end_date = st.date_input( 'End Date', value=st.session_state.get( 'tidesandcurrents_end_date', dt.date.today( ) ), key='tidesandcurrents_end_date', disabled=(
							tac_mode == 'station') )
			
			option_c1, option_c2 = st.columns( 2 )
			with option_c1:
				tac_datum = st.selectbox( 'Datum', options=TIDESANDCURRENTS_DATUMS, index=TIDESANDCURRENTS_DATUMS.index( st.session_state.get( 'tidesandcurrents_datum', 'MLLW' ) ), key='tidesandcurrents_datum', disabled=(
							tac_mode == 'station') )
			
			with option_c2:
				tac_units = st.selectbox( 'Units', options=TIDESANDCURRENTS_UNITS, index=TIDESANDCURRENTS_UNITS.index( st.session_state.get( 'tidesandcurrents_units', 'metric' ) ), key='tidesandcurrents_units', disabled=(
							tac_mode == 'station') )
			
			option_c3, option_c4 = st.columns( 2 )
			with option_c3:
				tac_time_zone = st.selectbox( 'Time Zone', options=TIDESANDCURRENTS_TIME_ZONES, index=TIDESANDCURRENTS_TIME_ZONES.index( st.session_state.get( 'tidesandcurrents_time_zone', 'gmt' ) ), key='tidesandcurrents_time_zone', disabled=(
							tac_mode == 'station') )
			
			with option_c4:
				tac_interval = st.selectbox( 'Interval', options=TIDESANDCURRENTS_INTERVALS, index=TIDESANDCURRENTS_INTERVALS.index( st.session_state.get( 'tidesandcurrents_interval', 'hilo' ) ), key='tidesandcurrents_interval', disabled=(
							tac_mode != 'tide-predictions') )
			
			tac_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'tidesandcurrents_timeout', 20 ) ), step=1, key='tidesandcurrents_timeout' )
			
			st.caption( 'NOAA CO-OPS station metadata uses Station ID only. Water-level and '
			            'tide-prediction requests require Station ID plus begin and end '
			            'dates.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				tac_submit = st.button( 'Submit', key='tidesandcurrents_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='tidesandcurrents_clear', on_click=_clear_tidesandcurrents_state, width='stretch' )
			
			if tac_submit:
				st.session_state[ 'environmental_active_source' ] = 'NOAA Tides & Currents'
		
		# ----------------------------
		# -------- Expander EPA UV Index
		# ----------------------------
		with st.expander( label='EPA UV Index', icon='☀️', expanded=False ):
			UVINDEX_MODES = [ 'daily-zip', 'daily-city-state', 'hourly-zip', 'hourly-city-state' ]
			UVINDEX_STATE_CODES = [ 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
					'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
					'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
					'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
					'WI', 'WY', 'AS', 'GU', 'MP', 'PR', 'VI' ]
			
			def _clear_uvindex_state( ) -> None:
				st.session_state[ 'uvindex_clear_request' ] = True
			
			if 'uvindex_results' not in st.session_state:
				st.session_state[ 'uvindex_results' ] = { }
			
			if 'uvindex_clear_request' not in st.session_state:
				st.session_state[ 'uvindex_clear_request' ] = False
			
			if st.session_state.get( 'uvindex_mode', 'daily-zip' ) not in UVINDEX_MODES:
				st.session_state[ 'uvindex_mode' ] = 'daily-zip'
			
			if st.session_state.get( 'uvindex_state', 'VA' ) not in UVINDEX_STATE_CODES:
				st.session_state[ 'uvindex_state' ] = 'VA'
			
			if 'uvindex_zip_code' not in st.session_state:
				st.session_state[ 'uvindex_zip_code' ] = ''
			
			if 'uvindex_city' not in st.session_state:
				st.session_state[ 'uvindex_city' ] = ''
			
			if 'uvindex_timeout' not in st.session_state:
				st.session_state[ 'uvindex_timeout' ] = 20
			
			if st.session_state.get( 'uvindex_clear_request', False ):
				st.session_state[ 'uvindex_mode' ] = 'daily-zip'
				st.session_state[ 'uvindex_zip_code' ] = ''
				st.session_state[ 'uvindex_city' ] = ''
				st.session_state[ 'uvindex_state' ] = 'VA'
				st.session_state[ 'uvindex_timeout' ] = 20
				st.session_state[ 'uvindex_results' ] = { }
				st.session_state[ 'uvindex_clear_request' ] = False
			
			uvindex_mode = st.selectbox( 'Mode', options=UVINDEX_MODES, index=UVINDEX_MODES.index( st.session_state.get( 'uvindex_mode', 'daily-zip' ) ), key='uvindex_mode' )
			
			uvindex_zip_code = st.text_input( 'Zip Code', value=st.session_state.get( 'uvindex_zip_code', '' ), key='uvindex_zip_code', disabled=(
						uvindex_mode not in [ 'daily-zip',
						'hourly-zip' ]), placeholder='Example: 22201' )
			
			city_c1, city_c2 = st.columns( 2 )
			with city_c1:
				uvindex_city = st.text_input( 'City', value=st.session_state.get( 'uvindex_city', '' ), key='uvindex_city', disabled=(
							uvindex_mode not in [ 'daily-city-state',
							'hourly-city-state' ]), placeholder='Example: Arlington' )
			
			with city_c2:
				uvindex_state = st.selectbox( 'State', options=UVINDEX_STATE_CODES, index=UVINDEX_STATE_CODES.index( st.session_state.get( 'uvindex_state', 'VA' ) ), key='uvindex_state', disabled=(
							uvindex_mode not in [ 'daily-city-state', 'hourly-city-state' ]) )
			
			uvindex_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'uvindex_timeout', 20 ) ), step=1, key='uvindex_timeout' )
			
			st.caption( 'UV Index forecasts can be retrieved hourly or daily, by ZIP code or '
			            'by city and state.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				uvindex_submit = st.button( 'Submit', key='uvindex_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='uvindex_clear', on_click=_clear_uvindex_state, width='stretch' )
			
			if uvindex_submit:
				st.session_state[ 'environmental_active_source' ] = 'EPA UV Index'
		
		# ----------------------------
		# -------- Expander Purple Air
		# ----------------------------
		with st.expander( label='Purple Air', icon='🟣', expanded=False ):
			PURPLEAIR_MODES = [ 'sensors', 'sensor' ]
			PURPLEAIR_FIELD_OPTIONS = [ 'name', 'pm2.5', 'temperature', 'humidity', 'latitude',
					'longitude', 'last_seen', 'location_type', 'model', 'hardware', 'pm2.5_cf_1_a',
					'pm2.5_cf_1_b', 'pressure', 'firmware_version', 'rssi' ]
			
			PURPLEAIR_SENSOR_LIST_DEFAULT_FIELDS = [ 'name', 'pm2.5', 'temperature', 'humidity',
					'latitude', 'longitude', 'last_seen', 'location_type' ]
			
			PURPLEAIR_SENSOR_DETAIL_DEFAULT_FIELDS = [ 'name', 'model', 'hardware', 'pm2.5_cf_1_a',
					'pm2.5_cf_1_b', 'temperature', 'humidity', 'pressure', 'latitude', 'longitude',
					'last_seen', 'firmware_version', 'rssi' ]
			
			PURPLEAIR_LOCATION_TYPES = { 'Outdoor': 0, 'Indoor': 1 }
			
			def _clear_purpleair_state( ) -> None:
				'''
					Purpose:
					--------
					Flag the PurpleAir expander state for reset on the next rerun.

					Parameters:
					-----------
					None

					Returns:
					--------
					None

				'''
				st.session_state[ 'purpleair_clear_request' ] = True
			
			def _coerce_optional_float( value: object ) -> float | None:
				'''
					Purpose:
					--------
					Convert an optional UI value into a float or None.

					Parameters:
					-----------
					value (object):
						Value supplied by a Streamlit input control.

					Returns:
					--------
					float | None:
						Float value when supplied; otherwise None.

				'''
				text = str( value or '' ).strip( )
				
				if not text:
					return None
				
				return float( text )
			
			if 'purpleair_results' not in st.session_state:
				st.session_state[ 'purpleair_results' ] = { }
			
			if 'purpleair_clear_request' not in st.session_state:
				st.session_state[ 'purpleair_clear_request' ] = False
			
			if st.session_state.get( 'purpleair_mode', 'sensors' ) not in PURPLEAIR_MODES:
				st.session_state[ 'purpleair_mode' ] = 'sensors'
			
			if 'purpleair_sensor_index' not in st.session_state:
				st.session_state[ 'purpleair_sensor_index' ] = ''
			
			if 'purpleair_nwlng' not in st.session_state:
				st.session_state[ 'purpleair_nwlng' ] = ''
			
			if 'purpleair_nwlat' not in st.session_state:
				st.session_state[ 'purpleair_nwlat' ] = ''
			
			if 'purpleair_selng' not in st.session_state:
				st.session_state[ 'purpleair_selng' ] = ''
			
			if 'purpleair_selat' not in st.session_state:
				st.session_state[ 'purpleair_selat' ] = ''
			
			if 'purpleair_location_type_label' not in st.session_state:
				st.session_state[ 'purpleair_location_type_label' ] = 'Outdoor'
			
			if 'purpleair_max_age' not in st.session_state:
				st.session_state[ 'purpleair_max_age' ] = 0
			
			if 'purpleair_modified_since' not in st.session_state:
				st.session_state[ 'purpleair_modified_since' ] = 0
			
			if 'purpleair_fields' not in st.session_state:
				st.session_state[ 'purpleair_fields' ] = PURPLEAIR_SENSOR_LIST_DEFAULT_FIELDS
			
			if 'purpleair_timeout' not in st.session_state:
				st.session_state[ 'purpleair_timeout' ] = 20
			
			if st.session_state.get( 'purpleair_clear_request', False ):
				st.session_state[ 'purpleair_mode' ] = 'sensors'
				st.session_state[ 'purpleair_sensor_index' ] = ''
				st.session_state[ 'purpleair_nwlng' ] = ''
				st.session_state[ 'purpleair_nwlat' ] = ''
				st.session_state[ 'purpleair_selng' ] = ''
				st.session_state[ 'purpleair_selat' ] = ''
				st.session_state[ 'purpleair_location_type_label' ] = 'Outdoor'
				st.session_state[ 'purpleair_max_age' ] = 0
				st.session_state[ 'purpleair_modified_since' ] = 0
				st.session_state[ 'purpleair_fields' ] = PURPLEAIR_SENSOR_LIST_DEFAULT_FIELDS
				st.session_state[ 'purpleair_timeout' ] = 20
				st.session_state[ 'purpleair_results' ] = { }
				st.session_state[ 'purpleair_clear_request' ] = False
			
			purpleair_mode = st.selectbox( 'Mode', options=PURPLEAIR_MODES, index=PURPLEAIR_MODES.index( st.session_state.get( 'purpleair_mode', 'sensors' ) ), key='purpleair_mode' )
			
			default_fields = (
					PURPLEAIR_SENSOR_DETAIL_DEFAULT_FIELDS if purpleair_mode == 'sensor' else PURPLEAIR_SENSOR_LIST_DEFAULT_FIELDS)
			
			if not st.session_state.get( 'purpleair_fields', [ ] ):
				st.session_state[ 'purpleair_fields' ] = default_fields
			
			purpleair_sensor_index = st.text_input( 'Sensor Index', value=st.session_state.get( 'purpleair_sensor_index', '' ), key='purpleair_sensor_index', disabled=(
						purpleair_mode != 'sensor'), placeholder='Example: 78307' )
			
			purpleair_fields = st.multiselect( 'Fields', options=PURPLEAIR_FIELD_OPTIONS, default=[
					field for field in st.session_state.get( 'purpleair_fields', default_fields ) if
					field in PURPLEAIR_FIELD_OPTIONS ], key='purpleair_fields', help='Select only the PurpleAir fields needed for the request.' )
			
			bbox_c1, bbox_c2 = st.columns( 2 )
			with bbox_c1:
				purpleair_nwlng = st.text_input( 'NW Longitude', value=st.session_state.get( 'purpleair_nwlng', '' ), key='purpleair_nwlng', disabled=(
							purpleair_mode != 'sensors') )
			
			with bbox_c2:
				purpleair_nwlat = st.text_input( 'NW Latitude', value=st.session_state.get( 'purpleair_nwlat', '' ), key='purpleair_nwlat', disabled=(
							purpleair_mode != 'sensors') )
			
			bbox_c3, bbox_c4 = st.columns( 2 )
			with bbox_c3:
				purpleair_selng = st.text_input( 'SE Longitude', value=st.session_state.get( 'purpleair_selng', '' ), key='purpleair_selng', disabled=(
							purpleair_mode != 'sensors') )
			
			with bbox_c4:
				purpleair_selat = st.text_input( 'SE Latitude', value=st.session_state.get( 'purpleair_selat', '' ), key='purpleair_selat', disabled=(
							purpleair_mode != 'sensors') )
			
			opt_c1, opt_c2 = st.columns( 2 )
			with opt_c1:
				location_labels = list( PURPLEAIR_LOCATION_TYPES.keys( ) )
				current_location_label = st.session_state.get( 'purpleair_location_type_label', 'Outdoor' )
				
				if current_location_label not in location_labels:
					current_location_label = 'Outdoor'
				
				purpleair_location_type_label = st.selectbox( 'Location Type', options=location_labels, index=location_labels.index( current_location_label ), key='purpleair_location_type_label', disabled=(
							purpleair_mode != 'sensors') )
			
			with opt_c2:
				purpleair_max_age = st.number_input( 'Max Age', min_value=0, max_value=1000000, value=int( st.session_state.get( 'purpleair_max_age', 0 ) ), step=1, key='purpleair_max_age', disabled=(
							purpleair_mode != 'sensors'), help='Maximum age filter in seconds. Use 0 to preserve broad behavior.' )
			
			purpleair_modified_since = st.number_input( 'Modified Since', min_value=0, max_value=2147483647, value=int( st.session_state.get( 'purpleair_modified_since', 0 ) ), step=1, key='purpleair_modified_since', disabled=(
						purpleair_mode != 'sensors'), help='Unix timestamp filter. Use 0 to disable the filter.' )
			
			purpleair_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'purpleair_timeout', 20 ) ), step=1, key='purpleair_timeout' )
			
			st.caption( 'PurpleAir requires an API key and uses a points system. Select only '
			            'the fields needed for the request.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				purpleair_submit = st.button( 'Submit', key='purpleair_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='purpleair_clear', on_click=_clear_purpleair_state, width='stretch' )
			
			if purpleair_submit:
				st.session_state[ 'environmental_active_source' ] = 'Purple Air'
		
		# ----------------------------
		# -------- Expander Open Air Quality
		# ----------------------------
		with st.expander( label='Open Air Quality', icon='🌬️', expanded=False ):
			OPENAQ_MODES = [ 'countries', 'providers', 'parameters', 'locations', 'latest',
					'parameter_latest' ]
			
			def _clear_openaq_state( ) -> None:
				st.session_state[ 'openaq_clear_request' ] = True
			
			def _coerce_optional_integer( value: object ) -> int | None:
				text = str( value or '' ).strip( )
				if not text:
					return None
				
				return int( text )
			
			if 'openaq_results' not in st.session_state:
				st.session_state[ 'openaq_results' ] = { }
			
			if 'openaq_clear_request' not in st.session_state:
				st.session_state[ 'openaq_clear_request' ] = False
			
			if st.session_state.get( 'openaq_mode', 'locations' ) not in OPENAQ_MODES:
				st.session_state[ 'openaq_mode' ] = 'locations'
			
			if 'openaq_location_id' not in st.session_state:
				st.session_state[ 'openaq_location_id' ] = ''
			
			if 'openaq_parameter_id' not in st.session_state:
				st.session_state[ 'openaq_parameter_id' ] = ''
			
			if 'openaq_country_id' not in st.session_state:
				st.session_state[ 'openaq_country_id' ] = ''
			
			if 'openaq_coordinates' not in st.session_state:
				st.session_state[ 'openaq_coordinates' ] = ''
			
			if 'openaq_radius' not in st.session_state:
				st.session_state[ 'openaq_radius' ] = 25000
			
			if 'openaq_providers_id' not in st.session_state:
				st.session_state[ 'openaq_providers_id' ] = ''
			
			if 'openaq_parameters_id' not in st.session_state:
				st.session_state[ 'openaq_parameters_id' ] = ''
			
			if 'openaq_limit' not in st.session_state:
				st.session_state[ 'openaq_limit' ] = 25
			
			if 'openaq_page' not in st.session_state:
				st.session_state[ 'openaq_page' ] = 1
			
			if 'openaq_timeout' not in st.session_state:
				st.session_state[ 'openaq_timeout' ] = 20
			
			if st.session_state.get( 'openaq_clear_request', False ):
				st.session_state[ 'openaq_mode' ] = 'locations'
				st.session_state[ 'openaq_location_id' ] = ''
				st.session_state[ 'openaq_parameter_id' ] = ''
				st.session_state[ 'openaq_country_id' ] = ''
				st.session_state[ 'openaq_coordinates' ] = ''
				st.session_state[ 'openaq_radius' ] = 25000
				st.session_state[ 'openaq_providers_id' ] = ''
				st.session_state[ 'openaq_parameters_id' ] = ''
				st.session_state[ 'openaq_limit' ] = 25
				st.session_state[ 'openaq_page' ] = 1
				st.session_state[ 'openaq_timeout' ] = 20
				st.session_state[ 'openaq_results' ] = { }
				st.session_state[ 'openaq_clear_request' ] = False
			
			openaq_mode = st.selectbox( 'Mode', options=OPENAQ_MODES, index=OPENAQ_MODES.index( st.session_state.get( 'openaq_mode', 'locations' ) ), key='openaq_mode', help=(
				'Select one of the wrapper-supported OpenAQ v3 '
				'modes. '
				'Discovery modes return lookup tables; latest modes '
				'return '
				'measurement records.') )
			
			openaq_location_id = st.text_input( 'Location ID', value=st.session_state.get( 'openaq_location_id', '' ), key='openaq_location_id', disabled=(
						openaq_mode != 'latest'), placeholder='Example: 8118' )
			
			openaq_parameter_id = st.text_input( 'Parameter ID', value=st.session_state.get( 'openaq_parameter_id', '' ), key='openaq_parameter_id', disabled=(
						openaq_mode != 'parameter_latest'), placeholder='Example: 2' )
			
			openaq_country_id = st.text_input( 'Country ID', value=st.session_state.get( 'openaq_country_id', '' ), key='openaq_country_id', disabled=(
						openaq_mode != 'locations'), placeholder='Optional numeric country ID' )
			
			openaq_coordinates = st.text_input( 'Coordinates', value=st.session_state.get( 'openaq_coordinates', '' ), key='openaq_coordinates', disabled=(
						openaq_mode != 'locations'), placeholder='latitude,longitude' )
			
			openaq_radius = st.number_input( 'Radius (meters)', min_value=1, max_value=500000, value=int( st.session_state.get( 'openaq_radius', 25000 ) ), step=1, key='openaq_radius', disabled=(
						openaq_mode != 'locations') )
			
			filter_c1, filter_c2 = st.columns( 2 )
			with filter_c1:
				openaq_providers_id = st.text_input( 'Providers ID', value=st.session_state.get( 'openaq_providers_id', '' ), key='openaq_providers_id', disabled=(
							openaq_mode not in [ 'countries',
							'locations' ]), placeholder='Optional provider ID' )
			
			with filter_c2:
				openaq_parameters_id = st.text_input( 'Parameters ID', value=st.session_state.get( 'openaq_parameters_id', '' ), key='openaq_parameters_id', disabled=(
							openaq_mode not in [ 'countries',
							'locations' ]), placeholder='Optional parameter ID' )
			
			page_c1, page_c2 = st.columns( 2 )
			with page_c1:
				openaq_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'openaq_limit', 25 ) ), step=1, key='openaq_limit', disabled=(
							openaq_mode == 'latest') )
			
			with page_c2:
				openaq_page = st.number_input( 'Page', min_value=1, max_value=10000, value=int( st.session_state.get( 'openaq_page', 1 ) ), step=1, key='openaq_page', disabled=(
							openaq_mode == 'latest') )
			
			openaq_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'openaq_timeout', 20 ) ), step=1, key='openaq_timeout' )
			
			st.caption( 'OpenAQ v3 requires an API key. Discovery modes return countries, '
			            'providers, parameters, or locations. Latest modes retrieve current '
			            'measurements by location or parameter.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				openaq_submit = st.button( 'Submit', key='openaq_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='openaq_clear', on_click=_clear_openaq_state )
			
			if openaq_submit:
				st.session_state[ 'environmental_active_source' ] = 'Open Air Quality'
		
		# ----------------------------
		# -------- Expander NASA FIRMS
		# ----------------------------
		with st.expander( label='NASA FIRMS', icon='🔥', expanded=False ):
			FIRMS_MODES = [ 'area', 'data-availability' ]
			FIRMS_SOURCES = [ 'LANDSAT_NRT', 'MODIS_NRT', 'MODIS_SP', 'VIIRS_NOAA20_NRT',
					'VIIRS_NOAA20_SP', 'VIIRS_NOAA21_NRT', 'VIIRS_SNPP_NRT', 'VIIRS_SNPP_SP' ]
			
			FIRMS_SENSORS = [ 'ALL', 'LANDSAT_NRT', 'MODIS_NRT', 'MODIS_SP', 'VIIRS_NOAA20_NRT',
					'VIIRS_NOAA20_SP', 'VIIRS_NOAA21_NRT', 'VIIRS_SNPP_NRT', 'VIIRS_SNPP_SP' ]
			
			def _clear_firms_state( ) -> None:
				st.session_state[ 'firms_clear_request' ] = True
			
			def _validate_firms_area_coordinates( value: str ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					raise ValueError( 'Area Coordinates must be "world" or west,south,east,'
					                  'north.' )
				
				if text.lower( ) == 'world':
					return 'world'
				
				parts = [ item.strip( ) for item in text.split( ',' ) if item.strip( ) ]
				
				if len( parts ) != 4:
					raise ValueError( 'Area Coordinates must be "world" or four comma-separated values: '
					                  'west,south,east,north.' )
				
				west, south, east, north = [ float( item ) for item in parts ]
				
				if west < -180 or east > 180:
					raise ValueError( 'Longitude bounds must be within -180 and 180.' )
				
				if south < -90 or north > 90:
					raise ValueError( 'Latitude bounds must be within -90 and 90.' )
				
				if west >= east:
					raise ValueError( 'West longitude must be less than east longitude.' )
				
				if south >= north:
					raise ValueError( 'South latitude must be less than north latitude.' )
				
				return f'{west:g},{south:g},{east:g},{north:g}'
			
			if 'firms_results' not in st.session_state:
				st.session_state[ 'firms_results' ] = { }
			
			if 'firms_clear_request' not in st.session_state:
				st.session_state[ 'firms_clear_request' ] = False
			
			if st.session_state.get( 'firms_mode', 'area' ) not in FIRMS_MODES:
				st.session_state[ 'firms_mode' ] = 'area'
			
			if st.session_state.get( 'firms_source', 'VIIRS_SNPP_NRT' ) not in FIRMS_SOURCES:
				st.session_state[ 'firms_source' ] = 'VIIRS_SNPP_NRT'
			
			if st.session_state.get( 'firms_sensor', 'ALL' ) not in FIRMS_SENSORS:
				st.session_state[ 'firms_sensor' ] = 'ALL'
			
			if 'firms_area_coordinates' not in st.session_state:
				st.session_state[ 'firms_area_coordinates' ] = 'world'
			
			if 'firms_day_range' not in st.session_state:
				st.session_state[ 'firms_day_range' ] = 1
			
			if 'firms_date' not in st.session_state:
				st.session_state[ 'firms_date' ] = ''
			
			if 'firms_timeout' not in st.session_state:
				st.session_state[ 'firms_timeout' ] = 20
			
			if st.session_state.get( 'firms_clear_request', False ):
				st.session_state[ 'firms_mode' ] = 'area'
				st.session_state[ 'firms_source' ] = 'VIIRS_SNPP_NRT'
				st.session_state[ 'firms_area_coordinates' ] = 'world'
				st.session_state[ 'firms_day_range' ] = 1
				st.session_state[ 'firms_date' ] = ''
				st.session_state[ 'firms_sensor' ] = 'ALL'
				st.session_state[ 'firms_timeout' ] = 20
				st.session_state[ 'firms_results' ] = { }
				st.session_state[ 'firms_clear_request' ] = False
			
			firms_mode = st.selectbox( 'Mode', options=FIRMS_MODES, index=FIRMS_MODES.index( st.session_state.get( 'firms_mode', 'area' ) ), key='firms_mode' )
			
			firms_source = st.selectbox( 'Source', options=FIRMS_SOURCES, index=FIRMS_SOURCES.index( st.session_state.get( 'firms_source', 'VIIRS_SNPP_NRT' ) ), key='firms_source', disabled=(
						firms_mode != 'area') )
			
			firms_area_coordinates = st.text_input( 'Area Coordinates', value=st.session_state.get( 'firms_area_coordinates', 'world' ), key='firms_area_coordinates', disabled=(
						firms_mode != 'area'), placeholder='west,south,east,north or world' )
			
			firms_day_range = st.number_input( 'Day Range', min_value=1, max_value=5, value=int( st.session_state.get( 'firms_day_range', 1 ) ), step=1, key='firms_day_range', disabled=(
						firms_mode != 'area'), help='NASA FIRMS Area API supports 1 through 5 days per request.' )
			
			firms_date = st.text_input( 'Date', value=st.session_state.get( 'firms_date', '' ), key='firms_date', disabled=(
						firms_mode != 'area'), placeholder='YYYY-MM-DD or blank for most recent' )
			
			firms_sensor = st.selectbox( 'Sensor', options=FIRMS_SENSORS, index=FIRMS_SENSORS.index( st.session_state.get( 'firms_sensor', 'ALL' ) ), key='firms_sensor', disabled=(
						firms_mode != 'data-availability') )
			
			firms_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'firms_timeout', 20 ) ), step=1, key='firms_timeout' )
			
			st.caption( 'NASA FIRMS requires a MAP_KEY in FIRMS_MAP_KEY or '
			            'NASA_FIRMS_MAP_KEY. '
			            'Area mode retrieves active fire detections. Data availability mode '
			            'returns supported date ranges by sensor.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				firms_submit = st.button( 'Submit', key='firms_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='firms_clear', on_click=_clear_firms_state, width='stretch' )
			
			if firms_submit:
				st.session_state[ 'environmental_active_source' ] = 'NASA FIRMS'
		
		# ----------------------------
		# -------- Expander USGS Water Data
		# ----------------------------
		with st.expander( label='USGS Water Data', icon='💧', expanded=False ):
			USGSWATERDATA_MODES = [ 'monitoring-locations', 'time-series-metadata',
					'latest-continuous', 'latest-daily' ]
			
			USGSWATERDATA_STATE_CODES = [ '', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
					'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
					'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
					'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
					'WV', 'WI', 'WY', 'AS', 'GU', 'MP', 'PR', 'VI' ]
			
			USGSWATERDATA_SITE_TYPES = [ '', 'AG', 'AT', 'ES', 'FA', 'FA-CI', 'FA-CS', 'FA-DV',
					'FA-FON', 'FA-GC', 'FA-HP', 'FA-LF', 'FA-OIL', 'FA-QC', 'FA-SEW', 'FA-SPS',
					'FA-TEP', 'FA-WDS', 'FA-WIW', 'GL', 'GW', 'GW-CR', 'GW-EX', 'GW-HZ', 'GW-IW',
					'GW-MW', 'GW-TH', 'LK', 'OC', 'OC-CO', 'SB', 'SP', 'ST', 'ST-CA', 'ST-DCH',
					'ST-TS', 'WE' ]
			
			def _clear_usgswaterdata_state( ) -> None:
				st.session_state[ 'usgswaterdata_clear_request' ] = True
			
			if 'usgswaterdata_results' not in st.session_state:
				st.session_state[ 'usgswaterdata_results' ] = { }
			
			if 'usgswaterdata_clear_request' not in st.session_state:
				st.session_state[ 'usgswaterdata_clear_request' ] = False
			
			if st.session_state.get( 'usgswaterdata_mode', 'monitoring-locations' ) not in USGSWATERDATA_MODES:
				st.session_state[ 'usgswaterdata_mode' ] = 'monitoring-locations'
			
			if st.session_state.get( 'usgswaterdata_state_code', '' ) not in USGSWATERDATA_STATE_CODES:
				st.session_state[ 'usgswaterdata_state_code' ] = ''
			
			if st.session_state.get( 'usgswaterdata_site_type', '' ) not in USGSWATERDATA_SITE_TYPES:
				st.session_state[ 'usgswaterdata_site_type' ] = ''
			
			if 'usgswaterdata_monitoring_location_id' not in st.session_state:
				st.session_state[ 'usgswaterdata_monitoring_location_id' ] = ''
			
			if 'usgswaterdata_county_code' not in st.session_state:
				st.session_state[ 'usgswaterdata_county_code' ] = ''
			
			if 'usgswaterdata_parameter_code' not in st.session_state:
				st.session_state[ 'usgswaterdata_parameter_code' ] = ''
			
			if 'usgswaterdata_limit' not in st.session_state:
				st.session_state[ 'usgswaterdata_limit' ] = 25
			
			if 'usgswaterdata_timeout' not in st.session_state:
				st.session_state[ 'usgswaterdata_timeout' ] = 20
			
			if st.session_state.get( 'usgswaterdata_clear_request', False ):
				st.session_state[ 'usgswaterdata_mode' ] = 'monitoring-locations'
				st.session_state[ 'usgswaterdata_monitoring_location_id' ] = ''
				st.session_state[ 'usgswaterdata_state_code' ] = ''
				st.session_state[ 'usgswaterdata_county_code' ] = ''
				st.session_state[ 'usgswaterdata_site_type' ] = ''
				st.session_state[ 'usgswaterdata_parameter_code' ] = ''
				st.session_state[ 'usgswaterdata_limit' ] = 25
				st.session_state[ 'usgswaterdata_timeout' ] = 20
				st.session_state[ 'usgswaterdata_results' ] = { }
				st.session_state[ 'usgswaterdata_clear_request' ] = False
			
			usgswd_mode = st.selectbox( 'Mode', options=USGSWATERDATA_MODES, index=USGSWATERDATA_MODES.index( st.session_state.get( 'usgswaterdata_mode', 'monitoring-locations' ) ), key='usgswaterdata_mode' )
			
			usgswd_monitoring_location_id = st.text_input( 'Monitoring Location ID', value=st.session_state.get( 'usgswaterdata_monitoring_location_id', '' ), key='usgswaterdata_monitoring_location_id', placeholder='Example: USGS-01491000' )
			
			meta_c1, meta_c2 = st.columns( 2 )
			with meta_c1:
				usgswd_state_code = st.selectbox( 'State Code', options=USGSWATERDATA_STATE_CODES, index=USGSWATERDATA_STATE_CODES.index( st.session_state.get( 'usgswaterdata_state_code', '' ) ), key='usgswaterdata_state_code', disabled=(
							usgswd_mode != 'monitoring-locations'), format_func=lambda value: 'All' if value == '' else value )
			
			with meta_c2:
				usgswd_county_code = st.text_input( 'County Code', value=st.session_state.get( 'usgswaterdata_county_code', '' ), key='usgswaterdata_county_code', disabled=(
							usgswd_mode != 'monitoring-locations'), placeholder='Optional county code' )
			
			usgswd_site_type = st.selectbox( 'Site Type', options=USGSWATERDATA_SITE_TYPES, index=USGSWATERDATA_SITE_TYPES.index( st.session_state.get( 'usgswaterdata_site_type', '' ) ), key='usgswaterdata_site_type', disabled=(
						usgswd_mode != 'monitoring-locations'), format_func=lambda value: 'All' if value == '' else value, help='USGS site type code, such as ST for stream or GW for well.' )
			
			usgswd_parameter_code = st.text_input( 'Parameter Code', value=st.session_state.get( 'usgswaterdata_parameter_code', '' ), key='usgswaterdata_parameter_code', disabled=(
						usgswd_mode == 'monitoring-locations'), placeholder='Example: '
			                                                                '00060' )
			
			usgswd_limit = st.number_input( 'Limit', min_value=1, max_value=50000, value=int( st.session_state.get( 'usgswaterdata_limit', 25 ) ), step=1, key='usgswaterdata_limit' )
			
			usgswd_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'usgswaterdata_timeout', 20 ) ), step=1, key='usgswaterdata_timeout' )
			
			st.caption( 'USGS Water Data uses the modern OGC API collections for monitoring '
			            'locations, time-series metadata, latest continuous values, '
			            'and latest '
			            'daily values.' )
			
			btn_c1, btn_c2 = st.columns( 2 )
			with btn_c1:
				usgswd_submit = st.button( 'Submit', key='usgswaterdata_submit', width='stretch' )
			
			with btn_c2:
				st.button( 'Clear', key='usgswaterdata_clear', on_click=_clear_usgswaterdata_state, width='stretch' )
			
			if usgswd_submit:
				st.session_state[ 'environmental_active_source' ] = 'USGS Water Data'
	
	with right:
		st.markdown( '### Results' )
		active_source = st.session_state.get( 'environmental_active_source', '' )
		if not active_source:
			st.info( 'Select a source, configure the request, and submit it to display results.' )
		else:
			st.caption( f'Active Source: {active_source}' )
		
		# -------- Air Now
		if active_source == 'Air Now':
			if airnow_submit:
				try:
					f = AirNow( )
					
					latitude_value = None
					longitude_value = None
					
					if str( airnow_latitude or '' ).strip( ):
						latitude_value = float( airnow_latitude )
					
					if str( airnow_longitude or '' ).strip( ):
						longitude_value = float( airnow_longitude )
					
					result = f.fetch( mode=str( airnow_mode ),
						zip_code=str( airnow_zip_code ).strip( ), latitude=latitude_value,
						longitude=longitude_value, date=str( airnow_date ),
						distance=int( airnow_distance ), time=int( airnow_timeout ) )
					
					st.session_state[ 'airnow_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'AirNow request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'airnow_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'zipCode' in params:
						st.markdown( f"**Zip Code:** {params.get( 'zipCode', '' )}" )
					if 'distance' in params:
						st.markdown( f"**Distance:** {params.get( 'distance', '' )} miles" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_aqi = summary.get( 'max_aqi', None )
						st.metric( 'Peak AQI', '' if max_aqi is None else str( max_aqi ) )
					
					with sum_c3:
						top_category = str( summary.get( 'top_category', '' ) or '' )
						if top_category:
							st.markdown( f"**Category:** {top_category}" )
						else:
							st.markdown( '**Category:** N/A' )
					
					reporting_area = str( summary.get( 'reporting_area', '' ) or '' )
					dominant_parameter = str( summary.get( 'dominant_parameter', '' ) or '' )
					
					if reporting_area:
						st.markdown( f"**Reporting Area:** {reporting_area}" )
					if dominant_parameter:
						st.markdown( f"**Dominant Parameter in Result Set:** {dominant_parameter}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Air Quality Results' )
					df_airnow = pd.DataFrame( rows )
					
					if not df_airnow.empty:
						st.dataframe( df_airnow, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Reporting Area', '' ) or f'Record {idx}' )
							aqi = item.get( 'AQI', '' )
							parameter = str( item.get( 'Parameter Name', '' ) or '' )
							
							with st.expander( f'Record {idx}: AQI {aqi} - {label} - {parameter}', expanded=False ):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									st.markdown( f"**Date Observed:** "
									             f"{item.get( 'Date Observed', '' )}" )
									st.markdown( f"**Hour Observed:** "
									             f"{item.get( 'Hour Observed', '' )}" )
									st.markdown( f"**Reporting Area:** "
									             f"{item.get( 'Reporting Area', '' )}" )
									st.markdown( f"**State Code:** {item.get( 'State Code', '' )}" )
								
								with right_c:
									st.markdown( f"**Parameter Name:** "
									             f"{item.get( 'Parameter Name', '' )}" )
									st.markdown( f"**AQI:** {item.get( 'AQI', '' )}" )
									st.markdown( f"**Category:** "
									             f"{item.get( 'Category', '' )}" )
									st.markdown( f"**Action Day:** {item.get( 'Action Day', '' )}" )
								
								discussion = str( item.get( 'Discussion', '' ) or '' )
								if discussion:
									st.markdown( f"**Discussion:** {discussion}" )
					else:
						st.info( 'No displayable AirNow rows were found.' )
				else:
					st.info( 'No AirNow records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='Air Now', result=result )
		
		# -------- NOAA Climate Data
		if active_source == 'NOAA Climate Data':
			if climatedata_submit:
				try:
					if climatedata_start_date > climatedata_end_date:
						raise ValueError( 'Start Date must be on or before End Date.' )
					
					if climatedata_mode == 'data' and not str( climatedata_dataset ).strip( ):
						raise ValueError( 'Dataset is required for data mode.' )
					
					f = ClimateData( )
					result = f.fetch( mode=str( climatedata_mode ), keyword=str( climatedata_keyword ).strip( ), dataset=str( climatedata_dataset ).strip( ), start_date=str( climatedata_start_date ), end_date=str( climatedata_end_date ), stations=str( climatedata_stations ).strip( ), data_types=str( climatedata_data_types ).strip( ), limit=int( climatedata_limit ), offset=int( climatedata_offset ), time=int( climatedata_timeout ) )
					
					st.session_state[ 'climatedata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NOAA Climate Data request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'climatedata_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'dataset' in params:
						st.markdown( f"**Dataset:** {params.get( 'dataset', '' )}" )
					if 'stations' in params:
						st.markdown( f"**Stations:** {params.get( 'stations', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_dataset = str( summary.get( 'first_dataset', '' ) or '' )
						if first_dataset:
							st.markdown( f"**Dataset/Type:** {first_dataset}" )
						else:
							st.markdown( '**Dataset/Type:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Climate Results' )
					df_climatedata = pd.DataFrame( rows )
					if not df_climatedata.empty:
						st.dataframe( df_climatedata, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Title', '' ) or item.get( 'Station', '' ) or item.get( 'Date', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable NOAA climate rows were found.' )
				else:
					st.info( 'No NOAA climate records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='NOAA Climate Data', result=result )
		
		# -------- NASA EONET
		if active_source == 'NASA EONET':
			if eonet_submit:
				try:
					clean_start_date = _validate_eonet_date( 'Start Date', eonet_start_date )
					clean_end_date = _validate_eonet_date( 'End Date', eonet_end_date )
					
					if bool( clean_start_date ) != bool( clean_end_date ):
						raise ValueError( 'Start Date and End Date must either both be supplied or '
						                  'both be blank.' )
					
					if clean_start_date and clean_end_date:
						start_value = dt.datetime.strptime( clean_start_date, '%Y-%m-%d' ).date( )
						end_value = dt.datetime.strptime( clean_end_date, '%Y-%m-%d' ).date( )
						
						if start_value > end_value:
							raise ValueError( 'Start Date must be on or before End Date.' )
					
					clean_bbox = _validate_eonet_bbox( eonet_bbox )
					
					f = EoNet( )
					result = f.fetch( mode=str( eonet_mode ), source=str( eonet_source ).strip( ), category=str( eonet_category ).strip( ), status=str( eonet_status ).strip( ), limit=int( eonet_limit ), days=int( eonet_days ), start_date=clean_start_date, end_date=clean_end_date, bbox=clean_bbox, time=int( eonet_timeout ) )
					
					st.session_state[ 'eonet_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NASA EONET request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'eonet_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'category' in params:
						st.markdown( f"**Category Filter:** {params.get( 'category', '' )}" )
					if 'status' in params:
						st.markdown( f"**Status:** {params.get( 'status', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						st.metric( 'Open Records', int( summary.get( 'open_count', 0 ) or 0 ) )
					
					with sum_c3:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Event:** {first_title}" )
						else:
							st.markdown( '**First Event:** N/A' )
					
					first_categories = str( summary.get( 'first_categories', '' ) or '' )
					if first_categories:
						st.markdown( f"**Categories:** {first_categories}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### EONET Results' )
					df_eonet = pd.DataFrame( rows )
					
					if not df_eonet.empty:
						st.dataframe( df_eonet, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = (item.get( 'Latitude', None ) or item.get( 'lat', None ))
							longitude = (item.get( 'Longitude', None ) or item.get( 'lon', None ))
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': float( latitude ),
										'lon': float( longitude ) } )
						
						if map_rows:
							st.markdown( '#### Event Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Title', '' ) or item.get( 'Category', '' ) or item.get( 'ID', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable EONET rows were found.' )
				else:
					st.info( 'No EONET records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='NASA EONET', result=result )
		
		# -------- EPA Envirofacts
		if active_source == 'EPA Envirofacts':
			if envirofacts_submit:
				try:
					f = EnviroFacts( )
					result = f.fetch( table_name=str( envirofacts_table_name ).strip( ), state_code=str( envirofacts_state_code ).strip( ), facility_name=str( envirofacts_facility_name ).strip( ), limit=int( envirofacts_limit ), time=int( envirofacts_timeout ) )
					
					st.session_state[ 'envirofacts_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Envirofacts request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'envirofacts_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'table_name' in result:
						st.markdown( f"**Table:** {result.get( 'table_name', '' )}" )
				
				with meta_c2:
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_facility = str( summary.get( 'first_facility', '' ) or '' )
						if first_facility:
							st.markdown( f"**First Facility:** {first_facility}" )
						else:
							st.markdown( '**First Facility:** N/A' )
					
					with sum_c3:
						first_state = str( summary.get( 'first_state', '' ) or '' )
						if first_state:
							st.markdown( f"**First State:** {first_state}" )
						else:
							st.markdown( '**First State:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Envirofacts Results' )
					df_envirofacts = pd.DataFrame( rows )
					
					if not df_envirofacts.empty:
						st.dataframe( df_envirofacts, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Facility Name', '' ) or item.get( 'Primary Name', '' ) or item.get( 'Name', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable Envirofacts rows were found.' )
				else:
					st.info( 'No Envirofacts records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			_promote_environmental_result( source='EPA Envirofacts', result=result )
		
		# -------- NOAA Tides & Currents
		if active_source == 'NOAA Tides & Currents':
			if tac_submit:
				try:
					if not str( tac_station_id or '' ).strip( ):
						raise ValueError( 'Station ID is required for NOAA Tides & Currents.' )
					
					if tac_mode != 'station' and tac_begin_date > tac_end_date:
						raise ValueError( 'Begin Date must be on or before End Date.' )
					
					f = TidesAndCurrents( )
					result = f.fetch( mode=str( tac_mode ).strip( ), station_id=str( tac_station_id ).strip( ), begin_date=dt.datetime.strftime( tac_begin_date, '%Y%m%d' ), end_date=dt.datetime.strftime( tac_end_date, '%Y%m%d' ), datum=str( tac_datum ).strip( ), units=str( tac_units ).strip( ), time_zone=str( tac_time_zone ).strip( ), interval=str( tac_interval ).strip( ), time=int( tac_timeout ) )
					
					st.session_state[ 'tidesandcurrents_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NOAA Tides & Currents request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'tidesandcurrents_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'station_id' in result:
						st.markdown( f"**Station ID:** {result.get( 'station_id', '' )}" )
				
				with meta_c2:
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_station = str( summary.get( 'first_station', '' ) or '' )
						if first_station:
							st.markdown( f"**Station:** {first_station}" )
						else:
							st.markdown( '**Station:** N/A' )
					
					with sum_c3:
						first_value = str( summary.get( 'first_value', '' ) or '' )
						if first_value:
							st.markdown( f"**Sample Value:** {first_value}" )
						else:
							st.markdown( '**Sample Value:** N/A' )
					
					first_time = str( summary.get( 'first_time', '' ) or '' )
					if first_time:
						st.markdown( f"**First Time:** {first_time}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Tides & Currents Results' )
					df_tac = pd.DataFrame( rows )
					
					if not df_tac.empty:
						st.dataframe( df_tac, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Name', '' ) or item.get( 'Time', '' ) or item.get( 'T', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable Tides & Currents rows were found.' )
				else:
					st.info( 'No Tides & Currents records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='NOAA Tides & Currents', result=result )
		
		# -------- EPA UV Index
		if active_source == 'EPA UV Index':
			if uvindex_submit:
				try:
					if uvindex_mode in [ 'daily-zip', 'hourly-zip' ]:
						if not str( uvindex_zip_code or '' ).strip( ):
							raise ValueError( 'Zip Code is required for ZIP-based UV Index modes.' )
						
						if not re.fullmatch( r'\d{5}(?:-\d{4})?', str( uvindex_zip_code ).strip( ) ):
							raise ValueError( 'Zip Code must be a valid 5-digit ZIP or ZIP+4 value.' )
					
					if uvindex_mode in [ 'daily-city-state', 'hourly-city-state' ]:
						if not str( uvindex_city or '' ).strip( ):
							raise ValueError( 'City is required for city/state UV Index modes.' )
						
						if not str( uvindex_state or '' ).strip( ):
							raise ValueError( 'State is required for city/state UV Index modes.' )
					
					f = UvIndex( )
					result = f.fetch( mode=str( uvindex_mode ), zip_code=str( uvindex_zip_code ).strip( ), city=str( uvindex_city ).strip( ), state=str( uvindex_state ).strip( ), time=int( uvindex_timeout ) )
					
					st.session_state[ 'uvindex_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'EPA UV Index request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'uvindex_results', { } )
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'zip_code' in params:
						st.markdown( f"**Zip Code:** {params.get( 'zip_code', '' )}" )
					if 'city' in params:
						st.markdown( f"**City/State:** {params.get( 'city', '' )}, "
						             f"{params.get( 'state', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_uv = summary.get( 'max_uv', None )
						st.metric( 'Peak UV', '' if max_uv is None else str( max_uv ) )
					
					with sum_c3:
						first_alert = str( summary.get( 'first_alert', '' ) or '' )
						if first_alert:
							st.markdown( f"**Alert:** {first_alert}" )
						else:
							st.markdown( '**Alert:** N/A' )
					
					first_location = str( summary.get( 'first_location', '' ) or '' )
					if first_location:
						st.markdown( f"**Location:** {first_location}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### UV Index Results' )
					df_uvindex = pd.DataFrame( rows )
					
					if not df_uvindex.empty:
						st.dataframe( df_uvindex, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'City', '' ) or item.get( 'Zip', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable UV Index rows were found.' )
				else:
					st.info( 'No UV Index records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='EPA UV Index', result=result )
		
		# -------- Purple Air
		if active_source == 'Purple Air':
			if purpleair_submit:
				try:
					selected_fields = ','.join( purpleair_fields or default_fields )
					
					if purpleair_mode == 'sensor' and not str( purpleair_sensor_index ).strip( ):
						raise ValueError( 'Sensor Index is required for sensor mode.' )
					
					if purpleair_mode == 'sensors':
						if not str( purpleair_nwlng ).strip( ):
							raise ValueError( 'NW Longitude is required for sensors mode.' )
						
						if not str( purpleair_nwlat ).strip( ):
							raise ValueError( 'NW Latitude is required for sensors mode.' )
						
						if not str( purpleair_selng ).strip( ):
							raise ValueError( 'SE Longitude is required for sensors mode.' )
						
						if not str( purpleair_selat ).strip( ):
							raise ValueError( 'SE Latitude is required for sensors mode.' )
					
					f = PurpleAir( )
					result = f.fetch( mode=str( purpleair_mode ), sensor_index=(
							None if not str( purpleair_sensor_index ).strip( ) else int( purpleair_sensor_index )), nwlng=_coerce_optional_float( purpleair_nwlng ), nwlat=_coerce_optional_float( purpleair_nwlat ), selng=_coerce_optional_float( purpleair_selng ), selat=_coerce_optional_float( purpleair_selat ), location_type=int( PURPLEAIR_LOCATION_TYPES.get( purpleair_location_type_label, 0 ) ), max_age=int( purpleair_max_age ), modified_since=int( purpleair_modified_since ), fields=selected_fields, time=int( purpleair_timeout ) )
					
					st.session_state[ 'purpleair_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'PurpleAir request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'purpleair_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'sensor_index' in params:
						st.markdown( f"**Sensor Index:** {params.get( 'sensor_index', '' )}" )
					if 'fields' in params:
						st.markdown( f"**Fields:** {params.get( 'fields', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_pm25 = summary.get( 'max_pm25', None )
						st.metric( 'Peak PM2.5', '' if max_pm25 is None else str( max_pm25 ) )
					
					with sum_c3:
						first_name = str( summary.get( 'first_name', '' ) or '' )
						if first_name:
							st.markdown( f"**First Sensor:** {first_name}" )
						else:
							st.markdown( '**First Sensor:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### PurpleAir Results' )
					df_purpleair = pd.DataFrame( rows )
					
					if not df_purpleair.empty:
						st.dataframe( df_purpleair, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = item.get( 'Latitude', None )
							longitude = item.get( 'Longitude', None )
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': latitude, 'lon': longitude } )
						
						if map_rows:
							st.markdown( '#### Sensor Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Name', '' ) or item.get( 'Sensor Index', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable PurpleAir rows were found.' )
				else:
					st.info( 'No PurpleAir records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='Purple Air', result=result )
		
		# -------- Open Air Quality
		if active_source == 'Open Air Quality':
			if openaq_submit:
				try:
					if openaq_mode == 'latest' and not str( openaq_location_id ).strip( ):
						raise ValueError( 'Location ID is required for latest mode.' )
					
					if openaq_mode == 'parameter_latest' and not str( openaq_parameter_id ).strip( ):
						raise ValueError( 'Parameter ID is required for parameter_latest mode.' )
					
					f = OpenAQ( )
					result = f.fetch( mode=str( openaq_mode ), location_id=_coerce_optional_integer( openaq_location_id ), parameter_id=_coerce_optional_integer( openaq_parameter_id ), country_id=_coerce_optional_integer( openaq_country_id ), coordinates=str( openaq_coordinates ).strip( ), radius=int( openaq_radius ), providers_id=str( openaq_providers_id ).strip( ), parameters_id=str( openaq_parameters_id ).strip( ), limit=int( openaq_limit ), page=int( openaq_page ), time=int( openaq_timeout ) )
					
					st.session_state[ 'openaq_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'OpenAQ request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'openaq_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'location_id' in params:
						st.markdown( f"**Location ID:** {params.get( 'location_id', '' )}" )
					if 'parameter_id' in params:
						st.markdown( f"**Parameter ID:** {params.get( 'parameter_id', '' )}" )
					if 'country_id' in params:
						st.markdown( f"**Country ID:** {params.get( 'country_id', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_name = str( summary.get( 'first_name', '' ) or '' )
						if first_name:
							st.markdown( f"**First Result:** {first_name}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_parameter = str( summary.get( 'first_parameter', '' ) or '' )
						if first_parameter:
							st.markdown( f"**First Parameter:** {first_parameter}" )
						else:
							st.markdown( '**First Parameter:** N/A' )
					
					first_country = str( summary.get( 'first_country', '' ) or '' )
					if first_country:
						st.markdown( f"**Country:** {first_country}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### OpenAQ Results' )
					df_openaq = pd.DataFrame( rows )
					
					if not df_openaq.empty:
						st.dataframe( df_openaq, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = item.get( 'Latitude', None )
							longitude = item.get( 'Longitude', None )
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': latitude, 'lon': longitude } )
						
						if map_rows:
							st.markdown( '#### Location Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Name', '' ) or item.get( 'Location Id', '' ) or item.get( 'Parameter', '' ) or item.get( 'Display '
							                                                                                                                 'Name', '' ) or item.get( 'Id', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable OpenAQ rows were found.' )
				else:
					st.info( 'No OpenAQ records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='Open Air Quality', result=result )
		
		# -------- NASA FIRMS
		if active_source == 'NASA FIRMS':
			if firms_submit:
				try:
					clean_area_coordinates = _validate_firms_area_coordinates( firms_area_coordinates )
					
					clean_date = str( firms_date or '' ).strip( )
					if clean_date:
						dt.datetime.strptime( clean_date, '%Y-%m-%d' )
					
					f = Firms( )
					result = f.fetch( mode=str( firms_mode ), source=str( firms_source ), area_coordinates=clean_area_coordinates, day_range=int( firms_day_range ), date=clean_date, sensor=str( firms_sensor ), time=int( firms_timeout ) )
					
					st.session_state[ 'firms_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NASA FIRMS request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'firms_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'source' in params:
						st.markdown( f"**Source:** {params.get( 'source', '' )}" )
					if 'sensor' in params:
						st.markdown( f"**Sensor:** {params.get( 'sensor', '' )}" )
					if 'area_coordinates' in params:
						st.markdown( f"**Area:** {params.get( 'area_coordinates', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_date = str( summary.get( 'first_date', '' ) or '' )
						if first_date:
							st.markdown( f"**First Date:** {first_date}" )
						else:
							st.markdown( '**First Date:** N/A' )
					
					with sum_c3:
						first_sensor = str( summary.get( 'first_sensor', '' ) or '' )
						if first_sensor:
							st.markdown( f"**Sensor:** {first_sensor}" )
						else:
							st.markdown( '**Sensor:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### FIRMS Results' )
					df_firms = pd.DataFrame( rows )
					
					if not df_firms.empty:
						st.dataframe( df_firms, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = (
										item.get( 'latitude', None ) or item.get( 'Latitude', None ))
							longitude = (
										item.get( 'longitude', None ) or item.get( 'Longitude', None ))
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': float( latitude ),
										'lon': float( longitude ) } )
						
						if map_rows:
							st.markdown( '#### Fire Detection Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'acq_date', '' ) or item.get( 'data_id', '' ) or item.get( 'sensor', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable FIRMS rows were found.' )
				else:
					st.info( 'No FIRMS records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
			
			_promote_environmental_result( source='NASA FIRMS', result=result )
		
		# -------- USGS Water Data
		if active_source == 'USGS Water Data':
			if usgswd_submit:
				try:
					if usgswd_mode in [ 'time-series-metadata', 'latest-continuous',
							'latest-daily' ]:
						if not str( usgswd_monitoring_location_id or '' ).strip( ):
							raise ValueError( 'Monitoring Location ID is required for time-series '
							                  'metadata and latest-value modes.' )
					
					if str( usgswd_parameter_code or '' ).strip( ):
						if not re.fullmatch( r'\d{5}', str( usgswd_parameter_code ).strip( ) ):
							raise ValueError( 'Parameter Code must be a 5-digit USGS parameter code.' )
					
					f = USGSWaterData( )
					result = f.fetch( mode=str( usgswd_mode ), monitoring_location_id=str( usgswd_monitoring_location_id ).strip( ), state_code=str( usgswd_state_code ).strip( ), county_code=str( usgswd_county_code ).strip( ), site_type=str( usgswd_site_type ).strip( ), parameter_code=str( usgswd_parameter_code ).strip( ), limit=int( usgswd_limit ), time=int( usgswd_timeout ) )
					
					st.session_state[ 'usgswaterdata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS Water Data request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgswaterdata_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'monitoring_location_id' in params:
						st.markdown( f"**Location:** "
						             f"{params.get( 'monitoring_location_id', '' )}" )
					if 'parameter_code' in params:
						st.markdown( f"**Parameter:** {params.get( 'parameter_code', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_location = str( summary.get( 'first_location', '' ) or '' )
						if first_location:
							st.markdown( f"**First Location:** {first_location}" )
						else:
							st.markdown( '**First Location:** N/A' )
					
					with sum_c3:
						first_value = str( summary.get( 'first_value', '' ) or '' )
						if first_value:
							st.markdown( f"**Sample Value:** {first_value}" )
						else:
							st.markdown( '**Sample Value:** N/A' )
					
					first_time = str( summary.get( 'first_time', '' ) or '' )
					if first_time:
						st.markdown( f"**First Time:** {first_time}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### USGS Water Data Results' )
					df_usgswaterdata = pd.DataFrame( rows )
					
					if not df_usgswaterdata.empty:
						st.dataframe( df_usgswaterdata, use_container_width=True, hide_index=True )
						
						map_rows = [ ]
						for item in rows:
							latitude = (
										item.get( 'Latitude', None ) or item.get( 'lat', None ) or item.get( 'latitude', None ))
							longitude = (
										item.get( 'Longitude', None ) or item.get( 'lon', None ) or item.get( 'longitude', None ))
							
							if latitude is not None and longitude is not None:
								map_rows.append( { 'lat': float( latitude ),
										'lon': float( longitude ) } )
						
						if map_rows:
							st.markdown( '#### Monitoring Location Map' )
							st.map( map_rows )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Monitoring Location ID', '' ) or item.get( 'Location ID', '' ) or item.get( 'Monitoring Location Name', '' ) or item.get( 'Time', '' ) or f'Record {idx}' )
							
							with st.expander( f'Record {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No displayable USGS Water Data rows were found.' )
				else:
					st.info( 'No USGS Water Data records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

# ==============================================================================
# ASTRONOMICAL MODE
# ==============================================================================
elif mode == 'Astronomical':
	st.session_state.setdefault( 'astronomical_active_source', '' )
	st.session_state.setdefault( 'astronomical_documents', [ ] )
	st.session_state.setdefault( 'astronomical_raw_result', None )
	
	def _build_astronomical_documents( source: str, result: Any ) -> List[ Document ]:
		"""Build canonical documents from an Astronomical-mode result.

		Purpose:
			Converts heterogeneous astronomical, scientific, demographic, and public-data
			responses into LangChain documents for the shared right-side viewer.

		Args:
			source (str): Astronomical source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents representing the result payload.
		"""
		documents: List[ Document ] = [ ]
		
		def _append_document( value: Any, index: int = 1 ) -> None:
			if isinstance( value, Document ):
				metadata = dict( value.metadata or { } )
				metadata.setdefault( 'mode', 'Astronomical' )
				metadata.setdefault( 'source', source )
				documents.append( Document( page_content=value.page_content or '', metadata=metadata ) )
				return
			
			metadata: Dict[ str, Any ] = { 'mode': 'Astronomical', 'source': source,
					'item': index, }
			if isinstance( value, dict ):
				content = json.dumps( normalize( value ), indent=2, ensure_ascii=False, default=str )
			elif isinstance( value, (list, tuple, set) ):
				content = json.dumps( normalize( list( value ) ), indent=2, ensure_ascii=False, default=str )
			else:
				content = str( value )
			
			if content.strip( ):
				documents.append( Document( page_content=content, metadata=metadata ) )
		
		if result is None:
			return documents
		
		if isinstance( result, list ):
			for index, item in enumerate( result, start=1 ):
				_append_document( item, index )
		else:
			_append_document( result )
		
		return documents
	
	def _promote_astronomical_result( source: str, result: Any ) -> List[ Document ]:
		"""Promote a source result into the shared Astronomical document state.

		Purpose:
			Writes the active result to canonical document, raw-text, and source state while
			preserving every source-specific result key and specialized renderer.

		Args:
			source (str): Astronomical source that produced the result.
			result (Any): Source-specific result payload.

		Returns:
			List[Document]: Canonical documents written to shared session state.
		"""
		documents = _build_astronomical_documents( source=source, result=result )
		st.session_state[ 'astronomical_raw_result' ] = result
		st.session_state[ 'astronomical_documents' ] = documents
		st.session_state[ 'documents' ] = documents
		st.session_state[ 'raw_documents' ] = list( documents )
		st.session_state[ 'raw_text' ] = '\n\n'.join(
			document.page_content for document in documents if
			isinstance( document.page_content, str ) and document.page_content.strip( ) )
		st.session_state[ 'processed_text' ] = ''
		st.session_state[ 'active_loader' ] = source
		return documents
	
	st.subheader( '🌌 Physics & Astronomical Data' )
	st.divider( )
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	
	with left:
		# ----------------------------
		# -------- Expander US Naval Observatory
		# ----------------------------
		with st.expander( label='US Naval Observatory', icon='⚓', expanded=False ):
			if 'navalobservatory_results' not in st.session_state:
				st.session_state[ 'navalobservatory_results' ] = { }
			
			if 'navalobservatory_clear_request' not in st.session_state:
				st.session_state[ 'navalobservatory_clear_request' ] = False
			
			if st.session_state.get( 'navalobservatory_clear_request', False ):
				st.session_state[ 'navalobservatory_date' ] = dt.date.today( )
				st.session_state[ 'navalobservatory_time' ] = dt.time( 12, 0 )
				st.session_state[ 'navalobservatory_latitude' ] = 38.9072
				st.session_state[ 'navalobservatory_longitude' ] = -77.0369
				st.session_state[ 'navalobservatory_location_label' ] = ''
				st.session_state[ 'navalobservatory_timeout' ] = 20
				st.session_state[ 'navalobservatory_results' ] = { }
				st.session_state[ 'navalobservatory_clear_request' ] = False
			
			def _clear_navalobservatory_state( ) -> None:
				st.session_state[ 'navalobservatory_clear_request' ] = True
			
			naval_date = st.date_input( 'Date', value=st.session_state.get( 'navalobservatory_date', dt.date.today( ) ), key='navalobservatory_date', help='USNO date parameter in YYYY-MM-DD format.' )
			
			naval_time = st.time_input( 'Time (UTC)', value=st.session_state.get( 'navalobservatory_time', dt.time( 12, 0 ) ), key='navalobservatory_time', help='USNO time parameter in 24-hour format.' )
			
			c1, c2 = st.columns( 2 )
			with c1:
				naval_latitude = st.number_input( 'Latitude', min_value=-90.0, max_value=90.0, value=float( st.session_state.get( 'navalobservatory_latitude', 38.9072 ) ), step=0.0001, format='%.6f', key='navalobservatory_latitude', help='Decimal degrees. North positive.' )
			
			with c2:
				naval_longitude = st.number_input( 'Longitude', min_value=-180.0, max_value=180.0, value=float( st.session_state.get( 'navalobservatory_longitude', -77.0369 ) ), step=0.0001, format='%.6f', key='navalobservatory_longitude', help='Decimal degrees. East positive, west negative.' )
			
			naval_location_label = st.text_input( 'Location Label', value=st.session_state.get( 'navalobservatory_location_label', '' ), key='navalobservatory_location_label', placeholder='Example: Washington, DC' )
			
			naval_timeout = st.number_input( 'Timeout (seconds)', min_value=5, max_value=120, value=int( st.session_state.get( 'navalobservatory_timeout', 20 ) ), step=1, key='navalobservatory_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				naval_submit = st.button( 'Submit', key='navalobservatory_submit', use_container_width=True, width='stretch' )
			with b2:
				naval_clear = st.button( 'Clear', key='navalobservatory_clear', on_click=_clear_navalobservatory_state, width='stretch' )
			
			if naval_submit:
				st.session_state[ 'astronomical_active_source' ] = 'US Naval Observatory'
		
		# ----------------------------
		# -------- Expander Satellite Center
		# ----------------------------
		with st.expander( label='Satellite Center', icon='🛰️', expanded=False ):
			if 'satellitecenter_results' not in st.session_state:
				st.session_state[ 'satellitecenter_results' ] = { }
			
			if 'satellitecenter_clear_request' not in st.session_state:
				st.session_state[ 'satellitecenter_clear_request' ] = False
			
			if st.session_state.get( 'satellitecenter_clear_request', False ):
				st.session_state[ 'satellitecenter_mode' ] = 'observatories'
				st.session_state[ 'satellitecenter_query' ] = ''
				st.session_state[ 'satellitecenter_start_time' ] = ''
				st.session_state[ 'satellitecenter_end_time' ] = ''
				st.session_state[ 'satellitecenter_coordinate_systems' ] = 'gse'
				st.session_state[ 'satellitecenter_resolution_factor' ] = 1
				st.session_state[ 'satellitecenter_timeout' ] = 20
				st.session_state[ 'satellitecenter_results' ] = { }
				st.session_state[ 'satellitecenter_clear_request' ] = False
			
			def _clear_satellitecenter_state( ) -> None:
				st.session_state[ 'satellitecenter_clear_request' ] = True
			
			def _safe_dataframe( rows: Any ) -> pd.DataFrame:
				try:
					if isinstance( rows, pd.DataFrame ):
						return rows
					
					if isinstance( rows, list ):
						if rows and all( isinstance( x, dict ) for x in rows ):
							return pd.DataFrame( rows )
						return pd.DataFrame( { 'Value': [ str( x ) for x in rows ] } )
					
					if isinstance( rows, dict ):
						return pd.json_normalize( rows )
					
					return pd.DataFrame( { 'Value': [ str( rows ) ] } )
				except Exception:
					return pd.DataFrame( )
			
			def _render_satellite_table( title: str, rows: Any ) -> None:
				df_local = _safe_dataframe( rows )
				if not df_local.empty:
					st.markdown( title )
					st.dataframe( df_local, use_container_width=True, hide_index=True )
				else:
					st.info( 'No displayable rows were found.' )
			
			satellite_mode = st.selectbox( 'Mode', options=[ 'observatories', 'ground_stations',
					'locations' ], index=[ 'observatories', 'ground_stations',
					'locations' ].index( st.session_state.get( 'satellitecenter_mode', 'observatories' ) ), key='satellitecenter_mode' )
			
			satellite_query = st.text_area( 'Observatory Query', height=90, key='satellitecenter_query', placeholder=(
				'Examples:\n'
				'iss\n'
				'mms1,mms2\n'
				'themisb\n'
				'\n'
				'Used for locations mode only. Leave '
				'blank for observatories '
				'and ground stations.'), disabled=(satellite_mode != 'locations') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				satellite_start_time = st.text_input( 'Start Time (UTC)', value=st.session_state.get( 'satellitecenter_start_time', '' ), key='satellitecenter_start_time', placeholder='2026-03-15T00:00:00Z', disabled=(
							satellite_mode != 'locations') )
			
			with c2:
				satellite_end_time = st.text_input( 'End Time (UTC)', value=st.session_state.get( 'satellitecenter_end_time', '' ), key='satellitecenter_end_time', placeholder='2026-03-15T02:00:00Z', disabled=(
							satellite_mode != 'locations') )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				satellite_coordinate_systems = st.text_input( 'Coordinate Systems', value=st.session_state.get( 'satellitecenter_coordinate_systems', 'gse' ), key='satellitecenter_coordinate_systems', placeholder='gse or geo,gsm', disabled=(
							satellite_mode != 'locations') )
			
			with c4:
				satellite_resolution_factor = st.number_input( 'Resolution Factor', min_value=1, max_value=1000, value=int( st.session_state.get( 'satellitecenter_resolution_factor', 1 ) ), step=1, key='satellitecenter_resolution_factor', disabled=(
							satellite_mode != 'locations') )
			
			with c5:
				satellite_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'satellitecenter_timeout', 20 ) ), step=1, key='satellitecenter_timeout' )
			
			st.caption( 'No API key is required for SSCWeb. For locations mode, use UTC ISO '
			            '8601 timestamps and observatory IDs returned by the observatories '
			            'service.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				satellite_submit = st.button( 'Submit', key='satellitecenter_submit', use_container_width=True, width='stretch' )
			
			with b2:
				st.button( 'Clear', key='satellitecenter_clear', on_click=_clear_satellitecenter_state, width='stretch' )
			
			if satellite_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Satellite Center'
		
		# ----------------------------
		# -------- Expander Astro Catalog
		# ----------------------------
		with st.expander( label='Astro Catalog', icon='✨', expanded=False ):
			if 'astrocatalog_results' not in st.session_state:
				st.session_state[ 'astrocatalog_results' ] = { }
			
			if 'astrocatalog_clear_request' not in st.session_state:
				st.session_state[ 'astrocatalog_clear_request' ] = False
			
			if st.session_state.get( 'astrocatalog_clear_request', False ):
				st.session_state[ 'astrocatalog_mode' ] = 'object_query'
				st.session_state[ 'astrocatalog_query' ] = ''
				st.session_state[ 'astrocatalog_quantity' ] = ''
				st.session_state[ 'astrocatalog_attributes' ] = ''
				st.session_state[ 'astrocatalog_arguments' ] = ''
				st.session_state[ 'astrocatalog_ra' ] = ''
				st.session_state[ 'astrocatalog_dec' ] = ''
				st.session_state[ 'astrocatalog_radius' ] = 2
				st.session_state[ 'astrocatalog_format' ] = 'json'
				st.session_state[ 'astrocatalog_timeout' ] = 20
				st.session_state[ 'astrocatalog_results' ] = { }
				st.session_state[ 'astrocatalog_clear_request' ] = False
			
			def _clear_astrocatalog_state( ) -> None:
				st.session_state[ 'astrocatalog_clear_request' ] = True
			
			astro_mode = st.selectbox( 'Mode', options=[ 'object_query', 'cone_search' ], index=[
					'object_query',
					'cone_search' ].index( st.session_state.get( 'astrocatalog_mode', 'object_query' ) ), key='astrocatalog_mode' )
			
			astro_query = st.text_area( 'Object Query', height=80, key='astrocatalog_query', placeholder=(
				'Examples:\n'
				'SN1987A\n'
				'AT2024abc\n'
				'GW170817\n'
				'\n'
				'Used for object_query mode.'), disabled=(astro_mode != 'object_query') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				astro_quantity = st.text_input( 'Quantity', value=st.session_state.get( 'astrocatalog_quantity', '' ), key='astrocatalog_quantity', placeholder='Example: photometry', disabled=(
							astro_mode != 'object_query') )
			
			with c2:
				astro_attributes = st.text_input( 'Attributes', value=st.session_state.get( 'astrocatalog_attributes', '' ), key='astrocatalog_attributes', placeholder='Example: time,magnitude,band', disabled=(
							astro_mode != 'object_query') )
			
			astro_arguments = st.text_area( 'Arguments', height=80, key='astrocatalog_arguments', placeholder=(
				'Optional query arguments.\n'
				'Examples:\n'
				'time=2450000\n'
				'band=V'), disabled=(astro_mode != 'object_query') )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				astro_ra = st.text_input( 'RA', value=st.session_state.get( 'astrocatalog_ra', '' ), key='astrocatalog_ra', placeholder='13:09:48.09', disabled=(
							astro_mode != 'cone_search') )
			
			with c4:
				astro_dec = st.text_input( 'Dec', value=st.session_state.get( 'astrocatalog_dec', '' ), key='astrocatalog_dec', placeholder='+27:57:34.8', disabled=(
							astro_mode != 'cone_search') )
			
			with c5:
				astro_radius = st.number_input( 'Radius (arcsec)', min_value=1, max_value=3600, value=int( st.session_state.get( 'astrocatalog_radius', 2 ) ), step=1, key='astrocatalog_radius', disabled=(
							astro_mode != 'cone_search') )
			
			c6, c7 = st.columns( 2 )
			with c6:
				astro_format = st.selectbox( 'Format', options=[ 'json', 'csv', 'tsv' ], index=[
						'json', 'csv',
						'tsv' ].index( st.session_state.get( 'astrocatalog_format', 'json' ) ), key='astrocatalog_format' )
			
			with c7:
				astro_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'astrocatalog_timeout', 20 ) ), step=1, key='astrocatalog_timeout' )
			
			st.caption( 'No API key is required for Open Astronomy Catalog. '
			            'Use object_query for named events and cone_search for coordinate '
			            'searches.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				astro_submit = st.button( 'Submit', key='astrocatalog_submit', width='stretch' )
			with b2:
				st.button( 'Clear', key='astrocatalog_clear', on_click=_clear_astrocatalog_state, width='stretch' )
			
			if astro_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Astro Catalog'
		
		# ----------------------------
		# -------- Expander Astro Query
		# ----------------------------
		with st.expander( label='Astro Query', icon='🔭', expanded=False ):
			if 'astroquery_results' not in st.session_state:
				st.session_state[ 'astroquery_results' ] = { }
			
			if 'astroquery_clear_request' not in st.session_state:
				st.session_state[ 'astroquery_clear_request' ] = False
			
			if st.session_state.get( 'astroquery_clear_request', False ):
				st.session_state[ 'astroquery_mode' ] = 'object_search'
				st.session_state[ 'astroquery_query' ] = ''
				st.session_state[ 'astroquery_ra' ] = ''
				st.session_state[ 'astroquery_dec' ] = ''
				st.session_state[ 'astroquery_radius' ] = 0.5
				st.session_state[ 'astroquery_radius_unit' ] = 'deg'
				st.session_state[ 'astroquery_row_limit' ] = 100
				st.session_state[ 'astroquery_results' ] = { }
				st.session_state[ 'astroquery_clear_request' ] = False
			
			def _clear_astroquery_state( ) -> None:
				st.session_state[ 'astroquery_clear_request' ] = True
			
			astroquery_mode = st.selectbox( 'Mode', options=[ 'object_search', 'object_ids',
					'region_search' ], index=[ 'object_search', 'object_ids',
					'region_search' ].index( st.session_state.get( 'astroquery_mode', 'object_search' ) ), key='astroquery_mode' )
			
			astroquery_query = st.text_area( 'Object Query', height=80, key='astroquery_query', placeholder=(
				'Examples:\n'
				'M81\n'
				'Sirius\n'
				'NGC 1300\n'
				'\n'
				'Used for object_search and object_ids.'), disabled=(
						astroquery_mode == 'region_search') )
			
			c1, c2, c3 = st.columns( 3 )
			with c1:
				astroquery_ra = st.text_input( 'RA', value=st.session_state.get( 'astroquery_ra', '' ), key='astroquery_ra', placeholder='13:09:48.09', disabled=(
							astroquery_mode != 'region_search'), help='Right Ascension of the search center, e.g. 13:09:48.09.' )
			
			with c2:
				astroquery_dec = st.text_input( 'Dec', value=st.session_state.get( 'astroquery_dec', '' ), key='astroquery_dec', placeholder='-23:22:53.3', disabled=(
							astroquery_mode != 'region_search'), help='Declination of the search center, e.g. -23:22:53.3.' )
			
			with c3:
				astroquery_radius = st.number_input( 'Radius', min_value=0.001, max_value=60.0, value=float( st.session_state.get( 'astroquery_radius', 0.5 ) ), step=0.1, key='astroquery_radius', disabled=(
							astroquery_mode != 'region_search'), help='Cone-search radius around the RA/Dec sky position.' )
			
			c4, c5 = st.columns( 2 )
			with c4:
				astroquery_radius_unit = st.selectbox( 'Radius Unit', options=[ 'deg', 'arcmin',
						'arcsec' ], index=[ 'deg', 'arcmin',
						'arcsec' ].index( st.session_state.get( 'astroquery_radius_unit', 'deg' ) ), key='astroquery_radius_unit', disabled=(
							astroquery_mode != 'region_search') )
			
			with c5:
				astroquery_row_limit = st.number_input( 'Row Limit', min_value=1, max_value=10000, value=int( st.session_state.get( 'astroquery_row_limit', 100 ) ), step=1, key='astroquery_row_limit' )
			
			st.caption( 'No API key is required for basic astroquery SIMBAD queries. '
			            'Use object_search for a named object, object_ids for alternate '
			            'names, '
			            'and region_search for a cone search around RA/Dec.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				astroquery_submit = st.button( 'Submit', key='astroquery_submit', width='stretch' )
			with b2:
				st.button( 'Clear', key='astroquery_clear', on_click=_clear_astroquery_state, width='stretch' )
			
			if astroquery_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Astro Query'
		
		# ----------------------------
		# -------- Expander Star Map
		# ----------------------------
		with st.expander( label='Star Map', icon='⭐', expanded=False ):
			if 'starmap_results' not in st.session_state:
				st.session_state[ 'starmap_results' ] = { }
			
			if 'starmap_clear_request' not in st.session_state:
				st.session_state[ 'starmap_clear_request' ] = False
			
			if st.session_state.get( 'starmap_clear_request', False ):
				st.session_state[ 'starmap_mode' ] = 'object_link'
				st.session_state[ 'starmap_query' ] = ''
				st.session_state[ 'starmap_ra' ] = 15.2976
				st.session_state[ 'starmap_dec' ] = -17.5892
				st.session_state[ 'starmap_zoom' ] = 5
				st.session_state[ 'starmap_image_source' ] = 'DSS2'
				st.session_state[ 'starmap_box_color' ] = 'yellow'
				st.session_state[ 'starmap_show_box' ] = True
				st.session_state[ 'starmap_show_grid' ] = True
				st.session_state[ 'starmap_show_lines' ] = True
				st.session_state[ 'starmap_show_boundaries' ] = True
				st.session_state[ 'starmap_show_const_names' ] = False
				st.session_state[ 'starmap_timeout' ] = 20
				st.session_state[ 'starmap_results' ] = { }
				st.session_state[ 'starmap_clear_request' ] = False
			
			def _clear_starmap_state( ) -> None:
				st.session_state[ 'starmap_clear_request' ] = True
			
			starmap_mode = st.selectbox( 'Mode', options=[ 'object_link', 'coordinate_link',
					'snapshot' ], index=[ 'object_link', 'coordinate_link',
					'snapshot' ].index( st.session_state.get( 'starmap_mode', 'object_link' ) ), key='starmap_mode' )
			
			starmap_query = st.text_area( 'Object Query', height=80, key='starmap_query', placeholder='Examples: Polaris, M31, NGC 1300, Used for object_link mode '
			                                                                                          'only.', disabled=(
						starmap_mode != 'object_link') )
			
			c1, c2, c3 = st.columns( 3 )
			with c1:
				starmap_ra = st.number_input( 'RA (hours)', min_value=0.0, max_value=24.0, value=float( st.session_state.get( 'starmap_ra', 15.2976 ) ), step=0.0001, format='%.4f', key='starmap_ra', disabled=(
							starmap_mode == 'object_link'), help='Right Ascension of the sky center in hours. Example: 15.2976' )
			
			with c2:
				starmap_dec = st.number_input( 'Dec (degrees)', min_value=-90.0, max_value=90.0, value=float( st.session_state.get( 'starmap_dec', -17.5892 ) ), step=0.0001, format='%.4f', key='starmap_dec', disabled=(
							starmap_mode == 'object_link'), help='Declination of the sky center in degrees. Example: -17.5892' )
			
			with c3:
				starmap_zoom = st.number_input( 'Zoom', min_value=1, max_value=18, value=int( st.session_state.get( 'starmap_zoom', 5 ) ), step=1, key='starmap_zoom', help='Smaller values show a wider field; larger values zoom in.' )
			
			c4, c5 = st.columns( 2 )
			with c4:
				starmap_image_source = st.selectbox( 'Image Source', options=[ 'DSS2', 'SDSS',
						'SDSS-III', 'GALEX', 'IRAS', 'RASS', 'H-Alpha' ], index=[ 'DSS2', 'SDSS',
						'SDSS-III', 'GALEX', 'IRAS', 'RASS',
						'H-Alpha' ].index( st.session_state.get( 'starmap_image_source', 'DSS2' ) ), key='starmap_image_source', disabled=(
							starmap_mode != 'snapshot'), help='Sky survey source used for snapshot generation.' )
			
			with c5:
				starmap_box_color = st.text_input( 'Box Color', value=st.session_state.get( 'starmap_box_color', 'yellow' ), key='starmap_box_color', placeholder='yellow', disabled=(
							starmap_mode == 'snapshot') )
			
			c6, c7 = st.columns( 2 )
			with c6:
				starmap_show_box = st.checkbox( 'Show Box', value=st.session_state.get( 'starmap_show_box', True ), key='starmap_show_box', disabled=(
							starmap_mode == 'snapshot') )
			
			with c7:
				starmap_show_grid = st.checkbox( 'Show Grid', value=st.session_state.get( 'starmap_show_grid', True ), key='starmap_show_grid', disabled=(
							starmap_mode == 'object_link') )
			
			c8, c9 = st.columns( 2 )
			with c8:
				starmap_show_lines = st.checkbox( 'Show Constellation Lines', value=st.session_state.get( 'starmap_show_lines', True ), key='starmap_show_lines', disabled=False )
			
			with c9:
				starmap_show_boundaries = st.checkbox( 'Show Constellation Boundaries', value=st.session_state.get( 'starmap_show_boundaries', True ), key='starmap_show_boundaries', disabled=False )
			
			starmap_show_const_names = st.checkbox( 'Show Constellation Names', value=st.session_state.get( 'starmap_show_const_names', False ), key='starmap_show_const_names', disabled=(
						starmap_mode != 'snapshot') )
			
			starmap_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'starmap_timeout', 20 ) ), step=1, key='starmap_timeout' )
			
			st.caption( 'No API key is required. '
			            'Use object_link for a named object, coordinate_link for '
			            'RA/Dec-centered interactive maps, '
			            'and snapshot for a static sky image page.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				starmap_submit = st.button( 'Submit', key='starmap_submit', width='stretch' )
			with b2:
				st.button( 'Clear', key='starmap_clear', on_click=_clear_starmap_state, width='stretch' )
			
			if starmap_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Star Map'
		
		# ----------------------------
		# -------- Expander Simbad
		# ----------------------------
		with st.expander( label='SIMBAD', icon='🌌', expanded=False ):
			if 'simbad_results' not in st.session_state:
				st.session_state[ 'simbad_results' ] = { }
			
			if 'simbad_clear_request' not in st.session_state:
				st.session_state[ 'simbad_clear_request' ] = False
			
			if st.session_state.get( 'simbad_clear_request', False ):
				st.session_state[ 'simbad_mode' ] = 'object_search'
				st.session_state[ 'simbad_query' ] = 'Polaris'
				st.session_state[ 'simbad_ra' ] = '02:31:49.09'
				st.session_state[ 'simbad_dec' ] = '+89:15:50.8'
				st.session_state[ 'simbad_radius' ] = 0.5
				st.session_state[ 'simbad_radius_unit' ] = 'deg'
				st.session_state[ 'simbad_row_limit' ] = 100
				st.session_state[ 'simbad_results' ] = { }
				st.session_state[ 'simbad_clear_request' ] = False
			
			def _clear_simbad_state( ) -> None:
				st.session_state[ 'simbad_clear_request' ] = True
			
			simbad_mode = st.selectbox( 'Mode', options=[ 'object_search', 'object_ids',
					'region_search' ], index=[ 'object_search', 'object_ids',
					'region_search' ].index( st.session_state.get( 'simbad_mode', 'object_search' ) ), key='simbad_mode', help='Choose named-object lookup, alternate identifiers, or cone search.' )
			
			simbad_query = st.text_area( 'Object Name', height=80, key='simbad_query', placeholder=(
				'Examples:\n'
				'Polaris\n'
				'M 31\n'
				'NGC 1300\n'
				'\n'
				'Used for object_search and object_ids.'), disabled=(
						simbad_mode == 'region_search') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				simbad_ra = st.text_input( 'Right Ascension', value=st.session_state.get( 'simbad_ra', '02:31:49.09' ), key='simbad_ra', placeholder='13:09:48.09', disabled=(
							simbad_mode != 'region_search'), help='Hourangle format recommended for this wrapper.' )
			
			with c2:
				simbad_dec = st.text_input( 'Declination', value=st.session_state.get( 'simbad_dec', '+89:15:50.8' ), key='simbad_dec', placeholder='-23:22:53.3', disabled=(
							simbad_mode != 'region_search') )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				simbad_radius = st.number_input( 'Radius', min_value=0.001, max_value=60.0, value=float( st.session_state.get( 'simbad_radius', 0.5 ) ), step=0.1, key='simbad_radius', disabled=(
							simbad_mode != 'region_search'), help='Cone-search radius around the RA/Dec position.' )
			
			with c4:
				simbad_radius_unit = st.selectbox( 'Radius Unit', options=[ 'deg', 'arcmin',
						'arcsec' ], index=[ 'deg', 'arcmin',
						'arcsec' ].index( st.session_state.get( 'simbad_radius_unit', 'deg' ) ), key='simbad_radius_unit', disabled=(
							simbad_mode != 'region_search') )
			
			with c5:
				simbad_row_limit = st.number_input( 'Row Limit', min_value=1, max_value=10000, value=int( st.session_state.get( 'simbad_row_limit', 100 ) ), step=1, key='simbad_row_limit' )
			
			st.caption( 'SIMBAD named-object queries do not require an API key. '
			            'Use object_search for a record lookup, object_ids for aliases, '
			            'and region_search for a cone search around sky coordinates.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				simbad_submit = st.button( 'Submit', key='simbad_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='simbad_clear', on_click=_clear_simbad_state, width='stretch' )
			
			if simbad_submit:
				st.session_state[ 'astronomical_active_source' ] = 'SIMBAD'
		
		# ----------------------------
		# -------- Expander Space Weather
		# ----------------------------
		with st.expander( label='Space Weather', icon='☄️', expanded=False ):
			if 'spaceweather_results' not in st.session_state:
				st.session_state[ 'spaceweather_results' ] = { }
			
			if 'spaceweather_clear_request' not in st.session_state:
				st.session_state[ 'spaceweather_clear_request' ] = False
			
			if st.session_state.get( 'spaceweather_clear_request', False ):
				st.session_state[ 'spaceweather_mode' ] = 'cme'
				st.session_state[ 'spaceweather_start_date' ] = '2026-03-01'
				st.session_state[ 'spaceweather_end_date' ] = '2026-03-15'
				st.session_state[ 'spaceweather_location' ] = 'ALL'
				st.session_state[ 'spaceweather_catalog' ] = 'ALL'
				st.session_state[ 'spaceweather_notification_type' ] = 'all'
				st.session_state[ 'spaceweather_most_accurate_only' ] = True
				st.session_state[ 'spaceweather_complete_entry_only' ] = True
				st.session_state[ 'spaceweather_speed' ] = 0
				st.session_state[ 'spaceweather_half_angle' ] = 0
				st.session_state[ 'spaceweather_keyword' ] = ''
				st.session_state[ 'spaceweather_timeout' ] = 20
				st.session_state[ 'spaceweather_results' ] = { }
				st.session_state[ 'spaceweather_clear_request' ] = False
			
			def _clear_spaceweather_state( ) -> None:
				st.session_state[ 'spaceweather_clear_request' ] = True
			
			spaceweather_mode = st.selectbox( 'Mode', options=[ 'cme', 'cme_analysis', 'gst', 'ips',
					'flr', 'sep', 'mpc', 'rbe', 'hss', 'wsa_enlil', 'notifications' ], index=[
					'cme', 'cme_analysis', 'gst', 'ips', 'flr', 'sep', 'mpc', 'rbe', 'hss',
					'wsa_enlil',
					'notifications' ].index( st.session_state.get( 'spaceweather_mode', 'cme' ) ), key='spaceweather_mode', help='Choose the documented DONKI endpoint to '
			                                                                                                                     'query.' )
			
			d1, d2 = st.columns( 2 )
			with d1:
				spaceweather_start_date = st.text_input( 'Start Date', value=st.session_state.get( 'spaceweather_start_date', '2026-03-01' ), key='spaceweather_start_date', placeholder='2026-03-01' )
			
			with d2:
				spaceweather_end_date = st.text_input( 'End Date', value=st.session_state.get( 'spaceweather_end_date', '2026-03-15' ), key='spaceweather_end_date', placeholder='2026-03-15' )
			
			c1, c2 = st.columns( 2 )
			with c1:
				spaceweather_location = st.text_input( 'Location', value=st.session_state.get( 'spaceweather_location', 'ALL' ), key='spaceweather_location', placeholder='ALL or Earth', disabled=(
							spaceweather_mode != 'ips') )
			
			with c2:
				spaceweather_catalog = st.text_input( 'Catalog', value=st.session_state.get( 'spaceweather_catalog', 'ALL' ), key='spaceweather_catalog', placeholder='ALL or SWRC_CATALOG', disabled=(
							spaceweather_mode not in [ 'ips', 'cme_analysis' ]) )
			
			c3, c4 = st.columns( 2 )
			with c3:
				spaceweather_notification_type = st.text_input( 'Notification Type', value=st.session_state.get( 'spaceweather_notification_type', 'all' ), key='spaceweather_notification_type', placeholder='all or FLR', disabled=(
							spaceweather_mode != 'notifications') )
			
			with c4:
				spaceweather_keyword = st.text_input( 'Keyword', value=st.session_state.get( 'spaceweather_keyword', '' ), key='spaceweather_keyword', placeholder='swpc_annex', disabled=(
							spaceweather_mode != 'cme_analysis') )
			
			c5, c6 = st.columns( 2 )
			with c5:
				spaceweather_speed = st.number_input( 'Speed', min_value=0, max_value=5000, value=int( st.session_state.get( 'spaceweather_speed', 0 ) ), step=10, key='spaceweather_speed', disabled=(
							spaceweather_mode != 'cme_analysis') )
			
			with c6:
				spaceweather_half_angle = st.number_input( 'Half Angle', min_value=0, max_value=180, value=int( st.session_state.get( 'spaceweather_half_angle', 0 ) ), step=1, key='spaceweather_half_angle', disabled=(
							spaceweather_mode != 'cme_analysis') )
			
			c7, c8, c9 = st.columns( 3 )
			with c7:
				spaceweather_most_accurate_only = st.checkbox( 'Most Accurate Only', value=bool( st.session_state.get( 'spaceweather_most_accurate_only', True ) ), key='spaceweather_most_accurate_only', disabled=(
							spaceweather_mode != 'cme_analysis') )
			
			with c8:
				spaceweather_complete_entry_only = st.checkbox( 'Complete Entry Only', value=bool( st.session_state.get( 'spaceweather_complete_entry_only', True ) ), key='spaceweather_complete_entry_only', disabled=(
							spaceweather_mode != 'cme_analysis') )
			
			with c9:
				spaceweather_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'spaceweather_timeout', 20 ) ), step=1, key='spaceweather_timeout' )
			
			st.caption( 'Examples: cme for coronal mass ejections, gst for geomagnetic '
			            'storms, '
			            'ips with Location=Earth, notifications with Type=all, or '
			            'cme_analysis with Catalog=ALL and Speed=500.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				spaceweather_submit = st.button( 'Submit', key='spaceweather_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='spaceweather_clear', on_click=_clear_spaceweather_state, width='stretch' )
			
			if spaceweather_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Space Weather'
		
		# ----------------------------
		# -------- Expander Astro Catalog
		# ----------------------------
		with st.expander( label='Star Chart', icon='🪐', expanded=False ):
			if 'starchart_results' not in st.session_state:
				st.session_state[ 'starchart_results' ] = { }
			
			if 'starchart_clear_request' not in st.session_state:
				st.session_state[ 'starchart_clear_request' ] = False
			
			if st.session_state.get( 'starchart_clear_request', False ):
				st.session_state[ 'starchart_mode' ] = 'object_chart'
				st.session_state[ 'starchart_query' ] = 'Polaris'
				st.session_state[ 'starchart_ra' ] = 2.5302
				st.session_state[ 'starchart_dec' ] = 89.2642
				st.session_state[ 'starchart_zoom' ] = 5
				st.session_state[ 'starchart_image_source' ] = 'DSS2'
				st.session_state[ 'starchart_box_color' ] = 'yellow'
				st.session_state[ 'starchart_show_box' ] = True
				st.session_state[ 'starchart_show_grid' ] = True
				st.session_state[ 'starchart_show_lines' ] = True
				st.session_state[ 'starchart_show_boundaries' ] = True
				st.session_state[ 'starchart_show_const_names' ] = False
				st.session_state[ 'starchart_width' ] = 900
				st.session_state[ 'starchart_height' ] = 450
				st.session_state[ 'starchart_magnitude' ] = 7.5
				st.session_state[ 'starchart_timeout' ] = 20
				st.session_state[ 'starchart_results' ] = { }
				st.session_state[ 'starchart_clear_request' ] = False
			
			def _clear_starchart_state( ) -> None:
				'''
					Purpose:
					--------
					Flag the Star Chart expander state for reset on the next rerun.

					Parameters:
					-----------
					None

					Returns:
					--------
					None
				'''
				st.session_state[ 'starchart_clear_request' ] = True
			
			starchart_mode = st.selectbox( 'Mode', options=[ 'object_search', 'object_chart',
					'coordinate_chart', 'static_chart' ], index=[ 'object_search', 'object_chart',
					'coordinate_chart',
					'static_chart' ].index( st.session_state.get( 'starchart_mode', 'object_chart' ) ), key='starchart_mode' )
			
			starchart_query = st.text_area( 'Object Query', height=80, key='starchart_query', placeholder=(
				'Examples:\n'
				'Polaris\n'
				'M31\n'
				'NGC 1300\n'
				'\n'
				'Used for object_search and object_chart.'), disabled=(
						starchart_mode not in [ 'object_search', 'object_chart' ]) )
			
			c1, c2 = st.columns( 2 )
			with c1:
				starchart_ra = st.number_input( 'RA', value=float( st.session_state.get( 'starchart_ra', 2.5302 ) ), format='%.4f', key='starchart_ra', disabled=(
							starchart_mode not in [ 'coordinate_chart', 'static_chart' ]) )
			
			with c2:
				starchart_dec = st.number_input( 'Dec', value=float( st.session_state.get( 'starchart_dec', 89.2642 ) ), format='%.4f', key='starchart_dec', disabled=(
							starchart_mode not in [ 'coordinate_chart', 'static_chart' ]) )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				starchart_zoom = st.number_input( 'Zoom', min_value=0, max_value=18, value=int( st.session_state.get( 'starchart_zoom', 5 ) ), step=1, key='starchart_zoom' )
			
			with c4:
				starchart_width = st.number_input( 'Width', min_value=100, max_value=2400, value=int( st.session_state.get( 'starchart_width', 900 ) ), step=50, key='starchart_width', disabled=(
							starchart_mode != 'static_chart') )
			
			with c5:
				starchart_height = st.number_input( 'Height', min_value=100, max_value=2400, value=int( st.session_state.get( 'starchart_height', 450 ) ), step=50, key='starchart_height', disabled=(
							starchart_mode != 'static_chart') )
			
			c6, c7, c8 = st.columns( 3 )
			with c6:
				starchart_image_source = st.selectbox( 'Image Source', options=[ 'DSS2',
						'SDSS' ], index=[ 'DSS2',
						'SDSS' ].index( st.session_state.get( 'starchart_image_source', 'DSS2' ) ), key='starchart_image_source' )
			
			with c7:
				starchart_box_color = st.text_input( 'Box Color', value=st.session_state.get( 'starchart_box_color', 'yellow' ), key='starchart_box_color', disabled=(
							starchart_mode not in [ 'object_chart', 'coordinate_chart' ]) )
			
			with c8:
				starchart_magnitude = st.number_input( 'Magnitude', min_value=0.0, max_value=20.0, value=float( st.session_state.get( 'starchart_magnitude', 7.5 ) ), step=0.1, key='starchart_magnitude', disabled=(
							starchart_mode != 'static_chart') )
			
			c9, c10 = st.columns( 2 )
			
			with c9:
				starchart_show_box = st.checkbox( 'Show Box', value=bool( st.session_state.get( 'starchart_show_box', True ) ), key='starchart_show_box', disabled=(
							starchart_mode not in [ 'object_chart', 'coordinate_chart' ]) )
			
			with c10:
				starchart_show_grid = st.checkbox( 'Show Grid', value=bool( st.session_state.get( 'starchart_show_grid', True ) ), key='starchart_show_grid', disabled=(
							starchart_mode not in [ 'coordinate_chart', 'static_chart' ]) )
			
			c11, c12 = st.columns( 2 )
			with c11:
				starchart_show_lines = st.checkbox( 'Show Lines', value=bool( st.session_state.get( 'starchart_show_lines', True ) ), key='starchart_show_lines', disabled=(
							starchart_mode not in [ 'coordinate_chart', 'static_chart' ]) )
			
			with c12:
				starchart_show_boundaries = st.checkbox( 'Show Boundaries', value=bool( st.session_state.get( 'starchart_show_boundaries', True ) ), key='starchart_show_boundaries', disabled=(
							starchart_mode not in [ 'coordinate_chart', 'static_chart' ]) )
			
			starchart_show_const_names = st.checkbox( 'Show Constellation Names', value=bool( st.session_state.get( 'starchart_show_const_names', False ) ), key='starchart_show_const_names', disabled=(
						starchart_mode != 'static_chart') )
			
			starchart_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'starchart_timeout', 20 ) ), step=1, key='starchart_timeout' )
			
			st.caption( 'Examples: object_search with Polaris, object_chart with M31, '
			            'coordinate_chart with RA=2.5302 and Dec=89.2642, or '
			            'static_chart with DSS2 and 900x450 output.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				starchart_submit = st.button( 'Submit', key='starchart_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='starchart_clear', on_click=_clear_starchart_state, width='stretch' )
			
			if starchart_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Star Chart'
		
		# ----------------------------
		# -------- Expander Near Earth Object
		# ----------------------------
		with st.expander( label='Near Earth Objects', icon='☄️', expanded=False ):
			if 'nearbyobjects_results' not in st.session_state:
				st.session_state[ 'nearbyobjects_results' ] = { }
			
			if 'nearbyobjects_clear_request' not in st.session_state:
				st.session_state[ 'nearbyobjects_clear_request' ] = False
			
			if st.session_state.get( 'nearbyobjects_clear_request', False ):
				st.session_state[ 'nearbyobjects_mode' ] = 'close_approaches'
				st.session_state[ 'nearbyobjects_start_date' ] = '2026-03-01'
				st.session_state[ 'nearbyobjects_end_date' ] = '2026-03-31'
				st.session_state[ 'nearbyobjects_query' ] = 'Apophis'
				st.session_state[ 'nearbyobjects_query_type' ] = 'sstr'
				st.session_state[ 'nearbyobjects_dist_max' ] = '10LD'
				st.session_state[ 'nearbyobjects_body' ] = 'Earth'
				st.session_state[ 'nearbyobjects_sort' ] = 'date'
				st.session_state[ 'nearbyobjects_limit' ] = 20
				st.session_state[ 'nearbyobjects_dv' ] = 6.0
				st.session_state[ 'nearbyobjects_dur' ] = 360
				st.session_state[ 'nearbyobjects_stay' ] = 8
				st.session_state[ 'nearbyobjects_launch' ] = '2020-2045'
				st.session_state[ 'nearbyobjects_h' ] = 26.0
				st.session_state[ 'nearbyobjects_occ' ] = 7
				st.session_state[ 'nearbyobjects_include_physical' ] = True
				st.session_state[ 'nearbyobjects_include_close_approaches' ] = True
				st.session_state[ 'nearbyobjects_ca_body' ] = 'Earth'
				st.session_state[ 'nearbyobjects_include_discovery' ] = True
				st.session_state[ 'nearbyobjects_timeout' ] = 20
				st.session_state[ 'nearbyobjects_results' ] = { }
				st.session_state[ 'nearbyobjects_clear_request' ] = False
			
			def _clear_nearbyobjects_state( ) -> None:
				'''
					Purpose:
					--------
					Flag the Near Earth Objects expander state for reset on the next rerun.

					Parameters:
					-----------
					None

					Returns:
					--------
					None
				'''
				st.session_state[ 'nearbyobjects_clear_request' ] = True
			
			nearby_mode = st.selectbox( 'Mode', options=[ 'close_approaches', 'object_lookup',
					'nhats_summary', 'nhats_object', 'fireballs' ], index=[ 'close_approaches',
					'object_lookup', 'nhats_summary', 'nhats_object',
					'fireballs' ].index( st.session_state.get( 'nearbyobjects_mode', 'close_approaches' ) ), key='nearbyobjects_mode', help='Choose close approaches, single-object lookup, NHATS screening, '
			                                                                                                                                'or fireball data.' )
			
			d1, d2 = st.columns( 2 )
			with d1:
				nearby_start_date = st.text_input( 'Start Date', value=st.session_state.get( 'nearbyobjects_start_date', '2026-03-01' ), key='nearbyobjects_start_date', placeholder='2026-03-01', disabled=(
							nearby_mode not in [ 'close_approaches', 'fireballs' ]) )
			
			with d2:
				nearby_end_date = st.text_input( 'End Date', value=st.session_state.get( 'nearbyobjects_end_date', '2026-03-31' ), key='nearbyobjects_end_date', placeholder='2026-03-31', disabled=(
							nearby_mode != 'close_approaches') )
			
			nearby_query = st.text_area( 'Object Query / Designation', height=80, key='nearbyobjects_query', placeholder=(
				'Examples:\n'
				'Apophis\n'
				'Eros\n'
				'99942\n'
				'2000 SG344'), disabled=(nearby_mode not in [ 'object_lookup', 'nhats_object' ]) )
			
			c1, c2 = st.columns( 2 )
			with c1:
				nearby_query_type = st.selectbox( 'Query Type', options=[ 'sstr', 'spk',
						'des' ], index=[ 'sstr', 'spk',
						'des' ].index( st.session_state.get( 'nearbyobjects_query_type', 'sstr' ) ), key='nearbyobjects_query_type', disabled=(
							nearby_mode != 'object_lookup') )
			
			with c2:
				nearby_dist_max = st.text_input( 'Distance Max', value=st.session_state.get( 'nearbyobjects_dist_max', '10LD' ), key='nearbyobjects_dist_max', placeholder='10LD or 0.05AU', disabled=(
							nearby_mode != 'close_approaches') )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				nearby_body = st.text_input( 'Body', value=st.session_state.get( 'nearbyobjects_body', 'Earth' ), key='nearbyobjects_body', placeholder='Earth', disabled=(
							nearby_mode != 'close_approaches') )
			
			with c4:
				nearby_sort = st.text_input( 'Sort', value=st.session_state.get( 'nearbyobjects_sort', 'date' ), key='nearbyobjects_sort', placeholder='date or dist', disabled=(
							nearby_mode != 'close_approaches') )
			
			with c5:
				nearby_limit = st.number_input( 'Limit', min_value=1, max_value=500, value=int( st.session_state.get( 'nearbyobjects_limit', 20 ) ), step=1, key='nearbyobjects_limit' )
			
			st.markdown( '#### NHATS Filters' )
			
			n1, n2, n3 = st.columns( 3 )
			with n1:
				nearby_dv = st.number_input( 'ΔV', min_value=0.0, max_value=20.0, value=float( st.session_state.get( 'nearbyobjects_dv', 6.0 ) ), step=0.1, key='nearbyobjects_dv', disabled=(
							nearby_mode not in [ 'nhats_summary', 'nhats_object' ]) )
			
			with n2:
				nearby_dur = st.number_input( 'Duration', min_value=1, max_value=3000, value=int( st.session_state.get( 'nearbyobjects_dur', 360 ) ), step=1, key='nearbyobjects_dur', disabled=(
							nearby_mode not in [ 'nhats_summary', 'nhats_object' ]) )
			
			with n3:
				nearby_stay = st.number_input( 'Stay', min_value=0, max_value=365, value=int( st.session_state.get( 'nearbyobjects_stay', 8 ) ), step=1, key='nearbyobjects_stay', disabled=(
							nearby_mode not in [ 'nhats_summary', 'nhats_object' ]) )
			
			n4, n5, n6 = st.columns( 3 )
			with n4:
				nearby_launch = st.text_input( 'Launch Window', value=st.session_state.get( 'nearbyobjects_launch', '2020-2045' ), key='nearbyobjects_launch', placeholder='2020-2045', disabled=(
							nearby_mode not in [ 'nhats_summary', 'nhats_object' ]) )
			
			with n5:
				nearby_h = st.number_input( 'H Max', min_value=0.0, max_value=40.0, value=float( st.session_state.get( 'nearbyobjects_h', 26.0 ) ), step=0.1, key='nearbyobjects_h', disabled=(
							nearby_mode != 'nhats_summary') )
			
			with n6:
				nearby_occ = st.number_input( 'OCC Max', min_value=0, max_value=20, value=int( st.session_state.get( 'nearbyobjects_occ', 7 ) ), step=1, key='nearbyobjects_occ', disabled=(
							nearby_mode != 'nhats_summary') )
			
			st.markdown( '#### SBDB Options' )
			s1, s2, s3 = st.columns( 3 )
			with s1:
				nearby_include_physical = st.checkbox( 'Physical Params', value=bool( st.session_state.get( 'nearbyobjects_include_physical', True ) ), key='nearbyobjects_include_physical', disabled=(
							nearby_mode != 'object_lookup') )
			
			with s2:
				nearby_include_close_approaches = st.checkbox( 'CA Data', value=bool( st.session_state.get( 'nearbyobjects_include_close_approaches', True ) ), key='nearbyobjects_include_close_approaches', disabled=(
							nearby_mode != 'object_lookup') )
			
			with s3:
				nearby_include_discovery = st.checkbox( 'Discovery', value=bool( st.session_state.get( 'nearbyobjects_include_discovery', True ) ), key='nearbyobjects_include_discovery', disabled=(
							nearby_mode != 'object_lookup') )
			
			nearby_ca_body = st.text_input( 'CA Body', value=st.session_state.get( 'nearbyobjects_ca_body', 'Earth' ), key='nearbyobjects_ca_body', placeholder='Earth', disabled=(
						nearby_mode != 'object_lookup') )
			
			nearby_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'nearbyobjects_timeout', 20 ) ), step=1, key='nearbyobjects_timeout' )
			
			st.caption( 'Examples: close_approaches with Distance Max=10LD and Body=Earth; '
			            'object_lookup with Query=Apophis and Query Type=sstr; '
			            'nhats_object with Query=99942; fireballs with Start '
			            'Date=2014-01-01.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				nearby_submit = st.button( 'Submit', key='nearbyobjects_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='nearbyobjects_clear', on_click=_clear_nearbyobjects_state, width='stretch' )
			
			if nearby_submit:
				st.session_state[ 'astronomical_active_source' ] = 'Near Earth Objects'
	
	with right:
		active_source = st.session_state.get( 'astronomical_active_source', '' )
		result_keys: Dict[ str, str ] = { 'US Naval Observatory': 'navalobservatory_results',
				'Satellite Center': 'satellitecenter_results',
				'Astro Catalog': 'astrocatalog_results', 'Astro Query': 'astroquery_results',
				'Star Map': 'starmap_results', 'SIMBAD': 'simbad_results',
				'Space Weather': 'spaceweather_results', 'Star Chart': 'starchart_results',
				'Near Earth Objects': 'nearbyobjects_results', }
		
		if active_source in result_keys:
			active_result = st.session_state.get( result_keys[ active_source ] )
			_promote_astronomical_result( source=active_source, result=active_result )
		
		st.markdown( '#### Results' )
		if not active_source:
			st.info( 'Select a source, configure the request, and submit it to display results.' )
		else:
			st.caption( f'Active Source: {active_source}' )
		
		if active_source == 'US Naval Observatory':
			naval_output = st.empty( )
			
			if naval_submit:
				try:
					f = NavalObservatory( )
					
					result = f.fetch( mode='celnav', date_value=naval_date.strftime( '%Y-%m-%d' ), time_value=naval_time.strftime( '%H:%M:%S' ), latitude=float( naval_latitude ), longitude=float( naval_longitude ), location_label=naval_location_label, time=int( naval_timeout ) )
					
					st.session_state[ 'navalobservatory_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
			
			result = st.session_state.get( 'navalobservatory_results', { } )
			
			if not result:
				naval_output.text( 'No results.' )
			else:
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': result.get( 'mode', '' ), 'url': result.get( 'url', '' ),
						'params': params, 'location_label': result.get( 'location_label', '' ), } )
				
				if not data:
					st.info( 'No results returned.' )
				else:
					st.markdown( '#### Observation Summary' )
					c1, c2 = st.columns( 2 )
					
					with c1:
						if params.get( 'date', '' ):
							st.markdown( f"**Date:** {params.get( 'date', '' )}" )
						if params.get( 'time', '' ):
							st.markdown( f"**Time:** {params.get( 'time', '' )}" )
						if result.get( 'location_label', '' ):
							st.markdown( f"**Location Label:** {result.get( 'location_label', '' )}" )
					
					with c2:
						if params.get( 'coords', '' ):
							st.markdown( f"**Coordinates:** {params.get( 'coords', '' )}" )
					
					bodies: List[ Dict[ str, Any ] ] = [ ]
					
					if isinstance( data, dict ):
						for key in [ 'data', 'bodies', 'results', 'celestialBodies',
								'celestial_bodies' ]:
							value = data.get( key, None )
							if isinstance( value, list ):
								bodies = [ item for item in value if isinstance( item, dict ) ]
								break
					
					if bodies:
						st.markdown( '#### Celestial Bodies' )
						df_bodies = pd.DataFrame( bodies )
						if not df_bodies.empty:
							st.dataframe( df_bodies, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable celestial body rows were found.' )
					else:
						top_fields = { }
						
						if isinstance( data, dict ):
							for key in [ 'gha', 'dec', 'hc', 'zn', 'altitude', 'azimuth', 'sunrise',
									'sunset', 'moonrise', 'moonset' ]:
								if key in data:
									top_fields[ key ] = data.get( key )
						
						if top_fields:
							st.markdown( '#### Key Values' )
							st.json( top_fields )
						else:
							st.markdown( '#### Result' )
							st.json( data )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Satellite Center
		if active_source == 'Satellite Center':
			st.markdown( 'Results' )
			if satellite_submit:
				try:
					f = SatelliteCenter( )
					result = f.fetch( mode=satellite_mode, query=satellite_query, start_time=satellite_start_time, end_time=satellite_end_time, coordinate_systems=satellite_coordinate_systems, resolution_factor=int( satellite_resolution_factor ), time=int( satellite_timeout ) )
					
					st.session_state[ 'satellitecenter_results' ] = {
							'request': { 'mode': satellite_mode, 'query': satellite_query,
									'start_time': satellite_start_time,
									'end_time': satellite_end_time,
									'coordinate_systems': satellite_coordinate_systems,
									'resolution_factor': int( satellite_resolution_factor ),
									'timeout': int( satellite_timeout ), }, 'data': result or { }, }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Satellite Center request failed.' )
					st.exception( exc )
			
			result_wrapper = st.session_state.get( 'satellitecenter_results', { } )
			
			if not result_wrapper:
				st.text( 'No results.' )
			else:
				if (
						isinstance( result_wrapper, dict ) and 'request' in result_wrapper and 'data' in result_wrapper):
					request_meta = result_wrapper.get( 'request', { } )
					result = result_wrapper.get( 'data', { } )
				else:
					request_meta = { 'mode': satellite_mode, 'query': satellite_query,
							'start_time': satellite_start_time, 'end_time': satellite_end_time,
							'coordinate_systems': satellite_coordinate_systems,
							'resolution_factor': int( satellite_resolution_factor ),
							'timeout': int( satellite_timeout ), }
					result = result_wrapper if isinstance( result_wrapper, dict ) else { }
				
				render_mode = str( request_meta.get( 'mode', 'observatories' ) )
				
				st.markdown( '#### Request Metadata' )
				st.json( request_meta )
				
				if render_mode == 'observatories':
					items = result.get( 'Observatory', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in items:
							if isinstance( item, dict ):
								location_value = ''
								geo_value = item.get( 'GeoLocation', { } )
								
								if isinstance( geo_value, dict ):
									lat_value = geo_value.get( 'Latitude', '' )
									lon_value = geo_value.get( 'Longitude', '' )
									if str( lat_value ).strip( ) or str( lon_value ).strip( ):
										location_value = f'{lat_value}, {lon_value}'
								
								summary_rows.append( { 'Id': item.get( 'Id', '' ),
										'Name': item.get( 'Name', '' ),
										'Resolution': item.get( 'Resolution', '' ),
										'StartTime': item.get( 'StartTime', '' ),
										'EndTime': item.get( 'EndTime', '' ),
										'GeoLocation': location_value, } )
						
						_render_satellite_table( f'#### Observatories ({len( summary_rows )})', summary_rows )
					else:
						st.info( 'No observatories were returned.' )
				
				elif render_mode == 'ground_stations':
					items = [ ]
					
					if isinstance( result, dict ):
						items = result.get( 'GroundStation', [ ] )
						if not items:
							items = result.get( 'GroundStations', [ ] )
					
					if items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in items:
							if isinstance( item, dict ):
								coords = item.get( 'Location', { } )
								location_value = ''
								
								if isinstance( coords, dict ):
									lat_value = coords.get( 'Latitude', '' )
									lon_value = coords.get( 'Longitude', '' )
									if str( lat_value ).strip( ) or str( lon_value ).strip( ):
										location_value = f'{lat_value}, {lon_value}'
								
								summary_rows.append( { 'Id': item.get( 'Id', '' ),
										'Name': item.get( 'Name', '' ),
										'Code': item.get( 'Code', '' ),
										'Location': location_value, } )
						
						_render_satellite_table( f'#### Ground Stations ({len( summary_rows )})', summary_rows )
					else:
						st.markdown( '#### Ground Stations' )
						st.json( result )
				
				elif render_mode == 'locations':
					st.markdown( '#### Locations' )
					if isinstance( result, dict ):
						if 'Data' in result and isinstance( result.get( 'Data' ), list ):
							_render_satellite_table( '##### Position Samples', result.get( 'Data', [ ] ) )
						elif 'Coordinates' in result and isinstance( result.get( 'Coordinates' ), list ):
							_render_satellite_table( '##### Coordinates', result.get( 'Coordinates', [ ] ) )
						else:
							flat_rows: List[ Dict[ str, Any ] ] = [ ]
							for key, value in result.items( ):
								if isinstance( value, (str, int, float, bool) ) or value is None:
									flat_rows.append( { 'Field': key, 'Value': value } )
							
							if flat_rows:
								_render_satellite_table( '##### Summary', flat_rows )
							else:
								st.json( result )
					else:
						st.write( result )
				
				else:
					st.json( result )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Astro Catalog
		if active_source == 'Astro Catalog':
			st.markdown( 'Results' )
			
			if astro_submit:
				try:
					f = AstroCatalog( )
					result = f.fetch( mode=astro_mode, query=astro_query, quantity=astro_quantity, attributes=astro_attributes, arguments=astro_arguments, ra=astro_ra, dec=astro_dec, radius=int( astro_radius ), data_format=astro_format, time=int( astro_timeout ) )
					
					st.session_state[ 'astrocatalog_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Astronomy Catalog request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'astrocatalog_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Metadata' )
				st.json( { 'mode': astro_mode, 'query': astro_query, 'quantity': astro_quantity,
						'attributes': astro_attributes, 'arguments': astro_arguments,
						'ra': astro_ra, 'dec': astro_dec, 'radius': int( astro_radius ),
						'format': astro_format, } )
				
				parsed_result = result
				
				if isinstance( result, str ):
					text_value = result.strip( )
					
					if astro_format == 'json':
						try:
							parsed_result = json.loads( text_value )
						except Exception:
							parsed_result = result
					else:
						parsed_result = result
				
				if isinstance( parsed_result, list ):
					st.markdown( f'#### Result Rows ({len( parsed_result )})' )
					df_catalog = pd.DataFrame( parsed_result )
					if not df_catalog.empty:
						st.dataframe( df_catalog, use_container_width=True, hide_index=True )
					else:
						st.text_area( 'Results', value=str( parsed_result ), height=320 )
				
				elif isinstance( parsed_result, dict ):
					candidate_rows: List[ Dict[ str, Any ] ] = [ ]
					
					for key in [ 'results', 'items', 'data', 'objects' ]:
						value = parsed_result.get( key, None )
						if isinstance( value, list ) and value:
							candidate_rows = [ item for item in value if isinstance( item, dict ) ]
							break
					
					if candidate_rows:
						st.markdown( f'#### Result Rows ({len( candidate_rows )})' )
						df_catalog = pd.DataFrame( candidate_rows )
						
						if not df_catalog.empty:
							st.dataframe( df_catalog, use_container_width=True, hide_index=True )
						else:
							st.json( parsed_result )
					
					else:
						title_value = (
									parsed_result.get( 'name' ) or parsed_result.get( 'alias' ) or parsed_result.get( 'event' ) or parsed_result.get( 'id' ) or 'Catalog Result')
						
						st.markdown( f'### {title_value}' )
						top_fields: Dict[ str, Any ] = { }
						for key in [ 'name', 'alias', 'ra', 'dec', 'redshift', 'type',
								'claimedtype', 'schema' ]:
							if key in parsed_result:
								top_fields[ key ] = parsed_result.get( key )
						
						if top_fields:
							st.json( top_fields )
						
						for key in [ 'summary', 'description', 'comments' ]:
							if key in parsed_result and str( parsed_result.get( key ) ).strip( ):
								st.markdown( f'#### {key.title( )}' )
								st.write( str( parsed_result.get( key ) ) )
						
						if not top_fields:
							st.json( parsed_result )
				
				elif isinstance( parsed_result, str ):
					st.markdown( '#### Result Text' )
					st.text_area( 'Results', value=parsed_result, height=320 )
				
				else:
					st.text_area( 'Results', value=str( parsed_result ), height=320 )
				
				with st.expander( 'Raw Result', expanded=False ):
					if isinstance( result, (dict, list) ):
						st.json( result )
					else:
						st.text_area( 'Raw', value=str( result ), height=240 )
		
		# -------- Astro Query
		if active_source == 'Astro Query':
			st.markdown( 'Results' )
			if astroquery_submit:
				try:
					f = AstroQuery( )
					result = f.fetch( mode=astroquery_mode, query=astroquery_query, ra=astroquery_ra, dec=astroquery_dec, radius=float( astroquery_radius ), radius_unit=astroquery_radius_unit, row_limit=int( astroquery_row_limit ) )
					
					st.session_state[ 'astroquery_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Astro Query request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'astroquery_results', { } )
			if not result:
				st.text( 'No results.' )
			else:
				meta_col1, meta_col2 = st.columns( 2 )
				with meta_col1:
					if isinstance( result, dict ) and 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if isinstance( result, dict ) and 'query' in result:
						st.markdown( f"**Query:** {result.get( 'query', '' )}" )
					if isinstance( result, dict ) and 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_col2:
					if isinstance( result, dict ) and 'row_limit' in result:
						st.markdown( f"**Row Limit:** {result.get( 'row_limit', '' )}" )
					if isinstance( result, dict ) and 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if isinstance( result, dict ) and 'radius' in result:
						st.markdown( f"**Radius:** {result.get( 'radius', '' )} "
						             f"{result.get( 'radius_unit', '' )}" )
				
				rows = result.get( 'rows', [ ] ) if isinstance( result, dict ) else [ ]
				columns = result.get( 'columns', [ ] ) if isinstance( result, dict ) else [ ]
				
				if columns:
					st.markdown( '#### Columns' )
					st.write( ', '.join( [ str( c ) for c in columns ] ) )
				
				if not rows:
					st.info( 'No rows returned.' )
				else:
					df_rows = pd.DataFrame( rows )
					
					if not df_rows.empty and columns:
						ordered_columns = [ c for c in columns if c in df_rows.columns ]
						if ordered_columns:
							df_rows = df_rows[ ordered_columns ]
					
					st.markdown( f'#### Result Rows ({len( df_rows )})' )
					
					if not df_rows.empty:
						st.dataframe( df_rows, use_container_width=True, hide_index=True )
					else:
						st.info( 'No displayable rows were returned.' )
					
					with st.expander( 'Row Details', expanded=False ):
						for idx, row in enumerate( rows, start=1 ):
							label = row.get( 'MAIN_ID', f'Row {idx}' )
							with st.expander( f'Row {idx}: {label}', expanded=False ):
								st.json( row )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Star Map
		if active_source == 'Star Map':
			st.markdown( 'Results' )
			
			if starmap_submit:
				try:
					f = StarMap( )
					result = f.fetch( mode=starmap_mode, query=starmap_query, ra=float( starmap_ra ), dec=float( starmap_dec ), zoom=int( starmap_zoom ), image_source=starmap_image_source, box_color=starmap_box_color, show_box=bool( starmap_show_box ), show_grid=bool( starmap_show_grid ), show_lines=bool( starmap_show_lines ), show_boundaries=bool( starmap_show_boundaries ), show_const_names=bool( starmap_show_const_names ), time=int( starmap_timeout ) )
					
					st.session_state[ 'starmap_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Star Map request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'starmap_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_col1, meta_col2 = st.columns( 2 )
				
				with meta_col1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'object' in result:
						st.markdown( f"**Object:** {result.get( 'object', '' )}" )
					if 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_col2:
					if 'zoom' in result:
						st.markdown( f"**Zoom:** {result.get( 'zoom', '' )}" )
					if 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if 'image_source' in result:
						st.markdown( f"**Image Source:** {result.get( 'image_source', '' )}" )
				
				if result.get( 'interactive_url', '' ):
					st.markdown( f"**Interactive URL:** "
					             f"{result.get( 'interactive_url', '' )}" )
				
				if result.get( 'snapshot_page_url', '' ):
					st.markdown( f"**Snapshot Page URL:** {result.get( 'snapshot_page_url', '' )}" )
				
				preferred_image_url = result.get( 'preferred_image_url', '' )
				if preferred_image_url:
					st.markdown( '#### Preferred Image' )
					st.image( preferred_image_url, use_container_width=True )
				
				image_links = result.get( 'image_links', { } ) or { }
				if image_links:
					st.markdown( '#### Available Image Links' )
					st.json( image_links )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'html_preview', '' ):
					st.markdown( '#### HTML Preview' )
					st.text_area( '', value=result.get( 'html_preview', '' ), height=220, key='starmap_html_preview' )
		
		# -------- SIMBAD
		if active_source == 'SIMBAD':
			st.markdown( 'Results' )
			
			if simbad_submit:
				try:
					f = AstroQuery( )
					result = f.fetch( mode=simbad_mode, query=simbad_query, ra=simbad_ra, dec=simbad_dec, radius=float( simbad_radius ), radius_unit=simbad_radius_unit, row_limit=int( simbad_row_limit ) )
					
					st.session_state[ 'simbad_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'SIMBAD request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'simbad_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if isinstance( result, dict ) and 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if isinstance( result, dict ) and 'query' in result:
						st.markdown( f"**Query:** {result.get( 'query', '' )}" )
					if isinstance( result, dict ) and 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_c2:
					if isinstance( result, dict ) and 'row_limit' in result:
						st.markdown( f"**Row Limit:** {result.get( 'row_limit', '' )}" )
					if isinstance( result, dict ) and 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if isinstance( result, dict ) and 'radius' in result:
						st.markdown( f"**Radius:** {result.get( 'radius', '' )} "
						             f"{result.get( 'radius_unit', '' )}" )
				
				columns = result.get( 'columns', [ ] ) if isinstance( result, dict ) else [ ]
				rows = result.get( 'rows', [ ] ) if isinstance( result, dict ) else [ ]
				
				if columns:
					st.markdown( '#### Columns' )
					st.write( columns )
				
				if rows:
					st.markdown( '#### Rows' )
					df_simbad = pd.DataFrame( rows )
					st.dataframe( df_simbad, use_container_width=True, hide_index=True )
				else:
					st.text( 'No rows returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Space Weather
		if active_source == 'Space Weather':
			if spaceweather_submit:
				try:
					f = SpaceWeather( )
					result = f.fetch( mode=spaceweather_mode, start_date=str( spaceweather_start_date ), end_date=str( spaceweather_end_date ), location=str( spaceweather_location or 'ALL' ), catalog=str( spaceweather_catalog or 'ALL' ), notification_type=str( spaceweather_notification_type or 'all' ), most_accurate_only=bool( spaceweather_most_accurate_only ), complete_entry_only=bool( spaceweather_complete_entry_only ), speed=int( spaceweather_speed ), half_angle=int( spaceweather_half_angle ), keyword=str( spaceweather_keyword or '' ), time=int( spaceweather_timeout ) )
					
					st.session_state[ 'spaceweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Space Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'satellitecenter_results', { } )
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Result Summary' )
				
				if satellite_mode == 'observatories':
					items = result.get( 'Observatory', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						df_sat = pd.DataFrame( items )
						if not df_sat.empty:
							st.caption( f'Observatories returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable observatory rows were found.' )
						
						for idx, item in enumerate( items, start=1 ):
							label = item.get( 'Id', f'Observatory {idx}' )
							with st.expander( f'Observatory {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No observatories returned.' )
				
				elif satellite_mode == 'ground_stations':
					items = result.get( 'GroundStation', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						df_sat = pd.DataFrame( items )
						if not df_sat.empty:
							st.caption( f'Ground stations returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable ground-station rows were found.' )
						
						for idx, item in enumerate( items, start=1 ):
							label = item.get( 'Id', f'Ground Station {idx}' )
							with st.expander( f'Ground Station {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No ground stations returned.' )
				
				else:
					data_items = result.get( 'Data', [ ] ) if isinstance( result, dict ) else [ ]
					
					if data_items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in data_items:
							if isinstance( item, dict ):
								summary_rows.append( { 'Id': item.get( 'Id', '' ),
										'CoordinateSystem': item.get( 'CoordinateSystem', '' ),
										'Start': item.get( 'StartTime', '' ),
										'End': item.get( 'EndTime', '' ) } )
						
						df_sat = pd.DataFrame( summary_rows )
						if not df_sat.empty:
							st.caption( f'Location sets returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable location-set summary rows were found.' )
						
						for idx, item in enumerate( data_items, start=1 ):
							label = item.get( 'Id', f'Trajectory {idx}' )
							with st.expander( f'Location Set {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No location data returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Star Chart
		if active_source == 'Star Chart':
			if starchart_submit:
				try:
					f = StarChart( )
					result = f.fetch( mode=starchart_mode, query=str( starchart_query or '' ), ra=float( starchart_ra ), dec=float( starchart_dec ), zoom=int( starchart_zoom ), image_source=str( starchart_image_source ), box_color=str( starchart_box_color or 'yellow' ), show_box=bool( starchart_show_box ), show_grid=bool( starchart_show_grid ), show_lines=bool( starchart_show_lines ), show_boundaries=bool( starchart_show_boundaries ), show_const_names=bool( starchart_show_const_names ), width=int( starchart_width ), height=int( starchart_height ), magnitude=float( starchart_magnitude ), time=int( starchart_timeout ) )
					
					st.session_state[ 'starchart_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Star Chart request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'starchart_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				search_payload = result.get( 'search', { } ) if isinstance( result, dict ) else { }
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				chart_url = result.get( 'chart_url', '' ) if isinstance( result, dict ) else ''
				image_url = result.get( 'image_url', '' ) if isinstance( result, dict ) else ''
				
				st.markdown( '#### Chart Summary' )
				
				meta_c1, meta_c2 = st.columns( 2 )
				with meta_c1:
					if mode_value:
						st.markdown( f'**Mode:** {mode_value}' )
					if result.get( 'url', '' ):
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
					if params:
						if 'query' in params and params.get( 'query', '' ):
							st.markdown( f"**Query:** {params.get( 'query', '' )}" )
						if 'ra' in params and 'dec' in params:
							st.markdown( f"**Coordinates:** RA={params.get( 'ra', '' )}, "
							             f"Dec={params.get( 'dec', '' )}" )
				
				with meta_c2:
					if chart_url:
						st.markdown( f'**Chart Link:** {chart_url}' )
					if image_url:
						st.markdown( f'**Image URL:** {image_url}' )
					if 'zoom' in params:
						st.markdown( f"**Zoom:** {params.get( 'zoom', '' )}" )
				
				if image_url:
					st.markdown( '#### Preview' )
					st.image( image_url, use_container_width=True )
				
				if search_payload:
					st.markdown( '#### Object Search' )
					
					if isinstance( search_payload, list ):
						items = [ item for item in search_payload if isinstance( item, dict ) ]
					elif isinstance( search_payload, dict ):
						items = [ search_payload ]
					else:
						items = [ ]
					
					if items:
						for idx, item in enumerate( items, start=1 ):
							title_value = (
									item.get( 'name' ) or item.get( 'title' ) or item.get( 'object' ) or f'Result {idx}')
							
							with st.expander( f'Result {idx}: {title_value}', expanded=False ):
								summary_parts: List[ str ] = [ ]
								
								for key in [ 'ra', 'dec', 'type', 'constellation', 'magnitude' ]:
									if key in item and str( item.get( key ) ).strip( ):
										summary_parts.append( f'{key}={item.get( key )}' )
								
								if summary_parts:
									st.caption( ' | '.join( summary_parts ) )
								
								st.json( item )
					else:
						st.json( search_payload )
				
				if data:
					st.markdown( '#### Chart Data' )
					if isinstance( data, list ):
						df_chart = pd.DataFrame( data )
						if not df_chart.empty:
							st.dataframe( df_chart, use_container_width=True, hide_index=True )
						else:
							st.json( data )
					elif isinstance( data, dict ):
						key_fields = { }
						for key in [ 'name', 'title', 'ra', 'dec', 'type', 'constellation',
								'magnitude' ]:
							if key in data:
								key_fields[ key ] = data.get( key )
						
						if key_fields:
							st.json( key_fields )
						else:
							st.json( data )
					else:
						st.write( data )
				
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
		
		# -------- Nearby Objects
		if active_source == 'Near Earth Objects':
			if nearby_submit:
				try:
					f = NearbyObjects( )
					result = f.fetch( mode=nearby_mode, start_date=str( nearby_start_date ), end_date=str( nearby_end_date ), query=str( nearby_query or '' ).strip( ), query_type=str( nearby_query_type ), dist_max=str( nearby_dist_max or '10LD' ), body=str( nearby_body or 'Earth' ), sort=str( nearby_sort or 'date' ), limit=int( nearby_limit ), dv=float( nearby_dv ), dur=int( nearby_dur ), stay=int( nearby_stay ), launch=str( nearby_launch or '2020-2045' ), h=float( nearby_h ), occ=int( nearby_occ ), include_physical=bool( nearby_include_physical ), include_close_approaches=bool( nearby_include_close_approaches ), ca_body=str( nearby_ca_body or 'Earth' ), include_discovery=bool( nearby_include_discovery ), time=int( nearby_timeout ) )
					
					st.session_state[ 'nearbyobjects_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Near Earth Objects request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'nearbyobjects_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					if 'count' in result:
						st.markdown( f"**Count:** {result.get( 'count', '' )}" )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'fields', [ ] ) and result.get( 'data', [ ] ):
					st.markdown( '#### Results' )
					df_nearby = pd.DataFrame( result.get( 'data', [ ] ), columns=result.get( 'fields', [ ] ) )
					st.dataframe( df_nearby, use_container_width=True, hide_index=True )
				
				elif 'data' in result:
					st.markdown( '#### Results' )
					data = result.get( 'data', { } )
					if isinstance( data, list ):
						if data:
							df_nearby = pd.DataFrame( data )
							st.dataframe( df_nearby, use_container_width=True, hide_index=True )
						else:
							st.text( 'No rows returned.' )
					else:
						st.json( data )
				
				if result.get( 'signature', { } ):
					with st.expander( 'Signature', expanded=False ):
						st.json( result.get( 'signature', { } ) )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

# ==============================================================================
# POPULATION MODE
# ==============================================================================
elif mode == 'Demographic':
	st.subheader( f'🩺 Demographics & Health Data' )
	st.divider( )
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	if 'demographic_active_source' not in st.session_state:
		st.session_state[ 'demographic_active_source' ] = ''
	
	with left:
		# ---------------------
		# ---- Expander U.S. Census Bureau
		# ---------------------
		with st.expander( label='U.S. Census Bureau', icon='📊', expanded=False ):
			CENSUS_MODES = [ 'variables', 'data' ]
			
			def _clear_census_state( ) -> None:
				st.session_state[ 'census_clear_request' ] = True
			
			def _validate_census_year( value: object ) -> str:
				text = str( value or '' ).strip( )
				if not re.fullmatch( r'\d{4}', text ):
					raise ValueError( 'Year must be a four-digit Census API vintage year.' )
				
				return text
			
			def _validate_census_dataset( value: object ) -> str:
				text = str( value or '' ).strip( ).strip( '/' )
				if not text:
					raise ValueError( 'Dataset is required.' )
				
				if not re.fullmatch( r'[A-Za-z0-9_\-/]+', text ):
					raise ValueError( 'Dataset may only contain letters, numbers, underscores, hyphens, '
					                  'and forward slashes.' )
				
				return text
			
			def _validate_census_fields( value: object ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					raise ValueError( 'Fields are required for Census data mode.' )
				
				fields = [ item.strip( ) for item in text.split( ',' ) if item.strip( ) ]
				
				if not fields:
					raise ValueError( 'Fields are required for Census data mode.' )
				
				for field in fields:
					if not re.fullmatch( r'[A-Za-z0-9_]+', field ):
						raise ValueError( f'Invalid Census field name: {field}' )
				
				return ','.join( fields )
			
			def _validate_census_geography_clause( name: str, value: object, required: bool = False ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					if required:
						raise ValueError( f'{name} is required.' )
					return ''
				
				if ':' not in text:
					raise ValueError( f'{name} must use Census geography syntax such as state:*.' )
				
				return text
			
			if 'census_results' not in st.session_state:
				st.session_state[ 'census_results' ] = { }
			
			if 'census_clear_request' not in st.session_state:
				st.session_state[ 'census_clear_request' ] = False
			
			if st.session_state.get( 'census_mode', 'variables' ) not in CENSUS_MODES:
				st.session_state[ 'census_mode' ] = 'variables'
			
			if 'census_year' not in st.session_state:
				st.session_state[ 'census_year' ] = '2022'
			
			if 'census_dataset' not in st.session_state:
				st.session_state[ 'census_dataset' ] = 'acs/acs5'
			
			if 'census_fields' not in st.session_state:
				st.session_state[ 'census_fields' ] = 'NAME,B01001_001E'
			
			if 'census_for' not in st.session_state:
				st.session_state[ 'census_for' ] = 'state:*'
			
			if 'census_in' not in st.session_state:
				st.session_state[ 'census_in' ] = ''
			
			if 'census_predicates' not in st.session_state:
				st.session_state[ 'census_predicates' ] = ''
			
			if 'census_timeout' not in st.session_state:
				st.session_state[ 'census_timeout' ] = 20
			
			if st.session_state.get( 'census_clear_request', False ):
				st.session_state[ 'census_mode' ] = 'variables'
				st.session_state[ 'census_year' ] = '2022'
				st.session_state[ 'census_dataset' ] = 'acs/acs5'
				st.session_state[ 'census_fields' ] = 'NAME,B01001_001E'
				st.session_state[ 'census_for' ] = 'state:*'
				st.session_state[ 'census_in' ] = ''
				st.session_state[ 'census_predicates' ] = ''
				st.session_state[ 'census_timeout' ] = 20
				st.session_state[ 'census_results' ] = { }
				st.session_state[ 'census_clear_request' ] = False
			
			census_mode = st.selectbox( 'Mode', options=CENSUS_MODES, index=CENSUS_MODES.index( st.session_state.get( 'census_mode', 'variables' ) ), key='census_mode', help=(
				'variables = dataset variable metadata; '
				'data = tabular Census query using get/for/in.') )
			
			census_year = st.text_input( 'Year', value=st.session_state.get( 'census_year', '2022' ), key='census_year', placeholder='2022' )
			
			census_dataset = st.text_input( 'Dataset', value=st.session_state.get( 'census_dataset', 'acs/acs5' ), key='census_dataset', placeholder='acs/acs5' )
			
			census_fields = st.text_area( 'Fields (get)', value=st.session_state.get( 'census_fields', 'NAME,B01001_001E' ), height=90, key='census_fields', placeholder='NAME,B01001_001E', disabled=(
						census_mode != 'data') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				census_for = st.text_input( 'For', value=st.session_state.get( 'census_for', 'state:*' ), key='census_for', placeholder='state:*', disabled=(
							census_mode != 'data') )
			
			with c2:
				census_in = st.text_input( 'In', value=st.session_state.get( 'census_in', '' ), key='census_in', placeholder='state:24', disabled=(
							census_mode != 'data') )
			
			census_predicates = st.text_area( 'Predicates', value=st.session_state.get( 'census_predicates', '' ), height=90, key='census_predicates', placeholder='SEX=1\nAGE=15', disabled=(
						census_mode != 'data'), help='Optional newline-delimited key=value filters.' )
			
			census_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'census_timeout', 20 ) ), step=1, key='census_timeout' )
			
			st.caption( 'Examples: dataset = acs/acs5, fields = NAME,B01001_001E, '
			            'for = state:*' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				census_submit = st.button( 'Submit', key='census_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='census_clear', on_click=_clear_census_state, width='stretch' )
			
			if census_submit:
				st.session_state[ 'demographic_active_source' ] = 'u_s_census_bureau'
		
		# ---------------------
		# ---- Expander CDC SOCRATA
		# ---------------------
		with st.expander( label='CDC Socrata', icon='🩺', expanded=False ):
			SOCRATA_MODES = [ 'rows', 'metadata' ]
			
			SOCRATA_CDC_DOMAINS = [ 'data.cdc.gov', 'chronicdata.cdc.gov' ]
			
			def _clear_socrata_state( ) -> None:
				st.session_state[ 'socrata_clear_request' ] = True
			
			def _validate_socrata_dataset_id( value: object ) -> str:
				text = str( value or '' ).strip( ).replace( '.json', '' ).strip( '/' )
				
				if not text:
					raise ValueError( 'Dataset ID is required.' )
				
				if not re.fullmatch( r'[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}', text ):
					raise ValueError( 'Dataset ID must use the Socrata four-by-four format, '
					                  'such as q8xq-ygsk.' )
				
				return text.lower( )
			
			if 'socrata_results' not in st.session_state:
				st.session_state[ 'socrata_results' ] = { }
			
			if 'socrata_clear_request' not in st.session_state:
				st.session_state[ 'socrata_clear_request' ] = False
			
			if st.session_state.get( 'socrata_mode', 'rows' ) not in SOCRATA_MODES:
				st.session_state[ 'socrata_mode' ] = 'rows'
			
			if st.session_state.get( 'socrata_domain', 'data.cdc.gov' ) not in SOCRATA_CDC_DOMAINS:
				st.session_state[ 'socrata_domain' ] = 'data.cdc.gov'
			
			if 'socrata_dataset_id' not in st.session_state:
				st.session_state[ 'socrata_dataset_id' ] = 'q8xq-ygsk'
			
			if 'socrata_select' not in st.session_state:
				st.session_state[ 'socrata_select' ] = ''
			
			if 'socrata_where' not in st.session_state:
				st.session_state[ 'socrata_where' ] = ''
			
			if 'socrata_order' not in st.session_state:
				st.session_state[ 'socrata_order' ] = ''
			
			if 'socrata_group' not in st.session_state:
				st.session_state[ 'socrata_group' ] = ''
			
			if 'socrata_limit' not in st.session_state:
				st.session_state[ 'socrata_limit' ] = 25
			
			if 'socrata_offset' not in st.session_state:
				st.session_state[ 'socrata_offset' ] = 0
			
			if 'socrata_timeout' not in st.session_state:
				st.session_state[ 'socrata_timeout' ] = 20
			
			if st.session_state.get( 'socrata_clear_request', False ):
				st.session_state[ 'socrata_mode' ] = 'rows'
				st.session_state[ 'socrata_domain' ] = 'data.cdc.gov'
				st.session_state[ 'socrata_dataset_id' ] = 'q8xq-ygsk'
				st.session_state[ 'socrata_select' ] = ''
				st.session_state[ 'socrata_where' ] = ''
				st.session_state[ 'socrata_order' ] = ''
				st.session_state[ 'socrata_group' ] = ''
				st.session_state[ 'socrata_limit' ] = 25
				st.session_state[ 'socrata_offset' ] = 0
				st.session_state[ 'socrata_timeout' ] = 20
				st.session_state[ 'socrata_results' ] = { }
				st.session_state[ 'socrata_clear_request' ] = False
			
			socrata_mode = st.selectbox( 'Mode', options=SOCRATA_MODES, index=SOCRATA_MODES.index( st.session_state.get( 'socrata_mode', 'rows' ) ), key='socrata_mode', help='rows = query dataset rows; metadata = inspect dataset metadata.' )
			
			socrata_domain = st.selectbox( 'Domain', options=SOCRATA_CDC_DOMAINS, index=SOCRATA_CDC_DOMAINS.index( st.session_state.get( 'socrata_domain', 'data.cdc.gov' ) ), key='socrata_domain', help='CDC Socrata portal domain.' )
			
			socrata_dataset_id = st.text_input( 'Dataset ID', value=st.session_state.get( 'socrata_dataset_id', 'q8xq-ygsk' ), key='socrata_dataset_id', placeholder='q8xq-ygsk' )
			
			socrata_select = st.text_area( 'Select', value=st.session_state.get( 'socrata_select', '' ), height=80, key='socrata_select', placeholder='locationname,datavaluetype,datavalue', disabled=(
						socrata_mode != 'rows') )
			
			socrata_where = st.text_area( 'Where', value=st.session_state.get( 'socrata_where', '' ), height=100, key='socrata_where', placeholder="year = '2020'", disabled=(
						socrata_mode != 'rows') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				socrata_order = st.text_input( 'Order', value=st.session_state.get( 'socrata_order', '' ), key='socrata_order', placeholder='locationname ASC', disabled=(
							socrata_mode != 'rows') )
			
			with c2:
				socrata_group = st.text_input( 'Group', value=st.session_state.get( 'socrata_group', '' ), key='socrata_group', placeholder='locationname', disabled=(
							socrata_mode != 'rows') )
			
			c3, c4, c5 = st.columns( 3 )
			with c3:
				socrata_limit = st.number_input( 'Limit', min_value=1, max_value=50000, value=int( st.session_state.get( 'socrata_limit', 25 ) ), step=1, key='socrata_limit', disabled=(
							socrata_mode != 'rows'), help='Socrata SODA 2.0 endpoints allow $limit values up to 50,000.' )
			
			with c4:
				socrata_offset = st.number_input( 'Offset', min_value=0, max_value=1000000, value=int( st.session_state.get( 'socrata_offset', 0 ) ), step=1, key='socrata_offset', disabled=(
							socrata_mode != 'rows') )
			
			with c5:
				socrata_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'socrata_timeout', 20 ) ), step=1, key='socrata_timeout' )
			
			st.caption( 'Example dataset: q8xq-ygsk on data.cdc.gov. '
			            'Use SoQL clauses for select, where, order, and group.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				socrata_submit = st.button( 'Submit', key='socrata_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='socrata_clear', on_click=_clear_socrata_state, width='stretch' )
			
			if socrata_submit:
				st.session_state[ 'demographic_active_source' ] = 'cdc_socrata'
		
		# ---------------------
		# ---- Expander US Health Data
		# ---------------------
		with st.expander( label='U.S. Health', icon='🏥', expanded=False ):
			HEALTHDATA_MODES = [ 'rows', 'metadata' ]
			HEALTHDATA_DOMAINS = [ 'healthdata.gov' ]
			
			def _clear_healthdata_state( ) -> None:
				st.session_state[ 'healthdata_clear_request' ] = True
			
			def _validate_healthdata_dataset_id( value: object ) -> str:
				text = str( value or '' ).strip( ).replace( '.json', '' ).strip( '/' )
				
				if not text:
					raise ValueError( 'Dataset ID is required.' )
				
				if not re.fullmatch( r'[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}', text ):
					raise ValueError( 'Dataset ID must use the Socrata four-by-four format, '
					                  'such as abcd-1234.' )
				
				return text.lower( )
			
			if 'healthdata_results' not in st.session_state:
				st.session_state[ 'healthdata_results' ] = { }
			
			if 'healthdata_clear_request' not in st.session_state:
				st.session_state[ 'healthdata_clear_request' ] = False
			
			if st.session_state.get( 'healthdata_mode', 'rows' ) not in HEALTHDATA_MODES:
				st.session_state[ 'healthdata_mode' ] = 'rows'
			
			if st.session_state.get( 'healthdata_domain', 'healthdata.gov' ) not in HEALTHDATA_DOMAINS:
				st.session_state[ 'healthdata_domain' ] = 'healthdata.gov'
			
			if 'healthdata_dataset_id' not in st.session_state:
				st.session_state[ 'healthdata_dataset_id' ] = ''
			
			if 'healthdata_select' not in st.session_state:
				st.session_state[ 'healthdata_select' ] = ''
			
			if 'healthdata_where' not in st.session_state:
				st.session_state[ 'healthdata_where' ] = ''
			
			if 'healthdata_order' not in st.session_state:
				st.session_state[ 'healthdata_order' ] = ''
			
			if 'healthdata_group' not in st.session_state:
				st.session_state[ 'healthdata_group' ] = ''
			
			if 'healthdata_limit' not in st.session_state:
				st.session_state[ 'healthdata_limit' ] = 25
			
			if 'healthdata_offset' not in st.session_state:
				st.session_state[ 'healthdata_offset' ] = 0
			
			if 'healthdata_timeout' not in st.session_state:
				st.session_state[ 'healthdata_timeout' ] = 20
			
			if st.session_state.get( 'healthdata_clear_request', False ):
				st.session_state[ 'healthdata_mode' ] = 'rows'
				st.session_state[ 'healthdata_domain' ] = 'healthdata.gov'
				st.session_state[ 'healthdata_dataset_id' ] = ''
				st.session_state[ 'healthdata_select' ] = ''
				st.session_state[ 'healthdata_where' ] = ''
				st.session_state[ 'healthdata_order' ] = ''
				st.session_state[ 'healthdata_group' ] = ''
				st.session_state[ 'healthdata_limit' ] = 25
				st.session_state[ 'healthdata_offset' ] = 0
				st.session_state[ 'healthdata_timeout' ] = 20
				st.session_state[ 'healthdata_results' ] = { }
				st.session_state[ 'healthdata_clear_request' ] = False
			
			healthdata_mode = st.selectbox( 'Mode', options=HEALTHDATA_MODES, index=HEALTHDATA_MODES.index( st.session_state.get( 'healthdata_mode', 'rows' ) ), key='healthdata_mode', help='rows = query dataset rows; metadata = inspect dataset metadata.' )
			
			healthdata_domain = st.selectbox( 'Domain', options=HEALTHDATA_DOMAINS, index=HEALTHDATA_DOMAINS.index( st.session_state.get( 'healthdata_domain', 'healthdata.gov' ) ), key='healthdata_domain', help='HealthData.gov Socrata portal domain.' )
			
			healthdata_dataset_id = st.text_input( 'Dataset ID', value=st.session_state.get( 'healthdata_dataset_id', '' ), key='healthdata_dataset_id', placeholder='abcd-1234' )
			
			healthdata_select = st.text_area( 'Select', value=st.session_state.get( 'healthdata_select', '' ), height=80, key='healthdata_select', placeholder='column1,column2', disabled=(
						healthdata_mode != 'rows') )
			
			healthdata_where = st.text_area( 'Where', value=st.session_state.get( 'healthdata_where', '' ), height=100, key='healthdata_where', placeholder="year = '2024'", disabled=(
						healthdata_mode != 'rows') )
			
			c1, c2 = st.columns( 2 )
			with c1:
				healthdata_order = st.text_input( 'Order', value=st.session_state.get( 'healthdata_order', '' ), key='healthdata_order', placeholder='column1 ASC', disabled=(
							healthdata_mode != 'rows') )
			
			with c2:
				healthdata_group = st.text_input( 'Group', value=st.session_state.get( 'healthdata_group', '' ), key='healthdata_group', placeholder='column1', disabled=(
							healthdata_mode != 'rows') )
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				healthdata_limit = st.number_input( 'Limit', min_value=1, max_value=50000, value=int( st.session_state.get( 'healthdata_limit', 25 ) ), step=1, key='healthdata_limit', disabled=(
							healthdata_mode != 'rows'), help='Socrata SODA 2.0 endpoints allow $limit values up to 50,000.' )
			
			with c4:
				healthdata_offset = st.number_input( 'Offset', min_value=0, max_value=1000000, value=int( st.session_state.get( 'healthdata_offset', 0 ) ), step=1, key='healthdata_offset', disabled=(
							healthdata_mode != 'rows') )
			
			with c5:
				healthdata_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'healthdata_timeout', 20 ) ), step=1, key='healthdata_timeout' )
			
			st.caption( 'HealthData.gov exposes open API access through Socrata. '
			            'Use SoQL-style clauses for select, where, order, and group.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				healthdata_submit = st.button( 'Submit', key='healthdata_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='healthdata_clear', on_click=_clear_healthdata_state, width='stretch' )
			
			if healthdata_submit:
				st.session_state[ 'demographic_active_source' ] = 'u_s_health'
		
		# ---------------------
		# ---- Expander WHO Global Health
		# ---------------------
		with st.expander( label='WHO Global', icon='🌍', expanded=False ):
			WHO_MODES = [ 'indicator_registry', 'athena' ]
			WHO_QUERY_PRESETS = [ 'Indicator', 'Dimension', 'DIMENSION/COUNTRY/DimensionValues',
					'DIMENSION/REGION', 'WHOSIS_000001', 'Custom...' ]
			
			WHO_FORMATS = [ 'json', 'xml', 'csv', 'csv&profile=text', 'csv&profile=verbose' ]
			
			def _clear_who_state( ) -> None:
				st.session_state[ 'who_clear_request' ] = True
			
			def _validate_who_query_path( value: object ) -> str:
				text = str( value or '' ).strip( ).lstrip( '/' )
				
				if not text:
					raise ValueError( 'Query Path is required for WHO Athena mode.' )
				
				if text.startswith( 'http://' ) or text.startswith( 'https://' ):
					raise ValueError( 'Query Path must be a path segment only, not a full URL.' )
				
				if '..' in text:
					raise ValueError( 'Query Path cannot contain parent-directory markers.' )
				
				if not re.fullmatch( r"[A-Za-z0-9_\-/$(),.'% =]+(?:\?.*)?", text ):
					raise ValueError( 'Query Path contains unsupported characters for this request.' )
				
				return text
			
			if 'who_results' not in st.session_state:
				st.session_state[ 'who_results' ] = { }
			
			if 'who_clear_request' not in st.session_state:
				st.session_state[ 'who_clear_request' ] = False
			
			if st.session_state.get( 'who_mode', 'indicator_registry' ) not in WHO_MODES:
				st.session_state[ 'who_mode' ] = 'indicator_registry'
			
			if 'who_query_path' not in st.session_state:
				st.session_state[ 'who_query_path' ] = ''
			
			if st.session_state.get( 'who_query_path', '' ) in WHO_QUERY_PRESETS:
				default_who_query_choice = st.session_state.get( 'who_query_path', 'Indicator' )
			elif str( st.session_state.get( 'who_query_path', '' ) ).strip( ):
				default_who_query_choice = 'Custom...'
			else:
				default_who_query_choice = 'Indicator'
			
			if 'who_query_choice' not in st.session_state:
				st.session_state[ 'who_query_choice' ] = default_who_query_choice
			
			if st.session_state.get( 'who_query_choice', 'Indicator' ) not in WHO_QUERY_PRESETS:
				st.session_state[ 'who_query_choice' ] = default_who_query_choice
			
			if 'who_custom_query_path' not in st.session_state:
				st.session_state[ 'who_custom_query_path' ] = (
						'' if default_who_query_choice != 'Custom...' else st.session_state.get( 'who_query_path', '' ))
			
			if st.session_state.get( 'who_format', 'json' ) not in WHO_FORMATS:
				st.session_state[ 'who_format' ] = 'json'
			
			if 'who_timeout' not in st.session_state:
				st.session_state[ 'who_timeout' ] = 20
			
			if st.session_state.get( 'who_clear_request', False ):
				st.session_state[ 'who_mode' ] = 'indicator_registry'
				st.session_state[ 'who_query_path' ] = ''
				st.session_state[ 'who_query_choice' ] = 'Indicator'
				st.session_state[ 'who_custom_query_path' ] = ''
				st.session_state[ 'who_format' ] = 'json'
				st.session_state[ 'who_timeout' ] = 20
				st.session_state[ 'who_results' ] = { }
				st.session_state[ 'who_clear_request' ] = False
			
			who_mode = st.selectbox( 'Mode', options=WHO_MODES, index=WHO_MODES.index( st.session_state.get( 'who_mode', 'indicator_registry' ) ), key='who_mode', help=(
				'indicator_registry = WHO metadata landing content; '
				'athena = configurable WHO GHO query path.') )
			
			who_query_choice = st.selectbox( 'Query Path Preset', options=WHO_QUERY_PRESETS, index=WHO_QUERY_PRESETS.index( st.session_state.get( 'who_query_choice', 'Indicator' ) ), key='who_query_choice', disabled=(
						who_mode != 'athena'), help='Common WHO GHO OData query paths.' )
			
			who_custom_query_path = st.text_area( 'Custom Query Path', value=st.session_state.get( 'who_custom_query_path', '' ), height=100, key='who_custom_query_path', placeholder="WHOSIS_000001?$filter=Dim1 eq 'MLE'", disabled=(
						who_mode != 'athena' or who_query_choice != 'Custom...'), help='Path appended after the WHO GHO API base endpoint.' )
			
			who_format = st.selectbox( 'Format', options=WHO_FORMATS, index=WHO_FORMATS.index( st.session_state.get( 'who_format', 'json' ) ), key='who_format', disabled=(
						who_mode != 'athena') )
			
			who_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'who_timeout', 20 ) ), step=1, key='who_timeout' )
			
			st.caption( 'WHO supports GHO OData paths such as Indicator, Dimension, '
			            'DIMENSION/COUNTRY/DimensionValues, and direct indicator-code '
			            'queries.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				who_submit = st.button( 'Submit', key='who_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='who_clear', on_click=_clear_who_state, width='stretch' )
			
			if who_submit:
				st.session_state[ 'demographic_active_source' ] = 'who_global'
		
		# ---------------------
		# ---- Expander United Nations Data
		# ---------------------
		with st.expander( label='United Nations', icon='🇺🇳', expanded=False ):
			UN_MODES = [ 'datasets', 'sdmx_query' ]
			UN_QUERY_PRESETS = [ 'dataflow', 'datastructure', 'codelist', 'conceptscheme',
					'dataflow/all/all/latest', 'datastructure/all/all/latest',
					'codelist/all/all/latest', 'conceptscheme/all/all/latest', 'Custom...' ]
			
			def _clear_un_state( ) -> None:
				st.session_state[ 'un_clear_request' ] = True
			
			def _validate_un_query_path( value: object ) -> str:
				text = str( value or '' ).strip( ).lstrip( '/' )
				
				if not text:
					raise ValueError( 'Query Path is required for sdmx_query mode.' )
				
				if text.startswith( 'http://' ) or text.startswith( 'https://' ):
					raise ValueError( 'Query Path must be a path segment only, not a full URL.' )
				
				if '..' in text:
					raise ValueError( 'Query Path cannot contain parent-directory markers.' )
				
				if not re.fullmatch( r'[A-Za-z0-9_\-./(),:*?=&%]+', text ):
					raise ValueError( 'Query Path contains unsupported characters for a UNdata REST request.' )
				
				return text
			
			if 'un_results' not in st.session_state:
				st.session_state[ 'un_results' ] = { }
			
			if 'un_clear_request' not in st.session_state:
				st.session_state[ 'un_clear_request' ] = False
			
			if st.session_state.get( 'un_mode', 'datasets' ) not in UN_MODES:
				st.session_state[ 'un_mode' ] = 'datasets'
			
			if 'un_query_path' not in st.session_state:
				st.session_state[ 'un_query_path' ] = ''
			
			if st.session_state.get( 'un_query_path', '' ) in UN_QUERY_PRESETS:
				default_un_query_choice = st.session_state.get( 'un_query_path', 'dataflow' )
			elif str( st.session_state.get( 'un_query_path', '' ) ).strip( ):
				default_un_query_choice = 'Custom...'
			else:
				default_un_query_choice = 'dataflow'
			
			if 'un_query_choice' not in st.session_state:
				st.session_state[ 'un_query_choice' ] = default_un_query_choice
			
			if st.session_state.get( 'un_query_choice', 'dataflow' ) not in UN_QUERY_PRESETS:
				st.session_state[ 'un_query_choice' ] = default_un_query_choice
			
			if 'un_custom_query_path' not in st.session_state:
				st.session_state[ 'un_custom_query_path' ] = (
						'' if default_un_query_choice != 'Custom...' else st.session_state.get( 'un_query_path', '' ))
			
			if 'un_timeout' not in st.session_state:
				st.session_state[ 'un_timeout' ] = 20
			
			if st.session_state.get( 'un_clear_request', False ):
				st.session_state[ 'un_mode' ] = 'datasets'
				st.session_state[ 'un_query_path' ] = ''
				st.session_state[ 'un_query_choice' ] = 'dataflow'
				st.session_state[ 'un_custom_query_path' ] = ''
				st.session_state[ 'un_timeout' ] = 20
				st.session_state[ 'un_results' ] = { }
				st.session_state[ 'un_clear_request' ] = False
			
			un_mode = st.selectbox( 'Mode', options=UN_MODES, index=UN_MODES.index( st.session_state.get( 'un_mode', 'datasets' ) ), key='un_mode', help=(
				'datasets = UNdata dataset catalog landing content; '
				'sdmx_query = direct REST SDMX query path.') )
			
			un_query_choice = st.selectbox( 'Query Path Preset', options=UN_QUERY_PRESETS, index=UN_QUERY_PRESETS.index( st.session_state.get( 'un_query_choice', 'dataflow' ) ), key='un_query_choice', disabled=(
						un_mode != 'sdmx_query'), help='Common UNdata SDMX REST artifact paths.' )
			
			un_custom_query_path = st.text_area( 'Custom Query Path', value=st.session_state.get( 'un_custom_query_path', '' ), height=120, key='un_custom_query_path', placeholder='data/DF_SDG_GLH/..SI_POV_DAY1...........?', disabled=(
						un_mode != 'sdmx_query' or un_query_choice != 'Custom...'), help='Path appended after https://data.un.org/WS/rest/' )
			
			un_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'un_timeout', 20 ) ), step=1, key='un_timeout' )
			
			st.caption( 'UNdata exposes SDMX REST artifacts such as dataflow, datastructure, '
			            'codelist, and conceptscheme. Use Custom for dataset-specific paths.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				un_submit = st.button( 'Submit', key='un_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='un_clear', on_click=_clear_un_state, width='stretch' )
			
			if un_submit:
				st.session_state[ 'demographic_active_source' ] = 'united_nations'
		
		# ---------------------
		# ---- Expander World Population
		# ---------------------
		with st.expander( label='World Population', icon='👥', expanded=False ):
			WORLDPOP_MODES = [ 'catalog', 'search', 'raster_metadata' ]
			WORLDPOP_ASSET_PRESETS = [ 'data/pop', 'data/pop/wpgp', 'data/pop/wpgp?iso3=GHA',
					'data/pop/wpgp?iso3=AUS', 'data/pop/wpgp?iso3=USA', 'data/pop/wpgp?iso3=GBR',
					'data/pop/wpgp?iso3=NGA', 'data/pop/wpgp?iso3=KEN', 'data/pop/wpgp?iso3=IND',
					'data/pop/wpgp?iso3=BRA', 'Custom...' ]
			
			def _clear_worldpop_state( ) -> None:
				st.session_state[ 'worldpop_clear_request' ] = True
			
			def _validate_worldpop_query( value: object ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					raise ValueError( 'Query is required for World Population search mode.' )
				
				return text
			
			def _validate_worldpop_asset_path( value: object ) -> str:
				text = str( value or '' ).strip( ).lstrip( '/' )
				
				if not text:
					raise ValueError( 'Asset Path is required for raster_metadata mode.' )
				
				if text.startswith( 'http://' ) or text.startswith( 'https://' ):
					raise ValueError( 'Asset Path must be a path segment only, not a full URL.' )
				
				if '..' in text:
					raise ValueError( 'Asset Path cannot contain parent-directory markers.' )
				
				if not re.fullmatch( r'[A-Za-z0-9_\-./?=&%]+', text ):
					raise ValueError( 'Asset Path contains unsupported characters for a WorldPop request.' )
				
				return text
			
			if 'worldpop_results' not in st.session_state:
				st.session_state[ 'worldpop_results' ] = { }
			
			if 'worldpop_clear_request' not in st.session_state:
				st.session_state[ 'worldpop_clear_request' ] = False
			
			if st.session_state.get( 'worldpop_mode', 'catalog' ) not in WORLDPOP_MODES:
				st.session_state[ 'worldpop_mode' ] = 'catalog'
			
			if 'worldpop_query' not in st.session_state:
				st.session_state[ 'worldpop_query' ] = ''
			
			if 'worldpop_asset_path' not in st.session_state:
				st.session_state[ 'worldpop_asset_path' ] = ''
			
			if st.session_state.get( 'worldpop_asset_path', '' ) in WORLDPOP_ASSET_PRESETS:
				default_asset_choice = st.session_state.get( 'worldpop_asset_path', 'data/pop/wpgp?iso3=GHA' )
			elif str( st.session_state.get( 'worldpop_asset_path', '' ) ).strip( ):
				default_asset_choice = 'Custom...'
			else:
				default_asset_choice = 'data/pop/wpgp?iso3=GHA'
			
			if 'worldpop_asset_choice' not in st.session_state:
				st.session_state[ 'worldpop_asset_choice' ] = default_asset_choice
			
			if st.session_state.get( 'worldpop_asset_choice', 'data/pop/wpgp?iso3=GHA' ) not in WORLDPOP_ASSET_PRESETS:
				st.session_state[ 'worldpop_asset_choice' ] = default_asset_choice
			
			if 'worldpop_custom_asset_path' not in st.session_state:
				st.session_state[ 'worldpop_custom_asset_path' ] = (
						'' if default_asset_choice != 'Custom...' else st.session_state.get( 'worldpop_asset_path', '' ))
			
			if 'worldpop_page' not in st.session_state:
				st.session_state[ 'worldpop_page' ] = 1
			
			if 'worldpop_page_size' not in st.session_state:
				st.session_state[ 'worldpop_page_size' ] = 25
			
			if 'worldpop_timeout' not in st.session_state:
				st.session_state[ 'worldpop_timeout' ] = 20
			
			if st.session_state.get( 'worldpop_clear_request', False ):
				st.session_state[ 'worldpop_mode' ] = 'catalog'
				st.session_state[ 'worldpop_query' ] = ''
				st.session_state[ 'worldpop_asset_path' ] = ''
				st.session_state[ 'worldpop_asset_choice' ] = 'data/pop/wpgp?iso3=GHA'
				st.session_state[ 'worldpop_custom_asset_path' ] = ''
				st.session_state[ 'worldpop_page' ] = 1
				st.session_state[ 'worldpop_page_size' ] = 25
				st.session_state[ 'worldpop_timeout' ] = 20
				st.session_state[ 'worldpop_results' ] = { }
				st.session_state[ 'worldpop_clear_request' ] = False
			
			worldpop_mode = st.selectbox( 'Mode', options=WORLDPOP_MODES, index=WORLDPOP_MODES.index( st.session_state.get( 'worldpop_mode', 'catalog' ) ), key='worldpop_mode', help=(
				'catalog = API landing content; '
				'search = catalog-style search; '
				'raster_metadata = direct asset or metadata path.') )
			
			worldpop_query = st.text_area( 'Query', value=st.session_state.get( 'worldpop_query', '' ), height=90, key='worldpop_query', placeholder='population Ghana 2020', disabled=(
						worldpop_mode != 'search') )
			
			worldpop_asset_choice = st.selectbox( 'Asset Path Preset', options=WORLDPOP_ASSET_PRESETS, index=WORLDPOP_ASSET_PRESETS.index( st.session_state.get( 'worldpop_asset_choice', 'data/pop/wpgp?iso3=GHA' ) ), key='worldpop_asset_choice', disabled=(
						worldpop_mode != 'raster_metadata'), help='Common WorldPop API metadata paths. Use Custom for another path.' )
			
			worldpop_custom_asset_path = st.text_area( 'Custom Asset Path', value=st.session_state.get( 'worldpop_custom_asset_path', '' ), height=100, key='worldpop_custom_asset_path', placeholder='data/pop/wpgp?iso3=GHA', disabled=(
					worldpop_mode != 'raster_metadata' or worldpop_asset_choice != 'Custom...') )
			
			c1, c2, c3 = st.columns( 3 )
			with c1:
				worldpop_page = st.number_input( 'Page', min_value=1, max_value=100000, value=int( st.session_state.get( 'worldpop_page', 1 ) ), step=1, key='worldpop_page', disabled=(
							worldpop_mode != 'search') )
			
			with c2:
				worldpop_page_size = st.number_input( 'Page Size', min_value=1, max_value=500, value=int( st.session_state.get( 'worldpop_page_size', 25 ) ), step=1, key='worldpop_page_size', disabled=(
							worldpop_mode != 'search') )
			
			with c3:
				worldpop_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'worldpop_timeout', 20 ) ), step=1, key='worldpop_timeout' )
			
			st.caption( 'WorldPop exposes API access to population and demographic datasets. '
			            'Raster metadata mode appends a selected path to the current wrapper '
			            'base URL.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				worldpop_submit = st.button( 'Submit', key='worldpop_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='worldpop_clear', on_click=_clear_worldpop_state, width='stretch' )
			
			if worldpop_submit:
				st.session_state[ 'demographic_active_source' ] = 'world_population'
		
		# ---------------------
		# ---- Expander CDC WONDER
		# ---------------------
		with st.expander( label='CDC Wonder', icon='🧬', expanded=False ):
			WONDER_MODES = [ 'metadata_template', 'query_xml' ]
			
			WONDER_DATASETS = [ 'D76', 'D140', 'D176', 'D158', 'D159', 'D160', 'D161', 'D162',
					'D163', 'D164', 'D165', 'D166', 'D167', 'D168', 'D169', 'D170', 'D171', 'D172',
					'D173', 'D174', 'D175', 'Other' ]
			
			def _clear_wonder_state( ) -> None:
				st.session_state[ 'wonder_clear_request' ] = True
			
			def _validate_wonder_dataset_id( value: object ) -> str:
				text = str( value or '' ).strip( ).upper( )
				
				if not text:
					raise ValueError( 'CDC WONDER Dataset ID is required.' )
				
				if not re.fullmatch( r'D\d{1,4}', text ):
					raise ValueError( 'CDC WONDER Dataset ID must use the format D followed by digits, '
					                  'such as D76.' )
				
				return text
			
			def _validate_wonder_xml( value: object ) -> str:
				text = str( value or '' ).strip( )
				
				if not text:
					raise ValueError( 'Request XML is required for query_xml mode.' )
				
				if '<request-parameters>' not in text and '<query-parameters>' not in text:
					raise ValueError( 'Request XML should contain CDC WONDER request/query parameters.' )
				
				return text
			
			if 'wonder_results' not in st.session_state:
				st.session_state[ 'wonder_results' ] = { }
			
			if 'wonder_clear_request' not in st.session_state:
				st.session_state[ 'wonder_clear_request' ] = False
			
			if st.session_state.get( 'wonder_mode', 'metadata_template' ) not in WONDER_MODES:
				st.session_state[ 'wonder_mode' ] = 'metadata_template'
			
			if 'wonder_dataset_id' not in st.session_state:
				st.session_state[ 'wonder_dataset_id' ] = 'D76'
			
			if st.session_state.get( 'wonder_dataset_id', 'D76' ) in WONDER_DATASETS:
				default_dataset_choice = st.session_state.get( 'wonder_dataset_id', 'D76' )
			else:
				default_dataset_choice = 'Other'
			
			if 'wonder_dataset_choice' not in st.session_state:
				st.session_state[ 'wonder_dataset_choice' ] = default_dataset_choice
			
			if st.session_state.get( 'wonder_dataset_choice', 'D76' ) not in WONDER_DATASETS:
				st.session_state[ 'wonder_dataset_choice' ] = default_dataset_choice
			
			if 'wonder_custom_dataset_id' not in st.session_state:
				st.session_state[ 'wonder_custom_dataset_id' ] = (
						'' if default_dataset_choice != 'Other' else st.session_state.get( 'wonder_dataset_id', '' ))
			
			if 'wonder_request_xml' not in st.session_state:
				st.session_state[ 'wonder_request_xml' ] = ''
			
			if 'wonder_timeout' not in st.session_state:
				st.session_state[ 'wonder_timeout' ] = 20
			
			if st.session_state.get( 'wonder_clear_request', False ):
				st.session_state[ 'wonder_mode' ] = 'metadata_template'
				st.session_state[ 'wonder_dataset_id' ] = 'D76'
				st.session_state[ 'wonder_dataset_choice' ] = 'D76'
				st.session_state[ 'wonder_custom_dataset_id' ] = ''
				st.session_state[ 'wonder_request_xml' ] = ''
				st.session_state[ 'wonder_timeout' ] = 20
				st.session_state[ 'wonder_results' ] = { }
				st.session_state[ 'wonder_clear_request' ] = False
			
			wonder_mode = st.selectbox( 'Mode', options=WONDER_MODES, index=WONDER_MODES.index( st.session_state.get( 'wonder_mode', 'metadata_template' ) ), key='wonder_mode', help=(
				'metadata_template = build a starter XML request; '
				'query_xml = submit a raw XML request to CDC WONDER.') )
			
			wonder_dataset_choice = st.selectbox( 'Dataset ID', options=WONDER_DATASETS, index=WONDER_DATASETS.index( st.session_state.get( 'wonder_dataset_choice', 'D76' ) ), key='wonder_dataset_choice', help='Common CDC WONDER database identifiers. Use Other for newer IDs.' )
			
			wonder_custom_dataset_id = st.text_input( 'Custom Dataset ID', value=st.session_state.get( 'wonder_custom_dataset_id', '' ), key='wonder_custom_dataset_id', placeholder='D76', disabled=(
						wonder_dataset_choice != 'Other') )
			
			wonder_request_xml = st.text_area( 'Request XML', value=st.session_state.get( 'wonder_request_xml', '' ), height=240, key='wonder_request_xml', placeholder='<request-parameters>...</request-parameters>', disabled=(
						wonder_mode != 'query_xml') )
			
			wonder_timeout = st.number_input( 'Timeout', min_value=1, max_value=120, value=int( st.session_state.get( 'wonder_timeout', 20 ) ), step=1, key='wonder_timeout' )
			
			st.caption( 'CDC WONDER requires POST requests with request_xml and acceptance '
			            'of data-use restrictions. The wrapper submits '
			            'accept_datause_restrictions=true.' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				wonder_submit = st.button( 'Submit', key='wonder_submit', width='stretch' )
			
			with b2:
				st.button( 'Clear', key='wonder_clear', on_click=_clear_wonder_state, width='stretch' )
			
			if wonder_submit:
				st.session_state[ 'demographic_active_source' ] = 'cdc_wonder'
		
		# ---------------------
		# ---- Expander Pub Med
		# ---------------------
		with st.expander( label='Pub Med Search', icon='🏥', expanded=False ):
			def _clear_pubmed_state( ) -> None:
				st.session_state[ 'pubmed_clear_request' ] = True
			
			if 'pubmed_results' not in st.session_state:
				st.session_state[ 'pubmed_results' ] = { }
			
			if 'pubmed_clear_request' not in st.session_state:
				st.session_state[ 'pubmed_clear_request' ] = False
			
			if 'pubmed_query' not in st.session_state:
				st.session_state[ 'pubmed_query' ] = ''
			
			if 'pubmed_max_docs' not in st.session_state:
				st.session_state[ 'pubmed_max_docs' ] = 5
			
			if st.session_state.get( 'pubmed_clear_request', False ):
				st.session_state[ 'pubmed_results' ] = { }
				st.session_state[ 'pubmed_query' ] = ''
				st.session_state[ 'pubmed_max_docs' ] = 5
				st.session_state[ 'pubmed_clear_request' ] = False
			
			pubmed_query = st.text_input( 'PubMed Query', value=st.session_state.get( 'pubmed_query', '' ), key='pubmed_query', placeholder='Example: machine learning cancer diagnosis' )
			
			pubmed_max_docs = st.number_input( 'Max Documents', min_value=1, max_value=100, value=int( st.session_state.get( 'pubmed_max_docs', 5 ) ), step=1, key='pubmed_max_docs' )
			
			st.caption( 'PubMed search uses the LangChain PubMedSearchLoader and promotes '
			            'returned documents into the shared loader state for downstream use.' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				pubmed_submit = st.button( 'Submit', key='pubmed_submit', use_container_width=True )
			
			with b2:
				pubmed_clear = st.button( 'Clear', key='pubmed_clear', on_click=_clear_pubmed_state, use_container_width=True )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'PubMedSearchLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='pubmed_loader_output.txt', mime='text/plain', key='pubmed_save', width='stretch' )
				else:
					st.button( 'Save', key='pubmed_save_disabled', disabled=True, width='stretch' )
			
			if pubmed_submit:
				st.session_state[ 'demographic_active_source' ] = 'pub_med_search'
		
		# ---------------------
		# ---- Expander Open City
		# ---------------------
		with st.expander( label='Open City Data', icon='🏙️', expanded=False ):
			OPEN_CITY_DOMAINS = [ 'data.sfgov.org', 'data.cityofnewyork.us',
					'data.cityofchicago.org', 'data.lacity.org', 'data.seattle.gov',
					'data.austintexas.gov', 'data.cincinnati-oh.gov', 'data.baltimorecity.gov',
					'data.cityofboston.gov', 'data.nashville.gov', 'Other' ]
			
			def _clear_open_city_state( ) -> None:
				st.session_state[ 'open_city_clear_request' ] = True
			
			def _validate_open_city_domain( value: object ) -> str:
				text = str( value or '' ).strip( ).lower( )
				text = text.replace( 'https://', '' ).replace( 'http://', '' )
				text = text.strip( '/' )
				
				if not text:
					raise ValueError( 'City ID is required.' )
				
				if '/' in text:
					raise ValueError( 'City ID must be a domain only, not a URL path.' )
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]+\.[a-z]{2,}', text ):
					raise ValueError( 'City ID must be a valid Socrata portal domain, such as '
					                  'data.sfgov.org.' )
				
				return text
			
			def _validate_open_city_dataset_id( value: object ) -> str:
				text = str( value or '' ).strip( ).replace( '.json', '' ).strip( '/' )
				
				if not text:
					raise ValueError( 'Dataset ID is required.' )
				
				if not re.fullmatch( r'[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}', text ):
					raise ValueError( 'Dataset ID must use the Socrata four-by-four format, '
					                  'such as vw6y-z8j6.' )
				
				return text.lower( )
			
			if 'open_city_results' not in st.session_state:
				st.session_state[ 'open_city_results' ] = { }
			
			if 'open_city_clear_request' not in st.session_state:
				st.session_state[ 'open_city_clear_request' ] = False
			
			if 'open_city_id' not in st.session_state:
				st.session_state[ 'open_city_id' ] = 'data.sfgov.org'
			
			if st.session_state.get( 'open_city_id', 'data.sfgov.org' ) in OPEN_CITY_DOMAINS:
				default_open_city_choice = st.session_state.get( 'open_city_id', 'data.sfgov.org' )
			else:
				default_open_city_choice = 'Other'
			
			if 'open_city_choice' not in st.session_state:
				st.session_state[ 'open_city_choice' ] = default_open_city_choice
			
			if st.session_state.get( 'open_city_choice', 'data.sfgov.org' ) not in OPEN_CITY_DOMAINS:
				st.session_state[ 'open_city_choice' ] = default_open_city_choice
			
			if 'open_city_custom_id' not in st.session_state:
				st.session_state[ 'open_city_custom_id' ] = (
						'' if default_open_city_choice != 'Other' else st.session_state.get( 'open_city_id', '' ))
			
			if 'open_city_dataset_id' not in st.session_state:
				st.session_state[ 'open_city_dataset_id' ] = ''
			
			if 'open_city_limit' not in st.session_state:
				st.session_state[ 'open_city_limit' ] = 100
			
			if st.session_state.get( 'open_city_clear_request', False ):
				st.session_state[ 'open_city_results' ] = { }
				st.session_state[ 'open_city_id' ] = 'data.sfgov.org'
				st.session_state[ 'open_city_choice' ] = 'data.sfgov.org'
				st.session_state[ 'open_city_custom_id' ] = ''
				st.session_state[ 'open_city_dataset_id' ] = ''
				st.session_state[ 'open_city_limit' ] = 100
				st.session_state[ 'open_city_clear_request' ] = False
			
			open_city_choice = st.selectbox( 'City ID', options=OPEN_CITY_DOMAINS, index=OPEN_CITY_DOMAINS.index( st.session_state.get( 'open_city_choice', 'data.sfgov.org' ) ), key='open_city_choice', help='Common Socrata open-data city domains. Use Other for a custom portal.' )
			
			open_city_custom_id = st.text_input( 'Custom City ID', value=st.session_state.get( 'open_city_custom_id', '' ), key='open_city_custom_id', disabled=(
						open_city_choice != 'Other'), placeholder='data.example.gov' )
			
			dataset_id = st.text_input( 'Dataset ID', value=st.session_state.get( 'open_city_dataset_id', '' ), key='open_city_dataset_id', placeholder='vw6y-z8j6' )
			
			limit = st.number_input( 'Limit', min_value=1, max_value=5000, value=int( st.session_state.get( 'open_city_limit', 100 ) ), step=10, key='open_city_limit' )
			
			st.caption( 'Open City Data uses LangChain OpenCityDataLoader backed by Socrata. '
			            'Use the API tab on the city dataset page to find the dataset ID.' )
			
			b1, b2, b3 = st.columns( 3 )
			with b1:
				open_city_submit = st.button( 'Submit', key='open_city_submit', width='stretch' )
			
			with b2:
				open_city_clear = st.button( 'Clear', key='open_city_clear', on_click=_clear_open_city_state, width='stretch' )
			
			with b3:
				can_save = (
							st.session_state.get( 'active_loader' ) == 'OpenCityLoader' and isinstance( st.session_state.get( 'raw_text' ), str ) and st.session_state.get( 'raw_text' ).strip( ))
				
				if can_save:
					st.download_button( 'Save', data=st.session_state.get( 'raw_text' ), file_name='open_city_loader_output.txt', mime='text/plain', key='open_city_save', width='stretch' )
				else:
					st.button( 'Save', key='open_city_save_disabled', disabled=True, width='stretch' )
			
			if open_city_submit:
				st.session_state[ 'demographic_active_source' ] = 'open_city_data'
	
	with right:
		st.markdown( '##### Results' )
		active_source = st.session_state.get( 'demographic_active_source', '' )
		display_names: Dict[ str, str ] = { 'u_s_census_bureau': 'U.S. Census Bureau',
				'cdc_socrata': 'CDC Socrata', 'u_s_health': 'U.S. Health',
				'who_global': 'WHO Global', 'united_nations': 'United Nations',
				'world_population': 'World Population', 'cdc_wonder': 'CDC Wonder',
				'pub_med_search': 'Pub Med Search', 'open_city_data': 'Open City Data', }
		if not active_source:
			st.info( 'Select a source, configure the request, and submit it to display results.' )
		else:
			st.caption( f"Active Source: {display_names.get( active_source, active_source )}" )
		
		# -------- U.S. Census Bureau
		if active_source == 'u_s_census_bureau':
			st.markdown( '##### U.S. Census Bureau' )
			result = st.session_state.get( 'census_results', { } )
			if census_submit:
				try:
					clean_year = _validate_census_year( census_year )
					clean_dataset = _validate_census_dataset( census_dataset )
					if census_mode == 'data':
						clean_fields = _validate_census_fields( census_fields )
						clean_for = _validate_census_geography_clause( name='For', value=census_for, required=True )
						clean_in = _validate_census_geography_clause( name='In', value=census_in, required=False )
					else:
						clean_fields = str( census_fields or '' ).strip( )
						clean_for = str( census_for or '' ).strip( )
						clean_in = str( census_in or '' ).strip( )
					
					f = CensusData( )
					result = f.fetch( mode=str( census_mode ), year=clean_year, dataset=clean_dataset, fields=clean_fields, geography_for=clean_for, geography_in=clean_in, predicates=str( census_predicates or '' ).strip( ), time=int( census_timeout ) )
					
					st.session_state[ 'census_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Census request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'variables':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					variables = payload.get( 'variables', { } ) if isinstance( payload, dict ) else { }
					
					rows: List[ Dict[ str, Any ] ] = [ ]
					if isinstance( variables, dict ):
						for name, meta in variables.items( ):
							if isinstance( meta, dict ):
								rows.append( { 'Name': name, 'Label': meta.get( 'label', '' ),
										'Concept': meta.get( 'concept', '' ),
										'PredicateType': meta.get( 'predicateType', '' ),
										'Group': meta.get( 'group', '' ),
										'Limit': meta.get( 'limit', '' ), } )
					
					_render_summary_kv( '#### Summary', { 'Year': census_year,
							'Dataset': census_dataset, 'VariableCount': len( rows ), } )
					_render_rows_table( '#### Variables', rows )
				
				elif result.get( 'mode', '' ) == 'data':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					rows = payload.get( 'rows', [ ] ) if isinstance( payload, dict ) else [ ]
					_render_summary_kv( '#### Summary', { 'Year': census_year,
							'Dataset': census_dataset, 'Fields': census_fields, 'For': census_for,
							'In': census_in,
							'RowCount': len( rows ) if isinstance( rows, list ) else 0, } )
					_render_rows_table( '#### Data Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
		
		# -------- CDC SOCRATA
		elif active_source == 'cdc_socrata':
			st.markdown( '##### CDC Socrata' )
			result = st.session_state.get( 'socrata_results', { } )
			if socrata_submit:
				try:
					clean_dataset_id = _validate_socrata_dataset_id( socrata_dataset_id )
					f = Socrata( )
					result = f.fetch( mode=str( socrata_mode ), domain=str( socrata_domain ), dataset_id=clean_dataset_id, select=str( socrata_select ), where=str( socrata_where ), order=str( socrata_order ), group=str( socrata_group ), limit=int( socrata_limit ), offset=int( socrata_offset ), time=int( socrata_timeout ) )
					st.session_state[ 'socrata_results' ] = result or { }
					st.rerun( )
				except Exception as exc:
					st.error( 'CDC Socrata request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				if result.get( 'mode', '' ) == 'metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					_render_summary_kv( '#### Summary', {
							'Name': payload.get( 'name', '' ) if isinstance( payload, dict ) else '',
							'Description': payload.get( 'description', '' ) if isinstance( payload, dict ) else '',
							'RowsUpdatedAt': payload.get( 'rowsUpdatedAt', '' ) if isinstance( payload, dict ) else '',
							'ViewType': payload.get( 'viewType', '' ) if isinstance( payload, dict ) else '',
							'Columns': len( payload.get( 'columns', [ ] ) ) if isinstance( payload, dict ) else 0, } )
					rows: List[ Dict[ str, Any ] ] = [ ]
					columns_payload = payload.get( 'columns', [ ] ) if isinstance( payload, dict ) else [ ]
					for item in columns_payload:
						if isinstance( item, dict ):
							rows.append( { 'Name': item.get( 'name', '' ),
									'FieldName': item.get( 'fieldName', '' ),
									'DataType': item.get( 'dataTypeName', '' ),
									'Description': item.get( 'description', '' ), } )
					
					_render_rows_table( '#### Columns', rows )
				
				elif result.get( 'mode', '' ) == 'rows':
					rows = result.get( 'data', [ ] ) if isinstance( result, dict ) else [ ]
					
					_render_summary_kv( '#### Summary', { 'Domain': socrata_domain,
							'DatasetId': socrata_dataset_id, 'Limit': int( socrata_limit ),
							'Offset': int( socrata_offset ),
							'RowCount': len( rows ) if isinstance( rows, list ) else 0, } )
					_render_rows_table( '#### Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
		
		# -------- US Health Data
		elif active_source == 'u_s_health':
			st.markdown( '##### U.S. Health' )
			result = st.session_state.get( 'healthdata_results', { } )
			
			if healthdata_submit:
				try:
					clean_dataset_id = _validate_healthdata_dataset_id( healthdata_dataset_id )
					
					f = HealthData( )
					result = f.fetch( mode=str( healthdata_mode ), domain=str( healthdata_domain ), dataset_id=clean_dataset_id, select=str( healthdata_select ), where=str( healthdata_where ), order=str( healthdata_order ), group=str( healthdata_group ), limit=int( healthdata_limit ), offset=int( healthdata_offset ), time=int( healthdata_timeout ) )
					
					st.session_state[ 'healthdata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'HealthData request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', {
							'Name': payload.get( 'name', '' ) if isinstance( payload, dict ) else '',
							'Description': payload.get( 'description', '' ) if isinstance( payload, dict ) else '',
							'RowsUpdatedAt': payload.get( 'rowsUpdatedAt', '' ) if isinstance( payload, dict ) else '',
							'ViewType': payload.get( 'viewType', '' ) if isinstance( payload, dict ) else '',
							'Columns': len( payload.get( 'columns', [ ] ) ) if isinstance( payload, dict ) else 0, } )
					
					rows: List[ Dict[ str, Any ] ] = [ ]
					columns_payload = payload.get( 'columns', [ ] ) if isinstance( payload, dict ) else [ ]
					for item in columns_payload:
						if isinstance( item, dict ):
							rows.append( { 'Name': item.get( 'name', '' ),
									'FieldName': item.get( 'fieldName', '' ),
									'DataType': item.get( 'dataTypeName', '' ),
									'Description': item.get( 'description', '' ), } )
					
					_render_rows_table( '#### Columns', rows )
				
				elif result.get( 'mode', '' ) == 'rows':
					rows = result.get( 'data', [ ] ) if isinstance( result, dict ) else [ ]
					
					_render_summary_kv( '#### Summary', { 'Domain': healthdata_domain,
							'DatasetId': healthdata_dataset_id, 'Limit': int( healthdata_limit ),
							'Offset': int( healthdata_offset ),
							'RowCount': len( rows ) if isinstance( rows, list ) else 0, } )
					_render_rows_table( '#### Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
		
		# -------- WHO Global Health
		elif active_source == 'who_global':
			st.markdown( '##### WHO Global' )
			result = st.session_state.get( 'who_results', { } )
			
			if who_submit:
				try:
					if who_mode == 'athena':
						selected_query_path = (
								who_custom_query_path if who_query_choice == 'Custom...' else who_query_choice)
						clean_query_path = _validate_who_query_path( selected_query_path )
					else:
						clean_query_path = ''
					
					f = GlobalHealthData( )
					result = f.fetch( mode=str( who_mode ), query_path=clean_query_path, fmt=str( who_format ), time=int( who_timeout ) )
					
					st.session_state[ 'who_query_path' ] = clean_query_path
					st.session_state[ 'who_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'WHO Global Health request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'indicator_registry':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
							'HasHtml': (
										isinstance( payload, dict ) and bool( payload.get( 'html', '' ) )), } )
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview( '#### Indicator Registry Preview', str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'athena':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ) and isinstance( payload.get( 'value', [ ] ), list ):
						rows = payload.get( 'value', [ ] )
						_render_summary_kv( '#### Summary', {
								'QueryPath': st.session_state.get( 'who_query_path', '' ),
								'Format': who_format, 'ResultCount': len( rows ), } )
						_render_rows_table( '#### Athena Results', rows )
					elif isinstance( payload, dict ) and payload.get( 'text', '' ):
						_render_summary_kv( '#### Summary', {
								'QueryPath': st.session_state.get( 'who_query_path', '' ),
								'Format': who_format, 'HasText': True, } )
						st.markdown( '#### Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
		
		# -------- United Nations Data
		elif active_source == 'united_nations':
			st.markdown( '##### United Nations' )
			result = st.session_state.get( 'un_results', { } )
			
			if un_submit:
				try:
					if un_mode == 'sdmx_query':
						selected_query_path = (
								un_custom_query_path if un_query_choice == 'Custom...' else un_query_choice)
						clean_query_path = _validate_un_query_path( selected_query_path )
					else:
						clean_query_path = ''
					
					f = UnitedNations( )
					result = f.fetch( mode=str( un_mode ), query_path=clean_query_path, time=int( un_timeout ) )
					
					st.session_state[ 'un_query_path' ] = clean_query_path
					st.session_state[ 'un_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'United Nations request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'datasets':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
							'HasHtml': (
										isinstance( payload, dict ) and bool( payload.get( 'html', '' ) )), } )
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview( '#### Dataset Catalog Preview', str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'sdmx_query':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
							'QueryPath': st.session_state.get( 'un_query_path', '' ),
							'TextPayload': (
										isinstance( payload, dict ) and bool( payload.get( 'text', '' ) )),
							'JsonPayload': isinstance( payload, (dict, list) ), } )
					
					if isinstance( payload, dict ) and payload.get( 'text', '' ):
						st.markdown( '#### Query Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					elif isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview( '#### Query Response', str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
		
		# -------- World Population
		elif active_source == 'world_population':
			st.markdown( '##### World Population' )
			result = st.session_state.get( 'worldpop_results', { } )
			
			if worldpop_submit:
				try:
					if worldpop_mode == 'search':
						clean_query = _validate_worldpop_query( worldpop_query )
						clean_asset_path = ''
					elif worldpop_mode == 'raster_metadata':
						selected_asset_path = (
								worldpop_custom_asset_path if worldpop_asset_choice == 'Custom...' else worldpop_asset_choice)
						clean_asset_path = _validate_worldpop_asset_path( selected_asset_path )
						clean_query = ''
					else:
						clean_query = ''
						clean_asset_path = ''
					
					f = WorldPopulation( )
					result = f.fetch( mode=str( worldpop_mode ), query=clean_query, asset_path=clean_asset_path, page=int( worldpop_page ), page_size=int( worldpop_page_size ), time=int( worldpop_timeout ) )
					
					st.session_state[ 'worldpop_query' ] = clean_query
					st.session_state[ 'worldpop_asset_path' ] = clean_asset_path
					st.session_state[ 'worldpop_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'World Population request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'catalog':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
							'HasHtml': (
										isinstance( payload, dict ) and bool( payload.get( 'html', '' ) )), } )
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview( '#### Catalog Preview', str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'search':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ) and isinstance( payload.get( 'results', [ ] ), list ):
						rows = payload.get( 'results', [ ] )
						_render_summary_kv( '#### Summary', {
								'Query': st.session_state.get( 'worldpop_query', '' ),
								'Page': worldpop_page, 'PageSize': worldpop_page_size,
								'ResultCount': len( rows ), } )
						_render_rows_table( '#### Search Results', rows )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'raster_metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv( '#### Summary', {
							'AssetPath': st.session_state.get( 'worldpop_asset_path', '' ),
							'HasText': (
										isinstance( payload, dict ) and bool( payload.get( 'text', '' ) )), } )
					
					if isinstance( payload, dict ) and payload.get( 'text', '' ):
						st.markdown( '#### Metadata Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
		
		# -------- CDC WONDER
		elif active_source == 'cdc_wonder':
			st.markdown( '##### CDC Wonder' )
			result = st.session_state.get( 'wonder_results', { } )
			
			if wonder_submit:
				try:
					selected_dataset_id = (
							wonder_custom_dataset_id if wonder_dataset_choice == 'Other' else wonder_dataset_choice)
					clean_dataset_id = _validate_wonder_dataset_id( selected_dataset_id )
					
					if wonder_mode == 'query_xml':
						clean_request_xml = _validate_wonder_xml( wonder_request_xml )
					else:
						clean_request_xml = str( wonder_request_xml or '' ).strip( )
					
					f = Wonder( )
					result = f.fetch( mode=str( wonder_mode ), dataset_id=clean_dataset_id, request_xml=clean_request_xml, time=int( wonder_timeout ) )
					
					st.session_state[ 'wonder_dataset_id' ] = clean_dataset_id
					st.session_state[ 'wonder_results' ] = result or { }
					
					if (
							wonder_mode == 'metadata_template' and isinstance( result, dict ) and isinstance( result.get( 'data', { } ), dict )):
						template_xml = result.get( 'data', { } ).get( 'request_xml', '' )
						st.session_state[ 'wonder_request_xml' ] = template_xml
					
					st.rerun( )
				
				except Exception as exc:
					st.error( 'CDC WONDER request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'metadata_template':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ):
						_render_summary_kv( '#### Template Summary', {
								'DatasetId': payload.get( 'dataset_id', '' ),
								'Notes': payload.get( 'notes', '' ), } )
						
						template_xml = str( payload.get( 'request_xml', '' ) )
						if template_xml:
							_render_xml_preview( '#### Starter XML', template_xml )
						else:
							st.info( 'No starter XML returned.' )
				
				elif result.get( 'mode', '' ) == 'query_xml':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ):
						xml_text = str( payload.get( 'xml', '' ) )
						
						_render_summary_kv( '#### Response Summary', {
								'DatasetId': st.session_state.get( 'wonder_dataset_id', '' ),
								'Characters': len( xml_text ),
								'HasXml': bool( xml_text.strip( ) ), } )
						
						_render_xml_preview( '#### XML Response', xml_text )
					else:
						st.info( 'No XML response returned.' )
				
				_render_fallback_raw( result )
		
		# -------- Pub Med
		elif active_source == 'pub_med_search':
			st.markdown( '##### Pub Med Search' )
			if pubmed_clear:
				remaining = _clear_loader_documents( 'PubMedSearchLoader' )
				st.info( f'PubMed Loader state cleared. Remaining documents: {remaining}.' )
			
			if pubmed_submit:
				if not pubmed_query or not pubmed_query.strip( ):
					st.info( 'Enter a PubMed query.' )
				else:
					try:
						loader = PubMedSearchLoader( )
						documents = loader.load( query=pubmed_query.strip( ), max_docs=int( pubmed_max_docs ) ) or [ ]
						
						count = _promote_loader_documents( documents, 'PubMedSearchLoader' )
						
						items: list[ dict[ str, Any ] ] = [ ]
						
						for i, doc in enumerate( documents, start=1 ):
							metadata = (
									doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
							content = str( getattr( doc, 'page_content', '' ) or '' )
							
							items.append( { 'Index': i, 'Title': (
									metadata.get( 'Title' ) or metadata.get( 'title' ) or ''),
									'Published': (
												metadata.get( 'Published' ) or metadata.get( 'published' ) or ''),
									'Copyright': (
											metadata.get( 'Copyright Information' ) or metadata.get( 'copyright' ) or ''),
									'Summary': content, 'Metadata': metadata, } )
						
						st.session_state[ 'pubmed_results' ] = { 'mode': 'pubmed',
								'query': pubmed_query.strip( ), 'max_docs': int( pubmed_max_docs ),
								'count': count, 'items': items, }
						
						st.success( f'Loaded {count} PubMed document(s).' )
					
					except Exception as exc:
						st.error( 'PubMed request failed.' )
						st.exception( exc )
			
			result = st.session_state.get( 'pubmed_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'Query': result.get( 'query', '' ), 'MaxDocs': result.get( 'max_docs', 0 ),
						'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Title': item.get( 'Title', '' ),
									'Published': item.get( 'Published', '' ), } for item in items ]
					df_pubmed = pd.DataFrame( table_rows )
					st.markdown( '#### Results' )
					st.dataframe( df_pubmed, use_container_width=True, hide_index=True )
					first = items[ 0 ]
					_render_summary_kv( '#### First Result', { 'Title': first.get( 'Title', '' ),
							'Published': first.get( 'Published', '' ),
							'Copyright': first.get( 'Copyright', '' ), } )
					
					st.markdown( '#### Abstract Preview' )
					st.code( str( first.get( 'Summary', '' ) )[ :8000 ] )
					
					with st.expander( 'Records', expanded=False ):
						for item in items:
							record_label = str( item.get( 'Title', '' ) or f"Record "
							                                               f"{item.get( 'Index', '' )}" )
							with st.expander( f"Record {item.get( 'Index', '' )}: {record_label}", expanded=False ):
								st.markdown( f"**Published:** {item.get( 'Published', '' )}" )
								st.markdown( f"**Copyright:** {item.get( 'Copyright', '' )}" )
								st.markdown( '##### Summary' )
								st.code( str( item.get( 'Summary', '' ) )[ :8000 ] )
								
								metadata = item.get( 'Metadata', { } )
								if metadata:
									with st.expander( 'Metadata', expanded=False ):
										st.json( metadata )
				else:
					st.info( 'No PubMed records returned.' )
				
				_render_fallback_raw( result )
		
		# -------- Open City
		elif active_source == 'open_city_data':
			st.markdown( '##### Open City Data' )
			if open_city_clear:
				remaining = _clear_loader_documents( 'OpenCityLoader' )
				st.info( f'Open City Data Loader state cleared. Remaining documents: {remaining}.' )
			
			if open_city_submit:
				try:
					selected_city_id = (
							open_city_custom_id if open_city_choice == 'Other' else open_city_choice)
					clean_city_id = _validate_open_city_domain( selected_city_id )
					clean_dataset_id = _validate_open_city_dataset_id( dataset_id )
					
					loader = OpenCityLoader( )
					documents = loader.load( city_id=clean_city_id, dataset_id=clean_dataset_id, limit=int( limit ) ) or [ ]
					
					count = _promote_loader_documents( documents, 'OpenCityLoader' )
					
					items: list[ dict[ str, Any ] ] = [ ]
					for i, doc in enumerate( documents, start=1 ):
						metadata = (
							doc.metadata if isinstance( getattr( doc, 'metadata', { } ), dict ) else { })
						content = str( getattr( doc, 'page_content', '' ) or '' )
						items.append( { 'Index': i, 'Source': metadata.get( 'source', '' ),
								'Row': content, 'Metadata': metadata, } )
					
					st.session_state[ 'open_city_id' ] = clean_city_id
					st.session_state[ 'open_city_results' ] = { 'mode': 'open_city',
							'city_id': clean_city_id, 'dataset_id': clean_dataset_id,
							'limit': int( limit ), 'count': count, 'items': items, }
					
					st.success( f'Loaded {count} Open City document(s).' )
				
				except Exception as exc:
					st.error( 'Open City request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'open_city_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_summary_kv( '#### Summary', { 'Mode': result.get( 'mode', '' ),
						'CityId': result.get( 'city_id', '' ),
						'DatasetId': result.get( 'dataset_id', '' ),
						'Limit': result.get( 'limit', 0 ), 'Returned': result.get( 'count', 0 ), } )
				
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if items:
					table_rows = [
							{ 'Index': item.get( 'Index', '' ), 'Source': item.get( 'Source', '' ),
									'Preview': str( item.get( 'Row', '' ) )[ :200 ], } for item in
							items ]
					
					df_open_city = pd.DataFrame( table_rows )
					
					st.markdown( '#### Results' )
					st.dataframe( df_open_city, use_container_width=True, hide_index=True )
					
					first = items[ 0 ]
					st.markdown( '#### First Row Preview' )
					st.code( str( first.get( 'Row', '' ) )[ :8000 ] )
					
					with st.expander( 'Records', expanded=False ):
						for item in items:
							with st.expander( f"Record {item.get( 'Index', '' )}", expanded=False ):
								st.markdown( f"**Source:** {item.get( 'Source', '' )}" )
								st.markdown( '##### Row' )
								st.code( str( item.get( 'Row', '' ) )[ :8000 ] )
								
								metadata = item.get( 'Metadata', { } )
								if metadata:
									with st.expander( 'Metadata', expanded=False ):
										st.json( metadata )
				else:
					st.info( 'No city records returned.' )
				
				_render_fallback_raw( result )

# ==============================================================================
# TEXT GENERATION MODE
# ==============================================================================
elif mode == 'Generation':
	st.subheader( '🧠  Generative AI' )
	st.divider( )
	
	# ---------- Generation Result State
	if 'generation_documents' not in st.session_state:
		st.session_state[ 'generation_documents' ] = [ ]
	
	if 'generation_raw_result' not in st.session_state:
		st.session_state[ 'generation_raw_result' ] = None
	
	if 'generation_active_source' not in st.session_state:
		st.session_state[ 'generation_active_source' ] = ''
	
	def _promote_generation_result( provider: str, result: Any ) -> None:
		"""Promote a provider response into the shared generation document state.

		Purpose:
		    Converts the active provider response into LangChain documents used by the shared
		    right-column viewer while retaining the original response object for provider-specific
		    rendering.

		Args:
		    provider (str): Name of the provider that generated the response.
		    result (Any): Response returned by the provider wrapper.

		Returns:
		    None: This function updates Streamlit session state.
		"""
		if not str( provider or '' ).strip( ):
			raise ValueError( 'Provider cannot be empty.' )
		
		documents: list[ Document ] = [ ]
		
		if isinstance( result, list ) and all( isinstance( item, Document ) for item in result ):
			documents = list( result )
		else:
			if isinstance( result, (dict, list) ):
				page_content = json.dumps( normalize( result ), indent=2, ensure_ascii=False )
			else:
				page_content = str( result or '' )
			
			documents = [ Document( page_content=page_content, metadata={ 'mode': 'Generation',
					'provider': provider, } ) ]
		
		for document in documents:
			if not isinstance( getattr( document, 'metadata', None ), dict ):
				document.metadata = { }
			
			document.metadata.setdefault( 'mode', 'Generation' )
			document.metadata.setdefault( 'provider', provider )
		
		st.session_state[ 'documents' ] = list( documents )
		st.session_state[ 'raw_documents' ] = list( documents )
		st.session_state[ 'raw_text' ] = '\n\n'.join(
			document.page_content for document in documents if
			isinstance( getattr( document, 'page_content', None ), str ) )
		st.session_state[ 'generation_documents' ] = list( documents )
		st.session_state[ 'generation_raw_result' ] = result
		st.session_state[ 'generation_active_source' ] = provider
	
	def _clear_generation_result( provider: str ) -> None:
		"""Clear the shared generation result for the active provider.

		Purpose:
		    Removes the shared generated document only when the provider being cleared currently owns
		    the active result.

		Args:
		    provider (str): Name of the provider whose controls are being cleared.

		Returns:
		    None: This function updates Streamlit session state.
		"""
		if st.session_state.get( 'generation_active_source', '' ) != provider:
			return
		
		st.session_state[ 'documents' ] = [ ]
		st.session_state[ 'raw_documents' ] = [ ]
		st.session_state[ 'raw_text' ] = ''
		st.session_state[ 'generation_documents' ] = [ ]
		st.session_state[ 'generation_raw_result' ] = None
		st.session_state[ 'generation_active_source' ] = ''
	
	left, right = st.columns( [ 0.4, 0.6 ], gap='xxsmall', border=True )
	
	# ------------------------------------------------------------------
	# LEFT COLUMN — GENERATION CONTROLS
	# ------------------------------------------------------------------
	with left:
		# ---------------------
		# ---- Expander GPT
		# ---------------------
		with st.expander( label='ChatGPT', expanded=True ):
			CHAT_REASONING_EFFORTS = [ 'minimal', 'low', 'medium', 'high' ]
			
			def _clear_chat_state( ) -> None:
				st.session_state[ 'chat_clear_request' ] = True
				_clear_generation_result( 'ChatGPT' )
			
			def _normalize_chat_domains( value: object ) -> list[ str ]:
				text = str( value or '' ).strip( )
				
				if not text:
					return [ ]
				
				values: list[ str ] = [ ]
				entries = re.split( r'[\n,;]+', text )
				
				for entry in entries:
					raw_value = str( entry or '' ).strip( ).lower( )
					
					if not raw_value:
						continue
					
					if not raw_value.startswith( 'http://' ) and not raw_value.startswith( 'https://' ):
						raw_value = f'https://{raw_value}'
					
					parsed = urlparse( raw_value )
					domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
					domain = re.sub( r':\d+$', '', domain )
					domain = domain.lstrip( '.' )
					
					if domain.startswith( 'www.' ):
						domain = domain[ 4: ]
					
					if not domain:
						continue
					
					if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
						raise ValueError( f'Invalid search domain: {domain}' )
					
					if domain not in values:
						values.append( domain )
				
				return values
			
			if 'chat_clear_request' not in st.session_state:
				st.session_state[ 'chat_clear_request' ] = False
			
			if 'chat_prompt' not in st.session_state:
				st.session_state[ 'chat_prompt' ] = ''
			
			if 'chat_system' not in st.session_state:
				st.session_state[ 'chat_system' ] = ''
			
			if 'chat_domains' not in st.session_state:
				st.session_state[ 'chat_domains' ] = ''
			
			if 'chat_json_mode' not in st.session_state:
				st.session_state[ 'chat_json_mode' ] = False
			
			if 'chat_reasoning' not in st.session_state:
				st.session_state[ 'chat_reasoning' ] = False
			
			if 'chat_web_search' not in st.session_state:
				st.session_state[ 'chat_web_search' ] = False
			
			if 'chat_store' not in st.session_state:
				st.session_state[ 'chat_store' ] = True
			
			if 'chat_stream' not in st.session_state:
				st.session_state[ 'chat_stream' ] = False
			
			if 'chat_seed' not in st.session_state:
				st.session_state[ 'chat_seed' ] = 0
			
			if st.session_state.get( 'chat_reasoning_effort', 'low' ) not in CHAT_REASONING_EFFORTS:
				st.session_state[ 'chat_reasoning_effort' ] = 'low'
			
			if st.session_state.get( 'chat_clear_request', False ):
				st.session_state[ 'chat_prompt' ] = ''
				st.session_state[ 'chat_system' ] = ''
				st.session_state[ 'chat_domains' ] = ''
				st.session_state[ 'chat_json_mode' ] = False
				st.session_state[ 'chat_reasoning' ] = False
				st.session_state[ 'chat_web_search' ] = False
				st.session_state[ 'chat_store' ] = True
				st.session_state[ 'chat_stream' ] = False
				st.session_state[ 'chat_seed' ] = 0
				st.session_state[ 'chat_reasoning_effort' ] = 'low'
				st.session_state[ 'chat_clear_request' ] = False
			
			chat_prompt = st.text_area( 'Prompt', value=st.session_state.get( 'chat_prompt', '' ), height=120, key='chat_prompt' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_chat_models = (
						cfg.GPT_MODELS if hasattr( cfg, 'GPT_MODELS' ) and cfg.GPT_MODELS else [
								'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1' ])
				
				chat_model = _model_selector( key_prefix='chat', label='Model', options=_chat_models, default_model=(
						'gpt-5-mini' if 'gpt-5-mini' in _chat_models else _chat_models[ 0 ]), )
			
			with p_row1[ 1 ]:
				chat_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.05, key='chat_temperature' )
			
			with p_row2[ 0 ]:
				chat_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768, value=2048, step=1, key='chat_max_tokens' )
			
			with p_row2[ 1 ]:
				chat_top_p = st.slider( 'Top-P', min_value=0.0, max_value=1.0, value=1.0, step=0.01, key='chat_top_p' )
			
			with p_row3[ 0 ]:
				chat_seed = st.number_input( 'Seed', min_value=0, max_value=2_147_483_647, value=int( st.session_state.get( 'chat_seed', 0 ) ), step=1, key='chat_seed', help='Use 0 to omit the seed parameter.' )
			
			with p_row3[ 1 ]:
				chat_json_mode = st.checkbox( 'JSON Mode', value=bool( st.session_state.get( 'chat_json_mode', False ) ), key='chat_json_mode', help=(
					'Current wrapper behavior adds JSON-only instructions. '
					'A later Chat class drop-in should wire this to Responses '
					'API text.format.') )
			
			with p_row4[ 0 ]:
				chat_reasoning = st.checkbox( 'Reasoning', value=bool( st.session_state.get( 'chat_reasoning', False ) ), key='chat_reasoning' )
			
			with p_row4[ 1 ]:
				chat_web_search = st.checkbox( 'Web Search', value=bool( st.session_state.get( 'chat_web_search', False ) ), key='chat_web_search' )
			
			with p_row5[ 0 ]:
				chat_store = st.checkbox( 'Store', value=bool( st.session_state.get( 'chat_store', True ) ), key='chat_store' )
			
			with p_row5[ 1 ]:
				chat_stream = st.checkbox( 'Stream', value=bool( st.session_state.get( 'chat_stream', False ) ), key='chat_stream' )
			
			_chat_supports_reasoning = (
					str( chat_model ).strip( ).lower( ).startswith( 'gpt-5' ) or str( chat_model ).strip( ).lower( ).startswith( 'o' ))
			
			if _chat_supports_reasoning and chat_reasoning:
				chat_reasoning_effort = st.selectbox( 'Reasoning Effort', options=CHAT_REASONING_EFFORTS, index=CHAT_REASONING_EFFORTS.index( st.session_state.get( 'chat_reasoning_effort', 'low' ) ), key='chat_reasoning_effort' )
			else:
				chat_reasoning_effort = None
			
			chat_system = st.text_area( 'System', value=st.session_state.get( 'chat_system', '' ), height=120, key='chat_system' )
			
			if chat_web_search:
				chat_domains = st.text_area( 'Preferred Search Domains (one per line or comma-separated)', value=st.session_state.get( 'chat_domains', '' ), height=90, key='chat_domains', help='Examples: openai.com, platform.openai.com, arxiv.org' )
			else:
				chat_domains = ''
			
			btn_row = st.columns( 2 )
			
			with btn_row[ 0 ]:
				chat_submit = st.button( 'Submit', key='chat_submit' )
			
			with btn_row[ 1 ]:
				st.button( 'Clear', key='chat_clear', on_click=_clear_chat_state )
			
			# -----------------------------
			# Submit Button
			# -----------------------------
			if chat_submit:
				try:
					if not str( chat_prompt or '' ).strip( ):
						raise ValueError( 'Prompt cannot be empty.' )
					
					if chat_json_mode:
						has_json_instruction = (
								'json' in str( chat_prompt or '' ).lower( ) or 'json' in str( chat_system or '' ).lower( ))
						
						if not has_json_instruction:
							chat_system = (
										str( chat_system or '' ).strip( ) + '\n\nReturn valid JSON '
										                                    'only.').strip( )
					
					chat_domains_list = (
							_normalize_chat_domains( chat_domains ) if chat_web_search else [ ])
					
					fetcher = Chat( )
					params = { 'model': chat_model, 'temperature': float( chat_temperature ),
							'max_tokens': int( chat_max_tokens ), 'top_p': float( chat_top_p ),
							'seed': int( chat_seed ) if int( chat_seed ) > 0 else None,
							'system': chat_system if str( chat_system ).strip( ) else None,
							'response_format': 'json' if chat_json_mode else None,
							'reasoning_effort': (
									chat_reasoning_effort if _chat_supports_reasoning and chat_reasoning and chat_reasoning_effort else None),
							'web_search': bool( chat_web_search ),
							'search_domains': chat_domains_list if chat_domains_list else None,
							'store': bool( chat_store ), 'stream': bool( chat_stream ),
							'parallel_tool_calls': True, 'tool_choice': 'auto', }
					
					params = { key: value for key, value in params.items( ) if value is not None }
					
					result = _invoke_provider( fetcher, chat_prompt, params )
					_promote_generation_result( provider='ChatGPT', result=result )
				
				except Exception as exc:
					st.error( str( exc ) )
		
		# ---------------------
		# ---- Expander Groq
		# ---------------------
		with st.expander( label='Grok', expanded=False ):
			GROK_REASONING_EFFORTS = [ 'none', 'low', 'medium', 'high' ]
			
			def _clear_grok_state( ) -> None:
				st.session_state[ 'grok_clear_request' ] = True
				_clear_generation_result( 'Grok' )
			
			def _normalize_grok_domains( value: object ) -> list[ str ]:
				text = str( value or '' ).strip( )
				
				if not text:
					return [ ]
				
				values: list[ str ] = [ ]
				entries = re.split( r'[\n,;]+', text )
				
				for entry in entries:
					domain = str( entry or '' ).strip( ).lower( )
					
					if not domain:
						continue
					
					domain = re.sub( r'^https?://', '', domain )
					domain = domain.split( '/' )[ 0 ]
					domain = re.sub( r':\d+$', '', domain )
					domain = domain.lstrip( '.' )
					
					if domain.startswith( 'www.' ):
						domain = domain[ 4: ]
					
					if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
						raise ValueError( f'Invalid Grok web-search domain: {domain}' )
					
					if domain not in values:
						values.append( domain )
				
				if len( values ) > 5:
					raise ValueError( 'xAI web-search allowed domains are limited to five domains.' )
				
				return values
			
			def _normalize_grok_stop_lines( value: object ) -> list[ str ]:
				text = str( value or '' )
				
				if not text.strip( ):
					return [ ]
				
				return [ line.strip( ) for line in text.splitlines( ) if line.strip( ) ]
			
			if 'grok_clear_request' not in st.session_state:
				st.session_state[ 'grok_clear_request' ] = False
			
			if 'groq_prompt_chat' not in st.session_state:
				st.session_state[ 'groq_prompt_chat' ] = ''
			
			if 'groq_system_chat' not in st.session_state:
				st.session_state[ 'groq_system_chat' ] = ''
			
			if 'groq_domains_chat' not in st.session_state:
				st.session_state[ 'groq_domains_chat' ] = ''
			
			if 'groq_stop_chat' not in st.session_state:
				st.session_state[ 'groq_stop_chat' ] = ''
			
			if 'groq_json_mode_chat' not in st.session_state:
				st.session_state[ 'groq_json_mode_chat' ] = False
			
			if 'groq_reasoning_chat' not in st.session_state:
				st.session_state[ 'groq_reasoning_chat' ] = False
			
			if 'groq_web_search_chat' not in st.session_state:
				st.session_state[ 'groq_web_search_chat' ] = False
			
			if 'groq_store_chat' not in st.session_state:
				st.session_state[ 'groq_store_chat' ] = True
			
			if 'groq_stream_chat' not in st.session_state:
				st.session_state[ 'groq_stream_chat' ] = False
			
			if 'groq_seed_chat' not in st.session_state:
				st.session_state[ 'groq_seed_chat' ] = 0
			
			if st.session_state.get( 'groq_reasoning_effort_chat', 'low' ) not in GROK_REASONING_EFFORTS:
				st.session_state[ 'groq_reasoning_effort_chat' ] = 'low'
			
			if st.session_state.get( 'grok_clear_request', False ):
				st.session_state[ 'groq_prompt_chat' ] = ''
				st.session_state[ 'groq_system_chat' ] = ''
				st.session_state[ 'groq_domains_chat' ] = ''
				st.session_state[ 'groq_stop_chat' ] = ''
				st.session_state[ 'groq_json_mode_chat' ] = False
				st.session_state[ 'groq_reasoning_chat' ] = False
				st.session_state[ 'groq_web_search_chat' ] = False
				st.session_state[ 'groq_store_chat' ] = True
				st.session_state[ 'groq_stream_chat' ] = False
				st.session_state[ 'groq_seed_chat' ] = 0
				st.session_state[ 'groq_reasoning_effort_chat' ] = 'low'
				st.session_state[ 'grok_clear_request' ] = False
			
			groq_prompt = st.text_area( 'Prompt', value=st.session_state.get( 'groq_prompt_chat', '' ), height=120, key='groq_prompt_chat', )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_grok_models = (
						cfg.GROK_MODELS if hasattr( cfg, 'GROK_MODELS' ) and cfg.GROK_MODELS else [
								'grok-4.3', 'grok-4.20', 'grok-4.20-reasoning',
								'grok-4.20-multi-agent', 'grok-4-1-fast', 'grok-4-fast-reasoning',
								'grok-4', 'grok-code-fast-1', 'grok-3-mini' ])
				
				groq_model = _model_selector( key_prefix='groq', label='Model', options=_grok_models, default_model=(
						'grok-4.3' if 'grok-4.3' in _grok_models else _grok_models[ 0 ]), )
			
			with p_row1[ 1 ]:
				groq_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.05, key='groq_temperature_chat', )
			
			with p_row2[ 0 ]:
				groq_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768, value=2048, step=1, key='groq_max_tokens_chat', )
			
			with p_row2[ 1 ]:
				groq_top_p = st.slider( 'Top-P', min_value=0.0, max_value=1.0, value=1.0, step=0.01, key='groq_top_p_chat', )
			
			with p_row3[ 0 ]:
				groq_seed = st.number_input( 'Seed', min_value=0, max_value=2_147_483_647, value=int( st.session_state.get( 'groq_seed_chat', 0 ) ), step=1, key='groq_seed_chat', help='Use 0 to omit the seed parameter.' )
			
			with p_row3[ 1 ]:
				groq_json_mode = st.checkbox( 'JSON Mode', value=bool( st.session_state.get( 'groq_json_mode_chat', False ) ), key='groq_json_mode_chat', help='Adds JSON-only instructions through the current Grok wrapper.' )
			
			with p_row4[ 0 ]:
				groq_reasoning = st.checkbox( 'Reasoning', value=bool( st.session_state.get( 'groq_reasoning_chat', False ) ), key='groq_reasoning_chat' )
			
			with p_row4[ 1 ]:
				groq_web_search = st.checkbox( 'Web Search', value=bool( st.session_state.get( 'groq_web_search_chat', False ) ), key='groq_web_search_chat' )
			
			with p_row5[ 0 ]:
				groq_store = st.checkbox( 'Store', value=bool( st.session_state.get( 'groq_store_chat', True ) ), key='groq_store_chat' )
			
			with p_row5[ 1 ]:
				groq_stream = st.checkbox( 'Stream', value=bool( st.session_state.get( 'groq_stream_chat', False ) ), key='groq_stream_chat' )
			
			_groq_model_name = str( groq_model or '' ).strip( ).lower( )
			_groq_is_reasoning_model = (
					'reasoning' in _groq_model_name or _groq_model_name.startswith( 'grok-4' ) or _groq_model_name.startswith( 'grok-4.3' ) or _groq_model_name.startswith( 'grok-4.20' ))
			
			_groq_supports_reasoning_effort = (
					_groq_model_name == 'grok-4.3' or _groq_model_name == 'grok-4.20-multi-agent')
			
			if _groq_supports_reasoning_effort and groq_reasoning:
				groq_reasoning_effort = st.selectbox( 'Reasoning Effort', options=GROK_REASONING_EFFORTS, index=GROK_REASONING_EFFORTS.index( st.session_state.get( 'groq_reasoning_effort_chat', 'low' ) ), key='groq_reasoning_effort_chat' )
			else:
				groq_reasoning_effort = None
			
			groq_system = st.text_area( 'System', value=st.session_state.get( 'groq_system_chat', '' ), height=120, key='groq_system_chat' )
			
			if groq_web_search:
				groq_domains = st.text_area( 'Allowed Search Domains', value=st.session_state.get( 'groq_domains_chat', '' ), height=90, key='groq_domains_chat', help='Optional. xAI allows up to five allowed domains.' )
			else:
				groq_domains = ''
			
			groq_stop = st.text_area( 'Stop Sequences', value=st.session_state.get( 'groq_stop_chat', '' ), height=80, key='groq_stop_chat', disabled=_groq_is_reasoning_model, help='One stop sequence per line. Disabled for reasoning models.' )
			
			btn_row = st.columns( 2 )
			
			with btn_row[ 0 ]:
				groq_submit = st.button( 'Submit', key='groq_submit' )
			
			with btn_row[ 1 ]:
				st.button( 'Clear', key='groq_clear', on_click=_clear_grok_state )
			
			# -----------------------------
			# Submit Button
			# -----------------------------
			if groq_submit:
				try:
					if not str( groq_prompt or '' ).strip( ):
						raise ValueError( 'Prompt cannot be empty.' )
					
					if groq_json_mode:
						has_json_instruction = (
								'json' in str( groq_prompt or '' ).lower( ) or 'json' in str( groq_system or '' ).lower( ))
						
						if not has_json_instruction:
							groq_system = (
										str( groq_system or '' ).strip( ) + '\n\nReturn valid JSON '
										                                    'only.').strip( )
					
					groq_domains_list = (
							_normalize_grok_domains( groq_domains ) if groq_web_search else [ ])
					
					stop_lines = _normalize_grok_stop_lines( groq_stop )
					
					if stop_lines and _groq_is_reasoning_model:
						stop_lines = [ ]
					
					fetcher = Grok( )
					params = { 'model': groq_model, 'temperature': float( groq_temperature ),
							'max_tokens': int( groq_max_tokens ), 'top_p': float( groq_top_p ),
							'seed': int( groq_seed ) if int( groq_seed ) > 0 else None,
							'system': groq_system if str( groq_system ).strip( ) else None,
							'response_format': 'json' if groq_json_mode else None,
							'reasoning_effort': (
									groq_reasoning_effort if _groq_supports_reasoning_effort and groq_reasoning and groq_reasoning_effort else None),
							'web_search': bool( groq_web_search ),
							'search_domains': groq_domains_list if groq_domains_list else None,
							'stop': stop_lines if stop_lines else None,
							'stream': bool( groq_stream ), 'store': bool( groq_store ),
							'parallel_tool_calls': True, 'tool_choice': 'auto', }
					
					params = { key: value for key, value in params.items( ) if value is not None }
					result = _invoke_provider( fetcher, groq_prompt, params )
					_promote_generation_result( provider='Grok', result=result )
				
				except Exception as exc:
					st.error( str( exc ) )
		
		# ---------------------
		# ---- Expander Claude
		# ---------------------
		with st.expander( label='Claude', expanded=False ):
			def _clear_claude_state( ) -> None:
				'''
					Purpose:
					--------
					Flag the Claude generation state for reset on the next rerun.

					Parameters:
					-----------
					None

					Returns:
					--------
					None
				'''
				st.session_state[ 'claude_clear_request' ] = True
				_clear_generation_result( 'Claude' )
			
			def _normalize_claude_domains( value: object ) -> list[ str ]:
				'''
					Purpose:
					--------
					Normalize newline-, comma-, or semicolon-delimited Claude web-search
					domain entries.

					Parameters:
					-----------
					value (object):
						Domain text supplied by the user.

					Returns:
					--------
					list[str]:
						Canonical, de-duplicated domain names.
				'''
				text = str( value or '' ).strip( )
				
				if not text:
					return [ ]
				
				values: list[ str ] = [ ]
				entries = re.split( r'[\n,;]+', text )
				
				for entry in entries:
					raw_value = str( entry or '' ).strip( ).lower( )
					
					if not raw_value:
						continue
					
					if not raw_value.startswith( 'http://' ) and not raw_value.startswith( 'https://' ):
						raw_value = f'https://{raw_value}'
					
					parsed = urlparse( raw_value )
					domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
					domain = re.sub( r':\d+$', '', domain )
					domain = domain.lstrip( '.' )
					
					if domain.startswith( 'www.' ):
						domain = domain[ 4: ]
					
					if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
						raise ValueError( f'Invalid Claude web-search domain: {domain}' )
					
					if domain not in values:
						values.append( domain )
				
				return values
			
			def _normalize_claude_stop_lines( value: object ) -> list[ str ]:
				'''
					Purpose:
					--------
					Normalize stop sequences entered one per line.

					Parameters:
					-----------
					value (object):
						Stop-sequence textarea value.

					Returns:
					--------
					list[str]:
						Non-empty stop sequences.
				'''
				text = str( value or '' )
				
				if not text.strip( ):
					return [ ]
				
				return [ line.strip( ) for line in text.splitlines( ) if line.strip( ) ]
			
			if 'claude_clear_request' not in st.session_state:
				st.session_state[ 'claude_clear_request' ] = False
			
			if 'claude_prompt_chat' not in st.session_state:
				st.session_state[ 'claude_prompt_chat' ] = ''
			
			if 'claude_system_chat' not in st.session_state:
				st.session_state[ 'claude_system_chat' ] = ''
			
			if 'claude_stop_chat' not in st.session_state:
				st.session_state[ 'claude_stop_chat' ] = ''
			
			if 'claude_domains_chat' not in st.session_state:
				st.session_state[ 'claude_domains_chat' ] = ''
			
			if 'claude_blocked_domains_chat' not in st.session_state:
				st.session_state[ 'claude_blocked_domains_chat' ] = ''
			
			if 'claude_thinking_chat' not in st.session_state:
				st.session_state[ 'claude_thinking_chat' ] = False
			
			if 'claude_web_search_chat' not in st.session_state:
				st.session_state[ 'claude_web_search_chat' ] = False
			
			if 'claude_thinking_budget_chat' not in st.session_state:
				st.session_state[ 'claude_thinking_budget_chat' ] = 1024
			
			if st.session_state.get( 'claude_clear_request', False ):
				st.session_state[ 'claude_prompt_chat' ] = ''
				st.session_state[ 'claude_system_chat' ] = ''
				st.session_state[ 'claude_stop_chat' ] = ''
				st.session_state[ 'claude_domains_chat' ] = ''
				st.session_state[ 'claude_blocked_domains_chat' ] = ''
				st.session_state[ 'claude_thinking_chat' ] = False
				st.session_state[ 'claude_web_search_chat' ] = False
				st.session_state[ 'claude_thinking_budget_chat' ] = 1024
				st.session_state[ 'claude_clear_request' ] = False
			
			claude_prompt = st.text_area( 'Prompt', value=st.session_state.get( 'claude_prompt_chat', '' ), height=140, key='claude_prompt_chat', )
			
			# -----------------------------
			# Model / Output Controls
			# -----------------------------
			model_row = st.columns( [ 0.55, 0.45 ] )
			
			with model_row[ 0 ]:
				_claude_models = (
					cfg.CLAUDE_MODELS if hasattr( cfg, 'CLAUDE_MODELS' ) and cfg.CLAUDE_MODELS else [
							'claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5',
							'claude-3-5-haiku-latest', ])
				
				claude_model = _model_selector( key_prefix='claude', label='Model', options=_claude_models, default_model=(
						'claude-sonnet-4-6' if 'claude-sonnet-4-6' in _claude_models else
						_claude_models[ 0 ]), )
			
			with model_row[ 1 ]:
				claude_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=65536, value=2048, step=1, key='claude_max_tokens_chat', )
			
			# -----------------------------
			# Sampling Controls
			# -----------------------------
			sampling_row = st.columns( 3 )
			
			with sampling_row[ 0 ]:
				claude_temperature = st.slider( 'Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='claude_temperature_chat', )
			
			with sampling_row[ 1 ]:
				claude_top_p = st.slider( 'Top-P', min_value=0.0, max_value=1.0, value=1.0, step=0.01, key='claude_top_p_chat', )
			
			with sampling_row[ 2 ]:
				claude_top_k = st.number_input( 'Top-k', min_value=0, max_value=500, value=0, step=1, key='claude_top_k_chat', )
			
			# -----------------------------
			# Feature Toggles
			# -----------------------------
			option_row = st.columns( 2 )
			
			with option_row[ 0 ]:
				claude_thinking = st.checkbox( 'Reasoning', value=bool( st.session_state.get( 'claude_thinking_chat', False ) ), key='claude_thinking_chat', help='Anthropic exposes this as extended thinking with a token budget.', )
			
			with option_row[ 1 ]:
				claude_web_search = st.checkbox( 'Web Search', value=bool( st.session_state.get( 'claude_web_search_chat', False ) ), key='claude_web_search_chat', )
			
			# -----------------------------
			# Reasoning Budget
			# -----------------------------
			if claude_thinking:
				claude_thinking_budget = st.number_input( 'Thinking Budget', min_value=1024, max_value=max( 1024, int( claude_max_tokens ) - 1 ), value=min( int( st.session_state.get( 'claude_thinking_budget_chat', 1024 ) ), max( 1024, int( claude_max_tokens ) - 1 ) ), step=1024, key='claude_thinking_budget_chat', help='Must be less than Max Tokens.' )
			else:
				claude_thinking_budget = None
			
			# -----------------------------
			# Instruction Controls
			# -----------------------------
			claude_system = st.text_area( 'System', value=st.session_state.get( 'claude_system_chat', '' ), height=110, key='claude_system_chat', )
			
			claude_stop = st.text_area( 'Stop Sequences (one per line)', value=st.session_state.get( 'claude_stop_chat', '' ), height=80, key='claude_stop_chat', )
			
			# -----------------------------
			# Web Search Controls
			# -----------------------------
			if claude_web_search:
				search_row = st.columns( 2 )
				
				with search_row[ 0 ]:
					claude_domains = st.text_area( 'Allowed Search Domains', value=st.session_state.get( 'claude_domains_chat', '' ), height=90, key='claude_domains_chat', help='Optional allowlist, one domain per line or comma-separated.' )
				
				with search_row[ 1 ]:
					claude_blocked_domains = st.text_area( 'Blocked Search Domains', value=st.session_state.get( 'claude_blocked_domains_chat', '' ), height=90, key='claude_blocked_domains_chat', help='Optional blocklist, one domain per line or comma-separated.' )
			else:
				claude_domains = ''
				claude_blocked_domains = ''
			
			# -----------------------------
			# Action Controls
			# -----------------------------
			btn_row = st.columns( 2 )
			
			with btn_row[ 0 ]:
				claude_submit = st.button( 'Submit', key='claude_submit_chat', use_container_width=True )
			
			with btn_row[ 1 ]:
				st.button( 'Clear', key='claude_clear_chat', on_click=_clear_claude_state, use_container_width=True )
		
		# ---------------------
		# ---- Expander GEMINI
		# ---------------------
		with st.expander( label='Gemini', expanded=False ):
			GEMINI_THINKING_LEVELS = [ 'minimal', 'low', 'medium', 'high' ]
			
			def _clear_gemini_state( ) -> None:
				st.session_state[ 'gemini_clear_request' ] = True
				_clear_generation_result( 'Gemini' )
			
			def _normalize_gemini_domains( value: object ) -> list[ str ]:
				text = str( value or '' ).strip( )
				
				if not text:
					return [ ]
				
				values: list[ str ] = [ ]
				entries = re.split( r'[\n,;]+', text )
				
				for entry in entries:
					raw_value = str( entry or '' ).strip( ).lower( )
					
					if not raw_value:
						continue
					
					if not raw_value.startswith( 'http://' ) and not raw_value.startswith( 'https://' ):
						raw_value = f'https://{raw_value}'
					
					parsed = urlparse( raw_value )
					domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
					domain = re.sub( r':\d+$', '', domain )
					domain = domain.lstrip( '.' )
					
					if domain.startswith( 'www.' ):
						domain = domain[ 4: ]
					
					if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
						raise ValueError( f'Invalid Gemini grounding domain: {domain}' )
					
					if domain not in values:
						values.append( domain )
				
				return values
			
			def _normalize_gemini_stop_lines( value: object ) -> list[ str ]:
				text = str( value or '' )
				
				if not text.strip( ):
					return [ ]
				
				return [ line.strip( ) for line in text.splitlines( ) if line.strip( ) ]
			
			if 'gemini_clear_request' not in st.session_state:
				st.session_state[ 'gemini_clear_request' ] = False
			
			if 'gemini_prompt_chat' not in st.session_state:
				st.session_state[ 'gemini_prompt_chat' ] = ''
			
			if 'gemini_system_chat' not in st.session_state:
				st.session_state[ 'gemini_system_chat' ] = ''
			
			if 'gemini_domains_chat' not in st.session_state:
				st.session_state[ 'gemini_domains_chat' ] = ''
			
			if 'gemini_stop_chat' not in st.session_state:
				st.session_state[ 'gemini_stop_chat' ] = ''
			
			if 'gemini_json_mode_chat' not in st.session_state:
				st.session_state[ 'gemini_json_mode_chat' ] = False
			
			if 'gemini_grounding_chat' not in st.session_state:
				st.session_state[ 'gemini_grounding_chat' ] = False
			
			if 'gemini_reasoning_chat' not in st.session_state:
				st.session_state[ 'gemini_reasoning_chat' ] = False
			
			if 'gemini_include_thoughts_chat' not in st.session_state:
				st.session_state[ 'gemini_include_thoughts_chat' ] = False
			
			if 'gemini_seed_chat' not in st.session_state:
				st.session_state[ 'gemini_seed_chat' ] = 0
			
			if st.session_state.get( 'gemini_thinking_level_chat', 'low' ) not in GEMINI_THINKING_LEVELS:
				st.session_state[ 'gemini_thinking_level_chat' ] = 'low'
			
			if st.session_state.get( 'gemini_clear_request', False ):
				st.session_state[ 'gemini_prompt_chat' ] = ''
				st.session_state[ 'gemini_system_chat' ] = ''
				st.session_state[ 'gemini_domains_chat' ] = ''
				st.session_state[ 'gemini_stop_chat' ] = ''
				st.session_state[ 'gemini_json_mode_chat' ] = False
				st.session_state[ 'gemini_grounding_chat' ] = False
				st.session_state[ 'gemini_reasoning_chat' ] = False
				st.session_state[ 'gemini_include_thoughts_chat' ] = False
				st.session_state[ 'gemini_seed_chat' ] = 0
				st.session_state[ 'gemini_thinking_level_chat' ] = 'low'
				st.session_state[ 'gemini_clear_request' ] = False
			
			gemini_prompt = st.text_area( 'Prompt', value=st.session_state.get( 'gemini_prompt_chat', '' ), height=160, key='gemini_prompt_chat', )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_gemini_models = (
					cfg.GEMINI_MODELS if hasattr( cfg, 'GEMINI_MODELS' ) and cfg.GEMINI_MODELS else [
							'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
							'gemini-2.5-flash-lite' ])
				
				gemini_model = _model_selector( key_prefix='gemini', label='Model', options=_gemini_models, default_model=(
						'gemini-2.5-flash' if 'gemini-2.5-flash' in _gemini_models else
						_gemini_models[ 0 ]), )
			
			with p_row1[ 1 ]:
				gemini_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.05, key='gemini_temperature_chat', )
			
			with p_row2[ 0 ]:
				gemini_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768, value=2048, step=1, key='gemini_max_tokens_chat', )
			
			with p_row2[ 1 ]:
				gemini_top_p = st.slider( 'Top-p', min_value=0.0, max_value=1.0, value=1.0, step=0.01, key='gemini_top_p_chat', )
			
			with p_row3[ 0 ]:
				gemini_top_k = st.number_input( 'Top-k', min_value=0, max_value=500, value=0, step=1, key='gemini_top_k_chat', )
			
			with p_row3[ 1 ]:
				gemini_candidate_count = st.number_input( 'Candidates', min_value=1, max_value=8, value=1, step=1, key='gemini_candidate_count_chat', )
			
			with p_row4[ 0 ]:
				gemini_seed = st.number_input( 'Seed', min_value=0, max_value=2_147_483_647, value=int( st.session_state.get( 'gemini_seed_chat', 0 ) ), step=1, key='gemini_seed_chat', help='Use 0 to omit the seed parameter.' )
			
			with p_row4[ 1 ]:
				gemini_json_mode = st.checkbox( 'JSON Mode', value=bool( st.session_state.get( 'gemini_json_mode_chat', False ) ), key='gemini_json_mode_chat', help='Requests JSON output through the current Gemini wrapper.' )
			
			with p_row5[ 0 ]:
				gemini_grounding = st.checkbox( 'Grounding', value=bool( st.session_state.get( 'gemini_grounding_chat', False ) ), key='gemini_grounding_chat', help='Enable Google Search grounding for supported Gemini models.' )
			
			with p_row5[ 1 ]:
				gemini_reasoning = st.checkbox( 'Reasoning', value=bool( st.session_state.get( 'gemini_reasoning_chat', False ) ), key='gemini_reasoning_chat', help='Uses Gemini thinking configuration where supported.' )
			
			_gemini_model_name = str( gemini_model or '' ).strip( ).lower( )
			_gemini_supports_thinking_level = _gemini_model_name.startswith( 'gemini-3' )
			_gemini_supports_thinking_budget = _gemini_model_name.startswith( 'gemini-2.5' )
			
			if _gemini_supports_thinking_level and gemini_reasoning:
				r_row = st.columns( 2 )
				
				with r_row[ 0 ]:
					gemini_thinking_level = st.selectbox( 'Thinking Level', options=GEMINI_THINKING_LEVELS, index=GEMINI_THINKING_LEVELS.index( st.session_state.get( 'gemini_thinking_level_chat', 'low' ) ), key='gemini_thinking_level_chat', )
				
				with r_row[ 1 ]:
					gemini_include_thoughts = st.checkbox( 'Include Thoughts', value=bool( st.session_state.get( 'gemini_include_thoughts_chat', False ) ), key='gemini_include_thoughts_chat', )
			else:
				gemini_thinking_level = None
				gemini_include_thoughts = False
			
			if gemini_reasoning and _gemini_supports_thinking_budget:
				st.caption( 'Gemini 2.5 models use thinking_budget, but the current Gemini '
				            'generator only exposes thinking_level. This UI preserves the '
				            'current '
				            'wrapper contract until the Gemini class is updated.' )
			
			if gemini_reasoning and not (
					_gemini_supports_thinking_level or _gemini_supports_thinking_budget):
				st.caption( 'Reasoning controls are only sent for Gemini 3 model names under '
				            'the current wrapper contract.' )
			
			gemini_stop = st.text_area( 'Stop Sequences (one per line)', value=st.session_state.get( 'gemini_stop_chat', '' ), height=80, key='gemini_stop_chat', )
			
			gemini_system = st.text_area( 'System', value=st.session_state.get( 'gemini_system_chat', '' ), height=110, key='gemini_system_chat', )
			
			if gemini_grounding:
				gemini_domains = st.text_area( 'Preferred Search Domains (one per line or comma-separated)', value=st.session_state.get( 'gemini_domains_chat', '' ), height=90, key='gemini_domains_chat', help='Used as preferred source guidance for grounded Gemini responses.' )
			else:
				gemini_domains = ''
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				gemini_submit = st.button( 'Submit', key='gemini_submit_chat' )
			
			with btn_row[ 1 ]:
				st.button( 'Clear', key='gemini_clear_chat', on_click=_clear_gemini_state )
			
			# ------ Submit Button
			if gemini_submit:
				try:
					if not str( gemini_prompt or '' ).strip( ):
						raise ValueError( 'Prompt cannot be empty.' )
					
					if gemini_json_mode:
						has_json_instruction = (
								'json' in str( gemini_prompt or '' ).lower( ) or 'json' in str( gemini_system or '' ).lower( ))
						
						if not has_json_instruction:
							gemini_system = (
										str( gemini_system or '' ).strip( ) + '\n\nReturn valid JSON '
										                                      'only.').strip( )
					
					gemini_domains_list = (
							_normalize_gemini_domains( gemini_domains ) if gemini_grounding else [ ])
					
					stop_lines = _normalize_gemini_stop_lines( gemini_stop )
					fetcher = Gemini( )
					params = { 'model': gemini_model, 'temperature': float( gemini_temperature ),
							'max_tokens': int( gemini_max_tokens ), 'top_p': float( gemini_top_p ),
							'top_k': int( gemini_top_k ) if int( gemini_top_k ) > 0 else None,
							'candidate_count': int( gemini_candidate_count ),
							'seed': int( gemini_seed ) if int( gemini_seed ) > 0 else None,
							'system': (
								gemini_system if str( gemini_system or '' ).strip( ) else None),
							'response_format': 'json' if gemini_json_mode else None,
							'stop_sequences': stop_lines if stop_lines else None,
							'grounding': bool( gemini_grounding ), 'search_domains': (
								gemini_domains_list if gemini_domains_list else None),
							'reasoning': bool( gemini_reasoning and _gemini_supports_thinking_level ),
							'thinking_level': (
									gemini_thinking_level if _gemini_supports_thinking_level and gemini_reasoning else None),
							'include_thoughts': bool( gemini_include_thoughts ) if _gemini_supports_thinking_level and gemini_reasoning else False, }
					
					params = { key: value for key, value in params.items( ) if value is not None }
					
					result = _invoke_provider( fetcher, gemini_prompt, params )
					_promote_generation_result( provider='Gemini', result=result )
				
				except Exception as exc:
					st.error( str( exc ) )
		
		# ---------------------
		# ---- Expander Mistral
		# ---------------------
		with st.expander( label='Mistral', expanded=False ):
			def _clear_mistral_state( ) -> None:
				st.session_state[ 'mistral_clear_request' ] = True
				_clear_generation_result( 'Mistral' )
			
			if 'mistral_clear_request' not in st.session_state:
				st.session_state[ 'mistral_clear_request' ] = False
			
			if 'mistral_prompt_chat' not in st.session_state:
				st.session_state[ 'mistral_prompt_chat' ] = ''
			
			if 'mistral_system_chat' not in st.session_state:
				st.session_state[ 'mistral_system_chat' ] = ''
			
			if 'mistral_safe_mode_chat' not in st.session_state:
				st.session_state[ 'mistral_safe_mode_chat' ] = False
			
			if 'mistral_seed_chat' not in st.session_state:
				st.session_state[ 'mistral_seed_chat' ] = 0
			
			if st.session_state.get( 'mistral_clear_request', False ):
				st.session_state[ 'mistral_prompt_chat' ] = ''
				st.session_state[ 'mistral_system_chat' ] = ''
				st.session_state[ 'mistral_safe_mode_chat' ] = False
				st.session_state[ 'mistral_seed_chat' ] = 0
				st.session_state[ 'mistral_clear_request' ] = False
			
			mistral_prompt = st.text_area( 'Prompt', value=st.session_state.get( 'mistral_prompt_chat', '' ), height=120, key='mistral_prompt_chat' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			with p_row1[ 0 ]:
				_mistral_models = (
					cfg.MISTRAL_MODELS if hasattr( cfg, 'MISTRAL_MODELS' ) and cfg.MISTRAL_MODELS else [
							'mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest',
							'open-mistral-7b', 'Custom...', ])
				
				mistral_model = _model_selector( key_prefix='mistral', label='Model', options=_mistral_models, default_model=(
						'mistral-large-latest' if 'mistral-large-latest' in _mistral_models else
						_mistral_models[ 0 ]), )
			
			with p_row1[ 1 ]:
				mistral_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.05, key='mistral_temperature_chat', help='Mistral recommends tuning temperature or top-p, not both.' )
			
			with p_row2[ 0 ]:
				mistral_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768, value=1024, step=1, key='mistral_max_tokens_chat', )
			
			with p_row2[ 1 ]:
				mistral_top_p = st.slider( 'Top-p', min_value=0.0, max_value=1.0, value=1.0, step=0.01, key='mistral_top_p_chat', help='Mistral recommends tuning temperature or top-p, not both.' )
			
			with p_row3[ 0 ]:
				mistral_seed = st.number_input( 'Seed', min_value=0, max_value=2_147_483_647, value=int( st.session_state.get( 'mistral_seed_chat', 0 ) ), step=1, key='mistral_seed_chat', help='Use 0 to omit random_seed.' )
			
			with p_row3[ 1 ]:
				mistral_safe_mode = st.checkbox( 'Safe Mode', value=bool( st.session_state.get( 'mistral_safe_mode_chat', False ) ), key='mistral_safe_mode_chat', help='Maps to Mistral safe_prompt in the current wrapper.' )
			
			mistral_system = st.text_area( 'System', value=st.session_state.get( 'mistral_system_chat', '' ), height=100, key='mistral_system_chat', )
			
			st.caption( 'The current Mistral wrapper supports text output only. JSON mode and '
			            'stop sequences require a later complete Mistral class replacement.' )
			
			btn_row = st.columns( 2 )
			
			with btn_row[ 0 ]:
				mistral_submit = st.button( 'Submit', key='mistral_submit_chat' )
			
			with btn_row[ 1 ]:
				st.button( 'Clear', key='mistral_clear_chat', on_click=_clear_mistral_state )
			
			if mistral_submit:
				try:
					if not str( mistral_prompt or '' ).strip( ):
						raise ValueError( 'Prompt cannot be empty.' )
					
					if float( mistral_temperature ) > 0.0 and float( mistral_top_p ) < 1.0:
						st.warning( 'Mistral recommends adjusting either temperature or top-p, '
						            'but not both, for most use cases.' )
					
					fetcher = Mistral( )
					params = { 'model': mistral_model, 'temperature': float( mistral_temperature ),
							'max_tokens': int( mistral_max_tokens ),
							'top_p': float( mistral_top_p ),
							'seed': int( mistral_seed ) if int( mistral_seed ) > 0 else None,
							'safe_mode': bool( mistral_safe_mode ), 'system': (
									mistral_system if str( mistral_system or '' ).strip( ) else None), }
					
					params = { key: value for key, value in params.items( ) if value is not None }
					
					result = _invoke_provider( fetcher, mistral_prompt, params )
					_promote_generation_result( provider='Mistral', result=result )
				
				except Exception as exc:
					st.error( str( exc ) )

# ============================================
# DATA MANAGEMENT MODE
# ============================================
elif mode == 'Data Browse':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	
	tables = list_tables( )
	if tables:
		st.header( '' )
		browse_left, browse_center, browse_right = st.columns( 3 )
		with browse_left:
			table = st.selectbox( 'Select Table:', tables, key='table_name' )
		
		set_blue_divider( )
		df = read_table( table )
		render_data_editor( df, use_container_width=True, height=400 )
	else:
		st.info( 'No tables available.' )

# ============================================
# DATA UPLOAD MODE
# ============================================
elif mode == 'Data Upload':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Data Upload' )
		st.divider( )
		upl_c1, upl_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with upl_c1:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[
					'xlsx' ], key='data_upload_file' )
		
		with upl_c2:
			overwrite = st.checkbox( 'Overwrite existing tables', value=True, key='data_upload_overwrite' )
			
			if uploaded_file is not None:
				file_bytes = uploaded_file.getvalue( )
				upload_signature = (uploaded_file.name, len( file_bytes ), hash( file_bytes ),
						overwrite)
				
				if st.session_state[ 'data_upload_signature' ] != upload_signature:
					try:
						uploaded_file.seek( 0 )
						sheets = pd.read_excel( uploaded_file, sheet_name=None )
						if not sheets:
							raise ValueError( 'The uploaded workbook does not contain any worksheets.' )
						
						source_sheet_names = list( sheets.keys( ) )
						table_names = create_unique_upload_identifiers( source_sheet_names )
						sheet_table_map = dict( zip( source_sheet_names, table_names ) )
						df_tables = { table_name: sheets[ sheet_name ] for sheet_name, table_name in
								sheet_table_map.items( ) }
						imported_tables = write_dataframe_tables_to_database( df_tables, overwrite=overwrite )
						imported_sheet_names = { table_name: sheet_name for sheet_name, table_name
								in sheet_table_map.items( ) }
						
						st.session_state[ 'data_upload_tables' ] = imported_tables
						st.session_state[ 'data_upload_sheet_names' ] = imported_sheet_names
						st.session_state[ 'data_upload_filename' ] = uploaded_file.name
						st.session_state[ 'data_upload_signature' ] = upload_signature
						first_table_name = next( iter( imported_tables ) )
						activate_database_table( first_table_name )
						queue_database_success( f'Imported {len( imported_tables ):,} worksheet table(s) from '
						                        f'"{uploaded_file.name}". "{first_table_name}" is now the active table.' )
						st.rerun( )
					except Exception as e:
						st.session_state[ 'data_upload_signature' ] = None
						st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------
		# IMPORTED TABLE SELECTION
		# ------------------------------------------------------------------
		imported_tables = st.session_state.get( 'data_upload_tables', { } )
		imported_sheet_names = st.session_state.get( 'data_upload_sheet_names', { } )
		imported_filename = st.session_state.get( 'data_upload_filename', None )
		if uploaded_file is not None and imported_tables and (
				imported_filename == uploaded_file.name):
			table_options = list( imported_tables.keys( ) )
			if len( table_options ) > 1:
				selected_table = st.selectbox( 'Imported Table', options=table_options, format_func=lambda value: (
						f'{imported_sheet_names.get( value, value )} ({value})'), key='data_upload_selected_table' )
			else:
				selected_table = table_options[ 0 ]
			
			if st.session_state.get( 'active_database_table', '' ) != selected_table:
				activate_database_table( selected_table )
			df_uploaded = read_table( selected_table )
			st.session_state[ 'data_upload_tables' ][ selected_table ] = df_uploaded.copy( )
			declared_schema = create_declared_schema_map( create_schema( selected_table ) )
			profiles = profile_dataframe_schema( df_uploaded, declared_schema )
			schema = { column: str( profile[ 'analytical_role' ] ) for column, profile in
					profiles.items( ) }
			type_counts = pd.Series( schema ).value_counts( )
			
			# ------------------------------------------------------------------
			# SCHEMA METRICS
			# ------------------------------------------------------------------
			set_blue_divider( )
			st.markdown( '##### Types' )
			
			m1, m2, m3, m4, m5 = st.columns( 5, border=True )
			m1.metric( 'Rows', f'{len( df_uploaded ):,}' )
			m2.metric( 'Numeric', int( type_counts.get( 'numeric', 0 ) ) )
			m3.metric( 'Ordinal / ID', int( type_counts.get( 'ordinal', 0 ) ) + int( type_counts.get( 'identifier', 0 ) ) )
			m4.metric( 'Categorical', int( type_counts.get( 'categorical', 0 ) ) )
			m5.metric( 'Datetime', int( type_counts.get( 'datetime', 0 ) ) )
			
			# ------------------------------------------------------------------
			# DATABASE TABLE DISPLAY
			# ------------------------------------------------------------------
			set_blue_divider( )
			st.markdown( f'##### Data — {selected_table}' )
			
			render_data_editor( df_uploaded, use_container_width=True, height=500, hide_index=True, disabled=True, key=f'data_upload_editor_{selected_table}' )

# ============================================
# CRUD OPS MODE
# ============================================
elif mode == 'CRUD Ops':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with (((center))):
		st.subheader( 'CRUD Ops' )
		set_blue_divider( )
		
		st.markdown( '##### Select Data' )
		tables = [ table_name for table_name in list_tables( ) if
				not table_name.startswith( 'sqlite_' ) ]
		if not tables:
			st.info( 'No database tables are available.' )
		else:
			pending_table = st.session_state.get( 'crud_next_table', '' )
			if pending_table and pending_table in tables:
				st.session_state[ 'crud_table' ] = pending_table
				st.session_state[ 'crud_next_table' ] = ''
			elif st.session_state.get( 'crud_table', None ) not in tables:
				clear_keys( [ 'crud_table' ] )
			
			crud_c1, crud_c2, crud_c3, crud_c4, crud_c5 = st.columns( 5, border=True )
			with crud_c1:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
			
			st.divider( )
			
			df_database = read_table( table )
			database_schema = create_schema( table )
			supports_rowid = table_supports_rowid( table )
			df_database_rows = read_table_with_rowid( table ) if supports_rowid else pd.DataFrame( )
			declared_schema = create_declared_schema_map( database_schema )
			crud_profiles = profile_dataframe_schema( df_database, declared_schema )
			crud_type_counts = pd.Series( { column: str( profile[ 'analytical_role' ] ) for
					column, profile in crud_profiles.items( ) } ).value_counts( )
			with crud_c2:
				st.metric( 'Total Records', f'{len( df_database ):,}' )
			with crud_c3:
				st.metric( 'Numeric Fields', int( crud_type_counts.get( 'numeric', 0 ) ) )
			with crud_c4:
				st.metric( 'Ordinal Fields', int( crud_type_counts.get( 'ordinal', 0 ) ) )
			with crud_c5:
				st.metric( 'Categorical Fields', int( crud_type_counts.get( 'categorical', 0 ) ) )
			type_map = { str( item[ 1 ] ): str( item[ 2 ] or 'TEXT' ).upper( ) for item in
					database_schema }
			primary_key_column_names = { str( item[ 1 ] ) for item in database_schema if
					int( item[ 5 ] or 0 ) > 0 }
			schema_signature = (table,
					tuple( (str( item[ 1 ] ), str( item[ 2 ] or '' )) for item in database_schema ))
			previous_signature = st.session_state.get( 'crud_database_schema_signature', None )
			if previous_signature != schema_signature:
				owned_prefixes = ('crud_insert_', 'crud_update_', 'crud_record_',
						'crud_schema_drop_', 'crud_schema_rename_', 'crud_schema_type_',
						'crud_table_manage_')
				stale_keys = [ key for key in st.session_state.keys( ) if
						str( key ).startswith( owned_prefixes ) ]
				clear_keys( stale_keys )
				st.session_state[ 'crud_database_schema_signature' ] = schema_signature
			
			st.data_editor( df_database )
			
			if not supports_rowid:
				st.warning( 'This table was created WITHOUT ROWID. Record-level CRUD operations '
				            'are '
				            'unavailable, but schema and table-management operations remain '
				            'available.' )
			else:
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				set_blue_divider( )
				st.markdown( '##### ➕ Add Record' )
				with st.expander( label='Insert', expanded=True ):
					insert_data_values: Dict[ str, object ] = { }
					primary_key_columns = [ item for item in database_schema if
							int( item[ 5 ] or 0 ) > 0 ]
					auto_generated_columns = { str( primary_key_columns[ 0 ][ 1 ] ) } if (
							len( primary_key_columns ) == 1 and str( primary_key_columns[ 0 ][
								                                         2 ] or '' ).strip( ).upper( ) == 'INTEGER') else set( )
					insert_schema = [ item for item in database_schema if
							str( item[ 1 ] ) not in auto_generated_columns ]
					insert_columns = st.columns( 4 )
					for index, schema_item in enumerate( insert_schema ):
						column = str( schema_item[ 1 ] )
						declared_type = str( schema_item[ 2 ] or 'TEXT' ).upper( )
						target_column = insert_columns[ index % 4 ]
						with target_column:
							unique_values = df_database[
								column ].dropna( ).drop_duplicates( ).tolist( )
							choice_key = f'crud_insert_choice_{table}_{index}'
							new_key = f'crud_insert_new_{table}_{index}'
							if unique_values:
								display_map = { str( value ): value for value in unique_values }
								options = [ '<Enter New Value>' ] + list( display_map.keys( ) )
								selection = st.selectbox( column, options, key=choice_key, help='Select an existing unique field value or choose Enter '
								                                                                'New Value.' )
								if selection == '<Enter New Value>':
									value = st.text_input( f'New {column}', key=new_key )
								else:
									value = display_map[ selection ]
							else:
								value = st.text_input( column, key=new_key )
							insert_data_values[ column ] = value
					
					st.divider( )
					
					ins_c1, ins_c2 = st.columns( 2 )
					with ins_c1:
						if st.button( label='Insert Row', icon='➕', width='stretch', key='crud_insert_submit_button' ):
							try:
								persisted_values = {
										column: coerce_sqlite_value( value, type_map[ column ] ) for
										column, value in insert_data_values.items( ) }
								if persisted_values and database_record_exists( table, persisted_values ):
									st.warning( 'This record already exists. No row was '
									            'inserted.' )
								else:
									with create_connection( ) as conn:
										if persisted_values:
											columns = list( persisted_values.keys( ) )
											column_list = ', '.join(
												f'"{column}"' for column in columns )
											placeholders = ', '.join( [ '?' ] * len( columns ) )
											statement = f'INSERT INTO "{table}" ({column_list}) VALUES ({placeholders});'
											cursor = conn.execute( statement, [
													persisted_values[ column ] for column in
													columns ] )
										else:
											cursor = conn.execute( f'INSERT INTO "{table}" DEFAULT '
											                       f'VALUES;' )
										inserted_rowid = int( cursor.lastrowid )
										conn.commit( )
									if persisted_values and not database_record_matches( table, inserted_rowid, persisted_values ):
										raise RuntimeError( 'Inserted row could not be verified.' )
									if not persisted_values and read_database_record( table, inserted_rowid ) is None:
										raise RuntimeError( 'Inserted row could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row inserted successfully into "'
									                        f'{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to insert row: {ex}' )
					
					with ins_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_insert_reset_button', on_click=reset_session_prefixes, args=(
								(f'crud_insert_choice_{table}_', f'crud_insert_new_{table}_'),) )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				set_blue_divider( )
				st.markdown( '##### 📤 Update Data' )
				with st.expander( label='Update', expanded=True ):
					upd_c1, upd_c2, upd_c3, upd_c4 = st.columns( 4 )
					with upd_c1:
						rowid = st.number_input( 'Row ID', min_value=1, step=1, key='crud_update_database_rowid' )
					current_record = read_database_record( table, int( rowid ) )
					editable_update_columns = [ column for column in current_record if
							column not in primary_key_column_names ] if current_record else [ ]
					update_values: Dict[ str, object ] = { }
					if current_record is None:
						st.info( 'Row ID not found.' )
					else:
						update_columns = st.columns( 4 )
						for index, column in enumerate( current_record.keys( ) ):
							target_column = update_columns[ index % 4 ]
							with target_column:
								current_value = current_record[ column ]
								update_values[
									column ] = st.text_input( column, value='' if current_value is None else str( current_value ), key=f'crud_update_value_{table}_'
								                                                                                                       f'{int( rowid )}_{index}', disabled=column in primary_key_column_names, help='Primary-key '
								                                                                                                                                                                                    'fields are '
								                                                                                                                                                                                    'read-only.' if column in primary_key_column_names else None )
					
					st.divider( )
					
					btn_c1, btn_c2 = st.columns( 2 )
					with btn_c1:
						if st.button( label='Update Row', icon='⬆️', width='stretch', key='crud_update_submit_button', disabled=current_record is None or not editable_update_columns ):
							try:
								persisted_values = {
										column: coerce_sqlite_value( value, type_map[ column ] ) for
										column, value in update_values.items( ) if
										column not in primary_key_column_names }
								if not persisted_values:
									raise ValueError( 'No editable non-primary-key fields are '
									                  'available.' )
								set_clause = ', '.join(
									f'"{column}"=?' for column in persisted_values )
								parameters = list( persisted_values.values( ) ) + [ int( rowid ) ]
								with create_connection( ) as conn:
									cursor = conn.execute( f'UPDATE "{table}" SET {set_clause} '
									                       f'WHERE rowid=?;', parameters )
									conn.commit( )
								if cursor.rowcount == 0:
									st.warning( 'Row ID not found. No row was updated.' )
								else:
									if not database_record_matches( table, int( rowid ), persisted_values ):
										raise RuntimeError( 'Updated row could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row {int( rowid )} updated '
									                        f'successfully in "{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to update row: {ex}' )
					
					with btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_update_reset_button', on_click=reset_session_prefixes, args=(
								(f'crud_update_value_{table}_{int( rowid )}_',),) )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				set_blue_divider( )
				st.markdown( '##### ❌ Remove Record' )
				with st.expander( label='Delete', expanded=True ):
					delete_c1, delete_c2, delete_c3, delete_c4 = st.columns( 4 )
					with delete_c1:
						delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1, key='crud_delete_database_rowid' )
					
					st.divider( )
					
					delete_btn_c1, delete_btn_c2 = st.columns( 2 )
					with delete_btn_c1:
						if st.button( label='Delete Row', icon='❌', width='stretch', key='crud_delete_submit_button' ):
							with create_connection( ) as conn:
								cursor = conn.execute( f'DELETE FROM "{table}" WHERE rowid=?;', (
										int( delete_id ),) )
								conn.commit( )
							if cursor.rowcount == 0:
								st.warning( 'Row ID not found. No row was deleted.' )
							else:
								if read_database_record( table, int( delete_id ) ) is not None:
									st.error( 'Deleted row could not be verified.' )
								else:
									refresh_crud_database_state( table )
									queue_database_success( f'Row {int( delete_id )} deleted '
									                        f'successfully from "{table}".' )
									st.rerun( )
					with delete_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_delete_reset_button', on_click=reset_session_keys, args=(
								[ 'crud_delete_database_rowid' ],) )
				
				set_blue_divider( )
				
				# -------------------------------------------------------------------------------------
				# EDIT
				# -------------------------------------------------------------------------------------
				st.markdown( '##### 🛠️ Edit Data' )
				with st.expander( label='Update', expanded=True ):
					if df_database_rows.empty:
						st.info( 'The selected table does not contain any records.' )
					else:
						index_c1, index_c2 = st.columns( [ 0.20, 0.80 ] )
						with index_c1:
							max_row_position = len( df_database_rows ) - 1
							default_row_position = int( st.session_state.get( 'crud_record_row_position', 0 ) )
							default_row_position = max( 0, min( default_row_position, max_row_position ) )
							row_position = st.number_input( 'Select Row Position', min_value=0, max_value=max_row_position, value=default_row_position, step=1, key='crud_record_row_position' )
						selected_row = df_database_rows.iloc[ int( row_position ) ]
						selected_rowid = int( selected_row[ '__mathy_rowid__' ] )
						with index_c2:
							st.caption( f'SQLite Row ID: {selected_rowid}' )
						
						current_record = { column: selected_row[ column ] for column in
								df_database.columns }
						editable_record_columns = [ column for column in current_record if
								column not in primary_key_column_names ]
						updated_values: Dict[ str, object ] = { }
						with st.form( f'crud_record_edit_form_{table}_{selected_rowid}' ):
							record_columns = st.columns( 4 )
							for index, column in enumerate( df_database.columns ):
								target_column = record_columns[ index % 4 ]
								with target_column:
									current_value = current_record[ column ]
									updated_values[
										column ] = st.text_input( column, value='' if pd.isna( current_value ) else str( current_value ), key=f'crud_record_value_{table}_{selected_rowid}_{index}', disabled=column in primary_key_column_names, help='Primary-key fields are read-only.' if column in primary_key_column_names else None )
							
							st.divider( )
							
							record_btn_c1, record_btn_c2 = st.columns( 2 )
							with record_btn_c1:
								apply_record = st.form_submit_button( label='Apply Row Update', icon='✔️', width='stretch', disabled=not editable_record_columns )
							
							with record_btn_c2:
								st.form_submit_button( label='Reset', icon='🔄', width='stretch', on_click=reset_session_prefixes, args=(
										(f'crud_record_value_{table}_{selected_rowid}_',),) )
						
						if apply_record:
							try:
								persisted_values = {
										column: coerce_sqlite_value( value, type_map[ column ] ) for
										column, value in updated_values.items( ) if
										column not in primary_key_column_names }
								if not persisted_values:
									raise ValueError( 'No editable non-primary-key fields are '
									                  'available.' )
								set_clause = ', '.join(
									f'"{column}"=?' for column in persisted_values )
								with create_connection( ) as conn:
									cursor = conn.execute( f'UPDATE "{table}" SET {set_clause} '
									                       f'WHERE rowid=?;', list( persisted_values.values( ) ) + [
											selected_rowid ] )
									conn.commit( )
								if cursor.rowcount == 0:
									st.warning( 'The selected record no longer exists.' )
								else:
									if not database_record_matches( table, selected_rowid, persisted_values ):
										raise RuntimeError( 'Updated record could not be '
										                    'verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row position {int( row_position )} '
									                        f'updated successfully in "{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to update record: {ex}' )
				
				set_blue_divider( )
			
			# -------------------------------------------------------------------------------------
			# COLUMN CRUD
			# -------------------------------------------------------------------------------------
			st.markdown( '##### 🏗️ Edit Schema' )
			with st.expander( label='Rename Table', expanded=True ):
				protected_tables = { 'chat_history', 'embeddings', 'Prompts' }
				is_protected_table = table in protected_tables
				table_mgmt_c1, table_mgmt_c2 = st.columns( 2, border=True )
				with table_mgmt_c1:
					new_table_name = st.text_input( 'New Table Name', key='crud_table_manage_new_name', disabled=is_protected_table )
					if is_protected_table:
						st.info( 'This application infrastructure table is protected from '
						         'schema-level table changes.' )
					rename_table_c1, rename_table_c2 = st.columns( 2 )
					with rename_table_c1:
						if st.button( label='Rename Table', icon='✔️', width='stretch', key='crud_table_manage_rename_submit', disabled=is_protected_table ):
							try:
								throw_if( 'new_table_name', new_table_name )
								safe_table_name = create_identifier( new_table_name )
								if safe_table_name == table:
									st.info( 'The selected table already uses this name.' )
								elif safe_table_name.lower( ) in { name.lower( ) for name in
										list_tables( ) }:
									st.error( f'Table "{safe_table_name}" already exists.' )
								else:
									was_active = st.session_state.get( 'active_database_table', '' ) == table
									rename_table( table, safe_table_name )
									current_tables = list_tables( )
									if table in current_tables or safe_table_name not in current_tables:
										raise RuntimeError( 'Renamed table could not be '
										                    'verified.' )
									reconcile_import_table_metadata( table, safe_table_name )
									st.session_state[ 'crud_next_table' ] = safe_table_name
									if was_active:
										activate_database_table( safe_table_name )
									queue_database_success( f'Table "{table}" was renamed to "{safe_table_name}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to rename table: {ex}' )
					with rename_table_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_table_manage_rename_reset', on_click=reset_session_keys, args=(
								[ 'crud_table_manage_new_name' ],) )
				
				with table_mgmt_c2:
					confirm_delete = st.checkbox( 'Confirm permanent table deletion', value=False, key='crud_table_manage_delete_confirm', disabled=is_protected_table )
					if is_protected_table:
						st.info( 'This application infrastructure table is protected from '
						         'deletion.' )
					delete_table_c1, delete_table_c2 = st.columns( 2 )
					with delete_table_c1:
						if st.button( label='Delete Table', icon='❌', width='stretch', key='crud_table_manage_delete_submit', disabled=is_protected_table or not confirm_delete ):
							try:
								was_active = st.session_state.get( 'active_database_table', '' ) == table
								drop_table( table )
								if table in list_tables( ):
									raise RuntimeError( 'Deleted table could not be verified.' )
								reconcile_import_table_metadata( table )
								remaining_tables = [ name for name in list_tables( ) if
										not name.startswith( 'sqlite_' ) and name not in protected_tables ]
								replacement_table = remaining_tables[
									0 ] if remaining_tables else ''
								if was_active:
									if replacement_table:
										activate_database_table( replacement_table )
									else:
										clear_active_database_table( )
								if replacement_table:
									st.session_state[ 'crud_next_table' ] = replacement_table
								message = f'Table "{table}" was deleted.'
								if was_active and replacement_table:
									message += f' "{replacement_table}" is now the active table.'
								queue_database_success( message )
								st.rerun( )
							except Exception as ex:
								st.error( f'Unable to delete table: {ex}' )
					with delete_table_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_table_manage_delete_reset', on_click=reset_session_keys, args=(
								[ 'crud_table_manage_delete_confirm' ],) )
			
			with st.expander( label='Drop Columns', expanded=True ):
				label_c1, label_c2 = st.columns( 2, border=True )
				with label_c1:
					drop_columns = st.multiselect( 'Columns to Drop', df_database.columns.tolist( ), key='crud_schema_drop_columns' )
					drop_btn_c1, drop_btn_c2 = st.columns( 2 )
					with drop_btn_c1:
						if st.button( label='Drop', icon='❌', width='stretch', key='crud_schema_drop_submit_button' ):
							if not drop_columns:
								st.info( 'Select one or more columns to drop.' )
							elif len( drop_columns ) == len( df_database.columns ):
								st.error( 'Cannot Drop All Columns.' )
							else:
								try:
									for column in drop_columns:
										drop_column( table, column )
									remaining_columns = [ str( item[ 1 ] ) for item in
											create_schema( table ) ]
									if any( column in remaining_columns for column in
											drop_columns ):
										raise RuntimeError( 'Dropped column could not be '
										                    'verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Selected column(s) dropped'
									                        f' successfully from "{table}".' )
									st.rerun( )
								except Exception as ex:
									st.error( f'Unable to drop column: {ex}' )
					
					with drop_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_schema_drop_reset_button', on_click=reset_session_keys, args=(
								[ 'crud_schema_drop_columns' ],) )
				
				with label_c2:
					rename_target = st.selectbox( 'Rename Column', [ '<None>' ] + df_database.columns.tolist( ), key='crud_schema_rename_column' )
					
					new_column_name = st.text_input( 'New Column Name', key='crud_schema_rename_new_column_name' )
					
					rename_btn_c1, rename_btn_c2 = st.columns( 2 )
					with rename_btn_c1:
						if st.button( label='Rename', icon='✔️', width='stretch', key='crud_schema_rename_submit_button' ):
							if rename_target == '<None>' or not new_column_name:
								st.info( 'Select a column and enter a new name.' )
							elif new_column_name in df_database.columns:
								st.error( 'Column Name Already Exists.' )
							else:
								try:
									rename_column( table, rename_target, new_column_name )
									updated_names = [ str( item[ 1 ] ) for item in
											create_schema( table ) ]
									if new_column_name not in updated_names or rename_target in updated_names:
										raise RuntimeError( 'Renamed column could not be '
										                    'verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Column "{rename_target}" was renamed to "{new_column_name}" in "{table}".' )
									st.rerun( )
								except Exception as ex:
									st.error( f'Unable to rename column: {ex}' )
					
					with rename_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch', key='crud_schema_rename_reset_button', on_click=reset_session_keys, args=(
								[ 'crud_schema_rename_column',
										'crud_schema_rename_new_column_name' ],) )
			
			st.markdown( '##### 📝 Define Columns' )
			with st.expander( label='Data Types', expanded=True ):
				type_c1, type_c2, type_c3 = st.columns( 3 )
				with type_c1:
					type_column = st.selectbox( 'Column', df_database.columns.tolist( ), key='crud_schema_type_column' )
				current_type = type_map.get( type_column, 'TEXT' )
				with type_c2:
					st.text_input( 'Current Data Type', value=current_type, disabled=True, key=f'crud_schema_type_current_{table}_{type_column}' )
				supported_types = [ 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' ]
				default_type = current_type if current_type in supported_types else 'TEXT'
				with type_c3:
					new_type = st.selectbox( 'New Data Type', supported_types, index=supported_types.index( default_type ), key='crud_schema_type_new' )
				
				st.divider( )
				
				type_btn_c1, type_btn_c2 = st.columns( 2 )
				with type_btn_c1:
					if st.button( label='Define Type', icon='✔️', width='stretch', key='crud_schema_type_submit_button' ):
						if new_type == current_type:
							st.info( 'The selected column already uses this data type.' )
						else:
							try:
								change_column_type( table, type_column, new_type )
								updated_type_map = {
										str( item[ 1 ] ): str( item[ 2 ] or 'TEXT' ).upper( ) for
										item in create_schema( table ) }
								if updated_type_map.get( type_column, '' ) != new_type:
									raise RuntimeError( 'Updated column type could not be '
									                    'verified.' )
								refresh_crud_database_state( table )
								queue_database_success( f'Column "{type_column}" change'
								                        f'd to {new_type} in "{table}".' )
								st.rerun( )
							except Exception as ex:
								st.error( f'Unable to change data type: {ex}' )
				
				with type_btn_c2:
					st.button( label='Reset', icon='🔄', width='stretch', key='crud_schema_type_reset_button', on_click=reset_session_keys, args=(
							[ 'crud_schema_type_column', 'crud_schema_type_new' ],) )
			
			# -------------------------------------------------------------------------------------
			# DATABASE TABLE CRUD
			# -------------------------------------------------------------------------------------
			st.markdown( '##### 🆕 Create Table' )
			with st.expander( label='Create Data', expanded=True ):
				create_c1, create_c2 = st.columns( 2, border=True )
				with create_c1:
					crud_create_table_name = st.text_input( 'Table Name', key='crud_create_table_name', help='Enter the name of the new SQLite '
					                                                                                         'table.' )
				with create_c2:
					crud_create_column_count = st.slider( 'Number of Columns', min_value=1, max_value=20, value=5, step=1, key='crud_create_column_count', help='Number of user-defined columns. Mathy adds an Id primary key '
					                                                                                                                                            'automatically.' )
				
				st.caption( 'Mathy automatically adds: Id INTEGER PRIMARY KEY AUTOINCREMENT.' )
				crud_create_columns: List[ Dict[ str, str ] ] = [ ]
				crud_create_supported_types = [ 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' ]
				for row_start in range( 0, int( crud_create_column_count ), 5 ):
					schema_columns = st.columns( 5, border=True )
					row_end = min( row_start + 5, int( crud_create_column_count ) )
					for column_index in range( row_start, row_end ):
						with schema_columns[ column_index - row_start ]:
							st.markdown( f'**Column {column_index + 1}**' )
							column_name = st.text_input( 'Column Name', key=f'crud_create_column_name_{column_index}' )
							column_type = st.selectbox( 'Data Type', crud_create_supported_types, key=f'crud_create_column_type_{column_index}' )
							initial_value = st.text_input( 'Initial Value (optional)', key=f'crud_create_initial_value_{column_index}' )
							crud_create_columns.append( { 'name': column_name, 'type': column_type,
									'initial_value': initial_value } )
				
				st.divider( )
				create_btn_c1, create_btn_c2 = st.columns( 2 )
				with create_btn_c1:
					if st.button( label='Create Table', icon='➕', width='stretch', key='crud_create_table_submit' ):
						try:
							created_table, inserted_initial_row = create_crud_database_table( crud_create_table_name, crud_create_columns )
							st.session_state[ 'crud_next_table' ] = created_table
							activate_database_table( created_table )
							message = f'Table "{created_table}" was created successfully.'
							if inserted_initial_row:
								message += ' One initial row was inserted.'
							queue_database_success( message )
							st.rerun( )
						except Exception as ex:
							st.error( f'Unable to create table: {ex}' )
				with create_btn_c2:
					st.button( label='Reset', icon='🔄', width='stretch', key='crud_create_table_reset', on_click=reset_session_prefixes, args=(
							('crud_create_table_name', 'crud_create_column_count',
									'crud_create_column_', 'crud_create_initial_value_'),) )

# ============================================
# DATA FILTER MODE
# ============================================
elif mode == 'Data Filter':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Data Filter' )
		st.divider( )
		tables = list_tables( )
		if tables:
			explore_c1, explore_c2, explore_c3 = st.columns( 3, border=True )
			with explore_c1:
				table = st.selectbox( 'Table', tables, key='explore_table' )
			
			with explore_c2:
				page = st.number_input( 'Page', min_value=1, step=1 )
			
			with explore_c3:
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
			
			set_blue_divider( )
			offset = (page - 1) * page_size
			df_page = read_table( table, page_size, offset )
			render_data_editor( df_page, use_container_width=True, height=400 )

# ============================================
# DATA AGGREGATION MODE
# ============================================
elif mode == 'Data Aggregation':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with (center):
		st.subheader( 'Data Aggregation' )
		st.divider( )
		tables = list_tables( )
		st.session_state.get( 'aggregation', None )
		if tables:
			agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( 4, border=True )
			with agg_c1:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				declared_schema = create_declared_schema_map( create_schema( table ) )
				aggregation_profiles = profile_dataframe_schema( df, declared_schema )
			with agg_c2:
				numeric_columns = [ column for column, profile in aggregation_profiles.items( ) if
						profile[ 'analytical_role' ] == 'numeric' ]
				col = st.selectbox( 'Column', numeric_columns, key='col_box' ) if numeric_columns else None
				if not numeric_columns:
					st.info( 'No numeric columns are available.' )
			with agg_c3:
				aggregation = st.selectbox( 'Function', [ 'SUM', 'AVG',
						'COUNT' ], key='agg_box', disabled=col is None )
			with agg_c4:
				if col is not None:
					numeric_series = pd.to_numeric( df[ col ], errors='coerce' )
					if aggregation == 'SUM':
						st.metric( 'Result', numeric_series.sum( ) )
					elif aggregation == 'AVG':
						st.metric( 'Result', numeric_series.mean( ) )
					elif aggregation == 'COUNT':
						st.metric( 'Result', numeric_series.count( ) )

# ============================================
# SQL CONSOLE MODE
# ============================================
elif mode == 'SQL Console':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'SQL Console' )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Session State
		# ------------------------------------------------------------------
		if 'sql_console_query' not in st.session_state:
			st.session_state[ 'sql_console_query' ] = ''
		
		if 'sql_console_result' not in st.session_state:
			st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
		
		if 'sql_console_elapsed' not in st.session_state:
			st.session_state[ 'sql_console_elapsed' ] = 0.0
		
		if 'sql_console_executed' not in st.session_state:
			st.session_state[ 'sql_console_executed' ] = False
		
		if 'sql_console_table_name' not in st.session_state:
			st.session_state[ 'sql_console_table_name' ] = ''
		
		if 'sql_console_last_saved_table' not in st.session_state:
			st.session_state[ 'sql_console_last_saved_table' ] = None
		
		# ------------------------------------------------------------------
		# Action State
		# ------------------------------------------------------------------
		table_name = st.session_state[ 'sql_console_table_name' ]
		last_saved_table = st.session_state[ 'sql_console_last_saved_table' ]
		save_table = False
		undo_save = False
		
		def reset_sql_console( ) -> None:
			"""Reset SQL Console query and execution state.

			Purpose:
			    Clears the active SQL statement, stored query result, elapsed execution time,
			    execution-status flag, and destination table name so the SQL Console returns to
			    its initial empty state.

			Returns:
			    None: This function updates SQL Console values in Streamlit session state.
			"""
			st.session_state[ 'sql_console_query' ] = ''
			st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
			st.session_state[ 'sql_console_elapsed' ] = 0.0
			st.session_state[ 'sql_console_executed' ] = False
			st.session_state[ 'sql_console_table_name' ] = ''
		
		# ------------------------------------------------------------------
		# Query Editor
		# ------------------------------------------------------------------
		query = st.text_area( 'Enter SQL Query', key='sql_console_query' )
		
		run_c1, run_c2 = st.columns( 2 )
		with run_c1:
			run_query = st.button( 'Run Query', icon='▶️', use_container_width=True, key='sql_console_run_query' )
		
		with run_c2:
			st.button( 'Clear Query', icon='🔄', use_container_width=True, key='sql_console_clear_query', on_click=reset_sql_console )
		
		# ------------------------------------------------------------------
		# Query Execution
		# ------------------------------------------------------------------
		if run_query:
			if not is_safe_query( query ):
				st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
			else:
				try:
					start_time = time.perf_counter( )
					
					with create_connection( ) as conn:
						df_result = pd.read_sql_query( query, conn )
					
					end_time = time.perf_counter( )
					elapsed = end_time - start_time
					
					st.session_state[ 'sql_console_result' ] = df_result.copy( )
					st.session_state[ 'sql_console_elapsed' ] = elapsed
					st.session_state[ 'sql_console_executed' ] = True
					st.session_state[ 'sql_console_table_name' ] = ''
				
				except Exception as e:
					st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
					st.session_state[ 'sql_console_elapsed' ] = 0.0
					st.session_state[ 'sql_console_executed' ] = False
					st.session_state[ 'sql_console_table_name' ] = ''
					st.error( f'Execution failed: {e}' )
		
		# ------------------------------------------------------------------
		# Query Results
		# ------------------------------------------------------------------
		if st.session_state[ 'sql_console_executed' ]:
			df_result_source = st.session_state[ 'sql_console_result' ].copy( )
			result_profiles = profile_dataframe_schema( df_result_source )
			df_result = create_typed_analysis_dataframe( df_result_source, result_profiles )
			elapsed = float( st.session_state[ 'sql_console_elapsed' ] )
			
			set_blue_divider( )
			st.markdown( '##### Results' )
			render_data_editor( df_result, use_container_width=True, disabled=True, key='sql_console_result_editor' )
			
			row_count = len( df_result )
			column_count = len( df_result.columns )
			
			numeric_columns = [ column for column, profile in result_profiles.items( ) if
					profile[ 'analytical_role' ] == 'numeric' ]
			datetime_columns = [ column for column, profile in result_profiles.items( ) if
					profile[ 'analytical_role' ] == 'datetime' ]
			text_columns = [ column for column in df_result.columns if
					column not in numeric_columns and column not in datetime_columns ]
			
			# --------------------------------------------------------------
			# Execution and Schema Metrics
			# --------------------------------------------------------------
			set_blue_divider( )
			st.markdown( '##### Query Metrics' )
			
			col1, col2, col3, col4, col5, col6 = st.columns( 6, border=True )
			col1.metric( 'Rows Returned', f'{row_count:,}' )
			col2.metric( 'Columns Returned', f'{column_count:,}' )
			col3.metric( 'Numeric Columns', f'{len( numeric_columns ):,}' )
			col4.metric( 'Datetime Columns', f'{len( datetime_columns ):,}' )
			col5.metric( 'Text / Other', f'{len( text_columns ):,}' )
			col6.metric( 'Execution Time', f'{elapsed:.6f}' )
			
			# --------------------------------------------------------------
			# Result Schema
			# --------------------------------------------------------------
			set_blue_divider( )
			st.markdown( '##### Result Schema' )
			
			schema_rows: list[ dict[ str, object ] ] = [ ]
			for column in df_result.columns:
				series = df_result[ column ]
				null_count = int( series.isna( ).sum( ) )
				non_null_count = int( series.notna( ).sum( ) )
				distinct_count = int( series.nunique( dropna=True ) )
				
				schema_rows.append( { 'Column': column, 'Data Type': str( series.dtype ),
						'Non-Null': non_null_count, 'Null': null_count,
						'Distinct': distinct_count } )
			
			df_schema = pd.DataFrame( schema_rows, columns=[ 'Column', 'Data Type', 'Non-Null',
					'Null', 'Distinct' ] )
			
			render_data_editor( df_schema, use_container_width=True, hide_index=True, disabled=True, column_config={
					'Column': st.column_config.TextColumn( 'Column', width='large' ),
					'Data Type': st.column_config.TextColumn( 'Data Type', width='medium' ),
					'Non-Null': st.column_config.NumberColumn( 'Non-Null', format='%d' ),
					'Null': st.column_config.NumberColumn( 'Null', format='%d' ),
					'Distinct': st.column_config.NumberColumn( 'Distinct', format='%d' ) }, key='sql_console_schema_editor' )
			
			# --------------------------------------------------------------
			# Slow Query Warning
			# --------------------------------------------------------------
			if elapsed > 2.0:
				st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
			
			# --------------------------------------------------------------
			# Save, Export, and Undo Results
			# --------------------------------------------------------------
			set_blue_divider( )
			st.markdown( '##### Save / Export Results' )
			
			last_saved_table = st.session_state.get( 'sql_console_last_saved_table', None )
			
			left_container, right_container = st.columns( [ 0.50, 0.50 ] )
			
			with left_container:
				with st.container( border=True, height=87 ):
					table_name = st.text_input( 'New Table Name', key='sql_console_table_name', help='Creates a new SQLite table from the current query result.' )
			
			with right_container:
				with st.container( border=True, height=87 ):
					st.write( '' )
					
					action_c1, action_c2, action_c3 = st.columns( [ 0.34, 0.33,
							0.33 ], border=False )
					
					with action_c1:
						save_table = st.button( 'Save as New Table', icon='💾', use_container_width=True, key='sql_console_save_table' )
					
					with action_c2:
						csv = df_result.to_csv( index=False ).encode( 'utf-8' )
						
						st.download_button( 'Download CSV', csv, 'query_results.csv', 'text/csv', icon='📥', use_container_width=True, key='sql_console_download', disabled=df_result.empty )
					
					with action_c3:
						undo_save = st.button( 'Undo Last Save', icon='↩️', use_container_width=True, key='sql_console_undo_save', disabled=last_saved_table is None, help=(
								f'Deletes the last table saved by SQL Console: '
								f'{last_saved_table}' if last_saved_table else 'No table has been '
								                                               'saved during this '
								                                               'session.') )
			
			# --------------------------------------------------------------
			# Save Result to Database
			# --------------------------------------------------------------
			if save_table:
				if not table_name or not table_name.strip( ):
					st.error( 'Enter a table name.' )
				
				elif len( df_result.columns ) == 0:
					st.error( 'The query result does not contain any columns to save.' )
				
				elif df_result.columns.duplicated( ).any( ):
					duplicate_columns = df_result.columns[
						df_result.columns.duplicated( ) ].tolist( )
					
					st.error( f'The query result contains duplicate column names: '
					          f'{duplicate_columns}. '
					          f'Use SQL aliases to make each result column unique.' )
				
				else:
					try:
						safe_table_name = create_identifier( table_name )
						existing_tables = { existing_table.lower( ): existing_table for
								existing_table in list_tables( ) }
						
						if safe_table_name.lower( ) in existing_tables:
							st.error( f'Table "{existing_tables[ safe_table_name.lower( ) ]}" '
							          f'already '
							          f'exists. Enter a different table name.' )
						else:
							with create_connection( ) as conn:
								df_result.to_sql( safe_table_name, conn, if_exists='fail', index=False )
							
							st.session_state[ 'sql_console_last_saved_table' ] = safe_table_name
							
							queue_database_success( f'Query results saved as table "'
							                        f'{safe_table_name}" with '
							                        f'{row_count:,} row(s) and {column_count:, } column( s ).' )
							
							if safe_table_name != table_name.strip( ):
								st.info( f'The table name was normalized to "{safe_table_name}".' )
							
							st.rerun( )
					
					except Exception as e:
						st.error( f'Unable to save query results: {e}' )
		
		# --------------------------------------------------------------
		# Undo Last Table Save
		# --------------------------------------------------------------
		if undo_save:
			try:
				existing_tables = { existing_table.lower( ): existing_table for existing_table in
						list_tables( ) }
				
				tracked_table = existing_tables.get( last_saved_table.lower( ), None )
				
				if tracked_table is None:
					st.session_state[ 'sql_console_last_saved_table' ] = None
					
					st.warning( f'Table "{last_saved_table}" no longer exists in the '
					            f'database.' )
				
				else:
					drop_table( tracked_table )
					
					st.session_state[ 'sql_console_last_saved_table' ] = None
					
					queue_database_success( f'Table "{tracked_table}" was deleted. '
					                        f'The query and displayed results were '
					                        f'preserved.' )
				
				st.rerun( )
			
			except Exception as e:
				st.error( f'Unable to undo the last table save: {e}' )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
st.markdown( """
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""", unsafe_allow_html=True, )

# ---- Fixed Container
st.markdown( """
	<style>
	.foo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(27, 27, 27, 0.95);
		border-top: 1px solid #5A5A5A;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #0078FC;
		z-index: 1000;
	}
	.foo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""", unsafe_allow_html=True, )

# ---- Rendering Method
st.markdown( f"""
    <div class="foo-status-bar">
        <div class="foo-status-inner">
            <span> </span>
            <span> </span>
        </div>
    </div>
    """, unsafe_allow_html=True, )
