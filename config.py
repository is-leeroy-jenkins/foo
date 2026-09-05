'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                config.py
      Author:                  Terry Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2024  Terry Eppler

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
    config.py
  </summary>
  ******************************************************************************************
  '''
import os
from pathlib import Path

# -------------- APP-LEVEL UTILITIES -------------

def throw_if( name: str, value: object ) -> None:
	"""Raise ``ValueError`` when a required value is empty.

	Purpose:
		Provide a small, consistent guard for required arguments and configuration values. The
		function treats falsy values as invalid and raises a ``ValueError`` containing the
		caller-supplied argument or setting name.

	Args:
		name (str): Name of the argument or configuration value being validated.
		value (object): Value to validate.

	Returns:
		None.

	Raises:
		ValueError: Raised when ``value`` is falsy.
	"""
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def get_bool( name: str, default: bool = False ) -> bool:
	"""Read a Boolean environment variable using Fiddy's true-value convention.

	Purpose:
		Convert environment-variable text into a deterministic Boolean value. Missing variables
		return the caller-provided default. Values of ``1``, ``true``, ``yes``, ``y``, and
		``on`` are treated as ``True``; all other defined values are treated as ``False``.

	Args:
		name (str): Environment variable name.
		default (bool): Default value used when the environment variable is not defined.

	Returns:
		bool: Parsed Boolean value. If parsing fails, the original ``default`` value is returned.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value is None else value.strip( ).lower( ) in (
				'1',
				'true',
				'yes',
				'y',
				'on'
		)
	except Exception:
		return default

def get_int( name: str, default: int ) -> int:
	"""Read an integer environment variable with a deterministic fallback.

	Purpose:
		Parse an optional environment variable as an integer while preserving a safe default when
		the variable is missing, empty, or invalid.

	Args:
		name (str): Environment variable name.
		default (int): Default integer value used when parsing is not possible.

	Returns:
		int: Parsed integer value or the supplied default value.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else int( str( value ).strip( ) )
	except Exception:
		return default

def get_float( name: str, default: float ) -> float:
	"""Read a floating-point environment variable with a deterministic fallback.

	Purpose:
		Parse an optional environment variable as a float while preserving a safe default when the
		variable is missing, empty, or invalid.

	Args:
		name (str): Environment variable name.
		default (float): Default floating-point value used when parsing is not possible.

	Returns:
		float: Parsed floating-point value or the supplied default value.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else float( str( value ).strip( ) )
	except Exception:
		return default

def get_path( name: str, default: Path ) -> Path:
	"""Read a path environment variable and return a resolved ``Path``.

	Purpose:
		Resolve optional filesystem configuration from the environment. Missing variables return
		the resolved default path. Invalid values return the resolved default path rather than
		interrupting module import.

	Args:
		name (str): Environment variable name.
		default (Path): Default path used when the environment variable is not defined.

	Returns:
		Path: Resolved path value or resolved default path.
	"""
	try:
		throw_if( 'name', name )
		throw_if( 'default', default )
		value = os.getenv( name )
		return Path( value ).resolve( ) if value else default.resolve( )
	except Exception:
		return default.resolve( )

def get_text( name: str, default: str ) -> str:
	"""Read a text environment variable with a deterministic fallback.

	Purpose:
		Return an environment variable as text while preserving the supplied default when the
		variable is missing or empty.

	Args:
		name (str): Environment variable name.
		default (str): Default text value.

	Returns:
		str: Environment value or supplied default.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else str( value )
	except Exception:
		return default

# ------ CONSTANTS  -------------------

BASE_DIR = Path( __file__ ).resolve( ).parent
ROOT_DIR = Path( __file__ ).resolve( ).parent
LOG_DIR = get_path( 'LOG_DIR', ROOT_DIR / 'logging' )
LOG_PATH = get_text( 'LOG_PATH', str( LOG_DIR / 'Exceptions.db' ) )
LOG_FILE = get_text( 'LOG_FILE', 'Exceptions' )

# ------ ENVIRONMENT API KEYS  -------------------
ACCESS_DRIVER = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
AIRNOW_API_KEY = os.getenv( 'AIRNOW_API_KEY' )
CLAUDE_API_KEY = os.getenv( 'CLAUDE_API_KEY' )
CONGRESS_API_KEY = os.getenv( 'CONGRESS_API_KEY' )
CHROMA_API_KEY = os.getenv( 'CHROMA_API_KEY' )
CHROMA_TENET_ID = os.getenv( 'CHROMA_TENET_ID' )
GEOAPIFY_API_KEY = os.getenv( 'GEOAPIFY_API_KEY' )
GEOCODING_API_KEY = os.getenv( 'GEOCODING_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT_ID' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOVINFO_API_KEY = os.getenv( 'GOVINFO_API_KEY' )
GOOGLE_GENAI_USE_VERTEXAI = os.getenv( 'GOOGLE_GENAI_USE_VERTEXAI' )
GOOGLE_WEATHER_API_KEY = os.getenv( 'GOOGLE_WEATHER_API_KEY' )
GOOGLE_ACCOUNT_FILE = os.getenv( 'GOOGLE_ACCOUNT_CREDENTIALS' )
GOOGLE_DRIVE_TOKEN_PATH = os.getenv( 'GOOGLE_DRIVE_TOKEN_PATH' )
GOOGLE_DRIVE_FOLDER_ID = os.getenv( 'GOOGLE_DRIVE_FOLDER_ID' )
HUGGINGFACE_API_KEY = os.getenv( 'HUGGINGFACE_API_KEY' )
IPINFO_API_KEY = os.getenv( 'IPINFO_API_KEY' )
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
PINECONE_API_KEY = os.getenv( 'PINECONE_API_KEY' )
LANGSMITH_API_KEY = os.getenv( 'LANGSMITH_API_KEY' )
LLAMAINDEX_API_KEY = os.getenv( 'LLAMAINDEX_API_KEY' )
LLAMACLOUD_API_KEY = os.getenv( 'LLAMACLOUD_API_KEY' )
MISTRAL_API_KEY = os.getenv( 'MISTRAL_API_KEY' )
NASA_API_KEY = os.getenv( 'NASA_API_KEY' )
NASA_EARTHDATA_TOKEN = os.getenv( 'NASA_EARTHDATA_TOKEN' )
NEWS_API_KEY = os.getenv( 'NEWSAPI_API_KEY' )
THENEWS_API_KEY = os.getenv( 'THENEWSAPI_API_KEY' )
WEATHERAPI_API_KEY = os.getenv( 'WEATHERAPI_API_KEY' )
XAI_API_KEY = os.getenv( 'XAI_API_KEY' )
O365_CLIENT_ID = os.getenv( 'O365_CLIENT_ID' )
O365_CLIENT_SECRET = os.getenv( 'O365_CLIENT_SECRET' )
OPENAQ_API_KEY = os.getenv( 'OPENAQ_API_KEY' )
OPENSKY_API_CLIENT_ID = os.getenv( 'OPENSKY_API_CLIENT_ID' )
OPENSKY_API_CREDENTIALS = os.getenv( 'OPENSKY_API_CREDENTIALS' )
OPENSKY_API_CLIENT_SECRET = os.getenv( 'OPENSKY_API_CLIENT_ID' )
CENSUS_API_KEY = os.getenv( 'CENSUS_API_KEY' )
SOCRATA_API_KEY = os.getenv( 'SOCRATA_API_KEY' )
HEALTHDATA_API_KEY = os.getenv( 'HEALTHDATA_API_KEY' )
USGS_WATERDATA_API_KEY = os.getenv( 'USGS_API_KEY' )
DATA_GOV_API_KEY = os.getenv( 'DATAGOV_API_KEY' )
GOVINFO_API_KEY = os.getenv( 'GOVINFO_API_KEY' )
FIRMS_MAP_KEY = os.getenv( 'FIRMS_MAP_KEY' )
PURPLEAIR_API_KEY = os.getenv( 'PURPLEAIR_API_KEY' )
SKYMAP_TOKEN = os.getenv( 'SKY_MAP_TOKEN' )
 
# ---------------- CONSTANTS -----------------------
APP_TITLE = 'Foo'
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 30px 30px 0;'></div>"
SQLSERVER_DRIVER = r'DRIVER={ ODBC Driver 17 for SQL Server };SERVER=.\SQLExpress;'
DB_PATH = BASE_DIR / 'stores' / 'sqlite' / 'datamodels' / 'Data.db'
AGENTS ='''Mozilla/5.0 Windows NT 10.0; Win64; x64; AppleWebKit/537.36 (KHTML, like Gecko)
		Chrome/124.0 Safari/537.36'''
FAVICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/foo_logo.png'
DB = r'stores/sqlite/datamodels/Data.db'

MODE = [ 'Loading', 'Scraping', 'Retrieval',
		'Geospatial', 'Demographic', 'Environmntal',
		'Astronomical', 'Generation' ]

DB_MODES = [ 'Data Browse','Data Upload', 'CRUD Ops', 'Data Filter',
	'Data Aggregation', 'SQL Console' ]

MODE_MAP = \
{
		'Loading': 'Data Loading',
		'Scraping': 'Data Scraping',
		'Retrieval': 'Data Collections & Public Archives',
		'Geospatial': 'Weather & Geospatial Information',
		'Demographic': 'Health & Population Data',
		'Environmental': 'Environmental Information',
		'Astronomical': 'Physics & Astronomical Data',
		'Generation': 'AI Generation',
		'Data Browse': 'Data Browse',
		'Data Upload': 'Data Upload',
		'CRUD Ops': 'CRUD Ops',
		'Data Filter': 'Data Filter',
		'Data Aggregation': 'Data Aggregation',
		'SQL Console': 'SQL Console'
 }

CHUNKABLE_LOADERS = {
		'TextLoader': [ 'chars', 'tokens' ],
		'CsvLoader': [ 'chars' ],
		'PdfLoader': [ 'chars' ],
		'ExcelLoader': [ 'chars' ],
		'WordLoader': [ 'chars' ],
		'MarkdownLoader': [ 'chars' ],
		'HtmlLoader': [ 'chars' ],
		'JsonLoader': [ 'chars' ],
		'PowerPointLoader': [ 'chars' ],
}

REQUIRED_CORPORA = [
		'brown',
		'gutenberg',
		'reuters',
		'webtext',
		'inaugural',
		'state_union',
		'punkt',
		'stopwords',
]

SESSION_STATE_DEFAULTS = {
		# ------------ Ingestion
		'documents': None,
		'raw_documents': None,
		'active_loader': None,
		# ------------ Input
		'raw_text': None,
		'raw_tokens': None,
		'raw_text_view': None,
		# Processing
		'parser': None,
		'processed_text': None,
		'processed_text_view': None,
		# ------------ Performance
		'start_time': None,
		'end_time': None,
		'total_time': None,
		# ------------ Tokenization / Vocabulary
		'tokens': None,
		'vocabulary': None,
		'token_counts': None,
		'df_synsets': None,
		# ------------ SQLite / Excel
		'active_table': None,
		# ------------ Chunking
		'lines': None,
		'chunks': None,
		'chunk_modes': None,
		'chunked_documents': None,
		# ------------ Embeddings
		'embedder': None,
		'embeddings': None,
		'embedding_provider': None,
		'embedding_model': None,
		'embedding_source': None,
		'embedding_documents': None,
		'df_embedding_input': None,
		'df_embedding_output': None,
		# ------------ Retrieval / Search
		'search_results': None,
		# ------------ DataFrames
		'df_frequency': None,
		'df_tables': None,
		'df_schema': None,
		'df_preview': None,
		'df_count': None,
		'df_chunks': None,
		# ------------ Data
		'data_connection': None,
		# ------------ Sidebar / API Keys
		'api_keys': {
				'openai_api_key': None,
				'groq_api_key': None,
				'google_api_key': None,
				'pinecone_api_key': None,
				'google_credentials_path_api_key': None,
		},
		# ------------ XML Loader (explicit contract)
		'xml_loader': None,
		'xml_documents': None,
		'xml_split_documents': None,
		'xml_tree_loaded': None,
		'xml_namespaces': None,
		'xml_xpath_results': None,
		# ------------ WordNet Caches
		'wordnet_synsets_sig': None,
		'df_wordnet_synsets': None,
		'df_wordnet_lemmas': None,
}

# ----------------- Models

GPT_MODELS = [ 'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
               'gpt-5.1', 'gpt-5.2', 'gpt-4.1' ]

GEMINI_MODELS = [ 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                  'gemini-2.5-flash-lite' ]

GROK_MODELS = [ 'grok-4-1-fast-reasoning', 'grok-4-fast-reasoning', 'grok-4',
                'grok-code-fast-1', 'grok-3-mini', 'grok-2-image-1212' ]

CLAUDE_MODELS = [ 'claude-opus-4-6', 'claude-sonnet-4-6',
                  'claude-haiku-4-5' ]

MISTRAL_MODELS = [ 'mistral-large-latest', 'mistral-medium-latest',
                   'mistral-small-latest', 'mistral-ocr-latest'  ]

# ------------- API DEFINITIONS ------------------

ARXIV = r'''arXiv is a free distribution service and an open-access archive for nearly 2.4 million
		scholarly articles in the fields of physics, mathematics, computer science, quantitative
		biology, quantitative finance, statistics, electrical engineering and systems science, and
		economics. Materials on this site are not peer-reviewed by arXiv.
		
		https://arxiv.org/
'''

GOOGLE_DRIVE = r'''Google Drive is a secure, cloud-based storage service by Google that allows users
		to store, synchronize, and share files across computers, phones, and tablets. Offering 15GB
		of free storage, it acts as a centralized hub for documents, photos, and videos, enabling
		real-time collaboration with Google Workspace apps like Docs, Sheets, and Slides.
		
		https://workspace.google.com/products/drive/
'''

WIKIPEDIA= r'''A multilingual free online encyclopedia written and maintained by a community of
		volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing
		system called MediaWiki. Wikipedia is the largest and most-read reference work in history.
		
		https://www.wikipedia.org
'''

PUBMED = r'''The National Center for Biotechnology Information, National Library of Medicine
		comprises more than 35 million citations for biomedical literature from MEDLINE,
		life science journals, and online books. Citations may include links to full text content
		from PubMed Central and publisher web sites.
		
		https://pubmed.ncbi.nlm.nih.gov/
'''

THENEWS = r'''News Aggregators: Creating "one-stop-shop" apps or websites that compile headlines
		from multiple sources into a single feed. Market Intelligence: Tracking competitors,
		industry trends, and "sales triggers" (like mergers or funding rounds) to inform business
		decisions. Financial Analysis: Monitoring stock market news and economic shifts to assist
		in trading and risk management. Media Monitoring: Tracking brand mentions and public
		sentiment across the web to manage PR and customer feedback
'''

GOOGLE_CSE = r'''The Cse Service is the endpoint that returns the requested searches.
		You must identify a particular search engine to use in your request
		(using the cx query parameter) as well as the search query (using the q query parameter).
		In addition, you should provide a developer key (using the key query parameter).
		
		http://www.google.com/cse/
'''

GOOGLE_WEATHER = r'''The Weather API lets you request real-time, hyperlocal weather data for
		locations around the world. Weather information includes temperature, precipitation,
		humidity, and more.
		
		https://developers.google.com/maps/documentation/weather/overview
'''

US_NAVAL_OBSERVATORY = r'''Provides access to APIs from the US Naval Observatory's Celestial Navigation Data for
		Assumed Position and Time:  this data service provides all the astronomical information
		necessary to plot navigational lines of position from observations of the altitudes of
		celestial bodies.
		
		https://www.cnmoc.usff.navy.mil/usno/
'''

OPEN_SCIENCE = r'''Provides access to APIs from NASA's Open Science Data Repostitory (OSDR).
		NASA OSDR provides a RESTful Application Programming Interface (API) to its
		full-text search, data file retrieval, and metadata retrieval capabilities.
		The API provides a choice of standard web output formats,
		either JavaScript Object Notation (JSON) or Hyper Text Markup Language (HTML),
		of query results.
		
		https://science.nasa.gov/biological-physical/data/osdr/
'''

GOV_INFO = r'''The GovInfo Link Service provides services for developers and webmasters to access
        content and metadata on GovInfo. Current and planned services include a link service,
        list service, and search service. The link service is used to create embedded links to
        content and metadata on GovInfo and is currently enabled for the collections below.
        The collection code is listed in parenthesis after each collection name, and the available
        queries are listed below each collection. More information about each query is provided on
        the individual collection page.
        
        https://api.govinfo.gov/docs/
'''

CONGRESS = r'''Submit queries against the Congressional Research Service's (CRS) Appropriation
		Status Table.
		
		https://api.congress.gov/
'''

INTERNET_ARCHIVE = r'''The Internet Archive is a non-profit, digital library founded in 1996 that
		provides free, public access to vast collections of digitized materials, including over 800
		billion web pages via its "Wayback Machine". It acts as a "time machine" for the web, preserving
		books, movies, music, software, and websites to provide universal access to knowledge.
		
		https://archive.org/developers/index-apis.html
'''

THE_SATELLITE_CENTER = r'''Provides access to APIs from NASA's Satellite Situation Center Web (SSCWeb) service
		that is operated jointly by the NASA/GSFC Space Physics Data Facility (SPDF) and the
		National Space Science Data Center (NSSDC) to support a range of NASA science programs
		and to fulfill key international NASA responsibilities including those of NSSDC and the
		World Data Center-A for Rockets and Satellites.
		
		https://api.nasa.gov/
'''

NEAR_BY_OBJECTS = r'''Provides access to APIs from JPL’s SSD (Solar System Dynamics) and CNEOS
		(Center for Near-Earth Object Studies) API (Application Program Interface) service.
		This service provides an interface to machine-readable data (JSON-format) related to SSD
		and CNEOS.
		
		https://ssd-api.jpl.nasa.gov/
'''

ASTRONOMY_CATALOG = r'''The Open Astronomy Catalog (OAC) API is a RESTful interface designed for
		programmatic access to open-access astronomical data, specifically focusing on transient events.
		
		https://astrocats.space/
'''

ASTRO_QUERY = r'''Access to the astropy package that contains key functionality and common tools needed for
		performing astronomy and astrophysics with Python. It is at the core of the Astropy Project,
		which aims to enable the community to develop a robust ecosystem of affiliated packages
		covering a broad range of needs for astronomical research, data processing, and data analysis.
		
		https://astroquery.readthedocs.io/en/latest/
'''

OPEN_WEATHER = r'''
		Provides forecast weather retrieval by location name using the Open-Meteo
		Geocoding API and Open-Meteo Forecast API. This class is forecast-only by design and
		intentionally excludes archive / historical date-based retrieval so it does not overlap
		with the separate Historical Weather class.
		
		https://open-meteo.com/
'''

HISTORICAL_WEATHER = r'''Provides historical weather retrieval by location name and date using the
		Open-Meteo Geocoding API and Open-Meteo Historical Weather API. This class is intentionally
		designed around the actual user-facing need in the Foo fetcher expander: enter a location
		and a date, resolve that location to coordinates, then retrieve historical weather for that date.
		
		https://open-meteo.com/en/docs/historical-weather-api
'''

NASA_EARTH_OBSERVATORY = r'''NASA Earth Observatory's Natural Event Tracker (EONET) allows users to access imagery,
				often in near real-time (NRT), of natural events such as dust storms, forest fires, and
				tropical cyclones—empowering people all across the planet to locate, track, and potentially
				prepare for and manage events that affect communities in their paths.
				Version 3 API for events, categories, sources, and layers.
				
				https://eonet.gsfc.nasa.gov/docs/v3
'''

UN_DATA = r'''The UNdata API provides programmatic access to the United Nations' global statistical
		database, allowing developers and researchers to query and retrieve data directly for use
		in applications, websites, or local processing. The core UNdata API provides access to the
		broader UNdata platform, which contains over 60 million data points from across the UN system.
		
		https://data.un.org/Host.aspx?Content=API
'''

NASA_EONET = r'''NASA Earth Observatory's Natural Event Tracker (EONET) allows users to access imagery,
		often in near real-time (NRT), of natural events such as dust storms, forest fires, and tropical
		cyclones—empowering people all across the planet to locate, track, and potentially prepare for
		and manage events that affect communities in their paths. The EONET application programming
		interface (API) provides customization of features including curation and direct links to
		image sources.
		
		https://eonet.gsfc.nasa.gov/docs/v3
'''

NASA_FIRMS = r'''ASA’s Fire Information for Resource Management System (FIRMS) provides near
		real-time, satellite-derived active fire and hotspot data (within 3 hours of observation)
		to monitor wildfires. Using sensors from MODIS and VIIRS, it offers global coverage through
		an interactive map, email alerts, and GIS data. It is designed for firefighters, scientists,
		and natural resource managers.
		
		https://firms.modaps.eosdis.nasa.gov/api/
'''

EPA_ENVIROFACTS = r'''The Envirofacts Data Warehouse contains information from select EPA Environmental
		program office databases and provides access about environmental activities that may affect air,
		water, and land anywhere in the United States.
		
		https://www.epa.gov/enviro/envirofacts-data-service-api
'''

EPA_UV_INDEX = r'''The EPA UV Index predicts daily solar UV radiation intensity on a 1–11+ scale,
		helping to gauge sun-safe precautions. Developed with the National Weather Service, it factors
		in ozone, clouds, and elevation to forecast noon intensity. A UV Alert is issued if the
		index is 6+ and unusually high.
		
		https://enviro.epa.gov/envirofacts/uv/search
'''

SPACE_WEATHER = r'''NASA DONKI (Space Weather Database Of Notifications, Knowledge, Information) is
		is a comprehensive on-line tool for space weather forecasters, scientists, and the general
		space science community. DONKI chronicles the daily interpretations of space weather observations,
		analysis, models, forecasts, and notifications provided by the Space Weather Research Center (SWRC),
		comprehensive knowledge-base search functionality to support anomaly resolution and space
		science research, intelligent linkages, relationships, cause-and-effects between space weather
		activities and comprehensive webservice API access to information stored in DONKI.
		
		https://ccmc.gsfc.nasa.gov/tools/DONKI/#donki-webservice-calls-api
'''

NEAR_BY_OBJECTS = r''''SSD (Solar System Dynamics) and CNEOS (Center for Near-Earth Object Studies)
		API (Application Program Interface) service. This service provides an interface to
		machine-readable data (JSON-format) related to SSD and CNEOS.
		
		https://api.nasa.gov/
'''

SKY_MAP = r'''Provides static and link-based star chart generation using the SKY-MAP.ORG
		XML API, Site Linker, and Image Generator interfaces.
		
		https://wikisky.org/XML_API_V1.0.html
'''

NASA_OPEN_SCIENCE = r'''NASA’s Open Science Data Repository (OSDR) enables the reuse of comprehensive,
		multi-modal space life science data—including omics, physiological, phenotypic, behavioral,
		and environmental telemetry—to advance basic and applied research as well as operational
		outcomes for human space exploration.
		
		https://science.nasa.gov/biological-physical/data/osdr/
'''

OPEN_SKY = r'''The OpenSky Network consists of a multitude of sensors connected to the Internet by
		volunteers, industrial supporters, and academic/governmental organizations. All collected
		raw data is archived in a large historical database. The database is primarily used by
		researchers from different areas to analyze and improve air traffic control technologies
		and processes. The main technologies behind the OpenSky Network are the Automatic Dependent
		Surveillance-Broadcast (ADS-B) and Mode S. These technologies provide detailed (live) aircraft
		information over the publicly accessible 1090 MHz radio frequency channel.
		
		https://opensky-network.org/data/api
'''

USGS_EARTHQUAKES = r'''The USGS Earthquake Hazards Program monitors, reports, and researches global
		seismic activity to reduce losses and save lives. It operates the National Earthquake
		Information Center (NEIC) and Advanced National Seismic System (ANSS) to detect magnitude,
		location, and impacts, providing data for public safety, engineering, and hazard assessments.
		
		https://earthquake.usgs.gov/fdsnws/event/1/
'''

USGS_WATER = r'''The USGS Water Resources Mission Area monitors, assesses, and conducts research on
		the nation's water, providing data on streamflow, groundwater, water quality, and water use.
		With over 1.9 million sites across all 50 states, they provide real-time information crucial
		for water management, safety, and economic, environmental, and recreational decisions.
		
		https://api.waterdata.usgs.gov/
'''

NOAA_TIDES_CURRENTS = r'''NOAA Tides & Currents, managed by the Center for Operational Oceanographic
		Products and Services (CO-OPS), is the authoritative U.S. source for water level, tidal, and
		oceanographic data. It offers real-time monitoring and predictions for over 3,000 stations,
		crucial for navigation, safety, and coastal resilience.
		
		https://tidesandcurrents.noaa.gov/web_services_info.html
'''

NOAA_CLIMATE_DATA = r'''NOAA climate data is provided primarily through the National Centers for
		Environmental Information (NCEI), serving as the world's largest archive of atmospheric,
		coastal, and geophysical data. It offers free access to historical weather records, 30-year
		Climate Normals, and datasets on temperature, precipitation, and storms. Data is accessible
		via the NCEI Climate Data Online (CDO) portal and NOAA Climate.gov maps and tools.
		
		https://www.ncdc.noaa.gov/cdo-web/webservices/v2
'''

USGS_NATIONAL_MAP = r'''The USGS National Map is a premier collaborative program providing free,
		accurate, and public domain geospatial data for the United States and its territories.
		It offers topographic maps, 3D elevation data, and 8+ data layers (transportation,
		hydrography, structures) via an online viewer, data downloads, and web services for public,
		academic, and government use.
		
		https://www.usgs.gov/programs/national-geospatial-program/national-map
'''

JUPYTER_NOTEBOOK = r'''Jupyter Notebook is an open-source, browser-based application used for
		interactive computing, data analysis, and visualization. It allows users to create documents
		containing live code (primarily Python), markdown text, equations, and plots in one place.
		It is widely used in data science for exploratory analysis and data cleaning
		
		https://reference.langchain.com/python/langchain-community/document_loaders/notebook/NotebookLoader
'''

AWS_BUCKET = r'''The fundamental container for data in Amazon Simple Storage Service (S3), a highly
		scalable object storage service. These are logical containers used to store data, similar to
		folders but operating on a flat structure rather than a traditional hierarchical file system.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/s3_directory/S3DirectoryLoader
'''

AWS_FILE = r'''High-performance, fully managed file system access to S3 object storage for analytics,
		machine learning, and shared workloads, delivering ~1ms latency. It bridges object storage with
		file-based tools, offering intelligent prefetching and NFS compatibility. AWS also offers
		dedicated file services like Amazon EFS, FSx for Lustre, and FSx for Windows.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/s3_file/S3FileLoader
'''


GOOGLE_BUCKET = r'''Google Cloud Storage buckets are foundational, globally unique containers holding
		data as objects (files) within Google Cloud, ranging up to 5 TB each. They offer high durability
		and availability, featuring flexible storage classes (Standard, Nearline, Coldline, Archive)
		tailored for various access frequencies, and are managed via IAM policies, console, or API.
		
		https://reference.langchain.com/python/langchain-google-community/gcs_directory/GCSDirectoryLoader
'''

GOOGLE_FILE = r'''Highly scalable, secure, and durable storage solutions, primarily using Cloud Storage
		for object storage (unstructured data) and Filestore for managed file shares (structured data).
		It allows storing massive datasets, static website hosting, and data backups, with data organized
		into buckets and accessed via APIs, featuring flexible, tiered pricing based on access frequency.
		
		https://reference.langchain.com/python/langchain-google-community/gcs_file/GCSFileLoader
'''

GOOGLE_STT = r'''Google Speech-to-Text is a Google Cloud service that converts audio into text using
		advanced AI models, supporting 125+ languages and dialects. It enables real-time streaming or
		batch processing of audio files, commonly used in voice search, customer service, and transcription.
		The service offers high accuracy, including specialized models for video, phone calls, and, as of 2026,
		the advanced Chirp 3 model, which uses self-supervised learning for better accuracy across accents and languages.
		
		https://reference.langchain.com/python/langchain-classic/document_loaders/google_speech_to_text
'''

ONEDRIVE = r'''Microsoft OneDrive is a cloud storage service that allows you to store, sync, and back up
		files (photos, documents, etc.) online, making them accessible from any device, including Windows,
		macOS, Android, and iOS. It integrates with Microsoft 365, enabling real-time collaboration,
		file sharing, and automatic, secure backup, with 5GB of free storage for new users.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/onedrive/OneDriveLoader
'''

USGS_SCIENCE = r'''USGS ScienceBase is a collaborative digital repository and information
		management platform used by the U.S. Geological Survey (USGS) to catalog, manage, and
		release scientific data. It serves as a central hub for natural science data, offering tools
		for data stewardship, discovery, and access via a searchable, open-source catalog. It is
		heavily used for releasing public data products and hosting project-specific information.
		
		https://www.usgs.gov/sciencebase-instructions-and-documentation/sciencebase-geospatial-services
'''

GROKIPEDIA = r'''Grokipedia is an AI-powered online encyclopedia launched in October 2025 by xAI,
		the artificial intelligence company. It is designed as a direct, "truth-seeking"
		competitor to Wikipedia, which Musk has frequently criticized as being biased.
		
		https://github.com/AkeBoss-tech/grokipedia-api
'''

SOCRATA = r'''Socrata is a cloud-based Data-as-a-Service (DaaS) platform  that specializes in data
		publishing and visualization for government organizations. It provides tools for open data
		portals, performance management, and data integration to make public data accessible.
		
		https://dev.socrata.com/docs/endpoints.html
'''

HEALTH_DATA = r'''Health Data API is a digital interface that allows different software systems—
		such as electronic health record (EHR) platforms, patient portals, and wearable devices—
		to communicate and exchange medical information securely. These APIs are essential for
		"interoperability," enabling a unified view of a patient’s health history across multiple
		providers and services.
		
		https://developers.google.com/health/reference/rest
'''

GLOBAL_HEALTH = r'''The Global Health Observatory (GHO) API is the World Health Organization
		(WHO) primary gateway for accessing global health statistics programmatically. It allows
		researchers and developers to retrieve data on over 1,000 health indicators across 194 Member States.
		
		https://www.who.int/data/gho/info/gho-odata-api
'''

CENSUS_DATA = r'''The Census Bureau's Application Programming Interface (API) is a free data service
		that allows developers and researchers to programmatically access over 1,600 datasets
		containing raw statistical data on the U.S. population and economy
		
		https://www.census.gov/data/developers/data-sets.html
'''

WORLD_POPULATION = r'''World Population APIs provide programmatic access to demographic data,
		including total population, age, sex, and density, often broken down by region and time.
		
		https://www.worldpop.org/sdi/introapi/
'''

WONDER = r'''The CDC WONDER API (Wide-ranging Online Data for Epidemiologic Research) is a web
		service provided by the Centers for Disease Control and Prevention that allows for automated,
		programmatic access to public health data. It enables developers and researchers to query
		various CDC WONDER online databases and retrieve data directly into their own applications,
		widgets, or analytical workflows.
		
		https://wonder.cdc.gov/wonder/help/wonder-api.html
'''

AIR_NOW = r'''AirNow is the official U.S. government website and app providing real-time,
		local air quality data and forecasts using the color-coded Air Quality Index (AQI).
		It covers ozone and particle pollution ( and  ) via a partnership of the EPA, NOAA, and
		local agencies, offering a "Fire and Smoke Map" to monitor smoke impacts
		
		https://www.airnow.gov/?
'''

PURPLE_AIR = r'''PurpleAir provides low-cost, real-time air quality monitors and a public,
		crowdsourced map to measure, visualize, and share hyper-local,, particulate matter data.
		Using laser counters, these sensors empower communities to track pollution, particularly
		during wildfire events. Data is accessible via the PurpleAir Map and is used by researchers,
		public agencies, and individuals worldwide.
		
		https://api.purpleair.com/
'''

OPEN_AQ = r'''OpenAQ is a non-profit organization that aggregates, harmonizes, and shares open-source,
		global air quality data to fight "air inequality". It provides real-time and historical
		data—primarily on PM2.5, PM10, and other pollutants—from over 48,000 locations across 150 c
		ountries. The platform empowers researchers, journalists, and communities to access, analyze,
		and use air quality data through an open API, fostering collaboration to improve public
		health and policy.
		
		https://github.com/openaq/openaq-api
'''

# ------- AI DEFINITIONS --------------------------

GROK_AI = r'''Grok is a generative artificial intelligence chatbot developed by xAI, an AI company.
		It is designed to be a "maximum truth-seeking" AI with a "rebellious streak" and a sense of humor,
		intended to compete directly with other AI systems like OpenAI's ChatGPT and Google's Gemini
'''

CHATGPT_AI = r'''ChatGPT is an AI-powered conversational chatbot developed by OpenAI, designed to
		understand and generate human-like text, code, and images. Based on Generative Pre-trained
		Transformer (GPT) technology, it is trained on massive datasets to answer questions, write
		content, and summarize information. It is accessible via web and mobile apps, with free and
		paid "Plus" subscriptions.
'''

GEMINI_AI = r'''Google Gemini is a conversational generative AI chatbot and virtual assistant developed
		by Google, formerly known as Bard. Powered by advanced Large Language Models (LLMs), it acts
		as a personal, proactive assistant that integrates across Google apps (Gmail, Docs, Drive)
		to generate text, images, and code
'''

MISTRAL_AI = r'''Mistral AI is a prominent French artificial intelligence startup founded in April 2023
		by former researchers from Meta and Google DeepMind. Based in Paris, the company has rapidly
		become a leading European AI firm, focusing on creating efficient, open-weight large language
		models (LLMs) that rival established US-based tech giants like OpenAI and Anthropic. As of
		late 2025, the company is valued at over €11.7 billion (approx. $14 billion).
'''

CLAUDE_AI = r'''Claude is an advanced AI assistant developed by Anthropic, designed to be helpful, safe,
		and honest. Known for high-quality coding, summarizing, and reasoning abilities, Claude features a
		large context window (up to 200,000 tokens) and models like Opus, Sonnet, and Haiku. It is used for
		tasks like analyzing data, drafting content, and conversational AI.
'''

# -------- LOADER DEFINITIONS -------------------

TEXT_LOADER = r'''Provides LangChain's TextLoader functionality to parse plain-text files
		into Document objects.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader
'''

NLTK_LOADER = r'''The Natural Language Toolkit (NLTK) is a comprehensive, open-source Python library
		used for symbolic and statistical Natural Language Processing (NLP). Developed originally at
		the University of Pennsylvania by Steven Bird and Edward Loper, it has become a standard tool
		in academia for teaching and research in computational linguistics.
		
		https://www.nltk.org/
'''

HTML_LOADER = r'''Provides Langchain's UnstructuredHTMLLoader's functionality to parse HTML files
		into Document objects.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/html/UnstructuredHTMLLoader
'''

WEB_CRAWLER = r'''Web fetching with optional Playwright-backed page rendering.
'''

WEB_LOADER = r'''Functionality to load all text from HTML webpages into
		a document format that can be used downstream.
'''

GITHUB_LOADER = r'''The LangChain GitHub Loader is a suite of integrations designed to ingest data
		from GitHub repositories into a format compatible with Large Language Models (LLMs).
		These loaders are primarily used in Retrieval-Augmented Generation (RAG) pipelines to
		allow AI agents to "chat" with codebases, analyze issues, or summarize pull requests.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/github/GithubFileLoader
'''

WIKIPEDIA_LOADER = r'''The LangChain Wikipedia Loader (WikipediaLoader) is a component designed to
        fetch and convert Wikipedia pages into a standardized Document format for use in LLM applications.
        
        https://reference.langchain.com/python/langchain-community/document_loaders/wikipedia/WikipediaLoader
'''

ARXIV_LOADER = r'''arXiv is a free distribution service and an open-access archive for nearly 2.4 million
		scholarly articles in the fields of physics, mathematics, computer science, quantitative
		biology, quantitative finance, statistics, electrical engineering and systems science, and
		economics. Materials on this site are not peer-reviewed by arXiv.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/arxiv/ArxivLoader
'''

PDF_LOADER = r'''Public, SDK-oriented PDF loader with: Page-aware metadata, Two-stage chunking,
		Configurable chunk profiles, Table isolation, Optional OCR fallback.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/pdf/PyPDFLoader
'''

EXCEL_LOADER = r'''Provides LangChain's UnstructuredExcelLoader functionality
		to parse Excel spreadsheets into documents.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/excel/UnstructuredExcelLoader
'''

POWERPOINT_LOADER = r'''The UnstructuredPowerPointLoader (within LangChain) is a tool for parsing
		Microsoft PowerPoint (.ppt/.pptx) files to extract text and metadata, enabling AI applications
		to read and process presentations. It supports loading documents in "single" (full text) or
		"elements" (chunked by title/narrative) modes, ideal for Retrieval Augmented Generation (RAG) tasks
		
		https://reference.langchain.com/python/langchain-community/document_loaders/powerpoint/UnstructuredPowerPointLoader
'''

JSON_LOADER = r'''The LangChain JSONLoader is a specialized document loader used to transform JSON
		and JSON Lines data into standardized LangChain Document objects. It is a critical component
		for building applications like Retrieval-Augmented Generation (RAG) that need to process structured data
'''

MARKDOWN_LOADER = r'''LangChain's Markdown document loaders are specialized tools used to convert
		Markdown files into standardized LangChain Document objects. These objects are then used
		for downstream tasks like Retrieval Augmented Generation (RAG), embedding generation,
		or semantic chunking.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/markdown/UnstructuredMarkdownLoader
'''

XML_LOADER = r'''The UnstructuredXMLLoader in LangChain is a specialized tool designed to load and
		parse XML files into standardized LangChain Document objects. It leverages the Unstructured.io
		library to extract text content and preserve document structure for use in downstream LLM
		applications like RAG.
		
		https://reference.langchain.com/python/langchain-community/document_loaders/xml/UnstructuredXMLLoader
'''

CSV_LOADER = r'''The LangChain CSVLoader is a standard utility within the langchain-community package
		designed to transform structured CSV data into a list of standardized Document objects.
		This process is the foundational step for integrating tabular data into LLM-powered workflows,
		such as Retrieval Augmented Generation (RAG).
		
		https://reference.langchain.com/python/langchain-community/document_loaders/csv_loader/CSVLoader
'''


# -------- GENERATION PARAMETER DEFINITIONS -------------------

TEMPERATURE = r'''Optional. A number between 0 and 2. Higher values like 0.8 will make the output
		more random, while lower values like 0.2 will make it more focused and deterministic'''

TOP_P = r'''Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.'''

TOP_K = r'''Optional. The maximum number of tokens to consider when sampling. Gemini models use
		Top-p (nucleus) sampling or a combination of Top-k and nucleus sampling. Top-k sampling considers
		the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
		Note: The default value varies by Model and is specified by theModel.top_p attribute returned
		from the getModel function. An empty topK attribute indicates that the model doesn't apply
		top-k sampling and doesn't allow setting topK on requests.'''

PRESENCE_PENALTY = r'''Optional. Presence penalty applied to the next token's logprobs
		if the token has already been seen in the response. This penalty is binary on/off
		and not dependant on the number of times the token is used (after the first).'''

FREQUENCY_PENALTY = r'''Optional. Frequency penalty applied to the next token's logprobs,
		multiplied by the number of times each token has been seen in the respponse so far.
		A positive penalty will discourage the use of tokens that have already been used,
		proportional to the number of times the token has been used: The more a token is used,
		the more difficult it is for the model to use that token again increasing
		the vocabulary of responses.'''

MAX_OUTPUT_TOKENS = r'''Optional. The maximum number of tokens used in generating output content'''

ALLOWED_DOMAINS = r'''Optional. The allowed domains used in generating output content and
		grounding of generated content to reduce halucinations.'''

STOP_SEQUENCE = r'''Optional. Up to 4 string sequences where the API will stop generating further tokens.'''

STORE = 'Optional. Whether to maintain state from turn to turn, preserving reasoning and tool context '

STREAM = 'Optional. Whether to return the generated respose in asynchronous chunks'

TOOLS = '''Optional. An array of tools the model may call while generating a response. You can specify which
		tool to use by setting the tool_choice parameter. Used by the Reponses API
		and Reasoning models'''

INCLUDE = r'''Optional. Specifies additional output data to include in the model response enabling reasoning
			items to be used in multi-turn conversations when using the Responses API statelessly
			and Reasoning models.
			'''

REASONING = r'''Optional. Reasoning models introduce reasoning tokens in addition to input and output tokens.
				The models use these reasoning tokens to “think,” breaking down the prompt and
				considering multiple approaches to generating a response. After generating reasoning tokens,
				the model produces an answer as visible completion tokens and discards
				the reasoning tokens from its context. Used by the Reasoning models'''