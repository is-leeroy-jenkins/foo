'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                models.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="models.py" company="Terry D. Eppler">

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
    models.py

    Purpose:
        Defines lightweight Pydantic models used by Foo to normalize prompts, files,
        messages, tool declarations, locations, weather summaries, directions, and
        astronomy coordinates. These models provide documented structured data shapes
        for serialization, validation, and MkDocs-generated API reference pages.
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Prompt( BaseModel ):
	"""Represent a structured prompt definition.

	Purpose:
		Captures the core fields used to store, version, and reuse an instruction bundle
		for language-model workflows. The model gives Foo a consistent shape for prompt
		text, identifiers, versions, formats, and associated user questions.

	Attributes:
		instructions (Optional[str]): Primary instruction block, usually equivalent to a
			system message or reusable prompt body.
		format (Optional[str]): Format label such as ``chat``, ``json``, or ``completion``.
		question (Optional[str]): User question or task associated with the prompt.
	"""
	instructions: Optional[ str ]
	format: Optional[ str ]
	question: Optional[ str ]

class File( BaseModel ):
	"""Represent file metadata.

	Purpose:
		Stores common metadata for uploaded files, generated files, provider artifacts,
		and tool outputs. The model intentionally remains permissive so Foo can normalize
		file-like objects returned by different upstream APIs without coupling to a single
		provider response schema.

	Attributes:
		filename (Optional[str]): Original, assigned, or display filename.
		bytes (Optional[int]): File size in bytes when supplied by the upstream provider.
		created_at (Optional[int]): Unix timestamp indicating when the file was created.
		expires_at (Optional[int]): Unix timestamp indicating when the file expires.
		id (Optional[str]): Unique upstream file identifier.
		object (Optional[str]): Upstream object discriminator, such as ``file``.
		purpose (Optional[str]): Provider-specific purpose value for the file.
	"""
	filename: Optional[ str ]
	bytes: Optional[ int ]
	created_at: Optional[ int ]
	expires_at: Optional[ int ]
	id: Optional[ str ]
	object: Optional[ str ]
	purpose: Optional[ str ]

class Error( BaseModel ):
	"""Represent an upstream API error payload.

	Purpose:
		Normalizes simple error objects returned by upstream services. Foo can surface
		structured error codes and messages in the UI or documentation without parsing
		unstructured exception strings.

	Attributes:
		code (Optional[str]): Provider-specific error code when available.
		message (Optional[str]): Human-readable error message.
	"""
	code: Optional[ str ]
	message: Optional[ str ]

class Reasoning( BaseModel ):
	"""Represent reasoning metadata.

	Purpose:
		Stores reasoning configuration or summary fields returned by a model provider.
		The model keeps reasoning-related metadata structured without requiring callers
		to depend on a provider-specific response object.

	Attributes:
		effort (Optional[str]): Reasoning effort label such as ``low``, ``medium``, or
			``high`` when supported by the provider.
		summary (Optional[str]): Short reasoning summary when one is returned.
	"""
	effort: Optional[ str ]
	summary: Optional[ str ]

class Document( BaseModel ):
	"""Represent a simple structured document.

	Purpose:
		Captures a generic document-like output with a summary and longer description.
		Foo uses this shape for structured-output examples and workflows where a model
		or loader returns a compact narrative artifact.

	Attributes:
		summary (Optional[str]): High-level document summary.
		description (Optional[str]): Longer narrative description or extracted content.
	"""
	summary: Optional[ str ]
	description: Optional[ str ]

class Message( BaseModel ):
	"""Represent a chat message object.

	Purpose:
		Normalizes conversational state across model providers and internal workflows.
		The model supports message content, role, optional message type, and additional
		metadata so callers can serialize or inspect chat-like objects consistently.

	Attributes:
		content (Optional[str]): Message payload text.
		role (Optional[str]): Message role, such as ``system``, ``user``, ``assistant``,
			or ``tool``.
		type (Optional[str]): Optional provider or application message discriminator.
		data (Optional[Dict]): Optional metadata or structured payload associated with
			the message.
	"""
	content: Optional[ str ]
	role: Optional[ str ]
	type: Optional[ str ]
	data: Optional[ Dict ]

class Location( BaseModel ):
	"""Represent a coarse geographic location.

	Purpose:
		Stores city, country, region, timezone, and type information used to bias search,
		mapping, weather, and other location-aware tools. The model is intentionally
		high-level and does not store precise street-level location data.

	Attributes:
		type (Optional[str]): Object type discriminator.
		city (Optional[str]): City name.
		country (Optional[str]): Country name or country code.
		region (Optional[str]): State, province, territory, or region name.
		timezone (Optional[str]): IANA timezone string when known.
	"""
	type: Optional[ str ]
	city: Optional[ str ]
	country: Optional[ str ]
	region: Optional[ str ]
	timezone: Optional[ str ]

class GeoCoordinates( BaseModel ):
	"""Represent latitude and longitude coordinates.

	Purpose:
		Stores a decimal-degree coordinate pair and optional timezone for tools that need
		location precision beyond a city or region. The structure is suitable for maps,
		nearby search, weather lookups, and proximity-based retrieval.

	Attributes:
		type (Optional[str]): Object type discriminator.
		latitude (Optional[float]): Latitude in decimal degrees.
		longitude (Optional[float]): Longitude in decimal degrees.
		timezone (Optional[str]): IANA timezone string when known.
	"""
	type: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]

class Forecast( BaseModel ):
	"""Represent a simplified weather forecast.

	Purpose:
		Stores a compact weather result that can be returned by a provider wrapper, tool,
		or structured model response. The model keeps temperature, precipitation, and sky
		condition fields separate for display and downstream processing.

	Attributes:
		type (Optional[str]): Object type discriminator.
		temperature (Optional[int]): Temperature value using the provider's selected units.
		precipitation (Optional[int]): Precipitation percentage or provider-specific amount.
		sky_conditions (Optional[str]): Human-readable sky condition description.
	"""
	type: Optional[ str ]
	temperature: Optional[ int ]
	precipitation: Optional[ int ]
	sky_conditions: Optional[ str ]

class Directions( BaseModel ):
	"""Represent simplified route directions.

	Purpose:
		Stores route information returned by a mapping or navigation provider. The route
		field remains provider-neutral so it can hold a text route, polyline, structured
		step list, or other upstream route representation.

	Attributes:
		type (Optional[str]): Object type discriminator.
		route (Optional[Any]): Provider-specific route representation.
	"""
	type: Optional[ str ]
	route: Optional[ Any ]

class SkyCoordinates( BaseModel ):
	"""Represent astronomical sky coordinates.

	Purpose:
		Stores right ascension and declination values for astronomy-oriented structured
		outputs. Foo uses this shape when normalizing sky-object lookups, star charts,
		or other celestial-coordinate responses.

	Attributes:
		type (Optional[str]): Object type discriminator.
		declination (Optional[float]): Declination in decimal degrees.
		right_ascension (Optional[float]): Right ascension in provider-specific units.
	"""
	type: Optional[ str ]
	declination: Optional[ float ]
	right_ascension: Optional[ float ]

class Tool( BaseModel ):
	"""Represent a tool declaration.

	Purpose:
		Stores the common fields used to describe a model-callable tool. The base model
		captures the tool name, type, and description, while specialized subclasses add
		configuration such as parameters, filters, or environment settings.

	Attributes:
		name (Optional[str]): Tool or function name exposed to the model.
		type (Optional[str]): Tool type discriminator, commonly ``function`` or a vendor
			specific tool type.
		description (Optional[str]): Human-readable description of the tool.
	"""
	name: Optional[ str ]
	type: Optional[ str ]
	description: Optional[ str ]

class Function( Tool ):
	"""Represent a function tool declaration.

	Purpose:
		Extends the generic tool declaration with JSON-schema-like parameters and a
		strictness flag. This model is used when Foo needs to serialize a callable
		function definition for provider tool-calling APIs.

	Attributes:
		parameters (Optional[Dict[str, Any]]): JSON-schema-like input definition for the
			function.
		strict (Optional[bool]): Whether provider-side argument validation should be strict.
	"""
	parameters: Optional[ Dict[ str, Any ] ]
	strict: Optional[ bool ]

class FileSearch( Tool ):
	"""Represent a file-search tool configuration.

	Purpose:
		Stores configuration for a file-search tool invocation, including the vector stores
		available to the search operation, result limits, and optional filter criteria.
		The model supports serialization and rehydration of file-search configurations.

	Attributes:
		vector_store_ids (Optional[List[str]]): Vector store identifiers available to the
			search tool.
		max_num_results (Optional[int]): Maximum number of results to return.
		filters (Optional[Dict[str, Any]]): Optional metadata or provider-specific filters.
	"""
	vector_store_ids: Optional[ List[ str ] ]
	max_num_results: Optional[ int ]
	filters: Optional[ Dict[ str, Any ] ]

class WebSearch( Tool ):
	"""Represent a web-search tool configuration.

	Purpose:
		Stores configuration for a web-search tool invocation. The model captures context
		size and optional user-location metadata so callers can construct provider search
		requests in a structured, documented way.

	Attributes:
		type (Optional[str]): Tool type discriminator, commonly ``web_search``.
		search_context_size (Optional[str]): Desired amount of context returned by the
			provider, such as ``low``, ``medium``, or ``high``.
		user_location (Optional[Any]): Optional location descriptor used to bias results.
	"""
	type: Optional[ str ]
	search_context_size: Optional[ str ]
	user_location: Optional[ Any ]

class ComputerUse( Tool ):
	"""Represent a computer-use tool configuration.

	Purpose:
		Stores virtual display and environment settings for a computer-use or UI automation
		tool. The model is provider-neutral and supports browser, desktop, or other
		environment labels when they are available.

	Attributes:
		type (Optional[str]): Tool type discriminator, commonly ``computer_use``.
		display_height (Optional[int]): Virtual display height in pixels.
		display_width (Optional[int]): Virtual display width in pixels.
		environment (Optional[str]): Provider-specific environment label.
	"""
	type: Optional[ str ]
	display_height: Optional[ int ]
	display_width: Optional[ int ]
	environment: Optional[ str ]
