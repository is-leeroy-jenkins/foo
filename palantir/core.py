'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                core.py
	  Author:                  Terry D. Eppler (adapted by Bro)
	  Created:                 05-31-2022
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        08-25-2025
	******************************************************************************************
	<copyright file="core.py" company="Terry D. Eppler">
	
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
	    core.py — lightweight immutable result container and small core helpers.

	    Purpose:
	        Provides shared validation and result-container primitives used by Foo fetchers,
	        scrapers, loaders, generators, and writer utilities. The module centralizes the
	        simple required-value guard and the response-backed Result object so higher-level
	        components can exchange HTTP fetch outcomes through one consistent structure.
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
from typing import Dict, Optional, Any
from requests import Response

def throw_if( name: str, value: object ) -> None:
	"""Validate a required argument.

	Purpose:
		Validates that a required argument contains a usable value before the surrounding
		workflow continues. This guard centralizes early validation so provider wrappers,
		fetchers, and UI routines fail with consistent and readable error messages.

	Args:
		name (str): Name of the argument being validated.
		value (object): Argument value to validate.

	Raises:
		ValueError: Raised when the value is ``None`` or an empty string.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Result( ):
	"""Represent an HTTP response result.

	Purpose:
		Stores the normalized outcome of a single HTTP response in a small object that is
		easy to inspect, serialize, and pass between Foo components. The container keeps
		the original response object together with common response fields such as URL,
		status code, text, encoding, and headers.

	Attributes:
		url (Optional[str]): Final response URL.
		status_code (Optional[int]): HTTP status code returned by the server.
		text (Optional[str]): Response body text.
		encoding (Optional[str]): Response text encoding reported by ``requests``.
		headers (Optional[str]): Response headers captured from the source response.
		response (Optional[Response]): Original ``requests.Response`` object.
	"""
	url: Optional[ str ]
	status_code: Optional[ int ]
	text: Optional[ str ]
	encoding: Optional[ str ]
	headers: Optional[ str ]
	response: Optional[ Response ]
	
	def __init__( self, response: Response ) -> None:
		"""Initialize the result from a response object.

		Purpose:
			Copies the primary fields from a ``requests.Response`` object into stable
			instance members. This constructor performs only local assignment and keeps
			the original response available for callers that need access to the complete
			upstream response object.

		Args:
			response (Response): Source response object returned by ``requests``.
		"""
		self.response = response
		self.url = response.url
		self.status_code = response.status_code
		self.text = response.text
		self.encoding = response.encoding
		self.headers = response.headers
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable ordering of attributes and methods for interactive use,
			debugging, documentation surfaces, and UI inspector tooling that displays
			available result members.

		Returns:
			Ordered member names exposed by the result container.
		"""
		return [ 'url',
		         'status_code',
		         'text',
		         'encoding',
		         'headers',
		         'has_html',
		         'to_dict',
		         'from_response' ]
	
	def to_dict( self ) -> Dict[ str, Any ]:
		"""Convert the result to a dictionary.

		Purpose:
			Produces a plain dictionary representation of the result for serialization,
			testing, UI display, or downstream processing. Response headers are copied
			into a standard dictionary so callers receive a detached, JSON-friendly
			mapping rather than the mutable header object from ``requests``.

		Returns:
			Dictionary containing the result URL, status code, text, encoding, and headers.
		"""
		return \
			{
					'url': self.url,
					'status_code': self.status_code,
					'text': self.text,
					'encoding': self.encoding,
					'headers': dict( self.headers ),
			}
	
	@property
	def has_html( self ) -> bool:
		"""Indicate whether response text is available.

		Purpose:
			Reports whether the result contains text content in the response body. The
			property is used as a lightweight availability check by fetcher, scraper,
			and writer workflows before rendering or serializing response content.

		Returns:
			True when response text is represented as a string; otherwise, False.
		"""
		return isinstance( self.text, str )