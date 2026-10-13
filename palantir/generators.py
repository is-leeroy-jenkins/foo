'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                generators.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="generators.py" company="Terry D. Eppler">

	     generators.py
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
    generators.py — multi-provider generative AI clients and orchestration helpers.

    Purpose:
        Provides Foo's model-generation layer for xAI Grok, Google Gemini, Anthropic Claude,
        Mistral, and OpenAI. The module normalizes provider configuration, sampling and
        reasoning controls, structured output, web grounding, image and audio operations,
        file search, translation, transcription, and response extraction behind consistent
        Python interfaces used by the application and downstream tooling.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from anthropic import Anthropic as Claude
import base64
from boogr import Error, Logger
from core import Result
import config as cfg
from google import genai
from google import genai
from google.genai import types
from openai import OpenAI
from pathlib import Path
from typing import Any, Dict, Optional, Pattern, List, Tuple
from requests import Response
from xai_sdk import Client as Xai
from mistralai import Mistral as MistralAI
import re
import urllib


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

def encode_image( path: str ) -> str:
	"""Encode image.

	Purpose:
	    Reads an image from disk and returns its bytes as a Base64-encoded UTF-8 string for APIs that accept inline image data.

	Args:
	    path (str): Filesystem or resource path identifying the input or output.

	Returns:
	    str: Normalized text produced by the operation.
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

class Generator( ):
	"""Generator component.

	Purpose:
	    Defines the shared state and abstract generation contract used by model-provider wrappers.

	Attributes:
	    timeout (Optional[int]): Maximum request duration, in seconds, applied to provider calls.
	    headers (Optional[Dict[str, Any]]): HTTP headers sent with the current request.
	    response (Optional[Response]): Most recent raw response returned by the provider client.
	    url (Optional[str]): Most recent endpoint or resource URL used by the instance.
	    result (Optional[Result]): Most recent normalized Foo result produced by the instance.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	"""
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	url: Optional[ str ]
	result: Optional[ Result ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		self.timeout = None
		self.headers = None
		self.response = None
		self.url = None
		self.result = None
		self.query = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    list[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'timeout', 'headers', 'response', 'url', 'result', 'query', 'fetch' ]
	
	def fetch( self, query: str, url: str, time: int = 10 ) -> Result | None:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    url (str): Absolute endpoint or resource URL.
		    time (int): Maximum request duration in seconds.

		Returns:
		    Result | None: Normalized Foo result for the completed provider request, or ``None`` when the selected path does not create one.

		Raises:
		    NotImplementedError: Raised when the method cannot satisfy its documented notimplemented requirement.
		"""
		raise NotImplementedError( 'Must be implemented by a subclass.' )

class Grok( Generator ):
	"""Grok component.

	Purpose:
	    Wraps the xAI Responses or chat API for text generation, reasoning controls, structured output, and grounded web search.

	Attributes:
	    client (Optional[Xai]): Initialized provider SDK client used to execute API requests.
	    model (Optional[str]): Provider model identifier used for generation requests.
	    response (Optional[Any]): Most recent raw response returned by the provider client.
	    api_key (Optional[str]): Provider credential loaded from application configuration.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    params (Optional[Dict[str, Any]]): Request parameters assembled for the most recent provider call.
	    temperature (Optional[float]): Sampling temperature applied to eligible generation requests.
	    max_tokens (Optional[int]): Maximum output-token allowance for generation requests.
	    top_p (Optional[float]): Nucleus-sampling threshold applied to eligible generation requests.
	    reasoning_effort (Optional[str]): Configured reasoning effort for models that support the setting.
	    stream (Optional[bool]): Whether the active provider request returns incremental events.
	    store (Optional[bool]): Whether provider-side response retention is enabled for the active request.
	    messages (Optional[List[Dict[str, Any]]]): Role-based message payload submitted to a chat or response endpoint.
	    system_instructions (Optional[str]): System-level instructions assembled for the active request.
	    web_search (Optional[bool]): Whether grounded web search is enabled for the active request.
	    search_domains (Optional[List[str]]): Normalized domains used to constrain or guide grounded search.
	    parallel_tool_calls (Optional[bool]): Whether eligible tools may be invoked concurrently.
	    tool_choice (Optional[str]): Tool-selection mode applied to the active request.
	    tools (Optional[List[Dict[str, Any]]]): Tool definitions attached to the active model request.
	    headers (Any): HTTP headers sent with the current request.
	    timeout (Any): Maximum request duration, in seconds, applied to provider calls.
	    file_path (Any): Resolved filesystem path of the current source or output file.
	    content (Any): Current content retained by the Grok workflow between related operations.
	"""
	client: Optional[ Xai ]
	model: Optional[ str ]
	response: Optional[ Any ]
	api_key: Optional[ str ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	reasoning_effort: Optional[ str ]
	stream: Optional[ bool ]
	store: Optional[ bool ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	system_instructions: Optional[ str ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.model = 'grok-4-fast-reasoning'
		self.client = Xai( api_key=self.api_key, base_url='https://api.x.ai/v1' )
		self.messages = None
		self.temperature = 0.7
		self.top_p = 1.0
		self.max_tokens = 2048
		self.reasoning_effort = None
		self.headers = { }
		self.timeout = None
		self.file_path = None
		self.content = None
		self.query = None
		self.params = None
		self.response = None
		self.system_instructions = None
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
		self.tools = [ ]
		self.store = True
		self.stream = False
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'client', 'model', 'response', 'api_key', 'query', 'params', 'temperature',
			'max_tokens', 'top_p', 'reasoning_effort', 'stream', 'store', 'messages',
			'system_instructions', 'web_search', 'search_domains', 'parallel_tool_calls',
			'tool_choice', 'tools', 'normalize_domains', 'supports_reasoning_effort',
			'is_reasoning_model', 'build_instructions', 'build_tools', 'build_response_format',
			'extract_output_text', 'fetch', 'generate_text', 'search_web' ]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
		    Converts domain input into a deduplicated, validated list accepted by the provider search tool.

		Args:
		    domains (Any): Domain names or URLs used to constrain grounded web search.

		Returns:
		    List[str]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				if not value:
					continue
				
				value = re.sub( r'^https?://', '', value )
				value = value.split( '/' )[ 0 ]
				value = re.sub( r':\d+$', '', value )
				value = value.lstrip( '.' )
				if value.startswith( 'www.' ):
					value = value[ 4: ]
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', value ):
					raise ValueError( f'Invalid xAI web-search domain: {value}' )
				
				if value not in values:
					values.append( value )
			
			if len( values ) > 5:
				raise ValueError( 'xAI web-search allowed domains are limited to five domains.' )
			
			return values
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning_effort( self, model: str ) -> bool:
		"""Supports reasoning effort.

		Purpose:
		    Determines whether the selected model supports reasoning effort configuration.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name == 'grok-4.3'
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'supports_reasoning_effort( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning_object( self, model: str ) -> bool:
		"""Supports reasoning object.

		Purpose:
		    Determines whether the selected model supports reasoning object configuration.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name == 'grok-4.20-multi-agent'
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'supports_reasoning_object( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def is_reasoning_model( self, model: str ) -> bool:
		"""Is reasoning model.

		Purpose:
		    Evaluates whether the supplied or current value satisfies the reasoning model condition.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return ('reasoning' in name or name.startswith( 'grok-4' ) or name.startswith(
				'grok-4.3' ) or name.startswith( 'grok-4.20' ))
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'is_reasoning_model( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_instructions( self, system: str = None,
			response_format: str = None ) -> str | None:
		"""Build instructions.

		Purpose:
		    Combines system instructions and output-format constraints into a provider-ready instruction string.

		Args:
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			parts: List[ str ] = [ ]
			
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append(
					'Return valid JSON only. Do not include markdown fences, prose, '
					'or commentary outside the JSON value.' )
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'build_instructions( self, **kwargs ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, web_search: bool = False, search_domains: Any = None ) -> List[
		Dict[ str, Any ] ]:
		"""Build tools.

		Purpose:
		    Constructs provider tool definitions required by the selected capabilities.

		Args:
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.

		Returns:
		    List[Dict[str, Any]]: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			tools: List[ Dict[ str, Any ] ] = [ ]
			
			if not web_search:
				return tools
			
			domains = self.normalize_domains( search_domains )
			tool: Dict[ str, Any ] = { 'type': 'web_search' }
			
			if domains:
				tool[ 'filters' ] = { 'allowed_domains': domains }
			
			tools.append( tool )
			return tools
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'build_tools( self, **kwargs) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_format: str = None ) -> Dict[
		                                                                  str, Any ] | None:
		"""Build response format.

		Purpose:
		    Translates the requested response mode into the provider-specific structured-output configuration.

		Args:
		    response_format (str): Requested output representation, such as text or JSON.

		Returns:
		    Dict[str, Any] | None: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			mode = str( response_format or '' ).strip( ).lower( )
			
			if not mode or mode == 'auto':
				return None
			
			if mode in [ 'json', 'json_object' ]:
				return { 'type': 'json_object' }
			
			if mode == 'text':
				return { 'type': 'text' }
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = (
					'build_response_format( self, **args ) -> Dict[ str, Any ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def extract_output_text( self, response: Any ) -> str:
		"""Extract output text.

		Purpose:
		    Extracts final text from provider response objects, dictionaries, or streamed response events.

		Args:
		    response (Any): Provider response object or event stream to inspect.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'output_text' ) and response.output_text:
				return str( response.output_text )
			
			if hasattr( response, 'text' ) and response.text:
				return str( response.text )
			
			if isinstance( response, dict ):
				if response.get( 'output_text' ):
					return str( response.get( 'output_text' ) )
				
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
				
				output = response.get( 'output', [ ] )
				if isinstance( output, list ):
					parts: List[ str ] = [ ]
					
					for item in output:
						if not isinstance( item, dict ):
							continue
						
						content = item.get( 'content', [ ] )
						if isinstance( content, list ):
							for block in content:
								if isinstance( block, dict ) and block.get( 'text' ):
									parts.append( str( block.get( 'text' ) ) )
					
					if parts:
						return '\n'.join( parts ).strip( )
			
			if hasattr( response, '__iter__' ) and not isinstance( response, (str, bytes, dict) ):
				parts: List[ str ] = [ ]
				
				for event in response:
					event_type = getattr( event, 'type', '' )
					
					if event_type == 'response.output_text.delta':
						delta = getattr( event, 'delta', '' )
						if delta:
							parts.append( str( delta ) )
					
					elif event_type == 'response.completed':
						final_response = getattr( event, 'response', None )
						if final_response is not None:
							text = self.extract_output_text( final_response )
							if text:
								return text
				
				if parts:
					return ''.join( parts )
			
			output = getattr( response, 'output', None )
			if output:
				parts: List[ str ] = [ ]
				
				for item in output:
					content = getattr( item, 'content', None )
					
					if content:
						for block in content:
							text = getattr( block, 'text', None )
							
							if text:
								parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'extract_output_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def create_response( self, payload: Dict[ str, Any ] ) -> Any:
		"""Create response.

		Purpose:
		    Submits an assembled payload through the available provider response or chat endpoint.

		Args:
		    payload (Dict[str, Any]): Validated request dictionary forwarded to the provider client.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'payload', payload )
			
			if hasattr( self.client, 'responses' ) and hasattr( self.client.responses, 'create' ):
				return self.client.responses.create( **payload )
			
			if hasattr( self.client, 'chat' ) and hasattr( self.client.chat, 'create' ):
				messages = payload.get( 'input', [ ] )
				
				if isinstance( messages, str ):
					messages = [ { 'role': 'user', 'content': messages } ]
				
				chat_payload = { 'model': payload.get( 'model' ),
						'messages': messages, 'stream': payload.get( 'stream', False ) }
				
				if payload.get( 'temperature' ) is not None:
					chat_payload[ 'temperature' ] = payload.get( 'temperature' )
				
				if payload.get( 'top_p' ) is not None:
					chat_payload[ 'top_p' ] = payload.get( 'top_p' )
				
				if payload.get( 'max_output_tokens' ) is not None:
					chat_payload[ 'max_tokens' ] = payload.get( 'max_output_tokens' )
				
				if payload.get( 'tools' ):
					chat_payload[ 'tools' ] = payload.get( 'tools' )
				
				if payload.get( 'tool_choice' ):
					chat_payload[ 'tool_choice' ] = payload.get( 'tool_choice' )
				
				if payload.get( 'stop' ):
					chat_payload[ 'stop' ] = payload.get( 'stop' )
				
				if payload.get( 'response_format' ):
					chat_payload[ 'response_format' ] = payload.get( 'response_format' )
				
				if payload.get( 'reasoning_effort' ):
					chat_payload[ 'reasoning_effort' ] = payload.get( 'reasoning_effort' )
				
				if payload.get( 'reasoning' ):
					chat_payload[ 'reasoning' ] = payload.get( 'reasoning' )
				
				return self.client.chat.create( **chat_payload )
			
			raise RuntimeError( 'The xAI client does not expose responses.create or chat.create.' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'create_response( self, payload: Dict[ str, Any ] ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			web_search: bool = False, search_domains: Any = None,
			stop: List[ str ] = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    stop (List[str]): Sequences that stop generation when encountered.
		    stream (bool): Whether the provider should return incremental response events.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.query = str( query )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.stream = bool( stream )
			self.store = bool( store )
			self.web_search = bool( web_search )
			self.search_domains = self.normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.system_instructions = self.build_instructions( system=system,
				response_format=response_format )
			self.tools = self.build_tools( web_search=self.web_search,
				search_domains=self.search_domains )
			
			input_messages: List[ Dict[ str, str ] ] = [ ]
			
			if self.system_instructions:
				input_messages.append( {
							'role': 'system',
							'content': self.system_instructions
					} )
			
			input_messages.append( {
						'role': 'user',
						'content': self.query
				} )
			
			self.params = {
					'model': self.model,
					'input': input_messages,
					'max_output_tokens': self.max_tokens,
					'stream': self.stream,
					'store': self.store,
					'parallel_tool_calls': self.parallel_tool_calls
			}
			
			if seed is not None:
				self.params[ 'seed' ] = int( seed )
			
			format_payload = self.build_response_format( response_format )
			if format_payload:
				self.params[ 'response_format' ] = format_payload
			
			if self.tools:
				self.params[ 'tools' ] = self.tools
				self.params[ 'tool_choice' ] = self.tool_choice
			
			is_reasoning = self.is_reasoning_model( self.model )
			
			if self.supports_reasoning_effort( self.model ) and self.reasoning_effort:
				self.params[ 'reasoning_effort' ] = self.reasoning_effort
			elif self.supports_reasoning_object( self.model ) and self.reasoning_effort:
				self.params[ 'reasoning' ] = { 'effort': self.reasoning_effort }
			
			if not is_reasoning:
				self.params[ 'temperature' ] = self.temperature
				self.params[ 'top_p' ] = self.top_p
				
				if stop:
					self.params[ 'stop' ] = [ str( item ) for item in stop if str( item ).strip() ]
			
			self.response = self.create_response( self.params )
			return self.extract_output_text( self.response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'fetch( self, **args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			web_search: bool = False, search_domains: Any = None,
			stop: List[ str ] = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Generate text.

		Purpose:
		    Generates text from the supplied prompt while exposing provider sampling, reasoning, and tool controls.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    stop (List[str]): Sequences that stop generation when encountered.
		    stream (bool): Whether the provider should return incremental response events.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( query=query, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, seed=seed, system=system,
				response_format=response_format, reasoning_effort=reasoning_effort,
				web_search=web_search, search_domains=search_domains, stop=stop, stream=stream,
				store=store, parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'generate_text( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			search_domains: Any = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Search web.

		Purpose:
		    Generates a response with provider web-search grounding enabled and optional domain constraints.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    stream (bool): Whether the provider should return incremental response events.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( query=query, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, seed=seed, system=system,
				response_format=response_format, reasoning_effort=reasoning_effort,
				web_search=True, search_domains=search_domains, stop=None, stream=stream,
				store=store, parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'search_web( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception

class Gemini( Generator ):
	"""Gemini component.

	Purpose:
	    Wraps Google Gemini generation with configurable sampling, thinking controls, response formats, and Google Search grounding.

	Attributes:
	    api_key (Optional[str]): Provider credential loaded from application configuration.
	    client (Optional[Any]): Initialized provider SDK client used to execute API requests.
	    model (Optional[str]): Provider model identifier used for generation requests.
	    response (Optional[Any]): Most recent raw response returned by the provider client.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    params (Optional[Dict[str, Any]]): Request parameters assembled for the most recent provider call.
	    temperature (Optional[float]): Sampling temperature applied to eligible generation requests.
	    max_tokens (Optional[int]): Maximum output-token allowance for generation requests.
	    top_p (Optional[float]): Nucleus-sampling threshold applied to eligible generation requests.
	    top_k (Optional[int]): Top-k sampling limit applied when supported by the provider.
	    candidate_count (Optional[int]): Current candidate count retained by the Gemini workflow between related operations.
	    seed (Optional[int]): Current seed retained by the Gemini workflow between related operations.
	    system_instructions (Optional[str]): System-level instructions assembled for the active request.
	    response_format (Optional[str]): Current response format retained by the Gemini workflow between related operations.
	    stop_sequences (Optional[List[str]]): Current stop sequences retained by the Gemini workflow between related operations.
	    grounding (Optional[bool]): Current grounding retained by the Gemini workflow between related operations.
	    search_domains (Optional[List[str]]): Normalized domains used to constrain or guide grounded search.
	    reasoning (Optional[bool]): Whether model reasoning or thinking controls are active.
	    thinking_level (Optional[str]): Named thinking level selected for compatible Gemini models.
	    thinking_budget (Optional[int]): Thinking-token budget selected for compatible Gemini models.
	    include_thoughts (Optional[bool]): Whether supported thought summaries are requested.
	    tools (Optional[List[Any]]): Tool definitions attached to the active model request.
	    config (Optional[Any]): Provider-specific generation configuration for the active request.
	"""
	
	api_key: Optional[ str ]
	client: Optional[ Any ]
	model: Optional[ str ]
	response: Optional[ Any ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	seed: Optional[ int ]
	system_instructions: Optional[ str ]
	response_format: Optional[ str ]
	stop_sequences: Optional[ List[ str ] ]
	grounding: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	reasoning: Optional[ bool ]
	thinking_level: Optional[ str ]
	thinking_budget: Optional[ int ]
	include_thoughts: Optional[ bool ]
	tools: Optional[ List[ Any ] ]
	config: Optional[ Any ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.client = genai.Client( api_key=self.api_key )
		self.model = 'gemini-2.5-flash'
		self.response = None
		self.query = None
		self.params = None
		self.temperature = 0.7
		self.max_tokens = 2048
		self.top_p = 1.0
		self.top_k = None
		self.candidate_count = 1
		self.seed = None
		self.system_instructions = None
		self.response_format = None
		self.stop_sequences = [ ]
		self.grounding = False
		self.search_domains = [ ]
		self.reasoning = False
		self.thinking_level = None
		self.thinking_budget = None
		self.include_thoughts = False
		self.tools = [ ]
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools,
		    and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'api_key', 'client', 'model', 'response', 'query', 'params', 'temperature',
			'max_tokens', 'top_p', 'top_k', 'candidate_count', 'seed', 'system_instructions',
			'response_format', 'stop_sequences', 'grounding', 'search_domains', 'reasoning',
			'thinking_level', 'thinking_budget', 'include_thoughts', 'tools', 'config',
			'normalize_domains', 'normalize_stop_sequences', 'supports_thinking_level',
			'supports_thinking_budget', 'build_system_instruction', 'build_thinking_config',
			'build_tools', 'build_config', 'extract_text', 'fetch', 'generate_text', 'search_web' ]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
		    Converts domain input into a deduplicated, validated list accepted by the provider search tool.

		Args:
		    domains (Any): Domain names or URLs used to constrain grounded web search.

		Returns:
		    List[str]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				
				if not value:
					continue
				
				if not value.startswith( 'http://' ) and not value.startswith( 'https://' ):
					value = f'https://{value}'
				
				parsed = urllib.parse.urlparse( value )
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
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_stop_sequences( self, stop_sequences: Any ) -> List[ str ]:
		"""Normalize stop sequences.

		Purpose:
		    Converts stop-sequence input into a clean ordered list of non-empty strings.

		Args:
		    stop_sequences (Any): Sequences that stop generation when encountered.

		Returns:
		    List[str]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if stop_sequences is None:
				return [ ]
			
			if isinstance( stop_sequences, str ):
				parts = stop_sequences.splitlines( )
			elif isinstance( stop_sequences, (list, tuple, set) ):
				parts = [ str( item ) for item in stop_sequences if item is not None ]
			else:
				parts = [ str( stop_sequences ) ]
			
			return [ str( item ).strip( ) for item in parts if str( item ).strip( ) ]
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = (
					'normalize_stop_sequences( self, stop_sequences: Any ) -> List[ str ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def supports_thinking_level( self, model: str ) -> bool:
		"""Supports thinking level.

		Purpose:
		    Determines whether the selected model supports thinking level configuration.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			return str( model ).strip( ).lower( ).startswith( 'gemini-3' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'supports_thinking_level( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_thinking_budget( self, model: str ) -> bool:
		"""Supports thinking budget.

		Purpose:
		    Determines whether the selected model supports thinking budget configuration.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			return str( model ).strip( ).lower( ).startswith( 'gemini-2.5' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'supports_thinking_budget( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_system_instruction( self, system: str = None, response_format: str = None,
			grounding: bool = False, search_domains: Any = None ) -> str | None:
		"""Build system instruction.

		Purpose:
		    Combines system guidance, output constraints, and grounding preferences into a provider-ready instruction string.

		Args:
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    grounding (bool): Whether grounding behavior is enabled for the operation.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			parts: List[ str ] = [ ]
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append( 'Return valid JSON only. Do not include markdown fences, prose, '
				              'or commentary outside the JSON value.' )
			
			domains = self.normalize_domains( search_domains )
			if grounding and domains:
				parts.append( 'When using Google Search grounding, strongly prefer relevant '
				              f'sources from these domains when available: {", ".join( domains )}.' )
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'build_system_instruction( self, **args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def build_thinking_config( self, model: str, reasoning: bool = False,
			thinking_level: str = None, thinking_budget: int | None = None,
			include_thoughts: bool = False ) -> Any:
		"""Build thinking config.

		Purpose:
		    Builds Gemini thinking settings appropriate to the selected model family.

		Args:
		    model (str): Provider model identifier selected for the request.
		    reasoning (bool): Whether provider reasoning or thinking controls are enabled.
		    thinking_level (str): Named Gemini thinking level for models that support it.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    include_thoughts (bool): Whether supported thought summaries are requested in the response.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if not reasoning:
				return None
			
			thinking_data: Dict[ str, Any ] = { }
			
			if self.supports_thinking_level( model ):
				level = str( thinking_level or 'low' ).strip( ).lower( )
				
				if level not in [ 'minimal', 'low', 'medium', 'high' ]:
					level = 'low'
				
				thinking_data[ 'thinking_level' ] = level
			
			elif self.supports_thinking_budget( model ):
				if thinking_budget is not None:
					thinking_data[ 'thinking_budget' ] = int( thinking_budget )
				else:
					thinking_data[ 'thinking_budget' ] = -1
			
			else:
				return None
			
			if include_thoughts:
				thinking_data[ 'include_thoughts' ] = True
			
			if hasattr( types, 'ThinkingConfig' ):
				return types.ThinkingConfig( **thinking_data )
			
			return thinking_data
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'build_thinking_config( self, **args ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, grounding: bool = False ) -> List[ Any ]:
		"""Build tools.

		Purpose:
		    Constructs provider tool definitions required by the selected capabilities.

		Args:
		    grounding (bool): Whether grounding behavior is enabled for the operation.

		Returns:
		    List[Any]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if not grounding:
				return [ ]
			
			if hasattr( types, 'Tool' ) and hasattr( types, 'GoogleSearch' ):
				return [ types.Tool( google_search=types.GoogleSearch( ) ) ]
			
			return [ { 'google_search': { } } ]
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'build_tools( self, grounding: bool=False ) -> List[ Any ]'
			Logger( ).write( exception )
			raise exception
	
	def build_config( self, model: str, temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			candidate_count: int = 1, seed: int | None = None,
			system: str = None, response_format: str = None,
			stop_sequences: Any = None, grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> Any:
		"""Build config.

		Purpose:
		    Builds the provider generation configuration from validated sampling, reasoning, and tool settings.

		Args:
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    candidate_count (int): Candidate count supplied by the caller and interpreted according to the method contract.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    stop_sequences (Any): Sequences that stop generation when encountered.
		    grounding (bool): Whether grounding behavior is enabled for the operation.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    reasoning (bool): Whether provider reasoning or thinking controls are enabled.
		    thinking_level (str): Named Gemini thinking level for models that support it.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    include_thoughts (bool): Whether supported thought summaries are requested in the response.
		    response_json_schema (Dict[str, Any]): Response json schema supplied by the caller and interpreted according to the method contract.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			config_data: Dict[ str, Any ] = {
					'temperature': float( temperature ),
					'max_output_tokens': int( max_tokens ),
					'top_p': float( top_p ),
					'candidate_count': int( candidate_count )
			}
			
			if top_k is not None and int( top_k ) > 0:
				config_data[ 'top_k' ] = int( top_k )
			
			if seed is not None:
				config_data[ 'seed' ] = int( seed )
			
			system_instruction = self.build_system_instruction( system=system,
				response_format=response_format, grounding=grounding,
				search_domains=search_domains )
			
			if system_instruction:
				config_data[ 'system_instruction' ] = system_instruction
			
			clean_stop = self.normalize_stop_sequences( stop_sequences )
			if clean_stop:
				config_data[ 'stop_sequences' ] = clean_stop
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				config_data[ 'response_mime_type' ] = 'application/json'
			
			if response_json_schema:
				config_data[ 'response_mime_type' ] = 'application/json'
				config_data[ 'response_json_schema' ] = response_json_schema
			
			tools_value = self.build_tools( grounding=grounding )
			if tools_value:
				config_data[ 'tools' ] = tools_value
			
			thinking_config = self.build_thinking_config( model=model, reasoning=reasoning,
				thinking_level=thinking_level, thinking_budget=thinking_budget,
				include_thoughts=include_thoughts )
			
			if thinking_config:
				config_data[ 'thinking_config' ] = thinking_config
			
			if hasattr( types, 'GenerateContentConfig' ):
				return types.GenerateContentConfig( **config_data )
			
			return config_data
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'build_config( self, **args ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, response: Any ) -> str:
		"""Extract text.

		Purpose:
		    Extracts generated text from the provider-specific response structure.

		Args:
		    response (Any): Provider response object or event stream to inspect.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'text' ) and response.text:
				return str( response.text )
			
			if isinstance( response, dict ):
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
			
			candidates = getattr( response, 'candidates', None )
			if candidates:
				parts: List[ str ] = [ ]
				
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					candidate_parts = getattr( content, 'parts', None ) if content else None
					
					if not candidate_parts:
						continue
					
					for part in candidate_parts:
						text = getattr( part, 'text', None )
						
						if text:
							parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'extract_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    candidate_count (int): Candidate count supplied by the caller and interpreted according to the method contract.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    stop_sequences (Any): Sequences that stop generation when encountered.
		    grounding (bool): Whether grounding behavior is enabled for the operation.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    reasoning (bool): Whether provider reasoning or thinking controls are enabled.
		    thinking_level (str): Named Gemini thinking level for models that support it.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    include_thoughts (bool): Whether supported thought summaries are requested in the response.
		    response_json_schema (Dict[str, Any]): Response json schema supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.query = str( prompt )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.top_k = int( top_k ) if top_k is not None else None
			self.candidate_count = int( candidate_count )
			self.seed = int( seed ) if seed is not None else None
			self.system_instructions = (str( system ).strip( )
			                            if system and str( system ).strip( )
			                            else None)
			self.response_format = response_format
			self.stop_sequences = self.normalize_stop_sequences( stop_sequences )
			self.grounding = bool( grounding )
			self.search_domains = self.normalize_domains( search_domains )
			self.reasoning = bool( reasoning )
			self.thinking_level = thinking_level
			self.thinking_budget = thinking_budget
			self.include_thoughts = bool( include_thoughts )
			self.config = self.build_config( model=self.model, temperature=self.temperature,
				max_tokens=self.max_tokens, top_p=self.top_p, top_k=self.top_k,
				candidate_count=self.candidate_count, seed=self.seed,
				system=self.system_instructions, response_format=self.response_format,
				stop_sequences=self.stop_sequences, grounding=self.grounding,
				search_domains=self.search_domains, reasoning=self.reasoning,
				thinking_level=self.thinking_level, thinking_budget=self.thinking_budget,
				include_thoughts=self.include_thoughts, response_json_schema=response_json_schema )
			self.params = { 'model': self.model, 'contents': self.query, 'config': self.config }
			
			self.response = self.client.models.generate_content( **self.params )
			return self.extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'fetch( self, *args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Generate text.

		Purpose:
		    Generates text from the supplied prompt while exposing provider sampling, reasoning, and tool controls.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    candidate_count (int): Candidate count supplied by the caller and interpreted according to the method contract.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    stop_sequences (Any): Sequences that stop generation when encountered.
		    grounding (bool): Whether grounding behavior is enabled for the operation.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    reasoning (bool): Whether provider reasoning or thinking controls are enabled.
		    thinking_level (str): Named Gemini thinking level for models that support it.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    include_thoughts (bool): Whether supported thought summaries are requested in the response.
		    response_json_schema (Dict[str, Any]): Response json schema supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k,
				candidate_count=candidate_count, seed=seed, system=system,
				response_format=response_format, stop_sequences=stop_sequences,
				grounding=grounding, search_domains=search_domains, reasoning=reasoning,
				thinking_level=thinking_level, thinking_budget=thinking_budget,
				include_thoughts=include_thoughts, response_json_schema=response_json_schema )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			search_domains: Any = None, reasoning: bool = False,
			thinking_level: str = None, thinking_budget: int | None = None,
			include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Search web.

		Purpose:
		    Generates a response with provider web-search grounding enabled and optional domain constraints.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    candidate_count (int): Candidate count supplied by the caller and interpreted according to the method contract.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    stop_sequences (Any): Sequences that stop generation when encountered.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    reasoning (bool): Whether provider reasoning or thinking controls are enabled.
		    thinking_level (str): Named Gemini thinking level for models that support it.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    include_thoughts (bool): Whether supported thought summaries are requested in the response.
		    response_json_schema (Dict[str, Any]): Response json schema supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k, candidate_count=candidate_count,
				seed=seed, system=system, response_format=response_format,
				stop_sequences=stop_sequences, grounding=True, search_domains=search_domains,
				reasoning=reasoning, thinking_level=thinking_level, thinking_budget=thinking_budget,
				include_thoughts=include_thoughts, response_json_schema=response_json_schema )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'search_web( self, prompt: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception

class Claude( Generator ):
	"""Claude component.

	Purpose:
	    Wraps Anthropic Claude message generation and optional web-search tooling behind a consistent Foo interface.

	Attributes:
	    client (Optional[Claude]): Initialized provider SDK client used to execute API requests.
	    model (Optional[str]): Provider model identifier used for generation requests.
	    response (Optional[Any]): Most recent raw response returned by the provider client.
	    api_key (Optional[str]): Provider credential loaded from application configuration.
	    messages (Optional[List[Dict[str, Any]]]): Role-based message payload submitted to a chat or response endpoint.
	    params (Optional[Dict[str, Any]]): Request parameters assembled for the most recent provider call.
	    temperature (Optional[float]): Sampling temperature applied to eligible generation requests.
	    max_tokens (Optional[int]): Maximum output-token allowance for generation requests.
	    top_p (Optional[float]): Nucleus-sampling threshold applied to eligible generation requests.
	    top_k (Optional[int]): Top-k sampling limit applied when supported by the provider.
	    thinking_budget (Optional[int]): Thinking-token budget selected for compatible Gemini models.
	    system_instructions (Optional[str]): System-level instructions assembled for the active request.
	    web_search (Optional[bool]): Whether grounded web search is enabled for the active request.
	    search_domains (Optional[List[str]]): Normalized domains used to constrain or guide grounded search.
	    blocked_domains (Optional[List[str]]): Current blocked domains retained by the Claude workflow between related operations.
	    url (Any): Most recent endpoint or resource URL used by the instance.
	    headers (Any): HTTP headers sent with the current request.
	    timeout (Any): Maximum request duration, in seconds, applied to provider calls.
	    content (Any): Current content retained by the Claude workflow between related operations.
	    agents (Any): Configured user-agent string sent with web requests.
	"""
	client: Optional[ Claude ]
	model: Optional[ str ]
	response: Optional[ Any ]
	api_key: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	thinking_budget: Optional[ int ]
	system_instructions: Optional[ str ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	blocked_domains: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.api_key = cfg.CLAUDE_API_KEY
		self.url = r'https://api.anthropic.com'
		self.client = None
		self.messages = None
		self.model = 'claude-sonnet-4-6'
		self.max_tokens = 2048
		self.temperature = 0.7
		self.top_p = 1.0
		self.top_k = None
		self.thinking_budget = None
		self.headers = { }
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.system_instructions = None
		self.web_search = False
		self.search_domains = [ ]
		self.blocked_domains = [ ]
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools,
		    and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'content', 'url', 'client', 'timeout', 'headers', 'fetch', 'api_key', 'response',
			'params', 'agents', 'messages', 'temperature', 'top_p', 'top_k', 'thinking_budget',
			'system_instructions', 'web_search', 'search_domains', 'blocked_domains' ]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
		    Converts domain input into a deduplicated, validated list accepted by the provider search tool.

		Args:
		    domains (Any): Domain names or URLs used to constrain grounded web search.

		Returns:
		    List[str]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				_parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				_parts = [ str( x ) for x in domains if x is not None ]
			else:
				_parts = [ str( domains ) ]
			
			_values = [ ]
			for _entry in _parts:
				_value = str( _entry ).strip( ).lower( )
				if not _value:
					continue
				
				if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
					_value = f'https://{_value}'
				
				_parsed = urllib.parse.urlparse( _value )
				_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
				_domain = re.sub( r':\d+$', '', _domain )
				_domain = _domain.lstrip( '.' )
				
				if _domain.startswith( 'www.' ):
					_domain = _domain[ 4: ]
				
				if _domain and _domain not in _values:
					_values.append( _domain )
			
			return _values
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_thinking( self, model: str ) -> bool:
		"""Supports thinking.

		Purpose:
		    Supports thinking using the class state and returns data required by the surrounding workflow.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return _name.startswith( 'claude-' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_supports_thinking( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, response: Any ) -> str:
		"""Extract text.

		Purpose:
		    Extracts generated text from the provider-specific response structure.

		Args:
		    response (Any): Provider response object or event stream to inspect.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'content' ) and response.content:
				_parts = [ ]
				for _block in response.content:
					_type = getattr( _block, 'type', None )
					if _type == 'text':
						_text = getattr( _block, 'text', '' )
						if _text:
							_parts.append( _text )
				if _parts:
					return '\n'.join( _parts ).strip( )
			
			return str( response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_extract_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    system (str): System-level instructions that define response behavior.
		    stop_sequences (List[str]): Sequences that stop generation when encountered.
		    thinking (bool): Whether thinking behavior is enabled for the operation.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    blocked_domains (Any): Blocked domains supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.query = query
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.top_k = int( top_k ) if top_k is not None else None
			self.system_instructions = system if system and str( system ).strip( ) else None
			self.web_search = bool( web_search )
			self.client = Claude( api_key=self.api_key )
			self.search_domains = self.normalize_domains( search_domains )
			self.blocked_domains = self.normalize_domains( blocked_domains )
			self.thinking_budget = int( thinking_budget ) if thinking_budget is not None else None
			self.messages = [ { 'role': 'user', 'content': self.query } ]
			self.params = \
				{
						'model': self.model,
						'max_tokens': self.max_tokens,
						'messages': self.messages,
				}
			
			if self.system_instructions:
				self.params[ 'system' ] = self.system_instructions
			
			if stop_sequences:
				self.params[ 'stop_sequences' ] = stop_sequences
			
			if thinking and self.supports_thinking( self.model ):
				_budget = self.thinking_budget if self.thinking_budget is not None else 1024
				if _budget < 1024:
					_budget = 1024
				if _budget >= self.max_tokens:
					raise ValueError( 'Thinking Budget must be less than Max Tokens.' )
				
				self.params[ 'thinking' ] = \
					{
							'type': 'enabled',
							'budget_tokens': _budget,
					}
				
				if self.top_p is not None:
					self.params[ 'top_p' ] = min( 1.0, max( 0.95, self.top_p ) )
			else:
				self.params[ 'temperature' ] = self.temperature
				self.params[ 'top_p' ] = self.top_p
				
				if self.top_k is not None and self.top_k > 0:
					self.params[ 'top_k' ] = self.top_k
			
			if self.web_search:
				self.tools = [ ]
				self.web_tool = \
					{
							'type': 'web_search_20250305',
							'name': 'web_search',
					}
				
				if self.search_domains:
					self.web_tool[ 'allowed_domains' ] = self.search_domains
				
				if self.blocked_domains:
					self.web_tool[ 'blocked_domains' ] = self.blocked_domains
				
				self.tools.append( self.web_tool )
				self.params[ 'tools' ] = self.tools
			
			self.response = self.client.messages.create( **self.params )
			return self.extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'fetch( self, **args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Generate text.

		Purpose:
		    Generates text from the supplied prompt while exposing provider sampling, reasoning, and tool controls.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    system (str): System-level instructions that define response behavior.
		    stop_sequences (List[str]): Sequences that stop generation when encountered.
		    thinking (bool): Whether thinking behavior is enabled for the operation.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    blocked_domains (Any): Blocked domains supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( query=query, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k, system=system,
				stop_sequences=stop_sequences, thinking=thinking, thinking_budget=thinking_budget,
				web_search=web_search, search_domains=search_domains,
				blocked_domains=blocked_domains, )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'generate_text( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Search web.

		Purpose:
		    Generates a response with provider web-search grounding enabled and optional domain constraints.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    top_k (int | None): Maximum number of high-probability tokens considered during sampling.
		    system (str): System-level instructions that define response behavior.
		    stop_sequences (List[str]): Sequences that stop generation when encountered.
		    thinking (bool): Whether thinking behavior is enabled for the operation.
		    thinking_budget (int | None): Token budget allocated to Gemini thinking when supported.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    blocked_domains (Any): Blocked domains supplied by the caller and interpreted according to the method contract.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( query=query, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k, system=system,
				stop_sequences=stop_sequences, thinking=thinking, thinking_budget=thinking_budget,
				web_search=True, search_domains=search_domains, blocked_domains=blocked_domains, )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'search_web( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception

class Mistral( Generator ):
	"""Mistral component.

	Purpose:
	    Wraps Mistral text generation and response extraction for Foo model workflows.

	Attributes:
	    client (Optional[MistralAI]): Initialized provider SDK client used to execute API requests.
	    model (Optional[str]): Provider model identifier used for generation requests.
	    response (Optional[Any]): Most recent raw response returned by the provider client.
	    api_key (Optional[str]): Provider credential loaded from application configuration.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    params (Optional[Dict[str, Any]]): Request parameters assembled for the most recent provider call.
	    temperature (Optional[float]): Sampling temperature applied to eligible generation requests.
	    max_tokens (Optional[int]): Maximum output-token allowance for generation requests.
	    top_p (Optional[float]): Nucleus-sampling threshold applied to eligible generation requests.
	    messages (Optional[List[Dict[str, Any]]]): Role-based message payload submitted to a chat or response endpoint.
	    system_instructions (Optional[str]): System-level instructions assembled for the active request.
	    seed (Optional[int]): Current seed retained by the Mistral workflow between related operations.
	    safe_prompt (Optional[bool]): Current safe prompt retained by the Mistral workflow between related operations.
	    headers (Any): HTTP headers sent with the current request.
	    timeout (Any): Maximum request duration, in seconds, applied to provider calls.
	    content (Any): Current content retained by the Mistral workflow between related operations.
	    agents (Any): Configured user-agent string sent with web requests.
	"""
	client: Optional[ MistralAI ]
	model: Optional[ str ]
	response: Optional[ Any ]
	api_key: Optional[ str ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	system_instructions: Optional[ str ]
	seed: Optional[ int ]
	safe_prompt: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.api_key = cfg.MISTRAL_API_KEY
		self.model = 'mistral-large-latest'
		self.headers = { }
		self.client = None
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.query = None
		self.temperature = None
		self.max_tokens = None
		self.top_p = None
		self.messages = [ ]
		self.system_instructions = None
		self.seed = None
		self.safe_prompt = False
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools,
		    and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'content', 'client', 'timeout', 'headers', 'fetch', 'api_key', 'response',
			'params', 'agents', 'model', 'temperature', 'max_tokens', 'top_p', 'messages',
			'system_instructions', 'seed', 'safe_prompt', '_extract_text', 'create_schema', ]
	
	def _extract_text( self, response: Any ) -> str:
		"""Extract text.

		Purpose:
		    Extracts generated text from the provider-specific response structure.

		Args:
		    response (Any): Provider response object or event stream to inspect.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'choices' ) and response.choices:
				choice = response.choices[ 0 ]
				
				if hasattr( choice, 'message' ) and choice.message is not None:
					message = choice.message
					
					if hasattr( message, 'content' ):
						content = message.content
						
						if isinstance( content, str ):
							return content.strip( )
						
						if isinstance( content, list ):
							parts: List[ str ] = [ ]
							for item in content:
								if isinstance( item, str ) and item.strip( ):
									parts.append( item.strip( ) )
								elif hasattr( item, 'text' ):
									text_value = getattr( item, 'text', '' )
									if text_value:
										parts.append( str( text_value ).strip( ) )
							
							if parts:
								return '\n'.join( parts ).strip( )
						
						return str( content ).strip( )
			return str( response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = '_extract_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'mistral-large-latest', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			safe_mode: bool = False, system: str = None ) -> str | None:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    query (str): Search text, prompt, or provider query submitted by the caller.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    safe_mode (bool): Whether safe mode behavior is enabled for the operation.
		    system (str): System-level instructions that define response behavior.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.query = str( query ).strip( )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.seed = int( seed ) if seed is not None else None
			self.safe_prompt = bool( safe_mode )
			self.system_instructions = system if system and str( system ).strip( ) else None
			self.client = MistralAI( api_key=self.api_key )
			self.messages = [ ]
			if self.system_instructions:
				self.messages.append( {
						'role': 'system',
						'content': self.system_instructions,
				} )
			
			self.messages.append( {
					'role': 'user',
					'content': self.query,
			} )
			self.params = {
					'model': self.model,
					'messages': self.messages,
					'temperature': self.temperature,
					'max_tokens': self.max_tokens,
					'top_p': self.top_p,
					'stream': False,
					'response_format': { 'type': 'text' },
					'safe_prompt': self.safe_prompt,
			}
			
			if self.seed is not None and self.seed > 0:
				self.params[ 'random_seed' ] = self.seed
			
			self.response = self.client.chat.complete( **self.params )
			return self._extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = 'fetch( self, *args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Create schema.

		Purpose:
		    Builds a JSON-compatible function schema for model tool-calling and orchestration workflows.

		Args:
		    function (str): Function name exposed in the generated tool schema.
		    tool (str): Service or tool name referenced by the generated schema.
		    description (str): Human-readable explanation embedded in the generated schema.
		    parameters (dict): JSON Schema property definitions for the tool arguments.
		    required (list[str]): Argument names that callers must supply to the generated tool.

		Returns:
		    Dict[str, str] | None: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema = \
				{
						'name': func_name,
						'description': f'{desc} This function uses the {tool_name} service.',
						'parameters':
							{
									'type': 'object',
									'properties': parameters,
									'required': required
							}
				}
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = ('create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception

class Chat( Generator ):
	"""Chat component.

	Purpose:
	    Provides a unified OpenAI interface for text, image, audio, translation, file search, and document-analysis workflows.

	Attributes:
	    api_key (Optional[str]): Provider credential loaded from application configuration.
	    client (Optional[OpenAI]): Initialized provider SDK client used to execute API requests.
	    system_instructions (Optional[str]): System-level instructions assembled for the active request.
	    model (Optional[str]): Provider model identifier used for generation requests.
	    number (Optional[int]): Current number retained by the Chat workflow between related operations.
	    temperature (Optional[float]): Sampling temperature applied to eligible generation requests.
	    top_percent (Optional[float]): Current top percent retained by the Chat workflow between related operations.
	    frequency_penalty (Optional[float]): Current frequency penalty retained by the Chat workflow between related operations.
	    presence_penalty (Optional[float]): Current presence penalty retained by the Chat workflow between related operations.
	    max_completion_tokens (Optional[int]): Upper bound applied to completion tokens.
	    store (Optional[bool]): Whether provider-side response retention is enabled for the active request.
	    stream (Optional[bool]): Whether the active provider request returns incremental events.
	    modalities (Optional[List[str]]): Current modalities retained by the Chat workflow between related operations.
	    stops (Optional[List[str]]): Current stops retained by the Chat workflow between related operations.
	    response_format (Optional[str]): Current response format retained by the Chat workflow between related operations.
	    reasoning_effort (Optional[str]): Configured reasoning effort for models that support the setting.
	    input_text (Optional[str]): Current input text retained by the Chat workflow between related operations.
	    id (Optional[str]): Current id retained by the Chat workflow between related operations.
	    vector_store_ids (Optional[List[str]]): Current vector store ids retained by the Chat workflow between related operations.
	    metadata (Optional[Dict[str, Any]]): Current metadata retained by the Chat workflow between related operations.
	    tools (Optional[List[Dict[str, Any]]]): Tool definitions attached to the active model request.
	    vector_stores (Optional[Dict[str, str]]): Current vector stores retained by the Chat workflow between related operations.
	    web_search (Optional[bool]): Whether grounded web search is enabled for the active request.
	    search_domains (Optional[List[str]]): Normalized domains used to constrain or guide grounded search.
	    parallel_tool_calls (Optional[bool]): Whether eligible tools may be invoked concurrently.
	    tool_choice (Optional[str]): Tool-selection mode applied to the active request.
	    request (Optional[Dict[str, Any]]): Current request retained by the Chat workflow between related operations.
	    response (Optional[Any]): Most recent raw response returned by the provider client.
	    query (Optional[str]): Most recent search text or model prompt submitted by the instance.
	    image_url (Optional[str]): URL associated with the current image resource or endpoint.
	    input (Optional[Any]): Current input retained by the Chat workflow between related operations.
	    messages (Optional[Any]): Role-based message payload submitted to a chat or response endpoint.
	"""
	
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	system_instructions: Optional[ str ]
	model: Optional[ str ]
	number: Optional[ int ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	reasoning_effort: Optional[ str ]
	input_text: Optional[ str ]
	id: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	metadata: Optional[ Dict[ str, Any ] ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	vector_stores: Optional[ Dict[ str, str ] ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	response: Optional[ Any ]
	query: Optional[ str ]
	image_url: Optional[ str ]
	input: Optional[ Any ]
	messages: Optional[ Any ]
	
	def __init__( self, num: int = 1, temp: float = 0.8, top: float = 0.9,
			freq: float = 0.0, pres: float = 0.0, iters: int = 10000,
			store: bool = True, stream: bool = True ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Args:
		    num (int): Num supplied by the caller and interpreted according to the method contract.
		    temp (float): Temp supplied by the caller and interpreted according to the method contract.
		    top (float): Top supplied by the caller and interpreted according to the method contract.
		    freq (float): Freq supplied by the caller and interpreted according to the method contract.
		    pres (float): Pres supplied by the caller and interpreted according to the method contract.
		    iters (int): Iters supplied by the caller and interpreted according to the method contract.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    stream (bool): Whether the provider should return incremental response events.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=self.api_key )
		self.client.api_key = cfg.OPENAI_API_KEY
		self.system_instructions = None
		self.model = 'gpt-5-mini'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = iters
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = None
		self.input_text = None
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372' ]
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = { 'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372' }
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
		self.request = None
		self.response = None
		self.query = None
		self.image_url = None
		self.input = None
		self.messages = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools,
		    and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'api_key', 'client', 'system_instructions', 'model', 'number', 'temperature',
			'top_percent', 'frequency_penalty', 'presence_penalty', 'max_completion_tokens',
			'store', 'stream', 'response_format', 'reasoning_effort', 'web_search',
			'search_domains', 'parallel_tool_calls', 'tool_choice', 'tools', 'vector_store_ids',
			'request', 'response', 'fetch', 'generate_text', 'generate_image', 'analyze_image',
			'summarize_document', 'search_web', 'search_files', 'translate', 'transcribe',
			'get_format_options', 'get_model_options', 'get_effort_options', 'get_data', 'dump' ]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
		    Converts domain input into a deduplicated, validated list accepted by the provider search tool.

		Args:
		    domains (Any): Domain names or URLs used to constrain grounded web search.

		Returns:
		    List[str]: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				
				if not value:
					continue
				
				if not value.startswith( 'http://' ) and not value.startswith( 'https://' ):
					value = f'https://{value}'
				
				parsed = urllib.parse.urlparse( value )
				domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
				domain = re.sub( r':\d+$', '', domain )
				domain = domain.lstrip( '.' )
				
				if domain.startswith( 'www.' ):
					domain = domain[ 4: ]
				
				if not domain:
					continue
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
					raise ValueError( f'Invalid domain: {domain}' )
				
				if domain not in values:
					values.append( domain )
			
			return values
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning( self, model: str ) -> bool:
		"""Supports reasoning.

		Purpose:
		    Determines whether the selected model supports reasoning configuration.

		Args:
		    model (str): Provider model identifier selected for the request.

		Returns:
		    bool: ``True`` when the condition is satisfied; otherwise ``False``.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name.startswith( 'gpt-5' ) or name.startswith( 'o' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'supports_reasoning( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_instructions( self, system: str = None,
			response_format: str = None, web_search: bool = False,
			search_domains: Any = None ) -> str | None:
		"""Build instructions.

		Purpose:
		    Combines system instructions and output-format constraints into a provider-ready instruction string.

		Args:
		    system (str): System-level instructions that define response behavior.
		    response_format (str): Requested output representation, such as text or JSON.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.

		Returns:
		    str | None: Normalized text produced by the operation, or ``None`` when no text is available.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			parts: List[ str ] = [ ]
			
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append( 'Return valid JSON only. Do not include markdown fences, prose, '
				              'or commentary outside the JSON value.' )
			
			domains = self.normalize_domains( search_domains )
			if web_search and domains:
				parts.append( 'When using web search, strongly prefer sources from the following '
				              f'domains when they are relevant and available: '
				              f'{", ".join( domains )}.' )
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'build_instructions( self, **args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, response_format: str | Dict[ str, Any ] = None,
			json_schema: Dict[ str, Any ] = None, schema_name: str = 'structured_response',
			schema_description: str = 'Structured JSON response.' ) -> Dict[ str, Any ] | None:
		"""Build text format.

		Purpose:
		    Translates the requested text response mode into the OpenAI format configuration.

		Args:
		    response_format (str | Dict[str, Any]): Requested output representation, such as text or JSON.
		    json_schema (Dict[str, Any]): Json schema supplied by the caller and interpreted according to the method contract.
		    schema_name (str): Schema name supplied by the caller and interpreted according to the method contract.
		    schema_description (str): Schema description supplied by the caller and interpreted according to the method contract.

		Returns:
		    Dict[str, Any] | None: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if isinstance( response_format, dict ):
				return response_format
			
			mode = str( response_format or '' ).strip( ).lower( )
			
			if not mode or mode == 'auto':
				return None
			
			if mode == 'text':
				return { 'type': 'text' }
			
			if mode in [ 'json', 'json_object' ]:
				return { 'type': 'json_object' }
			
			if mode in [ 'json_schema', 'schema' ]:
				throw_if( 'json_schema', json_schema )
				return {
						'type': 'json_schema',
						'name': schema_name,
						'description': schema_description,
						'strict': True,
						'schema': json_schema
				}
			
			return None
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, **args ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, web_search: bool = False, search_domains: Any = None,
			file_search: bool = False, vector_store_ids: List[ str ] = None,
			max_file_results: int = 20 ) -> List[ Dict[ str, Any ] ]:
		"""Build tools.

		Purpose:
		    Constructs provider tool definitions required by the selected capabilities.

		Args:
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    file_search (bool): Whether file search behavior is enabled for the operation.
		    vector_store_ids (List[str]): Vector store ids supplied by the caller and interpreted according to the method contract.
		    max_file_results (int): Max file results supplied by the caller and interpreted according to the method contract.

		Returns:
		    List[Dict[str, Any]]: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			tools: List[ Dict[ str, Any ] ] = [ ]
			
			if web_search:
				tools.append( { 'type': 'web_search' } )
			
			if file_search:
				store_ids = vector_store_ids or self.vector_store_ids
				throw_if( 'vector_store_ids', store_ids )
				tools.append( {
							'type': 'file_search',
							'vector_store_ids': store_ids,
							'max_num_results': int( max_file_results )
					} )
			
			return tools
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, **args ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def extract_output_text( self, response: Any ) -> str:
		"""Extract output text.

		Purpose:
		    Extracts final text from provider response objects, dictionaries, or streamed response events.

		Args:
		    response (Any): Provider response object or event stream to inspect.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'output_text' ) and response.output_text:
				return str( response.output_text )
			
			if isinstance( response, dict ):
				if response.get( 'output_text' ):
					return str( response.get( 'output_text' ) )
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
			
			if hasattr( response, '__iter__' ) and not isinstance( response, (str, bytes, dict) ):
				parts: List[ str ] = [ ]
				
				for event in response:
					event_type = getattr( event, 'type', '' )
					
					if event_type == 'response.output_text.delta':
						delta = getattr( event, 'delta', '' )
						if delta:
							parts.append( str( delta ) )
					
					elif event_type == 'response.completed':
						final_response = getattr( event, 'response', None )
						if final_response is not None and hasattr( final_response, 'output_text' ):
							text = str( final_response.output_text or '' )
							if text:
								return text
				
				if parts:
					return ''.join( parts )
			
			output = getattr( response, 'output', None )
			if output:
				parts: List[ str ] = [ ]
				for item in output:
					content = getattr( item, 'content', None )
					if content:
						for block in content:
							text = getattr( block, 'text', None )
							if text:
								parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'extract_output_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gpt-5-mini', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			system: str = None, response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto',
			json_schema: Dict[ str, Any ] = None,
			schema_name: str = 'structured_response',
			schema_description: str = 'Structured JSON response.' ) -> str:
		"""Fetch.

		Purpose:
		    Dispatches the requested retrieval or generation operation using the class configuration and returns the normalized result.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str | Dict[str, Any]): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    stream (bool): Whether the provider should return incremental response events.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.
		    json_schema (Dict[str, Any]): Json schema supplied by the caller and interpreted according to the method contract.
		    schema_name (str): Schema name supplied by the caller and interpreted according to the method contract.
		    schema_description (str): Schema description supplied by the caller and interpreted according to the method contract.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.query = prompt
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_completion_tokens = int( max_tokens )
			self.top_percent = float( top_p )
			self.store = bool( store )
			self.stream = bool( stream )
			self.web_search = bool( web_search )
			self.search_domains = self.normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.response_format = (
				str( response_format ).strip( ).lower( ) if isinstance( response_format,
					str ) else response_format)
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.system_instructions = self.build_instructions( system=system,
				response_format=(response_format if isinstance( response_format, str ) else None),
				web_search=self.web_search, search_domains=self.search_domains )
			self.tools = self.build_tools( web_search=self.web_search,
				search_domains=self.search_domains, file_search=False )
			
			self.request = {
					'model': self.model,
					'input': self.query,
					'max_output_tokens': self.max_completion_tokens,
					'store': self.store,
					'stream': self.stream,
					'parallel_tool_calls': self.parallel_tool_calls
			}
			
			text_format = self.build_text_format( response_format=response_format,
				json_schema=json_schema, schema_name=schema_name,
				schema_description=schema_description )
			
			if text_format:
				self.request[ 'text' ] = { 'format': text_format }
			
			if self.system_instructions:
				self.request[ 'instructions' ] = self.system_instructions
			
			if seed is not None:
				self.request[ 'seed' ] = int( seed )
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.supports_reasoning( self.model ) and self.reasoning_effort:
				self.request[ 'reasoning' ] = { 'effort': self.reasoning_effort }
			else:
				self.request[ 'temperature' ] = self.temperature
				self.request[ 'top_p' ] = self.top_percent
			
			self.response = self.client.responses.create( **self.request )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'fetch( self, **args ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gpt-5-mini',
			temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto',
			json_schema: Dict[ str, Any ] = None ) -> str:
		"""Generate text.

		Purpose:
		    Generates text from the supplied prompt while exposing provider sampling, reasoning, and tool controls.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str | Dict[str, Any]): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    web_search (bool): Whether to attach the provider web-search tool.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    stream (bool): Whether the provider should return incremental response events.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.
		    json_schema (Dict[str, Any]): Json schema supplied by the caller and interpreted according to the method contract.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, seed=seed, system=system,
				response_format=response_format, reasoning_effort=reasoning_effort,
				web_search=web_search, search_domains=search_domains, store=store,
				stream=stream, parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice, json_schema=json_schema )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_image( self, prompt: str ) -> str:
		"""Generate image.

		Purpose:
		    Generates image output from a text prompt using the selected image model and rendering options.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.input_text = prompt
			self.response = self.client.images.generate( model='gpt-image-1',
				prompt=self.input_text, size='1024x1024' )
			
			if hasattr( self.response, 'data' ) and self.response.data:
				image = self.response.data[ 0 ]
				
				if hasattr( image, 'url' ) and image.url:
					return image.url
			
			return str( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def analyze_image( self, prompt: str, url: str ) -> str:
		"""Analyze image.

		Purpose:
		    Submits image content with instructions and returns the model analysis.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    url (str): Absolute endpoint or resource URL.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'url', url )
			self.input_text = prompt
			self.image_url = url
			self.input = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': self.input_text
									},
									{
											'type': 'input_image',
											'image_url': self.image_url
									}
							]
					}
			]
			self.response = self.client.responses.create( model=self.model, input=self.input )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def summarize_document( self, prompt: str, path: str ) -> str:
		"""Summarize document.

		Purpose:
		    Loads document content and produces a model-generated summary under the selected generation settings.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			file_path = Path( path )
			
			if not file_path.exists( ):
				raise FileNotFoundError( str( file_path ) )
			
			with file_path.open( 'rb' ) as stream:
				uploaded = self.client.files.create( file=stream, purpose='assistants' )
			
			self.messages = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': prompt
									},
									{
											'type': 'input_file',
											'file_id': uploaded.id
									}
							]
					}
			]
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gpt-5-mini',
			temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, search_domains: Any = None,
			store: bool = True, stream: bool = False, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str:
		"""Search web.

		Purpose:
		    Generates a response with provider web-search grounding enabled and optional domain constraints.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.
		    model (str): Provider model identifier selected for the request.
		    temperature (float): Sampling temperature controlling response variability.
		    max_tokens (int): Maximum number of tokens the provider may generate.
		    top_p (float): Nucleus-sampling probability threshold.
		    seed (int | None): Optional deterministic sampling seed supported by the provider.
		    system (str): System-level instructions that define response behavior.
		    response_format (str | Dict[str, Any]): Requested output representation, such as text or JSON.
		    reasoning_effort (str): Provider-specific reasoning effort level.
		    search_domains (Any): Domain names or URLs allowed or preferred for grounded web search.
		    store (bool): Whether the provider may retain the response according to its API semantics.
		    stream (bool): Whether the provider should return incremental response events.
		    parallel_tool_calls (bool): Whether multiple eligible tool calls may execute in parallel.
		    tool_choice (str): Provider tool-selection mode or explicit tool choice.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, seed=seed, system=system,
				response_format=response_format, reasoning_effort=reasoning_effort,
				web_search=True,
				search_domains=search_domains, store=store, stream=stream,
				parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str, ... ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def search_files( self, prompt: str ) -> str:
		"""Search files.

		Purpose:
		    Searches configured file-search resources and returns a model response grounded in matching files.

		Args:
		    prompt (str): Prompt supplied by the caller and interpreted according to the method contract.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.query = prompt
			self.tools = self.build_tools( web_search=False, file_search=True,
				vector_store_ids=self.vector_store_ids, max_file_results=20 )
			self.request = { 'model': self.model, 'tools': self.tools, 'input': prompt }
			self.response = self.client.responses.create( **self.request )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, text: str ) -> str:
		"""Translate.

		Purpose:
		    Translates supplied text into the requested target language.

		Args:
		    text (str): Text content supplied to the operation.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'text', text )
			return self.fetch(
				prompt=f'Translate the following text faithfully and preserve meaning:\n\n{text}',
				model=self.model, temperature=0.2, max_tokens=self.max_completion_tokens,
				top_p=self.top_percent, system=self.system_instructions )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'translate( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, text: str ) -> str:
		"""Transcribe.

		Purpose:
		    Transcribes an audio file with the selected speech model and output settings.

		Args:
		    text (str): Text content supplied to the operation.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'text', text )
			return text
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'transcribe( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def get_format_options( self ) -> List[ str ]:
		"""Get format options.

		Purpose:
		    Returns supported format choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'auto', 'text', 'json', 'json_schema' ]
	
	def get_model_options( self ) -> List[ str ]:
		"""Get model options.

		Purpose:
		    Returns supported model choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		if hasattr( cfg, 'GPT_MODELS' ) and cfg.GPT_MODELS:
			return list( cfg.GPT_MODELS )
		
		return [ 'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2', 'gpt-4.1' ]
	
	def get_effort_options( self ) -> List[ str ]:
		"""Get effort options.

		Purpose:
		    Returns supported effort choices for validation and user-interface selection.

		Returns:
		    List[str]: Ordered values or records produced by the operation.
		"""
		return [ 'minimal', 'low', 'medium', 'high' ]
	
	def get_data( self ) -> Dict[ str, Any ]:
		"""Get data.

		Purpose:
		    Returns the instance state as a serializable dictionary for inspection or persistence.

		Returns:
		    Dict[str, Any]: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.
		"""
		return {
				'num': self.number,
				'model': self.model,
				'temperature': self.temperature,
				'top_percent': self.top_percent,
				'frequency_penalty': self.frequency_penalty,
				'presence_penalty': self.presence_penalty,
				'max_completion_tokens': self.max_completion_tokens,
				'store': self.store,
				'stream': self.stream,
				'response_format': self.response_format,
				'reasoning_effort': self.reasoning_effort,
				'web_search': self.web_search,
				'search_domains': self.search_domains,
				'parallel_tool_calls': self.parallel_tool_calls,
				'tool_choice': self.tool_choice,
				'vector_store_ids': self.vector_store_ids,
				'request': self.request
		}
	
	def dump( self ) -> str:
		"""Dump.

		Purpose:
		    Serializes the current instance state into the requested representation.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			return str( self.get_data( ) )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'dump( self ) -> str'
			Logger( ).write( exception )
			raise exception
