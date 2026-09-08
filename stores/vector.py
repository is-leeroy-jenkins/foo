'''
    ******************************************************************************************
      Assembly:                Foo
      Filename:                vector.py
      Author:                  Terry D. Eppler
      Created:                 09-08-2026
      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-08-2026
    ******************************************************************************************
    <copyright file="vector.py" company="Terry D. Eppler">

         Foo is a python framework for web scraping information into ML pipelines.

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

    </copyright>
    <summary>
        LangChain Chroma vector-storage integration for Foo document chunks.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def throw_if( name: str, value: object ) -> None:
    """Input guard.

    Purpose:
        Validates that a required argument contains a usable value before the vector-storage
        operation continues.

    Args:
        name (str): Name value used by the operation.
        value (object): Value value used by the operation.

    Returns:
        None: This function performs validation and does not return a value.
    """
    if value is None:
        raise ValueError( f'Argument "{name}" cannot be empty!' )

    if isinstance( value, str ) and not value.strip( ):
        raise ValueError( f'Argument "{name}" cannot be empty!' )

    if isinstance( value, ( list, tuple, dict, set ) ) and len( value ) == 0:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


class ChromaStore( ):
    """LangChain Chroma vector-storage wrapper.

    Purpose:
        Persists chunked LangChain documents through ``langchain_chroma.Chroma`` using the same
        embedding implementation selected for the document-embedding workflow.
    """

    documents: List[ Document ]
    embedder: Embeddings | None
    collection_name: str
    persist_directory: str
    vector_store: Chroma | None

    def __init__( self ) -> None:
        """Initialize the Chroma vector-store wrapper.

        Purpose:
            Initializes document, embedding, collection, persistence, and vector-store members
            before a LangChain Chroma operation is requested.

        Returns:
            None: This method initializes instance state.
        """
        self.documents = [ ]
        self.embedder = None
        self.collection_name = ''
        self.persist_directory = ''
        self.vector_store = None

    def create( self, documents: List[ Document ], embedder: Embeddings,
        collection_name: str, persist_directory: str ) -> Chroma:
        """Create or replace a persistent LangChain Chroma collection.

        Purpose:
            Creates the configured Chroma vector store, resets the target collection to prevent
            duplicate records from repeated Store actions, and adds the current LangChain chunk
            documents using the selected embedding implementation.

        Args:
            documents (List[Document]): Chunked LangChain documents to persist.
            embedder (Embeddings): LangChain embedding implementation used by Chroma.
            collection_name (str): Chroma collection name assigned to the loader workflow.
            persist_directory (str): Local directory used for persistent Chroma storage.

        Returns:
            Chroma: Configured LangChain Chroma vector store containing the documents.
        """
        throw_if( 'documents', documents )
        throw_if( 'embedder', embedder )
        throw_if( 'collection_name', collection_name )
        throw_if( 'persist_directory', persist_directory )
        self.documents = list( documents )
        self.embedder = embedder
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        Path( self.persist_directory ).mkdir( parents=True, exist_ok=True )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_directory,
        )
        self.vector_store.reset_collection( )
        ids = [
            str( ( document.metadata or { } ).get( 'chunk_id', f'chunk-{index:06d}' ) )
            for index, document in enumerate( self.documents, start=1 )
        ]
        self.vector_store.add_documents( documents=self.documents, ids=ids )
        return self.vector_store
