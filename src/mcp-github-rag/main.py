from fastmcp import FastMCP
import os
import gc
import tempfile
import uuid
import shutil
from pathlib import Path
import hashlib
import git

# Define the FastMCP application
mcp = FastMCP(
    "GitHub RAG Chat",
    dependencies=[
        "gitpython",
        "llama-index",
        "ollama",
        "pandas",
        "huggingface-hub",
        "transformers"
    ]
)

# Cache for repositories and query engines
REPO_CACHE = {}
QUERY_ENGINE_CACHE = {}

def clone_repo(repo_url: str) -> str:
    """Clone a repository and return the path. If repository is already cloned in temp directory, reuse it."""
    # Create a deterministic directory name based on repo URL
    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    temp_dir = os.path.join(tempfile.gettempdir(), f"github_rag_{repo_hash}")
    
    # If directory exists and is a valid git repo, return it
    if os.path.exists(temp_dir):
        try:
            repo = git.Repo(temp_dir)
            if not repo.bare and repo.remote().url == repo_url:
                return temp_dir
        except:
            # If there's any error with existing repo, clean it up
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Create directory and clone repository
    os.makedirs(temp_dir, exist_ok=True)
    try:
        git.Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to clone repository: {str(e)}")

def process_repository(repo_url: str) -> dict:
    """Process a GitHub repository to extract content and prepare for RAG."""
    from gitingest import ingest
    
    # Check if the repository is already processed
    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    if repo_hash in REPO_CACHE:
        return REPO_CACHE[repo_hash]
    
    # Process with gitingest
    summary, tree, content = ingest(repo_url)
    
    result = {
        "summary": summary,
        "tree": tree,
        "content": content
    }
    
    # Cache the result
    REPO_CACHE[repo_hash] = result
    return result

def get_query_engine(repo_url: str) -> object:
    """Get or create a query engine for a repository."""
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.core import PromptTemplate
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import MarkdownNodeParser
    
    # Check if the query engine is already created
    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    if repo_hash in QUERY_ENGINE_CACHE:
        return QUERY_ENGINE_CACHE[repo_hash]
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process repository
        repo_data = process_repository(repo_url)
        content = repo_data["content"]
        repo_name = repo_url.split('/')[-1]
        
        # Write content to a markdown file in temp directory
        content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Load the documents
        loader = SimpleDirectoryReader(input_dir=temp_dir)
        docs = loader.load_data()
        
        # Setup LLM & embedding model
        llm = Ollama(model="llama3.2", request_timeout=120.0)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        
        # Create an index over loaded data
        Settings.embed_model = embed_model
        node_parser = MarkdownNodeParser()
        index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)
        
        # Create the query engine
        Settings.llm = llm
        query_engine = index.as_query_engine(streaming=False)  # Note: streaming not used in MCP
        
        # Customize prompt template
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a highly precise "
            "and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        # Cache the query engine
        QUERY_ENGINE_CACHE[repo_hash] = query_engine
        return query_engine

@mcp.tool()
def github_repository_summary(repo_url: str) -> str:
    """
    Retrieve a summary of a GitHub repository.
    
    Args:
        repo_url: The URL of the GitHub repository
        
    Returns:
        A summary of the repository structure and contents
    """
    try:
        repo_data = process_repository(repo_url)
        return repo_data["summary"]
    except Exception as e:
        return f"Error processing repository: {str(e)}"

@mcp.tool()
def github_repository_structure(repo_url: str) -> str:
    """
    Retrieve the structure of a GitHub repository.
    
    Args:
        repo_url: The URL of the GitHub repository
        
    Returns:
        A tree-like representation of the repository structure
    """
    try:
        repo_data = process_repository(repo_url)
        return repo_data["tree"]
    except Exception as e:
        return f"Error retrieving repository structure: {str(e)}"

@mcp.tool()
def github_repository_content(repo_url: str) -> str:
    """
    Retrieve the full content of a GitHub repository.
    
    Args:
        repo_url: The URL of the GitHub repository
        
    Returns:
        The full textual content of the repository
    """
    try:
        repo_data = process_repository(repo_url)
        return repo_data["content"]
    except Exception as e:
        return f"Error retrieving repository content: {str(e)}"

@mcp.tool()
def ask_github_repository(repo_url: str, question: str) -> str:
    """
    Ask a question about a GitHub repository using RAG.
    
    Args:
        repo_url: The URL of the GitHub repository
        question: The question to ask about the repository
        
    Returns:
        An answer to the question based on the repository content
    """
    try:
        # Get or create the query engine
        query_engine = get_query_engine(repo_url)
        
        # Query the engine
        response = query_engine.query(question)
        
        # Return the response
        return str(response)
    except Exception as e:
        return f"Error querying repository: {str(e)}"

@mcp.tool()
def clear_repository_cache() -> str:
    """
    Clear the repository and query engine cache.
    
    Returns:
        A confirmation message
    """
    global REPO_CACHE, QUERY_ENGINE_CACHE
    REPO_CACHE = {}
    QUERY_ENGINE_CACHE = {}
    gc.collect()
    return "Repository cache cleared successfully."