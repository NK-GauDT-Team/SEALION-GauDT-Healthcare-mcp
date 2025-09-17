import json
import boto3
from typing import Dict, List, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field
import logging
import re
import requests
import os
import httpx,asyncio,time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import functools
import asyncio
from typing import Callable, Awaitable, Any

Emit = Callable[[str, Any, str | None], Awaitable[None]]
# Load environment variables from .env file
load_dotenv()

async def emit_progress_message(step,messages,emit:Emit,data={}):
	return await emit("progress", {
					"step": step,
					"message": messages,
					"data":data
			}, id_=str(step))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FIRECRAWL_BASE = os.getenv("CRAWLER_BASE_URL", "http://ec2-13-251-44-139.ap-southeast-1.compute.amazonaws.com:3002")
SCRAPE_ENDPOINT = f"{FIRECRAWL_BASE}/v1/scrape"
HEADERS = {"Content-Type": "application/json"}

# Keep this equal to your Playwright workers to avoid queueing contention
MAX_CONCURRENCY = int(os.getenv("SCRAPER_CONCURRENCY", "3"))

class SageMakerSealionChat(BaseChatModel):
    """ChatModel implementation for SageMaker Sealion endpoint"""

    endpoint_name: str
    region_name: str = "us-east-1"
    max_tokens: int = 1024 * 1
    temperature: float = 0.7
    top_p: float = 0.9
    client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = boto3.client(
            'sagemaker-runtime',
            region_name=self.region_name
        )

    @property
    def _llm_type(self) -> str:
        return "sagemaker-sealion-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        run_pydantic: Optional[bool] = True,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result from messages"""

        # Format messages for Sealion
        formatted_prompt = self._format_messages(messages)

        try:
            if run_pydantic:
                # Prepare the payload
                payload = {
                    "inputs": formatted_prompt,
                    "parameters": {
                        "max_new_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "do_sample": True,
                        "return_full_text": False,
                        "stop": stop or ["Human:", "\n\nHuman:", "Question:", "Observation:"]
                    }
                }
            else:
                payload = {
                    "inputs": formatted_prompt,
                    "parameters": {
                        "max_new_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                }

            logger.debug(f"Sending to SageMaker: {formatted_prompt[:200]}...")

            # Invoke the endpoint
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            # Parse response
            result = json.loads(response['Body'].read().decode())

            # Extract generated text
            if isinstance(result, list):
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', result.get('output', ''))
            else:
                generated_text = str(result)

            logger.debug(f"Received from SageMaker: {generated_text[:200]}...")

            # Create AIMessage with proper ChatGeneration
            message = AIMessage(content=generated_text)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error calling SageMaker endpoint: {str(e)}")
            error_message = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages for Sealion model with better structure"""
        formatted_prompt = ""

        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_prompt += f"<|system|>\n{message.content}\n\n"
            elif isinstance(message, HumanMessage):
                formatted_prompt += f"<|user|>\n{message.content}\n\n"
            elif isinstance(message, AIMessage):
                formatted_prompt += f"<|assistant|>\n{message.content}\n\n"
            elif hasattr(message, 'content'):
                # Handle other message types
                formatted_prompt += f"{message.content}\n\n"

        # Ensure we end properly for the model to continue
        if not formatted_prompt.endswith("<|assistant|>\n"):
            formatted_prompt += "<|assistant|>\n"

        return formatted_prompt

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        run_pydantic: Optional[bool] = True,
        **kwargs
    ) -> ChatResult:
        """Generate text based on the prompt and SageMaker model"""
        loop = asyncio.get_running_loop()
        bound = functools.partial(
            self._generate, messages, stop, run_manager,run_pydantic, **kwargs
        )
        return await loop.run_in_executor(None, bound)

# Tool implementations (using your original functions)
def post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def get_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def load_sealion_chat(endpoint_name:str,query:str,
                        region_name:str="us-east-1",
                        temperature:float =0.3,
                        max_tokens:int = 1024*2):

    client = SageMakerSealionChat(
				endpoint_name=endpoint_name,
				region_name=region_name,
				temperature=temperature,
				max_tokens=max_tokens
    )
    prompt = (f"""
            You are an expert in extracting and summarizing all the medicines stated in the articles given.
            You will be given one or more articles from webpages. 
            Focus on fast, effective, and safe first-line treatments that can help manage the illness before considering prescription medicines.
            Extract and highlight key remedies, lifestyle practices, or natural techniques mentioned in the sources that may provide relief.
            Do not include unnecessary details—prioritize only practical solutions that directly address the problem. 
            
            Guidelines:  
            - Focus only on medicines, treatments, or actionable techniques relevant to the illness mentioned
            - Prioritize medicines that are safe, accessible, and can be implemented quickly.
            - Include both traditional and non-traditional methods if available.  
            - Eliminate unnecessary details, filler text, and unrelated information.  
            - Keep the summary concise but comprehensive, not exceeding 200-300 words.  
            
            RESTRICTIONS:
            - Only summarize/extract informations given by the sources. Do not make up answers.
            - Do not suggest prescription medicines or antibiotics.
            Here is the illness to address: {query}
        """)
    
    return client,prompt

async def scrape_one(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    """POST a single URL to Firecrawl with retries."""
    attempt = 0
    payload = make_payload(url)

    async with semaphore:
        while True:
            attempt += 1
            try:
                resp = await client.post(
                    SCRAPE_ENDPOINT,
                    headers=HEADERS,
                    json=payload,
                    timeout=timeout_s,
                )
                # Raise for HTTP errors
                resp.raise_for_status()

                # Firecrawl usually returns JSON. Guard against non-JSON responses.
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    data = {"raw": resp.text}

                return {
                    "url": url,
                    "status": resp.status_code,
                    "data": data,
                }

            except (httpx.HTTPError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt >= max_retries:
                    return {
                        "url": url,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "attempts": attempt,
                    }
                # Backoff
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

async def scrape_all(urls: List[str]) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [scrape_one(client, u, semaphore) for u in urls]
        return await asyncio.gather(*tasks)

def run_async_scraping(urls: List[str]) -> List[Dict[str, Any]]:
    """Wrapper to run async scraping in sync context"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, scrape_all(urls))
                return future.result(timeout=20)  # 1 minute timeout
        else:
            return loop.run_until_complete(scrape_all(urls))
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(scrape_all(urls))

def make_payload(url: str) -> Dict[str, Any]:
    """Create payload for Firecrawl API"""
    return {
        "url": url
    }

def run_async_agenerate(client, messages):
    """Wrapper to run async agenerate in sync context"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()     
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, client._agenerate(messages))
                return future.result(timeout=20)  # 1 minute timeout
        else:
            return loop.run_until_complete(client._agenerate(messages))
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(client._agenerate(messages))

async def gs_search_with_crawling(query: str, limit: int = 6, 
                            crawl_top: int = 5,
                            country_code : str=None,
                            emit:Emit = None) -> str:
    """Enhanced Google Search that automatically crawls top results
        provides country code as well to set the searches on specific country.
    """
    max_limit = 5
    
    try:
        # Load all the necessary credentials
        API_KEY = os.getenv("GS_API")
        CX = os.getenv("CX")
        results = []
        sec_endpoint = os.getenv("ENDPOINT_NAME2")

            # Load 2nd endpoint config
        client,prompt = load_sealion_chat(sec_endpoint,query)
        messages = [SystemMessage(content=prompt)]
        await emit_progress_message(0,f"Searching {query}...",emit=emit)
        
        # Initialize async client for http requests
        async_client = httpx.AsyncClient()        # Fetch search results
        for start in range(1, min(max_limit, 11), 5):
            clean_query = query.split("\"")[0]
            print("New Query: ",clean_query)
            url = (
                "https://www.googleapis.com/customsearch/v1"
                f"?key={API_KEY}&cx={CX}&q={query}&start={start}&num={max_limit}"
            ) 
            # Add Country code if stated
            if country_code:
                url += f"&cr={country_code}"
            
            res = requests.get(url).json()
            if "items" in res:
                for item in res["items"]:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })
            else:
                break

        if not results:
            return f"No search results found for '{query}'"

        # Format search results
        formatted_results = f"Search results for '{query}':\n\n"
        # for i, result in enumerate(results, 1):
        #     formatted_results += f"{i}. **{result['title']}**\n"
        #     formatted_results += f"   URL: {result['link']}\n"
        #     formatted_results += f"   Summary: {result['snippet']}\n\n"
        
        # Extract URLs for scraping (top N results)
        urls_to_scrape = [result['link'] for result in results]
        await emit_progress_message(0,f"Scraping top links: {urls_to_scrape}",emit=emit)
        
        # Scrape the top URLs using async Firecrawl
        formatted_results += f"\n--- DETAILED CONTENT FROM TOP {len(urls_to_scrape)} RESULTS (via Firecrawl) ---\n\n"
        extract_crawl = []
        try:
            scrape_results = run_async_scraping(urls_to_scrape)
            final_messages = []
            for result in scrape_results:
                if 'data' in result:
                    final_messages.extend([SystemMessage(content=prompt), result['data']])

            print("Run simoultaneous generation.")
            results = run_async_agenerate(client, final_messages)
            print("Type of results: ",type(results))
            for gen in results.generations:
                if gen.message:
                    print("Content length: ",len(gen.message.content))
                    print("Content: ",gen.message)
                    extract_crawl.append(gen.message.content)
            print("Finish all generation.")            
            # for i, result in enumerate(scrape_results, 1):
            #     print("Run")
            #     formatted_results += f"=== CONTENT {i}: {result.get('url', 'Unknown URL')} ===\n"
            #     # Extract content from Firecrawl response
            #     data = result.get('data', {})
            #     if isinstance(data, dict):
            #         content = data.get('content', data.get('markdown', data.get('text', str(data))))
            #     else:
            #         content = str(data)
                
            #     print("Sending Content: ",len(content))
            #     key_info = client._generate(messages + [f"\n\n{content[:4000]}"])
            #     if key_info.generations[0].message:
            #         extract_crawl.append(key_info.generations[0].message.content)
            #     print("Finish Extraction")
                
                # Extract key information
                # if content and len(content) > 50:
                #     key_info = extract_crawl_first_item_tool(content)
                #     formatted_results += f"{key_info}\n\n"
                # else:
                #     formatted_results += "No meaningful content extracted.\n\n"
            
        except Exception as scrape_error:
            formatted_results += f"Error during scraping: {str(scrape_error)}\n"
            logger.error(f"Scraping error: {scrape_error}")
        
        return extract_crawl
        
    except Exception as e:
        logger.error(f"Search and crawl failed: {e}")
        return f"Search and crawl failed: {str(e)}"
        
    #     extract_crawl = []
    #     for i, (url, content) in enumerate(crawl_results.items(), 1):
    #         # Extract only max_limit
    #         if len(extract_crawl) > max_limit:
    #             break
    #         # formatted_results += f"=== CONTENT {i}: {url} ===\n"
    #         # Extract key information from crawled content
    #         key_info = client._generate(messages + [f"\n\n{content}"])
    #         if key_info.generations[0].message:
    #             extract_crawl.append(key_info.generations[0].message.content)
        
    #     return "\n\n".join(extract_crawl)
    # except Exception as e:
    #     return f"Search and crawl failed: {str(e)}"


def crawl_multiple_urls(urls: List[str], max_workers: int = 3) -> Dict[str, str]:
    """Crawl multiple URLs concurrently and return results"""
    results = {}
    
    def crawl_single_url(url: str) -> tuple[str, str]:
        """Helper function to crawl a single URL"""
        try:
            content = crawl_url_with_polling_tool(url)
            return url, content
        except Exception as e:
            return url, f"Error crawling {url}: {str(e)}"
    
    # Use ThreadPoolExecutor for concurrent crawling
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all crawl tasks
        future_to_url = {executor.submit(crawl_single_url, url): url for url in urls}
        
        # Collect results as they complete
        for future in as_completed(future_to_url, timeout=180):  # 3 minutes total timeout
            url = future_to_url[future]
            try:
                url_result, content = future.result()
                results[url_result] = content
                logger.info(f"Successfully crawled: {url} with {content[:200]}")
            except Exception as e:
                results[url] = f"Error: {str(e)}"
                logger.error(f"Failed to crawl {url}: {str(e)}")
    
    return results

def crawl_url_with_polling_tool(
                                    target_url: str,
                                    crawl_base_url: Optional[str] = None,
                                    max_wait_time_seconds: int = 120,
                                    poll_interval_seconds: int = 10
                                ) -> str:
    """Crawl URL and return formatted content"""
    try:
        base = crawl_base_url or os.getenv("CRAWLER_BASE_URL", "http://ec2-47-129-166-168.ap-southeast-1.compute.amazonaws.com:3002/v1/crawl")
        
        # Step 1: initiate crawl
        init = post_json(base, {"url": target_url})
        if not init.get("success"):
            return f"Crawl initiation failed: {init}"
        
        status_url = init.get("url")
        if not status_url:
            return f"Missing status URL in initiation response: {init}"
        
        # Step 2: poll for results
        start = time.time()
        while time.time() - start < max_wait_time_seconds:
            try:
                result = get_json(status_url)
            except requests.RequestException:
                time.sleep(poll_interval_seconds)
                continue
            
            status = result.get("status", "unknown")
            completed = int(result.get("completed", 0) or 0)
            total = int(result.get("total", 0) or 0)
            data = result.get("data", [])
            
            if status == "completed" or (total > 0 and completed >= total and isinstance(data, list) and len(data) > 0):
                # Extract and format the first item
                if data and len(data) > 0:
                    first_item = data[0]
                    content = first_item.get('markdown', 'No content available')
                    return f"Content from {target_url}:\n\n{content[:7000]}"  # Limit content length
                return f"Crawl completed but no content found for {target_url}"
            
            time.sleep(poll_interval_seconds)
        
        return f"Crawling did not complete within {max_wait_time_seconds} seconds"
    except Exception as e:
        return f"Crawling failed: {str(e)}"


def gs_search(query: str, limit: int = 11) -> str:
    """Search Google Custom Search and return formatted results,
        Better results can be obtained by giving local language.
        Example: in Vietnam (query should be in vietnamese), Jakarta (should be indonesian)
    """
    try:
        API_KEY = os.getenv("GS_API")
        CX = os.getenv("CX")
        results = []
        
        # Fetch results
        for start in range(1, min(limit, 11), 5):
            url = (
                "https://www.googleapis.com/customsearch/v1"
                f"?key={API_KEY}&cx={CX}&q={query}&start={start}"
            )
            res = requests.get(url).json()
            if "items" in res:
                for item in res["items"]:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })
            else:
                break
        
        # Format results for the agent
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results[:5], 1):
            formatted_results += f"{i}. **{result['title']}**\n"
            formatted_results += f"   URL: {result['link']}\n"
            formatted_results += f"   Summary: {result['snippet']}\n\n"
        
        return formatted_results
    except Exception as e:
        return f"Search failed: {str(e)}"

def extract_crawl_first_item_tool(crawl_result: str) -> str:
    """Extract key information from crawl result text"""
    try:
        # Simple extraction of key health-related information
        # lines = crawl_result.split('\n')
        # key_lines = []
        
        # health_keywords = [
        #     'vaccine', 'vaccination', 'health', 'medical', 'hospital', 'clinic',
        #     'masuk angin', 'cold', 'fever', 'medicine', 'treatment', 'doctor',
        #     'emergency', 'insurance', 'requirement', 'prevention', 'symptom'
        # ]
        
        # for line in lines:
        #     line_lower = line.lower()
        #     if any(keyword in line_lower for keyword in health_keywords) and len(line.strip()) > 10:
        #         key_lines.append(line.strip())
        
        # if key_lines:
        #     return f"Key health information extracted:\n\n" + "\n".join(key_lines[:10])
        # else:
        #     return f"Summary of content (first 500 chars):\n\n{crawl_result[:500]}"
        return crawl_result
    except Exception as e:
        return f"Extraction failed: {str(e)}"
    
class SageMakerToolsVariable:
    # GOOGLE_SEARCH = Tool(
    #     name="google_search",
    #     description="""Search Google for current information about health, travel, and medical topics. 
    #                 Use this for finding recent information about diseases, treatments, travel requirements, and health advice. 
    #                 Input should be a search query string.
    #                 For more accurate searching, please use origin words of the user's destination for giving remedies of local context.""",
    #     func=gs_search
    # )

    # URL_CRAWLER = Tool(
    #     name="crawl_url", 
    #     description="Extract detailed content from a specific webpage URL. Use this to get full content from medical websites, health authorities, or travel health pages. Input should be a valid URL.",
    #     func=crawl_url_with_polling_tool
    # )

    # EXTRACT_INFO = Tool(
    #     name="extract_info",
    #     description="Extract and summarize key health-related information from text content. Use this to get the most important health information from crawled content. Input should be text content to analyze.",
    #     func=extract_crawl_first_item_tool
    # )

    GOOGLE_SEARCH_WITH_CRAWLING = Tool(
        name="google_search_and_crawl",
        description="""Search Google and automatically crawl the top 5 results for detailed content.
                        Use this when you need comprehensive, detailed information from multiple sources about health, travel, or medical topics.
                        You may use country_code = ID|VN|US|etc given by the query to send to search for specific countries.""",
        func=gs_search_with_crawling
    )
    
