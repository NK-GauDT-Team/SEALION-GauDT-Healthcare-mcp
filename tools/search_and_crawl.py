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
import concurrent

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
IS_SCRAPE = True
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


    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            run_pydantic: Optional[bool] = True,
            **kwargs
        ) -> ChatResult:
        """Async wrapper for SageMaker Async Inference call."""
        loop = asyncio.get_running_loop()
        bound = functools.partial(
            self._generate, messages, stop, run_manager, run_pydantic, **kwargs
        )
        # run in thread to avoid blocking event loop
        return await loop.run_in_executor(None, bound)


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        run_pydantic: Optional[bool] = True,
        **kwargs
    ) -> ChatResult:
        """Submit request to SageMaker Async Inference and fetch from S3."""

        formatted_prompt = self._format_messages(messages)
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True,
                "return_full_text": False,
                "stop": stop or ["Human:", "\n\nHuman:", "Question:", "Observation:"]
            },
        }

        try:
            # Step 1: Upload input payload to S3
            input_key = f"async-inputs/{int(time.time())}.json"
            async_bucket = f"sealion-mcp-async-results-bucket"
            import boto3
            s3_client = boto3.client("s3")
            s3_client.put_object(
                Bucket=async_bucket,
                Key=input_key,
                Body=json.dumps(payload)
            )
            input_location = f"s3://{async_bucket}/{input_key}"

            # Step 2: Send async request using InputLocation
            response = self.client.invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                InputLocation=input_location,
                InferenceId=f"req-{int(time.time())}",
                ContentType="application/json",
            )

            # Step 3: Parse output S3 location
            output_location = response["OutputLocation"]
            assert output_location.startswith("s3://"), f"Unexpected output location: {output_location}"
            no_prefix = output_location.replace("s3://", "")
            bucket, key = no_prefix.split("/", 1)

            # Step 4: Poll until result is available
            while True:
                try:
                    obj = s3_client.get_object(Bucket=bucket, Key=key)
                    body = obj["Body"].read().decode()
                    result = json.loads(body)
                    break
                except s3_client.exceptions.NoSuchKey:
                    time.sleep(2)  # wait and retry

            # Step 5: Extract generated text
            if isinstance(result, list):
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", result.get("output", ""))
            else:
                generated_text = str(result)

            message = AIMessage(content=generated_text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        except Exception as e:
            error_message = AIMessage(content=f"SageMaker async error: {str(e)}")
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
                        max_tokens:int = 1024*5):

    client = SageMakerSealionChat(
				endpoint_name=endpoint_name,
				region_name=region_name,
				temperature=temperature,
				max_tokens=max_tokens
    )
    prompt = (f"""
           You are an expert medical information extractor specializing in multilingual medicine identification and traditional remedies from Southeast Asian and global sources.
            Your task is to extract and summarize ALL medicines, treatments, and remedies mentioned in the provided articles, regardless of language or cultural origin.
            ## LANGUAGE REQUIREMENTS:
            - Extract medicine names in their ORIGINAL language (Indonesian, Malay, Thai, Vietnamese, Chinese, etc.)
            - Provide English translations in parentheses when available
            - Include destination country terms alongside scientific names 
            - Preserve spelling variations and regional differences
            ## EXTRACTION FOCUS:
            Focus on fast, effective, and safe first-line treatments for: {query}
            Extract ALL of the following:
            1. **Traditional Medicines**: Jamu, herbal remedies, TCM, Ayurveda
            2. **Local Plants/Herbs**: Include local names (e.g., "temulawak (Curcuma zanthorrhiza)", "sambiloto", "mengkudu")
            3. **Over-the-counter medicines**: Available without prescription
            4. **Home remedies**: Kitchen ingredients, simple preparations
            5. **Lifestyle treatments**: Dietary changes, physical practices
            6. **Natural techniques**: Massage, acupressure, traditional methods

            ## OUTPUT FORMAT:
            Organize by category with original names preserved:

            **Herbal/Traditional Medicines:**
            - [Original name] ([English/Scientific name if available]) - [Brief description]

            **Over-the-Counter Medicines:**
            - [Medicine name] - [Purpose and dosage if mentioned]

            **Home Remedies:**
            - [Remedy] - [Preparation and use]

            **Lifestyle/Natural Treatments:**
            - [Method] - [How to apply]

            ## QUALITY GUIDELINES:
            - Include dosages, preparation methods, and frequency when mentioned
            - Note any safety warnings or contraindications
            - Preserve cultural context and traditional usage notes
            - Include both common and scientific names when available
            - Don't translate traditional medicine names that don't have direct English equivalents
            

            ## RESTRICTIONS:
            - Extract ONLY from the provided sources - no additional information
            - NO prescription medicines or antibiotics
            - NO medical advice beyond what's stated in sources
            - Include warnings about consulting healthcare providers when mentioned

            ## SOURCES TO PROCESS:
            Please extract medicines and treatments for "{query}" from the following articles:

            [Articles will be provided here]
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
        while attempt < max_retries:
            attempt += 1
            try:
                resp = await client.post(
                    SCRAPE_ENDPOINT,
                    headers=HEADERS,
                    json=payload,
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    data = {"raw": resp.text}
                
                return {
                    "url": url,
                    "status": resp.status_code,
                    "data": data,
                    "attempts": attempt,
                }
                
            except (httpx.HTTPError, httpx.ConnectError, httpx.ReadTimeout) as e:
                print(f"Attempt {attempt}/{max_retries} failed for {url}: {type(e).__name__}: {e}")
                
                if attempt >= max_retries:
                    return {
                        "url": url,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "attempts": attempt,
                    }
                
                # Exponential backoff
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Catch any other unexpected exceptions
                print(f"Unexpected error for {url}: {type(e).__name__}: {e}")
                return {
                    "url": url,
                    "status": "error",
                    "error": f"Unexpected error: {type(e).__name__}: {e}",
                    "attempts": attempt,
                }
        
        # This shouldn't be reached, but just in case
        return {
            "url": url,
            "status": "error",
            "error": "Max retries exceeded without proper error handling",
            "attempts": attempt,
        }

async def scrape_all(urls: List[str]) -> List[Dict[str, Any]]:
    """Scrape all URLs concurrently."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    try:
        async with httpx.AsyncClient(http2=True) as client:
            tasks = [scrape_one(client, url, semaphore) for url in urls]
            # Use return_exceptions=True to prevent one failure from stopping everything
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            normalized = []
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    print(f"Unhandled exception for {url}: {type(result).__name__}: {result}")
                    normalized.append({
                        "url": url,
                        "status": "error",
                        "error": f"Unhandled exception: {type(result).__name__}: {result}",
                        "attempts": 0,
                    })
                else:
                    normalized.append(result)
            
            return normalized
            
    except Exception as e:
        print(f"Critical error in scrape_all: {type(e).__name__}: {e}")
        # Return error results for all URLs
        return [{
            "url": url,
            "status": "error",
            "error": f"Critical scraping error: {type(e).__name__}: {e}",
            "attempts": 0,
        } for url in urls]

def run_async_scraping_in_thread(urls: List[str]) -> List[Dict[str, Any]]:
    """Run async scraping in a new thread with its own event loop."""
    def run_in_thread():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(scrape_all(urls))
            finally:
                loop.close()
        except Exception as e:
            print(f"Thread execution error: {type(e).__name__}: {e}")
            return [{
                "url": url,
                "status": "error",
                "error": f"Thread execution error: {type(e).__name__}: {e}",
                "attempts": 0,
            } for url in urls]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        try:
            return future.result(timeout=120)  # Increased timeout
        except concurrent.futures.TimeoutError:
            print("Scraping operation timed out")
            return [{
                "url": url,
                "status": "error",
                "error": "Operation timed out",
                "attempts": 0,
            } for url in urls]
        except Exception as e:
            print(f"Future execution error: {type(e).__name__}: {e}")
            return [{
                "url": url,
                "status": "error",
                "error": f"Future execution error: {type(e).__name__}: {e}",
                "attempts": 0,
            } for url in urls]

def run_async_scraping(urls: List[str]) -> List[Dict[str, Any]]:
    """Wrapper to run async scraping in sync context."""
    if not urls:
        return []
    
    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            print("Detected running event loop, using thread execution")
            return run_async_scraping_in_thread(urls)
        except RuntimeError:
            # No running loop, we can create our own
            print("No running event loop, creating new one")
            return asyncio.run(scrape_all(urls))
            
    except Exception as e:
        print(f"Critical error in run_async_scraping: {type(e).__name__}: {e}")
        return [{
            "url": url,
            "status": "error",
            "error": f"Critical wrapper error: {type(e).__name__}: {e}",
            "attempts": 0,
        } for url in urls]


def make_payload(url: str) -> Dict[str, Any]:
    """Create payload for Firecrawl API"""
    return {
        "url": url
    }

def run_async_agenerate(client, prompts):
    """
    Wrapper to run async agenerate with multiple prompts in sync context.
    Each prompt gets combined with system_prompt.
    """
    import asyncio
    import concurrent

    async def _runner():
        tasks = [
            client._agenerate([m]) 
            for m in prompts
        ]
        responses = await asyncio.gather(*tasks)
        return responses

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Jupyter or already running event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(_runner()))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(_runner())
    except RuntimeError:
        # No loop exists, safe to just run
        return asyncio.run(_runner())


def is_firecrawl_friendly(url: str) -> bool:
    """
    Simple filter to keep only web pages Firecrawl v1 can scrape.
    Exclude PDFs, images, Google Maps, Drive, YouTube, etc.
    """
    # block common non-HTML formats
    blocked_ext = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".zip")
    if url.lower().endswith(blocked_ext):
        return False

    # block Google properties that Firecrawl usually can't extract meaningfully
    blocked_domains = [
        "google.com/maps", "google.com/drive",
        "youtube.com", "facebook.com", "twitter.com",
        "instagram.com", "tiktok.com"
    ]
    if any(b in url.lower() for b in blocked_domains):
        return False

    return True

async def gs_search_with_crawling(query: str, limit: int = 6, 
                            crawl_top: int = 5,
                            country_code : str=None,
                            emit:Emit = None) -> str:
    """Enhanced Google Search that automatically crawls top results
        provides country code as well to set the searches on specific country.
    """
    def remove_links(markdown_text):
        # Remove markdown links but keep text
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_text)

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
        clean_query = query.split("\"")[0]
        print("New Query:", clean_query)

        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={API_KEY}&cx={CX}&q={query}&num={max_limit*2}&start=1"
        ) 

        # Add Country code if stated
        if country_code:
            url += f"&cr={country_code}"

        res = requests.get(url).json()
        if "items" in res:
            for item in res["items"]:
                link = item.get('link', '')
                print(link)
                if is_firecrawl_friendly(link):
                    results.append({
                        'title': item.get('title', ''),
                        'link': link,
                        'snippet': item.get('snippet', '')
                    })

        # Limit to top 5 usable links
        results = results[:max_limit]

        print("Total url to scrape: ",len(results))
        # Format search results
        formatted_results = f"Search results for '{query}':\n\n"
        
        # Extract URLs for scraping (top N results)
        urls_to_scrape = [result['link'] for result in results]
        await emit_progress_message(0,f"Scraping top links: {urls_to_scrape}",emit=emit)
        print("Total url to scrape: ",urls_to_scrape)
        
        # Scrape the top URLs using async Firecrawl
        formatted_results += f"\n--- DETAILED CONTENT FROM TOP {len(urls_to_scrape)} RESULTS (via Firecrawl) ---\n\n"
        extract_crawl = []
        
        print("Starting scraping process...")
        
        # This will NEVER raise an exception now
        scrape_results = run_async_scraping(urls_to_scrape)
        
        print("Still perform docs text")  # This will now always execute
        print(f"Scraping completed. Processing {len(scrape_results)} results...")
        
        final_messages = []
        docs_text = ""
        successful_scrapes = 0
        
        # Process scrape results more robustly
        for i, result in enumerate(scrape_results):
            print(f"Processing result {i+1}/{len(scrape_results)}: {result['url']}")
            
            if result.get('status') == 'error':
                print(f"  Error: {result.get('error', 'Unknown error')}")
                continue
            
            # Check for successful data extraction
            try:
                if "data" in result and result["data"] and "data" in result["data"]:
                    markdown_content = result["data"]["data"].get("markdown", "")
                    if markdown_content:
                        # Clean and truncate content
                        clean_content = remove_links(markdown_content)[:3000]
                        docs_text += f"\n\n--- Content from {result['url']} ---\n"
                        docs_text += clean_content
                        successful_scrapes += 1
                        print(f"  Successfully extracted {len(clean_content)} characters")
                    else:
                        print("  No markdown content found")
                else:
                    print(f"  Unexpected data structure: {result.keys()}")
                    
            except Exception as e:
                print(f"  Error processing result data: {type(e).__name__}: {e}")
                continue
        
        print(f"Successfully processed {successful_scrapes} out of {len(scrape_results)} results")
        
        # Only proceed with summarization if we have content
        if docs_text.strip():
            print(f"Preparing to summarize {len(docs_text)} characters of content...")
            
            try:
                final_messages = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"Summarize the following content:\n\n{docs_text}")
                ]
                
                print("Running simultaneous generation...")
                endpoint_name = os.getenv("ENDPOINT_NAME","Unknown")
                client = SageMakerSealionChat(
                            endpoint_name=endpoint_name,
                            region_name="us-east-1",
                            temperature=0.1,
                            max_tokens=1024
                        )
                
                # Generate summaries
                results = await asyncio.gather(*(client._agenerate([m]) for m in final_messages))
                # print(f"Total LLM results: {len(results)}")
                # print("Results:", results)
                for gen in results:

                    if gen.generations and 'ready to process the articles' not in gen.generations[0].message.content:
                        content = gen.generations[0].message.content.strip()
                        print(f"Generated content length: {len(content)}")
                        print(f"Content preview: {content[:200]}...")
                        extract_crawl.append(content)
                
                print("Finished all generation.")
                
            except Exception as llm_error:
                print(f"LLM generation error: {type(llm_error).__name__}: {llm_error}")
                extract_crawl.append(f"Error during content summarization: {llm_error}")
        else:
            print("No content available for summarization")
            extract_crawl.append("No content could be extracted from the scraped URLs")
            
        print("\n" + "-"*100)
        return extract_crawl
        
    except Exception as e:
        error_msg = f"Search and crawl failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        print(f"Critical error in gs_search_with_crawling: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return [error_msg]

async def crawl_url_with_polling_tool(
    target_url: str,
    crawl_base_url: str = None,
    max_wait_time_seconds: int = 120,
    poll_interval_seconds: int = 10
) -> str:
    """Async crawl URL and return formatted content"""
    base = crawl_base_url or os.getenv(
        "CRAWLER_BASE_URL",
        "http://ec2-47-129-166-168.ap-southeast-1.compute.amazonaws.com:3002"
    )
    base += "/v1/crawl"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Step 1: initiate crawl
            init = (await client.post(base, json={"url": target_url})).json()
            if not init.get("success"):
                return f"Crawl initiation failed: {init}"

            status_url = init.get("url")
            if not status_url:
                return f"Missing status URL in initiation response: {init}"

            # Step 2: poll for results
            start = time.time()
            while time.time() - start < max_wait_time_seconds:
                try:
                    result = (await client.get(status_url)).json()
                except httpx.RequestError:
                    await asyncio.sleep(poll_interval_seconds)
                    continue

                status = result.get("status", "unknown")
                completed = int(result.get("completed", 0) or 0)
                total = int(result.get("total", 0) or 0)
                data = result.get("data", [])

                if status == "completed" or (
                    total > 0 and completed >= total and isinstance(data, list) and len(data) > 0
                ):
                    if data:
                        first_item = data[0]
                        content = first_item.get("markdown", "No content available")
                        return f"Content from {target_url}:\n\n{content[:7000]}"
                    return f"Crawl completed but no content found for {target_url}"

                await asyncio.sleep(poll_interval_seconds)

            return f"Crawling did not complete within {max_wait_time_seconds} seconds"

        except Exception as e:
            return f"Crawling failed for {target_url}: {str(e)}"

async def crawl_multiple_urls(urls: list[str], max_concurrent: int = 3) -> dict[str, str]:
    """Crawl multiple URLs concurrently and return results"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def sem_crawl(url: str):
        async with semaphore:
            content = await crawl_url_with_polling_tool(url)
            results[url] = content
    await asyncio.gather(*(sem_crawl(url) for url in urls))
    return results

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
    
