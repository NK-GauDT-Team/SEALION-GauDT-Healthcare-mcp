from tools.search_and_crawl import SageMakerToolsVariable,SageMakerSealionChat
from pydantic import BaseModel, Field
from typing import Dict
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Optional, Any, Dict
import re
import boto3
import json,os
import asyncio
from typing import Callable, Awaitable, Any
Emit = Callable[[str, Any, str | None], Awaitable[None]]

logger = logging.getLogger(__name__)

async def emit_progress_message(step,messages,emit:Emit,data={}):
	return await emit("progress", {
					"step": step,
					"message": messages,
					"data":data
			}, id_=str(step))
	
class SealionReActAgent:
		"""Enhanced ReAct-style agent for Sealion with better tool parsing"""
		
		def __init__(self, llm: BaseChatModel, tools: List[Tool], verbose: bool = True):
				self.llm = llm
				self.tools = tools
				self.verbose = verbose
				self.tool_map = {tool.name: tool for tool in tools}
				
		def _create_react_prompt(self, query: str, 
                           language: str = "detect"
                           ,origin_country:str=None,
                           destination_country:str=None) -> str:
				"""Create a better structured ReAct prompt"""
				
				tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
				
				# Detect if query is in Indonesian
				is_indonesian = any(word in query.lower() for word in ['saya', 'aku', 'nama', 'sedang', 'tolong', 'bantu'])
			
				system_prompt = f""" 
										You are a Travel Health Assistant helping travelers.  
										Your role is to provide locally available medicines, branded herbal medicines, 
          					or traditional herbal supplements that travelers can immediately buy and consume 
               			for first-line treatment of common illnesses.  

										You have access to these tools:  

										{tool_descriptions}  

										IMPORTANT RULES:  
										- You MUST use these tools to get current information before answering. Do not rely only on internal knowledge.  
										- Always recommend ready-made medicines or branded traditional supplements, not just raw ingredients.  
											- ✅ Example: Tolak Angin (contains ginger, mint, honey, etc.)  
											- ❌ Not acceptable: ginger tea or ginger root.  
										- Prioritize medicines that are easily purchasable in the {destination_country}
          						(pharmacies, convenience stores, traditional medicine shops).  
										- Provide multiple local options if available.  
										- When sending queries to the tools, you MUST use the **local language of the origin or destination country** (e.g., Bahasa Indonesia in Indonesia, Japanese in Japan, Mandarin in China, Thai in Thailand). This ensures you retrieve remedies that are actually sold and recognized locally.  

										Format (must follow strictly):  

										Thought: [your reasoning about what to do]  
										Action: [tool name to use, must be one of: google_search_and_crawl]  
										Action Input: [input for the tool]  
										Observation: [tool result will appear here]  
										... (repeat Thought/Action/Action Input/Observation as needed)  
										Thought: [final reasoning]  
										Final Answer: [your complete answer to the user]  

										Guidance steps (origin-to-destination mapping):  
										1) Origin country context (for comparison only):  
											- Search in {origin_country}, using the local language of {origin_country}, to identify common branded herbal medicines or traditional supplements used for the user’s illness.  
											- This step is for understanding what the traveler might already be familiar with.  
											- Do NOT stop here — continue to the destination search.  

										2) Destination-focused search (required for output):  
											- Search in {destination_country}, using the local language of {destination_country}, for herbal medicines, branded traditional remedies, or supplements that are available for the same illness.  
											- Look for products that match or are equivalent to the origin-country remedies, but available locally.  
											- Prioritize remedies that can be realistically bought at pharmacies, convenience stores, traditional medicine shops, or online marketplaces in {destination_country}.  

										3) Destination-ready output:  
											- Return a list of **specific branded herbal medicines or traditional supplements available in {destination_country}** that the traveler can buy and consume immediately.  
											- For each product include:  
												• Product/brand name (local name)  
												• What it treats (indication)  
												• Form (sachet, syrup, lozenge, capsule, etc.)  
												• Where to buy locally (e.g., pharmacy chains, convenience stores, online marketplaces)  
											- Optionally, mention the origin-country equivalent to help the traveler recognize similarity.  

										Guardrails:  
										- Never return just raw ingredients (e.g., “ginger,” “turmeric leaf”). Only ready-to-consume branded remedies or packaged traditional products.  
										- Always prioritize destination availability over origin familiarity.  
										- Use local-language queries for both origin and destination countries.  

										Question: {query}
										Thought: I need to search for current information about
          					""" 

				
				return system_prompt
		
		async def execute(self, query: str, max_iterations: int = 3,
              						origin_country:str=None,
                           destination_country:str=None,
                           emit:Emit = None,
                           timeout: float = 300.0) -> Dict[str, Any]:
				"""Execute the agent with better tool parsing and execution"""
				
				prompt = self._create_react_prompt(query,origin_country=origin_country,
                           										destination_country=destination_country)
				intermediate_steps = []
				
				# Start the conversation
				messages = [SystemMessage(content=prompt)]
				
				for iteration in range(max_iterations):
						try:
								logger.info(f"Starting iteration {iteration + 1}")
        				await emit_progress_message(iteration + 4,f"Starting iteration {iteration + 1}",emit=emit)
								# Get LLM response using async version
								result = await self.llm._agenerate(messages)
								response_text = result.generations[0].message.content.strip()

								if self.verbose:
									print(f"\n--- Iteration {iteration + 1} ---")
									print(f"LLM Response: {response_text[:300]}...")
         					await emit_progress_message(iteration + 4,f"Starting iteration {iteration + 1} - \
                									LLM Response: {response_text[:100]}",emit=emit)
								if "Final Answer:" in response_text:
									final_answer = response_text.split("Final Answer:")[-1].strip()

									return {
											"output": final_answer,
											"intermediate_steps": intermediate_steps
									}
								
										
								# Parse action with multiple patterns
								action_patterns = [
										r"Action:\s*([a-zA-Z_]+)",
										r"Action:\s*([a-zA-Z_]+)\s*\n",
										r"Action:\s*([a-zA-Z_]+)\s*$"
								]
								
								input_patterns = [
										r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|$))",
										r"Action Input:\s*(.+?)(?:\n|$)",
										r"Action Input:\s*\"(.+?)\"",
										r"Action Input:\s*'(.+?)'",
										r"Action Input:\s*(.+)"
								]
								
								action_name = None
								action_input = None
								
								# Try to find action
								for pattern in action_patterns:
										match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
										if match:
												action_name = match.group(1).strip()
												break
								
								# Try to find input
								for pattern in input_patterns:
										match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
										if match:
												action_input = match.group(1).strip()
												break
								
								if action_name and action_input:
										# Clean up action name and input
										action_name = action_name.lower().replace(' ', '_')
										action_input = action_input.strip('"\'')
										          
										if self.verbose:
												print(f"Executing: {action_name} with input: {action_input[:100]}...")
										
										cc = None
										# Execute the tool
										if action_name in self.tool_map:
												if "country_code=" in action_input: 
													action_input,cc = action_input.split("country_code=")[0],action_input.split("country_code=")[1]
												
												try:
														if cc:
															tool_result = await self.tool_map[action_name].func(action_input,country_code=cc,emit=emit)
														else:
															tool_result = await self.tool_map[action_name].func(action_input,emit=emit)
														# Get the tool result appened in the total steps
														intermediate_steps.append((action_name, action_input, tool_result))
														
														# Add to conversation
														messages.append(AIMessage(content=response_text))
														messages.append(HumanMessage(content=f"Observation: {tool_result}"))
														
														if self.verbose:
																print(f"Tool result: {tool_result[:100]}...")
              
												except Exception as e:
														error_msg = f"Error executing {action_name}: {str(e)}"
														await emit_progress_message(iteration + 4,f"Error encountered, executing recovery function.",
                                          							emit=emit,data={})
														logger.error(error_msg)
														messages.append(AIMessage(content=response_text))
														messages.append(HumanMessage(content=f"Observation: {error_msg}"))
										else:
												error_msg = f"Unknown action: {action_name}. Available actions: {list(self.tool_map.keys())}"
												messages.append(AIMessage(content=response_text))
												messages.append(HumanMessage(content=f"Observation: {error_msg}"))
												
								else:
										# No clear action found, ask for final answer
										if self.verbose:
												print("No action found, asking for final answer")
										await emit_progress_message(iteration + 4,f"Generating Final Answer..",emit=emit)
										await asyncio.sleep(1)
										messages.append(AIMessage(content=response_text))
										messages.append(HumanMessage(content="""Summarize the content of previous conversation to TODOLIST to the user.
																														Only generate response based on provided conversation. Do not make up answers.
																				"""))
										
						except Exception as e:
								logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
								break
				await emit_progress_message(iteration + 4,f"Error happened, executing fallback function to generate final answer.",emit=emit)
				# Fallback response if iterations exhausted
				fallback_msg = """"I need to search for current information first, 
														but let me provide some general guidance based on your query."""
				
				# Try one final generation asking for direct answer
				try:
						fallback_messages = [
								SystemMessage(content=f"""Previous discussion : {messages}.
															You are a helpful travel health assistant. 
															Provide practical advice for the user's health concern while traveling."""),
								HumanMessage(content=query + "\n\nPlease provide practical, actionable advice.")
						]
						
						final_result = await self.llm._agenerate(messages)
						fallback_response = final_result.generations[0].message.content.strip()
						print("Pass through fallback response")
						return {
								"output": fallback_response,
								"intermediate_steps": intermediate_steps
						}
				except:
						return {
								"output": fallback_msg,
								"intermediate_steps": intermediate_steps
						}

class SealionMCPAgent:
	"""Main enhanced Sealion MCP Agent"""

	def __init__(
			self,
			endpoint_name: str,
			region_name: str = "us-east-1",
			temperature: float = 0.3,  # Lower temperature for better tool following
			max_tokens: int = 1024     # Reduced for more focused responses
	):
		
		# Initialize the Chat Model
		self.llm = SageMakerSealionChat(
				endpoint_name=endpoint_name,
				region_name=region_name,
				temperature=temperature,
				max_tokens=max_tokens
		)
		
		# Define tools
		# self.tools = [SageMakerToolsVariable.GOOGLE_SEARCH, 
		# 				SageMakerToolsVariable.URL_CRAWLER, 
		# 				SageMakerToolsVariable.EXTRACT_INFO,
		# 				SageMakerToolsVariable.GOOGLE_SEARCH_WITH_CRAWLING]
		self.tools = [SageMakerToolsVariable.GOOGLE_SEARCH_WITH_CRAWLING]
		
		# Create custom ReAct agent
		self.agent = SealionReActAgent(
				llm=self.llm,
				tools=self.tools,
				verbose=True
		)
		
		logger.info("Successfully initialized Sealion MCP Agent with enhanced tool parsing")
	
	async def invoke(self, query: str, max_iterations: int = 3,
            				origin_country:str=None,
                    destination_country :str=None,
                    emit:Emit=None) -> Dict:
		"""
		Invoke the agent with a query
		
		Args:
			query (str): The query to process
			max_iterations (int): Maximum number of reasoning iterations
			origin_country (str): Origin country for context
			destination_country (str): Destination country for context
			emit (Emit): Callback for progress updates
			
		Returns:
			Dict: Response containing output and intermediate steps
			
		Raises:
			asyncio.TimeoutError: If execution exceeds timeout
			Exception: For other execution errors
		"""
		"""Invoke the agent with a query"""
		return await self.agent.execute(query, max_iterations=max_iterations,
              											origin_country=origin_country,
                           					destination_country = destination_country,
                                		emit=emit)

# Configuration class
class SealionConfig(BaseModel):
	"""Configuration for Sealion MCP Agent"""
	endpoint_name: str = Field(description="SageMaker endpoint name")
	region_name: str = Field(default="us-east-1", description="AWS region")
	temperature: float = Field(default=0.3, description="Temperature for generation")
	max_tokens: int = Field(default=1024, description="Maximum tokens to generate")

def create_sealion_agent(config: SealionConfig) -> SealionMCPAgent:
	"""Factory function to create a Sealion MCP Agent"""
	return SealionMCPAgent(
			endpoint_name=config.endpoint_name,
			region_name=config.region_name,
			temperature=config.temperature,
			max_tokens=config.max_tokens
	)
		
