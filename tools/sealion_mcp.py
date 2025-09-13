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


logger = logging.getLogger(__name__)

class SealionReActAgent:
		"""Enhanced ReAct-style agent for Sealion with better tool parsing"""
		
		def __init__(self, llm: BaseChatModel, tools: List[Tool], verbose: bool = True):
				self.llm = llm
				self.tools = tools
				self.verbose = verbose
				self.tool_map = {tool.name: tool for tool in tools}
				
		def _create_react_prompt(self, query: str, language: str = "detect") -> str:
				"""Create a better structured ReAct prompt"""
				
				tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
				
				# Detect if query is in Indonesian
				is_indonesian = any(word in query.lower() for word in ['saya', 'aku', 'nama', 'sedang', 'tolong', 'bantu'])
			
				system_prompt = f"""You are a travel health assistant helping travelers. 
								Your task  focus searching on remedies or herbal medicine atau traditional herbal 
								supplement that can ease the user's illness (first treatment focused).
								You have access to these tools:
								
								{tool_descriptions}
								
								IMPORTANT: You MUST use these tools to get current information before answering. 
								Don't rely only on internal knowledge.
								
								Use this exact format:
								Thought: [your reasoning about what to do]
								Action: [tool name to use, must be one of: google_search_and_crawl]
								Action Input: [input for the tool]
								Observation: [tool result will appear here]
								... (repeat Thought/Action/Action Input/Observation as needed)
								Thought: [final reasoning]
								Final Answer: [your complete answer to the user]
								
								Tool usage recommendations:
								- Use 'google_search_and_crawl' to get comprehensive information from multiple sources at once
								
								Guidance task to do:
								1. Search regarding the illness and the remedies taken in user's origin country.
								2. Search the remedies or herbal medicine atau traditional herbal supplement in destination country or local context.
								3. You must provide remedies or herbal medicine atau traditional herbal supplement that can be found locally especially
								in user's destination country.

								Question: {query}
								Thought: I need to search for current information about"""
				
				return system_prompt
		
		def execute(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
				"""Execute the agent with better tool parsing and execution"""
				
				prompt = self._create_react_prompt(query)
				intermediate_steps = []
				
				# Start the conversation
				messages = [SystemMessage(content=prompt)]
				
				for iteration in range(max_iterations):
						try:
								logger.info(f"Starting iteration {iteration + 1}")
								
								# Get LLM response
								result = self.llm._generate(messages)
								response_text = result.generations[0].message.content.strip()
								print("Raw Result from LLM Generation: ",response_text)

								if self.verbose:
									print(f"\n--- Iteration {iteration + 1} ---")
									print(f"LLM Response: {response_text[:300]}...")
								
								# Check for Final Answer
								if "Final Answer:" in response_text:
									final_answer = response_text.split("Final Answer:")[-1].strip()
									print("Pass through final answer: ")
									print("-"*100)
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
															tool_result = self.tool_map[action_name].func(action_input,country_code=cc)
														else:
															tool_result = self.tool_map[action_name].func(action_input)
														intermediate_steps.append((action_name, action_input, tool_result))
														
														# Add to conversation
														messages.append(AIMessage(content=response_text))
														messages.append(HumanMessage(content=f"Observation: {tool_result}"))
														
														if self.verbose:
																print(f"Tool result: {tool_result[:200]}...")
																
												except Exception as e:
														error_msg = f"Error executing {action_name}: {str(e)}"
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
										
										messages.append(AIMessage(content=response_text))
										messages.append(HumanMessage(content="""Summarize the content of previous conversation to TODOLIST to the user.
																				Only generate response based on 
																				"""))
										
						except Exception as e:
								logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
								break
				
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
						
						final_result = self.llm._generate(fallback_messages)
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
	
	def invoke(self, query: str, max_iterations: int = 3) -> Dict:
		"""Invoke the agent with a query"""
		return self.agent.execute(query, max_iterations=max_iterations)

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
		
