
import boto3
import uuid
import json
import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

bedrock_agent_client = boto3.client('bedrock-agent')
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

###########################################################
# The initial restaurant booking agent is created by cdk:  
# https://github.com/aws-samples/amazon-bedrock-samples/tree/main/agents-and-function-calling/bedrock-agents/features-examples/13-create-agent-using-CDK
booking_agent_id = "5SEE4A1JDS" 
agent_alias_id = "8QL4VGLSD6"

# Define a MCP Client class
class MCPClient:
    # Use an Amazon Bedrock model
    MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    # connect to the server  
    async def connect_to_server(self, server_script_path: str):
        # the server script must be a python or nodejs script 
        if not server_script_path.endswith(('.py', '.js')):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        # use stdio transport
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

    # get MCP tools info and transform them to Bedrock agent function definitions
    async def get_tools(self):         
        # get the list of tools from the MCP server
        response = await self.session.list_tools()

        # extract the tool information
        mcp_tools = [{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        } for tool in response.tools]

        # convert the MCP tool info to Bedrock agent function definitions 
        agent_functions = self._make_bedrock_agent_functions_definitions(mcp_tools)
        return agent_functions
    
    # an extra step for format transformation 
    def _make_bedrock_agent_functions_definitions(self, format_a):
        format_b = []

        for function in format_a:
            # Create new function dictionary with basic properties
            new_function = {
                'name': function['name'],
                'description': function['description'],
                'parameters': {}
            }
            
            # Convert parameters
            properties = function['parameters']['properties']
            required_params = function['parameters']['required']
            
            for param_name, param_info in properties.items():
                new_function['parameters'][param_name] = {
                    'description': param_info['title'],
                    'required': param_name in required_params,
                    'type': param_info['type']
                }
            
            format_b.append(new_function)    
        return format_b
    
    # add mcp tools to a Return of Control action group in the agent
    async def update_bedrock_agent(self, agent_functions):
        agent_action_group_name = "mcp_tools"
        agent_action_group_description= "Actions for getting the weather or weather alert based on the input two-letter US state code"

        # create a new action group 
        agent_action_group_response = bedrock_agent_client.create_agent_action_group(
            agentId=booking_agent_id,
            agentVersion='DRAFT',
            actionGroupExecutor={
                'customControl': 'RETURN_CONTROL'
            },
            actionGroupName=agent_action_group_name,
            functionSchema={
                'functions': agent_functions
            },
            description=agent_action_group_description
        )

        # prepare the agent
        response = bedrock_agent_client.prepare_agent(
            agentId=booking_agent_id
        )
        print(response)

    # handle a tool call to MCP server and return the result
    async def handle_tool_call(self, function_name, function_args):

        # a simple format transformation from the ROC function call to the MCP tool call
        print(f"Function name: {function_name}")
        tool_name = function_name
        tool_args = {item['name']: item['value'] for item in function_args}
      
        print(f"Tool name: {tool_name}")
        print(f"Tool args: {tool_args}")

        # tool call to the MCP server
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    # send a query to the agent and get a response
    async def chat(self, query):

        # invoke bedrock agent
        # create a random id for session initiator id
        session_id:str = str(uuid.uuid1())
        enable_trace:bool = False
        end_session:bool = False

        print("\ninvoke the agent ...")
        # invoke the agent API
        agentResponse = bedrock_agent_runtime_client.invoke_agent(
            inputText=query,
            agentId=booking_agent_id,
            agentAliasId=agent_alias_id, 
            sessionId=session_id,
            enableTrace=enable_trace, 
            endSession= end_session
        )
        event_stream = agentResponse['completion']

        # process the agent output
        function_call = None
        agent_answer = None
        try:
            for event in event_stream:
                if 'returnControl' in event:
                    function_call = event
                elif 'chunk' in event:
                    data = event['chunk']['bytes']
                    agent_answer = data.decode('utf8')
                elif 'trace' in event:
                    print.info(json.dumps(event['trace'], indent=2))
                else:
                    raise Exception("unexpected event.", event)
        except Exception as e:
            raise Exception("unexpected event.", e)

        if function_call != None:
            print("\nreturn function call at the local host ...")
            # extract the info fromt the ROC function call
            function_name = function_call["returnControl"]["invocationInputs"][0]["functionInvocationInput"]["function"]
            function_args = function_call["returnControl"]["invocationInputs"][0]["functionInvocationInput"]["parameters"]

            # make mcp tool call to the MCP server
            tool_response = await self.handle_tool_call(function_name, function_args)
            tool_result = tool_response.content[0].text
            print(tool_result)
            
            # invoke agent the second time with the function call result
            print("\ninvoke the agent the second time due to ROC ...")
            agentResponse = bedrock_agent_runtime_client.invoke_agent(
                agentId=booking_agent_id,
                agentAliasId=agent_alias_id, 
                sessionId=session_id,
                enableTrace=enable_trace, 
                sessionState={
                    'invocationId': function_call["returnControl"]["invocationId"],
                    'returnControlInvocationResults': [{
                            'functionResult': {
                                'actionGroup': function_call["returnControl"]["invocationInputs"][0]["functionInvocationInput"]["actionGroup"],
                                'function': function_call["returnControl"]["invocationInputs"][0]["functionInvocationInput"]["function"],
                                'responseBody': {
                                    "TEXT": {
                                        'body': tool_result
                                    }
                                }
                            }
                            }]}
            )
            # print(agentResponse)
            event_stream = agentResponse['completion']

            # process agent output
            agent_answer = None
            try:
                for event in event_stream:      
                    if 'chunk' in event:
                        data = event['chunk']['bytes']
                        agent_answer = data.decode('utf8')
                    elif 'trace' in event:
                        print.info(json.dumps(event['trace'], indent=2))
                    else:
                        raise Exception("unexpected event.", event)
            except Exception as e:
                raise Exception("unexpected event.", e)

        return agent_answer
    
    # From: https://community.aws/content/2uFvyCPQt7KcMxD9ldsJyjZM1Wp/model-context-protocol-mcp-and-amazon-bedrock
    async def chat_loop(self):
        print("\nMCP Client Started!\nType your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.chat(query)
                print("\nResponse:")
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

# run the MCP client 
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    await client.connect_to_server(sys.argv[1])

    mcp_tools = await client.get_tools()
    print("\nmcp_tools:\n", mcp_tools)

    await client.update_bedrock_agent(mcp_tools)
    print("\nupdated bedrock agent action groups")

    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
