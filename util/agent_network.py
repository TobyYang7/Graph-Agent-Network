import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from agent import Agent

class AgentNetwork:
    def __init__(self, adjacency_matrix, system_prompts=None):
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.num_agents = len(self.adjacency_matrix)
        
        if system_prompts is None:
            system_prompts = ["You are a helpful assistant."] * self.num_agents
        elif len(system_prompts) != self.num_agents:
            raise ValueError("Number of system prompts must match the number of agents")
        
        self.agents = [Agent(system_prompt) for system_prompt in system_prompts]
        
    def process_inputs(self, inputs, **kwargs):
        if len(inputs) != self.num_agents:
            raise ValueError("Number of inputs must match the number of agents")
        
        responses = []
        for i, user_input in enumerate(inputs):
            response = self.agents[i].get_response(user_input, **kwargs)
            responses.append({
                'agent_index': i,
                'input': user_input,
                'output': response['content'],
                'total_tokens': response['total_tokens']
            })
        
        return responses

    def get_connected_agents(self, agent_index):
        return [i for i, connected in enumerate(self.adjacency_matrix[agent_index]) if connected]

# 使用示例
if __name__ == "__main__":
    # 示例邻接矩阵（3个agents的网络）
    adjacency_matrix = [
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ]
    
    # 自定义system prompts
    system_prompts = [
        "You are a creative storyteller.",
        "You are a logical problem solver.",
        "You are an empathetic counselor."
    ]
    
    network = AgentNetwork(adjacency_matrix, system_prompts)
    
    # 准备输入
    inputs = [
        "Tell me a short story about friendship.",
        "How can we solve the problem of climate change?",
        "How can I improve my communication skills?"
    ]
    
    # 处理输入并获取响应
    responses = network.process_inputs(inputs)
    
    # 打印结果
    for response in responses:
        print(f"\nAgent {response['agent_index']}:")
        print(f"Input: {response['input']}")
        print(f"Output: {response['output']}")
        print(f"Total tokens: {response['total_tokens']}")
        
        # 打印连接的agents
        connected_agents = network.get_connected_agents(response['agent_index'])
        print(f"Connected to agents: {connected_agents}")