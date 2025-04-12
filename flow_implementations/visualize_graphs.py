import graphviz

# 方法一：自動研究流程
def visualize_method_one():
    dot = graphviz.Digraph('Method One', comment='自動研究流程')
    
    # 添加節點
    dot.node('START', 'START', shape='oval', style='filled', fillcolor='#5D8AA8', fontcolor='white')
    dot.node('perform_deep_research', 'perform_deep_research', shape='box', style='filled', fillcolor='#6495ED', fontcolor='white')
    dot.node('extract_additional_urls', 'extract_additional_urls', shape='box', style='filled', fillcolor='#6495ED', fontcolor='white')
    dot.node('process_urls_with_reader', 'process_urls_with_reader', shape='box', style='filled', fillcolor='#7B68EE', fontcolor='white')
    dot.node('create_embeddings', 'create_embeddings', shape='box', style='filled', fillcolor='#9370DB', fontcolor='white')
    dot.node('store_embeddings', 'store_embeddings', shape='box', style='filled', fillcolor='#9370DB', fontcolor='white')
    dot.node('create_final_summary', 'create_final_summary', shape='box', style='filled', fillcolor='#8A2BE2', fontcolor='white')
    dot.node('END', 'END', shape='oval', style='filled', fillcolor='#4682B4', fontcolor='white')
    
    # 添加邊緣連接
    dot.edge('START', 'perform_deep_research')
    dot.edge('perform_deep_research', 'extract_additional_urls')
    dot.edge('extract_additional_urls', 'process_urls_with_reader')
    dot.edge('process_urls_with_reader', 'create_embeddings')
    dot.edge('create_embeddings', 'store_embeddings')
    dot.edge('store_embeddings', 'create_final_summary')
    dot.edge('create_final_summary', 'END')
    
    # 添加子圖
    with dot.subgraph(name='cluster_research') as c:
        c.attr(label='研究階段')
        c.node('perform_deep_research')
        c.node('extract_additional_urls')
    
    with dot.subgraph(name='cluster_processing') as c:
        c.attr(label='內容處理階段')
        c.node('process_urls_with_reader')
    
    with dot.subgraph(name='cluster_vectorization') as c:
        c.attr(label='向量化階段')
        c.node('create_embeddings')
        c.node('store_embeddings')
    
    with dot.subgraph(name='cluster_summary') as c:
        c.attr(label='摘要階段')
        c.node('create_final_summary')
    
    # 保存圖形
    dot.render('/Users/kevinluo/github_items/new_app_folders_20250405/my-custom-mcp/mcp_RAG/_Jina/flow_implementations/method_one/method_one_graph', format='png', cleanup=True)
    print("Method One graph visualization generated successfully!")

# 方法二：靈活輸入與 RAG 集成
def visualize_method_two():
    dot = graphviz.Digraph('Method Two', comment='靈活輸入與 RAG 集成')
    
    # 添加節點
    dot.node('START', 'START', shape='oval', style='filled', fillcolor='#5D8AA8', fontcolor='white')
    dot.node('route_input_type', 'route_input_type', shape='diamond', style='filled', fillcolor='#FF7F50', fontcolor='white')
    dot.node('process_urls_with_reader', 'process_urls_with_reader', shape='box', style='filled', fillcolor='#6495ED', fontcolor='white')
    dot.node('create_embeddings', 'create_embeddings', shape='box', style='filled', fillcolor='#9370DB', fontcolor='white')
    dot.node('store_embeddings', 'store_embeddings', shape='box', style='filled', fillcolor='#9370DB', fontcolor='white')
    dot.node('process_query_with_rag', 'process_query_with_rag', shape='box', style='filled', fillcolor='#7B68EE', fontcolor='white')
    dot.node('route_completion', 'route_completion', shape='diamond', style='filled', fillcolor='#FF7F50', fontcolor='white')
    dot.node('END', 'END', shape='oval', style='filled', fillcolor='#4682B4', fontcolor='white')
    
    # 添加邊緣連接
    dot.edge('START', 'route_input_type')
    dot.edge('route_input_type', 'process_urls_with_reader', label='URL輸入')
    dot.edge('route_input_type', 'process_query_with_rag', label='查詢輸入')
    dot.edge('process_urls_with_reader', 'create_embeddings')
    dot.edge('create_embeddings', 'store_embeddings')
    dot.edge('store_embeddings', 'route_completion')
    dot.edge('process_query_with_rag', 'route_completion')
    dot.edge('route_completion', 'START', label='返回起始節點')
    dot.edge('route_completion', 'END', label='結束會話')
    
    # 保存圖形
    dot.render('/Users/kevinluo/github_items/new_app_folders_20250405/my-custom-mcp/mcp_RAG/_Jina/flow_implementations/method_two/method_two_graph', format='png', cleanup=True)
    print("Method Two graph visualization generated successfully!")

# 方法三：Agentic RAG 智能助手
def visualize_method_three():
    dot = graphviz.Digraph('Method Three', comment='Agentic RAG 智能助手')
    
    # 添加節點
    dot.node('START', 'START', shape='oval', style='filled', fillcolor='#5D8AA8', fontcolor='white')
    dot.node('input_analyzer', 'input_analyzer', shape='box', style='filled', fillcolor='#6495ED', fontcolor='white')
    dot.node('task_planner', 'task_planner', shape='box', style='filled', fillcolor='#6495ED', fontcolor='white')
    dot.node('tool_selector', 'tool_selector', shape='box', style='filled', fillcolor='#FF7F50', fontcolor='white')
    dot.node('task_executor', 'task_executor', shape='box', style='filled', fillcolor='#7B68EE', fontcolor='white')
    dot.node('reflection_engine', 'reflection_engine', shape='diamond', style='filled', fillcolor='#FF7F50', fontcolor='white')
    dot.node('response_generator', 'response_generator', shape='box', style='filled', fillcolor='#8A2BE2', fontcolor='white')
    dot.node('END', 'END', shape='oval', style='filled', fillcolor='#4682B4', fontcolor='white')
    
    # 添加邊緣連接
    dot.edge('START', 'input_analyzer')
    dot.edge('input_analyzer', 'task_planner')
    dot.edge('task_planner', 'tool_selector')
    dot.edge('tool_selector', 'task_executor')
    dot.edge('task_executor', 'reflection_engine')
    dot.edge('reflection_engine', 'tool_selector', label='需要更多研究')
    dot.edge('reflection_engine', 'response_generator', label='研究完成')
    dot.edge('response_generator', 'END')
    
    # 保存圖形
    dot.render('/Users/kevinluo/github_items/new_app_folders_20250405/my-custom-mcp/mcp_RAG/_Jina/flow_implementations/method_three/method_three_graph', format='png', cleanup=True)
    print("Method Three graph visualization generated successfully!")

if __name__ == "__main__":
    visualize_method_one()
    visualize_method_two()
    visualize_method_three()
    print("All graph visualizations generated successfully!")
