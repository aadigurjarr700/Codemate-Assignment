import os
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from mistralai import Mistral

# Import your own functions
from functions import FunctionLibrary

load_dotenv()

@dataclass
class FunctionCall:
    function_name: str
    parameters: Dict[str, Any]
    output_variable: str
    depends_on: List[str] = None
    description: str = ""

@dataclass
class ExecutionPlan:
    query: str
    function_calls: List[FunctionCall]
    execution_order: List[int]
    estimated_time: float = 0.0
    confidence_score: float = 0.0

@dataclass
class ExecutionResult:
    """Result of executing a function plan"""
    success: bool
    results: Dict[str, Any]
    errors: List[str]
    execution_time: float
    plan: ExecutionPlan

@dataclass
class ExecutionContext:
    """Stores variables and results during execution"""
    variables: Dict[str, Any]
    execution_log: List[str]
    start_time: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'variables'):
            self.variables = {}
        if not hasattr(self, 'execution_log'):
            self.execution_log = []
        if not hasattr(self, 'start_time'):
            self.start_time = datetime.now()

class QueryAnalyzer:
    def __init__(self):
        self.action_keywords = {
            'retrieve': ['get', 'retrieve', 'fetch', 'find', 'show', 'list'],
            'calculate': ['calculate', 'compute', 'sum', 'total', 'average'],
            'send': ['send', 'email', 'notify', 'alert'],
        }

        self.entity_patterns = {
            'date_range': [
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
            ],
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'data_types': ['invoices', 'customers', 'orders', 'products', 'employees']
        }

    def extract_intent(self, query: str) -> List[str]:
        intents = []
        query_lower = query.lower()
        for intent, keywords in self.action_keywords.items():
            if any(k in query_lower for k in keywords):
                intents.append(intent)
        return intents or ['retrieve']

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {}
        for key, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, query, re.IGNORECASE))
            if matches:
                entities[key] = matches
        return entities

    def parse_time_references(self, query: str) -> Dict[str, str]:
        now = datetime.now()
        if "march 2024" in query.lower():
            return {
                "start_date": "2024-03-01",
                "end_date": "2024-03-31"
            }
        return {}

class LLMInterface:
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.model_name = model_name
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set in .env")
        self.client = Mistral(api_key=self.api_key)

    def generate_function_sequence(self, query: str, available_functions: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(query, available_functions, context)
        
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        
        # Extract JSON from markdown code blocks if present
        json_content = self._extract_json_from_response(content)
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from Mistral:\n{content}")
    
    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from markdown code blocks or return content as-is"""
        # Check if content is wrapped in markdown code blocks
        if "```json" in content:
            # Extract content between ```json and ```
            start_marker = "```json"
            end_marker = "```"
            start_idx = content.find(start_marker) + len(start_marker)
            end_idx = content.find(end_marker, start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()
        elif "```" in content and "{" in content:
            # Handle case where it's just ``` without json specifier
            start_idx = content.find("```") + 3
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                return content[start_idx:end_idx].strip()
        
        # If no code blocks found, return content as-is
        return content.strip()

    def _build_prompt(self, query: str, functions: List[str], context: Dict) -> str:
        return f"""
You are an AI assistant that converts natural language queries into structured function call sequences.

Available functions:
{json.dumps(functions, indent=2)}

User Query: "{query}"

Context:
- intents: {context.get('intents')}
- entities: {context.get('entities')}
- time_info: {context.get('time_info')}

Respond with valid JSON:
{{
  "reasoning": "...",
  "function_calls": [ {{
      "function_name": "...",
      "parameters": {{ ... }},
      "output_variable": "...",
      "depends_on": [ ... ],
      "description": "..."
  }} ],
  "execution_order": [...],
  "confidence": 0-1.0
}}

Only output the JSON. No additional text.
""".strip()

class FunctionExecutor:
    """Executes the planned function sequence"""
    
    def __init__(self, function_library):
        self.function_library = function_library
        self.logger = logging.getLogger(__name__)
        
    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute the complete function plan"""
        start_time = time.time()
        context = ExecutionContext(
            variables={},
            execution_log=[],
            start_time=datetime.now()
        )
        
        errors = []
        results = {}
        
        self.logger.info(f"Starting execution of plan: {plan.query}")
        context.execution_log.append(f"[{datetime.now()}] Starting execution: {plan.query}")
        
        try:
            # Execute functions in the specified order
            for step_idx, func_idx in enumerate(plan.execution_order):
                function_call = plan.function_calls[func_idx]
                
                self.logger.info(f"Step {step_idx + 1}: Executing {function_call.function_name}")
                
                try:
                    # Check dependencies
                    if not self._check_dependencies(function_call, context):
                        error_msg = f"Dependencies not met for {function_call.function_name}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
                        continue
                    
                    # Resolve parameters (replace variables with actual values)
                    resolved_params = self._resolve_parameters(function_call.parameters, context)
                    
                    # Execute the function
                    result = self._execute_function(
                        function_call.function_name, 
                        resolved_params
                    )
                    
                    # Store result in context
                    context.variables[function_call.output_variable] = result
                    results[function_call.output_variable] = result
                    
                    log_msg = f"[{datetime.now()}] ✅ {function_call.function_name} completed"
                    context.execution_log.append(log_msg)
                    self.logger.info(f"Step {step_idx + 1} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Error in {function_call.function_name}: {str(e)}"
                    errors.append(error_msg)
                    log_msg = f"[{datetime.now()}] ❌ {function_call.function_name} failed: {str(e)}"
                    context.execution_log.append(log_msg)
                    self.logger.error(error_msg)
                    
                    # Decide whether to continue or stop
                    if not self._should_continue_on_error(function_call, plan):
                        break
            
            execution_time = time.time() - start_time
            success = len(errors) == 0
            
            # Final log entry
            status = "✅ SUCCESS" if success else "❌ FAILED"
            context.execution_log.append(f"[{datetime.now()}] {status} - Execution completed in {execution_time:.2f}s")
            
            return ExecutionResult(
                success=success,
                results=results,
                errors=errors,
                execution_time=execution_time,
                plan=plan
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical execution error: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            return ExecutionResult(
                success=False,
                results=results,
                errors=errors,
                execution_time=execution_time,
                plan=plan
            )
    
    def _check_dependencies(self, function_call: FunctionCall, context: ExecutionContext) -> bool:
        """Check if all dependencies are satisfied"""
        if not function_call.depends_on:
            return True
            
        for dep in function_call.depends_on:
            if dep not in context.variables:
                self.logger.warning(f"Missing dependency: {dep}")
                return False
        return True
    
    def _resolve_parameters(self, parameters: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Replace variable references with actual values"""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # This is a variable reference
                var_name = value[1:]  # Remove the $ prefix
                if var_name in context.variables:
                    resolved[key] = context.variables[var_name]
                else:
                    self.logger.warning(f"Variable {var_name} not found, using raw value")
                    resolved[key] = value
            else:
                resolved[key] = value
                
        return resolved
    
    def _execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a single function with given parameters"""
        try:
            # Get the actual function from the library
            if hasattr(self.function_library, function_name):
                func = getattr(self.function_library, function_name)
                result = func(**parameters)
                return result
            else:
                # Simulate function execution for demo purposes
                return self._simulate_function_execution(function_name, parameters)
                
        except Exception as e:
            self.logger.error(f"Function execution failed: {function_name} - {str(e)}")
            raise
    
    def _simulate_function_execution(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Simulate function execution when actual implementation is not available"""
        self.logger.info(f"Simulating execution of {function_name} with params: {parameters}")
        
        # Return realistic mock data based on function type
        if "get_" in function_name:
            return {"data": f"Mock data from {function_name}", "count": 42, "parameters": parameters}
        elif "calculate_" in function_name:
            return {"result": 12345.67, "calculation": function_name}
        elif "send_" in function_name:
            return {"status": "sent", "recipient": parameters.get("recipient", "unknown")}
        elif "create_" in function_name:
            return {"created": True, "item_id": "mock_id_123", "type": function_name}
        else:
            return {"status": "completed", "function": function_name, "params": parameters}
    
    def _should_continue_on_error(self, failed_function: FunctionCall, plan: ExecutionPlan) -> bool:
        """Determine if execution should continue after an error"""
        # For now, stop on any error. In a real system, you might want more sophisticated logic
        return False
    
    def print_execution_summary(self, result: ExecutionResult):
        """Print a human-readable summary of the execution"""
        print(f"\n{'='*60}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Query: {result.plan.query}")
        print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        print(f"Functions Executed: {len(result.results)}")
        
        if result.errors:
            print(f"\n❌ ERRORS ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")
        
        if result.results:
            print(f"\n📊 RESULTS:")
            for var_name, value in result.results.items():
                print(f"  {var_name}: {self._format_result(value)}")
        
        print(f"\n{'='*60}")
    
    def _format_result(self, value: Any) -> str:
        """Format result for display"""
        if isinstance(value, dict):
            return json.dumps(value, indent=2)[:200] + "..." if len(str(value)) > 200 else json.dumps(value, indent=2)
        elif isinstance(value, list):
            return f"List with {len(value)} items"
        else:
            return str(value)[:100] + "..." if len(str(value)) > 100 else str(value)

class FunctionOrchestrator:
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.function_library = FunctionLibrary()
        self.query_analyzer = QueryAnalyzer()
        self.llm_interface = LLMInterface(model_name)

    def process_query(self, query: str) -> ExecutionPlan:
        intents = self.query_analyzer.extract_intent(query)
        entities = self.query_analyzer.extract_entities(query)
        time_info = self.query_analyzer.parse_time_references(query)

        context = {
            "intents": intents,
            "entities": entities,
            "time_info": time_info
        }

        available_functions = self.function_library.list_all_functions()
        llm_output = self.llm_interface.generate_function_sequence(query, available_functions, context)

        function_calls = [
            FunctionCall(
                function_name=fc["function_name"],
                parameters=fc["parameters"],
                output_variable=fc["output_variable"],
                depends_on=fc.get("depends_on", []),
                description=fc.get("description", "")
            )
            for fc in llm_output["function_calls"]
        ]

        # Map function names in execution_order to their indices
        name_to_index = {fc.function_name: idx for idx, fc in enumerate(function_calls)}
        execution_order = []
        for name in llm_output["execution_order"]:
            if name in name_to_index:
                execution_order.append(name_to_index[name])
            else:
                # If function name not found, try to find by partial match
                for func_name, idx in name_to_index.items():
                    if name in func_name or func_name in name:
                        execution_order.append(idx)
                        break

        return ExecutionPlan(
            query=query,
            function_calls=function_calls,
            execution_order=execution_order,
            confidence_score=llm_output.get("confidence", 0.0)
        )

    def explain_plan(self, plan: ExecutionPlan) -> str:
        explanation = f"\nQuery: {plan.query}\nConfidence: {plan.confidence_score:.2f}\n"
        for idx in plan.execution_order:
            call = plan.function_calls[idx]
            explanation += f"\nStep {idx + 1}: {call.function_name}\n"
            explanation += f"  Description: {call.description}\n"
            explanation += f"  Params: {json.dumps(call.parameters)}\n"
            if call.depends_on:
                explanation += f"  Depends on: {call.depends_on}\n"
        return explanation

class EnhancedFunctionOrchestrator:
    """Extended orchestrator that can both plan and execute"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.orchestrator = FunctionOrchestrator(model_name)
        self.executor = FunctionExecutor(self.orchestrator.function_library)
    
    def process_and_execute_query(self, query: str) -> ExecutionResult:
        """Process query and execute the plan"""
        # Generate execution plan
        plan = self.orchestrator.process_query(query)
        
        # Show the plan
        print(self.orchestrator.explain_plan(plan))
        
        # Execute the plan
        result = self.executor.execute_plan(plan)
        
        # Show execution summary
        self.executor.print_execution_summary(result)
        
        return result

if __name__ == "__main__":
    # Test the orchestrator - planning only
    orchestrator = FunctionOrchestrator()
    queries = [
        "Get all invoices for March 2024, calculate the total amount, and send me a summary via email"
    ]
    
    for query in queries:
        plan = orchestrator.process_query(query)
        print(orchestrator.explain_plan(plan))
        print(json.dumps({
            "query": plan.query,
            "function_calls": [asdict(fc) for fc in plan.function_calls],
            "execution_order": plan.execution_order,
            "confidence_score": plan.confidence_score
        }, indent=2))
        print("=" * 60)
    
    # Test the enhanced orchestrator - planning and execution
    print("\n" + "="*80)
    print("TESTING ENHANCED ORCHESTRATOR (WITH EXECUTION)")
    print("="*80)
    
    enhanced_orchestrator = EnhancedFunctionOrchestrator()
    
    test_queries = [
        "Get all invoices for March 2024, calculate the total amount, and send me a summary via email",
        "Find low stock products and create purchase orders"
    ]
    
    for query in test_queries:
        print(f"\n🚀 Processing: {query}")
        result = enhanced_orchestrator.process_and_execute_query(query)
        print("\n" + "="*80)