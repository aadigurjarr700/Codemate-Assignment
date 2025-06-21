import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Import your existing classes
from execute import ExecutionPlan, FunctionCall, ExecutionResult

logger = logging.getLogger(__name__)

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

# Enhanced FunctionOrchestrator with execution capability
class EnhancedFunctionOrchestrator:
    """Extended orchestrator that can both plan and execute"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        from execute import FunctionOrchestrator  # Import your existing class
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
    # Test the enhanced orchestrator
    enhanced_orchestrator = EnhancedFunctionOrchestrator()
    
    test_queries = [
        "Get all invoices for March 2024, calculate the total amount, and send me a summary via email",
        "Find low stock products and create purchase orders"
    ]
    
    for query in test_queries:
        print(f"\n🚀 Processing: {query}")
        result = enhanced_orchestrator.process_and_execute_query(query)
        print("\n" + "="*80)