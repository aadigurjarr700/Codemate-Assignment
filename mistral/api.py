# app.py - Complete Flask backend for Function Orchestrator
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your existing classes
try:
    from execute import ExecutionPlan, FunctionCall, ExecutionResult, FunctionOrchestrator
except ImportError as e:
    logger.error(f"Failed to import from execute.py: {e}")
    # Create mock classes for development/testing
    @dataclass
    class ExecutionPlan:
        query: str
        function_calls: List[Any]
        execution_order: List[int]
    
    @dataclass
    class FunctionCall:
        function_name: str
        parameters: Dict[str, Any]
        output_variable: str
        depends_on: List[str]
    
    @dataclass
    class ExecutionResult:
        success: bool
        results: Dict[str, Any]
        errors: List[str]
        execution_time: float
        plan: ExecutionPlan
    
    class FunctionOrchestrator:
        def __init__(self, model_name: str = "mistral-large-latest"):
            self.function_library = MockFunctionLibrary()
        
        def process_query(self, query: str) -> ExecutionPlan:
            return ExecutionPlan(
                query=query,
                function_calls=[],
                execution_order=[]
            )
        
        def explain_plan(self, plan: ExecutionPlan) -> str:
            return f"Mock plan explanation for: {plan.query}"
    
    class MockFunctionLibrary:
        def get_invoices(self, **kwargs):
            return {"invoices": ["invoice1", "invoice2"], "count": 2}
        
        def calculate_total(self, **kwargs):
            return {"total": 5000.00}

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
                if func_idx >= len(plan.function_calls):
                    continue
                    
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
        if "get_" in function_name or "fetch_" in function_name:
            return {"data": f"Mock data from {function_name}", "count": 42, "parameters": parameters}
        elif "calculate_" in function_name or "compute_" in function_name:
            return {"result": 12345.67, "calculation": function_name}
        elif "send_" in function_name or "email_" in function_name:
            return {"status": "sent", "recipient": parameters.get("recipient", "unknown")}
        elif "create_" in function_name or "generate_" in function_name:
            return {"created": True, "item_id": "mock_id_123", "type": function_name}
        else:
            return {"status": "completed", "function": function_name, "params": parameters}
    
    def _should_continue_on_error(self, failed_function: FunctionCall, plan: ExecutionPlan) -> bool:
        """Determine if execution should continue after an error"""
        # For now, stop on any error. In a real system, you might want more sophisticated logic
        return False

class EnhancedFunctionOrchestrator:
    """Extended orchestrator that can both plan and execute"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        try:
            self.orchestrator = FunctionOrchestrator(model_name)
        except Exception as e:
            logger.warning(f"Failed to initialize FunctionOrchestrator: {e}, using mock")
            self.orchestrator = FunctionOrchestrator(model_name)
        
        self.executor = FunctionExecutor(self.orchestrator.function_library)
    
    def process_and_execute_query(self, query: str) -> ExecutionResult:
        """Process query and execute the plan"""
        # Generate execution plan
        plan = self.orchestrator.process_query(query)
        
        # Execute the plan
        result = self.executor.execute_plan(plan)
        
        return result

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize orchestrator
try:
    orchestrator = EnhancedFunctionOrchestrator()
    logger.info("Function orchestrator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize orchestrator: {e}")
    orchestrator = None

@app.route('/')
def index():
    """Serve the main frontend page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({"error": "Frontend template not found"}), 404

@app.route('/api/process', methods=['POST'])
def process_query():
    """Process a query and return execution results"""
    try:
        if not orchestrator:
            return jsonify({
                'success': False,
                'error': 'Orchestrator not initialized'
            }), 500
            
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        query = data['query']
        logger.info(f"Processing query: {query}")
        
        # Process and execute the query
        result = orchestrator.process_and_execute_query(query)
        
        # Convert result to JSON-serializable format
        response_data = {
            'success': result.success,
            'results': result.results,
            'errors': result.errors,
            'execution_time': result.execution_time,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'plan': {
                'query': result.plan.query,
                'function_calls': [
                    {
                        'function_name': fc.function_name,
                        'parameters': fc.parameters,
                        'output_variable': fc.output_variable,
                        'depends_on': fc.depends_on
                    } for fc in result.plan.function_calls
                ],
                'execution_order': result.plan.execution_order
            }
        }
        
        logger.info(f"Query processed successfully: {result.success}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/plan', methods=['POST'])
def get_execution_plan():
    """Get execution plan without executing"""
    try:
        if not orchestrator:
            return jsonify({
                'success': False,
                'error': 'Orchestrator not initialized'
            }), 500
            
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        query = data['query']
        logger.info(f"Getting plan for query: {query}")
        
        # Generate plan only
        plan = orchestrator.orchestrator.process_query(query)
        explanation = orchestrator.orchestrator.explain_plan(plan)
        
        response_data = {
            'success': True,
            'query': query,
            'plan': {
                'query': plan.query,
                'function_calls': [
                    {
                        'function_name': fc.function_name,
                        'parameters': fc.parameters,
                        'output_variable': fc.output_variable,
                        'depends_on': fc.depends_on
                    } for fc in plan.function_calls
                ],
                'execution_order': plan.execution_order
            },
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Plan generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting plan: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/functions', methods=['GET'])
def get_available_functions():
    """Get list of available functions"""
    try:
        if not orchestrator:
            return jsonify({
                'success': False,
                'error': 'Orchestrator not initialized'
            }), 500
            
        functions = []
        if hasattr(orchestrator.orchestrator, 'function_library'):
            lib = orchestrator.orchestrator.function_library
            for attr_name in dir(lib):
                if not attr_name.startswith('_') and callable(getattr(lib, attr_name)):
                    func = getattr(lib, attr_name)
                    functions.append({
                        'name': attr_name,
                        'docstring': func.__doc__ if func.__doc__ else 'No description available'
                    })
        
        return jsonify({
            'success': True,
            'functions': functions,
            'total_count': len(functions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting functions: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Function Orchestrator API',
        'orchestrator_status': 'initialized' if orchestrator else 'failed'
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed system status"""
    try:
        status_info = {
            'service': 'Function Orchestrator API',
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'orchestrator_initialized': orchestrator is not None,
            'available_endpoints': [
                'GET /',
                'POST /api/process',
                'POST /api/plan', 
                'GET /api/functions',
                'GET /api/health',
                'GET /api/status'
            ]
        }
        
        if orchestrator:
            try:
                # Try to get function count
                lib = orchestrator.orchestrator.function_library
                function_count = len([attr for attr in dir(lib) 
                                    if not attr.startswith('_') and callable(getattr(lib, attr))])
                status_info['available_functions'] = function_count
            except Exception:
                status_info['available_functions'] = 'unknown'
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({
            'service': 'Function Orchestrator API',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'POST /api/process',
            'POST /api/plan',
            'GET /api/functions',
            'GET /api/health',
            'GET /api/status'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        logger.info("Created templates directory")
    
    # Print startup information
    print("\n" + "="*60)
    print("🚀 FUNCTION ORCHESTRATOR WEB API")
    print("="*60)
    print(f"📍 API Base URL: http://localhost:5000")
    print(f"🌐 Frontend URL: http://localhost:5000")
    print(f"📊 Health Check: http://localhost:5000/api/health")
    print(f"📋 Status: http://localhost:5000/api/status")
    print("="*60)
    print("Available Endpoints:")
    print("  GET  /                 - Frontend interface")
    print("  POST /api/process      - Execute query")
    print("  POST /api/plan         - Get execution plan")
    print("  GET  /api/functions    - List available functions")
    print("  GET  /api/health       - Health check")
    print("  GET  /api/status       - Detailed status")
    print("="*60)
    
    if orchestrator:
        print("✅ Orchestrator initialized successfully")
    else:
        print("❌ Orchestrator failed to initialize (running in mock mode)")
    
    print("Starting server...")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5002,
        threaded=True
    )