"""
Function Library for AI Pipeline
50 Business Functions with Clear Descriptions
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class FunctionLibrary:
    """
    A comprehensive library of 50 business functions that can be called by the AI model
    to fulfill various user requests.
    """
    
    def __init__(self):
        self.functions = {
            # DATA RETRIEVAL FUNCTIONS (1-10)
            "get_invoices": {
                "description": "Retrieve invoices from database based on date range or filters",
                "inputs": {
                    "start_date": "string (YYYY-MM-DD)",
                    "end_date": "string (YYYY-MM-DD)", 
                    "customer_id": "string (optional)",
                    "status": "string (paid/pending/overdue, optional)"
                },
                "outputs": {"invoices": "list of invoice objects"},
                "example": "get_invoices('2024-03-01', '2024-03-31')"
            },
            
            "get_customers": {
                "description": "Retrieve customer information from CRM database",
                "inputs": {
                    "customer_id": "string (optional)",
                    "name_filter": "string (optional)",
                    "active_only": "boolean (default: true)"
                },
                "outputs": {"customers": "list of customer objects"},
                "example": "get_customers(name_filter='ABC Corp')"
            },
            
            "get_products": {
                "description": "Retrieve product catalog information",
                "inputs": {
                    "category": "string (optional)",
                    "price_range": "dict with min/max (optional)",
                    "in_stock": "boolean (optional)"
                },
                "outputs": {"products": "list of product objects"},
                "example": "get_products(category='electronics', in_stock=true)"
            },
            
            "get_orders": {
                "description": "Retrieve order history and details",
                "inputs": {
                    "start_date": "string (YYYY-MM-DD)",
                    "end_date": "string (YYYY-MM-DD)",
                    "customer_id": "string (optional)",
                    "status": "string (optional)"
                },
                "outputs": {"orders": "list of order objects"},
                "example": "get_orders('2024-01-01', '2024-03-31', status='completed')"
            },
            
            "get_employee_data": {
                "description": "Retrieve employee information and records",
                "inputs": {
                    "employee_id": "string (optional)",
                    "department": "string (optional)",
                    "active_only": "boolean (default: true)"
                },
                "outputs": {"employees": "list of employee objects"},
                "example": "get_employee_data(department='sales')"
            },
            
            "get_financial_reports": {
                "description": "Retrieve financial reports and statements",
                "inputs": {
                    "report_type": "string (income/balance/cashflow)",
                    "period": "string (monthly/quarterly/yearly)",
                    "year": "integer",
                    "month": "integer (optional)"
                },
                "outputs": {"report": "financial report object"},
                "example": "get_financial_reports('income', 'monthly', 2024, 3)"
            },
            
            "get_inventory": {
                "description": "Retrieve current inventory levels and stock information",
                "inputs": {
                    "product_id": "string (optional)",
                    "location": "string (optional)",
                    "low_stock_only": "boolean (default: false)"
                },
                "outputs": {"inventory": "list of inventory items"},
                "example": "get_inventory(low_stock_only=true)"
            },
            
            "get_sales_data": {
                "description": "Retrieve sales performance data and metrics",
                "inputs": {
                    "start_date": "string (YYYY-MM-DD)",
                    "end_date": "string (YYYY-MM-DD)",
                    "salesperson_id": "string (optional)",
                    "region": "string (optional)"
                },
                "outputs": {"sales_data": "sales metrics object"},
                "example": "get_sales_data('2024-03-01', '2024-03-31', region='north')"
            },
            
            "get_support_tickets": {
                "description": "Retrieve customer support tickets and issues",
                "inputs": {
                    "status": "string (open/closed/pending)",
                    "priority": "string (high/medium/low, optional)",
                    "start_date": "string (YYYY-MM-DD, optional)",
                    "customer_id": "string (optional)"
                },
                "outputs": {"tickets": "list of support ticket objects"},
                "example": "get_support_tickets('open', priority='high')"
            },
            
            "get_user_activities": {
                "description": "Retrieve user activity logs and system usage data",
                "inputs": {
                    "user_id": "string (optional)",
                    "start_date": "string (YYYY-MM-DD)",
                    "end_date": "string (YYYY-MM-DD)",
                    "activity_type": "string (optional)"
                },
                "outputs": {"activities": "list of activity objects"},
                "example": "get_user_activities(start_date='2024-03-01', end_date='2024-03-31')"
            },
            
            # CALCULATION & ANALYSIS FUNCTIONS (11-20)
            "calculate_total": {
                "description": "Calculate sum of numerical values from a dataset",
                "inputs": {
                    "data": "list or dict",
                    "field": "string (field name to sum)"
                },
                "outputs": {"total": "number"},
                "example": "calculate_total(invoice_data, 'amount')"
            },
            
            "calculate_average": {
                "description": "Calculate average of numerical values",
                "inputs": {
                    "data": "list or dict",
                    "field": "string (field name)"
                },
                "outputs": {"average": "number"},
                "example": "calculate_average(sales_data, 'revenue')"
            },
            
            "analyze_trends": {
                "description": "Analyze data trends over time periods",
                "inputs": {
                    "data": "list of data points",
                    "time_field": "string",
                    "value_field": "string",
                    "period": "string (daily/weekly/monthly)"
                },
                "outputs": {"trend_analysis": "trend analysis object"},
                "example": "analyze_trends(sales_data, 'date', 'amount', 'monthly')"
            },
            
            "generate_summary": {
                "description": "Generate text summary of data or analysis results",
                "inputs": {
                    "data": "any data object",
                    "summary_type": "string (brief/detailed)",
                    "focus_areas": "list of strings (optional)"
                },
                "outputs": {"summary": "string"},
                "example": "generate_summary(financial_data, 'detailed', ['revenue', 'expenses'])"
            },
            
            "calculate_percentage": {
                "description": "Calculate percentage changes or ratios",
                "inputs": {
                    "current_value": "number",
                    "previous_value": "number",
                    "calculation_type": "string (change/ratio)"
                },
                "outputs": {"percentage": "number"},
                "example": "calculate_percentage(150000, 120000, 'change')"
            },
            
            "perform_forecast": {
                "description": "Generate forecasts based on historical data",
                "inputs": {
                    "historical_data": "list of data points",
                    "periods_ahead": "integer",
                    "method": "string (linear/exponential)"
                },
                "outputs": {"forecast": "forecast object"},
                "example": "perform_forecast(monthly_sales, 6, 'linear')"
            },
            
            "calculate_roi": {
                "description": "Calculate return on investment metrics",
                "inputs": {
                    "investment": "number",
                    "returns": "number",
                    "time_period": "number (years)"
                },
                "outputs": {"roi": "ROI metrics object"},
                "example": "calculate_roi(50000, 65000, 1)"
            },
            
            "analyze_performance": {
                "description": "Analyze performance metrics and KPIs",
                "inputs": {
                    "data": "performance data object",
                    "metrics": "list of metric names",
                    "benchmark": "dict (optional)"
                },
                "outputs": {"performance_analysis": "analysis object"},
                "example": "analyze_performance(sales_data, ['conversion_rate', 'avg_deal_size'])"
            },
            
            "calculate_variance": {
                "description": "Calculate variance between actual and budgeted/expected values",
                "inputs": {
                    "actual_values": "list of numbers",
                    "expected_values": "list of numbers"
                },
                "outputs": {"variance_analysis": "variance object"},
                "example": "calculate_variance([100, 150, 200], [90, 140, 190])"
            },
            
            "generate_insights": {
                "description": "Generate business insights from analyzed data",
                "inputs": {
                    "analysis_results": "analysis object",
                    "context": "string (business context)",
                    "insight_type": "string (operational/strategic)"
                },
                "outputs": {"insights": "list of insight strings"},
                "example": "generate_insights(trend_analysis, 'quarterly_review', 'strategic')"
            },
            
            # COMMUNICATION FUNCTIONS (21-30)
            "send_email": {
                "description": "Send email to specified recipients",
                "inputs": {
                    "to": "list of email addresses",
                    "subject": "string",
                    "body": "string",
                    "cc": "list of email addresses (optional)",
                    "attachments": "list of file paths (optional)"
                },
                "outputs": {"status": "success/failure message"},
                "example": "send_email(['user@company.com'], 'Monthly Report', summary_text)"
            },
            
            "send_sms": {
                "description": "Send SMS text message",
                "inputs": {
                    "phone_number": "string",
                    "message": "string",
                    "urgent": "boolean (default: false)"
                },
                "outputs": {"status": "delivery status"},
                "example": "send_sms('+1234567890', 'Alert: Low inventory detected')"
            },
            
            "create_notification": {
                "description": "Create in-app notification for users",
                "inputs": {
                    "user_id": "string",
                    "title": "string",
                    "message": "string",
                    "priority": "string (high/medium/low)",
                    "action_required": "boolean"
                },
                "outputs": {"notification_id": "string"},
                "example": "create_notification('user123', 'Report Ready', 'Your report is ready for review')"
            },
            
            "schedule_meeting": {
                "description": "Schedule meeting in calendar system",
                "inputs": {
                    "title": "string",
                    "start_time": "string (ISO datetime)",
                    "duration": "integer (minutes)",
                    "attendees": "list of email addresses",
                    "location": "string (optional)"
                },
                "outputs": {"meeting_id": "string"},
                "example": "schedule_meeting('Review Meeting', '2024-03-15T14:00:00', 60, ['team@company.com'])"
            },
            
            "send_slack_message": {
                "description": "Send message to Slack channel or user",
                "inputs": {
                    "channel": "string",
                    "message": "string",
                    "mention_users": "list of user IDs (optional)"
                },
                "outputs": {"message_timestamp": "string"},
                "example": "send_slack_message('#general', 'Monthly report completed', ['@john'])"
            },
            
            "create_announcement": {
                "description": "Create company-wide announcement",
                "inputs": {
                    "title": "string",
                    "content": "string",
                    "target_audience": "string (all/department/role)",
                    "expiry_date": "string (YYYY-MM-DD, optional)"
                },
                "outputs": {"announcement_id": "string"},
                "example": "create_announcement('System Maintenance', 'Scheduled downtime notice', 'all')"
            },
            
            "send_webhook": {
                "description": "Send data to external webhook endpoint",
                "inputs": {
                    "url": "string",
                    "data": "dict",
                    "headers": "dict (optional)",
                    "method": "string (POST/PUT, default: POST)"
                },
                "outputs": {"response": "webhook response object"},
                "example": "send_webhook('https://api.external.com/hook', report_data)"
            },
            
            "create_alert": {
                "description": "Create system alert for monitoring",
                "inputs": {
                    "alert_type": "string",
                    "message": "string",
                    "severity": "string (critical/warning/info)",
                    "recipients": "list of contact methods"
                },
                "outputs": {"alert_id": "string"},
                "example": "create_alert('inventory', 'Low stock alert', 'warning', ['email', 'sms'])"
            },
            
            "update_dashboard": {
                "description": "Update real-time dashboard with new data",
                "inputs": {
                    "dashboard_id": "string",
                    "widget_id": "string",
                    "data": "dict",
                    "refresh_interval": "integer (seconds, optional)"
                },
                "outputs": {"update_status": "success/failure"},
                "example": "update_dashboard('main_dash', 'sales_widget', sales_summary)"
            },
            
            "log_activity": {
                "description": "Log activity or event in system audit trail",
                "inputs": {
                    "user_id": "string",
                    "action": "string",
                    "resource": "string",
                    "details": "dict (optional)"
                },
                "outputs": {"log_id": "string"},
                "example": "log_activity('user123', 'report_generated', 'monthly_sales', details)"
            },
            
            # FILE & DOCUMENT FUNCTIONS (31-40)
            "create_pdf_report": {
                "description": "Generate PDF report from data and template",
                "inputs": {
                    "template": "string (template name)",
                    "data": "dict",
                    "output_path": "string",
                    "include_charts": "boolean (default: true)"
                },
                "outputs": {"file_path": "string"},
                "example": "create_pdf_report('monthly_template', report_data, '/reports/march.pdf')"
            },
            
            "export_to_excel": {
                "description": "Export data to Excel spreadsheet",
                "inputs": {
                    "data": "list or dict",
                    "sheet_name": "string",
                    "file_path": "string",
                    "include_formatting": "boolean (default: true)"
                },
                "outputs": {"file_path": "string"},
                "example": "export_to_excel(invoice_data, 'March Invoices', '/exports/invoices.xlsx')"
            },
            
            "create_csv_export": {
                "description": "Export data to CSV file",
                "inputs": {
                    "data": "list of dicts",
                    "file_path": "string",
                    "columns": "list of column names (optional)"
                },
                "outputs": {"file_path": "string"},
                "example": "create_csv_export(customer_data, '/exports/customers.csv')"
            },
            
            "generate_chart": {
                "description": "Create chart/graph from data",
                "inputs": {
                    "data": "dict or list",
                    "chart_type": "string (bar/line/pie/scatter)",
                    "title": "string",
                    "output_path": "string",
                    "dimensions": "dict (width/height, optional)"
                },
                "outputs": {"chart_path": "string"},
                "example": "generate_chart(monthly_sales, 'line', 'Sales Trend', '/charts/trend.png')"
            },
            
            "backup_data": {
                "description": "Create backup of specified data",
                "inputs": {
                    "data_source": "string",
                    "backup_location": "string",
                    "compression": "boolean (default: true)",
                    "encryption": "boolean (default: false)"
                },
                "outputs": {"backup_path": "string"},
                "example": "backup_data('customer_db', '/backups/', compression=true)"
            },
            
            "compress_files": {
                "description": "Compress files into archive",
                "inputs": {
                    "file_paths": "list of strings",
                    "output_path": "string",
                    "format": "string (zip/tar.gz, default: zip)"
                },
                "outputs": {"archive_path": "string"},
                "example": "compress_files(['/reports/march.pdf'], '/archives/reports.zip')"
            },
            
            "upload_to_cloud": {
                "description": "Upload files to cloud storage",
                "inputs": {
                    "file_path": "string",
                    "cloud_provider": "string (aws/gcp/azure)",
                    "bucket_name": "string",
                    "access_level": "string (private/public)"
                },
                "outputs": {"cloud_url": "string"},
                "example": "upload_to_cloud('/reports/march.pdf', 'aws', 'company-reports')"
            },
            
            "create_document": {
                "description": "Create formatted document from template and data",
                "inputs": {
                    "template_type": "string (word/google_docs)",
                    "template_id": "string",
                    "data": "dict",
                    "output_name": "string"
                },
                "outputs": {"document_id": "string"},
                "example": "create_document('word', 'contract_template', client_data, 'New Contract')"
            },
            
            "scan_document": {
                "description": "OCR scan document and extract text",
                "inputs": {
                    "file_path": "string",
                    "language": "string (default: en)",
                    "output_format": "string (text/json)"
                },
                "outputs": {"extracted_text": "string or dict"},
                "example": "scan_document('/uploads/invoice.pdf', 'en', 'json')"
            },
            
            "merge_documents": {
                "description": "Merge multiple documents into single file",
                "inputs": {
                    "file_paths": "list of strings",
                    "output_path": "string",
                    "document_type": "string (pdf/word)"
                },
                "outputs": {"merged_file_path": "string"},
                "example": "merge_documents(['/docs/part1.pdf', '/docs/part2.pdf'], '/final.pdf', 'pdf')"
            },
            
            # SYSTEM & AUTOMATION FUNCTIONS (41-50)
            "execute_sql_query": {
                "description": "Execute SQL query on database",
                "inputs": {
                    "query": "string",
                    "database": "string",
                    "parameters": "dict (optional)"
                },
                "outputs": {"results": "list of records"},
                "example": "execute_sql_query('SELECT * FROM orders WHERE date > ?', 'main_db', {'date': '2024-03-01'})"
            },
            
            "call_api": {
                "description": "Make HTTP API call to external service",
                "inputs": {
                    "url": "string",
                    "method": "string (GET/POST/PUT/DELETE)",
                    "headers": "dict (optional)",
                    "data": "dict (optional)",
                    "timeout": "integer (seconds, default: 30)"
                },
                "outputs": {"response": "API response object"},
                "example": "call_api('https://api.service.com/data', 'GET', headers={'Auth': 'token'})"
            },
            
            "schedule_task": {
                "description": "Schedule automated task for future execution",
                "inputs": {
                    "task_name": "string",
                    "function_call": "string",
                    "parameters": "dict",
                    "schedule": "string (cron format or datetime)",
                    "recurring": "boolean (default: false)"
                },
                "outputs": {"task_id": "string"},
                "example": "schedule_task('monthly_report', 'generate_report', params, '0 0 1 * *', true)"
            },
            
            "validate_data": {
                "description": "Validate data against specified rules or schema",
                "inputs": {
                    "data": "any",
                    "validation_rules": "dict",
                    "strict_mode": "boolean (default: false)"
                },
                "outputs": {"validation_result": "validation object"},
                "example": "validate_data(customer_data, email_validation_rules)"
            },
            
            "transform_data": {
                "description": "Transform data format or structure",
                "inputs": {
                    "data": "any",
                    "transformation_type": "string (json_to_csv/xml_to_json/etc)",
                    "mapping_rules": "dict (optional)"
                },
                "outputs": {"transformed_data": "transformed data object"},
                "example": "transform_data(xml_data, 'xml_to_json')"
            },
            
            "cache_data": {
                "description": "Store data in cache for faster retrieval",
                "inputs": {
                    "key": "string",
                    "data": "any",
                    "expiry_time": "integer (seconds, optional)",
                    "cache_type": "string (memory/redis, default: memory)"
                },
                "outputs": {"cache_status": "success/failure"},
                "example": "cache_data('monthly_report_march', report_data, 3600)"
            },
            
            "clear_cache": {
                "description": "Clear cached data by key or clear all cache",
                "inputs": {
                    "key": "string (optional, if not provided clears all)",
                    "cache_type": "string (memory/redis, default: memory)"
                },
                "outputs": {"clear_status": "success/failure"},
                "example": "clear_cache('monthly_report_march')"
            },
            
            "monitor_system": {
                "description": "Check system health and performance metrics",
                "inputs": {
                    "components": "list of system components",
                    "metrics": "list of metrics to check",
                    "alert_thresholds": "dict (optional)"
                },
                "outputs": {"system_status": "system health object"},
                "example": "monitor_system(['database', 'api'], ['cpu', 'memory', 'disk'])"
            },
            
            "encrypt_data": {
                "description": "Encrypt sensitive data using specified algorithm",
                "inputs": {
                    "data": "string or dict",
                    "encryption_type": "string (AES/RSA, default: AES)",
                    "key": "string (optional, generates if not provided)"
                },
                "outputs": {"encrypted_data": "encrypted data object"},
                "example": "encrypt_data(customer_ssn_data, 'AES')"
            },
            
            "audit_compliance": {
                "description": "Run compliance audit checks on data or processes",
                "inputs": {
                    "audit_type": "string (data_privacy/financial/security)",
                    "scope": "string (department/system/all)",
                    "standards": "list of compliance standards"
                },
                "outputs": {"audit_report": "compliance audit object"},
                "example": "audit_compliance('data_privacy', 'customer_data', ['GDPR', 'CCPA'])"
            }
        }
    
    def get_function_info(self, function_name: str) -> Dict:
        """Get detailed information about a specific function"""
        return self.functions.get(function_name, {})
    
    def search_functions(self, keyword: str) -> List[str]:
        """Search for functions containing a keyword in name or description"""
        matching_functions = []
        for func_name, func_info in self.functions.items():
            if (keyword.lower() in func_name.lower() or 
                keyword.lower() in func_info.get('description', '').lower()):
                matching_functions.append(func_name)
        return matching_functions
    
    def get_functions_by_category(self, category: str) -> List[str]:
        """Get functions by category (data_retrieval, calculation, communication, etc.)"""
        category_ranges = {
            'data_retrieval': list(self.functions.keys())[0:10],
            'calculation': list(self.functions.keys())[10:20], 
            'communication': list(self.functions.keys())[20:30],
            'file_document': list(self.functions.keys())[30:40],
            'system_automation': list(self.functions.keys())[40:50]
        }
        return category_ranges.get(category, [])
    
    def list_all_functions(self) -> List[str]:
        """Get list of all available function names"""
        return list(self.functions.keys())

# Example usage and testing
if __name__ == "__main__":
    # Initialize the function library
    lib = FunctionLibrary()
    
    # Example: Search for email-related functions
    email_functions = lib.search_functions("email")
    print("Email-related functions:", email_functions)
    
    # Example: Get info about a specific function
    invoice_func_info = lib.get_function_info("get_invoices")
    print("\nget_invoices function info:")
    print(json.dumps(invoice_func_info, indent=2))
    
    # Example: Get all calculation functions
    calc_functions = lib.get_functions_by_category("calculation")
    print("\nCalculation functions:", calc_functions)
    
    # List first 10 functions as sample
    print("\nFirst 10 functions in library:")
    for i, func_name in enumerate(lib.list_all_functions()[:10]):
        print(f"{i+1}. {func_name}: {lib.functions[func_name]['description']}")