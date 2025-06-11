# Plugin Internal Structure

## Complete Plugin Example: AWS Operations Plugin

Let's look at a full AWS operations plugin to understand the internal structure:

```
aws_operations_plugin/
├── __init__.py
├── plugin.py              # Main plugin class
├── categories.py          # Intent categories
├── handlers/              # Handler implementations
│   ├── __init__.py
│   ├── billing_handler.py
│   ├── outage_handler.py
│   └── security_handler.py
├── prompts/              # System prompts
│   ├── __init__.py
│   ├── billing_prompts.py
│   ├── outage_prompts.py
│   └── security_prompts.py
├── tools/               # Tool integrations
│   ├── __init__.py
│   ├── aws_cost_explorer.py
│   ├── aws_health_checker.py
│   └── aws_security_analyzer.py
├── config/              # Configuration schemas
│   ├── __init__.py
│   └── aws_config.py
└── tests/               # Plugin tests
    ├── __init__.py
    ├── test_handlers.py
    └── test_integration.py
```

## 1. Main Plugin Class (`plugin.py`)

```python
# aws_operations_plugin/plugin.py
from typing import List, Dict, Any
from agentflow import AgentPlugin, IntentCategory, BaseHandler
from .categories import get_aws_categories
from .handlers import AWSBillingHandler, AWSOutageHandler, AWSSecurityHandler
from .prompts import AWS_PROMPTS
from .tools import AWSCostExplorer, AWSHealthChecker, AWSSecurityAnalyzer
from .config import AWSPluginConfig

class AWSOperationsPlugin:
    """AWS Operations Plugin for AgentFlow"""
    
    def __init__(self, aws_access_key: str = None, aws_secret_key: str = None, 
                 aws_region: str = "us-east-1", config: Dict[str, Any] = None):
        """
        Initialize AWS plugin with credentials and configuration
        
        Args:
            aws_access_key: AWS access key (or use IAM role)
            aws_secret_key: AWS secret key (or use IAM role)
            aws_region: Default AWS region
            config: Plugin-specific configuration
        """
        self.config = AWSPluginConfig(**(config or {}))
        
        # Initialize AWS tools with credentials
        self.cost_explorer = AWSCostExplorer(
            access_key=aws_access_key,
            secret_key=aws_secret_key,
            region=aws_region
        )
        
        self.health_checker = AWSHealthChecker(
            access_key=aws_access_key,
            secret_key=aws_secret_key,
            region=aws_region
        )
        
        self.security_analyzer = AWSSecurityAnalyzer(
            access_key=aws_access_key,
            secret_key=aws_secret_key,
            region=aws_region
        )
        
        # Initialize handlers with tools
        self.billing_handler = AWSBillingHandler(
            cost_explorer=self.cost_explorer,
            config=self.config.billing_config
        )
        
        self.outage_handler = AWSOutageHandler(
            health_checker=self.health_checker,
            config=self.config.outage_config
        )
        
        self.security_handler = AWSSecurityHandler(
            security_analyzer=self.security_analyzer,
            config=self.config.security_config
        )
    
    def get_intent_categories(self) -> List[IntentCategory]:
        """Return AWS-specific intent categories"""
        return get_aws_categories()
    
    def get_handlers(self) -> List[BaseHandler]:
        """Return AWS-specific handlers"""
        return [
            self.billing_handler,
            self.outage_handler, 
            self.security_handler
        ]
    
    def get_prompts(self) -> Dict[str, str]:
        """Return AWS-specific system prompts"""
        return AWS_PROMPTS
    
    def get_tools_config(self) -> Dict[str, dict]:
        """Return AWS tool configurations for FastMCP"""
        return {
            'aws_cost_analysis': {
                'type': 'fastmcp_tool',
                'config': {
                    'tool_name': 'aws_cost_explorer',
                    'timeout': 30,
                    'retry_count': 3
                }
            },
            'aws_service_health': {
                'type': 'fastmcp_tool',
                'config': {
                    'tool_name': 'aws_health_dashboard',
                    'timeout': 15,
                    'retry_count': 2
                }
            },
            'aws_security_scan': {
                'type': 'fastmcp_tool',
                'config': {
                    'tool_name': 'aws_security_analyzer',
                    'timeout': 45,
                    'retry_count': 1
                }
            }
        }
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return {
            'name': 'aws_operations',
            'version': '1.2.0',
            'description': 'AWS operations and management plugin',
            'author': 'AgentFlow Community',
            'requires': ['boto3>=1.26.0', 'agentflow>=1.0.0'],
            'categories': ['cloud', 'infrastructure', 'billing'],
            'permissions': ['aws:read', 'aws:analyze']
        }
    
    async def on_plugin_load(self):
        """Called when plugin is loaded"""
        # Validate AWS credentials
        try:
            await self.health_checker.verify_credentials()
            print(f"AWS Plugin loaded successfully for region {self.health_checker.region}")
        except Exception as e:
            print(f"Warning: AWS credentials validation failed: {e}")
    
    async def on_plugin_unload(self):
        """Called when plugin is unloaded"""
        # Cleanup resources
        await self.cost_explorer.close()
        await self.health_checker.close()
        await self.security_analyzer.close()
        print("AWS Plugin unloaded successfully")

# Plugin factory function (required)
def create_plugin(**kwargs) -> AWSOperationsPlugin:
    """Factory function to create plugin instance"""
    return AWSOperationsPlugin(**kwargs)
```

## 2. Intent Categories (`categories.py`)

```python
# aws_operations_plugin/categories.py
from agentflow import IntentCategory
from typing import List

def get_aws_categories() -> List[IntentCategory]:
    """Define AWS-specific intent categories with examples and patterns"""
    
    return [
        IntentCategory(
            name="AWS_BILLING",
            description="AWS cost, billing, and budget-related questions",
            examples=[
                "Why is my AWS bill so high this month?",
                "Show me cost breakdown by service",
                "What's driving the EC2 costs?",
                "Set up a budget alert",
                "Compare costs with last month"
            ],
            confidence_threshold=0.7,
            keywords=[
                "bill", "billing", "cost", "expensive", "budget", "spend", 
                "charge", "fee", "price", "money", "dollar", "usage"
            ],
            patterns=[
                r"\$\d+",  # Dollar amounts
                r"bill.*high",
                r"cost.*breakdown",
                r"budget.*alert"
            ]
        ),
        
        IntentCategory(
            name="AWS_OUTAGE",
            description="AWS service outages, performance issues, and health checks",
            examples=[
                "S3 is responding slowly",
                "Lambda functions are timing out", 
                "RDS connection issues",
                "EC2 instances are unhealthy",
                "API Gateway returning 500 errors"
            ],
            confidence_threshold=0.8,  # High confidence for outages
            keywords=[
                "down", "slow", "timeout", "error", "failed", "outage",
                "unavailable", "503", "500", "connection", "latency"
            ],
            patterns=[
                r"(s3|lambda|rds|ec2|api gateway).*down",
                r"timeout",
                r"5\d\d.*error",
                r"connection.*failed"
            ]
        ),
        
        IntentCategory(
            name="AWS_SECURITY",
            description="AWS security, IAM, and compliance questions",
            examples=[
                "Review my IAM policies",
                "Check for security vulnerabilities",
                "Who has admin access?",
                "Audit recent API calls",
                "Security group configuration"
            ],
            confidence_threshold=0.6,
            keywords=[
                "security", "iam", "policy", "permission", "access", "audit",
                "vulnerability", "compliance", "role", "group", "admin"
            ],
            patterns=[
                r"iam.*policy",
                r"security.*group",
                r"admin.*access",
                r"audit.*log"
            ]
        ),
        
        IntentCategory(
            name="AWS_RESOURCE_MANAGEMENT",
            description="EC2, EBS, VPC, and other resource management",
            examples=[
                "List all running EC2 instances",
                "Create a new VPC",
                "Resize EBS volume",
                "Check unused resources",
                "Optimize resource allocation"
            ],
            confidence_threshold=0.6,
            keywords=[
                "ec2", "ebs", "vpc", "instance", "volume", "subnet",
                "create", "delete", "resize", "optimize", "list"
            ],
            patterns=[
                r"(create|delete|list).*instance",
                r"resize.*volume",
                r"vpc.*configuration"
            ]
        )
    ]
```

## 3. Handler Implementation (`handlers/billing_handler.py`)

```python
# aws_operations_plugin/handlers/billing_handler.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from agentflow import BaseHandler
from ..tools.aws_cost_explorer import AWSCostExplorer
from ..config.aws_config import BillingConfig

class AWSBillingHandler(BaseHandler):
    """Handler for AWS billing and cost-related requests"""
    
    def __init__(self, cost_explorer: AWSCostExplorer, config: BillingConfig):
        self.cost_explorer = cost_explorer
        self.config = config
        self.priority = 8  # High priority for billing issues
    
    async def can_handle(self, intent: dict, context: dict) -> bool:
        """Check if this handler can process the intent"""
        return intent.get('type') == 'AWS_BILLING'
    
    def get_priority(self) -> int:
        """Return handler priority"""
        return self.priority
    
    async def handle(
        self, 
        message: str, 
        intent: dict,
        context: dict, 
        session: 'Session',
        thread: Optional['ConversationThread']
    ) -> dict:
        """Process AWS billing request"""
        
        start_time = datetime.now()
        
        try:
            # Determine what type of billing analysis to perform
            analysis_type = self._determine_analysis_type(message, intent)
            
            # Gather billing data in parallel
            tasks = []
            
            if analysis_type in ['overview', 'breakdown']:
                tasks.append(self._get_current_month_costs())
                tasks.append(self._get_previous_month_costs())
                tasks.append(self._get_cost_by_service())
            
            if analysis_type in ['trends', 'forecast']:
                tasks.append(self._get_cost_trends())
                tasks.append(self._get_forecast())
            
            if analysis_type in ['anomalies', 'alerts']:
                tasks.append(self._detect_cost_anomalies())
                tasks.append(self._check_budget_alerts())
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and generate response
            response_data = self._process_billing_results(results, analysis_type)
            
            # Generate human-readable response
            response_text = await self._generate_billing_response(
                response_data, message, context
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'content': response_text,
                'intent_type': 'AWS_BILLING',
                'analysis_type': analysis_type,
                'data': response_data,
                'tools_used': ['aws_cost_explorer'],
                'processing_time': processing_time,
                'confidence': intent.get('confidence', 0.8),
                'suggested_actions': self._get_suggested_actions(response_data)
            }
            
        except Exception as e:
            return {
                'content': f"I encountered an error analyzing your AWS billing: {str(e)}",
                'error': str(e),
                'intent_type': 'AWS_BILLING',
                'fallback': True,
                'suggested_actions': ['Check AWS credentials', 'Verify permissions']
            }
    
    def _determine_analysis_type(self, message: str, intent: dict) -> str:
        """Determine what type of billing analysis to perform"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['breakdown', 'by service', 'detailed']):
            return 'breakdown'
        elif any(word in message_lower for word in ['trend', 'over time', 'monthly']):
            return 'trends'
        elif any(word in message_lower for word in ['forecast', 'predict', 'next month']):
            return 'forecast'
        elif any(word in message_lower for word in ['anomaly', 'spike', 'unusual']):
            return 'anomalies'
        elif any(word in message_lower for word in ['alert', 'budget', 'threshold']):
            return 'alerts'
        else:
            return 'overview'
    
    async def _get_current_month_costs(self) -> Dict[str, Any]:
        """Get current month's costs"""
        end_date = datetime.now()
        start_date = end_date.replace(day=1)
        
        return await self.cost_explorer.get_costs(
            start_date=start_date,
            end_date=end_date,
            granularity='MONTHLY'
        )
    
    async def _get_previous_month_costs(self) -> Dict[str, Any]:
        """Get previous month's costs for comparison"""
        current_month_start = datetime.now().replace(day=1)
        previous_month_end = current_month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        
        return await self.cost_explorer.get_costs(
            start_date=previous_month_start,
            end_date=previous_month_end,
            granularity='MONTHLY'
        )
    
    async def _get_cost_by_service(self) -> Dict[str, Any]:
        """Get cost breakdown by AWS service"""
        end_date = datetime.now()
        start_date = end_date.replace(day=1)
        
        return await self.cost_explorer.get_costs_by_dimension(
            start_date=start_date,
            end_date=end_date,
            dimension='SERVICE',
            granularity='MONTHLY'
        )
    
    async def _get_cost_trends(self) -> Dict[str, Any]:
        """Get cost trends over the last 6 months"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months
        
        return await self.cost_explorer.get_costs(
            start_date=start_date,
            end_date=end_date,
            granularity='MONTHLY'
        )
    
    async def _get_forecast(self) -> Dict[str, Any]:
        """Get cost forecast for next month"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)
        
        return await self.cost_explorer.get_cost_forecast(
            start_date=start_date,
            end_date=end_date
        )
    
    async def _detect_cost_anomalies(self) -> Dict[str, Any]:
        """Detect cost anomalies using AWS Cost Anomaly Detection"""
        return await self.cost_explorer.get_anomalies(
            date_interval={
                'StartDate': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'EndDate': datetime.now().strftime('%Y-%m-%d')
            }
        )
    
    async def _check_budget_alerts(self) -> Dict[str, Any]:
        """Check active budget alerts"""
        return await self.cost_explorer.get_budget_alerts()
    
    def _process_billing_results(self, results: list, analysis_type: str) -> Dict[str, Any]:
        """Process raw billing results into structured data"""
        processed_data = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle exceptions in results
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if analysis_type == 'overview':
            if len(valid_results) >= 3:
                processed_data.update({
                    'current_month': valid_results[0],
                    'previous_month': valid_results[1],
                    'service_breakdown': valid_results[2]
                })
        
        # Add more processing logic for other analysis types...
        
        return processed_data
    
    async def _generate_billing_response(
        self, 
        data: Dict[str, Any], 
        original_message: str, 
        context: dict
    ) -> str:
        """Generate human-readable response using LLM"""
        
        # Prepare context for LLM
        prompt_context = {
            'original_message': original_message,
            'billing_data': data,
            'user_context': context.get('user_context', {}),
            'analysis_type': data.get('analysis_type', 'overview')
        }
        
        # Use the billing prompt template
        from ..prompts.billing_prompts import BILLING_ANALYSIS_PROMPT
        
        prompt = BILLING_ANALYSIS_PROMPT.format(**prompt_context)
        
        # This would use the LLM client from the framework
        # For now, return a template response
        return self._generate_template_response(data)
    
    def _generate_template_response(self, data: Dict[str, Any]) -> str:
        """Generate template response (fallback)"""
        analysis_type = data.get('analysis_type', 'overview')
        
        if analysis_type == 'overview':
            return f"""
            ## AWS Billing Overview
            
            Based on your billing analysis:
            
            **Current Month**: ${data.get('current_month', {}).get('total', 'N/A')}
            **Previous Month**: ${data.get('previous_month', {}).get('total', 'N/A')}
            
            **Top Services by Cost**:
            {self._format_service_costs(data.get('service_breakdown', {}))}
            
            **Recommendations**:
            - Review EC2 instance utilization
            - Consider Reserved Instances for predictable workloads
            - Set up cost alerts for unusual spending
            """
        
        return "Billing analysis completed successfully."
    
    def _format_service_costs(self, service_data: dict) -> str:
        """Format service costs for display"""
        if not service_data or 'Groups' not in service_data:
            return "- No service breakdown available"
        
        formatted = []
        for group in service_data['Groups'][:5]:  # Top 5 services
            service = group['Keys'][0] if group['Keys'] else 'Unknown'
            amount = group['Metrics']['UnblendedCost']['Amount']
            formatted.append(f"- {service}: ${float(amount):.2f}")
        
        return '\n'.join(formatted)
    
    def _get_suggested_actions(self, data: Dict[str, Any]) -> list:
        """Generate suggested actions based on billing data"""
        actions = []
        
        # Add logic to suggest actions based on the analysis
        analysis_type = data.get('analysis_type')
        
        if analysis_type == 'overview':
            actions.extend([
                'Set up budget alerts',
                'Review top spending services',
                'Check for unused resources'
            ])
        elif analysis_type == 'anomalies':
            actions.extend([
                'Investigate cost spikes',
                'Review recent deployments',
                'Check for unauthorized usage'
            ])
        
        return actions
```

## 4. Tool Integration (`tools/aws_cost_explorer.py`)

```python
# aws_operations_plugin/tools/aws_cost_explorer.py
import boto3
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError

class AWSCostExplorer:
    """AWS Cost Explorer integration for billing analysis"""
    
    def __init__(self, access_key: str = None, secret_key: str = None, 
                 region: str = "us-east-1"):
        """Initialize AWS Cost Explorer client"""
        
        session_kwargs = {'region_name': region}
        if access_key and secret_key:
            session_kwargs.update({
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            })
        
        self.session = boto3.Session(**session_kwargs)
        self.client = self.session.client('ce')  # Cost Explorer
        self.budgets_client = self.session.client('budgets')
        self.region = region
    
    async def get_costs(
        self, 
        start_date: datetime, 
        end_date: datetime,
        granularity: str = 'MONTHLY',
        metrics: list = None
    ) -> Dict[str, Any]:
        """Get cost data for a date range"""
        
        if metrics is None:
            metrics = ['UnblendedCost']
        
        try:
            # Run in thread pool since boto3 is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_cost_and_usage(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Granularity=granularity,
                    Metrics=metrics
                )
            )
            
            return self._process_cost_response(response)
            
        except ClientError as e:
            raise Exception(f"AWS Cost Explorer error: {e}")
        except NoCredentialsError:
            raise Exception("AWS credentials not found or invalid")
    
    async def get_costs_by_dimension(
        self,
        start_date: datetime,
        end_date: datetime, 
        dimension: str,
        granularity: str = 'MONTHLY'
    ) -> Dict[str, Any]:
        """Get costs grouped by dimension (SERVICE, AZ, etc.)"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_dimension_values(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Dimension=dimension,
                    Context='COST_AND_USAGE'
                )
            )
            
            # Get costs for each dimension value
            dimension_costs = await loop.run_in_executor(
                None,
                lambda: self.client.get_cost_and_usage(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Granularity=granularity,
                    Metrics=['UnblendedCost'],
                    GroupBy=[{
                        'Type': 'DIMENSION',
                        'Key': dimension
                    }]
                )
            )
            
            return self._process_dimension_response(dimension_costs)
            
        except ClientError as e:
            raise Exception(f"AWS Cost Explorer dimension error: {e}")
    
    async def get_cost_forecast(
        self,
        start_date: datetime,
        end_date: datetime,
        metric: str = 'UNBLENDED_COST'
    ) -> Dict[str, Any]:
        """Get cost forecast"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_cost_forecast(
                    TimePeriod={
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': end_date.strftime('%Y-%m-%d')
                    },
                    Metric=metric,
                    Granularity='MONTHLY'
                )
            )
            
            return self._process_forecast_response(response)
            
        except ClientError as e:
            raise Exception(f"AWS Cost forecast error: {e}")
    
    async def get_anomalies(self, date_interval: Dict[str, str]) -> Dict[str, Any]:
        """Get cost anomalies"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_anomalies(
                    DateInterval=date_interval
                )
            )
            
            return self._process_anomalies_response(response)
            
        except ClientError as e:
            raise Exception(f"AWS Cost anomalies error: {e}")
    
    async def get_budget_alerts(self) -> Dict[str, Any]:
        """Get budget alerts"""
        
        try:
            loop = asyncio.get_event_loop()
            
            # First get list of budgets
            budgets_response = await loop.run_in_executor(
                None,
                lambda: self.budgets_client.describe_budgets(
                    AccountId=self.session.get_credentials().access_key[:12]  # Account ID approximation
                )
            )
            
            alerts = []
            for budget in budgets_response.get('Budgets', []):
                budget_name = budget['BudgetName']
                
                # Get notifications for each budget
                notifications_response = await loop.run_in_executor(
                    None,
                    lambda: self.budgets_client.describe_notifications_for_budget(
                        AccountId=self.session.get_credentials().access_key[:12],
                        BudgetName=budget_name
                    )
                )
                
                alerts.append({
                    'budget_name': budget_name,
                    'budget': budget,
                    'notifications': notifications_response.get('Notifications', [])
                })
            
            return {'budget_alerts': alerts}
            
        except ClientError as e:
            raise Exception(f"AWS Budget alerts error: {e}")
    
    def _process_cost_response(self, response: dict) -> Dict[str, Any]:
        """Process cost response into structured format"""
        results = response.get('ResultsByTime', [])
        
        processed = {
            'total': 0.0,
            'currency': 'USD',
            'time_periods': []
        }
        
        for result in results:
            period_data = {
                'start': result['TimePeriod']['Start'],
                'end': result['TimePeriod']['End'],
                'amount': float(result['Total']['UnblendedCost']['Amount']),
                'currency': result['Total']['UnblendedCost']['Unit']
            }
            processed['time_periods'].append(period_data)
            processed['total'] += period_data['amount']
        
        return processed
    
    def _process_dimension_response(self, response: dict) -> Dict[str, Any]:
        """Process dimension-grouped cost response"""
        results = response.get('ResultsByTime', [])
        
        processed = {
            'Groups': [],
            'total': 0.0
        }
        
        for result in results:
            for group in result.get('Groups', []):
                processed['Groups'].append({
                    'Keys': group['Keys'],
                    'Metrics': group['Metrics']
                })
                
                # Add to total
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                processed['total'] += amount
        
        return processed
    
    def _process_forecast_response(self, response: dict) -> Dict[str, Any]:
        """Process forecast response"""
        return {
            'forecast_results': response.get('ForecastResultsByTime', []),
            'total_forecast': response.get('Total', {})
        }
    
    def _process_anomalies_response(self, response: dict) -> Dict[str, Any]:
        """Process anomalies response"""
        return {
            'anomalies': response.get('Anomalies', []),
            'count': len(response.get('Anomalies', []))
        }
    
    async def verify_credentials(self) -> bool:
        """Verify AWS credentials work"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.get_usage_forecast(
                    TimePeriod={
                        'Start': datetime.now().strftime('%Y-%m-%d'),
                        'End': datetime.now().strftime('%Y-%m-%d')
                    },
                    Metric='USAGE_QUANTITY',
                    Granularity='DAILY'
                )
            )
            return True
        except Exception:
            return False
    
    async def close(self):
        """Cleanup resources"""
        # boto3 clients don't need explicit cleanup
        pass
```

## 5. Configuration Schema (`config/aws_config.py`)

```python
# aws_operations_plugin/config/aws_config.py
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional

class BillingConfig(BaseModel):
    """Configuration for billing operations"""
    cost_threshold_alert: float = 1000.0  # Alert if daily cost > $1000
    currency: str = "USD"
    default_metrics: List[str] = ["UnblendedCost"]
    forecast_days: int = 30
    anomaly_detection_enabled: bool = True
    
    @validator('cost_threshold_alert')
    def threshold_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Cost threshold must be positive')
        return v

class OutageConfig(BaseModel):
    """Configuration for outage detection"""
    health_check_regions: List[str] = ["us-east-1", "us-west-2"]
    alert_channels: List[str] = []
    auto_escalate_p0: bool = True
    response_time_threshold_seconds: int = 300
    
class SecurityConfig(BaseModel):
    """Configuration for security operations"""
    scan_intervals_hours: int = 24
    vulnerability_severity_threshold: str = "MEDIUM"
    compliance_frameworks: List[str] = ["SOC2", "ISO27001"]
    alert_on_admin_actions: bool = True
    
class AWSPluginConfig(BaseModel):
    """Main AWS plugin configuration"""
    regions: List[str] = ["us-east-1"]
    default_region: str = "us-east-1"
    
    billing_config: BillingConfig = BillingConfig()
    outage_config: OutageConfig = OutageConfig()
    security_config: SecurityConfig = SecurityConfig()
    
    # Plugin-specific settings
    enable_cost_optimization: bool = True
    enable_security_scanning: bool = True
    enable_health_monitoring: bool = True
    
    # Integration settings
    slack_webhook_url: Optional[str] = None
    email_notifications: List[str] = []
    
    class Config:
        extra = "allow"  # Allow additional custom fields
```

## 6. System Prompts (`prompts/billing_prompts.py`)

```python
# aws_operations_plugin/prompts/billing_prompts.py

BILLING_ANALYSIS_PROMPT = """
You are an AWS billing expert helping users understand their cloud costs.

User Question: {original_message}
Billing Data: {billing_data}
Analysis Type: {analysis_type}
User Context: {user_context}

Based on the billing data provided, give a clear, actionable response that:

1. **Directly answers the user's question**
2. **Highlights key cost insights** (increases, decreases, anomalies)
3. **Identifies top cost drivers** (services, regions, usage patterns)
4. **Provides specific recommendations** for cost optimization
5. **Suggests next steps** the user should take

Format your response with:
- **Summary**: 1-2 sentence overview
- **Key Findings**: 3-5 bullet points of important insights
- **Cost Breakdown**: Table or list of major cost components
- **Recommendations**: 3-5 actionable suggestions
- **Next Steps**: What the user should do next

Be specific with dollar amounts and percentages. If you notice concerning trends (>20% increase, unusual spikes), emphasize them.

Keep the tone professional but friendly. Avoid technical jargon unless necessary.
"""

COST_OPTIMIZATION_PROMPT = """
You are an AWS cost optimization specialist. Review the following cost data and provide optimization recommendations:

Cost Data: {cost_data}
Usage Patterns: {usage_patterns}
Current Architecture: {architecture_info}

Provide recommendations in these categories:
1. **Quick Wins** (immediate savings, low effort)
2. **Reserved Instances/Savings Plans** (commitment-based savings)
3. **Right-sizing** (instance optimization)
4. **Architecture Changes** (longer-term optimizations)

For each recommendation:
- Estimated monthly savings
- Implementation effort (Low/Medium/High)
- Risk level (Low/Medium/High)
- Steps to implement

Focus on the highest-impact, lowest-risk optimizations first.
"""

AWS_PROMPTS = {
    'billing_analysis': BILLING_ANALYSIS_PROMPT,
    'cost_optimization': COST_OPTIMIZATION_PROMPT,
    'outage_response': """
    You are an AWS incident response specialist. A user is reporting a service issue.
    
    Issue: {message}
    Service Health: {health_data}
    User Environment: {user_context}
    
    Provide immediate troubleshooting steps and escalation guidance.
    """,
    'security_analysis': """
    You are an AWS security specialist reviewing potential security issues.
    
    Security Concern: {message}
    Security Scan Results: {security_data}
    User Permissions: {user_context}
    
    Assess the security situation and provide remediation steps.
    """
}
```

## 7. Plugin Entry Point (`__init__.py`)

```python
# aws_operations_plugin/__init__.py
"""
AWS Operations Plugin for AgentFlow

This plugin provides AWS billing, outage, and security management capabilities
for the AgentFlow framework.
"""

from .plugin import AWSOperationsPlugin, create_plugin
from .categories import get_aws_categories
from .handlers import AWSBillingHandler, AWSOutageHandler, AWSSecurityHandler
from .config import AWSPluginConfig

__version__ = "1.2.0"
__author__ = "AgentFlow Community"
__description__ = "AWS operations and management plugin"

# Plugin metadata for discovery
PLUGIN_INFO = {
    'name': 'aws_operations',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'requires': ['boto3>=1.26.0', 'agentflow>=1.0.0'],
    'categories': ['cloud', 'infrastructure', 'billing', 'security'],
    'entry_point': create_plugin
}

# Export main components
__all__ = [
    'AWSOperationsPlugin',
    'create_plugin', 
    'get_aws_categories',
    'AWSBillingHandler',
    'AWSOutageHandler', 
    'AWSSecurityHandler',
    'AWSPluginConfig',
    'PLUGIN_INFO'
]
```

## Plugin Loading Process

When the framework loads this plugin:

1. **Discovery**: Plugin manager finds the plugin via entry points or file system
2. **Import**: Imports the plugin module and calls `create_plugin()`
3. **Registration**: Registers intent categories, handlers, prompts, and tools
4. **Initialization**: Calls `on_plugin_load()` for any setup
5. **Integration**: Plugin components become part of the agent workflow

The plugin structure provides:
- **Modularity**: Each component has a specific responsibility
- **Testability**: Each component can be tested independently
- **Configuration**: Type-safe configuration with validation
- **Extensibility**: Easy to add new handlers or categories
- **Production Ready**: Proper error handling and async support

This shows how a real plugin would be structured internally with all the necessary components to handle AWS operations within the AgentFlow framework.