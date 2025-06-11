# AgentFlow Plugin Usage Examples

## Plugin Architecture Overview

Plugins are self-contained modules that extend the framework with domain-specific functionality. Each plugin can contribute:
- Intent categories
- Message handlers  
- System prompts
- Tool configurations
- Domain-specific logic

## Real-World Plugin Examples

### 1. Company-Specific Plugins

```python
# company_plugins/support_plugin.py
from agentflow import AgentPlugin, IntentCategory, BaseHandler

class CompanySupportPlugin:
    """Plugin for Acme Corp support operations"""
    
    def __init__(self, jira_client, slack_client, knowledge_base):
        self.jira = jira_client
        self.slack = slack_client  
        self.kb = knowledge_base
    
    def get_intent_categories(self) -> List[IntentCategory]:
        return [
            IntentCategory(
                "ACME_OUTAGE", 
                "Production outages specific to Acme infrastructure",
                ["Redis is down", "Database connection timeout", "API rate limits"]
            ),
            IntentCategory(
                "ACME_DEPLOYMENT",
                "Deployment and release questions", 
                ["When is next release?", "Can I deploy to staging?", "Rollback procedure"]
            ),
            IntentCategory(
                "ACME_ONCALL",
                "On-call escalation and incident management",
                ["Page on-call engineer", "Escalate to SRE", "Critical incident"]
            )
        ]
    
    def get_handlers(self) -> List[BaseHandler]:
        return [
            AcmeOutageHandler(self.jira, self.slack),
            AcmeDeploymentHandler(self.jira, self.kb),
            AcmeOncallHandler(self.slack, self.jira)
        ]
    
    def get_prompts(self) -> Dict[str, str]:
        return {
            'acme_outage_response': """
                You are Acme Corp's incident response assistant. When handling outages:
                
                1. Immediately assess severity (P0/P1/P2/P3)
                2. Check our monitoring dashboard: https://acme.grafana.com
                3. Look for recent deployments in Jira
                4. If P0/P1, auto-page on-call via Slack
                
                Current incident: {message}
                System status: {diagnostics}
                Recent deployments: {recent_changes}
                
                Provide immediate troubleshooting steps and escalation if needed.
            """,
            'acme_deployment_guidance': """
                You are Acme's deployment assistant. Our deployment process:
                
                - Staging deploys: Auto-approved for engineers
                - Production deploys: Require approval from Tech Lead
                - Hotfixes: Can bypass normal approval with SRE sign-off
                - Release schedule: Tuesdays/Thursdays 2PM EST
                
                Request: {message}
                User role: {user_context}
                Current release status: {release_info}
                
                Guide the user through appropriate deployment procedures.
            """
        }
    
    def get_tools_config(self) -> Dict[str, dict]:
        return {
            'acme_jira_integration': {
                'type': 'fastmcp_tool',
                'config': {
                    'tool_name': 'jira_ticket_manager',
                    'base_url': 'https://acme.atlassian.net',
                    'project_key': 'INFRA'
                }
            },
            'acme_grafana_check': {
                'type': 'fastmcp_tool', 
                'config': {
                    'tool_name': 'grafana_dashboard',
                    'dashboard_url': 'https://acme.grafana.com/d/infra-overview'
                }
            },
            'acme_slack_escalation': {
                'type': 'fastmcp_tool',
                'config': {
                    'tool_name': 'slack_pager',
                    'oncall_channel': '#sre-oncall',
                    'incident_channel': '#incidents'
                }
            }
        }

def create_plugin(jira_client, slack_client, knowledge_base):
    return CompanySupportPlugin(jira_client, slack_client, knowledge_base)
```

### 2. Industry-Specific Plugin

```python
# industry_plugins/healthcare_plugin.py
class HealthcareCompliancePlugin:
    """Plugin for HIPAA-compliant healthcare support"""
    
    def get_intent_categories(self) -> List[IntentCategory]:
        return [
            IntentCategory(
                "HIPAA_VIOLATION",
                "Potential HIPAA compliance issues",
                ["Patient data exposed", "Unauthorized access", "PHI breach"]
            ),
            IntentCategory(
                "CLINICAL_SYSTEM",
                "Clinical system support",
                ["EMR is down", "Lab results not syncing", "Medication alerts"]
            ),
            IntentCategory(
                "AUDIT_REQUEST", 
                "Compliance audit and reporting",
                ["Generate audit report", "Access logs needed", "Compliance review"]
            )
        ]
    
    def get_handlers(self) -> List[BaseHandler]:
        return [
            HIPAAViolationHandler(),  # Auto-escalates, logs to compliance
            ClinicalSystemHandler(),  # Integrates with Epic/Cerner 
            AuditRequestHandler()     # Generates compliance reports
        ]
    
    def get_prompts(self) -> Dict[str, str]:
        return {
            'hipaa_incident_response': """
                CRITICAL: Potential HIPAA violation detected.
                
                Immediate actions:
                1. Document incident with timestamp
                2. Isolate affected systems
                3. Notify Privacy Officer within 1 hour
                4. Begin breach assessment
                
                Incident: {message}
                Systems involved: {affected_systems}
                
                DO NOT discuss specific patient information in this channel.
                Escalating to Privacy Officer immediately.
            """
        }

def create_plugin():
    return HealthcareCompliancePlugin()
```

### 3. Multi-Plugin Usage

```python
# main.py - Building agent with multiple plugins
async def create_company_agent():
    """Create agent with multiple plugins"""
    
    # Initialize external services
    jira_client = JiraClient(url="https://acme.atlassian.net", token=JIRA_TOKEN)
    slack_client = SlackClient(token=SLACK_TOKEN)
    knowledge_base = VectorKnowledgeBase(index="acme-support-kb")
    
    agent = await (AgentFrameworkBuilder()
        # Company-specific plugin
        .with_plugins([
            'company_plugins.support_plugin',
            'company_plugins.security_plugin', 
            'company_plugins.hr_plugin'
        ])
        # Industry plugins
        .with_plugins([
            'industry_plugins.healthcare_plugin',
            'industry_plugins.fintech_plugin'
        ])
        # Open source community plugins
        .with_plugins([
            'agentflow_community.aws_plugin',
            'agentflow_community.kubernetes_plugin',
            'agentflow_community.datadog_plugin'
        ])
        .with_config('production_config.yaml')
        .with_rate_limiting(200)
        .with_logging()
        .build(llm_client, fastmcp_client))
    
    return agent
```

## Plugin Distribution & Installation

### 1. Internal Company Plugin Package

```python
# setup.py for company plugins
from setuptools import setup, find_packages

setup(
    name="acme-agentflow-plugins",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "agentflow>=1.0.0",
        "jira-python>=2.0.0",
        "slack-sdk>=3.0.0"
    ],
    entry_points={
        'agentflow.plugins': [
            'support = acme_plugins.support_plugin:create_plugin',
            'security = acme_plugins.security_plugin:create_plugin',
            'hr = acme_plugins.hr_plugin:create_plugin',
        ]
    }
)

# Installation
# pip install acme-agentflow-plugins
```

### 2. Community Plugin Ecosystem

```python
# Third-party AWS plugin
# pip install agentflow-aws-plugin

from agentflow_aws import AWSPlugin

class AWSPlugin:
    """Community plugin for AWS operations"""
    
    def get_intent_categories(self) -> List[IntentCategory]:
        return [
            IntentCategory("AWS_BILLING", "AWS cost and billing questions", 
                         ["Why is my bill high?", "Cost breakdown", "Budget alerts"]),
            IntentCategory("AWS_OUTAGE", "AWS service issues",
                         ["S3 is slow", "Lambda timeout", "RDS connection"]),
            IntentCategory("AWS_SECURITY", "AWS security concerns",
                         ["Security group", "IAM policy", "VPC configuration"])
        ]
    
    def get_handlers(self) -> List[BaseHandler]:
        return [
            AWSBillingHandler(),    # Integrates with Cost Explorer API
            AWSOutageHandler(),     # Checks AWS Status page
            AWSSecurityHandler()    # Reviews IAM/Security groups
        ]
    
    def get_tools_config(self) -> Dict[str, dict]:
        return {
            'aws_cost_explorer': {
                'type': 'fastmcp_tool',
                'config': {'tool_name': 'aws_cost_analysis'}
            },
            'aws_health_check': {
                'type': 'fastmcp_tool', 
                'config': {'tool_name': 'aws_service_health'}
            }
        }

# Auto-discovery usage
agent = await (AgentFrameworkBuilder()
    .auto_discover_plugins('agentflow_aws')  # Finds and loads AWS plugin
    .auto_discover_plugins('agentflow_k8s')  # Finds and loads K8s plugin
    .build(llm_client, fastmcp_client))
```

## Plugin Configuration & Customization

### 1. Plugin-Specific Configuration

```yaml
# config.yaml
name: "acme_support_agent"

# Core framework config
session_config:
  ttl_seconds: 3600

# Plugin-specific configuration
plugin_configs:
  acme_support:
    jira_project: "INFRA"
    slack_channels:
      oncall: "#sre-oncall"
      incidents: "#incidents"
    escalation_levels:
      - level: 1
        response_time: 300  # 5 minutes
        contacts: ["oncall@acme.com"]
      - level: 2 
        response_time: 900  # 15 minutes
        contacts: ["engineering-leads@acme.com"]
  
  aws_operations:
    regions: ["us-east-1", "us-west-2"]
    cost_threshold: 1000  # Alert if daily cost > $1000
    auto_scaling_enabled: true
  
  security_compliance:
    audit_retention_days: 2555  # 7 years
    alert_on_admin_actions: true
    require_mfa_for_escalation: true
```

### 2. Runtime Plugin Management

```python
class PluginManager:
    """Runtime plugin management"""
    
    async def hot_reload_plugin(self, plugin_name: str):
        """Reload plugin without restarting agent"""
        # Unload current plugin
        if plugin_name in self.loaded_plugins:
            await self.unload_plugin(plugin_name)
        
        # Reload from source
        new_plugin = await self.load_plugin(plugin_name)
        
        # Re-register handlers and categories
        self.agent.handler_registry.refresh_handlers(new_plugin.get_handlers())
        self.agent.intent_classifier.refresh_categories(new_plugin.get_intent_categories())
    
    async def enable_plugin(self, plugin_name: str, user_id: str = None):
        """Enable plugin for specific user or globally"""
        if user_id:
            # Enable for specific user session
            session = await self.agent.session_manager.get_session(user_id)
            session.enabled_plugins.add(plugin_name)
        else:
            # Enable globally
            self.globally_enabled_plugins.add(plugin_name)
    
    async def disable_plugin(self, plugin_name: str, user_id: str = None):
        """Disable plugin"""
        # Similar logic for disabling
        pass

# Usage
await plugin_manager.enable_plugin('security_plugin', user_id='admin_user')
await plugin_manager.disable_plugin('hr_plugin')  # Disable globally
```

## Plugin Use Cases

### 1. Team-Specific Customization

```python
# Different teams get different plugin combinations

# SRE Team Agent
sre_agent = await (AgentFrameworkBuilder()
    .with_plugins([
        'company.infrastructure_plugin',   # Internal infra tools
        'community.aws_plugin',           # AWS operations
        'community.kubernetes_plugin',    # K8s management
        'community.datadog_plugin'        # Monitoring
    ])
    .build(llm_client, fastmcp_client))

# Product Team Agent  
product_agent = await (AgentFrameworkBuilder()
    .with_plugins([
        'company.product_plugin',         # Internal product tools
        'community.analytics_plugin',    # Data analysis
        'community.ab_testing_plugin',   # Experiment management
        'community.user_feedback_plugin' # Customer insights
    ])
    .build(llm_client, fastmcp_client))

# Security Team Agent
security_agent = await (AgentFrameworkBuilder()
    .with_plugins([
        'company.security_plugin',        # Internal security tools
        'company.compliance_plugin',      # Compliance management
        'community.vulnerability_plugin', # Vuln scanning
        'community.threat_intel_plugin'   # Threat intelligence
    ])
    .build(llm_client, fastmcp_client))
```

### 2. Progressive Plugin Enablement

```python
# Start with basic functionality
basic_agent = await (AgentFrameworkBuilder()
    .with_plugins(['core.basic_support'])
    .build(llm_client))

# Add company-specific features
await plugin_manager.enable_plugin('company.jira_integration')
await plugin_manager.enable_plugin('company.slack_integration')

# Add advanced features for power users
if user.is_admin:
    await plugin_manager.enable_plugin('company.admin_tools', user.id)
    await plugin_manager.enable_plugin('company.system_management', user.id)

# Temporary plugin for incident response
if incident_active:
    await plugin_manager.enable_plugin('emergency.incident_response')
```

### 3. A/B Testing with Plugins

```python
# Test new plugin with subset of users
if user.id in experimental_group:
    await plugin_manager.enable_plugin('experimental.new_feature', user.id)
else:
    await plugin_manager.enable_plugin('stable.current_feature', user.id)

# Gradual rollout
rollout_percentage = get_rollout_percentage('new_ai_features')
if hash(user.id) % 100 < rollout_percentage:
    await plugin_manager.enable_plugin('experimental.gpt4_responses', user.id)
```

## Benefits of Plugin Architecture

1. **Modularity**: Each domain/team can develop independently
2. **Reusability**: Share plugins across teams/companies
3. **Testing**: Test new features with plugins before core integration
4. **Customization**: Different user groups get different capabilities
5. **Marketplace**: Community can build and share plugins
6. **Maintenance**: Update domain logic without touching core framework
7. **Compliance**: Industry-specific plugins handle regulatory requirements

The plugin system turns the framework into a **platform** where the community can build and share domain-specific extensions, similar to how VS Code extensions work.