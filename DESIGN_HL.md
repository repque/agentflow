# AgentFlow Framework - High-Level Design

## Executive Summary

AgentFlow is a production-ready framework for building conversational AI agents that separates orchestration infrastructure (80% reusable) from domain logic (20% customizable). It enables teams to build sophisticated AI agents with consistent patterns, built-in best practices, and a thriving plugin ecosystem.

**Key Innovation**: Transform agentic app development from "build everything from scratch" to "configure domain behavior on robust infrastructure" - similar to how Express.js revolutionized web development.

## System Architecture Overview

```mermaid
graph TB
    subgraph "Application Examples"
        SB[Support Bot<br/>• QUERY<br/>• OUTAGE<br/>• DATA_ISSUE<br/>• ESCALATION]
        EA[E-commerce Assistant<br/>• PRODUCT_INQ<br/>• ORDER_STATUS<br/>• RETURNS<br/>• RECOMMEND]
        PA[Personal Assistant<br/>• SCHEDULE<br/>• REMINDER<br/>• EMAIL<br/>• TASK]
    end
    
    subgraph "AgentFlow Framework Core"
        IC[Intent Classification]
        AO[Agent Orchestrator]
        HR[Handler Registry]
        SM[Session Manager]
        TM[Thread Manager]
        MS[Memory Store]
    end
    
    subgraph "Infrastructure Layer"
        FMI[FastMCP Integration]
        LLC[LLM Clients]
        VDB[Vector Database]
    end
    
    SB --> AO
    EA --> AO
    PA --> AO
    
    IC --> AO
    AO --> HR
    SM --> TM
    TM --> MS
    
    AO --> FMI
    AO --> LLC
    MS --> VDB
    
    style SB fill:#e3f2fd
    style EA fill:#e8f5e8
    style PA fill:#fff3e0
    style AO fill:#f3e5f5
    style IC fill:#f3e5f5
    style HR fill:#f3e5f5
    style SM fill:#f3e5f5
    style TM fill:#f3e5f5
    style MS fill:#f3e5f5
```

## Core Design Principles

### 1. **Separation of Concerns**
```mermaid
graph TB
    subgraph "Domain Layer - What makes your agent unique (20% of effort)"
        IC[Intent Categories]
        DH[Domain Handlers]
        SP[System Prompts]
    end
    
    subgraph "Framework Layer - Reusable infrastructure (80% of effort)"
        OE[Orchestrator Engine]
        SM[Session Manager]
        TD[Thread Detection]
    end
    
    style IC fill:#e1f5fe
    style DH fill:#e1f5fe
    style SP fill:#e1f5fe
    style OE fill:#f3e5f5
    style SM fill:#f3e5f5
    style TD fill:#f3e5f5
```

### 2. **Plugin Architecture**
```mermaid
graph TB
    subgraph "Company Plugins"
        ASP[Acme Support Plugin]
        AHP[Acme HR Plugin]
        ASEP[Acme Security Plugin]
        ADP[Acme Deploy Plugin]
    end
    
    subgraph "Framework Core"
        FC[Framework Core]
    end
    
    subgraph "Community Plugins"
        AWP[AWS Plugin]
        SP[Slack Plugin]
        JP[Jira Plugin]
        KP[K8s Plugin]
    end
    
    ASP --> FC
    AHP --> FC
    ASEP --> FC
    ADP --> FC
    
    AWP --> FC
    SP --> FC
    JP --> FC
    KP --> FC
    
    style ASP fill:#e3f2fd
    style AHP fill:#e3f2fd
    style ASEP fill:#e3f2fd
    style ADP fill:#e3f2fd
    style FC fill:#fff3e0
    style AWP fill:#f3e5f5
    style SP fill:#f3e5f5
    style JP fill:#f3e5f5
    style KP fill:#f3e5f5
```

### 3. **User Isolation & Threading**
```mermaid
graph TB
    subgraph "User A Session"
        AT1[Thread 1: Login Issue<br/>├─ Issue<br/>├─ Resolution<br/>└─ Closed]
        AT2[Thread 2: Bug Report<br/>├─ Report<br/>├─ Investigation<br/>└─ Active]
        AM[User A Memory<br/>• Past Issues<br/>• Preferences<br/>• Context]
    end
    
    subgraph "Framework Core"
        FC[Framework Core<br/>Orchestrates all users<br/>independently]
    end
    
    subgraph "User B Session"
        BT1[Thread 1: Order Status<br/>├─ Status Query<br/>└─ Resolved]
        BT2[Thread 2: Return Request<br/>├─ Request<br/>├─ Processing<br/>└─ Active]
        BM[User B Memory<br/>• Order History<br/>• Preferences<br/>• Context]
    end
    
    AT1 --> FC
    AT2 --> FC
    AM --> FC
    
    FC --> BT1
    FC --> BT2
    FC --> BM
    
    style AT1 fill:#e8f5e8
    style AT2 fill:#e8f5e8
    style AM fill:#e8f5e8
    style FC fill:#f3e5f5
    style BT1 fill:#fff3e0
    style BT2 fill:#fff3e0
    style BM fill:#fff3e0
```

## Message Processing Flow

```mermaid
flowchart TD
    UM[User Message] --> SL[Session Lookup]
    SL --> |Load/Create User Session<br/>with conversation threads| IC[Intent Classify]
    IC --> |Classify: QUERY/OUTAGE/ORDER_STATUS<br/>using domain-specific categories| TD[Thread Detector]
    TD --> |Detect: NEW_TOPIC/CONTINUATION/RESOLUTION<br/>maintain conversation context| CG[Context Gatherer]
    
    CG --> ML[Memory Lookup<br/>similar cases]
    CG --> FT[FastMCP Tools<br/>diagnostics]
    CG --> KB[Knowledge Base<br/>search]
    
    ML --> |Parallel Processing| HR[Handler Registry]
    FT --> HR
    KB --> HR
    
    HR --> |Route to: QueryHandler/OutageHandler/OrderHandler<br/>domain-specific processing| RG[Response Generator]
    RG --> |Generate human-readable response<br/>using domain prompts + LLM| CE[Confidence Evaluator]
    CE --> |High confidence → Deliver<br/>Low confidence → Escalate| US[Update State]
    US --> |Save conversation, update memory<br/>user-specific storage| FR[Final Response]
    
    style UM fill:#e8f5e8
    style SL fill:#e1f5fe
    style IC fill:#e1f5fe
    style TD fill:#e1f5fe
    style CG fill:#f3e5f5
    style ML fill:#fff3e0
    style FT fill:#fff3e0
    style KB fill:#fff3e0
    style HR fill:#f3e5f5
    style RG fill:#f3e5f5
    style CE fill:#f3e5f5
    style US fill:#f3e5f5
    style FR fill:#e8f5e8
```

## Thread Detection System

```mermaid
flowchart TD
    NM[New Message:<br/>"Actually, I have a different question about S3"] --> MFA[Multi-Factor Analysis]
    
    MFA --> KA[Keyword Analysis<br/>"actually"<br/>"different"]
    MFA --> SA[Semantic Analysis<br/>Vector similarity<br/>< 0.3]
    MFA --> TA[Temporal Analysis<br/>30min gap]
    
    KA --> D[Decision Engine]
    SA --> D
    TA --> D
    
    D --> DEC[Decision:<br/>NEW_TOPIC<br/>Confidence: 0.8]
    DEC --> CNT[Create New Thread<br/>"S3_question_1435"]
    
    style MFA fill:#f3e5f5
    style KA fill:#e1f5fe
    style SA fill:#e1f5fe
    style TA fill:#e1f5fe
    style D fill:#fff3e0
    style DEC fill:#e8f5e8
    style CNT fill:#e8f5e8
```

**Thread Transition Types:**
- **NEW_TOPIC**: "Actually, different question about..."
- **FOLLOW_UP**: "And also, regarding that issue..."
- **CLARIFY**: "What do you mean by restart?"
- **ESCALATION**: "This is urgent, production is down"
- **RESOLUTION**: "Thanks, that fixed it!"
- **CONTINUE**: "Yes, that's exactly the problem"

## Plugin Ecosystem Design

### Plugin Internal Structure
```mermaid
graph TD
    subgraph "AWS Operations Plugin Structure"
        PM[plugin.py - Main orchestrator]
        CAT[categories.py - Intent definitions]
        
        subgraph "handlers/"
            BH[billing_handler.py]
            OH[outage_handler.py]
            SH[security_handler.py]
        end
        
        subgraph "tools/"
            ACE[aws_cost_explorer.py]
            AHC[aws_health_checker.py]
            ASA[aws_security_analyzer.py]
        end
        
        subgraph "prompts/"
            BP[billing_prompts.py]
        end
        
        subgraph "config/"
            AC[aws_config.py]
        end
        
        subgraph "tests/"
            TH[test_handlers.py]
        end
    end
    
    PLF[Plugin Loading Flow:<br/>Discovery → Import → Register → Initialize → Integrate]
    
    style PM fill:#f3e5f5
    style CAT fill:#e1f5fe
    style BH fill:#e8f5e8
    style OH fill:#e8f5e8
    style SH fill:#e8f5e8
```

### Plugin Distribution Model
```mermaid
graph TB
    subgraph "Internal Company Registry"
        ICR[Company Registry]
    end
    
    subgraph "Company Plugins"
        AS[Acme Support<br/>v2.1.0]
        ASec[Acme Security<br/>v1.5.0]
        AHR[Acme HR<br/>v3.0.0]
        AD[Acme Deploy<br/>v1.2.0]
    end
    
    subgraph "Community Plugin Registry"
        CPR[Community Registry]
    end
    
    subgraph "Community Plugins"
        AWS[AWS<br/>v3.2.1]
        SL[Slack<br/>v2.0.0]
        JI[Jira<br/>v1.8.0]
        K8[K8s<br/>v2.5.0]
    end
    
    ICR --> AS
    ICR --> ASec
    ICR --> AHR
    ICR --> AD
    
    CPR --> AWS
    CPR --> SL
    CPR --> JI
    CPR --> K8
    
    INST[Installation:<br/>pip install acme-agentflow-plugins<br/>pip install agentflow-aws-plugin]
    
    style ICR fill:#e3f2fd
    style AS fill:#e3f2fd
    style ASec fill:#e3f2fd
    style AHR fill:#e3f2fd
    style AD fill:#e3f2fd
    style CPR fill:#f3e5f5
    style AWS fill:#f3e5f5
    style SL fill:#f3e5f5
    style JI fill:#f3e5f5
    style K8 fill:#f3e5f5
```

## Configuration Architecture

```mermaid
graph TB
    subgraph "Configuration Priority Hierarchy"
        subgraph "1. Environment Variables (Highest Priority)"
            ENV[AGENT_SESSION_TTL_SECONDS=7200<br/>AGENT_CONFIDENCE_THRESHOLD=0.6]
        end
        
        subgraph "2. YAML/JSON Config Files"
            YAML[production_config.yaml<br/>staging_config.yaml]
        end
        
        subgraph "3. Plugin Configurations"
            PLUGIN[aws_plugin.yaml<br/>slack_plugin.yaml]
        end
        
        subgraph "4. Framework Defaults (Lowest Priority)"
            DEFAULTS[Built-in sensible defaults]
        end
    end
    
    subgraph "Configuration Schema (Type-Safe with Pydantic)"
        SC[SessionConfig<br/>• ttl_seconds<br/>• cleanup_int<br/>• max_sessions]
        MC[MemoryConfig<br/>• max_per_user<br/>• similarity<br/>• vector_dim]
        CC[ConfidenceConfig<br/>• thresholds<br/>• escalation<br/>• intent_specific]
        
        AFC[AgentFrameworkConfig<br/>• name & version<br/>• llm_config<br/>• tools_config<br/>• domain_settings]
    end
    
    SC --> AFC
    MC --> AFC
    CC --> AFC
    
    style ENV fill:#e8f5e8
    style YAML fill:#e1f5fe
    style PLUGIN fill:#fff3e0
    style DEFAULTS fill:#f3e5f5
    style AFC fill:#f9f9f9
```

## Builder Pattern Design

```mermaid
flowchart TD
    AFB[AgentFrameworkBuilder]
    
    AFB --> WC[.with_config 'production.yaml'<br/>Load & validate configuration]
    AFB --> WP[.with_plugins ['aws_plugin', 'slack_plugin']<br/>Auto-discover plugin modules<br/>Load intent categories, handlers, prompts, tools]
    AFB --> WM[.with_middleware [LoggingMiddleware, RateLimitMiddleware]<br/>Setup cross-cutting concerns]
    AFB --> WRL[.with_rate_limiting 120<br/>Add rate limiting middleware]
    AFB --> WL[.with_logging<br/>Add logging middleware]
    
    WC --> BUILD[.build llm_client, fastmcp_client]
    WP --> BUILD
    WM --> BUILD
    WRL --> BUILD
    WL --> BUILD
    
    BUILD --> CC[Create core components:<br/>IntentClassifier<br/>HandlerRegistry<br/>MemoryStore<br/>SessionManager<br/>ThreadManager<br/>ConfidenceEvaluator<br/>ToolManager]
    
    CC --> WD[Wire dependencies]
    WD --> AO[Return configured AgentOrchestrator]
    
    AO --> RESULT[Production-ready agent with domain-specific behavior]
    
    style AFB fill:#f3e5f5
    style BUILD fill:#e8f5e8
    style AO fill:#e1f5fe
    style RESULT fill:#fff3e0
```

## Memory & State Management

```mermaid
graph TB
    subgraph "Global Memory"
        SM[Semantic Memory<br/>• Platform Facts<br/>• API Docs<br/>• Procedures]
        PM[Procedural Memory<br/>• Workflows<br/>• Best Practices<br/>• Troubleshooting]
    end
    
    subgraph "User A Memory"
        UAE[Episodic:<br/>• Issue history<br/>• Solutions<br/>• Preferences]
        UAT[Active Threads:<br/>• login_issue<br/>• bug_report]
    end
    
    subgraph "User B Memory"
        UBE[Episodic:<br/>• Order history<br/>• Preferences<br/>• Returns]
        UBT[Active Threads:<br/>• order_status<br/>• return_req]
    end
    
    subgraph "User C Memory"
        UCE[Episodic:<br/>• Task history<br/>• Reminders<br/>• Contacts]
        UCT[Active Threads:<br/>• meeting_req<br/>• email_draft]
    end
    
    subgraph "Session State Management - In-Memory Sessions (TTL-based)"
        subgraph "Session Manager"
            USA[User Session A<br/>TTL: 45min<br/>Threads: 2<br/>State: active]
            USB[User Session B<br/>TTL: 23min<br/>Threads: 1<br/>State: active]
            USC[User Session C<br/>TTL: 58min<br/>Threads: 3<br/>State: active]
            
            CT[Cleanup Task every 5 minutes<br/>Removes expired sessions]
        end
    end
    
    style SM fill:#f3e5f5
    style PM fill:#f3e5f5
    style UAE fill:#e8f5e8
    style UAT fill:#e8f5e8
    style UBE fill:#fff3e0
    style UBT fill:#fff3e0
    style UCE fill:#e1f5fe
    style UCT fill:#e1f5fe
    style USA fill:#e8f5e8
    style USB fill:#fff3e0
    style USC fill:#e1f5fe
    style CT fill:#f9f9f9
```

## Middleware Pipeline

```mermaid
flowchart TD
    IM[Incoming Message] --> AM[Authentication Middleware]
    AM --> |Verify user identity<br/>Add user context| RL[Rate Limiting Middleware]
    RL --> |Check requests/minute<br/>Enforce limits per user| LM[Logging Middleware]
    LM --> |Log request details<br/>Add correlation ID| MM[Metrics Middleware]
    MM --> |Record performance metrics<br/>Track usage patterns| FP[Framework Processing]
    FP --> |Core message processing<br/>Agent Orchestrator| RF[Response Formatting]
    RF --> |Transform response format<br/>Add metadata| AL[Audit Log Middleware]
    AL --> |Record complete interaction<br/>Store for compliance| FR[Final Response]
    
    style IM fill:#e8f5e8
    style AM fill:#e1f5fe
    style RL fill:#e1f5fe
    style LM fill:#e1f5fe
    style MM fill:#e1f5fe
    style FP fill:#f3e5f5
    style RF fill:#fff3e0
    style AL fill:#e1f5fe
    style FR fill:#e8f5e8
```

**Middleware Benefits:**
- Cross-cutting concerns
- Easy to add/remove features
- Testable in isolation
- Production monitoring
```

## Deployment Architecture

```
Production Deployment Options:

┌─ Single Instance Deployment ─┐
│                               │
│  ┌─────────────────────────┐  │
│  │     AgentFlow App       │  │
│  │                         │  │
│  │ ┌─────┐ ┌─────┐ ┌─────┐ │  │
│  │ │Plugin│ │Core │ │LLM  │ │  │
│  │ │ Mgr │ │Engine│ │Client│ │  │
│  │ └─────┘ └─────┘ └─────┘ │  │
│  └─────────────────────────┘  │
│              │                │
│  ┌─────────────────────────┐  │
│  │    FastMCP Tools        │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘

┌─ Microservices Deployment ─┐
│                             │
│ ┌─────────┐ ┌─────────────┐ │
│ │Agent    │ │ Plugin      │ │
│ │Core     │ │ Services    │ │
│ │Service  │ │             │ │
│ └─────────┘ └─────────────┘ │
│      │             │        │
│ ┌─────────┐ ┌─────────────┐ │
│ │Session  │ │ Tool        │ │
│ │Manager  │ │ Gateway     │ │
│ │Service  │ │             │ │
│ └─────────┘ └─────────────┘ │
│      │             │        │
│ ┌─────────────────────────┐ │
│ │    Message Bus/API GW   │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘

┌─ Cloud-Native Deployment ─┐
│                            │
│  ┌─ Load Balancer ─┐       │
│  │                 │       │
│  ├─ Agent Pod 1 ───┤       │
│  ├─ Agent Pod 2 ───┤       │
│  ├─ Agent Pod N ───┤       │
│  │                 │       │
│  └─────────────────┘       │
│           │                │
│  ┌─────────────────┐       │
│  │   Redis Cache   │       │
│  │ (Session Store) │       │
│  └─────────────────┘       │
│           │                │
│  ┌─────────────────┐       │
│  │  Vector DB      │       │
│  │ (Memory Store)  │       │
│  └─────────────────┘       │
└────────────────────────────┘
```

## Scalability & Performance

```
Performance Characteristics:

┌─ Async Processing Benefits ─┐
│                             │
│ Traditional Sync:           │
│ ┌─────┐ ┌─────┐ ┌─────┐     │
│ │Req 1│ │Req 2│ │Req 3│     │
│ └─────┘ └─────┘ └─────┘     │
│ │   5s  │   5s  │   5s      │
│ └───────┴───────┴─────────► │
│        Total: 15s           │
│                             │
│ AgentFlow Async:            │
│ ┌─────┐                     │
│ │Req 1│                     │
│ │Req 2│ ◄─ Parallel         │
│ │Req 3│                     │
│ └─────┘                     │
│ │   5s                      │
│ └─────────────────────────► │
│        Total: 5s            │
└─────────────────────────────┘

Concurrent User Handling:
┌─────────────────────────────┐
│     500+ Concurrent Users   │
│                             │
│ User A ─┐                   │
│ User B ─┤                   │
│ User C ─┤ ► AgentFlow ───►   │
│ User D ─┤     Core          │
│ ...    ─┤                   │
│ User N ─┘                   │
│                             │
│ • Per-user isolation        │
│ • Independent processing    │
│ • Shared infrastructure     │
│ • Memory-efficient          │
└─────────────────────────────┘

Throughput Targets:
• Response Time: < 5s (95th percentile)
• Memory Ops: < 1s (similarity search)
• Concurrent Sessions: 500+ users
• Hourly Throughput: 2000+ messages/hour
```

## Framework Benefits Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Value Proposition                           │
├─────────────────────────────────────────────────────────────────┤
│ Before AgentFlow:                                               │
│ ┌─────────────────┐                                             │
│ │ Build from      │ ◄── 100% custom development                 │
│ │ Scratch         │     • Session management                    │
│ │                 │     • Thread detection                      │
│ │ • 6-12 months   │     • Memory systems                       │
│ │ • High risk     │     • Tool integration                     │
│ │ • No standards  │     • Confidence evaluation                │
│ │ • Team silos    │     • Error handling                       │
│ └─────────────────┘     • User isolation                       │
├─────────────────────────────────────────────────────────────────┤
│ With AgentFlow:                                                 │
│ ┌─────────────────┐                                             │
│ │ Configure       │ ◄── 20% domain customization                │
│ │ Domain Logic    │     • Intent categories                     │
│ │                 │     • Domain handlers                       │
│ │ • 2-4 weeks     │     • System prompts                       │
│ │ • Low risk      │     • Tool configurations                  │
│ │ • Best practices│                                             │
│ │ • Team sharing  │ ◄── 80% reusable infrastructure            │
│ └─────────────────┘     Built-in & battle-tested               │
├─────────────────────────────────────────────────────────────────┤
│ Result: 10x faster development with production-ready quality    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Roadmap

```
┌─ Phase 1: Core Framework (4-6 weeks) ─┐
│                                        │
│ ┌─────────────────────────────────────┐│
│ │ • Agent Orchestrator                ││
│ │ • Session Management                ││
│ │ • Intent Classification             ││
│ │ • Handler Registry                  ││
│ │ • Basic Configuration               ││
│ └─────────────────────────────────────┘│
└────────────────────────────────────────┘

┌─ Phase 2: Advanced Features (3-4 weeks) ─┐
│                                           │
│ ┌─────────────────────────────────────────┐│
│ │ • Thread Detection System               ││
│ │ • Memory Management                     ││
│ │ • Confidence Evaluation                 ││
│ │ • Middleware Pipeline                   ││
│ │ • FastMCP Integration                   ││
│ └─────────────────────────────────────────┘│
└───────────────────────────────────────────┘

┌─ Phase 3: Plugin System (2-3 weeks) ─┐
│                                       │
│ ┌─────────────────────────────────────┐│
│ │ • Plugin Interface & Protocol       ││
│ │ • Plugin Manager & Discovery        ││
│ │ • Example Plugins (AWS, Slack)      ││
│ │ • Plugin Packaging & Distribution   ││
│ └─────────────────────────────────────┘│
└───────────────────────────────────────┘

┌─ Phase 4: Production Features (2-3 weeks) ─┐
│                                             │
│ ┌─────────────────────────────────────────── ┐│
│ │ • Production Configuration Management    ││
│ │ • Comprehensive Testing Suite            ││
│ │ • Documentation & Examples               ││
│ │ • Performance Optimization              ││
│ │ • Deployment Templates                  ││
│ └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘

Total Timeline: 11-16 weeks for complete framework
```

## Success Metrics

```
Framework Adoption Metrics:
┌─────────────────────────────────────────────────────────────────┐
│ Developer Experience:                                           │
│ • Time to first working agent: < 1 day                         │
│ • Lines of custom code required: < 500 (vs 5000+ from scratch) │
│ • Plugin installation time: < 5 minutes                        │
│                                                                 │
│ Technical Performance:                                          │
│ • Framework overhead: < 100ms per request                      │
│ • Memory efficiency: < 50MB per 100 concurrent users           │
│ • Plugin load time: < 2 seconds                                │
│                                                                 │
│ Ecosystem Growth:                                               │
│ • Community plugins: 20+ within 6 months                       │
│ • Company adoption: 5+ teams within first year                 │
│ • Plugin marketplace: Active contribution model                │
└─────────────────────────────────────────────────────────────────┘
```

**AgentFlow transforms conversational AI development from "build everything" to "configure domain behavior" - enabling teams to focus on what makes their agent unique while leveraging battle-tested infrastructure for everything else.**