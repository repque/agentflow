# Conversation Thread Detection System

## Thread Detection Logic

```python
import re
from typing import List, Dict, Optional, Tuple
from enum import Enum

class ThreadTransitionType(Enum):
    """Types of thread transitions"""
    NEW_TOPIC = "new_topic"           # Completely different subject
    FOLLOW_UP = "follow_up"           # Related to previous thread
    CLARIFICATION = "clarification"   # Asking for more details
    ESCALATION = "escalation"         # Moving to higher urgency
    RESOLUTION = "resolution"         # Marking something as solved
    CONTINUATION = "continuation"     # Same thread continues

class ThreadDetector:
    """Detects conversation thread transitions within user sessions"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.transition_keywords = {
            'new_topic': [
                'actually', 'by the way', 'also', 'another thing', 
                'different question', 'new issue', 'separate problem',
                'moving on', 'next', 'additionally'
            ],
            'follow_up': [
                'and', 'also', 'furthermore', 'in addition',
                'related to that', 'about that', 'regarding'
            ],
            'clarification': [
                'what do you mean', 'can you explain', 'how do I',
                'what exactly', 'clarify', 'more details'
            ],
            'escalation': [
                'urgent', 'critical', 'emergency', 'production down',
                'high priority', 'asap', 'immediately'
            ],
            'resolution': [
                'solved', 'fixed', 'resolved', 'working now',
                'that worked', 'problem solved', 'issue resolved',
                'thanks', 'perfect', 'got it'
            ]
        }
    
    async def detect_thread_transition(
        self, 
        new_message: str,
        current_thread: Optional[ConversationThread],
        session_context: dict
    ) -> Tuple[ThreadTransitionType, float, dict]:
        """
        Detect if new message starts a new thread or continues existing one
        
        Returns:
            - ThreadTransitionType: Type of transition
            - float: Confidence score (0.0-1.0)
            - dict: Additional context about the transition
        """
        
        # If no current thread, this is definitely a new topic
        if not current_thread:
            return ThreadTransitionType.NEW_TOPIC, 1.0, {
                'reason': 'no_existing_thread',
                'suggested_thread_id': self._generate_thread_id(new_message)
            }
        
        # Check for explicit resolution markers
        if self._check_resolution_markers(new_message):
            return ThreadTransitionType.RESOLUTION, 0.9, {
                'reason': 'explicit_resolution',
                'resolve_current_thread': True
            }
        
        # Perform multi-factor analysis
        keyword_analysis = await self._analyze_keywords(new_message)
        semantic_analysis = await self._analyze_semantic_similarity(
            new_message, current_thread
        )
        temporal_analysis = self._analyze_temporal_patterns(current_thread)
        
        # Combine analyses for final decision
        transition_type, confidence, context = await self._combine_analyses(
            keyword_analysis,
            semantic_analysis, 
            temporal_analysis,
            new_message,
            current_thread
        )
        
        return transition_type, confidence, context
    
    async def _analyze_keywords(self, message: str) -> dict:
        """Analyze message for thread transition keywords"""
        message_lower = message.lower()
        
        scores = {}
        for transition_type, keywords in self.transition_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            scores[transition_type] = score / len(keywords) if keywords else 0
        
        return {
            'scores': scores,
            'dominant_type': max(scores, key=scores.get) if scores else None,
            'confidence': max(scores.values()) if scores else 0.0
        }
    
    async def _analyze_semantic_similarity(
        self, 
        new_message: str, 
        current_thread: ConversationThread
    ) -> dict:
        """Analyze semantic similarity between new message and current thread"""
        
        if not current_thread.messages:
            return {'similarity': 0.0, 'analysis': 'no_thread_history'}
        
        # Get recent messages from current thread
        recent_messages = current_thread.messages[-3:]  # Last 3 messages
        thread_context = " ".join([msg.get('content', '') for msg in recent_messages])
        
        # Use LLM to analyze semantic similarity
        similarity_prompt = f"""
        Analyze the semantic similarity between this new message and the current conversation thread.
        
        Current thread context: {thread_context[:500]}
        Current thread topic: {current_thread.initial_topic}
        New message: {new_message}
        
        Rate similarity on a scale of 0.0 to 1.0:
        - 1.0: Same topic, direct continuation
        - 0.7-0.9: Related topic, follow-up question
        - 0.3-0.6: Loosely related, possible connection
        - 0.0-0.2: Completely different topic
        
        Return JSON format:
        {{
            "similarity": 0.85,
            "reasoning": "Brief explanation",
            "topic_shift": true/false
        }}
        """
        
        try:
            result = await self.llm_client.agenerate(similarity_prompt)
            return result
        except Exception as e:
            # Fallback to basic keyword matching
            return await self._basic_similarity_fallback(new_message, current_thread)
    
    async def _basic_similarity_fallback(
        self, 
        new_message: str, 
        current_thread: ConversationThread
    ) -> dict:
        """Basic similarity analysis as fallback"""
        new_words = set(new_message.lower().split())
        thread_words = set(current_thread.initial_topic.lower().split())
        
        if not thread_words:
            return {'similarity': 0.0, 'reasoning': 'no_thread_context'}
        
        intersection = new_words.intersection(thread_words)
        similarity = len(intersection) / max(len(new_words), len(thread_words))
        
        return {
            'similarity': similarity,
            'reasoning': f'keyword_overlap_{len(intersection)}_words',
            'topic_shift': similarity < 0.3
        }
    
    def _analyze_temporal_patterns(self, current_thread: ConversationThread) -> dict:
        """Analyze temporal patterns for thread detection"""
        now = datetime.now()
        time_since_last = now - current_thread.last_message_at
        
        # Thread likely stale if inactive for >30 minutes
        is_stale = time_since_last.total_seconds() > 1800
        
        # Urgency increases likelihood of new thread
        urgency_factor = min(time_since_last.total_seconds() / 3600, 1.0)  # 0-1 over 1 hour
        
        return {
            'time_since_last_minutes': time_since_last.total_seconds() / 60,
            'is_stale': is_stale,
            'urgency_factor': urgency_factor,
            'thread_age_hours': (now - current_thread.created_at).total_seconds() / 3600
        }
    
    async def _combine_analyses(
        self,
        keyword_analysis: dict,
        semantic_analysis: dict,
        temporal_analysis: dict,
        new_message: str,
        current_thread: ConversationThread
    ) -> Tuple[ThreadTransitionType, float, dict]:
        """Combine all analyses to make final thread transition decision"""
        
        # Resolution check (highest priority)
        if keyword_analysis['scores'].get('resolution', 0) > 0.3:
            return ThreadTransitionType.RESOLUTION, 0.9, {
                'reason': 'resolution_detected',
                'resolve_current_thread': True
            }
        
        # Escalation check (high priority)
        if keyword_analysis['scores'].get('escalation', 0) > 0.2:
            return ThreadTransitionType.ESCALATION, 0.8, {
                'reason': 'escalation_detected',
                'increase_priority': True
            }
        
        # New topic detection
        semantic_similarity = semantic_analysis.get('similarity', 0.5)
        new_topic_keywords = keyword_analysis['scores'].get('new_topic', 0)
        
        # Strong indicators of new topic
        if (semantic_similarity < 0.3 and new_topic_keywords > 0.1) or new_topic_keywords > 0.3:
            return ThreadTransitionType.NEW_TOPIC, 0.8, {
                'reason': 'semantic_and_keyword_mismatch',
                'semantic_similarity': semantic_similarity,
                'suggested_thread_id': self._generate_thread_id(new_message)
            }
        
        # Thread is stale, likely new topic
        if temporal_analysis['is_stale'] and semantic_similarity < 0.6:
            return ThreadTransitionType.NEW_TOPIC, 0.7, {
                'reason': 'stale_thread_low_similarity',
                'suggested_thread_id': self._generate_thread_id(new_message)
            }
        
        # Follow-up detection
        if (semantic_similarity > 0.6 and 
            keyword_analysis['scores'].get('follow_up', 0) > 0.1):
            return ThreadTransitionType.FOLLOW_UP, 0.8, {
                'reason': 'semantic_similarity_with_followup_keywords'
            }
        
        # Clarification request
        if keyword_analysis['scores'].get('clarification', 0) > 0.2:
            return ThreadTransitionType.CLARIFICATION, 0.7, {
                'reason': 'clarification_keywords_detected'
            }
        
        # Default to continuation if semantic similarity is reasonable
        if semantic_similarity > 0.4:
            return ThreadTransitionType.CONTINUATION, 0.6, {
                'reason': 'adequate_semantic_similarity',
                'semantic_similarity': semantic_similarity
            }
        
        # Fall back to new topic
        return ThreadTransitionType.NEW_TOPIC, 0.5, {
            'reason': 'low_similarity_fallback',
            'semantic_similarity': semantic_similarity,
            'suggested_thread_id': self._generate_thread_id(new_message)
        }
    
    def _check_resolution_markers(self, message: str) -> bool:
        """Check for explicit resolution markers"""
        resolution_patterns = [
            r'\b(thank you|thanks?|perfect|got it|that worked?)\b',
            r'\b(solved?|fixed?|resolved?|working now)\b',
            r'\b(issue resolved?|problem solved?)\b'
        ]
        
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in resolution_patterns)
    
    def _generate_thread_id(self, initial_message: str) -> str:
        """Generate a thread ID based on message content"""
        # Extract key topics/entities for thread ID
        words = initial_message.lower().split()[:5]  # First 5 words
        clean_words = [re.sub(r'[^a-z0-9]', '', word) for word in words if len(word) > 2]
        
        if clean_words:
            topic_hint = "_".join(clean_words[:3])
            timestamp = datetime.now().strftime("%H%M")
            return f"thread_{topic_hint}_{timestamp}"
        else:
            return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Integration with SupportSession
class ThreadManager:
    """Manages conversation threads within a support session"""
    
    def __init__(self, thread_detector: ThreadDetector):
        self.detector = thread_detector
    
    async def process_message_with_thread_detection(
        self,
        session: SupportSession,
        message: str,
        intent: dict
    ) -> Tuple[ConversationThread, bool]:
        """
        Process message and handle thread detection
        
        Returns:
            - ConversationThread: Active thread for this message
            - bool: Whether a new thread was created
        """
        
        current_thread = session.get_current_thread()
        
        # Detect thread transition
        transition_type, confidence, context = await self.detector.detect_thread_transition(
            message, current_thread, session.context
        )
        
        # Handle different transition types
        if transition_type == ThreadTransitionType.RESOLUTION:
            if current_thread:
                current_thread.mark_resolved(context.get('resolution_summary'))
            # Continue with current thread for this message, then it will be resolved
            new_thread_created = False
            active_thread = current_thread
            
        elif transition_type == ThreadTransitionType.NEW_TOPIC:
            # Create new thread
            thread_id = context.get('suggested_thread_id', self.detector._generate_thread_id(message))
            new_thread = ConversationThread(thread_id, message, intent['type'])
            session.threads[thread_id] = new_thread
            session.current_thread_id = thread_id
            new_thread_created = True
            active_thread = new_thread
            
        elif transition_type == ThreadTransitionType.ESCALATION:
            # Mark current thread as escalated, but continue with it
            if current_thread:
                current_thread.status = "escalated"
            new_thread_created = False
            active_thread = current_thread
            
        else:  # CONTINUATION, FOLLOW_UP, CLARIFICATION
            # Continue with current thread
            new_thread_created = False
            active_thread = current_thread
        
        # Add message to active thread
        if active_thread:
            message_data = {
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'intent': intent,
                'transition_type': transition_type.value,
                'confidence': confidence
            }
            active_thread.add_message(message_data)
        
        # Clean up stale threads
        session.cleanup_stale_threads()
        
        return active_thread, new_thread_created
```

## Usage Example

```python
# In the main workflow
async def process_message(self, user_id: str, message: str) -> dict:
    """Main async workflow orchestration per user with thread detection"""
    session = await self._get_or_create_session(user_id)
    intent = await self._classify_intent(message, session)
    
    # Thread detection and management
    active_thread, new_thread_created = await self.thread_manager.process_message_with_thread_detection(
        session, message, intent
    )
    
    # Use thread context for better responses
    thread_context = {
        'current_thread': active_thread,
        'new_thread_created': new_thread_created,
        'thread_history': active_thread.messages if active_thread else [],
        'thread_topic': active_thread.initial_topic if active_thread else None
    }
    
    context = await self._gather_context(intent, session, thread_context)
    response = await self._route_and_handle(intent, context, session)
    confidence = await self._evaluate_confidence(response, context)
    
    return await self._finalize_response(response, confidence, session)
```

## Thread Detection Triggers

1. **New Topic Detection**:
   - Keywords: "actually", "by the way", "different question"
   - Low semantic similarity (<0.3) with current thread
   - Time gap >30 minutes + low similarity

2. **Resolution Detection**:
   - Keywords: "thanks", "solved", "that worked"
   - Explicit confirmation patterns
   - Positive sentiment with closure indicators

3. **Escalation Detection**:
   - Urgency keywords: "urgent", "critical", "production down"
   - Sentiment shift to negative/frustrated
   - Request for higher-level support

4. **Clarification Detection**:
   - Question patterns: "what do you mean", "can you explain"
   - Reference to previous responses
   - Request for more details

5. **Continuation Detection**:
   - High semantic similarity (>0.6)
   - Natural conversation flow
   - Temporal continuity