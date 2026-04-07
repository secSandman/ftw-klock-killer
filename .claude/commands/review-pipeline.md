Read the following files in full:
- `orchestrator/pipeline.py`
- `agents/base_agent.py`
- `bus/message_bus.py`
- `bus/decision_tracker.py`
- `bus/taxonomy.py`

Also read `data/taxonomy.json`.

You are acting as a senior distributed systems architect reviewing the multi-agent pipeline for correctness, performance, and scalability.

**Pipeline flow:**
```
ORCHESTRATOR → pruner.in → PRUNER → pruner.out → GRUG → grug.out 
             → BALANCER → balancer.out → ZIPPY → zippy.out → ORCHESTRATOR
```

**Your audit must cover:**

1. **Message Bus Design** — The bus is a JSONL file with cursor tracking. What are the failure modes? (e.g. partial writes, cursor desync, concurrent access). Is this appropriate for the stated use case? What would break under parallel agent runs?

2. **FeatureSlice Schema** — Review the `FeatureSlice` dataclass. Is the schema complete? Are there fields that agents need but don't have? Are there fields that are defined but never used?

3. **Error Propagation** — Pipeline.run() has `if agent_out is None: return self._error(...)`. What happens to the bus state after an error? Can a failed run leave orphaned messages that corrupt the next run?

4. **Timeout Values** — PRUNER=2s, GRUG=10s, BALANCER=2s, ZIPPY=5s. Are these appropriate? What happens when GRUG runs F1-F4 on a 10,000-line C file from the Doom corpus?

5. **Taxonomy Enforcement** — All agents must validate tags against `data/taxonomy.json`. Review whether each agent's taxonomy tags are actually validated at publish time vs just set. Any agent that skips validation is a liability.

6. **Decision Audit** — Review `DecisionTracker` usage across all agents. Are all meaningful routing decisions logged? Are there silent decisions (places where an agent changes behavior without logging)?

7. **Stateless vs Stateful** — The pipeline creates a new MessageBus per Pipeline instance. Between two `pipeline.run()` calls, does bus state carry over? Should it? What's the expected behavior for multi-turn conversations?

8. **Missing integration** — The pipeline returns `routing_decision` but never makes an actual LLM API call. What's the minimal code change to wire BALANCER's routing decision to an actual Anthropic/OpenAI call? Write it.

9. **Scalability gaps** — If this pipeline needed to handle 100 concurrent files, what are the top 3 architectural changes needed?

10. **Proposed improvements** — Rank and write the top 3 changes by impact/effort ratio.
