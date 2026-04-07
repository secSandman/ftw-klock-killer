Read `agents/base_agent.py` in full to understand the FeatureSlice and BaseAgent interface.
Read `agents/pruner.py` as a reference implementation example.
Read `data/taxonomy.json` to understand valid taxonomy tags.
Read `bus/message_bus.py` to understand channel names.

The user wants to create a new agent. Their specification:
$ARGUMENTS

**Your task:**

1. **Clarify the agent's role** — Based on the user's description, identify:
   - What is this agent's single responsibility? (One job, one job well)
   - What channel does it read from? (must be an existing channel or a new one)
   - What channel does it write to?
   - Where does it fit in the pipeline? (before PRUNER? after ZIPPY? parallel?)

2. **Research grounding** — Every agent must be grounded in published research. Identify:
   - What academic paper or established algorithm best matches this agent's strategy?
   - Write a 3-4 line research docstring citation (author, year, title, DOI/arXiv if known)

3. **Scaffold the agent** — Write the complete agent file following this structure:
   ```python
   """
   <agent_name>.py — <AgentName> Agent (The <Archetype> / <Role>)
   ================================================================
   Strategy: <one-line strategy>
   Action:   <one-line action>

   Research grounding:
     - <Author> (<Year>). "<Title>." <Journal/Conference>.
       <DOI or arXiv URL>
     - <mapping paragraph>
   """
   
   class <AgentName>Agent(BaseAgent):
       AGENT_ID  = "<AGENTNAME>"
       IN_CHANNEL  = "<in_channel>"
       OUT_CHANNEL = "<out_channel>"
       
       def process(self, slice_in: FeatureSlice) -> FeatureSlice:
           # implementation
   ```

4. **Taxonomy tags** — List the taxonomy tags this agent will use. If new tags are needed, show the addition to `data/taxonomy.json`.

5. **Pipeline wiring** — Show the change needed in `orchestrator/pipeline.py` to add this agent to the run sequence.

6. **Test stub** — Write a `tests/test_<agent_name>.py` file with at least 5 tests covering the core logic.

7. **README update** — Write the markdown row for this agent to add to the crew table.

Be explicit. Write runnable code, not pseudocode. The agent must work on first import.
