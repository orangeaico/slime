# SWE-Agent Execution Flow Reference

## Overview

This document traces the complete execution flow when `sweagent run-batch` is called, documenting all Python modules and classes invoked during the process.

## Command Reference

The analyzed command from `/home/himanshu/swe_livup/run_command.sh`:
```bash
sweagent run-batch \
    --config /home/saurav/SWE-agent/config/test_xml_v2.yaml \
    --num_workers 80 \
    --instances.type swe_bench \
    --instances.subset full \
    --instances.split test \
    --instances.filter '(astropy__astropy-11693|...'
```

---

## Execution Flow

### 1. Entry Point
**Module:** `sweagent.__main__`
**File:** `sweagent/__main__.py:1-4`

- Entry point when `sweagent` command is invoked
- Calls: `main()` from `sweagent.run.run`

---

### 2. CLI Router
**Module:** `sweagent.run.run`
**File:** `sweagent/run/run.py:72-98`
**Function:** `main()`

- Parses the command (e.g., "run-batch", "run", "inspect")
- For "run-batch" or "b" command:
  - Imports: `run_from_cli()` from `sweagent.run.run_batch`
  - Delegates execution to the batch runner

---

### 3. Batch Run CLI Parser
**Module:** `sweagent.run.run_batch`
**File:** `sweagent/run/run_batch.py:1442-1449`
**Function:** `run_from_cli()`

**Classes Used:**
- `BasicCLI` (from `sweagent.run.common`)
- `ConfigHelper` (from `sweagent.run.common`)
- `RunBatchConfig` (configuration model)

**Actions:**
- Parses command-line arguments
- Creates `RunBatchConfig` instance
- Calls: `run_from_config()`

---

### 4. Configuration & Initialization
**File:** `sweagent/run/run_batch.py:1438-1439`
**Function:** `run_from_config(config: RunBatchConfig)`

- Validates and processes configuration
- Creates `RunBatch` instance
- Calls: `RunBatch.from_config(config).main()`

---

### 5. RunBatch Class Initialization
**Class:** `RunBatch`
**File:** `sweagent/run/run_batch.py:314-419, 428-513`

#### Configuration Classes
- **`RunBatchConfig`** (lines 89-230) - Main configuration model
  - Fields: instances, agent, output_dir, num_workers, retry settings, etc.
- **`AgentConfig`** (from `sweagent.agent.agents`)
- **`BatchInstanceSourceConfig`** (from `sweagent.run.batch_instances`)

#### Instance Loading
- **`SWEBenchInstances`** or **`InstancesFromFile`** (from `sweagent.run.batch_instances`)
  - Loads problem instances based on configuration
- **`BatchInstance`** (from `sweagent.run.batch_instances`)
  - Represents individual problem statements

#### Progress & Logging
- **`RunBatchProgressManager`** (from `sweagent.run._progress`) - line 414
  - Tracks progress across all instances
- **Logger** from `sweagent.utils.log.get_logger()`
  - Emoji-based logging: 🏃

#### Hooks System
- **`CombinedRunHooks`** (from `sweagent.run.hooks.abstract`) - line 374
  - Aggregates multiple hooks
- **`SaveApplyPatchHook`** (from `sweagent.run.hooks.apply_patch`) - line 412
  - Saves and applies patches
- **Optional:** `SweBenchEvaluate` (from `sweagent.run.hooks.swe_bench_evaluate`)
  - Continuous evaluation for SWE-bench
- **Optional:** `MirrorEvaluate` (from `sweagent.run.hooks.mirror_evaluate`)
  - Custom evaluation hook

#### Threading Components
- **`FifoCapacityLimiter`** (lines 278-308)
  - Controls parallel execution capacity
- **`_LiveCoordinator`** (lines 233-275)
  - Coordinates progress rendering across threads

---

### 6. Main Execution Loop
**Method:** `RunBatch.main()`
**File:** `sweagent/run/run_batch.py:519-600`

**Execution Modes:**

1. **Single Worker Mode** (`main_single_worker()` - lines 601-614)
   - Sequential execution
   - One instance at a time

2. **Multi Worker Mode** (`main_multi_worker()` - lines 616-741)
   - Parallel execution using threading
   - Work queue with ThreadPool
   - Watcher thread for dynamic retry injection

**Additional Components:**
- **`RetryManager`** (optional, lines 532-553)
  - Manages parallel and sequential retry lanes
  - Spawns child `RunBatch` instances for retries

---

### 7. Per-Instance Execution
**Method:** `RunBatch.run_instance(instance: BatchInstance)`
**File:** `sweagent/run/run_batch.py:743-849`

**Flow:**
1. Acquire capacity (if using limiter)
2. Set up instance-specific logging
3. Check if instance should be skipped (`should_skip()`)
4. Call: `_run_instance(instance)`
5. Update progress manager
6. Release capacity

---

### 8. Instance Runner
**Method:** `RunBatch._run_instance(instance: BatchInstance)`
**File:** `sweagent/run/run_batch.py:807-849`

#### Agent Creation
**Line 811:** `agent = get_agent_from_config(self.agent_config)`

**Returns:** `Agent` class (from `sweagent.agent.agents`)

**Agent Dependencies:**
- **`AbstractModel`** subclass (from `sweagent.agent.models`)
  - Implementation: `LiteLLMModel` (most common)
- **`ToolHandler`** (from `sweagent.tools.tools`)
  - Manages available commands/tools
- **`HistoryProcessor`** (from `sweagent.agent.history_processors`)
  - Default: `DefaultHistoryProcessor`
- **`ActionSampler`** (from `sweagent.agent.action_sampler`)
  - Configured via `ActionSamplerConfig`

#### Environment Creation
**Line 824:** `env = SWEEnv.from_config(instance.env)`

**Class:** `SWEEnv` (from `sweagent.environment.swe_env`)

**Environment Dependencies:**
- **`AbstractDeployment`** (from `swerex.deployment.abstract`)
  - Created via: `get_deployment()` (from `swerex.deployment.config`)
  - Manages container/execution environment
- **`Repo`** (from `sweagent.environment.repo`)
  - Repository management and setup

#### Hooks Registration
- **Line 821:** `SetStatusAgentHook` (from `sweagent.agent.hooks.status`)
- **Line 825:** `SetStatusEnvironmentHook` (from `sweagent.environment.hooks.status`)
- **Line 828:** `SetStatusDeploymentHook` (from `swerex.deployment.hooks.status`)

#### Mirror Patch Application
**Method:** `_apply_mirror_patch_if_needed()` (lines 905-988)
- Applies optional repository patches before agent execution
- Handles both `mirror_patch` and `test_mirror_patch`

---

### 9. Agent Execution
**Module:** `sweagent.agent.agents`
**Method:** `agent.run(problem_statement, env, output_dir)`

#### Key Classes Used

**Problem Statement:**
- **`ProblemStatement`** (from `sweagent.agent.problem_statement`)
  - Contains issue description and metadata

**Parsers:**
- **`ThoughtActionParser`** (from `sweagent.tools.parsing`)
  - Parses "Thought: ... Action: ..." format
- **`ActionOnlyParser`** (from `sweagent.tools.parsing`)
  - Parses direct actions without thought

**Review/Retry Logic:**
- **`RetryLoopConfig`** (from `sweagent.agent.reviewer`)
- **`ScoreRetryLoop`** (from `sweagent.agent.reviewer`)
- **`ChooserRetryLoop`** (from `sweagent.agent.reviewer`)
- **`ReviewSubmission`** (from `sweagent.agent.reviewer`)

**Template System:**
- **`TemplateConfig`** (from `sweagent.agent.agents`)
  - Jinja2 templates for prompts
  - Fields: `system_template`, `instance_template`, `next_step_template`

---

### 10. Model Interaction
**Module:** `sweagent.agent.models`
**Class:** `LiteLLMModel` (primary implementation of `AbstractModel`)

#### Key Components

**Configuration:**
- **`GenericAPIModelConfig`** (from `sweagent.agent.models`)
  - Fields: name, temperature, api_key, cost_limits, etc.
- **`RetryConfig`** (from `sweagent.agent.models`)
  - Retry logic for API failures

**External Libraries:**
- **`litellm.completion()`** - LLM API calls
  - Unified interface for OpenAI, Anthropic, etc.
- **`AutoTokenizer`** (from `transformers`)
  - Tokenization for token counting
- **`tenacity`** - Retry mechanism with exponential backoff

**Logging:**
- **`RawModelLogger`** (from `sweagent.agent.raw_model_logger`)
  - Logs raw API requests/responses

**Cost Tracking:**
- **`InstanceStats`** (from `sweagent.agent.models`)
  - Tracks per-instance API costs and calls

**Exception Handling:**
- `ContextWindowExceededError`
- `CostLimitExceededError`
- `ContentPolicyViolationError`
- `ModelConfigurationError`

---

### 11. Environment Interaction
**Module:** `sweagent.environment.swe_env`
**Class:** `SWEEnv`

#### SWE-ReX Integration
Uses `swerex` library for container management:

**Runtime Actions:**
- **`BashAction`** (from `swerex.runtime.abstract`)
- **`BashInterruptAction`** (from `swerex.runtime.abstract`)
- **`ReadFileRequest`** (from `swerex.runtime.abstract`)
- **`WriteFileRequest`** (from `swerex.runtime.abstract`)
- **`CreateBashSessionRequest`** (from `swerex.runtime.abstract`)

**Deployment:**
- **`DeploymentConfig`** (from `swerex.deployment.config`)
- **`DockerDeploymentConfig`** (from `swerex.deployment.config`)
  - Default: Python 3.11 container

**Key Methods:**
- `env.start()` - Initialize environment
- `env.communicate()` - Execute commands
- `env.close()` - Cleanup environment

---

### 12. Tool/Command Handling
**Module:** `sweagent.tools.tools`
**Class:** `ToolHandler`

#### Components
- **`ToolConfig`** (from `sweagent.tools.tools`)
  - Configuration for available tools/commands
- **Command Bundle** (from `sweagent.tools.bundle`)
  - Default command set for shell interaction

#### Parsing
- **Module:** `sweagent.tools.parsing`
  - Extracts actions from LLM responses
  - Handles format errors

---

### 13. Results & Output
**File:** `sweagent/run/run_batch.py:847-848, 577`

#### Result Classes
- **`AgentRunResult`** (from `sweagent.types`)
  - Contains: trajectory, info dict, exit_status
- **`Trajectory`** (from `sweagent.types`)
  - List of `TrajectoryStep` objects
- **`TrajectoryStep`** (from `sweagent.types`)
  - Individual step: action, observation, thought

#### Output Functions
- **`save_predictions()`** (from `sweagent.run.common`)
  - Saves per-instance predictions
- **`merge_predictions()`** (from `sweagent.run.merge_predictions`)
  - Aggregates predictions into `preds.json`

#### Exit Statuses
- `submitted`
- `early_exit`
- `submitted_no_patch`
- Various error statuses

---

### 14. Retry Management (Optional)
**Class:** `RetryManager`
**File:** `sweagent/run/run_batch.py:1022-1437`

#### Features
- **Parallel Retry Lanes** (`retry_parallel`)
  - Multiple independent retry chains
- **Sequential Retries** (`retry_seq`)
  - Chained retries within each lane
- **Dynamic Instance Injection**
  - Watcher thread monitors reject files
  - Enqueues new retry instances on-the-fly

#### ML-Based Retry Selection
**Module:** `sweagent.utils.retry_orchestrator`
- **Function:** `collect_retryable_ids()`
  - Strategies: `status_only`, `ml_only`, `status_and_ml`, `status_minus_resolved`
  - Uses ML model to predict retry success probability

#### Child RunBatch Instances
- Each retry lane spawns new `RunBatch` instances
- Modified configuration:
  - New output directory: `{base}_retry_{lane}_seq_{seq}`
  - Modified cache_salt and seed
  - Watches parent for new retry candidates

---

## Key Module Summary

### Core Modules

1. **sweagent.run.run_batch**
   - Orchestrates batch execution
   - Manages threading and retries
   - Progress tracking

2. **sweagent.agent.agents**
   - Agent logic and coordination
   - Action-observation loop
   - Submission handling

3. **sweagent.agent.models**
   - LLM interaction via `litellm`
   - Cost/token tracking
   - Retry logic for API failures

4. **sweagent.environment.swe_env**
   - Environment management
   - Container orchestration
   - Command execution

5. **sweagent.tools.tools**
   - Tool/command handling
   - Custom shell commands
   - File operations

6. **sweagent.run.batch_instances**
   - Instance loading (SWE-bench, files, HuggingFace)
   - Filtering and slicing
   - Problem statement configuration

### External Dependencies

7. **swerex** (SWE-ReX)
   - Container/deployment management
   - Runtime command execution
   - Docker integration

8. **litellm**
   - Unified LLM API interface
   - Supports OpenAI, Anthropic, etc.

9. **transformers**
   - Tokenization (`AutoTokenizer`)
   - Required for token counting

---

## Architecture Patterns

### Design Patterns Used

1. **Hook System**
   - Extensible via hooks at multiple levels
   - Agent hooks, environment hooks, run hooks
   - Combined hooks aggregate multiple hooks

2. **Configuration as Code**
   - Pydantic models for type-safe configuration
   - Nested configuration objects
   - CLI auto-generation from models

3. **Thread Safety**
   - FIFO capacity limiter for resource control
   - Thread-local logging
   - Work queues for parallel execution

4. **Retry Orchestration**
   - Multi-level retry system
   - ML-based retry selection
   - Hierarchical retry chains

5. **Modular Design**
   - Separation of concerns
   - Agent, environment, model abstraction
   - Tool/command extensibility

---

## Execution Flow Diagram

```
sweagent run-batch
    ↓
__main__.py → run.py (CLI router)
    ↓
run_batch.py:run_from_cli()
    ↓
run_batch.py:run_from_config()
    ↓
RunBatch.from_config()
    ├── Load instances (SWEBenchInstances/InstancesFromFile)
    ├── Initialize progress manager
    ├── Setup hooks
    └── Create RetryManager (if enabled)
    ↓
RunBatch.main()
    ├── main_single_worker() [if num_workers=1]
    │   └── Sequential execution
    └── main_multi_worker() [if num_workers>1]
        └── ThreadPool with work queue
    ↓
RunBatch.run_instance(instance)
    ↓
RunBatch._run_instance(instance)
    ├── Create Agent (get_agent_from_config)
    │   ├── Model (LiteLLMModel)
    │   ├── ToolHandler
    │   ├── HistoryProcessor
    │   └── ActionSampler
    ├── Create Environment (SWEEnv.from_config)
    │   ├── Deployment (Docker via swerex)
    │   └── Repo setup
    ├── Apply mirror patches
    └── agent.run(problem_statement, env, output_dir)
        ├── Format prompts (Templates)
        ├── Model.query() → litellm.completion()
        ├── Parse response (ThoughtActionParser)
        ├── env.communicate(action)
        ├── Observe result
        └── Repeat until submission/exit
    ↓
save_predictions() → AgentRunResult
    ↓
merge_predictions() → preds.json
```

---

## File Locations Reference

- Entry: `sweagent/__main__.py`
- CLI: `sweagent/run/run.py`
- Batch: `sweagent/run/run_batch.py`
- Agent: `sweagent/agent/agents.py`
- Models: `sweagent/agent/models.py`
- Environment: `sweagent/environment/swe_env.py`
- Tools: `sweagent/tools/tools.py`
- Instances: `sweagent/run/batch_instances.py`
- Retry: `sweagent/utils/retry_orchestrator.py`

---

## Notes

- The execution is highly modular with extensive use of hooks
- Configuration models use Pydantic for validation
- Parallel execution support via threading (not multiprocessing)
- Comprehensive logging with emoji indicators
- Cost tracking and limits at multiple levels
- Retry system supports both rule-based and ML-based selection

---

**Generated:** 2026-02-26
**Source:** `/home/himanshu/swe_livup/run_command.sh`
**Codebase:** SWE-agent @ `/home/himanshu/swe_livup`
