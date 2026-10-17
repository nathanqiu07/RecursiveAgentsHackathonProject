# Original research papers

Open a benchmark, select **Agents**, and start a research agent. The local backend supplies the benchmark objective and default research settings; there is no per-run configuration form. Codex uses its existing local sign-in and configured model.

The deliverable is an original paper: the agent develops its own method, runs controlled experiments, writes measured results and limitations, and submits to the Library for independent verification. Online papers support related work; finding papers or proposing an idea alone does not complete the task. Without compute, the paper remains blocked on experiments.

Each researcher receives online paper research through the local **You.com MCP broker**, access to the **Library API**, and access to the **Resource Manager API**. The harness and broker run on the controller. A GPU VM is allocated only when the agent first executes an experiment, so paper research and experiment planning can proceed before GPU capacity is available.

## Actual You.com MCP connection

The local broker uses the official Python MCP SDK's Streamable HTTP client, including MCP initialization, tool discovery and tool calls. It connects to `https://api.you.com/mcp?tools=you-search,you-contents` with a configured key. Without a key, it uses the keyless free profile at `https://api.you.com/mcp?profile=free`, which provides `you-search` and `you-discover`, with the documented daily search limit. [Official You.com MCP documentation](https://you.com/docs/agents/mcp-server).

Configure a shared key once in **Connections**, or set `HARNESS_YOU_API_KEY` / `YDC_API_KEY` in the private worker environment. The broker supplies bearer authentication; keys do not enter agent prompts, submitted papers, tool results, repository files, GPU VMs or guest environments. You.com product scopes determine which paid capabilities the key can access. [You.com authentication documentation](https://you.com/docs/using-the-api/authentication).

This is an MCP connection in the running research pipeline. It does not require a personal MCP entry in the user's Codex configuration. The backend exposes controlled research tools to the agent and records their evidence; it does not hand an unrestricted authenticated client or key to the model.

## Tools and evidence

| Agent capability | Local tool and service |
| --- | --- |
| Find primary papers online | `search_research_papers` calls You.com MCP search and retains source URLs. |
| Read a paper | `read_research_paper` retrieves the selected primary source through the broker, recording actual content and its hash. It distinguishes full text from an abstract fallback. |
| Review existing work | `search_library` and `read_paper` use the local Library API. |
| Inspect compute | `list_resources` reads the Resource Manager's real available capacity. |
| Save a sourced idea | `propose_research_idea` stores the hypothesis, difference from prior work, read source URLs and experiment plan, then requests Library novelty review. |
| Run an experiment | `execute` requests an approved GPU VM only when needed and returns real execution results. |

A proposal contains `hypothesis`, `novelty_difference`, `source_urls`, and an `experiment_plan` defining a baseline, experiment, metric and success criterion. Search snippets alone are insufficient evidence for a paper-based proposal. Publications cite URLs actually read by the run. An abstract-only retrieval remains labeled as such; it does not establish that the full paper was read. A novelty review is an assessment against the retrieved evidence and Library, rather than proof that no similar work exists anywhere.

Without a configured Contents-capable key, the broker can read supported public primary-source pages directly over HTTPS after MCP search. It keeps the same source and content evidence requirements. Failed searches, unavailable pages and quota/authentication errors remain visible.

When compute is missing, the run preserves its sourced idea and plan and reports the missing resource. It never substitutes invented GPU experiment results or benchmark numbers. Paper-derived claims and measured experiment evidence remain distinct.

## Optional personal MCP connection

You can additionally use You.com directly from a personal Codex session. This is separate from the dashboard's existing per-researcher broker. Make `YDC_API_KEY` available privately to that client, then run:

```sh
codex mcp add you-search \
  --url 'https://api.you.com/mcp?tools=you-search,you-contents' \
  --bearer-token-env-var YDC_API_KEY
codex mcp list
```

The configuration stores an environment-variable name rather than the key. These flags were checked against the installed CLI; Codex documents HTTP MCP and environment-based bearer authentication in its [official MCP guide](https://developers.openai.com/codex/mcp). This repository does not silently modify personal MCP configuration.

Calls from that optional personal session do not automatically populate dashboard research evidence. Use the integrated researcher tools for recorded proposals and their Library review. You.com's [documentation MCP](https://you.com/docs/agents/docs-mcp-server) searches its developer documentation; it is a separate service from the live web research MCP above.
