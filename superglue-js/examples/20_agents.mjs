/**
 * Example 20: Agent system (AgentSpecJs, AgentEngine, Client.runAgent).
 *
 *   OPENAI_API_KEY=sk-... node examples/20_agents.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  AgentEngine,
  AgentSpecJs,
  createClient,
  HookStage,
} = require("../dist/facade.js");

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini";
if (!API_KEY) {
  console.error("Set OPENAI_API_KEY before running this example.");
  process.exit(1);
}

async function main() {
  console.log("=".repeat(60));
  console.log("Example 1: Minimal agent (persona only)");
  console.log("=".repeat(60));

  let spec = new AgentSpecJs(
    "SimpleHelper",
    "a concise and helpful assistant",
    null,
    null,
    "",
    16,
    3,
    null,
    null,
  );
  console.log("Compiled system prompt:");
  console.log(spec.compileSystemPrompt());
  console.log();

  let agent = new AgentEngine(spec, API_KEY, undefined, MODEL);
  let result = await agent.run("What is 2 + 2? Answer in one sentence.");
  console.log("Response:", result.content);
  console.log();

  console.log("=".repeat(60));
  console.log("Example 2: Agent with goals and constraints");
  console.log("=".repeat(60));

  const researchSpec = new AgentSpecJs(
    "ResearchAssistant",
    "an expert research assistant specialising in science and technology",
    [
      "Provide accurate, well-sourced answers backed by evidence.",
      "Explain complex concepts clearly for a general audience.",
      "Acknowledge the limits of your knowledge honestly.",
    ],
    [
      "Never speculate without clearly labelling it as speculation.",
      "Do not cite sources you cannot verify.",
      "Keep responses concise — prefer bullet points over long paragraphs.",
    ],
    MODEL,
    4,
    3,
    null,
    null,
  );

  console.log("Compiled system prompt:");
  console.log(researchSpec.compileSystemPrompt());
  console.log();

  const researcher = new AgentEngine(researchSpec, API_KEY);
  result = await researcher.run(
    "Explain quantum entanglement in 3 bullet points.",
  );
  console.log("Response:", result.content);
  console.log();

  console.log("=".repeat(60));
  console.log("Example 3: Agent with tools");
  console.log("=".repeat(60));

  const financeSpec = new AgentSpecJs(
    "FinanceAgent",
    "a sharp and data-driven financial analyst",
    [
      "Provide real-time market data when asked.",
      "Give concise, actionable insights based on the data.",
    ],
    [
      "Never give investment advice — always say 'this is not financial advice'.",
      "Always cite the data source or tool used.",
    ],
    "",
    16,
    3,
    null,
    null,
  );

  const financeAgent = new AgentEngine(financeSpec, API_KEY, undefined, MODEL);

  const prices = { AAPL: 182.5, MSFT: 415.2, GOOGL: 175.8 };
  const companies = {
    apple: { name: "Apple Inc.", sector: "Technology", employees: 164_000 },
    microsoft: { name: "Microsoft Corp.", sector: "Technology", employees: 221_000 },
  };

  await financeAgent.registerTool(
    "get_stock_price",
    "Get the latest stock price for a given ticker symbol.",
    {
      type: "object",
      properties: {
        ticker: { type: "string", description: "Ticker e.g. AAPL" },
      },
      required: ["ticker"],
    },
    async (args) => {
      const a = args ?? {};
      const ticker = String(a.ticker ?? "").toUpperCase();
      const price = prices[ticker];
      if (price === undefined) {
        return { error: `Unknown ticker: ${ticker}` };
      }
      return { ticker, price, currency: "USD" };
    },
  );

  await financeAgent.registerTool(
    "get_company_info",
    "Retrieve basic information about a publicly listed company.",
    {
      type: "object",
      properties: {
        name: { type: "string", description: "Company name or partial name" },
      },
      required: ["name"],
    },
    async (args) => {
      const a = args ?? {};
      const name = String(a.name ?? "").toLowerCase();
      for (const [k, v] of Object.entries(companies)) {
        if (name.includes(k) || k.includes(name)) return v;
      }
      return { error: `No company found for: ${name}` };
    },
  );

  result = await financeAgent.run(
    "What is Apple's current stock price and how many employees do they have?",
  );
  console.log("Response:", result.content);
  console.log("Tool rounds:", result.rounds);
  console.log();

  console.log("=".repeat(60));
  console.log("Example 4: Agent with hooks");
  console.log("=".repeat(60));

  const auditLog = [];

  const hookedSpec = new AgentSpecJs(
    "ComplianceAgent",
    "a compliance-aware business analyst",
    ["Answer business questions accurately"],
    ["Always maintain a professional tone"],
    "",
    16,
    3,
    null,
    null,
  );

  const hookedAgent = new AgentEngine(hookedSpec, API_KEY, undefined, MODEL);

  await hookedAgent.registerHook(
    HookStage.PRE_COMPLETION,
    (ctx) => {
      auditLog.push({
        stage: ctx.stage,
        content_preview: String(ctx.content ?? "").slice(0, 80),
      });
      return null;
    },
    "audit",
  );

  await hookedAgent.registerHook(
    HookStage.POST_COMPLETION,
    (ctx) => {
      if (ctx.stage === "post_completion") {
        const content = String(ctx.content ?? "");
        if (!content.endsWith("*")) {
          return (
            content +
            "\n\n*This information is for educational purposes only.*"
          );
        }
      }
      return null;
    },
    "disclaimer",
  );

  result = await hookedAgent.run(
    "Summarise the top 3 risks of entering the EV market in 2 sentences.",
  );
  console.log("Response:", result.content);
  console.log("Audit log entries:", auditLog.length, "—", auditLog);
  console.log();

  console.log("=".repeat(60));
  console.log("Example 5: Agent with guardrails");
  console.log("=".repeat(60));

  const safeSpec = new AgentSpecJs(
    "SafeAgent",
    "a privacy-conscious data assistant",
    ["Answer data-related questions safely"],
    ["Never process or repeat personally identifiable information"],
    "",
    16,
    2,
    null,
    null,
  );

  const safeAgent = new AgentEngine(safeSpec, API_KEY, undefined, MODEL);

  await safeAgent.registerGuardrail(
    (stage, content) => {
      if (
        stage === "input" &&
        /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/.test(content)
      ) {
        throw new Error(
          "Input contains a potential credit card number — blocked.",
        );
      }
      if (stage === "output") {
        return content.replace(
          /[\w.+-]+@[\w-]+\.[a-z]{2,}/gi,
          "[REDACTED_EMAIL]",
        );
      }
      return content;
    },
    "both",
    "pii-guard",
  );

  result = await safeAgent.run(
    "What is the best way to store user data securely?",
  );
  console.log("Safe response:", result.content);
  console.log();

  try {
    await safeAgent.run("My card number is 4111 1111 1111 1111. Is it valid?");
    console.log("ERROR: should have been blocked");
  } catch (e) {
    console.log(`Blocked as expected: ${e.message}`);
  }
  console.log();

  console.log("=".repeat(60));
  console.log("Example 6: Sequential multi-agent pipeline (manager → worker)");
  console.log("=".repeat(60));

  const sharedClient = createClient({
    apiKey: API_KEY,
    model: MODEL,
  });

  const managerSpec = new AgentSpecJs(
    "ProjectManager",
    "a seasoned software project manager",
    [
      "Break complex software tasks into 3–5 clear, actionable sub-tasks.",
      "Assign priorities (High / Medium / Low) to each sub-task.",
    ],
    [
      "Output exactly one numbered list — nothing else.",
      "Each sub-task must be completable in less than a day.",
    ],
    "",
    16,
    3,
    null,
    null,
  );

  const developerSpec = new AgentSpecJs(
    "SeniorDeveloper",
    "a pragmatic senior software engineer who writes clean, idiomatic Python",
    [
      "Implement the sub-task provided concisely.",
      "Include brief inline comments where the logic is non-obvious.",
    ],
    [
      "Write Python 3.12+ code only.",
      "Do not add placeholder TODO comments — either implement it or explain why not.",
    ],
    "",
    16,
    3,
    null,
    null,
  );

  const task = "Build a rate-limited HTTP client wrapper in Python.";
  console.log(`Task: ${task}\n`);

  const planResult = await sharedClient.runAgent(managerSpec, task);
  console.log("Manager output (plan):");
  console.log(planResult.content);
  console.log();

  if (planResult.content) {
    const devPrompt = `Implement sub-task 1 from this plan:\n${planResult.content}`;
    const codeResult = await sharedClient.runAgent(developerSpec, devPrompt);
    console.log("Developer output (code):");
    console.log(codeResult.content);
    console.log();
  }

  console.log("=".repeat(60));
  console.log("Example 7: Custom system_prompt override");
  console.log("=".repeat(60));

  const pirateSpec = new AgentSpecJs(
    "PirateBot",
    "ignored when system_prompt is set",
    null,
    null,
    "",
    16,
    3,
    "You are a pirate. Respond exclusively in pirate speak. " +
      "Ye shall never break character, arrr!",
    null,
  );

  console.log("Compiled prompt (override):");
  console.log(pirateSpec.compileSystemPrompt());
  console.log();

  const pirate = new AgentEngine(pirateSpec, API_KEY, undefined, MODEL);
  result = await pirate.run("Hello! How are you today?");
  console.log("Pirate response:", result.content);
  console.log();

  console.log("=".repeat(60));
  console.log("Example 8: AgentSpec reasoningEffort");
  console.log("=".repeat(60));

  const deepSpec = new AgentSpecJs(
    "DeepThinker",
    "a careful analyst",
    ["Give precise answers"],
    null,
    "",
    16,
    3,
    null,
    "high",
  );
  console.log("reasoningEffort:", deepSpec.reasoningEffort);
  const deepAgent = new AgentEngine(deepSpec, API_KEY, undefined, MODEL);
  result = await deepAgent.run("What is 12 × 13? One sentence.");
  console.log("Response:", result.content);
  console.log();

  console.log("All examples completed successfully.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
