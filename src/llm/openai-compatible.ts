// Provider for OpenAI-compatible chat servers, e.g. `mlx_lm.server`. Uses plain fetch; no SDK.
import { type Completion, type CompletionOptions, type Message, type Provider, ProviderError } from "./provider.ts";

export type OpenAICompatibleConfig = {
  baseUrl: string; // e.g. http://127.0.0.1:8080
  /**
   * Sent as `model`. Defaults to "default_model", the name mlx_lm.server maps to the model it was
   * started with; any other value makes it try to load that name as a model path or repo.
   */
  model?: string;
  name?: string;
  timeoutMs?: number;
  /** Injected for tests; defaults to global fetch. */
  fetch?: typeof fetch;
  now?: () => number;
};

type ChatResponse = {
  model?: string;
  system_fingerprint?: unknown;
  choices?: { message?: { content?: string | null }; finish_reason?: string | null }[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

/** A token count, or undefined if the server didn't send one: a missing count isn't 0 tokens. */
const tokens = (n: unknown): number | undefined => (typeof n === "number" && Number.isFinite(n) ? n : undefined);

/** mlx_lm.server's name for "the model this server was started with" (mlx_lm/server.py). */
export const MLX_DEFAULT_MODEL = "default_model";

export function openAICompatible(cfg: OpenAICompatibleConfig): Provider {
  const doFetch = cfg.fetch ?? fetch;
  const now = cfg.now ?? (() => performance.now());
  const base = cfg.baseUrl.replace(/\/+$/, "");

  return {
    name: cfg.name ?? cfg.model ?? base,
    async complete(messages: Message[], opts: CompletionOptions): Promise<Completion> {
      const body = {
        model: cfg.model ?? MLX_DEFAULT_MODEL,
        messages,
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        ...(opts.seed === undefined ? {} : { seed: opts.seed }),
        stream: false,
      };
      const start = now();
      let res: Response;
      try {
        res = await doFetch(`${base}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(cfg.timeoutMs ?? 300_000),
        });
      } catch (e) {
        throw new ProviderError(`request to ${base} failed: ${(e as Error).message}`);
      }
      if (!res.ok) throw new ProviderError(`HTTP ${res.status} from ${base}: ${(await res.text()).slice(0, 500)}`);
      let data: ChatResponse;
      try {
        data = (await res.json()) as ChatResponse;
      } catch (e) {
        throw new ProviderError(`invalid JSON body from ${base}: ${(e as Error).message}`);
      }
      // fetch resolves when headers arrive; stop the clock only once the whole body has been read.
      const latencyMs = now() - start;
      const choice = data.choices?.[0];
      // mlx_lm.server sends `content: null` when the model generated no text (it stopped at once, or wrote
      // only to a separate reasoning field). That is an empty answer for the harness to score, not a server
      // failure; a response with no message or content at all still is.
      const content = choice?.message?.content;
      const text = content === null ? "" : content;
      if (typeof text !== "string") throw new ProviderError(`no message content in response from ${base}`);
      return {
        text,
        latencyMs,
        model: data.model,
        finishReason: choice?.finish_reason ?? undefined,
        ...(typeof data.system_fingerprint === "string" ? { systemFingerprint: data.system_fingerprint } : {}),
        usage: {
          promptTokens: tokens(data.usage?.prompt_tokens),
          completionTokens: tokens(data.usage?.completion_tokens),
        },
      };
    },
  };
}

/** Throws a clear error if the server isn't up (the harness doesn't start it). */
export async function healthCheck(baseUrl: string, doFetch: typeof fetch = fetch): Promise<string[]> {
  const base = baseUrl.replace(/\/+$/, "");
  let res: Response;
  try {
    res = await doFetch(`${base}/v1/models`, { signal: AbortSignal.timeout(5_000) });
  } catch (e) {
    throw new ProviderError(
      `No model server at ${base} (${(e as Error).message}). Start one with: scripts/serve-model.sh <model-name>`,
    );
  }
  if (!res.ok) throw new ProviderError(`Model server at ${base} returned HTTP ${res.status} for /v1/models`);
  // mlx_lm.server sends its 200 headers before building the list, so a failure while listing (e.g. no
  // Hugging Face cache folder) arrives as an empty or cut-off body rather than an HTTP error.
  let data: { data?: { id?: string }[] };
  try {
    data = (await res.json()) as typeof data;
  } catch (e) {
    throw new ProviderError(
      `Model server at ${base} sent an unreadable /v1/models response (${(e as Error).message}); ` +
        `check the server's log. scripts/serve-model.sh sets up what mlx_lm.server needs to list models.`,
    );
  }
  return (data.data ?? []).map((m) => m.id ?? "").filter(Boolean);
}
