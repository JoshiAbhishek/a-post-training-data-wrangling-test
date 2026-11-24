// One interface over model backends. The harness only ever talks to a Provider.

export type Role = "system" | "user" | "assistant";
export type Message = { role: Role; content: string };

export type CompletionOptions = {
  temperature: number;
  maxTokens: number;
  seed?: number;
};

/** Token counts as the server reports them; undefined when it sends none (never guessed as 0). */
export type Usage = { promptTokens?: number; completionTokens?: number };

export type Completion = {
  text: string;
  usage: Usage;
  latencyMs: number;
  /** Model id reported by the server, when available. */
  model?: string;
  /** Why generation stopped, e.g. "stop" or "length" (hit the max-token limit). */
  finishReason?: string;
  /**
   * The server's software and hardware as it reports them (OpenAI's `system_fingerprint`); for
   * mlx_lm.server, its mlx-lm and MLX versions, the OS and the GPU.
   */
  systemFingerprint?: string;
};

export type Provider = {
  name: string;
  complete(messages: Message[], opts: CompletionOptions): Promise<Completion>;
};

export class ProviderError extends Error {}
