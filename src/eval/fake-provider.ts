// Test helpers: providers that return canned responses and record every request.
import type { Completion, CompletionOptions, Message, Provider } from "../llm/provider.ts";

export type Recorded = { messages: Message[]; opts: CompletionOptions };

/** Responds by calling `reply` with the request; records every call. */
export function fakeProvider(reply: (messages: Message[], call: number) => string) {
  const calls: Recorded[] = [];
  const provider: Provider = {
    name: "fake",
    async complete(messages, opts): Promise<Completion> {
      calls.push({ messages: structuredClone(messages), opts });
      return { text: reply(messages, calls.length), latencyMs: 100, usage: { promptTokens: 50, completionTokens: 10 } };
    },
  };
  return { provider, calls };
}

/** Responds with the given responses in order (the last one repeats). */
export function scripted(...responses: string[]) {
  return fakeProvider((_, call) => responses[Math.min(call, responses.length) - 1]!);
}
