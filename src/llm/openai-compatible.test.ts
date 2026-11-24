// Tested against an in-process fake server (an injected fetch), so no model or open port is needed.
import { describe, expect, it } from "vitest";
import { healthCheck, openAICompatible } from "./openai-compatible.ts";
import { ProviderError } from "./provider.ts";

type Call = { url: string; init?: RequestInit };

function fakeServer(handler: (url: string, body: unknown) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    return handler(url, init?.body ? JSON.parse(String(init.body)) : undefined);
  }) as typeof fetch;
  return { fetchImpl, calls };
}

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

describe("openAICompatible provider", () => {
  it("sends an OpenAI-style chat request and parses the reply", async () => {
    const server = fakeServer(() =>
      json({
        model: "phi-4-mini",
        choices: [{ message: { content: '{"unit":"line","rules":[]}' }, finish_reason: "length" }],
        usage: { prompt_tokens: 120, completion_tokens: 9 },
      }),
    );
    let t = 1000;
    const p = openAICompatible({ baseUrl: "http://127.0.0.1:8080/", fetch: server.fetchImpl, now: () => (t += 250) });
    const out = await p.complete([{ role: "user", content: "hi" }], { temperature: 0, maxTokens: 1024, seed: 7 });

    expect(out).toEqual({
      text: '{"unit":"line","rules":[]}',
      latencyMs: 250,
      model: "phi-4-mini",
      finishReason: "length",
      usage: { promptTokens: 120, completionTokens: 9 },
    });
    expect(server.calls[0]?.url).toBe("http://127.0.0.1:8080/v1/chat/completions");
    expect(JSON.parse(String(server.calls[0]?.init?.body))).toMatchObject({
      // mlx_lm.server maps exactly "default_model" to the model it was started with.
      model: "default_model",
      messages: [{ role: "user", content: "hi" }],
      temperature: 0,
      max_tokens: 1024,
      seed: 7,
      stream: false,
    });
  });

  it("returns the server's system_fingerprint when it sends one", async () => {
    const reply = (extra: object) =>
      fakeServer(() => json({ choices: [{ message: { content: "x" }, finish_reason: "stop" }], ...extra }));
    const complete = (server: ReturnType<typeof fakeServer>) =>
      openAICompatible({ baseUrl: "http://127.0.0.1:8080", fetch: server.fetchImpl }).complete([], {
        temperature: 0,
        maxTokens: 8,
      });
    const fingerprint = "0.32.2-0.32.4-test-os-test-gpu";
    expect(await complete(reply({ system_fingerprint: fingerprint }))).toMatchObject({
      systemFingerprint: fingerprint,
    });
    expect(await complete(reply({}))).not.toHaveProperty("systemFingerprint");
    expect(await complete(reply({ system_fingerprint: 7 }))).not.toHaveProperty("systemFingerprint");
  });

  it("measures latency until the body has been read, not just until headers arrive", async () => {
    let clock = 0;
    const slowBody = {
      ok: true,
      status: 200,
      json: async () => {
        clock += 900; // the server is still generating / sending the body
        return { choices: [{ message: { content: "x" } }] };
      },
    } as unknown as Response;
    const p = openAICompatible({
      baseUrl: "http://h",
      fetch: (async () => slowBody) as typeof fetch,
      now: () => clock,
    });
    const out = await p.complete([], { temperature: 0, maxTokens: 10 });
    expect(out.latencyMs).toBe(900);
  });

  it("leaves token counts the server didn't send undefined, not 0", async () => {
    const server = fakeServer(() =>
      json({ choices: [{ message: { content: "x" } }], usage: { completion_tokens: 3, prompt_tokens: null } }),
    );
    const p = openAICompatible({ baseUrl: "http://127.0.0.1:8080", fetch: server.fetchImpl });
    const out = await p.complete([{ role: "user", content: "hi" }], { temperature: 0, maxTokens: 8 });
    expect(out.usage).toEqual({ promptTokens: undefined, completionTokens: 3 });
    const none = fakeServer(() => json({ choices: [{ message: { content: "x" } }] }));
    const q = openAICompatible({ baseUrl: "http://127.0.0.1:8080", fetch: none.fetchImpl });
    expect((await q.complete([], { temperature: 0, maxTokens: 8 })).usage).toEqual({
      promptTokens: undefined,
      completionTokens: undefined,
    });
  });

  it("reports a non-JSON body as ProviderError", async () => {
    const p = openAICompatible({ baseUrl: "http://h", fetch: (async () => new Response("<html>")) as typeof fetch });
    await expect(p.complete([], { temperature: 0, maxTokens: 10 })).rejects.toThrow(/invalid JSON body/);
  });

  it("omits seed when not given", async () => {
    const server = fakeServer(() => json({ choices: [{ message: { content: "x" } }] }));
    await openAICompatible({ baseUrl: "http://h", fetch: server.fetchImpl }).complete([], {
      temperature: 0.7,
      maxTokens: 10,
    });
    expect(JSON.parse(String(server.calls[0]?.init?.body))).not.toHaveProperty("seed");
  });

  it("returns null content (mlx_lm.server's empty generation) as an empty answer", async () => {
    const p = openAICompatible({
      baseUrl: "http://h",
      fetch: fakeServer(() =>
        json({
          choices: [{ message: { role: "assistant", content: null }, finish_reason: "stop" }],
          usage: { prompt_tokens: 40, completion_tokens: 1 },
        }),
      ).fetchImpl,
    });
    const out = await p.complete([{ role: "user", content: "hi" }], { temperature: 0, maxTokens: 8 });
    expect(out).toMatchObject({ text: "", finishReason: "stop", usage: { promptTokens: 40, completionTokens: 1 } });
  });

  it("reports HTTP errors, missing content and network failures as ProviderError", async () => {
    const opts = { temperature: 0, maxTokens: 10 };
    const httpErr = openAICompatible({
      baseUrl: "http://h",
      fetch: fakeServer(() => new Response("boom", { status: 500 })).fetchImpl,
    });
    await expect(httpErr.complete([], opts)).rejects.toThrow(/HTTP 500/);

    const empty = openAICompatible({ baseUrl: "http://h", fetch: fakeServer(() => json({ choices: [] })).fetchImpl });
    await expect(empty.complete([], opts)).rejects.toBeInstanceOf(ProviderError);

    const noContent = openAICompatible({
      baseUrl: "http://h",
      fetch: fakeServer(() => json({ choices: [{ message: { role: "assistant" } }] })).fetchImpl,
    });
    await expect(noContent.complete([], opts)).rejects.toBeInstanceOf(ProviderError);

    const down = openAICompatible({
      baseUrl: "http://h",
      fetch: (async () => {
        throw new TypeError("fetch failed");
      }) as typeof fetch,
    });
    await expect(down.complete([], opts)).rejects.toThrow(/request to http:\/\/h failed/);
  });
});

describe("healthCheck", () => {
  it("returns served model ids", async () => {
    const server = fakeServer(() => json({ data: [{ id: "phi-4-mini-instruct-4bit" }] }));
    expect(await healthCheck("http://h", server.fetchImpl)).toEqual(["phi-4-mini-instruct-4bit"]);
    expect(server.calls[0]?.url).toBe("http://h/v1/models");
  });

  it("explains how to start the server when it's down", async () => {
    const down = (async () => {
      throw new TypeError("connect ECONNREFUSED");
    }) as typeof fetch;
    await expect(healthCheck("http://127.0.0.1:8080", down)).rejects.toThrow(/scripts\/serve-model\.sh/);
  });

  it("reports an empty 200 body (the server failed while listing models) as ProviderError", async () => {
    // What mlx_lm.server sends when its Hugging Face cache folder is missing: headers, then nothing.
    const emptyBody = fakeServer(() => new Response("", { status: 200 })).fetchImpl;
    const err = await healthCheck("http://h", emptyBody).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as Error).message).toMatch(/unreadable \/v1\/models response/);
  });
});
