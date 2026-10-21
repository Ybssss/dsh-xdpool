/**
 * The client-identity and conversation-identity headers on chat completions.
 *
 * These exist for one reason: the gateway attributes every
 * `/v2/chat/completions` to a client, and that attribution is what the user
 * sees in their billing console as the "使用端" column. The desktop app sends
 * `X-IDE-Type`/`X-IDE-Name`/`X-IDE-Version` and reads as "WorkBuddy"; without
 * them our requests attributed to nothing and the column showed "–" — which
 * read as broken billing even though every credit was deducted correctly.
 *
 * Around 2026-10-15 the international gateway began *refusing* requests that
 * lacked a conversation identity, with a 400 "conversation id missing" (code
 * 11128). The desktop app sends a conversation-id header set derived from its
 * session registry; the plugin sent none, so every pooled chat completion was
 * rejected. This file pins that contract.
 *
 * The regression this guards against is subtle, which is why the assertions are
 * on the exact header set rather than "some headers exist": the temptation when
 * refactoring is to fold these into the shared `chatHeaders()`, which would put
 * identity headers on the points-automation endpoints too — endpoints the
 * desktop app has never been shown to send them on. Both directions are wrong,
 * so both are pinned.
 */
import { describe, expect, it } from 'vitest'
import {
  WorkBuddyUpstreamClient,
  chatConversationIdentityOf,
  type WorkBuddyCredential,
} from '../src/upstream.ts'

/** A minimal credential; the headers under test do not depend on its details. */
function credential(): WorkBuddyCredential {
  return {
    accessToken: 'tok',
    refreshToken: 'ref',
    uid: 'u1',
    enterpriseId: 'e1',
    domain: 'd1',
  } as unknown as WorkBuddyCredential
}

/** A recording fetch that answers a chat with SSE and anything else with JSON. */
function recordingFetch() {
  const seen: { url: string; headers: Record<string, string> }[] = []
  const impl = async (url: unknown, init: RequestInit): Promise<Response> => {
    const target = String(url)
    seen.push({ url: target, headers: init.headers as Record<string, string> })
    // Chat completions stream; every other endpoint answers a JSON envelope.
    const isChat = target.includes('/v2/chat/completions')
    return new Response(isChat ? 'data: [DONE]\n\n' : '{"code":0,"data":[]}\n', {
      status: 200,
      headers: { 'Content-Type': isChat ? 'text/event-stream' : 'application/json' },
    })
  }
  return { seen, options: { fetchImpl: impl as unknown as typeof fetch } }
}

/**
 * Every path in the client that POSTs a chat completion.
 *
 * The optional `conversationKey` argument is left off here to pin the two-arg
 * call shape the tests and the compaction path already use; the gateway
 * contract requires a conversation identity on every request regardless.
 */
function chatPaths(): { name: string; run: (client: WorkBuddyUpstreamClient) => Promise<unknown> }[] {
  return [
    { name: 'completeChat', run: (c) => c.completeChat(credential(), '{}') },
    { name: 'chatStream', run: (c) => c.chatStream(credential(), '{}') },
  ]
}

describe('chat completions carry the desktop client identity', () => {
  for (const path of chatPaths()) {
    it(`${path.name} sends the three X-IDE headers`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      expect(seen).toHaveLength(1)
      expect(seen[0].url).toContain('/v2/chat/completions')
      expect(seen[0].headers['X-IDE-Type']).toBe('WorkBuddy')
      expect(seen[0].headers['X-IDE-Name']).toBe('WorkBuddy')
      expect(seen[0].headers['X-IDE-Version']).toBe('5.5.6')
    })

    it(`${path.name} keeps the shared auth and product headers alongside them`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      // The identity headers must not have REPLACED the auth contract.
      expect(seen[0].headers['Authorization']).toBe('Bearer tok')
      expect(seen[0].headers['X-Product']).toBe('SaaS')
      expect(seen[0].headers['X-User-Id']).toBe('u1')
    })

    it(`${path.name} sends the conversation identity the gateway verifies`, async () => {
      // The gateway refuses a chat completion that carries no conversation id,
      // so every path that POSTs one must send the stable session identifiers.
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      expect(seen[0].headers['X-Conversation-ID']).toBeTruthy()
      expect(seen[0].headers['X-Conversation-Request-ID']).toBeTruthy()
      expect(seen[0].headers['acp-connection-id']).toBeTruthy()
      // The per-request ids the desktop mints fresh on every call.
      expect(seen[0].headers['X-Conversation-Message-ID']).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      )
      expect(seen[0].headers['X-Request-ID']).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      )
      // The agent/purpose headers that travel with the ids on the same request.
      expect(seen[0].headers['X-Agent-Intent']).toBe('craft')
      expect(seen[0].headers['X-Agent-Purpose']).toBe('conversation')
      expect(seen[0].headers['X-Private-Data']).toBe('true')
      expect(seen[0].headers['x-codebuddy-request']).toBe('1')
      // The stable three are UUID-shaped, exactly as the desktop mints them.
      const uuidV4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
      expect(seen[0].headers['X-Conversation-ID']).toMatch(uuidV4)
      expect(seen[0].headers['X-Conversation-Request-ID']).toMatch(uuidV4)
      expect(seen[0].headers['acp-connection-id']).toMatch(uuidV4)
    })

    it(`${path.name} sends the desktop's W3C/B3 tracing context`, async () => {
      // `Zr()` in the desktop's daemon sends this alongside the ids. Its absence
      // was the remaining difference when the gateway answered "request illegal".
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      const h = seen[0].headers
      expect(h['traceparent']).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/)
      expect(h['b3']).toMatch(/^[0-9a-f]{32}-[0-9a-f]{16}-1-[0-9a-f]{16}$/)
      expect(h['X-B3-TraceId']).toMatch(/^[0-9a-f]{32}$/)
      expect(h['X-B3-SpanId']).toMatch(/^[0-9a-f]{16}$/)
      expect(h['X-B3-ParentSpanId']).toMatch(/^[0-9a-f]{16}$/)
      expect(h['X-B3-Sampled']).toBe('1')
      // X-Trace-ID repeats the traceparent trace id, as the desktop does.
      expect(h['X-Trace-ID']).toBe(h['X-B3-TraceId'])
      expect(h['traceparent']).toContain(h['X-B3-TraceId']!)
    })

    it(`${path.name} keeps the stable session ids stable within one conversation`, async () => {
      // The desktop keeps the stable three in a session registry; we derive them
      // from the conversation key so the same key always presents the same
      // identity, and two different conversations get different ones.
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await client.chatStream(credential(), '{}', undefined, 'conv-a')
      await client.chatStream(credential(), '{}', undefined, 'conv-a')
      await client.chatStream(credential(), '{}', undefined, 'conv-b')
      expect(seen).toHaveLength(3)
      // Same key, stable ids.
      expect(seen[0].headers['X-Conversation-ID']).toBe(seen[1].headers['X-Conversation-ID'])
      expect(seen[0].headers['X-Conversation-Request-ID']).toBe(seen[1].headers['X-Conversation-Request-ID'])
      expect(seen[0].headers['acp-connection-id']).toBe(seen[1].headers['acp-connection-id'])
      // Different key, different ids.
      expect(seen[0].headers['X-Conversation-ID']).not.toBe(seen[2].headers['X-Conversation-ID'])
      // But the per-request ids are fresh every time, even for the same key.
      expect(seen[0].headers['X-Conversation-Message-ID']).not.toBe(seen[1].headers['X-Conversation-Message-ID'])
      expect(seen[0].headers['X-Request-ID']).not.toBe(seen[1].headers['X-Request-ID'])
    })
  }
})

describe('chatConversationIdentityOf', () => {
  it('derives the same identity for the same key', () => {
    const a = chatConversationIdentityOf('k1')
    const b = chatConversationIdentityOf('k1')
    expect(a.conversationId).toBe(b.conversationId)
    expect(a.conversationRequestId).toBe(b.conversationRequestId)
    expect(a.acpConnectionId).toBe(b.acpConnectionId)
  })

  it('derives different identities for different keys', () => {
    const a = chatConversationIdentityOf('k1')
    const b = chatConversationIdentityOf('k2')
    expect(a.conversationId).not.toBe(b.conversationId)
    expect(a.conversationRequestId).not.toBe(b.conversationRequestId)
    expect(a.acpConnectionId).not.toBe(b.acpConnectionId)
  })

  it("produces ids shaped like the desktop's UUIDs", () => {
    // The desktop mints these with `randomUUID()`, so they are dashed UUIDs, not
    // bare hex. An earlier revision emitted 36 hex characters; the gateway
    // answered "request illegal" (11128), so the shape is pinned here.
    const uuidV4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    const id = chatConversationIdentityOf('k1')
    expect(id.conversationId).toMatch(uuidV4)
    expect(id.conversationRequestId).toMatch(uuidV4)
    expect(id.acpConnectionId).toMatch(uuidV4)
  })

  it('distinguishes the three ids of one conversation', () => {
    // Salted differently, so a conversation does not present the same value in
    // three headers the desktop populates independently.
    const id = chatConversationIdentityOf('k1')
    expect(new Set([id.conversationId, id.conversationRequestId, id.acpConnectionId]).size).toBe(3)
  })
})

describe('non-chat endpoints stay free of identity headers', () => {
  it('the points-automation task list does not claim to be the desktop app', async () => {
    const { seen, options } = recordingFetch()
    const client = new WorkBuddyUpstreamClient(options)
    await client.listTasks(credential())
    expect(seen).toHaveLength(1)
    expect(seen[0].url).not.toContain('/v2/chat/completions')
    expect(seen[0].headers['X-IDE-Type']).toBeUndefined()
    expect(seen[0].headers['X-IDE-Name']).toBeUndefined()
    expect(seen[0].headers['X-Conversation-ID']).toBeUndefined()
  })

  it('openConversation keeps sending its own identity headers', async () => {
    // The automation's conversation opener predates this fix and already sent
    // the three headers inline. It now shares the same constant, so pinning it
    // here catches one path drifting from the other.
    const { seen, options } = recordingFetch()
    const client = new WorkBuddyUpstreamClient(options)
    const opened = await client.openConversation(credential())
    // openConversation tolerates a stream that ends immediately.
    expect(seen).toHaveLength(1)
    expect(seen[0].headers['X-IDE-Type']).toBe('WorkBuddy')
    expect(opened === undefined || typeof opened === 'object').toBe(true)
  })
})
